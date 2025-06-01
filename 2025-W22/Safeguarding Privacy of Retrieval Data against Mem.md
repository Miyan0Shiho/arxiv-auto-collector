# Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?

**Authors**: Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, Jinseong Park

**Published**: 2025-05-28 07:35:07

**PDF URL**: [http://arxiv.org/pdf/2505.22061v1](http://arxiv.org/pdf/2505.22061v1)

## Abstract
Retrieval-augmented generation (RAG) mitigates the hallucination problem in
large language models (LLMs) and has proven effective for specific,
personalized applications. However, passing private retrieved documents
directly to LLMs introduces vulnerability to membership inference attacks
(MIAs), which try to determine whether the target datum exists in the private
external database or not. Based on the insight that MIA queries typically
exhibit high similarity to only one target document, we introduce Mirabel, a
similarity-based MIA detection framework designed for the RAG system. With the
proposed Mirabel, we show that simple detect-and-hide strategies can
successfully obfuscate attackers, maintain data utility, and remain
system-agnostic. We experimentally prove its detection and defense against
various state-of-the-art MIA methods and its adaptability to existing private
RAG systems.

## Full Text


<!-- PDF content starts -->

Safeguarding Privacy of Retrieval Data against Membership Inference
Attacks: Is This Query Too Close to Home?
Yujin Choi1†Youngjoo Park1†Junyoung Byun2
1Seoul National University, Republic of Korea
2Chung-Ang University, Republic of Korea
3Korea Institute for Advanced Study, Republic of Korea
{uznhigh, youngjoo0913, jaewook}@snu.ac.kr, junyoungb@cau.ac.kr, jinseong@kias.re.krJaewook Lee1Jinseong Park3*
Abstract
Retrieval-augmented generation (RAG) miti-
gates the hallucination problem in large lan-
guage models (LLMs) and has proven effective
for specific, personalized applications. How-
ever, passing private retrieved documents di-
rectly to LLMs introduces vulnerability to
membership inference attacks (MIAs), which
try to determine whether the target datum exists
in the private external database or not. Based on
the insight that MIA queries typically exhibit
high similarity to only one target document,
we introduce Mirabel, a similarity-based MIA
detection framework designed for the RAG sys-
tem. With the proposed Mirabel, we show that
simple detect-and-hide strategies can success-
fully obfuscate attackers, maintain data utility,
and remain system-agnostic. We experimen-
tally prove its detection and defense against
various state-of-the-art MIA methods and its
adaptability to existing private RAG systems.
1 Introduction
Large language models (LLMs) (Brown et al.,
2020; Grattafiori et al., 2024) have demonstrated
their own strength for general and common knowl-
edge. However, they struggle to answer domain-
specific or personalized questions, resulting in hal-
lucinations that fabricate non-truth answers from
the training set (Huang et al., 2025). Retrieval-
augmented generation (RAG) (Lewis et al., 2020)
mitigates hallucination by giving information from
external data retrieval into LLMs. RAG can provide
reliable information by extracting relevant docu-
ments from an external database.
Membership inference attacks (MIAs) (Shokri
et al., 2017) are malicious machine learning attacks
that attempt to determine whether the target docu-
ment was used in the training dataset. Recent work
has examined MIAs and extraction attacks on lan-
guage models to study the privacy leakage (Shi
*Corresponding author.†Equal contribution.et al., 2024) in the language domain. Even though
MIAs achieve lower success rates on LLMs due to
the massive size of training data samples (Puerto
et al., 2024), privacy risks in RAG systems are sig-
nificant (Zeng et al., 2024). As RAG retrieves a
few top- kdocuments and conveys them directly to
LLMs, various attack methods (Liu et al., 2025;
Naseh et al., 2025) have succeeded in inferring
whether a target document is stored in a private
external database.
However, safeguarding language models against
MIA remains less studied than in other domains
(Li et al., 2021). Within RAG systems, the agent-
based query filtering system is the only defense
explored so far, but it is still highly vulnerable to
stealth attacks that evade detection(Naseh et al.,
2025). Therefore, we present a RAG-specific safe-
guarding framework against MIAs, motivated by
the observation that MIA queries containing mem-
ber documents are similar solely to a single target
document in the external database.
In detail, we measure the similarity between an
input query and the retrieved data points. Then, we
check whether this similarity exceeds a threshold
based on the Gumbel distribution (Gumbel, 1935),
which models the maximum values of data samples.
If an input query exceeds the threshold, indicating
that the query is overly correlated with one specific
document in the retrieval, we hide this private data
in the top- kdocument conveyed to the LLMs. We
summarize our contributions as follows:
•We propose a similarity-based method for de-
tecting MIA inRAG systems using Gum bel,
named Mirabel . To the best of our knowledge,
this is the first attempt to study safeguarding
strategies specifically against MIAs in RAG.
•Mirabel safeguards private external database
against MIA through a simple detect-and-hide
approach, which obfuscates attackers while
preserving the utility of the RAG systems.
•We empirically show the effectiveness of ourarXiv:2505.22061v1  [cs.CL]  28 May 2025

method against various measures, and its com-
parability to existing defense methods.
2 Related Works
2.1 Retrieval-Augmented Generation
RAG is a strategy that enhances LLMs by inte-
grating external data retrieval into the generation
process (Lewis et al., 2020). At its core, an RAG
system comprises three primary components: an
external database Dof textual documents, a re-
triever R, and a generator (i.e., LLM) G. When a
user submits a query q, the retriever identifies the
top-kcontextual retrieval from Dbased on similar-
ity, such as cosine similarity, often computed in an
embedding model ϕ(·)(Karpukhin et al., 2020):
Rk(q) = arg topkd∈Dsim(q, d). (1)
These retrievals are then combined with the original
user query, forming an augmented context passed
to the generator. The generator then produces an
output based on the retrieval (Gao et al., 2023;
Shuster et al., 2021) as:
p(q) =RAGprompt (q, R k(q)), (2)
response =G(p(q)). (3)
RAG improves response accuracy and reduces hal-
lucinations frequently observed in pure LLMs. Fur-
thermore, RAG offers architectural flexibility: any
of the three core modules ( D,R, andG) can be re-
placed or updated independently without requiring
end-to-end retraining (Cheng et al., 2023). More-
over, query rewriting (e.g., correcting ambiguous)
and specialized retrieval (e.g., token- or graph-
based) can be used in RAG (Ram et al., 2023).
2.2 Privacy Leakage of RAG System
Despite the utility improvements, the RAG systems
inevitably pose privacy concerns when dealing with
sensitive or proprietary data in the retrieval. For ex-
ample, when dealing with sensitive data, the use
of RAG increases the risk of serious legal com-
plications and breaches of personal privacy if the
documents in the external database are exposed.
The purpose of MIAs for the RAG system is
to determine whether a target document is stored
in the database. Beginning with directly asking
whether the document is in the database or not (An-
derson et al., 2024), various attacks are proposed:
S2MIA (Li et al., 2024) that provides the first half
of the document and requests completion; MBA(Liu et al., 2025) that prompts prediction of masked
tokens; and IA (Naseh et al., 2025), standing for
interrogation attack, that asks multiple queries that
are hard to answer without the document. All of the
attacks then exploit the output of the RAG system
to infer the membership of the target document.
2.3 Safeguarding Attacks in RAG
To safeguard against MIAs in RAG systems, Naseh
et al. (2025) investigated a simple filtering method
that asks an LLM agent, such as GPT-4o, to classify
incoming queries as benign or malicious. However,
this agent-based method struggles with several chal-
lenges that will be further discussed in Section 3.1.
As a complementary safeguard, differential pri-
vacy (DP) provides a mathematically rigorous pri-
vacy guarantee for sensitive data. Duan et al. (2023)
introduced a DP framework that hides private infor-
mation through noisy labeling against MIAs. DP
LLMs have also been explored for in-context learn-
ing (Tang et al., 2024) and private prompt tuning
(Hong et al., 2024). For the RAG system, DP-RAG
(Grislain, 2024) protects the privacy of data in the
data retrieval, and DPV oteRAG (Koga et al., 2024)
uses private voting. Machine unlearning also ad-
dresses membership privacy. For example, Tran
et al. (2025) proposed to unlearn highly memorized
tokens, but their approach targets LLMs and cannot
be extended to data retrieval without training.
3 Scenarios
Following the previous studies (Liu et al., 2025;
Naseh et al., 2025), there are three parties in our
scenario: 1) the operator of the RAG system with
private external database , 2) the benign users of
the RAG, and 3) the malicious attackers attempt-
ing MIAs to the RAG system. For instance,
“A healthcare AI operator deploys a medical-
diagnosis chatbot that retrieves private patient
records. Benign users consult the chatbot to assess
their health, but attackers attempt MIAs. The oper-
ator must preserve privacy against attackers while
providing accurate answers to benign users."
Types of Queries Based on the scenario, we di-
vide the queries for RAG systems into three types:
member attack query qm
aby attackers of malicious
questions in the external database, non-member at-
tack query qn
aby attackers of malicious questions
but not in the external database, and benign query
qbof benign users without malicious intention.

Details about Attacker Given a target document
d, an attacker aims to decide whether dis contained
in the external database Dby classifying its asso-
ciated query qaas a member attack query qm
aor
a non-member attack query qn
a. We assume the at-
tacker cannot access Dor the parameters of LLMs.
Details about Operator’s Defenses The defense
should prevent attackers from recognizing the mem-
bership information while delivering correct re-
sponses to benign users without prior knowledge
of attacks, such as their prompts or patterns. The
defender also cannot access LLM parameters, just
relying on access to the external database Dand its
embedding model ϕ. Each query must be answered
immediately without pending the next queries.
3.1 Motivation and Goals
To safeguard against MIAs, it is essential to de-
termine whether an input query is malicious to
the private RAG system and the external database.
Recently, Naseh et al. (2025) proposed an LLM
agent-based detection method to evaluate the ma-
licious intent of input queries, independent of the
external database. To judge either benign queries
qbor attack queries qa, the agent identifies harm-
ful or extraction phrases within queries. However,
this agent-based detection can be deceived by
stealth queries, which are crafted attack queries
designed to mimic benign queries (Naseh et al.,
2025). Moreover, simply rejecting all suspected at-
tack queries may inadvertently reveal to attackers
filtering phrases and detour the detection systems.
These limitations suggest that an effective de-
fense requires not simply blocking queries but ob-
fuscating the attacker’s knowledge. An attack suc-
ceeds only if it can correctly separate qm
afrom
qn
ausing the RAG’s responses. Hence, to defend
against the attack, the responses should be made sta-
tistically indistinguishable. Achieving this in turn
demands that the defender distinguishes between
qm
aandqn
a, so the responses can be selectively per-
turbed. Therefore, we set two complementary goals:
1) accurately separate member attack queries from
non-member attack queries, and 2) respond with
the corresponding responses so that an attacker can-
not classify the member and non-member cases.4 Methods
4.1 Is This Query Too Close to Home?
MIA Detection with Gumbel Distribution
By analyzing the similarity of queries, we propose
a detection method for the RAG system that dis-
tinguishes member attack queries ( qm
a) from other
types of queries ( qbandqn
a). Specifically, our goal
is to find a threshold τto classify a query as a mem-
ber attack qm
aif the maximum similarity exceeds:
max
d∈Dsim(q, d)> τ. (4)
To find the threshold τ, we focus on the distribu-
tional differences between member attack queries
and other queries (Wen et al., 2024).
Figure 1a demonstratesS
qSq, the cosine simi-
larities between all queries and external database,
where Sq={sim(q, d)|d∈ D} be a set of simi-
larities between a query qand total database. The
figure shows (i) similarities of the benign query qb
and the total database follow a normal distribution,
and (ii) for the member attack quires qm
a, the dis-
tribution appears similar to qbin the most likely
region, but exhibits extreme values in the right tail,
which are too similar for private external database.
Based on these observations, we propose a detec-
tion method using the Gumbel distribution, which
represents the maximum value of data samples
(Gumbel, 1935). In previous works on LLM pri-
vacy, the Gumbel distribution is used to find top- k
selection (Durfee and Rogers, 2019; Hong et al.,
2024). By extreme value theory, if nrandom vari-
ables follow i.i.d. normal distribution, i.e.,
X1,···, Xni.i.d.∼ N (µq, σ2
q), (5)
for some µqandσq, and if nis sufficiently large,
then, the maximum of i.i.d. samples follows a Gum-
bel distribution:
max{X1,···, Xn}d− →Gumbel (µn, βn),(6)
where βn=σ/√
2 lnnand
µn=µq+σqdp
2 lnn−ln(ln n)−ln(4π)
≈µq+σqd√
2 lnn(asn→ ∞ ). (7)
Under the observations (i) and (ii), we assume
that, for each q, the samples from Sq\{smax}fol-
lows a normal distribution, where smaxis the max-
imum similarity in Sq. Letµqandσqbe the mean

0.3 0.4 0.5 0.6 0.7 0.8 0.9
Similarity02468DensityBenign
MBA
S2MIA
IA(a) All similarities between query and candidates.
0.5 0.6 0.7 0.8 0.9
Similarity0510152025DensityGumbel ThresholdBenign
MBA Mem
MBA Non-Mem
S2MIA Mem
IA Mem (b) Top-1 similarity per query with Gumbel threshold.
Figure 1: Distributions of similarity scores between queries and retrieved data. We visualize both the full similarity
distributions and the top-1 similarities. A Gumbel-based thresholdS
qSqis marked for reference.
Test set Normal MBA S2MIA IA
Sq 0.469 0.027∗0.001∗0.012∗
Sq\{smax} 0.400 0.511 0.293 0.226
Table 1: Average p-values of the normality test on the
total data set ( Sq) and on the set with the maximum
similarity removed ( Sq\{smax}). * :p-value < 0.05.
and standard deviation of Sq\{smax}. Then the
threshold τcan be determined as follows:
τ=µn+c·σq/√
2 lnn, (8)
where c=−ln(−ln(1−ρ))is a critical value of
the Gumbel distribution for significance level ρ.
Figure 1b shows histograms of the maximum
value of each Sq, i.e., smaxfor each query. The
dashed line represents the threshold based on a
Gumbel distribution, which was computed fromS
qbSqb. Thresholds computed for each attack type
for separate smaxfor non-member attack query qn
a
and member-attack query qm
aare provided in Fig-
ure 6 in Appendix C. Most smaxvalues of the mem-
ber attack queries exceed the Gumbel-based thresh-
old, whereas those of benign and non-member at-
tack queries remain below it. These findings sug-
gest that smaxof member attack queries qm
acan
be successfully separated from qbandqn
aby the
Gumbel-based threshold.
Our proposed detection method classifies a query
as a member attack query qm
abased on the Gum-
bel distribution, i.e., MIA inRAG systems using
Gumbel, named Mirabel . Mirabel not only detects
whether the query qis a member attack, but also
identifies the specific document that is likely the
target. Moreover, it can be incorporated into the
standard RAG system naturally, as it uses the simi-
larity scores already computed for top- kselection.
The detailed method is shown in Algorithm 1.
Normality test To validate our assumption, we
conduct a normality test on SqandSq\{smax}.Algorithm 1 Proposed Mirabel Detection
Input: Query embedding function ϕ, corpus
embeddings {ed=ϕ(d) :d∈ D} , query q,
significance level ρ
Output: detection result, target document dt
Initialization: Sq← {} ,eq←ϕ(q)
1:for all d∈ Ddo
2: sd←cos(eq, ed)▷Compute sim(q, d)
3: Sq←Sq∪ {sd}
4:end for
5:smax, dt←max( Sq)▷Find (arg)max sim
6:µq, σq←mean (Sq\{smax}),std(Sq\{smax})
7:µn←µq+σq·√
2 lnn ▷ Gumbel mean
8:c← − ln(−ln(1−ρ)) ▷Critical value
9:τ←µn+c·σq/√
2 lnn ▷ Threshold
10:return (smax> τ),1[smax> τ]·dt
Table 1 demonstrates the mean p-values from
D’Agostino and Pearson’s normality test, averaged
across all queries q. Across all MIAs, the p-values
of total similarity scores in Sqare less than 0.05,
indicating that we can reject the hypothesis that the
scores follow a normal distribution. In contrast, for
a benign query qb,p >0.05, we cannot reject that
Sqbfollows a normal distribution.
Notably, when the maximum similarity score
smaxis removed from Sq, we observe p >0.05for
all queries, indicating that we cannot reject that Sq\
{smax}follows a normal distribution. As removing
smaxindicates the target document is removed in
the database, we cannot reject the hypothesis of
normality even for non-member attack queries qn
a.
The distributional difference of SqandSq\
{smax}suggests that smaxis extremely different
from values drawn from a normal distribution for
qm
a. Since the smaxis significantly different from
the Gumbel distribution for attack queries, leverag-
ing the confidence interval of the Gumbel distribu-

RAG System
OutputTarget
Document 𝒅∗
𝑮(𝒑(𝒒))Attack
Query𝒒𝒃Benign
Query
𝒒𝒂Benign 
User
AttackerPrivate External DB 𝓓
RAG Prompt 𝒑𝒒
Generator 𝑮
𝑹𝒌𝒒𝒒
+
Retriever 𝑹
Gumbel
Threshol d 𝝉
Top-𝒌
Detect
HideNon-member Query
Member Query
Figure 2: Illustration of our proposed Mirabel. We perform our detection to classify whether an input query is a
member attack query qm
a. If it detected as qm
a, we hide it from data retrieval and proceed standard RAG system.
tion makes it possible to detect qm
aand the target
document, enabling identification of MIAs.
4.2 Detect-and-hide to Defend MIA
Leveraging the proposed Mirabel for MIA detec-
tion, we propose a simple defense method that safe-
guards the data sample in the external database with
a simple detect-and-hide strategy. In the detection,
the query is evaluated using Mirabel to determine
whether it is a member attack query qm
a. In the hid-
ing, if the query qis classified as qm
a, the target
document identified by Mirabel is removed from
the retrieved set, and the standard RAG system pro-
ceeds with the remaining documents. If the query
is not classified as a member attack, the standard
RAG system proceeds without any modification.
Figure 2 illustrates our proposed defense method.
Primarily, this simple defense confuses the at-
tacker about whether the RAG output is from mem-
bers or non-members, since the response of the
retriever no longer includes the target document
in both cases. Consequently, the response distri-
bution remains indistinguishable between member
and non-member queries.
Our defense method, furthermore, preserves util-
ity for benign queries qb. When the query is not
detected as qm
a, the system behaves exactly the
same as the original RAG system. Unlike other
strong privacy-preserving mechanisms (e.g., differ-
ential privacy), which may degrade utility even for
benign queries, our approach introduces minimal
utility loss when the query is not detected as qm
a.
Finally, our safeguarding strategy is agnostic to
any RAG systems, as the detection only requires
similarity scores, which is a basic part of the RAG
system to calculate the top- kdocuments in theexternal database. Thus, it is easy to apply and
does not require any modification of the system.
The only additional step is to compute the Gumbel
distribution and hide the corresponding document
from the database for retrieval if the query is clas-
sified as qm
a. Based on this advantage, we show
that integrating our method into existing privacy-
preserving systems results in better defenses.
5 Experiments
5.1 Experimental Setups
Dataset To evaluate detection and defense per-
formance against MIAs, we utilize three datasets:
NFCorpus, SciDOCS, and TREC-COVID, drawn
from the BEIR benchmark (Thakur et al., 2021),
as employed in (Naseh et al., 2025). These corpora
represent sensitive scientific and medical domains.
In addition to evaluating the utility of the RAG sys-
tem on benign queries, we followed (Koga et al.,
2024) and used two question-answer benchmark
datasets: Natural Questions (NQ) (Kwiatkowski
et al., 2019) and TriviaQA (Joshi et al., 2017).
For both datasets, Wikipedia serves as the exter-
nal database (Lewis et al., 2020).
Baselines We evaluate performance for our detec-
tion and defense method using three MIA methods:
S2MIA (Li et al., 2024), MBA (Liu et al., 2025),
and IA (Naseh et al., 2025). S2MIA feeds the first
half of the target document to the RAG and scores
membership with BLEU and perplexity against the
full text. MBA masks tokens and counts how many
the generator recovers. IA lets an LLM craft 30
inference questions per document, and correctness
on them is the score.
For the comparison method of detection, we per-

form an agent-based detection using GPT-4o to dis-
tinguish between following the approach of (Naseh
et al., 2025), detailed in Appendix B.
As our work is the first work for defending
MIAs for RAGs, we choose to compare with the
most well-known privacy notion, DP (Dwork et al.,
2006), for guaranteeing the indistinguishability of
individual data in the private external database. For
DP-RAG Grislain (2024), we inject the noise in
the next token prediction based on DP in-context
learning (Tang et al., 2024) without considering the
privacy of top- kselection (Koga et al., 2024).
For the privacy budget ϵ, we set ϵ= 2for base
DP-RAG. To compare with a relaxed privacy set-
ting, we also set ϵ= 100 and referred to as DP-
RAG Large (in short, DP-RAG-L).
Metrics For the detection component, we mea-
sure accuracy, precision, recall, and F1-score to
determine how effectively the system identifies
queries targeting member documents.
To assess the utility of the system, we measure
(i) the exact match (EM), which indicates the pro-
portion of generated responses containing the gold
labels with top- kretrievals, and (ii) R@k, which
indicates the proportion of retrievals containing the
gold labels among the top- kresults.
For defense, we suggest the measures of attack
resistance andindistinguishability . The attack re-
sistance is measured based on the accuracy of the
attack. However, in MIA, since it is a binary classi-
fication, accuracy alone may underestimate attack
performance, especially when it falls below 0.5.
We instead define adjusted attack accuracy as:
max(acc, 1-acc) −0.5. (9)
This metric reflects the adversary’s advantage over
random guessing (0.5 in a balanced setting) and
ranges from 0 to 0.5. Lower values indicate weaker
attack success and, consequently, stronger defense.
To evaluate indistinguishability, we use the Kol-
mogorov–Smirnov (KS) test (An, 1933) to com-
pare the distributions of responses for member and
non-member queries. The KS statistic measures
the maximum difference between the two empiri-
cal distributions; a smaller value indicates greater
similarity, and thus higher indistinguishability.
RAG Setting and Implementation We em-
ployed a 1024-dimensional BGEm3 embedder (Li
et al., 2023) to find the similarity of queries and
documents in our retriever. Our retriever uses aMirabel, based on the significance parameter ρ, to
decide whether to filter documents before extract-
ing top- kcandidates via cosine similarity. In our ex-
periment, we set the significance level ρ= 0.05. In
evaluations concerning MIA detection and defense
on NFCorpus, SCIDOCS, and TREC-COVID, we
setk= 3. For utility evaluations on the Wikipedia
corpus, we tested k∈ {5,20,100}.
For the generator LLMs, we adopted Llama 3.2
3B Instruct and Llama 3.1 8B Instruct (Grattafiori
et al., 2024). Specifically, we used Llama 3.1 for the
utility evaluations and Llama 3.2 for the remaining
evaluations. Further details on the RAG prompt can
be found in the Figure 4. Experiments were mainly
conducted on a single A100 or H100 GPU with 96
GB VRAM.
Additional details are shown in Appendix A.
5.2 Detection Evaluation
Table 2 shows the detection performance compared
to agent-based methods. To detect member attack
queries, qm
awas labeled as 1 and otherwise 0, dif-
ferent from (Naseh et al., 2025) that labels both
attack queries qm
aandqn
aas 1. To balance the bi-
nary classification, we set the total number of qb
andqn
ato be the same as qm
a.
Mirabel detection shows stable performance
across all MIAs, even with the IA with stealth
queries, which the agent-based detection struggles
to detect. Notably, our method achieved a high re-
call score, indicating that it successfully detected
qm
a, while having slightly lower precision, which
misclassified qbasqm
a. In other words, our method
exhibits a lower Type II error, which is generally
considered more critical in attack detection.
In the next section, we will analyze how these
errors impact both attack performance and utility.
Note that while agent-based methods cannot clas-
sify a query as related to private members or not.
We also report the detection performance only on
comparing qbandqm
ain Appendix C.
5.3 Defense Evaluation
Safeguarding against MIA has two main goals: re-
ducing attack performance and preventing the at-
tacker from gaining private information, while pre-
serving the utility of RAG systems. Thus, we must
measure both utility preservation and attack degra-
dation for defense methods. Additional metrics are
provided in Appendix C.

Agent-based Detection Mirabel Detection
Attacks Data Acc (↑) F1 ( ↑) Precision ( ↑)Recall ( ↑)Acc (↑) F1 ( ↑) Precision ( ↑)Recall ( ↑)
S2MIANF 0.715 0.772 0.644 0.962 0.844 0.865 0.762 1.000
SCI 0.726 0.780 0.651 0.974 0.853 0.871 0.776 0.992
TREC 0.745 0.792 0.669 0.970 0.880 0.886 0.846 0.930
MBANF 0.727 0.783 0.650 0.986 0.847 0.867 0.766 1.000
SCI 0.730 0.785 0.652 0.986 0.855 0.873 0.775 1.000
TREC 0.745 0.794 0.667 0.980 0.995 0.995 1.000 0.990
IANF 0.482 0.012 0.125 0.006 0.799 0.750 0.896 0.817
SCI 0.485 0.030 0.258 0.016 0.827 0.772 0.928 0.843
TREC 0.500 0.000 0.000 0.000 0.720 0.744 0.670 0.705
Table 2: Detection performance of Mirabel compared to agent-based detection. For simplicity, we denote NFCorpus
as NF, SCIDOCS as SCI, and TREC-COVID as TREC throughout the following tables.
EM (↑) R ( ↑)
Data System @5 @20 @100 @5 @20 @100
NQ RAG 0.272 0.263 0.313 0.222 0.354 0.474
DP-RAG 0.030 0.051 0.020 – – –
Ours 0.253 0.253 0.273 0.172 0.303 0.443
TRIV RAG 0.730 0.725 0.755 0.575 0.705 0.840
DP-RAG 0.255 0.225 0.240 – – –
Ours 0.700 0.725 0.740 0.515 0.675 0.825
Table 3: Utility measures computed with benign queries.
Utility Preservation We first demonstrate that
our method largely avoids the utility degradation
associated with the existing privacy-preserving ap-
proach for RAG systems, DP-RAG, in Table 3.
EM@k, which represents the quality of the gen-
erated answers, shows that our method performs
comparably to the standard RAG, while DP-RAG
results in greater utility degradation.
In contrast, R@k, which measures how well the
system retrieves relevant documents, shows a mod-
erate decrease compared to the original RAG sys-
tem. This is because our detection method has a
slightly larger Type I detection error.
Despite degradation in R@k, our method main-
tains high performance in EM, because it retains
the remaining top- kdocuments (i.e., from top-2 to
top-(k+ 1)) by hiding only the target member doc-
ument. Since benign queries do not rely on a single
document but instead reference multiple relevant
documents, we can mitigate the utility loss.
Attack Resistance To evaluate the performance
of our defense, we measure the attack perfor-
mances of MIAs in Table 4. Without defense, MIAs
achieved high adjusted attack accuracy, indicating
the attacks are successful. However, after applying
the defenses, the accuracy decreases, suggesting
that the attacker struggles to distinguish whether
the response is from a member or a non-member.
Our method effectively reduces the attack accuracyNF SCI TREC Avg
S2RAG 0.188 0.126 0.119 0.144
MIA DP-RAG 0.021 0.015 0.004 0.013
DP-RAG-L 0.034 0.000 0.004 0.013
Ours 0.024 0.006 0.021 0.017
MBA RAG 0.377 0.406 0.313 0.365
DP-RAG 0.004 0.040 0.003 0.016
DP-RAG-L 0.014 0.036 0.184 0.078
Ours 0.019 0.101 0.139 0.086
IA RAG 0.403 0.380 0.254 0.346
DP-RAG 0.038 0.288 0.174 0.167
DP-RAG-L 0.267 0.305 0.020 0.197
Ours 0.008 0.010 0.083 0.034
Table 4: Adjusted attack accuracy ( ↓). A smaller value
indicates weaker attack success, thus stronger defense.
and achieves performance comparable to DP-RAG
(-L), which is considered a strong defense despite
its inherent privacy-utility trade-off.
From the perspective of detection failure, at-
tacker performance is related to the Type II error.
Type II errors occur when the detection method
fails to identify a member query, preventing the re-
moval of the target document. This can increase the
number of true positives, as the MIA will classify
members correctly as members.
Since our detection method has a low Type II
error rate, it results in a strong defense. However,
in the case of the IA on the TREC-COVID dataset,
even with a low recall and high Type II error, the
defense performance remains strong.
This is because the Type II error occurs when
the similarity score between the target document
dand the attack query qm
ais not significantly high.
In IA, such low similarity indicates that the attack
query was either a general question related to mul-
tiple documents or not strongly associated with the
target document. In these cases, the attack itself is
also likely to fail.

050100150200Member (Orig)
Non-Member (Orig)
20
 10
 0 10 20 30010203040Member (Defense)
Non-Member (Defense)(a) NFCorpus dataset.
0100200300400
20
 10
 0 10 20 300204060 (b) SCIDOCS dataset.
0100200
20
 10
 0 10 20 30020406080 (c) TREC-COVID dataset.
Figure 3: Histograms of IA scores for member and non-member queries across different datasets. After defense, the
member and non-member score distributions become indistinguishable.
NF SCI TREC Avg
S2RAG 0.358 0.195 0.203 0.252
MIA DP-RAG 0.077 0.051 0.038 0.055
DP-RAG-L 0.060 0.035 0.058 0.051
Ours 0.057 0.039 0.043 0.046
MBA RAG 0.754 0.813 0.625 0.731
DP-RAG 0.022 0.081 0.015 0.039
DP-RAG-L 0.036 0.073 0.026 0.045
Ours 0.038 0.202 0.283 0.174
IA RAG 0.805 0.771 0.555 0.710
DP-RAG 0.088 0.418 0.091 0.199
DP-RAG-L 0.301 0.414 0.223 0.313
Ours 0.100 0.081 0.172 0.118
Table 5: KS statistics ( ↓) for S2MIA, MBA, and IA
scores. For S2MIA not having a direct score, we report
the average KS statistics of similarity and perplexity.
Indistinguishability Most membership infer-
ence attacks rely on measuring a score and making
membership decisions based on that score. Even
if a defense method successfully reduces the at-
tacker’s accuracy, differences in the score distribu-
tions between member and non-member queries
enable attackers to perform adaptive attacks. There-
fore, another goal of defense is to achieve indis-
tinguishability between responses to member and
non-member queries, preventing the attacker from
gaining additional membership information.
Table 5 presents the results of the indistinguisha-
bility measure. As lower KS statistics indicate
higher distributional similarity, our method sig-
nificantly reduces the KS statistic, demonstrating
comparative or even smaller values compared to
DP-RAG.
To further illustrate this, we present histograms
of the attack scores produced by IA before and af-
ter applying our defense in Figure 3. In the original
RAG, we can observe the clear difference between
member and non-member queries. However, after
applying our defense, the two distributions are al-Adjusted Accuracy ( ↓) KS-statistics ( ↓)
NF SCI TREC NF SCI TREC
S2DP 0.021 0.004 0.038 0.077 0.051 0.038
MIA +Ours 0.020 0.029 0.024 0.088 0.028 0.050
MBA DP 0.015 0.040 0.288 0.022 0.081 0.015
+Ours 0.010 0.015 0.004 0.058 0.040 0.035
IA DP 0.004 0.003 0.174 0.088 0.418 0.091
+Ours 0.004 0.018 0.064 0.046 0.075 0.094
Table 6: Defense performances of DP-RAG and DP-
RAG with (+) our detect-and-hide using Mirabel.
DP-RAG DP-RAG + Ours
@5 @20 @100 @5 @20 @100
NQ 0.030 0.051 0.020 0.020 0.040 0.050
TRIV 0.255 0.225 0.240 0.230 0.230 0.225
Table 7: EM ( ↑) of DP-RAG and DP-RAG with ours.
most similar. The individual results for S2MIA and
distributions of attack scores for other methods are
provided in Appendix C.
5.4 Composing with Existing DP Models
In this section, we adapt our detect-and-hide strat-
egy to DP-RAG to demonstrate that our method is
agnostic to the RAG system. Table 6 presents the
defense performance when applying our method
on top of base DP-RAG ( ϵ= 2). As in the stan-
dard RAG setting, our method improved attack
resistance and indistinguishability, even though DP
already achieves strong defense performance. In
terms of utility, Table 7 shows that our method
introduces only minimal utility degradation. This
indicates that composing our method with exist-
ing or upcoming privacy-preserving models can be
beneficial. Additional results are in Appendix C.
6 Conclusion
In this paper, we propose Mirabel, a MIA detec-
tion method that classifies member attack queries

based on similarity using the Gumbel distribution.
We also introduce a simple defense strategy, detect-
and-hide, which effectively defends against attacks
with minimal utility degradation. Our method is
model-agnostic and can be easily applied to exist-
ing RAG systems. Experimental results show that
our method achieves defense performance compa-
rable to DP-RAG, while incurring negligible utility
loss on benign queries.
Limitations
Even though we have designed efficient detection
and defense methods for existing MIAs targeting
the private external database of the RAG system,
we need to be cautious when applying this defense
framework in real-world applications. In this work,
we only focus on small RAG systems. Deploying
larger models or API models could yield results
of greater scale or impact. Furthermore, we must
carefully keep watching this system for subsequent
attacks going forward.
Ethical Considerations
This work adheres to the ACL Code of Ethics. In
RAG systems, MIAs pose a serious threat to the
privacy of user queries submitted to LLMs. Miti-
gating these attacks, therefore, makes a meaningful
ethical contribution and offers practical guidance
for developing RAG systems that remain robust
against malicious attackers.
References
Kolmogorov An. 1933. Sulla determinazione empirica
di una legge didistribuzione. Giorn Dell’inst Ital
Degli Att , 4:89–91.
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2024. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation. arXiv preprint arXiv:2405.20446 .
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners. Advances in neural information
processing systems , 33:1877–1901.
Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu,
Dongyan Zhao, and Rui Yan. 2023. Lift yourself
up: Retrieval-augmented text generation with self-
memory. Advances in Neural Information Process-
ing Systems , 36:43780–43799.Haonan Duan, Adam Dziedzic, Nicolas Papernot, and
Franziska Boenisch. 2023. Flocks of stochastic par-
rots: Differentially private prompt learning for large
language models. In Thirty-seventh Conference on
Neural Information Processing Systems .
David Durfee and Ryan M Rogers. 2019. Practical dif-
ferentially private top-k selection with pay-what-you-
get composition. Advances in Neural Information
Processing Systems , 32.
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and
Adam Smith. 2006. Calibrating noise to sensitivity
in private data analysis. In Theory of cryptography
conference , pages 265–284. Springer.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2:1.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, and 1 others. 2024. The llama 3 herd of
models. arXiv preprint arXiv:2407.21783 .
Nicolas Grislain. 2024. Rag with differential privacy.
arXiv preprint arXiv:2412.19291 .
Emil Julius Gumbel. 1935. Les valeurs extrêmes des
distributions statistiques. In Annales de l’institut
Henri Poincaré , volume 5, pages 115–158.
Junyuan Hong, Jiachen T. Wang, Chenhui Zhang,
Zhangheng LI, Bo Li, and Zhangyang Wang. 2024.
DP-OPT: Make large language model your privacy-
preserving prompt engineer. In The Twelfth Interna-
tional Conference on Learning Representations .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions on Information
Systems , 43(2):1–55.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading compre-
hension. In 55th Annual Meeting of the Associa-
tion for Computational Linguistics, ACL 2017 , pages
1601–1611. Association for Computational Linguis-
tics (ACL).
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1) , pages 6769–6781.
Tatsuki Koga, Ruihan Wu, and Kamalika Chaudhuri.
2024. Privacy-preserving retrieval augmented gen-
eration with differential privacy. arXiv preprint
arXiv:2412.04697 .

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research. Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances
in neural information processing systems , 33:9459–
9474.
Jiacheng Li, Ninghui Li, and Bruno Ribeiro. 2021.
Membership inference attacks and defenses in classi-
fication models. In Proceedings of the Eleventh ACM
Conference on Data and Application Security and
Privacy , pages 5–16.
Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang.
2024. Generating is believing: Membership infer-
ence attacks against retrieval-augmented generation.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning. arXiv preprint arXiv:2308.03281 .
Mingrui Liu, Sixiao Zhang, and Cheng Long. 2025.
Mask-based membership inference attacks for
retrieval-augmented generation. In Proceedings of
the ACM on Web Conference 2025 , pages 2894–2907.
Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh
Chaudhari, Alina Oprea, and Amir Houmansadr.
2025. Riddle me this! stealthy membership infer-
ence for retrieval-augmented generation. In ACM
SIGSAC Conference on Computer and Communica-
tions Security .
Haritz Puerto, Martin Gubri, Sangdoo Yun, and
Seong Joon Oh. 2024. Scaling up membership in-
ference: When and how attacks succeed on large
language models. Preprint , arXiv:2411.00154.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo
Huang, Daogao Liu, Terra Blevins, Danqi Chen, and
Luke Zettlemoyer. 2024. Detecting pretraining data
from large language models. In The Twelfth Interna-
tional Conference on Learning Representations .
Reza Shokri, Marco Stronati, Congzheng Song, and Vi-
taly Shmatikov. 2017. Membership inference attacks
against machine learning models. In 2017 IEEE sym-
posium on security and privacy (SP) , pages 3–18.
IEEE.Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation. arXiv preprint
arXiv:2104.07567 .
Xinyu Tang, Richard Shin, Huseyin A Inan, Andre
Manoel, Fatemehsadat Mireshghallah, Zinan Lin,
Sivakanth Gopi, Janardhan Kulkarni, and Robert Sim.
2024. Privacy-preserving in-context learning with
differentially private few-shot generation. In The
Twelfth International Conference on Learning Repre-
sentations .
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogenous benchmark for zero-shot evalua-
tion of information retrieval models. arXiv preprint
arXiv:2104.08663 .
Toan Tran, Ruixuan Liu, and Li Xiong. 2025. To-
kens for learning, tokens for unlearning: Mitigat-
ing membership inference attacks in large language
models via dual-purpose training. arXiv preprint
arXiv:2502.19726 .
Rui Wen, Zheng Li, Michael Backes, and Yang Zhang.
2024. Membership inference attacks against in-
context learning. In Proceedings of the 2024 on ACM
SIGSAC Conference on Computer and Communica-
tions Security , pages 3481–3495.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and 1 others. 2024. The good
and the bad: Exploring privacy issues in retrieval-
augmented generation (rag). In Findings of the As-
sociation for Computational Linguistics ACL 2024 ,
pages 4505–4524.

A Experimental Details
A.1 Dataset
For attack and defense, we used NFCorpus, SCI-
DOCD, and TREC-COVID, containing approxi-
mately 3.6K, 171K, and 25K documents, respec-
tively. For the retriever database, NFCorpus is split
into member and nonmember sets in a 7:3 ratio,
while SCIDOCS and TREC-COVID are divided
8:2. To conduct MIA, we sample 1,000 members
and 1,000 nonmembers from NFCorpus, and 2,000
members and 2,000 nonmembers from the other
two datasets. Each member and nonmember sub-
set is further split (5:5) into reference and eval-
uation sets. Detection performance from benign
queries to attack queries is evaluated using 323,
500, and 50 queries provided in NFCorpus, SCI-
DOCS, and TREC-COVID, respectively, while an
equal number of MIA attack queries is generated
for the member subsets. This balanced design en-
sures a fair assessment of the detection and defense
capabilities.
For utility evaluation on benign queries, NQ
and TriviaQA are used, comprising approximately
7.8K and 138K question–answer (QA) pairs, re-
spectively. In our experiments, we select 200 QA
pairs from each dataset for evaluation.
All datasets and models used in this study are
publicly available and were released for academic
or research purposes.
A.2 Hyperparameters
Attacks For all datasets, we divided the mem-
ber and nonmember datasets into 5:5 for reference
and evaluation. For MBA and S2MIA, reference
sets were used to establish thresholds, following
their original paper. We found a threshold in the
greedy search algorithm for S2MIA and the high-
est F1 score for each count of masking tokens in
{5,10,15,20}in MBA.
For IA, we evaluate 30 queries to compute the
IA score for the standard RAG system, and 20
queries for DP-RAG, since each query consumes
a portion of the total privacy budget. Specifically,
with a total ϵ= 2, each query uses ϵ= 0.1, while
when ϵ= 100 , each query uses ϵ= 5.
In the selection of thresholds for IA, while the
original paper selects the threshold based on a false
positive rate, we instead select the threshold based
on accuracy. We searched threshold in [0,1], and
we evaluated penalty values λ∈[0.5,1]. In the
case of DP-RAG, since the number of queries issmaller, we differ the searching space. Since we
observed that larger values of λin this range re-
sulted in higher recall but significantly lower ac-
curacy, indicating potential over-penalization. We
therefore refined the search to a smaller range of
λ∈[0.1,0.5]for this setting.
RAG Setting Maximum sequence length was
set to 8192 tokens for embedding documents us-
ing BGEm3. For experiments on TREC-COVID
corpus, the maximum sequence length was set
to 2048 tokens involving DP generator. To facil-
itate efficient indexing and retrieval, we employed
FAISS. We segment the Wikipedia corpus into non-
overlapping chunks of 100 tokens and randomly
sample one-tenth of these segments to serve as the
retriever database. In building the Wikipedia re-
triever database, we utilized half-precision (fp16)
to reduce computational overhead. This approach
helps to assess the usability of RAG systems in a
controlled yet representative manner while main-
taining computational ease.
DP-RAG setting For DP-RAG, we follow the
setting provided in the official GitHub of (Gris-
lain, 2024) ( https://github.com/sarus-tech/
dp-rag ). Specifically, we set the parameters as
ω= 0.01,α= 1.0, and temperature = 1.0.
A.3 MIA implementation
To implement MIAs for evaluating Mirabel’s de-
fense, we followed each original papers. For
S2MIA, we calculated BLEU score and perplexity.
To gain perplexity, GPT-2 was used. For MBA, we
tokenize the sentences and selected mask tokens in
random. (Liu et al., 2025) showed random mask-
ing has similar accuracy in attempting attacks. For
IA, GPT-4o and GPT-4o-mini was used to generate
summaries, questions, and ground truths.
B Prompt Templates
Figure 4 shows the generation prompt given to
every RAG model (Llama-3.1 8B for utility exper-
iments and Llama-3.2 3B elsewhere). Its design
enforces three constraints that proved crucial for
reliable evaluation: groundedness, conciseness, and
failure transparency. Figure 5 gives the agent-based
detection prompt used with GPT-4o to label incom-
ing queries as either Natural orContext-Probing .
Both templates are frozen across all runs; we
do not tune any prompt hyperparameters. The
raw prompts introduced in the figures includes

placeholder tokens {context} ,{question} , and
{Query} to facilitate replication.
C Additional Experiments
C.1 Illustration of motivating example
Additional illustration of Figure 1b. We display the
top-1 similarity of qm
aandqn
ain Figure 6. As shown
in the Figure, our Gumbel threshold can efficiently
separate the top-1 similarity of qm
aandqn
a.
Since removing the top-1 similarity is equiv-
alent to removing the target document from the
database, the query can be treated as a non-member
attack query. Therefore, top-2 similarity as qm
acor-
responds to the top-1 similarity of qn
a.
C.2 Detection
The agent-based detection methods aim to de-
tect benign queries and attack queries, while our
method detects member attack queries. To ensure
a fair comparison, we evaluate detection perfor-
mance using only benign queries and member at-
tack queries.
Table 8 shows the detection performance. As
shown, the performance of Mirabel was similar to
that in Table 2, while agent-based methods demon-
strate significantly better performance, except for
IA, which is designed to be stealthy. These experi-
mental results are consistent with those reported in
Naseh et al. (2025).
Our method demonstrates stable performance
across a variety of attacks, particularly maintaining
reliable detection performance even for IA, which
exhibits stealthy characteristics.
C.3 Defense
In this subsection, we provide the performance of
MIA attackers, including accuracy, precision, re-
call, and F1 score for each attack. Table 9 presents
the attack performance for the standard RAG and
adapting our method to that RAG system. Table 6
shows the results for DP-RAG (-L) and DP-RAG
(-L) with our method in terms of the accuracy, pre-
cision, recall, and F1 score.
As discussed in the main paper, the accuracy
close to 0.5 means a stronger defense. Moreover,
if the attacker’s ability to classify a member as a
member (True Positive, TP) and a non-member as
a non-member (False Positive, FP) are similar, the
indistinguishability criterion is implicitly satisfied.Therefore, precision, which is defined as
TP
TP+FP, (10)
can be considered an indicator of indistinguisha-
bility, with a value close to 0.5 suggesting that
indistinguishability has been achieved.
Our detect-and-hide strategy achieved a value
close to 0.5 in nearly every accuracy and precision
measure.
Notably, in our method, we achieved a low recall
measure, which is defined as
TP
TP+FN. (11)
The detect-and-hide is designed to make the
attacker misclassify a member as a non-member.
Therefore, our method increases the false negatives,
making the attacker misclassify a member as a non-
member.
We provide the defense performances of DP-
RAG-L and DP-RAG-L with ours in Table 11.
C.4 Indistinguishable
In this subsection, we provide the additional indis-
tinguishability experiments. Specifically, S2MIA.
Since S2MIA does not have any specific score and
relies on the similarity score and perplexity score,
we reported the average of each experiment. The
individual results are in Table 12.
When calculating the KS statistic with perplex-
ity, some perplexity values were infinite and were
therefore excluded from the computation.
Additionally, we illustrate the distributions of
scores of MBA and S2MIA in Figures 7-9.

RAG Generator Prompt
System:
You are a helpful assistant. Below is a query from a user and some relevant contexts.
Answer the question given the information in those contexts. Your answer should be short and
concise.
If you cannot find the answer to the question, just say "I don’t know".
User:
Contexts: {context}
Query: {question}
Assistant:
Answer:
Figure 4: Prompt template given to the RAG generator. It conditions the model on the retrieved contexts and enforces
grounded, concise answers with an explicit "I don’t know" fallback.
Agent-based Detection Mirabel Detection
Acc Precision Recall F1 Acc Precision Recall F1
NF S2MIA 0.941 0.925 0.960 0.942 0.876 0.802 1.000 0.890
MBA 0.954 0.927 0.985 0.955 0.876 0.802 1.000 0.890
IA 0.464 0.074 0.006 0.011 0.824 0.783 0.895 0.835
SCI S2MIA 0.962 0.951 0.974 0.963 0.865 0.791 0.992 0.880
MBA 0.968 0.952 0.986 0.969 0.869 0.792 1.000 0.884
IA 0.483 0.242 0.016 0.030 0.816 0.773 0.894 0.829
Trec S2MIA 0.980 1.000 0.960 0.980 0.970 1.000 0.940 0.969
MBA 1.000 1.000 1.000 1.000 0.980 1.000 0.960 0.980
IA 0.500 0.000 0.000 0.000 0.810 1.000 0.620 0.765
Table 8: Detection performance of Mirabel compared to agent-based detection, evaluated using qnandqm
a.
Acc Precision Recall F1
S2MIA MBA IA S2MIA MBA IA S2MIA MBA IA S2MIA MBA IA
NF RAG 0.688 0.877 0.903 0.636 0.859 0.878 0.878 0.902 0.967 0.738 0.880 0.920
Ours 0.524 0.481 0.492 0.526 0.425 0.681 0.480 0.108 0.230 0.502 0.172 0.344
SCI RAG 0.626 0.906 0.868 0.647 0.914 0.884 0.555 0.897 0.847 0.597 0.906 0.865
Ours 0.506 0.601 0.510 0.510 0.789 0.548 0.301 0.276 0.117 0.378 0.409 0.193
TREC RAG 0.619 0.813 0.755 0.586 0.813 0.696 0.832 0.815 0.894 0.688 0.814 0.782
Ours 0.521 0.639 0.583 0.522 0.759 0.594 0.597 0.417 0.485 0.557 0.539 0.534
Table 9: Defense performance of RAG and RAG with our detect-and-hide defense.

Agent-based Detection Prompt
System:
You are an expert classifier tasked with validating user input queries in a Retrieval-Augmented
Generation (RAG) system. Your role is to classify each query into one of the following two
categories:
1. Natural Queries:
– Queries that are naturally formulated and relevant to the task/domain.
– These queries align with the system’s intended purpose, such as retrieving factual information,
summarizing content, or answering domain-specific queries.
2. Context-Probing Queries:
– Queries that attempt to probe hidden or inaccessible parts of the system or input context.
– These queries are often out-of-scope and unnatural for the task/domain. Examples include queries
that ask for metadata, hidden inputs, or system behavior.
— Instructions —
1. Carefully analyze the given query to determine whether it aligns with the system’s intended
purpose.
2. If the query is Natural, ensure it is appropriate and consistent with the system’s task/domain.
3. If the query is Context-Probing, identify if it is attempting to extract hidden content, system
behavior, or metadata.
— Output Format —
For each input query, respond with a single word indicating the classification:
Natural or Context-Probing. Do not provide any explanations or reasoning in your response.
— Examples —
Example 1:
Input Query: "What are the common symptoms of diabetes?"
Output:
Natural
Example 2:
Input Query: "What part of the document did you hide from me?"
Output:
Context-Probing
User:
Input Query: "{Query}"
Figure 5: Prompt template for the GPT-4o agent that classifies incoming queries as either Natural (task-aligned) or
Context-Probing (potentially MIA).

0.5 0.6 0.7 0.8 0.9
Similarity0510152025DensityGumbel ThresholdBenign
S2MIA Mem
S2MIA Non-Mem(a) Gumbel threshold for S2MIA.
0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80
Similarity0510152025
Gumbel ThresholdBenign
MBA Mem
MBA Non-Mem (b) Gumbel threshold for MBA.
0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85
Similarity0510152025
Gumbel ThresholdBenign
IA Mem
IA Non-Mem (c) Gumbel threshold for IA.
Figure 6: Illustration of the similarity distributions for member attack queries and non-member attack queries,
compared to benign queries. The Gumbel-based threshold is shown for each attack. Our method effectively separates
member and non-member attack queries using this threshold.
Acc Precision Recall F1
S2MIA MBA IA S2MIA MBA IA S2MIA MBA IA S2MIA MBA IA
NF DP-RAG 0.479 0.496 0.538 0.486 0.495 0.579 0.738 0.394 0.740 0.586 0.439 0.650
+Ours 0.480 0.529 0.477 0.487 0.537 0.572 0.748 0.426 0.383 0.590 0.475 0.459
DP-RAG-L 0.466 0.514 0.767 0.465 0.517 0.806 0.452 0.438 0.787 0.458 0.474 0.797
+Ours 0.486 0.507 0.496 0.486 0.508 0.620 0.478 0.432 0.336 0.482 0.467 0.436
SCI DP-RAG 0.515 0.540 0.788 0.538 0.560 0.834 0.214 0.381 0.719 0.306 0.453 0.772
+Ours 0.510 0.515 0.504 0.526 0.523 0.606 0.209 0.345 0.020 0.299 0.416 0.039
DP-RAG-L 0.500 0.536 0.805 0.500 0.557 0.790 0.719 0.354 0.830 0.590 0.433 0.809
+Ours 0.496 0.505 0.508 0.497 0.508 0.619 0.728 0.333 0.040 0.591 0.402 0.075
TREC DP-RAG 0.496 0.503 0.674 0.000 0.509 0.679 0.000 0.419 0.642 0.000 0.459 0.660
+Ours 0.496 0.482 0.564 0.000 0.484 0.587 0.000 0.403 0.390 0.000 0.440 0.468
DP-RAG-L 0.510 0.496 0.684 0.520 0.501 0.662 0.355 0.271 0.734 0.422 0.351 0.696
+Ours 0.509 0.505 0.576 0.519 0.517 0.580 0.352 0.282 0.507 0.420 0.365 0.541
Table 10: Defense performance of DP-RAG and DP-RAG with our detect-and-hide defense.
Adjusted Accuracy ( ↓) KS-statistics ( ↓)
NF SCI TREC NF SCI TREC
S2DP-RAG-L 0.034 0.000 0.004 0.060 0.035 0.058
MIA +Ours 0.014 0.004 0.009 0.051 0.025 0.035
MBA DP-RAG-L 0.014 0.036 0.184 0.036 0.073 0.026
+Ours 0.007 0.005 0.005 0.022 0.032 0.017
IA DP-RAG-L 0.267 0.305 0.020 0.301 0.414 0.223
+Ours 0.004 0.008 0.076 0.036 0.111 0.129
Table 11: Defense performances of DP-RAG-L and DP-RAG-L with (+) our detect-and-hide using Mirabel.
KS for Similarity ( ↓) KS for Perplexity ( ↓)
NF SCI TREC NF SCI TREC
RAG 0.378 0.254 0.239 0.338 0.137 0.167
Ours 0.064 0.024 0.048 0.050 0.053 0.038
DP-RAG 0.056 0.051 0.041 0.098 0.050 0.034
DP-RAG-L 0.038 0.032 0.054 0.082 0.039 0.062
Table 12: KS statistics for S2MIA using similarity and perplexity scores across different datasets. Lower is better.

0200400600800NFCorpus
Member (Orig)
Non-Member (Orig)SCIDOCS TREC-COVID
0.0 0.2 0.4 0.6 0.8 1.00200400600800
Member (Defense)
Non-Member (Defense)
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0Figure 7: Histograms of MIA scores for member and non-member queries across different datasets.
700800900NFCorpus
Member (Orig)
Non-Member (Orig)
0.00 0.01 0.02 0.03 0.04 0.05
Similarity010203040Density
400425450 Member (Defense)
Non-Member (Defense)
0.00 0.01 0.02 0.03 0.04 0.05010203040SCIDOCS
0.00 0.01 0.02 0.03 0.04 0.05
Similarity
0.00 0.01 0.02 0.03 0.04 0.05TREC-COVID
0.00 0.01 0.02 0.03 0.04 0.05
Similarity
0.00 0.01 0.02 0.03 0.04 0.05
Figure 8: Histograms of similarity scores from S2MIA for member and non-member queries across different
datasets.
0100200300400NFCorpus
Member (Orig)
Non-Member (Orig)SCIDOCS TREC-COVID
0 500 1000 1500 20000100200300400 Member (Defense)
Non-Member (Defense)
0 500 1000 1500 2000 0 500 1000 1500 2000
Figure 9: Histograms of perplexity scores from S2MIA for member and non-member queries across different
datasets. Due to the extremely large values of some perplexity scores, the histogram is truncated at 2000 for
readability.
# Differentially Private Retrieval-Augmented Generation

**Authors**: Tingting Tang, James Flemings, Yongqin Wang, Murali Annavaram

**Published**: 2026-02-16 00:52:57

**PDF URL**: [https://arxiv.org/pdf/2602.14374v1](https://arxiv.org/pdf/2602.14374v1)

## Abstract
Retrieval-augmented generation (RAG) is a widely used framework for reducing hallucinations in large language models (LLMs) on domain-specific tasks by retrieving relevant documents from a database to support accurate responses. However, when the database contains sensitive corpora, such as medical records or legal documents, RAG poses serious privacy risks by potentially exposing private information through its outputs. Prior work has demonstrated that one can practically craft adversarial prompts that force an LLM to regurgitate the augmented contexts. A promising direction is to integrate differential privacy (DP), a privacy notion that offers strong formal guarantees, into RAG systems. However, naively applying DP mechanisms into existing systems often leads to significant utility degradation. Particularly for RAG systems, DP can reduce the usefulness of the augmented contexts leading to increase risk of hallucination from the LLMs. Motivated by these challenges, we present DP-KSA, a novel privacy-preserving RAG algorithm that integrates DP using the propose-test-release paradigm. DP-KSA follows from a key observation that most question-answering (QA) queries can be sufficiently answered with a few keywords. Hence, DP-KSA first obtains an ensemble of relevant contexts, each of which will be used to generate a response from an LLM. We utilize these responses to obtain the most frequent keywords in a differentially private manner. Lastly, the keywords are augmented into the prompt for the final output. This approach effectively compresses the semantic space while preserving both utility and privacy. We formally show that DP-KSA provides formal DP guarantees on the generated output with respect to the RAG database. We evaluate DP-KSA on two QA benchmarks using three instruction-tuned LLMs, and our empirical results demonstrate that DP-KSA achieves a strong privacy-utility tradeoff.

## Full Text


<!-- PDF content starts -->

Differentially Private Retrieval-Augmented Generation
Tingting Tang
University of Southern California
Los Angeles, California, USA
tangting@usc.eduJames Flemings
University of Southern California
Los Angeles, California, USA
jamesf17@usc.edu
Yongqin Wang
University of Southern California
Los Angeles, California, USA
yongqin@usc.eduMurali Annavaram
University of Southern California
Los Angeles, California, USA
annavara@usc.edu
Abstract
Retrieval-augmented generation (RAG) is a widely used framework
for reducing hallucinations in large language models (LLMs) on
domain-specific tasks by retrieving relevant documents from a data-
base to support accurate responses. However, when the database
contains sensitive corpora, such as medical records or legal doc-
uments, RAG poses serious privacy risks by potentially exposing
private information through its outputs. Prior work has demon-
strated that one can practically craft adversarial prompts that force
an LLM to regurgitate the augmented contexts. A promising di-
rection is to integrate differential privacy (DP), a privacy notion
that offers strong formal guarantees, into RAG systems. However,
naively applying DP mechanisms into existing systems often leads
to significant utility degradation. Particularly for RAG systems,
DP can reduce the usefulness of the augmented contexts lead-
ing to increase risk of hallucination from the LLMs. Motivated
by these challenges, we present DP-KSA , a novel privacy-preserving
RAG algorithm that integrates DP using the propose-test-release
(PTR) paradigm. DP-KSA follows from a key observation that most
question-answering (QA) queries can be sufficiently answered with
a few keywords. Hence, DP-KSA first obtains an ensemble of rele-
vant contexts, each of which will be used to generate a response
from an LLM. We utilize these responses to obtain the most frequent
keywords in a differentially private manner. Lastly, the keywords
are augmented into the prompt for the final output. This approach
effectively compresses the semantic space while preserving both
utility and privacy. We formally show that DP-KSA provides formal
DP guarantees on the generated output with respect to the RAG
database. We evaluate DP-KSA on two QA benchmarks using three
instruction-tuned LLMs, and our empirical results demonstrate that
DP-KSAachieves a strong privacy-utility tradeoff.
Keywords
Differential Privacy, Retrieval-Augmented Generation, Large Lan-
guage Models, Data Privacy
This work is licensed under the Creative Commons Attribu-
tion 4.0 International License. To view a copy of this license
visit https://creativecommons.org/licenses/by/4.0/ or send a
letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
Proceedings on Privacy Enhancing Technologies YYYY(X), 1–15
©YYYY Copyright held by the owner/author(s).
https://doi.org/XXXXXXX.XXXXXXX
Private External 
Database 𝐷
Generator 
Model 𝐹
Retrieved 
DocumentsQuery 𝒙
Response 𝒚
Generator 
Model 𝐹
Pre-training 
DataRAG
Pre-trainingAdversarial
bound
Retriever 
Model 𝑅Figure 1: Overview of the DP RAG problem setting. The ad-
versarial bound illustrates what capabilities the adversary
has. In this case, the adversary can only query the RAG sys-
tem and access the answer. The generator model has been
pre-trained on publicly available data, which the adversary
has access to. However, we are not concerned about preserv-
ing the privacy of the pre-training data.
1 Introduction
Large language models (LLMs) encode copious factual knowledge
in their parameters through pre-training on internet-scale data [ 6].
Hence, LLMs leverage its knowledge when prompted to accurately
answer queries [ 28,30]. However, the knowledge that the LLM was
pre-trained on (1) may not be precisely accessed and utilized and
(2) can eventually become outdated. This can lead to discrepancies
between generated content by the LLM and verifiable real-world
facts, known asfactuality hallucinations[16].
A widely-adopted approach to mitigate hallucinations is to up-
date the LLMs’ knowledge base by augmenting prompts with ex-
ternal knowledge retrieved from a database, known as retrieval-
augmented generation (RAG) [ 2,22]. This involves retrieving the
top documents from an external database that are semantically rele-
vant (similar) to a user query. However, a major concern is that the
external database can contain highly-sensitive information, such as
Personable Identifiable Information (PII). For example, healthcare
providers can leverage internal medical records to offer accurate
diagnoses and tailored care recommendations, while law firms may
rely on their legal case repositories to support clients in conducting
legal research and preparing documentation. Several have found
1arXiv:2602.14374v1  [cs.CR]  16 Feb 2026

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
that RAG on a sensitive corpus can leak private information about
individual documents in the corpus [ 17,29,36]. An attacker can
design a specific prompt to a RAG system and use the output to
reveal the information retrieved from the sensitive database.
In this work, we focus on preserving the privacy of the external
dataset with the following problem setup shown in Figure 1. A
RAG system receives a queryx. Our RAG system contains a private
external dataset 𝐷, a retriever model 𝑅that will retrieve the top
documents relevant to the queryx, and a generator model 𝐹that
will use the top retrieved documents and the query to generate
a responsey. Our setup assumes the adversary only has query
access to the RAG system to obtain responses. Hence, because the
response can contain information about the retrieved documents,
the adversary can adversarially craft prompts such that the response
reveals information about the external database. Therefore, our goal
is to design a privacy-preserving RAG system to protect private,
sensitive information against such attacks.
One promising privacy notion is Differential Privacy (DP) [ 11],
which provides a mathematical guarantee that each individual in a
database has limited affect on the outcome of a randomized algo-
rithm. In the context of RAG, each document in the RAG database
has limited influence on the output of the LLM. Although DP is the
standard privacy safeguard for applications utilizing information-
sensitive data, it tends to result in substantial utility degradation.
To highlight the challenges that privately generating text presents,
note that the response from the generator model 𝐹iteratively sam-
ples the next token 𝑦𝑡until a stop criteria is met, such as maximum
token length reached. Hence, the range of possible values that 𝑦𝑡
can take is equal to the vocabulary of the generator’s tokenizer,
which can be upwards of50 ,000. And if the maximum token length
is10, then the output space of possible responses is at most50 ,00010.
Hence, naively applying standard DP mechanisms, such as addi-
tive Gaussian noise, at every token generation could destroy any
meaningful utility from the generated tokens.
Thus, the goal our work is the following:
How can we integrate differential privacy into practical
RAG systems while preserving their utility?
To address the above question, we propose DP-KSA , a privacy-
preserving algorithm based on the propose-test-release (PTR) [ 12]
and subsample-and-aggregate [ 27] paradigm for private text gen-
eration illustrated in Figure 2.DP-KSAderives from a key observa-
tion that most queries from question-answering datasets can be
answered sufficiently with just a few keywords. Hence, we con-
vert the output space of responses for question-answering tasks
into a keyword subspace, then perform differentially privacy to
extract the keywords. This keyword subspace mostly preserves
the relative semantic representation of the original response space,
i.e. it maintains the utility of the LLM while operating in a low-
dimensional approximation of the entire sentence space. To achieve
this, a retriever model will first retrieve the top- 𝑁documents that
are most relevant to the query. Next, we partition the documents
into prompts, each prompt containing a retrieved document with
the corresponding query, and feed these prompts into the generator
model to obtain an ensemble of responses. Then DP-KSA transforms
the generated model responses into keywords that preserve relative
semantic meaning and privately extracts the frequently occurringkeywords in the responses via the propose-test-release paradigm.
Finally, the privately obtained keywords are augmented to a prompt
containing just the query, which is then fed to the LLM to generate
the final output.
We summarize thecontributions of our workas follows:
(1)We introduce DP-KSA , a simple RAG framework that pre-
serves the privacy of the external database by generating a
response for each of the retrieved documents from the re-
triever model, then extracting a small set of keywords from
the ensemble of responses.
(2)We formally show that the final outputs generated by DP-KSA
can achieve differential privacy, a strong privacy notion that
gives a probable guarantee of the privacy leakage of the
external database.
(3)We experimentally demonstrate that DP-KSA can achieve
strong privacy guarantees while preserving utility. In partic-
ular, we experimentally evaluated DP-KSA on two standard
benchmarking question-answering datasets with modern
instruction-tuned LLMs, Qwen 2.5 [34] and Llama 3 [13].
(4)We provide comprehensive ablation studies for important
hyperparameters of DP-KSA to demonstrate the robustness
ofDP-KSA as well as provide insights into the inner workings
ofDP-KSA.
2 Preliminary
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) is a hybrid architecture
that augments a large language model (LLM) with an external
retrieval mechanism to improve its performance on knowledge-
intensive tasks. Given a user queryx, a retriever model 𝑅is first
used to fetch the top- 𝑁most relevant documents from a private
corpus𝐷. Formally, the retriever returns a set of documents
𝐷x
1,...,𝐷x
𝑁←𝑅(x,𝐷;𝑁)(1)
where each𝐷x
𝑖∈𝐷is deemed semantically similar to the queryx.
Semantic similarity is typically assessed by computing the distance
between the vector representations of a query and a document.
Commonly used distance functions include the dot product and
cosine similarity. Each document 𝐷x
𝑖is then paired with the query
and passed into a generator model 𝐹, which synthesizes a response:
𝑦𝑖←𝐹(𝐷x
𝑖,x)(2)
The outputs 𝑦𝑖are natural language responses that reflect the LLM’s
understanding conditioned on both the query and the retrieved
document.
RAG offers several advantages over standalone language mod-
eling. By retrieving top- 𝑁relevant documents 𝐷x
1,...,𝐷x
𝑁from a
corpus𝐷using a retriever 𝑅, and conditioning generation on these
documents, RAG enables dynamic grounding without retraining
the generator 𝐹. This allows models to incorporate up-to-date or
domain-specific information at inference time, which is especially
valuable when 𝐷is frequently updated or too large to encode di-
rectly into model parameters. A key benefit of RAG is its ability to
reduce hallucinations. Since outputs 𝑦𝑖=𝐹(𝐷x
𝑖,x)are generated
with direct reference to retrieved content, responses are more likely
to be factually accurate. RAG is also modular as 𝑅and𝐹can be
2

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Figure 2: The proposedDP-KSAframework consists of three steps. First, it retrieves the top-𝑁documents most relevant to the
queryxfrom the private database 𝐷. Each retrieved document 𝐷x
𝑖is paired with queryxand passed to the generator model 𝐹to
produce responses. Next, DP-KSA applies a differentially private mechanism to extract the most frequent keywords from the
ensemble of responses. Finally, the selected keywords are combined with queryxand fed back into the generator 𝐹to produce
the final outputy.
trained or fine-tuned independently, enabling flexible adaptation
across domains.
2.2 Differential Privacy
Differential privacy (DP) is the gold standard for reasoning about
the privacy of machine learning algorithms.
Definition 2.1((𝜖,𝛿)-DP [ 11]).For𝜖≥0,𝛿∈[ 0,1], a randomized
algorithm𝑀is(𝜖,𝛿) -differentially private if for any pair of adjacent
datasets𝐷and𝐷′that differ in only one data point, it holds that
𝑃𝑟[𝑀(𝐷)∈𝐸]≤𝑒𝜖·𝑃𝑟[𝑀(𝐷′)∈𝐸]+𝛿
The above definition indicates that if two datasets are similar, a
DP algorithm should produce similar output 𝐸with a high proba-
bility so that attackers cannot infer the difference between them.
In our case, 𝑀functions as a RAG algorithm, producing answers to
queries by utilizing private retrieved document as context. If this
RAG algorithm adheres to differential privacy, it should generate
similar outputs even when the retrieved documents vary. Conse-
quently, this prohibits the generation of private information, such
as replicating the retrieved context.
2.2.1 Private Generation via Sample-and-Aggregate.There has been
a body of work on generating token sequences from an LLM with
DP, all of which rely on the idea of the sample-and-aggregate frame-
work in DP [ 27]. In this framework, the sensitive dataset is parti-
tioned into pairwise disjoint subsets. Then, the model will perform
inference on each subset to generate a response. Lastly, the re-
sponses are privately aggregated together using a DP mechanism[8,31,33], such as adding Gaussian noise to the aggregation. The
reason for using the sample-and-aggregate framework is that it
helps with calculating the global sensitivity, which is an important
property needed for DP mechanisms. If we have two neighboring
datasets𝐷and𝐷′that differ by one document 𝑑𝑖, then at most one
subset will differ since the document can only be contained in at
most one subset (due to the pairwise disjoint property of sample-
and-aggregate). Hence, changing one document changes at most
one response, which we can then use to argue the global sensitivity.
2.2.2 Private top- 𝑘selection.The private top- 𝑘selection problem
is one of the most fundamental problems in privacy-preserving
data analysis. The problem is given a set of candidates with corre-
sponding counts, return the top- 𝑘candidates based on the counts
in a privacy-preserving way [ 25]. In this work, we adopt an adap-
tive top-𝑘selection algorithm from Zhu and Wang [39], which is
based on the widely-known propose-test-release (PTR) framework
[12]. The idea is that as long as the count difference between the
𝑘-th and the ( 𝑘+1)-th highest candidates is larger than 2, we can
non-privately release the top-𝑘candidates.
3 Problem Settings
In this section, we describe our problem settings and the threat
model.
Problem Settings.Following the problem setting in Figure
1, suppose a RAG system receives an input queryxand wants to
generate a response using a generator model 𝐹. Typically𝐹is a
decoder-only model such as LLaMA or Qwen. Also suppose the
3

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
Figure 3: Histogram of token lengths of ground truth answers
in NQ dataset.
Figure 4: Histogram of token lengths of ground truth answers
in TQA dataset.
system has access to private external database 𝐷to assist with
answering the query. The goal is to generate the responseysuch
that it is(𝜖,𝛿) -DP (definition 2.1) with respect to a private RAG
database𝐷.
Threat Model.Additionally, our work assumes a realistic setup
where an adversary can only query the RAG system with any
promptx. The adversary has access only to (1) the responsey
and (2) the generator model 𝐹. Consequently, the adversary does
not have access to the external database 𝐷. We can assume the
adversary has access to the generator model, as this is usually
publicly available to download from the internet. That is, the pre-
training (and fine-tuning) data used to pre-train (and fine-tune)
the generator LLM model is considered public and we are only
concerned about preserving the privacy of the RAG database𝐷.
4 Methodology
In this section, we introduce DP-KSA , a DP RAG framework. We
begin by discussing the motivation behind the design of DP-KSA
in Section 4.1. Then we will go through DP-KSA in more detail in
Section 4.2.
4.1 Motivation
The challenge of generating differentially private text lies in the
high dimensionality of the output space. The generator model
𝐹typically generates text autoregressively, i.e. it iteratively sam-
ples the next token 𝑦𝑡+1from𝐹by using the previously sampled
tokens𝑦1,...,𝑦𝑡plus some provided context 𝑐to obtain𝑦𝑡+1∼𝐹(𝑦𝑡+1|𝑐,𝑦 1,...,𝑦𝑡). Then we feed 𝑦𝑡+1back to the model to sam-
ple the next token until the desired stop criteria is met (either max
token length reached or eos token sampled). However, the number
of possible tokens that 𝑦𝑡+1can realize is equal to the vocabulary
space of the generator model’s tokenizer, which could be upwards
to50,000. If the max generation length of the responseyis10, then
the output space of possible responses is50 ,00010. Hence, naively
applying DP noise each time we are generating the next token
results in a large magnitude of noise, which can have a deleterious
effect on the generated tokens and quickly destroy the utility.
To overcome the large dimensionality of private text generation,
the design of DP-KSA derives from a key observation that most
question-answering datasets can be answered sufficiently with just
a few keywords. To illustrate this, consider the following question-
answering example from the Natural Question (NQ) dataset [ 21], a
widely-used RAG benchmarking dataset:
Question:who lives in the imperial palace in tokyo?
Ground Truth Answer:the Imperial Family
From this example, we see that the ground truth only contains
three words (i.e., three tokens). Hence, extracting the keywords
"Imperial" and "Family" would completely preserve the semantic
meaning of the ground truth answer. To demonstrate that this
observation holds more generally, Figure 3 and Figure 4 show the
histograms of token lengths of ground truth answers in NQ dataset
and Trivia Question-Answering (TQA) dataset [ 18], respectively.
As we can see, the histograms are rightly skewed where most of
the ground-truth answers contain only one to four tokens. The
main takeaway here is that we can convert the task of accurately
answering these questions into obtaining only a small set of correct
tokens. Such a conversion effectively converts the QA task into
a keyword space which can be considered as a low-dimensional
approximation of the entire sequence space of correct answers.
Consequently, operating in this lower dimensional space will help
us preserve the utility as we integrate differential privacy into this
space.
4.2DP-KSA
As shown in Figure 2, at a high level, our algorithm proceeds in three
steps: (1) retrieval and partitioning, (2) DP keyword extraction, and
(3) zero-shot with DP information inference. Algorithm 1 succinctly
describesDP-KSAand we go through the details below.
Retrieval and partitioning.First, we use a retriever model 𝑅
to obtain the top 𝑁documents from the private database 𝐷that are
most relevant to the queryx(line 1). Specifically, each document in
the database is encoded into a dense vector representation by the
retriever𝑅. At query time, the retriever encodes the input query
into its own dense vector and ranks documents based on a similarity
measure, typically the dot product or cosine similarity, between the
query and document vectors. This process identifies the documents
most semantically similar to the query.
Then, for each retrieved document 𝐷x
𝑖, we partition them into
chunks with each chunk containing a retrieved document and the
input query. The document is augmented with the user query to
4

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Algorithm 1DP-KSA
Require: Generator model 𝐹, retriever model 𝑅, private database
𝐷, number of top documents 𝑁, user queryx, max number of
generated tokens𝑇 max
Ensure:Generated responsey
1:𝐷x
1,...,𝐷x
𝑁←𝑅( x,𝐷;𝑁)Retrieve𝑁most relevant documents
for queryx
2:for1≤𝑖≤𝑁do
3:y𝑖←𝐹(𝐷x
𝑖,x)
4:end for
5:Form histogramHby counting the number of responsesy 𝑖that
contain each token.
6:ˆ𝑘=FindBestK(H)select optimal number of keywords
7:𝑤 1,...,𝑤 ˆ𝑘←TopKWithPTR( ˆ𝑘,H)obtain keywords
8:y←𝐹(x,{𝑤 𝑖}ˆ𝑘
𝑖=1)zero-shot with keywords inference
9:Returny
Algorithm 2TopKWithPTR
Require:𝑘 – the number of top counted tokens to release;H–
histogram for the counts of each token; 𝛿– failure probability
1:Set𝑑𝑘:=H(𝑘)−H(𝑘+1) .
2:Setb𝑑𝑘:=max(2,𝑑 𝑘)+N(0,4𝜎2)−Φ(1−𝛿; 0,2𝜎).
3:Ifb𝑑𝑘>2,Returnthe exact top-𝑘tokens.
4:ElseTerminate with no keywords.
Algorithm 3FindBestK
Require:H– histogram for the counts of each token
1:Compute histogram gap 𝑑𝑘:=H(𝑘)−H(𝑘+1) for each𝑘=
1...𝑁−1.
2:Returnargmax𝑘{𝑑𝑘+𝑟(𝑘)+Gumbel(2/𝜖)}
produce a prompt, which is fed to the generator model 𝐹to generate
a response (line 3). Afterwards, we will obtain𝑁responses.
DP keyword extraction.Next, we extract keywords from the
ensemble of responsesy 𝑖∀𝑖to be used for the final prompt. The
intuition behind this approach is that the answers for QA datasets
typically contain a few "correct" tokens. Hence, if the retrieved
documents are relevant to the query, it is likely that the correct
tokens are contained in the ensemble of responses generated based
on different disjoint retrieved documents. This is inspired from
prior work’s use of PTR for DP text generation [33].
The key idea is that we convert keyword extraction into a private
top-𝑘selection problem, then use existing solutions in this space.
To this end, we form a histogramHby counting the frequency
of each word token among the responses based on the individual
retrieved documents. Then we use the histogram to obtain and
release the top- 𝑘tokens with the highest counts in a DP-manner.
Specifically, first we adaptively estimate the optimal ˆ𝑘to use (line
6), then extract the top- ˆ𝑘keywords usingTopKWithPTR(line 7).
One subtle limitation is that applying a private selection algo-
rithm, such as the exponential mechanism (EM) [ 25], for every
keyword will require applying the EM 𝑘times, which will decay
the privacy loss 𝜖. Hence, TopKWithPTR only needs to be appliedonce to obtain the top- ˆ𝑘by testing ifH (𝑘)−H(𝑘+1) >2. If the test
holds, then the top- 𝑘indices are exactly the same for all the neigh-
boring datasets, so we can release the exact top- 𝑘indices without
any privatization. However, the testH (𝑘)−H(𝑘+1) >2must be
performed in a differentially private way, which is done with PTR
(Algorithm 2). If the test fails, then no tokens can be release and
hence the final prompt will resort to zero-shot.
Moreover, the utility of TopKWithPTR is highest when 𝑘is chosen
to maximizeH (𝑘)−H(𝑘+1) . Hence, ˆ𝑘is estimated in a data-dependent
way by releasing argmax𝑘H(𝑘)−H(𝑘+1) using EM (Algorithm 3).
Here,𝑟(𝑘)is a regularizer independent of the dataset, e.g., we can
set𝑟(𝑘)=−∞ for any𝑘> 30and𝑘< 15, if we don’t want to
return more than 30 or less than 15 tokens.
Zero-shot + DP info inference.Lastly, we use the DP released
top-ˆ𝑘tokens along with the user queryxto generate the final
response from the generator modely ←𝐹( x,{𝑤𝑖}ˆ𝑘
𝑖=1)(line 9).
Note that the final response does not explicitly use the retrieved
documents for the final response, only the extracted keywords. If
the test in the TopKWithPTR fails, then the final response will not
contain any information from the external database.
5 Privacy Analysis
We now provide a formal guarantee of DP-KSA , in particular that it
achieves(𝜖,𝛿) -DP. We provide a proof sketch below and defer the
exact details in Appendix A.
Theorem 5.1.Letxbe a query received. Suppose we generate a
response tox, denoted asy, using DP-KSA (Algorithm 1) with 𝐹as
the generator model, 𝑅as the retrieval model, and 𝐷as the private
database. ThenDP-KSAsatisfies(𝜖,𝛿)-DP with respect to𝐷.
Proof sketch.The high-level idea of the privacy analysis is that the
final responseyis a function of the extracted keywords 𝑤1,...,𝑤 ˆ𝑘.
The keywords were obtained from the ensemble of responses, which
depend on the retrieved documents from the private dataset, and
an estimate for the number of keywords ˆ𝑘, which also depends
on the responses. Hence, it suffices to show that (1) FindBestK re-
turn ˆ𝑘that is differentially private with respect to 𝐷, then show (2)
TopKWithPTR is differentially private with respect to 𝐷. In regards
to (1), we use the exponential mechanism to achieve differential
privacy on ˆ𝑘. And (2) achieves differential privacy using standard
analysis of PTR framework. For both (1) and (2), the global sensitiv-
ity of the utility function 𝑑𝑘is2because that for two neighboring
external datasets 𝐷and𝐷′they differ by one document. Subse-
quently, they will differ by one retrieved document, say the 𝑖-th
𝐷x
𝑖≠𝐷′x
𝑖. Hence the sensitivity of the utility function 𝑑𝑘is2. Once
we prove these two claims, then we can invoke the post-processing
theorem of DP– stating that a DP quantity does not leak additional
privacy about the dataset– to argue that the final response is DP.
Remark5.2.The formal proof from Appendix A is relatively straight-
forward, as it relies on mostly well-established properties and theo-
rems. However, we utilize Renyi Differential Privacy (RDP) [ 26] to
perform our privacy analysis. Hence, we formally introduce and
define these properties in terms of RDP in the Appendix due to
spatial constraints.
5

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
6 Experiments
6.1 Experimental Setup
Dataset.To evaluate the effectiveness of our proposed method, we
conduct experiments on two widely used benchmark datasets in the
RAG literature, Natural Questions (NQ) [ 21] and Trivia Question
Answering (TQA) [ 18]. Both datasets are designed for open-domain
question answering and consist of a diverse set of real-world ques-
tions, each associated with multiple reference answers. Following
standard RAG evaluations [ 5,15,22], we use the Wikipedia corpus
as our external knowledge source for retrieval. Due to practical
constraints in computational resources, we follow prior work [ 20]
and use a subset of 100 questions from each dataset. Note that we
filter out questions with empty ground-truth references in both
datasets in order to correctly compute the evaluation metrics.
Model Architectures.Our system follows a standard retrieval-
augmented generation (RAG) pipeline, composed of a retriever and
a generator. For the retriever component, we adopt the Dense Pas-
sage Retriever (DPR) [ 19], a widely used dual-encoder architecture
built on top of BERT [ 6]. DPR encodes both the input question
and the candidate passages into dense vector embeddings and re-
trieves the most relevant documents by measuring similarity in
the embedding space. In our experiment, we use the inner prod-
uct as the similarity metric. In addition, given the large size of
the Wikipedia corpus (21 million passages), we use FAISS [ 7] to
accelerate retrieval. FAISS leverages approximate nearest neigh-
bor algorithms for efficient similarity search in high-dimensional
embedding spaces.
For the generator, we compare several state-of-the-art large
language models that have been instruction-tuned to follow user
prompts. Specifically, we evaluate Qwen 2.5 (3B) [ 34], Llama 3.2
(3B), and Llama 3.1 (8B) [ 13]. The instruction-tuned versions of
these models are particularly appropriate for QA tasks, as they are
optimized to respond effectively to task-specific prompts.
Baselines.We evaluate the effectiveness of DP-KSA (Algorithm
1) by comparing it against three representative baselines, each
reflecting a different level of privacy and retrieval capability.
The first baseline, denoted asnon-RAG (𝝐=0) , represents a
strictly private setup where only the question is provided to the
language model, without any access to external documents. Since no
retrieval is involved, this configuration guarantees inherent privacy
and serves as a minimal baseline. For DP-KSA to be considered
practically useful, it must achieve better performance than this
no-retrieval baseline under reasonable privacy budgets.
The second baseline, labeled asKSA (𝝐=∞) , performs the same
keyword extraction as DP-KSA but releases the top- 𝐾keywords
without introducing any noise. This non-private variant effectively
removes the privacy constraints while preserving the aggregation
structure of our method, making it a strong upper bound for our
privacy-preserving approach.
The third baseline is the standard retrieval-augmented genera-
tion pipeline, denoted asRAG (𝝐=∞) . In this setting, the ques-
tion is paired with the top-2 retrieved documents from the external
knowledge base, and the full documents are fed directly into the
language model without any privacy protection. This setting repre-
sents a theoretical upper bound in terms of utility, as it leverages
full retrieval with no noise or low-dimensional approximation.Settings.We follow previous work [ 20] and set𝛿=10−4. We
select𝜖={ 1,2,3,5,8}to achieve different levels of privacy. We set
the number of ensembles to 80 for our method DP-KSA and non-
private KSA method, which is found to be the optimal number of
ensembles and detailed later in the ablation studies in section 6.3.2 .
Metrics.Following prior works [32, 33], we adopt four widely
used evaluation metrics, F1, ROUGE-1, ROUGE-L and normalized
Levenshitein similarity. These metrics collectively capture both lexi-
cal overlap and semantic alignment between generated outputs and
ground-truth references. The F1 score computes the harmonic mean
of precision and recall, and is particularly well-suited for QA tasks
where partial correctness is meaningful. It reflects how well the
predicted answer tokens align with those in the reference, reward-
ing both completeness and accuracy. ROUGE-1 measures unigram
(i.e., single word) overlap between the prediction and the reference
text. This provides a straightforward measure of lexical similarity.
ROUGE-L, on the other hand, incorporates the sequential nature of
language by computing the longest common subsequence (LCS),
thus capturing not only which words are shared, but also whether
they appear in a similar order, which is an important indicator
of fluency and coherence in natural language generation. Leven-
shtein similarity is derived from the Levenshtein distance, which
quantifies the number of single-character insertions, deletions, or
substitutions required to transform one string into another. By nor-
malizing this value, we obtain a score that reflects how closely the
predicted answer resembles the reference at the character level,
offering a fine-grained perspective on textual similarity. Across all
four metrics, higher scores indicate better alignment between the
generated and ground-truth answers, and thus better performance.
6.2 Privacy-Utility Tradeoff ofDP-KSA
We show the privacy-utility tradeoff of DP-KSA across four evalu-
ation metrics and three different LLMs on NQ and TQA datasets.
We evaluate DP-KSA with different privacy budget 𝜖, ranging from
1 to 8. Smaller 𝜖indicates stronger privacy constraints. We also
compare our method with the non-RAG ( 𝜖=0), non-private KSA
(𝜖=∞) and non-private RAG (𝜖=∞) baselines.
6.2.1 Performance on NQ dataset.Figure 5 presents the perfor-
mance of DP-KSA on the NQ dataset. Across all models and metrics,
we observe a consistent performance improvement as the privacy
budget𝜖increases from 1 to 8, confirming that DP-KSA effectively
trades off privacy for utility. Notably, at moderate privacy levels
(e.g.𝜖=3 or 5), DP-KSA substantially outperforms the strictly pri-
vate non-RAG baseline, demonstrating its ability to retain useful
keywords even under noise. For instance, with Llama 3.2 (3B) as the
generator, the F1 score improves from 21.67 at 𝜖=0to 22.55 at𝜖=2,
and further to 25.18 at 𝜖=8, highlighting a strong privacy-utility
tradeoff. However, at 𝜖=1, performance tends to drop below or re-
main comparable to the non-RAG baseline ( 𝜖=0). This is likely due
to the higher noise level required under strong privacy constraints,
which hampers the success of the propose-test-release (PTR) con-
ditionH(𝑘)−H(𝑘+1) >2, resulting in fewer or no keywords being
released to support final response generation.
We find that DP-KSA often matches or even surpasses the perfor-
mance of the non-private KSA ( 𝜖=∞ ) baseline, despite the presence
of differential privacy noise. For example, using Qwen 2.5 (3B) as
6

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Figure 5: Results of DP-KSA on NQ dataset with different generator LLMs: Qwen 2.5 (3B), Llama 3.2 (3B), and Llama 3.1 (8B). We
use three baselines including non-RAG ( 𝜖=0), non-private RAG with top-2 retrieved documents ( 𝜖=∞ ), and non-private KSA
(𝜖=∞).
the generator, DP-KSA achieves ROUGE-1 and ROUGE-L scores at
a moderate privacy budget ( 𝜖=5) that are comparable to or exceed
those of non-private KSA. This suggests that a noisy histogram,
constructed across ensembles of multiple responses, effectively
preserves key semantic contents even in the presence of differ-
ential privacy noise. Additionally, unlike non-private KSA which
deterministically selects the most frequent keywords, DP-KSA ’s sto-
chastic keyword release mechanism introduces variation acrossruns, which may help reduce overfitting to overly dominant but po-
tentially less informative tokens. Overall, these findings underscore
that controlled randomness applied over an ensemble of semanti-
cally coherent outputs enables strong utility-privacy tradeoffs.
It is also worth noting that the non-private KSA baseline gen-
erally falls short of the upper bound set by the non-private RAG
(𝜖=∞ ) baseline. This performance gap is likely stems from the
information loss inherent in compressing an ensemble of responses
into a fixed set of keywords. Exploring more effective strategies
7

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
Figure 6: Results of DP-KSA on TQA dataset with different generator LLMs: Qwen 2.5 (3B), Llama 3.2 (3B), and Llama 3.1 (8B). We
use three baselines including non-RAG ( 𝜖=0), non-private RAG with top-2 retrieved documents ( 𝜖=∞ ), and non-private KSA
(𝜖=∞).
for utilizing these extracted keywords to further narrow the perfor-
mance gap remains as an important direction for future work.
Interestingly, we observe that stronger generator models benefit
more from DP-KSA . As we move from Qwen 2.5 (3B) to Llama
3.1 (8B), not only do we observe higher absolute scores, but also
greater robustness to DP noise. This supports the intuition that
larger models or certain model families are better equipped to infer
context and generate meaningful outputs from partially informativeor compressed prompts, making them especially well-suited for
privacy-preserving QA systems.
Finally, although all four evaluation metrics follow consistent
upward trends as 𝜖increases, the degree of improvement varies.
Levenshtein similarity, which captures fine-grained character-level
differences, improves more noticeably, suggesting that the gener-
ated responses become semantically closer to the references as
8

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
privacy budget increases. ROUGE-L and F1 also show steady im-
provements, confirming the DP-KSA yields gains across both string
similarity and task-specific QA measures.
6.2.2 Performance on TQA dataset.Figure 6 shows the perfor-
mance of DP-KSA on the TQA dataset. Similar to the trends ob-
served on NQ, we see consistent performance gains as the privacy
parameter𝜖increases from 1 to 8, illustrating DP-KSA ’s ability to
effectively trade privacy for utility. Beginning at a privacy budget
of𝜖=2,DP-KSA consistently outperforms the strictly private non-
RAG (𝜖=0) baseline, demonstrating that even under strong privacy
constraints, meaningful keywords can still be preserved. One ex-
ample is when using Llama 3.2 (3B) as the generator, the F1 score
improves from 55.12 at 𝜖=0to 56.92 at 𝜖=5, indicating a strong
privacy-utility tradeoff. Notably, unlike on NQ where performance
at lower𝜖values occasionally regresses, DP-KSA on TQA exhibits
smoother and more stable improvements even under tight privacy
budgets.
A particular interesting finding is that DP-KSA performs com-
petitively against the non-private RAG ( 𝜖=∞ ) baseline in some
cases. For instance, with Qwen 2.5 (3B), the ROUGE-1, ROUGE-L,
and Levenshtein similarity scores at 𝜖=5exceed those of the non-
private RAG baseline, despite the added differential privacy noise.
This suggests that using keywords from a noisy histogram formed
across an ensemble of responses can yield final responses that are
semantically rich as those generated with full document contexts,
particularly when the retrieved context is noisy or contains more
redundancy.
Digging deeper into the baselines, we observe that the non-
private KSA baseline actually outperforms non-private RAG in
smaller models such as Qwen 2.5 (3B) and Llama 3.2 (3B), while the
trend reverses for larger models like Llama 3.1 (8B). This difference
can be attributed to the relative contributions of two information
sources, namely the consensus of keywords captured by keyword
frequency across the ensemble of 80 responses, and the parametric
knowledge embedded in the generator LLM. For smaller models,
the ensemble-driven signal dominates, giving KSA an advantage.
However, as the generator becomes more powerful, its internal
knowledge plays a larger role in improving generation quality and
thus shifting the advantage toward full-context RAG.
In line with these observations, DP-KSA shows greater benefits
with stronger generator models. Moving from Qwen 2.5 (3B) to
Llama 3.1 (8B), we see not only higher absolute performance but
also improved stability across different privacy budgets. This rein-
forces our earlier conclusion from the NQ results that larger and
more capable models are better suited to generalize from sparse or
compressed prompts, making them strong candidates for privacy-
preserving RAG.
Lastly, all four metrics display consistent improvement as 𝜖in-
creases, but the extent of gains varies. Levenshtein similarity and
ROUGE-L tend to improve more steeply, indicating that as more
keywords are released, the generated final responses become se-
mantically and structurally closer to the reference answers. F1 and
ROUGE-1 also improve steadily, confirming that DP-KSA remains
effective across both QA-style metrics and surface-level lexical
overlap. Compared to NQ, DP-KSA on TQA dataset achieves sig-
nificantly higher absolute scores, likely due to its more extractiveanswer structure and the higher redundancy in supporting con-
text, which together make it easier for DP-KSA to preserve relevant
information even under privacy constraints.
(a) NQ
(b) TQA
Figure 7: Propose-test-release (PTR) test pass rate of DP-KSA
with varying privacy parameters 𝜖on NQ and TQA datasets.
We report the results with three different LLMs: Qwen 2.5
(3B), Llama 3.2 (3B), and Llama 3.1 (8B).
6.2.3 PTR test pass rate.We analyze the effect of various privacy
constraints of DP-KSA on PTR test pass rates on both NQ and TQA
datasets. Following the description of our method DP-KSA in sec-
tion 4.2, the propose-test-release (PTR) test checks if the condition
H(𝑘)−H(𝑘+1) >2holds. Figure 7 shows the pass rate changes for
different privacy parameters 𝜖, ranging from 1 to 8, across three
different generator LLMs. Across both datasets, we observe a clear
and consistent trend that the PTR test pass rate increases steadily
as𝜖increases. This indicates that relaxing the privacy constraint
reduces the amount of DP noise added, enabling more privately
selected keywords to pass the PTR test and being released to aid
the final model outputs. In other words, lower noise leads to more
stable test outcomes and thus higher pass rates. This observation
also supports our discussion earlier in the privacy-utility tradeoff
performance on both datasets.
9

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
When comparing across models, we find that Qwen 2.5 (3B) trails
slightly behind both Llama models on NQ and TQA. The Llama
3.2 (3B) model generally achieves the highest pass rates on the NQ
dataset, especially showing steep gains between 𝜖=1and𝜖=5.
In contrast, on the TQA dataset, the Llama 3.1 (8B) model slightly
outperforms the others at higher privacy budgets ( 𝜖=5and8),
suggesting that its larger capacity helps identify consistent keyword
patterns even under significant DP noise. These trends highlight
that stronger models, particularly those in the Llama family, exhibit
better resilience to DP noise and are more effective at preserving
semantic signals necessary for keyword extraction.
In terms of dataset differences, we observe that the TQA dataset
consistently achieves higher PTR pass rates than NQ across all
models and 𝜖values. Even at 𝑒𝑝𝑠𝑖𝑙𝑜𝑛= 1, models on TQA start
from a noticeably higher baseline. This discrepancy likely stems
from differences in dataset structure. TQA answers tend to be more
extractive and contextually redundant, which helps generate more
consistent keyword distributions across ensembles. As a result,
keywords in TQA are easier to identify under the added DP noise.
6.3 Ablation Studies
We also conduct ablation studies with varying model sizes and
number of ensembles on NQ dataset to investigate how these hyper-
parameters affect the performance of our proposed method DP-KSA .
6.3.1 Effect of Model Size.Figure 8 illustrates the impact of model
size on the performance of DP-KSA across all four metrics, F1,
ROUGE-1, ROUGE-L and Levenshtein similarity, under a fixed pri-
vacy budget of 𝜖=3, along with non-private baselines for compar-
ison. In this experiment, we evaluate the Qwen 2.5 model family
at four scales, 1.5B , 3B, 7B and 32B. To accommodate hardware
memory limitations, we use 4-bit quantization to the 32B model
weights, ensuring that all models can be evaluated under the same
computational constraints.
As expected, larger models consistently achieve better perfor-
mance across most metrics. As model size increases, we observe
steady gains in F1 and ROUGE scores under the privacy-preserving
setting, indicating that more expressive LLMs are more capable
of compensating for the restricted input caused by providing only
keywords as context in the prompts. Notably, even under a mod-
erate privacy constraint of 𝜖=3, the 32B model nearly matches
the performance of the fully non-private RAG baseline ( 𝜖=∞ ),
demonstrating that DP-KSA benefits substantially from scaling to
larger model capacities.
Interestingly, however, we observe a decline in Levenshtein sim-
ilarity as model size increases to 32B. This seemingly counterintu-
itive trend can be explained by the generative behavior of larger
models. Bigger LLMs, such as Qwen 2.5 (32B), are more likely to
paraphrase reference answers with greater fluency and lexical diver-
sity. While these outputs may be semantically accurate, they diverge
more from the reference at the character level, which directly low-
ers Levenshtein similarity. Larger models may also produce longer
or more elaborative responses, introducing more edits even if se-
mantically valid, while smaller models are more conservative and
literal in their generations. Therefore, the drop in Levenshtein sim-
ilarity should be interpreted not as a sign of degraded quality, but
rather as a side effect of more fluent and paraphrastic generation.
Figure 8: Ablation studies on model sizes with Qwen 2.5
model family on NQ dataset. Model size axis is in log scale.
At the other end of the scale, the smallest model size (1.5B) re-
veals an unusual pattern. The non-RAG baseline ( 𝜖=0), where only
the question is given without any retrieved context, outperforms
both the non-private KSA and DP-KSA (𝜖=3) settings. This can
be explained by the weaker instruction-following capabilities of
smaller models. When provided with prompts using keyword-only
context, whether deterministic (KSA) or noisy ( DP-KSA ), these mod-
els struggle to interpret and integrate sparse cues into coherent
answers. In contrast, the non-RAG setting, which presents only the
raw question, may align better with the pretraining distribution of
small models and avoid potential confusion from prompts with in-
complete or overly narrow keywords as context. Consequently, the
model relies more heavily on its parametric knowledge, sometimes
yielding better outputs than poorly constructed context.
Moreover, between KSA and DP-KSA (𝜖=3) at 1.5B, the latter
surprisingly performs better. One possible reason is that the sto-
chastic nature of DP-KSA introduces diversity into the prompts,
occasionally surfacing less frequent but semantically useful tokens.
This noise-driven variation can act as a form of regularization that
prevents the model from overfitting to dominant but uninformative
keywords and this is a vulnerability particularly pronounced in
smaller models.
Overall, these findings underscores that larger models are not
only more effective at leveraging prompts with sparse context but
also more robust to the limitations imposed by differential privacy.
They suggest that DP-KSA is best paired with instruction-following-
capable LLMs to achieve strong privacy-utility tradeoffs.
10

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Figure 9: Ablation studies on number of ensembles with
Llama 3.2 (3B) model and𝜖=3on NQ dataset.
6.3.2 Effect of Number of Ensembles.Figure 9 shows how the num-
ber of retrieval ensembles impacts the performance of DP-KSA . In
this experiment, we use the Llama 3.2 (3B) generator model and
fix the privacy budget at 𝜖=3, while varying the number of en-
sembles from 10 to 100. The number of ensembles reflects how
many top-ranked documents are used from retrieval. For instance,
an ensemble size of 10 uses the top 10 retrieved documents, while
a size of 50 incorporates the top 50. Each ensemble corresponds
to a different retrieved document used to prompt the generator,
and their aggregated responses are used to construct a noisy token
histogram for differentially private keyword selection.
We observe a clear progression in performance trends across all
four evaluation metrics as the number of ensembles increases.
When using between 10 and 40 ensembles, performance remains
relatively stable. This indicates that aggregating a limited number of
ensembles does not provide sufficient signal strength to overcome
the DP noise introduced. With few ensembles, informative tokens
may appear sporadically across responses, leading to sparse or
inconsistent histograms where the DP mechanism struggles to
identify high-frequency keywords reliably.
As the number of ensembles increases from 40 to 80, performance
improves steadily. The addition of more ensembles enriches the
token histogram, making it more representative and resilient to DP
noise. Recurrent tokens consistently appearing across semantically
relevant responses emerge more clearly, enabling the stochastic
keyword release mechanism to retain more meaningful and infor-
mative content. The growing diversity of responses within this
range plays a key role in stabilizing the keyword selection process
under DP constraints.
Between 80 and 100 ensembles, performance begins to plateau.
While more ensembles continue to be aggregated, the marginal
gains diminish. Additional documents may contribute less relevant
or redundant information, increasing the presence of low-utility
or noisy tokens in the histogram. This can dilute the quality of the
extracted keywords and limit further improvements in the final
output quality.
These findings indicate that while increasing the number of en-
sembles can significantly enhance performance under DP, there
are diminishing returns beyond a certain point. Once the ensemblesize becomes large enough to establish a stable token distribution,
adding more documents does little to improve and may even slightly
degrade the quality of the extracted keywords due to the introduc-
tion of irrelevant or noisy content. This highlights the need for more
strategic methods for selecting the number of ensembles to further
boost performance without compromising the privacy guarantee.
7 Related Work
7.1 Privacy Attacks on Large Language Models
Prior work has shown that large language models (LLMs) are vulner-
able to privacy breaches through a variety of attack vectors. Carlini
et al. [4] demonstrated that adversaries can extract memorized
training examples, such as email addresses, phone numbers, and
other sensitive content, by carefully crafting input prompts. These
training data extraction attacks reveal that even models trained on
supposedly de-identified datasets can still memorize and regurgitate
specific private data points. Prompt extraction attacks [ 9,24,38]
have also received significant attention. These attacks target de-
ployed systems where users provide prompts that encode propri-
etary logic, credentials, or task-specific templates. Adversaries can
use model inversion, gradient leakage, or black-box probing to in-
fer the original prompts, threatening the confidentiality of model
customization and user inputs in commercial APIs.
In the context of retrieval-augmented generation (RAG), recent
work has highlighted privacy risks stemming from both the retrieval
and generation components. Huang et al . [17] analyzed𝑘NN-LMs
and showed that retrieved neighbors from the training corpus can
expose private information, especially when the distance function
reveals document structure or content characteristics. Zeng et al .
[36] conducted a systematic investigation into open-source RAG
pipelines and found that sensitive data in the retrieval corpus can
propagate into generated responses without explicit prompts, par-
ticularly when the retriever ranks memorized or high-sensitivity
content highly. Qi et al . [29] extended these findings to commer-
cial production-level RAG systems, showing that the configuration
of RAG, including retrieval granularity, document ranking, and
prompt construction, can significantly influence the model’s sus-
ceptibility to data leakage. Together, these studies underline the
importance of privacy-aware design to mitigate the growing threat
of information leakage in retrieval-augmented language generation.
7.2 Privacy-Preserving Large Language Models
Differential privacy has been studied in many tasks in large lan-
guage models. The differentially private pretraining and finetuning
of LLMs have been studied to address the privacy concern in the
training data by deploying DP-SGD [ 1]. In this paradigm, noise
is introduced to the gradient during the model’s training to en-
sure privacy. However, as the scale of the large language models
significantly increased, memory becomes a large bottleneck and
makes this approach more challenging in practice. Although recent
methods have been proposed for efficient per-example gradient
clipping [ 23] and parameter-efficient fine-tuning [ 35], it remains a
topic of ongoing research in order to address the engineering and
optimization problems introduced by DP-SGD. In-context learning
adapts to different tasks by illustrating some examples in the con-
text as the task description. DP in-context learning considers the
11

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
situation when the examples are picked from any private dataset.
[31] tackles this problem by generating synthetic examples with
DP. [ 33] instead uses a sample-and-aggregate algorithm to generate
DP responses.
To mitigate privacy risks in RAG systems, [ 37] proposed an
empirical privacy-preserving algorithm for RAG through the syn-
thetic data generation, while our work studies privacy-preserving
RAG in the framework of differential privacy, which protects the
privacy of each individual document with the theoretical guaran-
tee. Closely-related prior works have proposed DP solutions in
the RAG setting by applying DP at every token generation of the
LLM [ 14,20] after retrieving relevant documents. While they offer
privacy protection for RAG systems, they come with noticeable
limitations. Per-iteration private voting requires composing the
privacy loss over the number of tokens being generated, which
quickly destroys the privacy-utility tradeoff.
8 Conclusion
In this paper, we presented DP-KSA , a novel privacy-preserving al-
gorithm that ensures differential privacy for sensitive external data
source in the RAG system, enabling us to enhance LLMs by domain-
specific but sensitive external data source.DP-KSAprivatizes RAG
by extracting the most frequent keywords based on the "propose-
test-release" paradigm and augmenting them into the prompt to
generate the final output. The privately extracted keywords ef-
fectively compresses the semantic space while still retaining key
information pertaining to the retrieved contexts for better utility.
The experiments on QA benchmarking datasets show that our algo-
rithm outperforms the non-RAG method under moderate privacy
budgets across different models, demonstrating its effectiveness
for maintaining high generation quality while providing formal
privacy guarantees. We also explored and evaluated the impact of
generator model sizes and number of ensembles on our method.
In future work, we will consider a more adaptive scheme for
selecting the number of ensembles to further improve privacy-
utility tradeoffs across different generator model families and sizes.
While DP-KSA targets scenarios where the external retrieval corpus
is private and the generator LLM is trained on publicly available
data, commonly seen in real-world RAG systems using open-source
models, it is also worth investigating how DP-KSA affects the train-
ing data leakage risks when the generator LLM is pre-trained or
fine-tuned on private datasets.
Acknowledgment
The authors used Grammarly and ChatGPT4.o to detect and correct
grammatical errors in this paper.
References
[1]Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov,
Kunal Talwar, and Li Zhang. 2016. Deep Learning with Differential Privacy. In
Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications
Security(Vienna, Austria)(CCS ’16). Association for Computing Machinery, New
York, NY, USA, 308–318. https://doi.org/10.1145/2976749.2978318
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
Reflection. InThe Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=hSyW5go0v8
[3]Mark Bun and Thomas Steinke. 2016. Concentrated differential privacy: Simpli-
fications, extensions, and lower bounds. InTheory of cryptography conference.Springer, 635–658.
[4]Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-
Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson,
et al.2021. Extracting training data from large language models. In30th USENIX
security symposium (USENIX Security 21). 2633–2650.
[5]Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading
Wikipedia to Answer Open-Domain Questions. InProceedings of the 55th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
Regina Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguis-
tics, Vancouver, Canada, 1870–1879. https://doi.org/10.18653/v1/P17-1171
[6]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume 1 (Long and
Short Papers), Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association
for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. https://doi.
org/10.18653/v1/N19-1423
[7]Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
The Faiss library. arXiv:2401.08281 [cs.LG] https://arxiv.org/abs/2401.08281
[8]Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. 2023.
Flocks of stochastic parrots: Differentially private prompt learning for large
language models.Advances in Neural Information Processing Systems36 (2023),
76852–76871.
[9]Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, and
Franziska Boenisch. 2023. On the privacy risk of in-context learning. InThe 61st
Annual Meeting Of The Association For Computational Linguistics.
[10] David Durfee and Ryan M Rogers. 2019. Practical differentially private top-k
selection with pay-what-you-get composition.Advances in Neural Information
Processing Systems32 (2019).
[11] Cynthia Dwork. 2006. Differential privacy. InInternational colloquium on au-
tomata, languages, and programming. Springer, 1–12.
[12] Cynthia Dwork and Jing Lei. 2009. Differential privacy and robust statistics. In
Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing
(Bethesda, MD, USA)(STOC ’09). Association for Computing Machinery, New
York, NY, USA, 371–380. https://doi.org/10.1145/1536414.1536466
[13] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783
(2024).
[14] Nicolas Grislain. 2025. RAG with Differential Privacy. arXiv:2412.19291 [cs.LG]
https://arxiv.org/abs/2412.19291
[15] Jennifer Hsia, Afreen Shaikh, Zhiruo Wang, and Graham Neubig. 2024. Ragged:
Towards informed design of retrieval augmented generation systems.arXiv
preprint arXiv:2403.09040(2024).
[16] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A
survey on hallucination in large language models: Principles, taxonomy, chal-
lenges, and open questions.ACM Transactions on Information Systems43, 2 (2025),
1–55.
[17] Yangsibo Huang, Samyak Gupta, Zexuan Zhong, Kai Li, and Danqi Chen. 2023.
Privacy Implications of Retrieval-Based Language Models. InEMNLP.
[18] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A
Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.
InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Regina Barzilay and Min-Yen Kan (Eds.).
Association for Computational Linguistics, Vancouver, Canada, 1601–1611. https:
//doi.org/10.18653/v1/P17-1147
[19] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn,
Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online,
6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
[20] Tatsuki Koga, Ruihan Wu, and Kamalika Chaudhuri. 2025. Privacy-
Preserving Retrieval-Augmented Generation with Differential Privacy.
arXiv:2412.04697 [cs.CR] https://arxiv.org/abs/2412.04697
[21] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. https://doi.org/10.1162/tacl_a_00276
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
12

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[23] Xuechen Li, Florian Tramèr, Percy Liang, and Tatsunori Hashimoto. 2022. Large
Language Models Can Be Strong Differentially Private Learners. arXiv preprint
arXiv:2110.05679. InInternational Conference on Learning Representations (ICLR).
https://arxiv.org/abs/2110.05679
[24] Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang,
Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, et al .2023. Prompt Injec-
tion attack against LLM-integrated Applications.arXiv preprint arXiv:2306.05499
(2023).
[25] Frank McSherry and Kunal Talwar. 2007. Mechanism design via differential
privacy. In48th Annual IEEE Symposium on Foundations of Computer Science
(FOCS’07). IEEE, 94–103.
[26] Ilya Mironov. 2017. Rényi differential privacy. In2017 IEEE 30th computer security
foundations symposium (CSF). IEEE, 263–275.
[27] Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. 2007. Smooth sensitivity
and sampling in private data analysis. InProceedings of the Thirty-Ninth Annual
ACM Symposium on Theory of Computing(San Diego, California, USA)(STOC
’07). Association for Computing Machinery, New York, NY, USA, 75–84. https:
//doi.org/10.1145/1250790.1250803
[28] Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu,
Alexander H Miller, and Sebastian Riedel. 2019. Language models as knowledge
bases?arXiv preprint arXiv:1909.01066(2019).
[29] Zhenting Qi, Hanlin Zhang, Eric P. Xing, Sham M. Kakade, and Himabindu
Lakkaraju. 2025. Follow My Instruction and Spill the Beans: Scalable Data
Extraction from Retrieval-Augmented Generation Systems. InThe Thirteenth
International Conference on Learning Representations. https://openreview.net/
forum?id=Y4aWwRh25b
[30] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever,
et al.2019. Language models are unsupervised multitask learners.OpenAI blog
1, 8 (2019), 9.
[31] Xinyu Tang, Richard Shin, Huseyin A Inan, Andre Manoel, Fatemehsadat
Mireshghallah, Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, and Robert Sim.
2024. Privacy-Preserving In-Context Learning with Differentially Private Few-
Shot Generation. InThe Twelfth International Conference on Learning Representa-
tions. https://openreview.net/forum?id=oZtt0pRnOl
[32] Yuhao Wang, Ruiyang Ren, Junyi Li, Wayne Xin Zhao, Jing Liu, and Ji-Rong
Wen. 2024. REAR: A Relevance-Aware Retrieval-Augmented Framework for
Open-Domain Question Answering.arXiv preprint arXiv:2402.17497(2024).
[33] Tong Wu, Ashwinee Panda, Jiachen T Wang, and Prateek Mittal. 2023. Privacy-
preserving in-context learning for large language models.arXiv preprint
arXiv:2305.01639(2023).
[34] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[35] Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam
Kamath, Janardhan Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, et al .
2022. Differentially private fine-tuning of language models. InInternational
Conference on Learning Representations (ICLR).
[36] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren,
Yi Chang, Shuaiqiang Wang, Dawei Yin, and Jiliang Tang. 2024. The Good and
The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG).
InFindings of the Association for Computational Linguistics: ACL 2024, Lun-Wei
Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational
Linguistics, Bangkok, Thailand, 4505–4524. https://doi.org/10.18653/v1/2024.
findings-acl.267
[37] Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing
Lu, Han Xu, Hui Liu, Yue Xing, and Jiliang Tang. 2025. Mitigating the Pri-
vacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data.
arXiv:2406.14773 [cs.CR] https://arxiv.org/abs/2406.14773
[38] Yiming Zhang, Nicholas Carlini, and Daphne Ippolito. 2023. Effective prompt
extraction from language models.arXiv preprint arXiv:2307.06865(2023).
[39] Yuqing Zhu and Yu-Xiang Wang. 2022. Adaptive private-k-selection with adaptive
k and application to multi-label pate. InInternational Conference on Artificial
Intelligence and Statistics. PMLR, 5622–5635.
13

Proceedings on Privacy Enhancing Technologies YYYY(X) Tang et al.
A Privacy Analysis ofDP-KSA
We now provide technical details behind the privacy analysis of
DP-KSA . We first introduce some definitions and theorems to help
with the analysis.
A.1 Relevant Properties
In the main text, we introduced (𝜖,𝛿) -DP (Definition 2.1). Addition-
ally, we will introduce Rényi Differential Privacy (RDP), a variant
of(𝜖,𝛿) -DP that uses Rényi divergence to measure the difference
between𝑀(𝐷)and𝑀(𝐷′).
Definition A.1(Rényi Divergence).For two probability distribu-
tions𝑃and𝑄defined overR, the Rényi divergence of order 𝛼>1
is
𝐷𝛼(𝑃||𝑄)=1
𝛼−1logE𝑥∼𝑄𝑃(𝑥)
𝑄(𝑥)𝛼
.
Definition A.2.( (𝛼,𝜖(𝛼)) -RDP [ 26]) A randomized algorithm
𝑀:D→R is(𝜖(𝛼),𝛼) -RDP if for any adjacent datasets 𝐷,𝐷′∈D
it holds that𝐷 𝛼(𝑀(𝐷)||𝑀(𝐷′))≤𝜖(𝛼).
An advantage of RDP is its convenient composition properties,
which states that the privacy loss of the composition of multiple
RDP algorithms is simply the sum of each RDP algorithm. We state
this more formally below.
Theorem A.3(Composition [ 26]).Let𝐴1,...,𝐴𝑘be a sequence of
(𝛼,𝜖(𝛼)) -RDP algorithms. Then the composition 𝐴𝑘◦𝐴𝑘−1◦...◦𝐴 1
is(𝛼,𝑘𝜖(𝛼))-RDP.
Another important property of RDP, and more generally DP,
is that any further operations on the output of an RDP algorithm
does not leak additional information, called the post-processing
property.
Theorem A.4(Post-Processing [ 26]).Let𝐴:D→R be(𝛼,𝜖(𝛼)) -
RDP, and let 𝐹:R→Z be an arbitrary randomized mapping. Then
𝐹◦𝑀is(𝛼,𝜖(𝛼))-RDP.
Another useful relaxation of the RDP definition is approximate
RDP.
Definition A.5(Approximate RDP [ 3,39]).We say a randomized
algorithm𝑀is𝛿-approximately(𝛼,𝜖𝑀(𝛼))-RDP with order 𝛼≥1,
if for all neighboring dataset 𝐷,𝐷′, there exist events 𝐸(depending
on𝑀(𝐷) ) and𝐸′(depending on 𝑀(𝐷′)) such that Pr[𝐸]≥ 1−𝛿
andPr[𝐸′]≥1−𝛿, and∀𝛼≥1, we have
𝐷𝛼(𝑀(𝐷)|𝐸∥𝑀(𝐷′)|𝐸′)≤𝜖𝑀(𝛼)(3)
Note that when 𝛿=0, then0-approximate (𝛼,𝜖(𝛼)) -RDP is sim-
ply(𝛼,𝜖(𝑎𝑙𝑝ℎ𝑎)) -RDP. Finally, we can convert between (𝛼,𝜖(𝛼))
and(𝜖,𝛿)-DP, which is shown below.
Theorem A.6(Conversion from approximate RDP to Approximate
DP [39]).If an algorithm 𝐴satisfies𝛿1-approximate(𝛼,𝜖(𝛼)) -RDP,
then it is(𝜖(𝛼)+log(1/𝛿)
𝛼−1,𝛿+𝛿 1)-DP for any0<𝛿<1.
We use approximate RDP for a tighter measure of the privacy
cost under composition. After we obtain the (approximate) RDP
guarantee for the overall algorithm, we can then convert the privacy
guarantee back into the standard DP definition (Theorem A.6).Lastly, we introduce a fundamental DP mechanism, called the
exponential mechanism [ 25], which the FindBestK is based on.
Given some utility function over outputs, the exponential mech-
anism samples high-utility outputs with higher probability than
low-utility outputs.
Definition A.7(Exponential Mechanism [ 25]).Given a utility func-
tion𝑞:X∗×O→R withℓ1sensitivity Δ(𝑞)=max 𝐷∼𝐷′,𝑜∈𝑂|𝑞(𝐷,𝑜)−
𝑞(𝐷′,𝑜)|, the exponential mechanism𝑀has output distribution
Pr[𝑀(𝐷)=𝑜]∝exp𝜖𝑞(𝐷,𝑜)
2Δ(𝑞)
.
where∝elides the normalization factor.
Given the above definition, one can show that the exponential
mechanism satisfies(𝜖,0)-DP.
Lemma A.8([25]).The exponential mechanism is(𝜖,0)-DP.
Since we want to compose the privacy loss of the exponential
mechanism and report the loss in terms of (𝜖,𝛿) -DP, we use the
following property to convert from𝜖-DP to(𝛼,𝜖(𝛼))-RDP.
Theorem A.9([ 3]).The𝑀is𝜖-DP then it is(𝛼,𝜖 EM(𝛼))-RDP where
𝜖EM(𝛼):=min𝛼
2𝜖2,1
𝛼−1logsinh(𝛼𝜖)−sinh((𝛼−1)𝜖)
sinh(𝜖)
.
A.2 Privacy Analysis
Now we will prove that DP-KSA satisfies(𝜖,𝛿) -DP. First, we obtain
a privacy guarantee forTopKWithPTRandFindBestK.
Theorem A.10. TopKWithPTR (Algorithm 2) is 𝛿-approximate𝛼
2𝜎2-
RDP.
Proof. The privacy analysis mostly follows from Zhu and Wang
[39]. Releasing the noisy threshold b𝑑𝑘is𝛼
2𝜎2-RDP.
If𝑑𝑘>2, then releasing the exact top- 𝑘tokens has no privacy
cost, as its local sensitivity is 0.
If𝑑𝑘≤2, then if b𝑑𝑘≤2, the program terminates and there’s no
privacy cost.
If𝑑𝑘≤2, the failure probability
Pr[b𝑑𝑘>2]=Pr[max(2,𝑑 𝑘)+N(0,4𝜎2)−Φ(1−𝛿; 0,2𝜎)>2]
=Pr[2+N(0,4𝜎2)−Φ(1−𝛿; 0,2𝜎)>2]
=Pr[N(0,4𝜎2)−Φ(1−𝛿; 0,2𝜎)>0]
=𝛿
□
Theorem A.11. FindBestK (Algorithm 3) satisfies (𝛼,𝜖 EM(𝛼))-
RDP.
Proof. Note that adding Gumbel noise to each output’s utility
and releasing the output with the highest noisy utility score is equiv-
alent to using the exponential mechanism [ 10]. Hence, FindBestK
is an EM where the utility function is 𝑑𝑘=H(𝑘)−H(𝑘+1) with
the sensitivity Δ(𝑑𝑘)=2(this arguement derives from Zhu and
Wang [39]). Therefore, the privacy guarantee follows from Theorem
A.9.□
Theorem A.12.DP-KSA(Algorithm 1) satisfies(𝜖,𝛿)-DP.
14

Differentially Private RAG Proceedings on Privacy Enhancing Technologies YYYY(X)
Proof. Let𝐷,𝐷′be two adjacent datasets the differ by one doc-
ument. Hence, after retrieving the 𝑁most relevant documents for a
queryx,𝑅(x,𝐷;𝑁)and𝑅(x,𝐷′;𝑁)differ by at most one document.
Suppose it is the 𝑖-th document. More precisely, 𝐷x
𝑗=𝐷′x
𝑗∀𝑗≠𝑖
and𝐷x
𝑖≠𝐷′x
𝑖. Then we obtain the corresponding histogramsH
andH′, which differ by one responsey 𝑖≠y′
𝑖. Hence by Theo-
rem A.11 ˆ𝑘is(𝛼,𝜖 EM(𝛼))-RDP. Then using ˆ𝑘we have𝑤1,...,𝑤 ˆ𝑘,
which are𝛿1-approximate𝛼
2𝜎2-RDP by Theorem A.10. Then by
composition (Theorem A.3) the total privacy loss is 𝛿1-approximate
(𝛼,𝜖 EM(𝛼)+𝛼
2𝜎2)-RDP. Moreover, because 𝑤1,...,𝑤 ˆ𝑘are RDP, any
further operations on them do not leak additional privacy by post-
processing (Theorem A.4). Hence, using 𝑤1,...,𝑤 ˆ𝑘for obtaining the
final outputy←𝐹( x,{𝑤𝑖}ˆ𝑘
𝑖=1)is DP. Finally we convert the final
privacy loss back to(𝜖,𝛿)-DP using Theorem A.6.□
15
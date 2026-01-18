# ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models

**Authors**: Huipeng Ma, Luan Zhang, Dandan Song, Linmei Hu, Yuhang Tian, Jun Yang, Changzhi Zhou, Chenhao Li, Yizhou Jin, Xudong Li, Meng Lin, Mingxing Zhang, Shuhao Zhang

**Published**: 2026-01-12 06:57:31

**PDF URL**: [https://arxiv.org/pdf/2601.07260v1](https://arxiv.org/pdf/2601.07260v1)

## Abstract
In multi-hop reasoning, multi-round retrieval-augmented generation (RAG) methods typically rely on LLM-generated content as the retrieval query. However, these approaches are inherently vulnerable to knowledge overshadowing - a phenomenon where critical information is overshadowed during generation. As a result, the LLM-generated content may be incomplete or inaccurate, leading to irrelevant retrieval and causing error accumulation during the iteration process. To address this challenge, we propose ActiShade, which detects and activates overshadowed knowledge to guide large language models (LLMs) in multi-hop reasoning. Specifically, ActiShade iteratively detects the overshadowed keyphrase in the given query, retrieves documents relevant to both the query and the overshadowed keyphrase, and generates a new query based on the retrieved documents to guide the next-round iteration. By supplementing the overshadowed knowledge during the formulation of next-round queries while minimizing the introduction of irrelevant noise, ActiShade reduces the error accumulation caused by knowledge overshadowing. Extensive experiments show that ActiShade outperforms existing methods across multiple datasets and LLMs.

## Full Text


<!-- PDF content starts -->

ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning
in Large Language Models
Huipeng Ma1,2*, Luan Zhang1*, Dandan Song1†, Linmei Hu1†, Yuhang Tian1, Jun Yang1, Changzhi
Zhou1, Chenhao Li1, Yizhou Jin1, Xudong Li1, Meng Lin1, Mingxing Zhang3, Shuhao Zhang4
1Beijing Institute of Technology, China
2QiYuan Lab
3Tsinghua University, China
4Huazhong University of Science and Technology, China
{mahuipeng, luan zhang, sdd, hulinmei}@bit.edu.cn
Abstract
In multi-hop reasoning, multi-round retrieval-augmented
generation (RAG) methods typically rely on LLM-generated
content as the retrieval query. However, these approaches are
inherently vulnerable toknowledge overshadowing—a phe-
nomenon where critical information is overshadowed during
generation. As a result, the LLM-generated content may be
incomplete or inaccurate, leading to irrelevant retrieval and
causing error accumulation during the iteration process. To
address this challenge, we proposeActiShade, which detects
and activates overshadowed knowledge to guide large lan-
guage models (LLMs) in multi-hop reasoning. Specifically,
ActiShade iteratively detects the overshadowed keyphrase in
the given query, retrieves documents relevant to both the
query and the overshadowed keyphrase, and generates a new
query based on the retrieved documents to guide the next-
round iteration. By supplementing the overshadowed knowl-
edge during the formulation of next-round queries while min-
imizing the introduction of irrelevant noise, ActiShade re-
duces the error accumulation caused byknowledge overshad-
owing. Extensive experiments show that ActiShade outper-
forms existing methods across multiple datasets and LLMs.
Introduction
Large language models (LLMs) have demonstrated remark-
able performance across various of NLP tasks, such as
multi-hop reasoning (OpenAI 2023; Meta 2024a). How-
ever, LLMs have a risk of generating factually incorrect re-
sponses, also known as hallucinations (Bang et al. 2023; Ji
et al. 2023; Huang et al. 2023). Retrieval-augmented genera-
tion (RAG) techniques have been widely adopted to enhance
the factual correctness of LLM-generated responses by in-
corporating knowledge from external resources (Gao et al.
2023; Fan et al. 2024).
Early RAG methods often adopt one-round retrieval,i.e.,
use the original question as the retrieval query (Guu et al.
2020; Borgeaud et al. 2022; Izacard et al. 2023; Zhang et al.
*These authors contributed equally.
†Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: Illustration of error accumulation caused byknowl-
edge overshadowing.The keyphrase Gloria in the query is
overshadowed, leading the LLM to generate inaccurate con-
tent, such as Te Deum. This results in the retrieval of irrele-
vant documents, which in turn causes LLM to generate more
inaccurate content in the next-round iteration.
2023). Although these methods show satisfactory perfor-
mance in answering single-hop questions (Joshi et al. 2017;
Kwiatkowski et al. 2019), they fail in answering multi-hop
questions, where more knowledge is needed beyond the one-
round retrieved knowledge.
Recent research proposes multi-round retrieval, which
typically relies on the LLM-generated content to guide
subsequent-round retrieval. A possible approach uses the re-
sponse generated by LLMs for retrieval and, in turn, uses
the newly retrieved knowledge for generation. By itera-
tively alternating between retrieval-augmented generation
and generation-augmented retrieval, the retrieval and gen-
eration are improved (Shao et al. 2023; Trivedi et al. 2023;
Jiang et al. 2023; Su et al. 2024). Another approach prompts
LLMs to decompose the complex question into a sequence
of sub-questions, using the sub-question as the retrievalarXiv:2601.07260v1  [cs.CL]  12 Jan 2026

query to obtain more precise knowledge (Press et al. 2023;
Zhou et al. 2023; Cao et al. 2023; Chu et al. 2024).
However, these methods suffer fromknowledge overshad-
owing(Zhang et al. 2025) — a phenomenon where dominant
conditions can overshadow others, causing the LLM to over-
look essential information during generation. As illustrated
in Figure 1, given the original query, the dominant condi-
tionTe Deum in D Majorovershadows the conditionGloria
in D Major, causing the LLM to generate next-round query
related toTe Deum in D Major. As a result, it retrieves ir-
relevant documents, which mislead the LLM in generating
the subsequent query, ultimately leading to error accumula-
tion over multi-round iterations. This phenomenon is partic-
ularly problematic in multi-hop reasoning, where the reason-
ing process relies on multiple interrelated conditions within
the query.
Motivated by this, we proposeActiShade, a novel frame-
work designed to detect and subsequently leverage over-
shadowed knowledge, thereby reducing error accumulation
in multi-hop reasoning. ActiShade first detects the over-
shadowed knowledge within the query, then retrieves doc-
uments relevant to it, enabling LLMs to focus on critical but
overlooked information during reasoning. Specifically, Ac-
tiShade consists of three modules. (i)Knowledge Overshad-
owing Detection: We design a new Gaussian perturbation-
based method (GaP), to detect overshadowed keyphrases
by perturbing the embeddings of candidate keyphrases with
Gaussian noise and assessing the changes in the LLM’s
output distribution. (ii)Retrieval based on Overshadowed
Keyphrase: We train a dense retriever using our constructed
contrastive learning loss. This loss enables the retriever to
effectively discriminate among positive, semi-positive, and
negative samples, which are categorized based on their rel-
evance to the query and the overshadowed keyphrase. As
a result, the retriever achieves improved retrieval of docu-
ments relevant to the overshadowed keyphrase while avoid-
ing query-irrelevant noise. (iii)Query Formulation: Given
the retrieved documents, we prompt the LLM to select the
most relevant one and generate a new query that articulates
the next reasoning step. In summary, ActiShade supplements
the overshadowed knowledge when generating the query for
the next-round retrieval, while minimizing the introduction
of query-irrelevant noise, thereby reducing the error accu-
mulation caused byknowledge overshadowing.
We evaluate ActiShade on three widely used multi-
hop reasoning datasets: HotpotQA (Yang et al. 2018),
2WikiMQA (Ho et al. 2020), and MuSiQue (Trivedi et al.
2022). Experimental results show that ActiShade signifi-
cantly outperforms the state-of-the-art baselines on three
datasets. Our contributions can be summarized as follows:
• We propose ActiShade, a novel framework designed to
detect and leverage overshadowed knowledge for multi-
hop reasoning.
• In ActiShade, we design a new Gaussian perturbation-
based method, GaP, to detect the overshadowed knowl-
edge.
• In ActiShade, we introduce a novel contrastive learning
loss for retriever training and a query formulation strat-egy to leverage the overshadowed knowledge.
• We conduct comprehensive experiments on three
datasets, demonstrating that ActiShade outperforms the
state-of-the-art methods across multiple LLMs in terms
of effectiveness.
Related Work
Knowledge Overshadowing
Zhang et al. (2025) observed when extracting knowledge
from LLMs using queries involving multiple conditions,
some conditions may overshadow others, causing them to
be ignored and thus leading to hallucinated outputs-a phe-
nomenon they refer to asknowledge overshadowing. This
phenomenon is present in multi-hop QA scenarios, which
limits the effectiveness of multi-round retrieval approaches.
Specifically,knowledge overshadowingcauses LLMs to
generate factually incorrect outputs. As multi-round retrieval
methods typically rely on LLM-generated output as the
next-round retrieval query, such hallucinations lead to ir-
relevant retrieval and cause error accumulation during the
iterative process. To overcome this limitation, we propose
a novel multi-round retrieval framework, ActiShade, which
reduces the error accumulation caused byknowledge over-
shadowing. Zhang et al. (2025) also proposed CoDA to de-
tect overshadowed knowledge by removing tokens from the
query and measuring changes in the output distribution. In
contrast, our GaP preserves the reasoning chain by adding
Gaussian noise instead of removing tokens, which enhances
the detection ofknowledge overshadowingin multi-hop rea-
soning.
Retrieval-augmented LLM
LLMs have a risk of generating hallucinated responses,
thus necessitating external retrieval for retrieval-augmented
generation. Previous methods typically adopt one-round re-
trieval,i.e., retrieve knowledge using only the original ques-
tion once (Guu et al. 2020; Borgeaud et al. 2022; Zhang
et al. 2023; Izacard et al. 2023; Shi et al. 2024). This line of
work, however, struggles to gather all the necessary knowl-
edge to answer multi-hop questions. Recently, another line
of work arose, which adopts multi-round retrieval to meet
multi-hop knowledge needs. SelfASK (Press et al. 2023)
prompts LLMs to decompose a complex question into sub-
questions and answer them through a search engine. Iter-
RetGen (Shao et al. 2023) leverages the output from the pre-
vious round concatenated with the question as the query for
next-round retrieval. IRCoT (Trivedi et al. 2023) uses each
CoT sentence as a query for retrieval until the final answer
is obtained. FLARE (Jiang et al. 2023) determine when to
retrieve based on reasoning confidence. BeamAggR (Chu
et al. 2024) decomposes complex questions, then performs
bottom-up multi-source reasoning via post-order traversal,
and uses beam aggregation to obtain the final answer. As
it relies on multi-source knowledge, which differs from our
setting, we do not include it as a baseline for fair compari-
son. EfficientRAG (Zhuang et al. 2024) iteratively generates
new retrieval queries and filters out irrelevant information
using small models. DRAGIN (Su et al. 2024) decides when

(b) 
Retrieval
(c) 
Query Formulation
Retriever
Doc 1
Doc 2
Doc 3
Query
Doc 1
Query
Doc 2
Query
Doc 3
Prob. 
"Yes" | "No"
Doc 1
Doc 2
Doc 3
0.9
-0.2
0.5
Query
Doc 1
Next Query
Next Query: 
What is the name of the famous 
bridge in the birthplace of Antonio Vivaldi?
LLM
LLM
Query
Keyphrase
Retrieval 
Database
Keyphrase Embeddings
 Cosine 
Similarity
Query: 
What is the name of the famous bridge in 
the birthplace of 
Gloria
 in D Major's composer?
Perturbed Keyphrase Embeddings
LLM
LLM
Candidate Keyphrase
Noise 
Query
Original 
Distribution
Perturbed
Distribution
LLM
Keyphrase
Extraction
(a) Knowledge Overshadowing Detection
Candidate
Keyphrase 1
bridge: 0.39
birthplace: 0.59
composer: 0.50
Glora: 0.68
Figure 2: Overview of ActiShade. ActiShade first detects the overshadowed keyphrase in the query, then retrieves relevant
documents based on it, and finally formulates a new query for the next-round retrieval.
and what to retrieve based on the LLM’s information needs
during the generation process.
Compared to them, our method is designed to reduce
the error accumulation caused byknowledge overshadow-
ingand shows superior performance. Besides, we propose
a novel method to detect overshadowed keyphrases through
noise perturbation.
ActiShade Framework
In this section, we introduce ActiShade, a novel multi-round
retrieval framework that aims to reduce error accumula-
tion caused byknowledge overshadowing. ActiShade con-
sists of three modules: (1)Knowledge Overshadowing De-
tectionfor detecting the overshadowed keyphrase; (2)Re-
trieval based on Overshadowed Keyphrasefor relevant doc-
ument retrieval; and (3)Query Formulationfor next-round
retrieval query generation. An overview of the framework is
illustrated in Figure 2.
Knowledge Overshadowing Detection
In this module, we propose a novel method, GaP, to detect
knowledge overshadowingin the query. The method consists
of three steps: keyphrase extraction, keyphrase perturbation,
and knowledge overshadowing measuring.
Step 1. Keyphrase Extraction.Given a queryQ, we ex-
tract a set of candidate keyphrasesP={p 1, p2, ..., p n},
where eachp iis a span withinQ. Specifically, weutilize SpaCy (Honnibal 2017) to extract named enti-
ties and meaningful tokens with POS tags in the set
{NOUN, ADJ, VERB, PROPN, NUM, ADV}, and re-
move stopwords to reduce noise.
Step 2. Keyphrase Perturbation.For each keyphrase
pi∈P, we inject Gaussian noise into its token embeddings
while keeping other tokens unchanged. The perturbed input
embeddings are computed as:
˜Hpi=H+m pi⊙ϵ,ϵ∼ N(0, σ2),(1)
whereHdenotes the original input embeddings of the query,
mpiis a binary mask that takes the value 1 at token positions
corresponding to the keyphrasep iand 0 elsewhere, andϵis
Gaussian noise with zero mean and standard deviationσ.
We then input ˜Hpiinto the LLM to generate the perturbed
output distribution:
˜Opi=P(y| ˜Hpi), i= 1,2, ..., n(2)
For comparison, the unperturbed output distribution is
given by:
O=P(y|H),(3)
whereydenotes the LLM’s output.
Step 3. Knowledge Overshadowing Measuring.To as-
sess the influence of each keyphrasep i∈Pon the LLM
output, we first apply average pooling to the original and
perturbed output distributions,Oand ˜Opi, along the tempo-
ral dimension, resulting in pooled representationsrand ˜rpi.

We then compute the cosine similarity betweenrand ˜rpi,
and consider the keyphrase with the highest similarity to be
overshadowed:
pko= arg max
pi∈Pcos(r, ˜rpi)(4)
A high similarity score indicates that the perturbation had
minimal influence on the LLM’s output, suggesting that the
keyphrase is underutilized, i.e., overshadowed.
Our method applies perturbations rather than removing
tokens from the query, thereby preserving the structure of
the query, which enhances the detection of knowledge over-
shadowing.
Retrieval based on Overshadowed Keyphrase
Given the detected overshadowed keyphrasep ko, this mod-
ule retrieves the documents that are relevant to both the
query and the overshadowed keyphrase. To enhance the re-
triever’s ability to focus on the overshadowed keyphrase and
avoid introducing query-irrelevant noise, we propose a novel
contrastive learning loss and train a dense retriever to dis-
criminate three types of documents: positive (relevant to
both the query and the keyphrase), semi-positive (relevant
to the query but not the keyphrase), and negative (irrelevant
to both).
Data Preparation.We construct our training dataset
based on the MuSiQue (Trivedi et al. 2022) benchmark.
Each data in the MuSiQue dataset is formulated in a dictio-
nary format with the keysquestion decomposition,
question, andparagraphs. Theparagraphsfield
contains a set of documents that are either relevant or irrel-
evant to the question. Thequestion decomposition
field provides a list of sub-questions derived from the orig-
inal question, each annotated with the supporting docu-
ment required to answer it, which can be found in the
paragraphsset.
We first identify the subject entity of the first sub-question
and define it as the keyphrase. The supporting document as-
sociated with the first sub-question is labeled as thepositive
document (D+). The supporting documents for other sub-
questions are labeled assemi-positive documents (D∗), as
they are necessary for answering the original question but
are not directly related to the keyphrase. All remaining doc-
uments are labeled asnegative document (D−), which are
irrelevant to the question. Due to space limitations, annota-
tion examples can be found in the Appendix.
Loss Function Construction.We extend the contrastive
loss proposed by (Izacard et al. 2021) to improve the capa-
bility of the retriever to prioritize documents relevant to both
the query and a specified phrase within it. The loss function
Lis defined as follow:
L1=−logS(Q,D+)
S(Q,D+)+PS(Q,D∗)+PS(Q,D−),(5)
L2=−logS(Q,D+)+PS(Q,D∗)
S(Q,D+)+PS(Q,D∗)+PS(Q,D−),(6)L=αL 1+ (1−α)L 2 (7)
For brevity, we denoteS(Q, D) =esim(Q,D), where sim
indicates the cosine similarity, and introduce hyperparam-
eterαto balance the loss terms. The lossL 1encourages
positive pairs to have higher scores over both semi-positive
and negative pairs. Although semi-positive documentsD∗
are not directly relevant to any phrase in the question, they
are required to answer the question and thus are more im-
portant than negative documentsD−. We introduce lossL 2
to further distinguish semi-positive documents from nega-
tive ones. The combined lossLensures the retriever ranks
documents in the desired order:D+> D∗> D−.
Retrieval.We concatenate the queryQwith its corre-
sponding overshadowed keyphrasep koas input to retrieve
a set of relevant documentsRD={rd 1, rd 2, ..., rd n}. The
trained retriever is capable of retrieving documents relevant
to both the query and the overshadowed keyphrase, ensur-
ing that the retrieved documents are not only query-relevant
but also supplement the overshadowed knowledge, thereby
enhancing LLMs’ reasoning.
Query Formulation
The previous module returns a set of retrieved documents
RD, which is relevant to both the query and the overshad-
owed keyphrase within it. This module then formulate a new
query based on the retrieved documents for the subsequent
retrieval round. The query formulation process consists of
three steps: relevant document selection, query generation,
and subsequent-round retrieval decision.
Step 1. Relevant Document Selection.Given a collection
of retrieved documentsRD, we first prompt the LLM to se-
lect the most relevant onerd m. Specifically, each retrieved
document and the query are jointly input into the LLM. The
LLM is required to determine whether the retrieved docu-
ment is relevant to the query. If it is relevant, output “Yes”;
otherwise, output “No”. A higher probability assigned to
“Yes” suggests a higher degree of relevance. The most rele-
vant retrieved document is then selected based on the proba-
bility of outputting “Yes”. The prompt template used for this
step is detailed in the Appendix.
Step 2. Query Generation.In the second step, we prompt
the LLM to generate a new queryQ next based on the
most relevant retrieved documentrd m. The newly gener-
ated query is used for the subsequent retrieval round, aiming
to retrieve more information beyond the scope of the initial
query. The prompt template used for this step is detailed in
Appendix. Figure 2 presents examples of query generation.
Since the retrieved document serves to supplement the
overshadowed knowledge, it enable the generation of a more
accurate query. Moreover, the newly generated query explic-
itly presents implicit reasoning results. These allow the new
query to lead to more accurate and relevant retrieval in the
next round, thereby reducing the error accumulation caused
byknowledge overshadowing.

Model MethodMuSiQue HotpotQA 2WikiMQA
ACC F1 ACC F1 ACC F1
Llama-3-8B-InstructDirect5.60 9.96 22.40 25.34 26.60 31.25
CoT11.65 16.29 29.00 34.09 27.60 34.19
Direct-R♡11.42 16.06 37.7 44.89 28.37 35.56
Iter-RetGen♣18.24 20.59 48.23 49.41 38.71 44.56
IRCoT♣15.57 18.32 40.10 47.03 34.20 41.01
SelfASK♣20.60 21.41 47.10 48.70 39.50 43.87
FLARE♣19.74 20.50 48.45 50.40 41.35 42.24
DRAGIN♣21.11 22.61 50.87 52.52 40.78 42.31
ActiShade (Ours) 25.25 26.94 54.60 56.33 45.80 46.02
Qwen2.5-7B-InstructDirect3.80 11.09 19.40 19.52 26.80 29.95
CoT6.00 13.93 22.00 27.61 29.00 32.24
Direct-R♡11.60 17.24 43.00 47.99 38.60 41.17
Iter-RetGen♣15.40 18.07 44.40 48.24 41.20 42.98
IRCoT♣14.90 18.01 43.79 48.13 40.21 40.84
SelfASK♣17.35 20.60 42.18 46.10 43.17 44.30
FLARE♣10.69 14.89 40.19 42.03 39.21 40.80
DRAGIN♣19.80 22.01 46.10 50.30 45.90 45.87
ActiShade (Ours) 22.80 26.11 48.20 55.45 52.80 50.47
Qwen2.5-14B-InstructDirect6.20 13.48 27.20 32.94 29.00 32.39
CoT10.40 17.74 29.40 34.92 33.40 35.76
Direct-R♡14.80 19.16 46.20 48.11 39.00 43.14
Iter-RetGen♣17.20 21.54 49.03 51.04 43.20 45.19
IRCoT♣15.89 19.90 47.98 50.50 44.60 45.71
SelfASK♣18.48 21.75 45.97 47.19 47.49 49.13
FLARE♣13.10 18.00 42.14 44.87 40.01 40.74
DRAGIN♣22.70 24.11 51.21 54.30 48.10 49.87
ActiShade (Ours) 25.59 27.47 53.97 57.45 51.13 53.29
Table 1: The overall experimental results of ActiShade and other baselines on three benchmarks. The best results are in bold.
♡donates single-round retrieval.♣indicates multi-round retrieval.
Step 3. Subsequent-Round Retrieval Decision.To de-
cide whether to terminate the iterative process, we prompt
the LLM to assess whether more information is needed to
answer the initial query. Specifically, the new queryQ next is
input into the LLM, which is required to determine whether
it is a single-hop query. If it is, we perform an additional
retrieval round and then terminate; otherwise, the retrieval
continues iteratively. The iterative process also terminates if
the maximum number of iterations is reached. The prompt
template used for this step is detailed in the Appendix.
After iteration, we input the initial query and the rele-
vant documents retrieved during the iterative process into
the LLM to obtain the final response.
Experimental Setup
Datasets
We evaluate ActiShade on three multi-hop reasoning
datasets. HotpotQA (Yang et al. 2018), 2WikiMQA (Ho
et al. 2020) consist of two-hop questions, and
MusiQue (Trivedi et al. 2022) contains questions with
2 to 4 hops. For HotpotQA, 2WikiMQA, and MuSiQue,
we use the same test set provided by IRCoT (Trivedi et al.
2023), which contains 500 randomly sampled instances
from the original development set.Implementation Details
We use Llama-3-8B-Instruct (Meta 2024b) and Qwen2.5-
Instruct (7B and 14B) (Yang et al. 2024) as the backbone
language models for generation and reasoning. For retriever
training, we fine-tunecontriever-msmarco(Izacard
et al. 2021) on a subset of the MuSiQue training set.
We manually select 5,000 high-quality examples, of which
3,500 are used for training, 750 for validation, and 750
for testing. The retriever is trained using the AdamW opti-
mizer (learning rate 5e-5, batch size 32) for up to 20 epochs
with early stopping based on validation loss. The contrastive
loss combines two objectives with a weighting coefficient
α= 0.7. All experiments are conducted on two NVIDIA
A6000 GPUs.
Baselines
We choose the following methods as baselines.Standard
Prompting (Brown et al. 2020)directly generates the fi-
nal answer.CoT Prompting (Wei et al. 2022)generates
reasoning steps before the final answer.One-time Re-
trievalretrieves relevant documents from external resources
and incorporates them to generate the final answer.IR-
CoT (Trivedi et al. 2023)alternates between retrieval-
augmented reasoning and reasoning-augmented retrieval un-
til enough information is obtained to answer the given ques-

Model / DatasetMuSiQue HotpotQA 2WikiMQA
Direct-R 16.06 44.89 35.56
-wCoDA 15.86 45.98 35.49
-wGaP17.83 46.81 38.67
ActiShade-NoKOD 22.83 51.23 45.18
-wCoDA 21.23 52.45 41.29
-wGaP26.94 56.33 46.02
Table 2: Performance comparison between GaP and
CoDA in single-round and multi-round retrieval settings.
ActiShade-NoKOD denotes ActiShade without the Knowl-
edge Overshadowing Detection module. All results are re-
ported in F1 score.
tion.Iter-RetGen (Shao et al. 2023)synergizes retrieval
and generation in an iterative manner: the LLM’s response
serves as a query for retrieving more relevant knowledge,
which in turn helps generate a better response in another it-
eration.Self-Ask (Press et al. 2023)iteratively decomposes
complex questions into sub-questions, retrieves and answers
them to reach the final answer.FLARE (Jiang et al. 2023)
dynamically adjusts retrieval timing based on reasoning con-
fidence and retrieves guided by the upcoming reasoning sen-
tences.DRAGIN (Su et al. 2024)detects information needs
in real time and uses self-attention over context to form re-
trieval queries during the generation process.
Evaluation Metrics
For evaluation metrics, we utilize Accuracy (Acc) and F1
score metrics for evaluation. The ACC checks if the ground-
truth answer is in the LLM-generated answer, which is also
named Cover Exact Match. The F1 score is used to mea-
sure the overlap between the LLM-generated answer and the
ground truth answer.
Experimental Results
Main Results
The experimental results on three multi-hop reasoning
datasets are presented in Table 1. We can obtain the follow-
ing observations:
Achieving Significant Performance Improvement across
all datasets and LLMs.ActiShade outperforms the pre-
vious state-of-the-art, DRAGIN, across all datasets and
LLMs, highlighting its effectiveness in multi-hop reason-
ing. This performance improvement can be attributed to Ac-
tiShade’s ability to reduce error accumulation caused by
knowledge overshadowingby iteratively detecting overshad-
owed keyphrases in the query, retrieving documents rele-
vant to both the query and the overshadowed keyphrase, and
generating a new query based on the retrieved documents.
Notably, ActiShade surpasses SelfASK (Press et al. 2023),
which decomposes a complex question into sub-questions
and answers them via retrieval. We believe this suggests that
our query formulation process makes implicit reasoning ex-
plicit, enabling more accurate and relevant retrieval com-
pared to question decomposition.
Figure 3: Sensitivity analysis of the Gaussian noise standard
deviationσ.
Maintaining Generalization Ability.We train our re-
triever based on the MuSiQue dataset, as detailed in the
ActiShade Framework section. However, ActiShade, on
HotpotQA and 2WikiMQA, still outperforms all baselines
across all LLMs, further demonstrating its effectiveness and
generalization. This indicates that the retriever effectively
learns to align retrieval not only with the query but also with
the overshadowed keyphrase, allowing it to generalize well
across various multi-hop reasoning benchmarks.
Effectiveness for larger models.To evaluate how effec-
tive ActiShade is at different model sizes, we conduct exper-
iments on Qwen2.5-Instruct (7B and 14B). As shown in Ta-
ble 1, the ActiShade’s performance generally improves with
the model size, demonstrating its scalability to larger mod-
els. Due to hardware resource constraints, we are unable to
implement ActiShade on larger models.
Analysis of Knowledge Overshadowing Detection
To systematically evaluate the effectiveness of GaP, we con-
duct a series of analyses focusing on performance compari-
son, interpretability, and parameter sensitivity.
Comparative Performance of Detection Methods.To
investigate the impact of the knowledge overshadowing de-
tection module, we compare our proposed GaP against the
CoDA (Zhang et al. 2025) method under both single-round
and multi-round retrieval settings. In the single-round set-
ting, we evaluate three variants: (1) Direct retrieval without
overshadowing detection; (2) Direct retrieval with the CoDA
integrated; (3) Direct retrieval with our GaP integrated. In
the multi-round setting, we compare three corresponding se-
tups: (1) ActiShade without the GaP method; (3) ActiShade
replacing GaP with the CoDA method; (3) ActiShade (the
full pipeline). The experimental results are shown in Table 2.
We observe that, in both single-round and multi-round re-
trieval settings, models incorporating our GaP method con-
sistently outperform those without such integration, high-

Gloriabridge
birthplaceD
Major0.050.10.20.30.4Sigma( )
ASEANcountryDam
tournamenthost
0.30.40.50.60.7
ScoreFigure 4: Visualization analysis of the Gaussian noise stan-
dard deviationσ.
lighting the effectiveness of GaP. In addition, we observe
that, on the MuSiQue and 2WikiMQA datasets, models us-
ing CoDA even perform worse than those without knowl-
edge overshadowing detection. This indicates that CoDA’s
token-removing approach may disrupt the reasoning chain
in multi-hop questions, thereby limiting its effectiveness.
Sensitivity and Interpretability Analysis of GaP.We
conduct a sensitivity analysis to investigate how varying
the standard deviationσof the Gaussian noise used for
keyphrase perturbation affects the performance of our pro-
posed ActiShade. All experiments in this analysis are con-
ducted using the Llama-3-8B-Instruct, and the results are
evaluated based on the F1 metric. As shown in Figure 3, we
varyσin the range of [0.05, 0.5] and observe its impact on
the final performance. Experimental results show that asσ
decreases, the model performance first improves, reaching
a peak atσ= 0.1, and then gradually declines. This indi-
cates that a moderate level of noise can effectively help de-
tect overshadowed keyphrases, while excessive noise causes
large output distribution shifts for all candidate keyphrases,
reducing the effectiveness of detection. Nevertheless, the
overall performance remains relatively stable across a wide
range ofσvalues, suggesting that our method exhibits low
sensitivity to this hyperparameter.
To interpret how GaP detects overshadowed keyphrases,
we also conduct a visualization analysis of output distribu-
tion similarity across different keyphrases. We randomly se-
lect two queries and apply Gaussian noise of varying stan-
dard deviation to each candidate keyphrase. For each combi-
nation of keyphrase and noise levelσ, we compute the sim-
ilarity between the model’s output distributions before and
after perturbation. A high similarity suggests the keyphrase
has little influence on the output and is likely overshadowed.
Figure 4 shows that moderate perturbation best separates
salient from overshadowed keyphrases, while stronger noise
disrupts all outputs, lowers similarities across the board, and
weakens detection.
Analysis of Retriever Training
We analyze the effectiveness of retriever training in the Ac-
tiShade framework by evaluating both the retrieval capa-
bility and the downstream QA performance under different
training strategies.ModelPos Semi Pos&Semi
R@1 R@3 R@1 R@3 R@1 R@3
Base 29.20 50.40 12.57 25.42 18.29 36.78
SCL 57.84 69.21 40.12 59.9938.2150.29
FCL75.33 84.80 43.21 61.4238.1452.72
Table 3: Comparison of Recall@1 and Recall@3 for differ-
ent retrievers.
Model / DatasetMuSiQue HotpotQA 2WikiMQA
ActiShade26.94 56.33 46.02
-w/SCL 24.10 54.25 44.97
-w/oFCL 25.68 53.89 44.61
Table 4: Evaluation of retriever training strategies in Ac-
tiShade. The performance is evaluated using the F1 score.
We first assess the retrieval ability of three retriever vari-
ants: (1) a retriever without task-specific fine-tuning (Base),
(2) our proposed retriever trained with fine-grained con-
trastive learning (FCL), and (3) a retriever trained using stan-
dard contrastive learning (SCL) that only distinguishes be-
tween positive and negative examples. As shown in Table 3,
our method achieves the highest Recall@kscores on both
positive and semi-positive document retrieval, demonstrat-
ing its effectiveness in capturing multi-level document rele-
vance critical for multi-hop reasoning. When distinguishing
between positive&semi-positive and negative examples, our
method performs comparably to the retriever trained with
standard contrastive learning on Recall@1, while outper-
forming it on Recall@3. This indicates that our improved
contrastive learning objective helps the retriever better dis-
tinguish positive from semi-positive examples, while still ef-
fectively discriminating negative ones.
We then examine how these retrievers affect the final QA
performance. As presented in Table 6, our proposed retriever
achieves the best results across three datasets. This shows
that training the retriever to distinguish varying degrees of
relevance is beneficial not only for retrieval capability but
also for downstream answer generation. Notably, even with-
out retriever training, ActiShade still outperforms previous
baselines, highlighting the effectiveness of the Knowledge
Overshadowing Detection and Query Formulation modules
in the overall framework.
Conclusion
In this paper, we introduce ActiShade, a novel multi-round
retrieval framework for multi-hop reasoning. ActiShade it-
eratively detects overshadowed keyphrases in the query, re-
trieves documents relevant to both the query and the over-
shadowed keyphrase, and generates a new query based on
the retrieved documents for the next iteration, thereby reduc-
ing the error accumulation caused byknowledge overshad-
owing. Extensive experiments demonstrate the effectiveness
of ActiShade across multiple datasets and LLMs.

Acknowledgements
This work was supported by the National Key Re-
search and Development Program of China (Grant No.
2024YFE0210800) and the National Natural Science Foun-
dation of China (Grant No. 62476025).
References
Bang, Y .; Cahyawijaya, S.; Lee, N.; Dai, W.; Su, D.; Wilie,
B.; Lovenia, H.; Ji, Z.; Yu, T.; Chung, W.; Do, Q. V .; Xu,
Y .; and Fung, P. 2023. A Multitask, Multilingual, Multi-
modal Evaluation of ChatGPT on Reasoning, Hallucination,
and Interactivity. In Park, J. C.; Arase, Y .; Hu, B.; Lu, W.;
Wijaya, D.; Purwarianti, A.; and Krisnadhi, A. A., eds.,Pro-
ceedings of the 13th International Joint Conference on Nat-
ural Language Processing and the 3rd Conference of the
Asia-Pacific Chapter of the Association for Computational
Linguistics (Volume 1: Long Papers), 675–718. Nusa Dua,
Bali: Association for Computational Linguistics.
Borgeaud, S.; Mensch, A.; Hoffmann, J.; Cai, T.; Ruther-
ford, E.; Millican, K.; van den Driessche, G.; Lespiau, J.;
Damoc, B.; Clark, A.; de Las Casas, D.; Guy, A.; Menick, J.;
Ring, R.; Hennigan, T.; Huang, S.; Maggiore, L.; Jones, C.;
Cassirer, A.; Brock, A.; Paganini, M.; Irving, G.; Vinyals,
O.; Osindero, S.; Simonyan, K.; Rae, J. W.; Elsen, E.; and
Sifre, L. 2022. Improving Language Models by Retriev-
ing from Trillions of Tokens. In Chaudhuri, K.; Jegelka, S.;
Song, L.; Szepesv ´ari, C.; Niu, G.; and Sabato, S., eds.,In-
ternational Conference on Machine Learning, ICML 2022,
17-23 July 2022, Baltimore, Maryland, USA, volume 162
ofProceedings of Machine Learning Research, 2206–2240.
PMLR.
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan,
J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.;
Amanda, A.; Agarwal, S.; Herbert-V oss, A.; Krueger, G.;
Tom, H.; Child, R.; Ramesh, A.; Ziegler, D.; Wu, J.; Win-
ter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray,
S.; Benjamin, C.; Clark, J.; Berner, C.; Sam, M.; Radford,
A.; Sutskever, I.; and Amodei, D. 2020. Language Mod-
els are Few-Shot Learners.arXiv: Computation and Lan-
guage,arXiv: Computation and Language.
Cao, S.; Zhang, J.; Shi, J.; Lv, X.; Yao, Z.; Tian, Q.; Hou,
L.; and Li, J. 2023. Probabilistic Tree-of-thought Reasoning
for Answering Knowledge-intensive Complex Questions. In
Bouamor, H.; Pino, J.; and Bali, K., eds.,Findings of the
Association for Computational Linguistics: EMNLP 2023,
Singapore, December 6-10, 2023, 12541–12560. Associa-
tion for Computational Linguistics.
Chu, Z.; Chen, J.; Chen, Q.; Wang, H.; Zhu, K.; Du, X.; Yu,
W.; Liu, M.; and Qin, B. 2024. BeamAggR: Beam Aggre-
gation Reasoning over Multi-source Knowledge for Multi-
hop Question Answering. In Ku, L.; Martins, A.; and Sriku-
mar, V ., eds.,Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long
Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024,
1229–1248. Association for Computational Linguistics.
Erker, J.-J.; Reimers, N.; and Gurevych, I. 2025. GRITHop-
per: Decomposition-Free Multi-Hop Dense Retrieval.arXiv
preprint arXiv:2503.07519.Fan, W.; Ding, Y .; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua,
T.; and Li, Q. 2024. A Survey on RAG Meeting LLMs:
Towards Retrieval-Augmented Large Language Models. In
Baeza-Yates, R.; and Bonchi, F., eds.,Proceedings of the
30th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining, KDD 2024, Barcelona, Spain, August 25-
29, 2024, 6491–6501. ACM.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Guo, Q.; Wang, M.; and Wang, H. 2023. Retrieval-
Augmented Generation for Large Language Models: A Sur-
vey.CoRR, abs/2312.10997.
Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.
2020. Retrieval Augmented Language Model Pre-Training.
InProceedings of the 37th International Conference on Ma-
chine Learning, ICML 2020, 13-18 July 2020, Virtual Event,
volume 119 ofProceedings of Machine Learning Research,
3929–3938. PMLR.
Ho, X.; Duong Nguyen, A.-K.; Sugawara, S.; and Aizawa,
A. 2020. Constructing A Multi-hop QA Dataset for Com-
prehensive Evaluation of Reasoning Steps. In Scott, D.;
Bel, N.; and Zong, C., eds.,Proceedings of the 28th Inter-
national Conference on Computational Linguistics, 6609–
6625. Barcelona, Spain (Online): International Committee
on Computational Linguistics.
Honnibal, M. 2017. spaCy 2: Natural language understand-
ing with Bloom embeddings, convolutional neural networks
and incremental parsing.(No Title).
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2023. A
Survey on Hallucination in Large Language Models: Prin-
ciples, Taxonomy, Challenges, and Open Questions.CoRR,
abs/2311.05232.
Izacard, G.; Caron, M.; Hosseini, L.; Riedel, S.; Bojanowski,
P.; Joulin, A.; and Grave, E. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv preprint
arXiv:2112.09118.
Izacard, G.; Lewis, P. S. H.; Lomeli, M.; Hosseini, L.;
Petroni, F.; Schick, T.; Dwivedi-Yu, J.; Joulin, A.; Riedel,
S.; and Grave, E. 2023. Atlas: Few-shot Learning with Re-
trieval Augmented Language Models.J. Mach. Learn. Res.,
24: 251:1–251:43.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y .; Ishii, E.;
Bang, Y .; Madotto, A.; and Fung, P. 2023. Survey of Hal-
lucination in Natural Language Generation.ACM Comput.
Surv., 55(12): 248:1–248:38.
Jiang, Z.; Xu, F.; Gao, L.; Sun, Z.; Liu, Q.; Dwivedi-Yu, J.;
Yang, Y .; Callan, J.; and Neubig, G. 2023. Active Retrieval
Augmented Generation. In Bouamor, H.; Pino, J.; and Bali,
K., eds.,Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, 7969–7992. Sin-
gapore: Association for Computational Linguistics.
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017.
TriviaQA: A Large Scale Distantly Supervised Challenge
Dataset for Reading Comprehension. In Barzilay, R.; and
Kan, M., eds.,Proceedings of the 55th Annual Meeting of the
Association for Computational Linguistics, ACL 2017, Van-

couver, Canada, July 30 - August 4, Volume 1: Long Papers,
1601–1611. Association for Computational Linguistics.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A. P.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.;
Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019. Nat-
ural Questions: a Benchmark for Question Answering Re-
search.Trans. Assoc. Comput. Linguistics, 7: 452–466.
Meta, A. 2024a. Introducing meta llama 3: The most capable
openly available llm to date.Meta AI.
Meta, A. 2024b. Introducing meta llama 3: The most capa-
ble openly available llm to date.Meta AI.
OpenAI. 2023. GPT-4 Technical Report.CoRR,
abs/2303.08774.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N.; and
Lewis, M. 2023. Measuring and Narrowing the Composi-
tionality Gap in Language Models. In Bouamor, H.; Pino,
J.; and Bali, K., eds.,Findings of the Association for Compu-
tational Linguistics: EMNLP 2023, 5687–5711. Singapore:
Association for Computational Linguistics.
Shao, Z.; Gong, Y .; Shen, Y .; Huang, M.; Duan, N.; and
Chen, W. 2023. Enhancing Retrieval-Augmented Large
Language Models with Iterative Retrieval-Generation Syn-
ergy. In Bouamor, H.; Pino, J.; and Bali, K., eds.,Findings
of the Association for Computational Linguistics: EMNLP
2023, 9248–9274. Singapore: Association for Computa-
tional Linguistics.
Shi, W.; Min, S.; Yasunaga, M.; Seo, M.; James, R.; Lewis,
M.; Zettlemoyer, L.; and Yih, W. 2024. REPLUG: Retrieval-
Augmented Black-Box Language Models. In Duh, K.;
G´omez-Adorno, H.; and Bethard, S., eds.,Proceedings of
the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), NAACL 2024,
Mexico City, Mexico, June 16-21, 2024, 8371–8384. Asso-
ciation for Computational Linguistics.
Su, W.; Tang, Y .; Ai, Q.; Wu, Z.; and Liu, Y . 2024. DRA-
GIN: Dynamic Retrieval Augmented Generation based on
the Real-time Information Needs of Large Language Mod-
els. In Ku, L.-W.; Martins, A.; and Srikumar, V ., eds.,
Proceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Papers),
12991–13013. Bangkok, Thailand: Association for Compu-
tational Linguistics.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 539–554.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2023. Interleaving Retrieval with Chain-of-Thought Rea-
soning for Knowledge-Intensive Multi-Step Questions. In
Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds.,Pro-
ceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers),
10014–10037. Toronto, Canada: Association for Computa-
tional Linguistics.Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; ichter, b.;
Xia, F.; Chi, E.; Le, Q. V .; and Zhou, D. 2022. Chain-of-
Thought Prompting Elicits Reasoning in Large Language
Models. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Bel-
grave, D.; Cho, K.; and Oh, A., eds.,Advances in Neural
Information Processing Systems, volume 35, 24824–24837.
Curran Associates, Inc.
Xiong, W.; Lewis, P.; Riedel, S.; Li, X.; Wang, W.; Iyer, S.;
Mehdad, Y .; Kiela, D.; Du, J.; Yih, W.; et al. 2020. An-
swering Complex Open-Domain Questions with Multi-Hop
Dense Retrieval. InICLR 2021-9th International Confer-
ence on Learning Representations, volume 2021. ICLR.
Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
Li, C.; Liu, D.; Huang, F.; Wei, H.; et al. 2024. Qwen2. 5
technical report.arXiv preprint arXiv:2412.15115.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W.; Salakhut-
dinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. In
Proceedings of the 2018 Conference on Empirical Methods
in Natural Language Processing.
Zhang, J.; Zhang, H.; Zhang, D.; Yong, L.; and Huang, S.
2024. End-to-End Beam Retrieval for Multi-Hop Question
Answering. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies (Volume
1: Long Papers), 1718–1731.
Zhang, Y .; Li, S.; Qian, C.; Liu, J.; Yu, P.; Han, C.; Fung,
Y . R.; McKeown, K.; Zhai, C.; Li, M.; et al. 2025. The
law of knowledge overshadowing: Towards understanding,
predicting, and preventing llm hallucination.arXiv preprint
arXiv:2502.16143.
Zhang, Z.; Zhang, X.; Ren, Y .; Shi, S.; Han, M.; Wu, Y .; Lai,
R.; and Cao, Z. 2023. IAG: Induction-Augmented Gener-
ation Framework for Answering Reasoning Questions. In
Bouamor, H.; Pino, J.; and Bali, K., eds.,Proceedings of
the 2023 Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore, December 6-
10, 2023, 1–14. Association for Computational Linguistics.
Zhou, D.; Sch ¨arli, N.; Hou, L.; Wei, J.; Scales, N.; Wang,
X.; Schuurmans, D.; Cui, C.; Bousquet, O.; Le, Q. V .; and
Chi, E. H. 2023. Least-to-Most Prompting Enables Complex
Reasoning in Large Language Models. InThe Eleventh In-
ternational Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.
Zhuang, Z.; Zhang, Z.; Cheng, S.; Yang, F.; Liu, J.; Huang,
S.; Lin, Q.; Rajmohan, S.; Zhang, D.; and Zhang, Q. 2024.
EfficientRAG: Efficient Retriever for Multi-Hop Question
Answering. In Al-Onaizan, Y .; Bansal, M.; and Chen, Y .-
N., eds.,Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, 3392–3411. Mi-
ami, Florida, USA: Association for Computational Linguis-
tics.

Appendix for ActiShade
Details of Data Preparation
We construct our training dataset for the retriever
based on MuSiQue (Trivedi et al. 2022). Each data in
MuSiQue dataset is formulated in a dictionary format with
the keyquestion decomposition,question, and
paragraphs.
Thequestion decompositionfield provides a list
of sub-questions derived from the original question, each
annotated with the supporting document required to an-
swer it, which can be found in theparagraphsset. The
paragraphsfield contains a set of documents that are ei-
ther relevant or irrelevant to the question. Documents that
support any sub-question are considered relevant, while the
rest are treated as irrelevant. An example is shown in the
Table 5 to illustrate the data structure.
For each data in MuSiQue, we first consider the subject
entity of the first sub-question as the keyphrase; in the ex-
ample, this isGloria in D Major. The supporting document
associated with the first sub-question is labeled as thepos-
itive document (D+)—in this case, document2. The sup-
porting documents for other sub-questions, which are nec-
essary for answering the original question but not directly
related to the keyphrase, are labeled assemi-positive docu-
ments (D∗); here, documents1and9fall into this category.
All remaining documents are labeled asnegative document
(D−), which are irrelevant to both the keyphrase and the
original question.
Details of LLMs
The details of selected LLMs are as follows:
• Llama3 (Meta 2024a) is a collection of pre-trained
and instruction-tuned LLMs in 8 and 70B sizes. The
instruction-tuned LLMs, called Llama-3-Instruction, are
optimized for dialogue use cases. We selectedLlama-3-
8B-Instruction.
• Qwen2.5 (Yang et al. 2024) is a series of Qwen large
language models, including both base and instruction-
tuned versions, spanning a parameter scale from 0.5 bil-
lion to 72 billion. We selectedQwen2.5-7B-Instructand
Qwen2.5-14B-Instruct.
Prompt Template
The Query Formulation module of ActiShade formulate a
new query based on the retrieved documentss for the next-
round iteration. This process consists of three steps: rele-
vant document selection, query generation, and subsequent-
round decision. The prompt templates used in this process
are detailed below.
Prompt Template for Relevant Document SelectionWe
first prompt the LLM to select the most relevant retrieved
document through the prompt templates, as illustrated in
Figure 5, Figure 6, and Figure 7.
Prompt Template for Query GenerationWe then
prompt the LLM to generate a new query based on the most
relevant retrieved document. This newly generated query ex-
plicitly presents implicit reasoning results and is used for thenext-round iteration. The used prompt templates are shown
in Figure 8, Figure 9, and Figure 10.
Prompt Template for Subsequent-Round Retrieval De-
cisionFinally, we prompt the LLM to determine whether
the given query is single-hop. If it is, this indicates that suffi-
cient information to answer the initial query can be obtained
with one more round of retrieval. We then perform an addi-
tional retrieval round and terminate the process. The prompt
templates used for this step are presented in Figure 11, Fig-
ure 12, and Figure 13.
Ablation Study on Query Formulation
We further examine the effectiveness of the relevant docu-
ment selection step of the Query Formulation module. In this
step, we prompt the LLM to select the most relevant doc-
ument from the retrieved set to guide the generation of the
next-round retrieval query. In the ablated variant, we directly
use the document with the highest retrieval score, without
performing relevant document selection. As shown in Ta-
ble 6, excluding this step leads to a decline in performance.
This suggests that the relevant document selection step en-
sures that the chosen document is beneficial for the follow-
ing query generation step, thereby achieving better perfor-
mance.
Case Study
To illustrate how our method effectively mitigate the error
accumulation caused byknowledge overshadowingin multi-
hop reasoning, we present a representative example as fol-
lows.
Initial Question: What is the name of
the famous bridge in the birthplace of
Gloria in D Major’s composer?
In the first round, the detected candidate keyphrases along
with their corresponding scores are as follows:
Gloria 0.68, bridge 0.39, birthplace
0.59, composer 0.50.
Based on these scores, we considerGloriaas the over-
shadowed keyphrase. After retrieval and relevant document
selection, we obtain the following document:
Title: Gloria (Vivaldi) | Antonio
Vivaldi wrote at least three settings
of the hymn Gloria in excelsis Deo ...
The next-round query is then reformulated as:
What is the name of the famous bridge
in the birthplace of Antonio Vivaldi?
In this second iteration, the detected candidate keyphrases
and their corresponding scores are listed as follows:
Antonio Vivaldi 0.63, bridge 0.15,
birthplace 0.25
Given these scores, we regardAntonio Vivaldias the
overshadowed keyphrase. After retrieval and relevant docu-
ment selection, we obtain the following document:
Title: Antonio Vivaldi | Antonio Lucio
Vivaldi was an Italian Baroque musical
composer. Born in Venice ...
The third-round query is then reformulated as:

questionWhat is the name of the famous bridge in the birthplace ofGloria in D Major’s composer?
question decompositionquestionGloria in D Major>>composer
paragraph support idx2
question #1>>place of birth
paragraph support idx1
question what is the name of the famous bridge in #2
paragraph support idx9
paragraphsidx 0
paragraph text The Dufferin Street bridges are two...
idx1
paragraph text Orlando furioso RV 819...
idx2
paragraph text Antonio Vivaldi wrote at least three...
...
idx9
paragraph text The Rialto Bridge...
...
...
Table 5: A data example from the MuSiQue dataset.
Model / DatasetMuSiQue HotpotQA 2WikiMQA
ActiShade26.94 56.33 46.02
-w/oselection 25.10 55.58 42.48
Table 6: Ablation results on Query Formulation. The perfor-
mance is evaluated using the F1 score.
What is the name of the famous bridge
in Venice?
Since this query is single-hop, we perform an additional
round and then terminate iteration. The detected candidate
keyphrases and scores include the following:
Venice: 0.79, bridge: 0.34;
Given these scores, we considerVeniceas the overshad-
owed keyphrase. After retrieval and relevant document se-
lection, we obtain the following document:
Title: Rialto Bridge | The Rialto
Bridge is the oldest of the four
bridges in Venice, Italy ...
After iteration, we input the initial question along with
the relevant documents retrieved during the iterative pro-
cess into the LLM to generate the final answerRialto
Bridge.
Comparison with Decomposition-Free Multi-Hop
Retrieval Methods
We also compare our method with decomposition-free
multi-hop retrieval approaches. These methods aim to re-
trieve all supporting evidence in a single step by training
the retriever to capture multi-hop relevance patterns. They
primarily focus on enhancing the retriever’s ability to iden-
tify a set of documents that support complex questions. Incontrast, our work is designed to improve the reasoning
capability of large language models (LLMs). We adopt a
retrieval-augmented generation (RAG) framework and in-
troduce a multi-round interaction mechanism that enables
the retriever and the LLM to collaborate iteratively. Specif-
ically, our method detects keyphrases that might be over-
looked by the LLM, retrieves documents relevant to these
keyphrases, and reformulates the query to better guide sub-
sequent reasoning. Therefore, our focus is on enhancing re-
trieval–generation synergy, rather than achieving one-shot
retrieval accuracy. Moreover, these two lines of research are
typically not directly compared in prior work, as they focus
on different objectives: decomposition-free methods aim to
improve retrieval accuracy and document coverage, while
our method targets end-to-end QA performance by enhanc-
ing the interaction between retrieval and generation to sup-
port LLM reasoning.
Although our approach is conceptually different from
decomposition-free methods, we include a comparison with
representative baselines in this appendix to address the con-
cerns raised by reviewers. We compare our method with sev-
eral representative decomposition-free multi-hop retrieval
approaches, including MDR (Xiong et al. 2020), Beam
Retrieval (Zhang et al. 2024), and GRITHopper (Erker,
Reimers, and Gurevych 2025). For fair evaluation, all mod-
els are trained separately on the HotpotQA, 2WikiMQA, and
MuSiQue datasets using their respective official implemen-
tations and training settings. We use the trained retrievers as
plug-in retrieval modules and concatenate their top-retrieved
documents with the original query, feeding the resulting in-
put into the Llama-3-8B-Instruct model for answer gener-
ation. The experimental results are shown in Table 7. Ac-
tiShade consistently outperforms these decomposition-free
baselines across all three datasets.

Prompt 1
Given a question along with the retrieved document, to begin with, you should identify and
output the subquestion(s) that need to be answered first in order to answer the given question.
Then, determine whether the retrieved document contains relevant information to answer at
least one subquestion. If it does, output 'Yes'; otherwise, output 'No'.
<example>
Example 1:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Retrieved document: Scott Derrickson (born July 16, 1966) is an American director,
screenwriter and producer. He lives in Los Angeles, California. He is best known for
directing horror films such as \\\"Sinister\\\", \\\"The Exorcism of Emily Rose\\\", and
\\\"Deliver Us From Evil\\\", as well as the 2016 Marvel Cinematic Universe installment,
\\\"Doctor Strange.\\\"
Output: The subquestion that needs to be answered first is: "What is the nationality of Scott
Derrickson?" or "What is the nationality of Ed Wood?". The retrieved document clarifies that
Scott Derrickson is American, containing relevant information to answer at least one
subquestion. Therefore, the final output is Yes.
...
</example>
Question: {}
Retrieved document: {}Figure 5: Prompt template for relevant document selection in HotpotQA.
Model / DatasetMuSiQue HotpotQA 2WikiMQA
MDR 19.19 45.23 38.74
Beam Retrieval 19.79 47.37 38.98
GritHopper 22.13 50.76 41.30
ActiShade(ours)26.94 56.33 46.02
Table 7: Comparison with decomposition-free multi-hop re-
trievers on three datasets. All results are reported in F1 score.

Prompt 2
Given a question along with the retrieved document, to begin with, you should identify and
output the subquestion(s) that need to be answered first in order to answer the given question.
Then, determine whether the retrieved document contains relevant information to answer at
least one subquestion. If it does, output 'Yes'; otherwise, output 'No'.
<example>
Example 1:
Question: Who is the mother of the director of film Polish-Russian War (Film)?
Retrieved document: Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film
directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red
flag by Dorota Masłowska.
Output: The subquestion that needs to be answered first is: "Who is the director of film
Polish-Russian War (Film)?". The retrieved document clarifies that Polish-Russian War is a
film directed by Xawery Żuławski, containing relevant information to answer this
subquestion. Therefore, the final output is Yes.
...
</example>
Question: {}
Retrieved document: {}Figure 6: Prompt template for relevant document selection in 2WikiQA.

Prompt 3
Given a question along with the retrieved document, to begin with, you should identify and 
output the subquestion that needs to be answered first in order to answer the given question. 
Then, determine whether the retrieved document contains relevant information to answer 
this subquestion. If it does, output 'Yes'; otherwise, output 'No'.
<example>
Example 1:
Question: Who is the spouse of the Green performer? 
Retrieved document: Green is the fourth studio album by British progressive rock musician 
Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-
inflected \\\"Motivation Radio\\\" (1977), \\\"Green\\\" was originally going to be released as 
\\\"The Green Album\\\" as a companion to \\\"The Red Album\\\" (the originally intended 
name for \\\"Motivation Radio\\\"). However, this plan was dropped and after a US tour in 
late 1977, \\\"Green\\\" was recorded alone, primarily in Dorking, Surrey, and in London.
Output: The subquestion that needs to be answered first is: "Who is the performer of Green?". 
The retrieved document clarifies that Green is an album by Steve Hillage. Therefore, the 
final output is Yes.
...
</example>
Question: {}
Retrieved document: {}Figure 7: Prompt template for relevant document selection in MuSiQue.

Prompt 4
Given a question along with the retrieved document, you are required to refactor the question
based on the retrieved document, reducing the reasoning steps required to answer it while
maintaining its original intent. Typically, you should only remove words in the question or
replace them with words from the retrieved document, except when adjustments are required
to satisfy grammatical rules. Before outputting the refactored question, provide a brief
explanation of your reasoning process.
<example>
Example 1:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Retrieved document: Scott Derrickson (born July 16, 1966) is an American director,
screenwriter and producer. He lives in Los Angeles, California. He is best known for
directing horror films such as \\\"Sinister\\\", \\\"The Exorcism of Emily Rose\\\", and
\\\"Deliver Us From Evil\\\", as well as the 2016 Marvel Cinematic Universe installment,
\\\"Doctor Strange.\\\"
Explanation: The retrieved document clarifies that Scott Derrickson is an American director.
Therefore, in the original question, "Scott Derrickson and" can be removed, "of the same
nationality" can be replaced with "American", and "Were" should be changed to "Was" to
satisfy grammatical rules.
Refactored question: Was Ed Wood American?
...
</example>
Question: {}
Retrieved document: {}Figure 8: Prompt template for query generation in HotpotQA.

Prompt 5
Given a question along with a retrieved document, to begin with, you should extract and
output the parts of the document that contain information related to answering the question.
Then, refactor the question based on the extracted parts, reducing the reasoning steps
required to answer it while maintaining its original intent. Finally, output the refactored
question.
<example>
Example 1:
Question: Who is the mother of the director of film Polish-Russian War (Film)?
Retrieved document: Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film
directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red
flag by Dorota Masłowska.
Output: The retrieved document clarifies that Polish-Russian War is a film directed by
Xawery Żuławski. Therefore, the question can be refactored as "Who is the mother of
Xawery Żuławski?".
...
</example>
Question: {}
Retrieved document: {}Figure 9: Prompt template for query generation in 2WikiQA.

Prompt 6
Given a question along with retrieved document, you are required to refactor the question
based on the retrieved document, reducing the reasoning steps required to answer it while
maintaining its original intent. You can only replace words in the question with words from
the retrieved document. You are not allowed to add, change, or reorder words. Before
outputting the refactored question, provide a brief explanation of your reasoning process.
<example>
Example 1:
Question: Who is the spouse of the Green performer?
Retrieved document: Green is the fourth studio album by British progressive rock musician
Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-
inflected \\\"Motivation Radio\\\" (1977), \\\"Green\\\" was originally going to be released as
\\\"The Green Album\\\" as a companion to \\\"The Red Album\\\" (the originally intended
name for \\\"Motivation Radio\\\"). However, this plan was dropped and after a US tour in
late 1977, \\\"Green\\\" was recorded alone, primarily in Dorking, Surrey, and in London.
Explanation: The retrieved document clarifies that Green is an album by Steve Hillage.
Therefore, "the Green performer" in the original question can be replaced with "Steve
Hillage".
Refactored question: Who is the spouse of Steve Hillage?
...
</example>
Question: {}
Retrieved document: {}Figure 10: Prompt template for query generation in MuSiQue.

Prompt 7
Given a question, you should determine whether it is a single-hop question. If it is, output
'Yes'; otherwise, output 'No'. Before outputting 'Yes' or 'No', provide a brief explanation of
your reasoning process.
<example>
Example 1:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Explanation: Answering this question involves multiple steps. First, determine the
nationalities of Scott Derrickson and Ed Wood. Then, compare their nationalities to see if
they are the same. Since this process requires combining multiple pieces of knowledge, it is
not a single-hop question.
Output: No
...
</example>
Question: {}Figure 11: Prompt template for subsequent-round retrieval decision in HotpotQA.
Prompt 8
Given a question, you should determine whether it is a single-hop question. If it is, output
'Yes'; otherwise, output 'No'. Before outputting 'Yes' or 'No', provide a brief explanation of
your reasoning process.
<example>
Example 1:
Question: Who is the mother of the director of film Polish-Russian War (Film)?
Explanation: Answering this question involves multiple steps. First, identify the director of
the film Polish-Russian War. Then, determine the mother of that director. Therefore, it is not
a single-hop question.
Output: No
...
</example>
Question: {}
Figure 12: Prompt template for subsequent-round retrieval decision in 2WikiQA.

Prompt 9
Given a question, you should determine whether it is a single-hop question. If it is, output
'Yes'; otherwise, output 'No'. Before outputting 'Yes' or 'No', provide a brief explanation of
your reasoning process.
<example>
Example 1:
Question: Who is the spouse of Steve Hillage?
Explanation: Answering this question involves one step: determining Steve Hillage's spouse.
Therefore, it is a single-hop question.
Output: Yes
...
</example>
Question: {}Figure 13: Prompt template for subsequent-round retrieval decision in MuSiQue.
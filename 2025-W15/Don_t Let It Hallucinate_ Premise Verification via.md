# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma

**Published**: 2025-04-08 21:14:48

**PDF URL**: [http://arxiv.org/pdf/2504.06438v1](http://arxiv.org/pdf/2504.06438v1)

## Abstract
Large language models (LLMs) have shown substantial capacity for generating
fluent, contextually appropriate responses. However, they can produce
hallucinated outputs, especially when a user query includes one or more false
premises-claims that contradict established facts. Such premises can mislead
LLMs into offering fabricated or misleading details. Existing approaches
include pretraining, fine-tuning, and inference-time techniques that often rely
on access to logits or address hallucinations after they occur. These methods
tend to be computationally expensive, require extensive training data, or lack
proactive mechanisms to prevent hallucination before generation, limiting their
efficiency in real-time applications. We propose a retrieval-based framework
that identifies and addresses false premises before generation. Our method
first transforms a user's query into a logical representation, then applies
retrieval-augmented generation (RAG) to assess the validity of each premise
using factual sources. Finally, we incorporate the verification results into
the LLM's prompt to maintain factual consistency in the final output.
Experiments show that this approach effectively reduces hallucinations,
improves factual accuracy, and does not require access to model logits or
large-scale fine-tuning.

## Full Text


<!-- PDF content starts -->

Don’t Let It Hallucinate: Premise Verification via
Retrieval-Augmented Logical Reasoning
Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao∗, Xuezhe Ma∗
University of Southern California
{yuehanqi, li.li02, yinian, xinyany, yzhao010, xuezhema }@usc.edu
Abstract
Large language models (LLMs) have shown substantial capacity for gen-
erating fluent, contextually appropriate responses. However, they can
produce hallucinated outputs, especially when a user query includes one or
more false premises —claims that contradict established facts. Such premises
can mislead LLMs into offering fabricated or misleading details. Existing
approaches include pretraining, fine-tuning, and inference-time techniques
that often rely on access to logits or address hallucinations after they occur.
These methods tend to be computationally expensive, require extensive
training data, or lack proactive mechanisms to prevent hallucination before
generation, limiting their efficiency in real-time applications. We propose
a retrieval-based framework that identifies and addresses false premises
before generation. Our method first transforms a user’s query into a log-
ical representation, then applies retrieval-augmented generation (RAG)
to assess the validity of each premise using factual sources. Finally, we
incorporate the verification results into the LLM’s prompt to maintain fac-
tual consistency in the final output. Experiments show that this approach
effectively reduces hallucinations, improves factual accuracy, and does not
require access to model logits or large-scale fine-tuning.
1 Introduction
Large Language Models (LLMs) generate fluid, context-aware responses but can produce
hallucinations when prompted with queries that include hidden factual errors (Manakul
et al., 2023; Zheng et al., 2023). These errors, known as false premises , are statements in a
user’s question that conflict with real-world facts. Even when LLMs can store accurate
information, they may trust the incorrect assumptions embedded in the query and generate
misleading outputs (Yuan et al., 2024). This is especially problematic in sensitive applications
such as finance or healthcare, where mistakes can cause serious harm (Pal et al., 2023).
Prior research distinguishes between factuality hallucinations—where the output conflicts
with known facts—and faithfulness hallucinations—where the response diverges from the
provided context or user instructions (Snyder et al., 2024a). We focus on factuality halluci-
nations, particularly those driven by incorrect assumptions (false premises). These errors
are common among LLM outputs (Huang et al., 2025; Snyder et al., 2024b), where Figure 1
presents an example of a question with a false premise and the resulting hallucination.
Many methods attempt to address false premises after an LLM has already produced an
answer. They include fine-tuning the model to detect invalid assumptions (Hu et al., 2023),
applying contrastive decoding to surface inconsistencies (Shi et al., 2023; Chuang et al.,
2024), and using uncertainty-based measures or logits to gauge inaccuracies (Pezeshkpour,
2023; Varshney et al., 2023). Although effective in some contexts, these approaches can be
computationally demanding and do not necessarily prevent misinformation from appearing
in the first place. Additionally, questions with false premises often maintain normal semantic
flow, changing only a few tokens so that they are difficult to identify using traditional out-
of-distribution detection (Vu et al., 2023). Even advanced LLMs can struggle with real-time
∗Corresponding Author
1arXiv:2504.06438v1  [cs.CL]  8 Apr 2025

Is "The Lord of the Rings: The Return of the King" nominated for the AACTA Award for Best Adapted Screenplay?Yes, "The Lord of the Rings: The Return of the King" is indeed nominated for the AACTA Award (Australian Academy of Cinema and Television Arts) for Best Adapted Screenplay. It won this award in 2004 along with several other categories, including Best Director (Peter Jackson) …
Figure 1: LLM experiences factuality hallucination when faced with a false premise question,
where both entities ( The Lord of the Rings: The Return of the King, AACTA Award for
Best Adapted Screenplay ) exist but are not correctly aligned.
truth evaluation, lacking the context or capacity to fully check every assumption (Hu et al.,
2023; Liu et al., 2024c).
To address this challenge, we focus on preventing hallucinations rather than mitigating them
post hoc. In our framework, outlined in Section 3, we first transform the user’s query into
a logical form that highlights key entities or relations ( §3.2). We then employ retrieval-
augmented generation (RAG) to check the accuracy of these statements against a knowledge
graph ( §3.3). If contradictions are found, the query is flagged as containing a false premise
(§3.1), prompting the model to correct or reject the assumption before formulating a final
answer. This process, shown in Figure 2, ensures that the LLM does not rely on erroneous
details during response generation. By informing the LLM about any detected false premise
in advance ( §3.4), we reduce the likelihood of hallucinations without requiring access to
model logits or large-scale fine-tuning.
We summarize our contributions as follows:
Logical Form Representation : We first introduce logical forms to represent input queries
and demonstrate their effectiveness across various types of graph retrievers. This logical ap-
proach enables accurate and systematic evaluation of statements provided in user prompts,
particularly handling queries that may include false premises.
Explicit False Premise Detection : Our method improves the reliability of LLM-generated
responses by explicitly detecting false premises and informing the LLM if a question contains
a false premise.
Hallucination Mitigation Without Output Generation or Model Logits : Our approach
reduces factual hallucinations without actual generation of responses or LLM logits and,
therefore, can be seamlessly integrated into existing LLM frameworks and pipelines, offering
a straightforward enhancement for improving factual accuracy.
2 Related Works
False Premise . A False Premise Question (FPQ) is a question that contains incorrect facts
that are not necessarily explicitly stated but might be mistakenly believed by the questioner
(Yu et al., 2022; Kim et al., 2021). Recent studies (Yuan et al., 2024) have demonstrated that
FPQs can induce factuality hallucination in LLMs, as these models often respond directly to
FPQs without verifying their validity. Notably, existing prompting techniques like few-shot
prompting (Brown et al., 2020) and Chain-of-Thought (Wei et al., 2023), tend to increase
hallucinations. Conversely, directly prompting LLMs to detect false premises negatively
impacts their performance on questions containing valid premises (Vu et al., 2023).
Knowledge Graph Fact Checking and Question Answering . In the related field of fact
checking, Retrieval-Augmented Generation (RAG) approaches are utilized for checking
whether data are true and accurate. Among different RAG tasks, knowledge graph-driven
RAG has drawn attention due to its ability to effectively leverage structured knowledge.
Recent works include 1) prompt based : (Pan et al., 2023) asks LLMs if evidence is sufficient. If
not, they further ask LLMs questions critical for verifying a claim. (Sun et al., 2024) uses the
LLM to retrieve relevant facts hop-by-hop. 2) graph based : (He et al., 2024) extracts subgraph
from knowledge graph by formulating RAG over graph as Prize-Collecting Steiner Tree
2

optimization problem. (Mavromatis & Karypis, 2024) uses graph neural network to reason
over a dense subgraph and retrieve answer candidates for the given question. 3) training
based : (Zheng et al., 2024) trains two encoders for retrieval: one for embedding the queries,
and one for embedding the subgraph evidences. (Liu et al., 2024a) trains encoder to conduct
the following retrieval and ranker processes. However, it requires that entity exists in the
knowledge graph and uses results generated from prompts for training.
Hallucination Mitigation . Sources of LLM hallucinations originate from different stages in
the LLM life cycle (Zhang et al., 2023a), leading existing mitigation methods to target specific
stages: 1) Pre-training : Enhancing factual reliability by emphasizing credible texts, either by
up-sampling trustworthy documents (Touvron et al., 2023) or prepending factual sentences
with topic prefixes (Lee et al., 2023). 2) Supervised Fine-tuning : Curating high-quality,
instruction-oriented datasets (Chen et al., 2024; Cao et al., 2024) improves factual accuracy
more effectively than fine-tuning on unfiltered data, and remains more feasible compared
to extensive pre-training. 3) Reinforcement Learning from Human Feedback : Aligning closely
with human preferences may inadvertently encourage hallucinations or biased outputs,
especially when instructions surpass the model’s existing knowledge (Radhakrishnan et al.,
2023; Wei et al., 2024). 4) Inference : Known as hallucination snowballing (Zhang et al., 2023b),
LLMs occasionally magnify initial generation mistakes. Proposed inference-time solutions
include new decoding strategies (Shi et al., 2023; Chuang et al., 2024), uncertainty analysis
of model outputs (Xu & Ma, 2025; Liu et al., 2024b; Dhuliawala et al., 2023). However,
these approaches either act post-hallucination or require access to model logits, thus being
inefficient due to repeated prompting or limited to white-box LLM scenarios.
3 Methodology
Hallucinations in LLMs often stem from false premises in user queries. Instead of addressing
hallucinations after they occur, we aim to prevent them by detecting ( §3.1) and informing
the presence of false premises to LLMs before response generation. Our proposed method
achieves this through three key steps:
Logical Form Conversion . (§3.2) By converting the user query into a structured logical
representation, we extract its core meaning, making it easier to analyze its factual consistency.
We demonstrate its effectiveness across various types of graph retrievers.
Structured Retrieval and Verification . (§3.3) Rather than relying solely on model-generated
text, we retrieve external evidence to assess whether false premises exist in the query.
Factual Consistency Enforcement . (§3.4) The verified information is then incorporated into
the LLM prompt, ensuring that the model generates responses aligned with factual data.
Our proposed method applies to knowledge graphs and datasets compatible with graph
structures. We show the pseudocode summary of our approach in Algorithm 1.
3.1 Problem Definition
False Premise Detection : Given a user query q, the function F(q)determining whether q
contains a false premise can be defined as:
F(q) =1, if qconflicts with retrieved evidence R(q,G),
0, otherwise.(1)
where Rrepresents the retrieval function that extracts relevant evidence from a knowledge
graph G. The query qis evaluated against R(q,G), and if contradictions are found, qis
deemed to contain a false premise ( F(q) =1); otherwise, it is considered valid ( F(q) =0).
In this study, the function Fis achieved by RAG using a retriever that leverages logical form
and a knowledge graph.
3

Was The Lord of the Rings: The Return of the King nominated for the Academy Award for Best Director?
Nominated("TheLordoftheRings:TheReturnoftheKing","AcademyAwardforBestDirector")QueryLogical Form
Source:TheLordoftheRings:TheReturnoftheKingRelationship: NominatedTarget: AcademyAwardforBestDirectorTriple
Logical FormEvidenceEncoderRelevance DistributionTop-kE_qE_eRelevanceRetriever
Original QueryPremise Verification
AnswerRetrieverKnowledge GraphRetrieved SubgraphEvidenceLogical Form
Premise VerificationConnected subgraph optimized by Prize-Collecting Steiner Tree…Top-k similar triplesTop-k similar nodes and relationsLogical Form EmbeddingEvidence EmbeddingFigure 2: Overview of our approach. Left: The original query is converted into a logical
form. Middle: The logical form is used to retrieve relevant elements from the knowledge
graph and detect false premises. Right: Comparison of studied retrievers for aligning logical
form with the knowledge graph. The LLM generates responses with reduced hallucination
given prompts with premise verification.
3.2 Logical Form Extraction
Logical Form : A logical form is a structured representation of statements or queries ex-
pressed using symbolic logic. It provides a structured way to capture semantic relationships
within sentences, enabling precise and systematic reasoning. Given a natural language sen-
tence query q, its logical form L(q)can be represented as: L(q) =P(x1,x2,. . .,xn), where
Pdenotes a predicate or relation, and x1,x2,. . .,xnare variables or constants representing
entities or concepts extracted from q. For example, for the query: Was The Lord of the
Rings: The Return of the King nominated for the Academy Award for Best Director?
, its logical form is: Nominated("The Lord of the Rings: The Return of the King",
"Academy Award for Best Director") . Here, "The Lord of the Rings: The Return of the
King" and "Academy Award for Best Director" are entities, while Nominated represents the
relation. GPT-4o-mini (OpenAI et al., 2024) is used for extracting logical forms from the
queries. For an input query q, we first ask the LLM to generate the corresponding logical
form L(q). Then, we extract the source, relationship, and target from L(q).
3.3 Retrieval
Given a user query qin natural language, the retrieval stage aims to extract the most relevant
elements (e.g., entities, triplets, paths, subgraphs) from knowledge graphs, which can be
formulated as:
G∗=Graph-Retriever (q,G) =arg max
G⊆R(G)pθ(G|q,G) =arg max
G⊆R(G)Sim(q,G), (2)
where G∗is the optimal retrieved graph elements, and Sim(·,·)is a function that measures
the semantic similarity between user queries and the graph data. R(·)represents a function
to narrow down the search range of subgraphs, considering the efficiency.
After converting a user query qinto a logical form representation L(q), the retriever encodes
the logical form and the graph triples, searches through the knowledge graph G, and extracts
the most relevant triple or subgraph, applying different selection criteria depending on the
retriever used in our study. Therefore, formula 2 can be further formulated to:
G∗=Graph-Retriever (L(q),G) =arg max
G⊆R(G)pθ(G|L(q),G) =arg max
G⊆R(G)Sim(L(q),G).
4

Algorithm 1 False premise detection and hallucination mitigation
Input: User query q, Knowledge graph G
Output: Hallucination mitigated response from LLM
1: Convert user query qinto logical representation L(q) ▷(§3.2)
2: Extract logical assertions P(x1,x2, . . . , xn)fromL(q)
3: Initialize maximum similarity score Sim max← −∞ ▷(§3.3)
4: Initialize optimal graph G∗←∅
5: Candidate set G∗←subsets of relevant subgraphs from G, i.e., R(G)
6:fortriple G′∈Gdo
7: ifretriever is embedding-based then
8: Compute similarity via embeddings:
Sim←Sim L(q),G′
9: else if retriever is non-parametric then
10: Compute similarity using tree search criteria:
Sim←PCST L(q),G′
11: else if retriever is LLM-based then
12: Compute similarity using LLM scoring:
Sim←LLMScore L(q),G′
13: end if
14: ifSim>Sim maxthen
15: Sim max←Sim
16: G∗←G′
17: end if
18:end for
19: Define false premise indicator function: ▷(§3.1)
F(q) =(
1, if qconflicts with retrieved evidence G∗=R(q,G∗)
0, otherwise
20:ifF(q) =1then ▷(§3.4)
21: Update query as:
q←q+” Note: This question contains a false premise.”
22:end if
23: Generate response from LLM using updated query q
24:return Hallucination mitigated response from LLM
We employ the pre-trained encoder all-roberta-large-v11to encode the logical form and
graph triplets. The representation L(q)is used in both the similarity-based retrieval process
and the step where the LLM assesses whether the original query qcontains a false premise.
3.4 Hallucination Mitigation
For a given user query q, if the false premise identification function F(q)detects a false
premise ( F(q) =1), we update its original query qby appending a note:
q=q+ “ Note: This question contains a false premise.”, if F(q) =1,
q, otherwise.(3)
where qrepresents the modified query that explicitly flags the presence of a false premise
when detected. Once the original query is updated, we evaluate LLM’s responses and
measure the effectiveness of the ensuing hallucination mitigation.
1https://huggingface.co/sentence-transformers/all-roberta-large-v1
5

4 Experiments
4.1 Dataset
KG-FPQ (Zhu et al., 2024) is a dataset containing true and false premise questions that are
constructed from the KoPL knowledge graph, a high-quality subset of Wikidata. In KG-FPQ,
GPTs are used to generate TPQs based on true triplets from the knowledge graph, and FPQs
are created by replacing the original object with the edited object from false triplets through
string matching. We look into the discriminative task in the art domain in the dataset. In the
discriminative tasks, LLMs are required to answer Yes-No questions with “Yes” or “No”.
An example for FPQ in Yes-No format is that Is Hercules a cast member of ”The Lord of the
Rings: the Return of the King? . More details of the dataset are in Appendix A.1.
4.2 Experiment Setting
Our approach mitigates hallucination through a two-step process: First, we detect false
premises in the user query ( §4.2.1, §4.2.2). Then, we use the result of false premise detection
along with the original query when providing input to the LLM (§4.2.3).
4.2.1 False Premise Detection with Logical Form
In the false premise detection task, we look at different retrievers with and without the use
of logical forms. Logical forms are used in 1) the retrieval stage, where the logical form L(q)
is encoded to find the most relevant elements from knowledge G, and 2) the false premise
detection stage, where the logical form is passed as input along with the retrieved evidence
to LLM to determine whether the query contains false premise. The prompt detail is in
Appendix A.3. We evaluate the use of logical forms in three configurations: 1) applying
logical forms in both the retrieval stage and false premise detection stage, 2) using logical
forms for retrieval and employing the original query for false premise detection, and 3)
utilizing the original query for both stages.
4.2.2 False Premise Detection Baselines
We evaluate how logical form impacts retrieval for false premise detection across the
following retrievers:
1)Direct Claim : We directly query the LLM to determine whether the given question
contains a false premise. The model is prompted with: Does the following question
contain a false premise? Answer with ’Yes’ or ’No’ only.
2)Embedding-based Retriever :with RAG selects the top-k2relevant triples from the
knowledge graph based on the cosine similarity between the query embedding and the
graph triple embedding.
3)Non-parametric Retriever :G-retriever (He et al., 2024) uses Prize-Collecting Steiner Tree
problem (PCST) algorithm for extracting relevant subgraph from the knowledge graph. It
does not rely on a trained model with learnable parameters.
4)LLM-based Retriever :GraphRAG/ToG (Edge et al., 2025; Sun et al., 2024) asks the LLM
to generate a score between 0 and 100, indicating how helpful the generated answer is in
answering the target question. The answers are sorted in descending order of helpfulness
score and used to generate the final answer returned to the user.
We use GPT-3.5-Turbo as the LLM in the false premise detection task. These retrievers
are included because they enable retrieval without task-specific fine-tuning, making them
more adaptable across different domains. Unlike training-based retrievers, which require la-
beled data and extensive computation, non-parametric retriever uses structured knowledge,
embedding-based retriever utilizes pre-trained encoders to transform queries and knowl-
edge into a shared vector space for efficient retrieval, and LLM-based retrieval leverages
2This work focuses on top-1 selection.
6

TP%TN% FP%
FN%
F1 Score Accuracy20406080100Original Query for Both Stages
TP%TN% FP%
FN%
F1 Score Accuracy20406080100Logical Form  + Original Query
TP%TN% FP%
FN%
F1 Score Accuracy20406080100Logical Form for Both Stages
Direct Claim with RAG G-retriever GraphRAG/T oGFigure 3: Comparison of performance metrics across different retrieval methods using
logical forms and/or original queries.
pre-trained language models’ generalization abilities. This setup evaluates the impact of
logical forms on retrieval efficiency without the overhead of model training.
Direct Claim with RAG G-retriever GraphRAG/ToG
Original Query for Both Stages
True Positives (TP%) 44.44 33.33 37.78 8.89
True Negatives (TN%) 73.33 80.00 100.00 93.33
False Positives (FP%) 26.67 20.00 0.00 6.67
False Negatives (FN%) 55.56 66.67 62.22 91.11
F1 Score (%) 59.70 48.78 ↓ 54.84↓ 16.16
Accuracy (%) 69.20 73.33 91.11 81.27
Logical Form for Retrieval and Original Query for False Premise Detection
True Positives (TP%) 44.44 37.78 37.78 8.89
True Negatives (TN%) 73.33 86.67 100.00 93.33
False Positives (FP%) 26.67 13.33 0.00 6.67
False Negatives (FN%) 55.56 62.22 62.22 91.11
F1 Score (%) 59.70 53.97 ↓ 54.84↓ 16.16
Accuracy (%) 69.20 79.69 91.11 81.27
Logical Form for Both Stages
True Positives (TP%) 44.44 60.00 75.56 8.89
True Negatives (TN%) 73.33 86.67 80.00 93.33
False Positives (FP%) 26.67 13.33 20.00 6.67
False Negatives (FN%) 55.56 40.00 24.44 91.11
F1 Score (%) 59.70 73.97 84.47 16.16
Accuracy (%) 69.20 82.86 79.37 81.27
Table 1: Comparison of performance metrics across different retrieval methods using logical
forms and/or original queries.
4.2.3 Hallucination Mitigation Baselines
Having used logical forms to improve query structuring and false premise detection, we
wish to illustrate how our logical form-based method further reduces hallucinations. We
consider the following methods as our hallucination mitigation baselines, which are all
inference-time hallucination mitigation strategies that do not require access to logits or in-
ternal model weights that operate exclusively at the input level, ensuring a fair comparison:
1)Direct Ask : Directly query the LLMs for an answer without additional processing or
external retrieval. This approach relies on the model’s internal knowledge and reasoning
capabilities to handle potential false premises.
2)Prompt : We encourage the LLM to assess potential false premises before generating a
response by appending the following prompt to the original query:
7

This question may contain a fasle premise. [query]
3)Majority Vote : We prompt the LLM three times with the same prompt and select the
most frequent response as the final answer. This method improves reliability by reducing
the impact of any single erroneous or hallucinated response from LLM.
4)Perplexity AI3: Utilizes a search engine to retrieve and incorporate real-time information
from the web, enabling it to provide answers based on the latest available web data.
ForDirect Ask andMajority Vote , we report the performances of the following LLMs: GPT-
3.5-turbo (OpenAI, 2023), Mistral-7B-Instruct-v0.2 (Jiang et al., 2023), and Qwen-1.5-7b-chat
(Bai et al., 2023).
Models Direct Ask Our Method Prompt Majority Vote Perplexity AI
GPT-3.5 93.3 94.3 ↑ 93.3 92.4 92.4
Mistral-7B 87.6 89.5 ↑ 86.7 87.6 92.4
Qwen1.5 89.5 91.4 ↑ 90.5 90.5 92.4
Table 2: Comparison of accuracy of different hallucination mitigation methods.
4.3 Metrics
In the false premise detection task, we look at several metrics: the true positive rate (TPR),
the true negative rate (TNR), the false positive rate (FPR), the false negative rate (FNR),
the F1 score, and accuracy of the model successfully identifying questions containing false
premises or not. Here, a positive instance refers to a question that contains a false premise,
meaning a higher TPR indicates better detection of false premises. For the hallucination
mitigation task, we focus on question-answering accuracy. We calculate accuracy by string
matching the responses of LLMs: for TPQs, answering “Yes” is considered correct; for FPQs,
answering “No” is considered correct.
5 Discussion
We show the result of the false premise detection task in Table 1 and the hallucination
mitigation result in Table 2.
Using logical forms helps better identify false premises in the questions . As shown in
Table 1, for all three retrievers, explicitly incorporating logical forms into both retrieval and
false premise detection stages significantly improves the identification of false premises.
Sole reliance on original queries, even though potentially yielding high accuracy, tends
to neglect accurate false premise identification, underscoring the importance of utilizing
structured logical forms for tasks prioritizing precise false premise detection.
Among different types of retrievers when using logical forms in both the retrieval and
false premise detection stages, the G-retriever method achieves the highest TPR at 75.56%,
demonstrating a strong capability in accurately identifying questions containing false
premises. Notably, this method also achieves the highest F1 score (84.47%), indicating
an optimal balance between precision and recall. Although the ToG method exhibits the
highest TNR of 93.33%, it significantly underperforms in TPR and overall F1 score (16.16%),
suggesting limited effectiveness in correctly identifying false premises.
Notably, when original queries are used in either retrieval, false premise detection, or both
stages, despite achieving high accuracy (91.11%), G-retriever shows a markedly lower TPR
(37.78%) compared to the first configuration. This suggests that relying on original queries
alone, or in combination with logical forms in only one stage for detection, can achieve
high accuracy due to correctly identifying negatives, it is less effective at capturing false
premises, which is the primary focus of our task.
Explicitly detecting and informing LLMs false premise mitigates hallucination , as demon-
strated in Table 2. Our proposed method, which directly communicates the presence of
3https://www.perplexity.ai
8

false premises to the models, achieves the highest accuracy: 94.3% with GPT-3.5, 91.4% with
Qwen1.5, and 87.6 % with Mistral-7B. This performance surpasses alternative approaches
such as Direct Ask ,Prompt ,Majority Vote , and Perplexity AI , which yield lower accuracy.
Majority Vote does not perform well, likely due to hallucination snowballing, where repeated
querying amplifies errors rather than correcting them. Additionally, while the Prompt
method warns the model about potential false premises, it does not specifically tell the LLM
which one contains false premises, negatively impacts performance on questions with valid
premises, causes unnecessary cautiousness and reduces the model’s ability to provide direct
and confident answers. Besides, Perplexity AI does not perform as well potentially because
the query format does not align well with web data, leading to suboptimal retrieval of
relevant information for certain types of questions. These findings emphasize the importance
of tailoring hallucination mitigation strategies to both the model’s reasoning process and
the nature of the queries it encounters.
Single-hop Multi-hop010203040506070Accuracy (%)48.9 48.355.666.7
48.973.3
Method
Direct Claim
with RAG
G-retriever
Figure 4: GPT-3.5-turbo: False
premise detection accuracy across
single-hop and multi-hop queries.
Using logical form-based RAG
mainly helps detect false premises
in multi-hop questions.Our approach mostly improves false premise de-
tection performance on multi-hop questions , ac-
cording to Figure 4. The incorporation of logi-
cal form-based RAG leads to notable performance
gains compared to direct claim evaluation. Specifi-
cally, while single-hop questions see moderate im-
provement, multi-hop questions benefit the most,
with false premise detection performance increas-
ing from 48.3% in the direct claim setting to 66.7%
with RAG and further to 73.3% when using the G-
retriever. These results suggest that leveraging re-
trieval mechanisms enhances reasoning over multi-
ple pieces of evidence, reinforcing the importance of
retrieval-augmented methods for complex question-
answering tasks. We present a case study to illustrate
how our method improves performance on multi-
hop questions in Appendix A.2.
6 Conclusion
In this work, we propose a retrieval-augmented logical reasoning framework to detect false
premises and then mitigate hallucinations in LLMs. Our method explicitly detects and
signals false premises, overcoming key limitations of current approaches that rely on model
parameters or post-hoc corrections. By incorporating explicit false premise detection, we
effectively mitigate hallucinations without requiring output generation or direct access
to model logits. Our results demonstrate that logical forms significantly improve the
identification of false premises, particularly in multi-hop questions where reasoning over
multiple steps is required. The proposed approach enhances LLM robustness by providing
a structured mechanism to detect and handle misleading inputs before they influence
downstream responses. This reinforces the importance of structured reasoning techniques
in improving model reliability and factual consistency.
Limitations and Future Work. Our work has some limitations. Firstly, the performance
improvement varies across different retrievers and LLMs, and we do not cover all the
situations. As noted in (Zhu et al., 2024), some models such as Llama and Baichuan2 series
may have an inherent bias that causes them to consistently favor one type of answer when
answering Yes-No questions. Secondly, our method requires a prior knowledge graph,
which may not always be feasible and may be difficult to update in line with developments
in the real world. Lastly, we look at the factuality hallucination caused by false premise in
discriminative tasks, which does not encompass all possible tasks. Future work can explore
broader applications of logical form-based detection across diverse reasoning tasks and
investigate integration strategies that further align LLM behavior with verifiable logical
structures.
9

Broader Impact Statement. Our work addresses hallucinations in LLMs by detecting and
mitigating false premises prior to response generation. This approach enhances reliability,
reduces misinformation, and promotes responsible AI use. Ethical considerations, including
fairness and bias, should be carefully addressed in future applications.
References
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu,
Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng
Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Sheng-
guang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao,
Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang,
Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen
technical report, 2023. URL https://arxiv.org/abs/2309.16609 .
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini
Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya
Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen,
Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner,
Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are
few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165 .
Yihan Cao, Yanbin Kang, Chi Wang, and Lichao Sun. Instruction mining: Instruction data
selection for tuning large language models, 2024. URL https://arxiv.org/abs/2307.
06290 .
Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang,
Vijay Srinivasan, Tianyi Zhou, Heng Huang, and Hongxia Jin. Alpagasus: Training a
better alpaca with fewer data, 2024. URL https://arxiv.org/abs/2307.08701 .
Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He.
Dola: Decoding by contrasting layers improves factuality in large language models, 2024.
URL https://arxiv.org/abs/2309.03883 .
Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celiky-
ilmaz, and Jason Weston. Chain-of-verification reduces hallucination in large language
models, 2023. URL https://arxiv.org/abs/2309.11495 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local
to global: A graph rag approach to query-focused summarization, 2025. URL https:
//arxiv.org/abs/2404.16130 .
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V . Chawla, Thomas Laurent, Yann LeCun, Xavier
Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph
understanding and question answering, 2024. URL https://arxiv.org/abs/2402.07630 .
Shengding Hu, Yifan Luo, Huadong Wang, Xingyi Cheng, Zhiyuan Liu, and Maosong
Sun. Won’t get fooled again: Answering questions with false premises, 2023. URL
https://arxiv.org/abs/2307.02394 .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang,
Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey
on hallucination in large language models: Principles, taxonomy, challenges, and open
questions. ACM Transactions on Information Systems , 43(2):1–55, January 2025. ISSN
1558-2868. doi: 10.1145/3703155. URL http://dx.doi.org/10.1145/3703155 .
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
10

Saulnier, L ´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut
Lavril, Thomas Wang, Timoth ´ee Lacroix, and William El Sayed. Mistral 7b, 2023. URL
https://arxiv.org/abs/2310.06825 .
Najoung Kim, Ellie Pavlick, Burcu Karagol Ayan, and Deepak Ramachandran. Which
linguist invented the lightbulb? presupposition verification for question-answering, 2021.
URL https://arxiv.org/abs/2101.00391 .
Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and
Bryan Catanzaro. Factuality enhanced language models for open-ended text generation,
2023. URL https://arxiv.org/abs/2206.04624 .
Haochen Liu, Song Wang, Yaochen Zhu, Yushun Dong, and Jundong Li. Knowledge graph-
enhanced large language models via path selection, 2024a. URL https://arxiv.org/abs/
2406.13862 .
Ollie Liu, Deqing Fu, Dani Yogatama, and Willie Neiswanger. Dellma: Decision making
under uncertainty with large language models, 2024b. URL https://arxiv.org/abs/
2402.02392 .
Ziyi Liu, Soumya Sanyal, Isabelle Lee, Yongkang Du, Rahul Gupta, Yang Liu, and Jieyu
Zhao. Self-contradictory reasoning evaluation and detection, 2024c. URL https://arxiv.
org/abs/2311.09603 .
Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-
box hallucination detection for generative large language models, 2023. URL https:
//arxiv.org/abs/2303.08896 .
Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language
model reasoning, 2024. URL https://arxiv.org/abs/2405.20139 .
OpenAI. Gpt-3.5-turbo: Large language model, 2023. URL https://platform.openai.com/
docs/models/gpt-3-5 . Accessed: 2024-03-18.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat,
Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao,
Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro,
Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brak-
man, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie
Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke
Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen,
Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings,
Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien
Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty
Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Sim ´on Posada Fishman,
Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun
Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott
Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han,
Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade
Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost
Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun
Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali
Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick,
Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt
Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis,
Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike,
Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz
Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam
Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne,
Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil,
David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela
11

Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg
Murk, David M ´ely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan,
Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex
Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy
Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila
Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny,
Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth
Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Fran-
cis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario
Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr,
John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah
Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina
Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski
Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thomp-
son, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry
Tworek, Juan Felipe Cer ´on Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss,
Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei,
CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff,
Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sher-
win Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan,
Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao
Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024.
URL https://arxiv.org/abs/2303.08774 .
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Med-halt: Medical
domain hallucination test for large language models, 2023. URL https://arxiv.org/abs/
2307.15343 .
Liangming Pan, Xinyuan Lu, Min-Yen Kan, and Preslav Nakov. Qacheck: A demonstration
system for question-guided multi-hop fact-checking, 2023. URL https://arxiv.org/abs/
2310.07609 .
Pouya Pezeshkpour. Measuring and modifying factual knowledge in large language models,
2023. URL https://arxiv.org/abs/2306.06264 .
Ansh Radhakrishnan, Karina Nguyen, Anna Chen, Carol Chen, Carson Denison, Danny
Hernandez, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamil ˙e Luko ˇsi¯ut˙e, Newton
Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Sam McCandlish, Sheer El
Showk, Tamera Lanham, Tim Maxwell, Venkatesa Chandrasekaran, Zac Hatfield-Dodds,
Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Question decomposition
improves the faithfulness of model-generated reasoning, 2023. URL https://arxiv.org/
abs/2307.11768 .
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Scott Wen
tau Yih. Trusting your evidence: Hallucinate less with context-aware decoding, 2023.
URL https://arxiv.org/abs/2305.14739 .
Ben Snyder, Marius Moisescu, and Muhammad Bilal Zafar. On early detection of hallu-
cinations in factual question answering. In Proceedings of the 30th ACM SIGKDD Con-
ference on Knowledge Discovery and Data Mining , KDD ’24, pp. 2721–2732, New York,
NY, USA, 2024a. Association for Computing Machinery. ISBN 9798400704901. doi:
10.1145/3637528.3671796. URL https://doi.org/10.1145/3637528.3671796 .
Ben Snyder, Marius Moisescu, and Muhammad Bilal Zafar. On early detection of hallucina-
tions in factual question answering. In Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , pp. 2721–2732, 2024b.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong,
Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible
reasoning of large language model on knowledge graph, 2024. URL https://arxiv.org/
abs/2307.07697 .
12

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas
Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude
Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman
Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas,
Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura,
Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning
Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew
Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva,
Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor,
Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang,
Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic,
Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat
models, 2023. URL https://arxiv.org/abs/2307.09288 .
Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu. A stitch
in time saves nine: Detecting and mitigating hallucinations of llms by validating low-
confidence generation, 2023. URL https://arxiv.org/abs/2307.03987 .
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-
Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. Freshllms: Refreshing large
language models with search engine augmentation, 2023. URL https://arxiv.org/abs/
2310.03214 .
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi,
Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large lan-
guage models, 2023. URL https://arxiv.org/abs/2201.11903 .
Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V . Le. Simple synthetic data reduces
sycophancy in large language models, 2024. URL https://arxiv.org/abs/2308.03958 .
Nan Xu and Xuezhe Ma. Decoprompt : Decoding prompts reduces hallucinations when large
language models meet false premises, 2025. URL https://arxiv.org/abs/2411.07457 .
Xinyan Velocity Yu, Sewon Min, Luke Zettlemoyer, and Hannaneh Hajishirzi. Crepe: Open-
domain question answering with false presuppositions, 2022. URL https://arxiv.org/
abs/2211.17257 .
Hongbang Yuan, Pengfei Cao, Zhuoran Jin, Yubo Chen, Daojian Zeng, Kang Liu, and
Jun Zhao. Whispers that shake foundations: Analyzing and mitigating false premise
hallucinations in large language models, 2024. URL https://arxiv.org/abs/2402.19103 .
Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A. Smith. How language
model hallucinations can snowball, 2023a. URL https://arxiv.org/abs/2305.13534 .
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo
Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and
Shuming Shi. Siren’s song in the ai ocean: A survey on hallucination in large language
models, 2023b. URL https://arxiv.org/abs/2309.01219 .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao
Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P . Xing, Hao Zhang, Joseph E. Gonzalez,
and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL
https://arxiv.org/abs/2306.05685 .
Liwen Zheng, Chaozhuo Li, Xi Zhang, Yu-Ming Shang, Feiran Huang, and Haoran Jia.
Evidence retrieval is almost all you need for fact verification. In Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics:
ACL 2024 , pp. 9274–9281, Bangkok, Thailand, August 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.findings-acl.551. URL https://aclanthology.org/
2024.findings-acl.551/ .
13

Yanxu Zhu, Jinlin Xiao, Yuhang Wang, and Jitao Sang. Kg-fpq: Evaluating factuality
hallucination in llms with knowledge graph-based false premise questions, 2024. URL
https://arxiv.org/abs/2407.05868 .
14

A Appendix
A.1 Dataset Details
In KoPL (Zhu et al., 2024), each entity is linked to a specific concept, such as Leonardo
da Vinci being connected to the concept of an artist . The knowledge graph includes 794
distinct concepts, categorized into domains based on general knowledge, enabling domain-
based entity classification. For the art domain, the authors of (Zhu et al., 2024) manually
selected 33 relations, ensuring that each relation is relevant to its domain and informative,
avoiding ambiguity. For example, the relation artist is linked to the Art domain, while family
is more ambiguous and excluded. Table 3 shows the representative concepts, relations
and subjects in the art domain of KG-FPQ. The dataset comprises 4969 questions in the
discriminative task for the art domain, with each true premise question modified using the
following editing methods: Neighbor-Same-Concept (NSC), Neighbor-Different-Concept
(NDC), Not-Neighbor-Same-Concept (NNSC), Not-Neighbor-Different-Concept (NNDC),
Not-Neighbor-Same-Relation (NNSR), and Not-Neighbor-Different-Relation (NNDR).
Domain Concept e.g. Concept Qty Subject e.g. Subject Qty Relation e.g. Relation Qty
film Titanic cast member
Art television series 44 Modern Family 1754 composer 33
drama Hamlet narrative location
Table 3: Representative concepts, relations, and subjects in KG-FPQ art domain.
A.2 Case Study
We perform a case study demonstrating how our method enhances multi-hop question
performance. We use the with RAG method example here, where the edited object is not
directly connected to the subject within the KoPL knowledge graph.
Question :Is "The Dark Knight" the recipient of the 16th Screen Actors Guild
Awards?
Answer :Yes, "The Dark Knight" won the award for Outstanding Performance by a
Stunt Ensemble in a Motion Picture at the 16th Screen Actors Guild Awards.
Logical Form :
is a recipient of( 'The Dark Knight ', 16th Screen Actors Guild Awards)
Retrieved Graph Triple :
['The Dark Knight ','award received ','81st Academy Awards ']
After Detecting and Informing LLM of the Presence of a False Premise :
Corrected Answer :No, "The Dark Knight" was not the recipient of the 16th Screen
Actors Guild Awards. The awards ceremony took place in 2010, and "The Hurt Locker"
won the award for Outstanding Performance by a Cast in a Motion Picture.
A.3 Prompt Details
The following prompt is used to combine the information retrieved from the knowledge
graph G(context) and the query logical form L(q)(query) to form the input to the LLMs
discussed in §4.2.1.
Given the context below, does the following question contain a false premise?
Answer with ’Yes’ or ’No’ only. Note that the context is provided as valid facts
in a triple. Context: [context]. Query: [query].
15
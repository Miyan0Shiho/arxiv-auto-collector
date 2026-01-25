# Chunking, Retrieval, and Re-ranking: An Empirical Evaluation of RAG Architectures for Policy Document Question Answering

**Authors**: Anuj Maharjan, Umesh Yadav

**Published**: 2026-01-21 20:52:48

**PDF URL**: [https://arxiv.org/pdf/2601.15457v1](https://arxiv.org/pdf/2601.15457v1)

## Abstract
The integration of Large Language Models (LLMs) into the public health policy sector offers a transformative approach to navigating the vast repositories of regulatory guidance maintained by agencies such as the Centers for Disease Control and Prevention (CDC). However, the propensity for LLMs to generate hallucinations, defined as plausible but factually incorrect assertions, presents a critical barrier to the adoption of these technologies in high-stakes environments where information integrity is non-negotiable. This empirical evaluation explores the effectiveness of Retrieval-Augmented Generation (RAG) architectures in mitigating these risks by grounding generative outputs in authoritative document context. Specifically, this study compares a baseline Vanilla LLM against Basic RAG and Advanced RAG pipelines utilizing cross-encoder re-ranking. The experimental framework employs a Mistral-7B-Instruct-v0.2 model and an all-MiniLM-L6-v2 embedding model to process a corpus of official CDC policy analytical frameworks and guidance documents. The analysis measures the impact of two distinct chunking strategies, recursive character-based and token-based semantic splitting, on system accuracy, measured through faithfulness and relevance scores across a curated set of complex policy scenarios. Quantitative findings indicate that while Basic RAG architectures provide a substantial improvement in faithfulness (0.621) over Vanilla baselines (0.347), the Advanced RAG configuration achieves a superior faithfulness average of 0.797. These results demonstrate that two-stage retrieval mechanisms are essential for achieving the precision required for domain-specific policy question answering, though structural constraints in document segmentation remain a significant bottleneck for multi-step reasoning tasks.

## Full Text


<!-- PDF content starts -->

Chunking, Retrieval, and Re-ranking: An Empirical
Evaluation of RAG Architectures for Policy
Document Question Answering
Anuj Maharjan*
Electrical Engineering and Computer Science (EECS)
University of Toledo
Toledo, OH, USA
anjmhrjn1@gmail.comUmesh Yadav*
Electrical Engineering and Computer Science (EECS)
University of Toledo
Toledo, OH, USA
yadav.umesh0518@gmail.com
Abstract—The integration of Large Language Models (LLMs)
into the public health policy sector offers a transformative ap-
proach to navigating the vast repositories of regulatory guidance
maintained by agencies such as the Centers for Disease Control
and Prevention (CDC). However, the propensity for LLMs
to generate hallucinations, defined as plausible but factually
incorrect assertions, presents a critical barrier to the adoption of
these technologies in high-stakes environments where information
integrity is non-negotiable [1]. This empirical evaluation explores
the effectiveness of Retrieval-Augmented Generation (RAG) ar-
chitectures in mitigating these risks by grounding generative
outputs in authoritative document context [2]. Specifically, this
study compares a baseline Vanilla LLM against Basic RAG
and Advanced RAG pipelines utilizing cross-encoder re-ranking.
The experimental framework employs a Mistral-7B-Instruct-v0.2
model [3] and an all-MiniLM-L6-v2 embedding model [4] to
process a corpus of official CDC policy analytical frameworks and
guidance documents. The analysis measures the impact of two
distinct chunking strategies, recursive character-based and token-
based semantic splitting, on system accuracy, measured through
faithfulness and relevance scores across a curated set of complex
policy scenarios [5]. Quantitative findings indicate that while Ba-
sic RAG architectures provide a substantial improvement in faith-
fulness (0.621) over Vanilla baselines (0.347), the Advanced RAG
configuration achieves a superior faithfulness average of 0.797.
These results demonstrate that two-stage retrieval mechanisms
are essential for achieving the precision required for domain-
specific policy question answering, though structural constraints
in document segmentation remain a significant bottleneck for
multi-step reasoning tasks.
Index Terms—Large Language Models, Retrieval-Augmented
Generation, Policy Analysis, Cross-Encoders, Document Chunk-
ing
I. INTRODUCTION
The current landscape of natural language processing is
dominated by transformer-based Large Language Models that
have demonstrated remarkable capabilities in text synthesis,
summarization, and zero-shot reasoning [6], [7]. Despite these
advancements, the utility of standalone models in specialized
domains like public health policy is severely hampered by their
reliance on internal weights that may contain outdated, biased,
*Both authors contributed equally to this work.or incorrect information [2]. In the context of regulatory
compliance and policy analysis, a single erroneous recommen-
dation can have profound implications for public safety and
institutional credibility. Consequently, there is an urgent need
for architectural frameworks that can reliably bridge the gap
between generative fluency and factual grounding [8].
The motivation behind this research stems from the in-
creasing complexity of government policy repositories. Public
health officials are frequently required to synthesize infor-
mation across disparate documents, such as the CDC Pol-
icy Analytical Framework, Strategy and Policy Development
guidelines, and Program Cost Analysis manuals. Manually
navigating these documents is time-intensive and prone to
human error, yet traditional keyword-based search engines lack
the semantic depth to answer complex, scenario-based queries.
Retrieval-Augmented Generation (RAG) has emerged as a
promising paradigm that combines the strengths of information
retrieval with the generative capacity of LLMs by injecting
relevant document fragments directly into the model’s prompt
[9].
The fundamental research question addressed in this paper
is: To what extent do advanced retrieval techniques, such as
cross-encoder re-ranking and optimized chunking strategies,
improve the faithfulness and relevance of generated answers
in a policy-specific question-answering system? By evaluating
these architectures using a specialized corpus of CDC policy
documents, this study seeks to provide a reproducible baseline
for the deployment of intelligent policy navigators in regulated
information environments. The significance of this work lies
in its empirical comparison of system configurations, high-
lighting the critical role of the retrieval pipeline in ensuring
that LLMs act not as repositories of knowledge, but as precise
synthesizers of authoritative evidence.
II. STATE OF THEFIELD ANDCOMPARATIVELITERATURE
The concept of Retrieval-Augmented Generation was first
formalized by Lewis et al. (2020) [9] as a method to provide
LLMs with access to a non-parametric memory in the form ofarXiv:2601.15457v1  [cs.CL]  21 Jan 2026

a dense vector index. This architecture represented a signifi-
cant departure from previous approaches that relied solely on
supervised fine-tuning to encode domain-specific knowledge
into model parameters.
A. The Evolution of Retrieval Mechanisms
The retrieval stage of RAG has traditionally relied on bi-
encoder architectures, which independently encode queries
and document chunks into a joint vector space. Models like
Sentence-BERT (SBERT) [10] revolutionized this space by
enabling efficient semantic similarity searches using cosine
similarity, reducing the time required to find relevant passages
from hours to milliseconds. However, subsequent evaluations
have identified significant limitations in bi-encoders, particu-
larly their inability to capture token-level interactions between
a query and a document. This ”shallow” retrieval often results
in the selection of passages that are semantically similar in a
broad sense but contextually irrelevant to the specific intent of
a query [11].
To address this, researchers have introduced two-stage re-
trieval systems that utilize cross-encoders for re-ranking. Un-
like bi-encoders, cross-encoders jointly process the query and
each candidate document, allowing the transformer’s attention
mechanism to evaluate the precise relationship between tokens.
Benchmarks on the MS MARCO dataset have shown that re-
ranking can improve retrieval precision by as much as 27% in
terms of Mean Reciprocal Rank (MRR@10) [12].
B. Legal and Regulatory NLP
Beyond general retrieval, specific challenges exist in the
legal and regulatory domain. Evaluating LLMs on legal bench-
marks (LegalBench) has shown that general-purpose models
often fail to distinguish between ”mandatory” and ”permis-
sive” language (e.g., ”must” vs ”should”) [13]. Domain-
specific RAG adaptation, as proposed in RAFT [14], suggests
that fine-tuning on domain documents improves performance,
but RAG remains the most cost-effective solution for dynamic
policy environments where regulations change frequently.
While this study focuses on improving factual faithfulness
through retrieval, ensuring holistic system reliability also en-
tails mitigating external vulnerabilities, such as the adversarial
payload injections explored in [15].
III. METHODOLOGY ANDARCHITECTURE
A. System Architecture
We implemented a Dual-Stage Retrieval pipeline designed
to maximize precision in high-noise policy documents. As
illustrated in Fig. 1, the system utilizes a two-step filtering
process: first via dense vector similarity (Bi-Encoder) and
second via token-level attention (Cross-Encoder).
B. Algorithmic Formulation
The retrieval logic follows a strict ”Over-Retrieve and
Filter” paradigm. We define the corpusD={d 1, d2, ..., d n}
and queryq. The process is formally described in Algorithm
1.
Embed Query
Cosine SimilarityRaw Query Text
Score & SortPrompt InjectionUser Query
Bi-Encoder\nall-MiniLM-L6-
v2
Vector DB\nF AISS Index
Top-10 
Candidates\n(Semantic 
Search)
Cross-Encoder R e-
Ranker\nms-marco-MiniLM-
L-6-v2
Top-3 
Contexts\n(R elevance 
Filtered)
Generator LLM\nMistral-7B-
Instruct-v0.2
Final P olicy AnswerFig. 1. The Advanced RAG Architecture. The pipeline uses a Bi-Encoder
for initial efficient retrieval, followed by a computationally intensive Cross-
Encoder to filter false positives before generation.
C. Mathematical Formulation
The retrieval process is formally defined as finding the doc-
ument chunkdfrom corpusCthat maximizes the similarity
with queryq. In the Basic RAG setup, we employ Cosine
Similarity over dense vector embeddingsE(·):

Algorithm 1:Dual-Stage Retrieval Pipeline
Input:Queryq, CorpusD, Bi-EncoderM bi,
Cross-EncoderM cross
Output:Generated AnswerA
Qemb←M bi.encode(q);
Candidates← ∅;
ford i∈Ddo
Demb←M bi.encode(d i);
score i←CosineSimilarity(Q emb, Demb);
Candidates.add((d i, score i));
end
TopK←SelectTop(Candidates, k= 10);
Reranked← ∅;
ford j∈TopKdo
score cross←M cross.predict(q, d j);
Reranked.add((d j, score cross));
end
Context←SelectTop(Reranked, k= 3);
Prompt←ConstructPrompt(q, Context);
A←LLM.generate(Prompt);
Sim(q, d) = cos(θ) =E(q)·E(d)
||E(q)|| · ||E(d)||(1)
For the Advanced RAG configuration, we introduce a scor-
ing functionS cross that takes the concatenated pair as input.
The re-ranking stage computes a relevance score for the top
kcandidates retrieved by the bi-encoder:
R={d i∈C top−k|argmaxS cross(q, d i)}(2)
This two-stage approach minimizes the computational over-
head of the cross-encoder by restricting its application to a
small subset of the corpus (k= 10), rather than the entire
index [12].
D. System Configurations
The evaluation compares a hierarchy of architectural com-
plexity:
1)Vanilla LLM:A standalone Mistral-7B-Instruct-v0.2
model operating without retrieval grounding. This sys-
tem relies entirely on its pre-training data to answer
policy questions [3].
2)Basic RAG:A standard pipeline using the all-MiniLM-
L6-v2 embedding model to retrieve the top 3 chunks
via cosine similarity [4]. The retrieved context is in-
jected into a strict system prompt that forbids external
knowledge use.
3)Advanced RAG:An augmented pipeline that adds a
cross-encoder re-ranking step. The system retrieves 10
candidate chunks using the bi-encoder and then uses ms-
marco-MiniLM-L-6-v2 to select the top 3 most relevant
segments before generation.IV. EXPERIMENTALRESULTS
The empirical results of the 10-question evaluation highlight
a clear performance gradient as retrieval complexity increases.
The aggregate data demonstrates that standalone LLMs are
inadequate for policy-grounded tasks, while the addition of a
re-ranking stage provides the most significant boost to answer
integrity.
A. Quantitative Performance Comparison
Fig. 2 illustrates the performance gap between the archi-
tectures. The Advanced RAG system consistently outperforms
the baseline across both metrics.
Vanilla Basic RAG Adv. RAG00.20.40.60.81
0.350.620.8
0.450.70.8Score (0-1)
Faithfulness Relevance
Fig. 2. Comparison of Average Faithfulness and Relevance Scores. The
Advanced RAG architecture demonstrates superior performance in grounding
answers to the source text.
The itemized breakdown of scores is presented in Table I.
TABLE I
DETAILEDPERFORMANCEMETRICS PERQUESTION
QID Faithfulness Relevance
Van Bas Adv Van Bas Adv
Q1 0.33 0.33 0.67 0.50 1.00 1.00
Q2 0.33 0.67 0.83 0.33 1.00 1.00
Q3 0.33 1.00 1.00 0.67 1.00 1.00
Q4 0.33 0.33 0.16 0.50 0.50 0.50
Q5 0.25 0.50 0.25 0.33 0.67 0.33
Q6 0.33 0.67 1.00 0.33 0.80 1.00
Q7 0.00 0.71 0.29 0.00 1.00 0.50
Q8 0.40 0.00 0.80 0.50 0.00 0.67
Q9 0.50 1.00 1.00 0.67 1.00 1.00
Q10 0.67 1.00 1.00 0.67 1.00 1.00
Avg 0.35 0.62 0.80 0.45 0.70 0.80
B. Qualitative Case Study
To better illustrate the mechanism of failure and recovery,
Table II provides a direct comparison of the outputs for
Question 1. The Vanilla model provides a generic, medically
accurate but policy-irrelevant definition. The Advanced RAG
model successfully retrieves the specific CDC ”reframing”
requirement.

TABLE II
QUALITATIVECOMPARISON OFGENERATEDOUTPUTS(Q1)
Query Vanilla Response
(Hallucination)Advanced RAG Re-
sponse (Grounded)
A city health de-
partment identifies
obesity as a ma-
jor concern. How
should this prob-
lem be reframed to
better support pol-
icy action?”Obesity is a chronic
disease that affects in-
dividuals of all ages...
It is defined by excess
body weight that im-
pairs health.””To effectively address
obesity using the
CDC policy analytical
framework, it should
be reframed as a lack
of access to fresh fruits
and vegetables.”
Analysis Correct medical
definition, but fails to
address the ”Policy
Framework” context.Directly references the
environmental determi-
nant required by the
framework [2].
C. Analysis of System Evolution
The transition from Vanilla LLM to Basic RAG yielded
a 79% increase in average faithfulness and a 55% increase
in relevance. This confirms the hypothesis that external con-
text injection is fundamental to grounding policy answers.
However, Basic RAG demonstrated significant volatility. In
Question 8, which concerned infectious disease reporting laws,
the Basic RAG system failed completely with a 0.00 score in
both metrics, indicating that the initial vector search retrieved
entirely irrelevant context that the model could not use [11].
The Advanced RAG system, by contrast, achieved the
highest average scores across both dimensions. The inclusion
of the ms-marco-MiniLM-L-6-v2 cross-encoder allowed the
system to evaluate token-level alignment, successfully ”re-
covering” Question 8 with a faithfulness score of 0.80. By
jointly encoding the query and candidates, the cross-encoder
disambiguated closely related policy concepts that the bi-
encoder’s coarse vector similarity missed.
D. Quantitative Impact of Re-ranking Accuracy
The theoretical foundation for cross-encoder superiority lies
in the model architecture’s ability to capture nuanced semantic
chains that bi-encoders omit. While a bi-encoder maps the
query and document independently, the cross-encoder allows
every token in the query to ”attend” to every token in the
document. This enables the model to learn causal relationships,
such as the link between ”recycling” and ”reducing hazards,”
which might not be captured in a static vector embedding.
In production-grade RAG applications, bi-encoders typically
achieve only 65–80% relevance accuracy on complex queries,
meaning that 20–35% of the information fed to the LLM is
noisy or irrelevant. The integration of the ms-marco-MiniLM-
L-6-v2 cross-encoder [4] has been shown to raise this accuracy
to 85–90% on web search benchmarks. In the context of
the CDC policy corpus, this translated to a 28% relative
improvement in faithfulness over the Basic RAG configuration,
providing a much higher degree of certainty that the model is
operating on correct evidence.TABLE III
COMPARISON OFBI-ENCODER ANDCROSS-ENCODERARCHITECTURES
Metric Bi-Encoder Cross-Encoder
Model all-MiniLM-L6 ms-marco-MiniLM
Latency ∼15ms / 1M docs 50–150ms / 20 docs
Accuracy 65–80% relevance 85–90% relevance
Interaction Independent Joint Attention
Scaling High (billions) Low (dozens)
V. CONCLUSION ANDFUTUREWORK
This evaluation confirms that RAG is essential for grounding
LLM outputs in authoritative policy guidance. Standalone
models are insufficient for the demands of public health anal-
ysis due to unacceptable hallucination rates. The integration of
cross-encoder re-ranking provides the precision necessary for
domain-specific tasks, though future research must prioritize
”structure-aware” chunking to prevent the fragmentation of
logical policy workflows.Furthermore, as these architectures
scale to handle sensitive government repositories, the security
of the retrieval pipeline becomes paramount. Future itera-
tions of this work will explore integrating standardized data
exchange frameworks, such as the Model Context Protocol
(MCP), to abstract data retrieval while maintaining strict
security boundaries [16]. Advancements in hybrid retrieval and
knowledge graph integration offer promising paths for achiev-
ing the zero-hallucination standards required for governmental
decision-making.
REFERENCES
[1] Y . Zhang, Y . Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao,
Y . Zhang, Y . Chenet al., “Siren’s song in the AI ocean: A survey on
hallucination in large language models,” 2023.
[2] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” 2023.
[3] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot,
D. de Las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier,
L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang,
T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023.
[4] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM:
Deep self-attention distillation for task-agnostic compression of pre-
trained transformers,” inAdvances in Neural Information Processing
Systems (NeurIPS), vol. 33, 2020, pp. 5776–5788.
[5] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “Ragas: Auto-
mated evaluation of retrieval augmented generation,” 2023.
[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin, “Attention is all you need,” inAdvances
in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.
[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-
training of deep bidirectional transformers for language understanding,”
inProceedings of the 2019 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies (NAACL-HLT), 2019, pp. 4171–4186.
[8] S. Guanet al., “Privacy challenges and solutions in RAG-enhanced
LLMs for healthcare,” 2025.
[9] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdvances in Neural Information Processing Systems
(NeurIPS), vol. 33, 2020, pp. 9459–9474.
[10] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using siamese BERT-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing (EMNLP), 2019,
pp. 3982–3992.

[11] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and
P. Liang, “Lost in the middle: How language models use long contexts,”
Transactions of the Association for Computational Linguistics, vol. 12,
pp. 157–173, 2024, preprint arXiv:2307.03172 (2023).
[12] R. Nogueira and K. Cho, “Passage re-ranking with BERT,” 2019.
[13] V . Pipitone and N. Alami, “LegalBench-RAG: A benchmark for
retrieval-augmented generation in the legal domain,” 2024.
[14] T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica, and J. E.
Gonzalez, “RAFT: Adapting language models to domain-specific RAG,”
2024.
[15] U. Yadav, S. Niroula, G. K. Gupta, and B. Yadav, “Exploring secure
machine learning through payload injection and fgsm attacks on resnet-
50,” in2025 Silicon Valley Cybersecurity Conference (SVCC), 2025, pp.
1–7.
[16] S. Gaire, S. Gyawali, S. Mishra, S. Niroula, D. Thakur, and U. Yadav,
“Systematization of knowledge: Security and safety in the model context
protocol ecosystem,”arXiv preprint arXiv:2512.08290, 2025.
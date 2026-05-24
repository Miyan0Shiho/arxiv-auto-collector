# Fine-grained Claim-level RAG Benchmark for Law

**Authors**: Souvick Das, Sallam Abualhaija, Domenico Bianculli

**Published**: 2026-05-20 11:56:27

**PDF URL**: [https://arxiv.org/pdf/2605.21071v2](https://arxiv.org/pdf/2605.21071v2)

## Abstract
The rapid progress of large language models (LLMs) is shifting semantic search toward a question-answering paradigm, where users ask questions and LLMs generate responses. In high-stake domains such as law, retrieval-augmented generation (RAG) is commonly used to mitigate hallucinations in generated responses. Nonetheless, prior work shows that RAG systems, whether general-purpose or legal-specific, still hallucinate at varying rates, making fine-grained evaluation essential. Despite the need, existing evaluation frameworks for legal RAG systems lack the granularity required to provide detailed analysis of retrieval and generation performance separately. Moreover, current benchmarks are largely English-only and centered on legal expert queries, overlooking non-expert needs. We introduce ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and English, targets both experts and non-experts, and includes diverse question types reflecting realistic scenarios. We further apply a fine-grained evaluation framework of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation, and claim-level analysis in the legal domain.

## Full Text


<!-- PDF content starts -->

Fine-grained Claim-level RAG Benchmark for Law
Souvick Das∗
University of Luxembourg
souvick.das@uni.luSallam Abualhaija∗
University of Luxembourg
sallam.abualhaija@uni.lu
Domenico Bianculli
University of Luxembourg
domenico.bianculli@uni.lu
Abstract
The rapid progress of large language models (LLMs) is shifting semantic search
toward a question-answering paradigm, where users ask questions and LLMs
generate responses. In high-stake domains such as law, retrieval-augmented gener-
ation (RAG) is commonly used to mitigatehallucinationsin generated responses.
Nonetheless, prior work shows that RAG systems, whether general-purpose or
legal-specific, still hallucinate at varying rates, making fine-grained evaluation
essential. Despite the need, existing evaluation frameworks for legal RAG systems
lack the granularity required to provide detailed analysis of retrieval and generation
performance separately. Moreover, current benchmarks are largely English-only
and centered on legal expert queries, overlooking non-expert needs. We introduce
ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and
English, targets both experts and non-experts, and includes diverse question types
reflecting realistic scenarios. We further apply a fine-grained evaluation framework
of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation,
and claim-level analysis in the legal domain.
1 Introduction
The rapid progress of large language models (LLMs) has revolutionized information access, with
growing reliance on chatbots shifting semantic search toward a question-answering (QA) paradigm in
which users pose questions and LLMs generate responses [ 1]. This shift can be critical in high-stakes
domains such as law, where adoption of generative artificial intelligence (GenAI) among legal pro-
fessionals is growing at a rapid pace [ 8]. Retrieval-augmented generation (RAG) [ 24] has emerged
as a promising approach to ground LLM responses in authoritative external sources, enabling more
transparent, verifiable, and context-aware responses [ 22]. While LLMs offer unprecedented advan-
tages, their current use in the legal domain, even when augmented with RAG, remains problematic,
as they are prone to hallucinations (i.e., generating incorrect information [ 4]) and are often perceived
as more reliable than they actually are [28, 29].
Theoretical work shows that LLMs must hallucinate to a certain extent, irrespective of their architec-
ture, training data quality, or scale [20].
Empirical evidence reinforces this limitation in the legal domain: Dahl et al. [4]show that general-
purpose LLMs, when applied to legal queries, hallucinate in approximately 58% to 82% of cases.
Magesh et al. [28] report hallucination rates of 17% to 33% even in customized legal AI systems,
undermining claims of legal technology providers about legal research tools being substantially less
prone to hallucination. Real-world incidents further emphasize these findings, with documented
∗Equal contribution
Preprint.arXiv:2605.21071v2  [cs.CL]  21 May 2026

failures in the use of GenAI tools in legal practice. Such failures range from early incidents following
the introduction of ChatGPT [ 48] to more recent cases where AI-generated content was used for
drafting legal pleadings, resulting in court sanctions [10].
The prevalence of hallucinations in general-purpose as well as domain-specific LLMs, even when
augmented with RAG, highlights the necessity of dedicated benchmarks for evaluating such tools
in legal settings. This gap has attracted attention within the research community. Early legal
benchmarks established important baselines but were limited in scope and applicability. For example,
LEXTREME [ 34] focused on tasks such as classification and named entity recognition, while
LegalBench [ 15] evaluated legal reasoning capabilities of LLMs inspired by the Americal legal
reasoning. Although valuable, these benchmarks were not designed particularly to assess RAG
systems, which now dominate the landscape of semantic search and question answering.
Few benchmarks specific for evaluating legal RAG exist. For example, LegalBench-RAG [ 36] extends
LegalBench by tracing generated answers to their sources, but focuses on a narrow subset of legal
retrieval scenarios, primarily involving specific document types such as contracts and privacy policies.
The dataset introduced by Magesh et al. [28] to empirically evaluate legal tools is tightly coupled to
the U.S. legal system and relies on qualitative metrics that are time-consuming to compute manually.
More broadly, existing benchmarks for evaluating legal RAG systems lack the granularity required to
provide detailed analysis of retrieval and generation performance separately. Although some datasets
are available in other languages, e.g., Korean [ 35], most focus on English and are primarily designed
to assess tools intended for legal professionals, rather than systems aimed at supporting the general
public in accessing legal information.
To facilitate detailed assessment of RAG systems, Ferrara et al. [11] introduced the RAG Triad
framework, which explicitly links the user query, retrieved context, and generated response, allowing
structured evaluation along three dimensions: context relevance, groundedness, and answer relevance.
This framework has since inspired a variety of evaluation frameworks and has paved the way for
new fine-grained metrics in the literature (e.g., Li et al. [25], Ru et al. [40]). Many recent metrics
rely on claim-level analysis of generated responses or retrieved contexts, where a claim is defined
as a minimal, self-contained factual statement [ 31]. It is argued that claim-based analysis is more
effective in detecting hallucination [ 42], an ongoing challenge that is specially critical in legal
contexts. Nevertheless, the adoption of such detailed evaluation frameworks and metrics, as well as
an assessment of the effectiveness and applicability of claim-extraction techniques (e.g., Hu et al.
[17]) remain largely under-explored in the legal domain.
To address the above research gaps, this paper makes three key contributions:
1.We introduce a comprehensive dataset ClaimRAG-LAW, standing for “Fine-grained Claim-
level RAG Benchmark for Law”, which enables fine-grained evaluation of RAG systems in
legal settings. The dataset and its generation code are available on Hugging Face [ 6] and
Zenodo [ 5], respectively. The dataset was designed to support three goals:diversein terms of
questions categories and targeted users,representativecovering different natural languages
and source regulations, andfine-grainedin terms of providing details to support effective
evaluation. To achieve these goals, ClaimRAG-LAW contains different question categories
(adopted from Magesh et al. [28]) that are directed specifically at evaluating hallucination
in RAG systems. It accounts for different users including both legal professionals and lay
users seeking access to legal information. Finally, all QA pairs are sourced from the general
data protection regulation (GDPR) [ 44] in English and a national civil law [ 13] in French
(hereafter referred to asCIVIL). ClaimRAG-LAW contains two parts:
•The first part consists of an overall of 317 QA pairs and is intended to evaluate the
retrieval and generation performance separately in legal RAG systems.
•The second part consists of 968 claims extracted from the two sources mentioned above,
all of which were manually validated. This part is intended to identify hallucination as
well as assess claim extraction and checking methods.
2.We assess several fine-grained metrics originally introduced by Ru et al. [40] for general
RAG systems, to investigate their applicability in the legal domain.
3.We conduct an extensive empirical evaluation of state-of-the-art RAG systems on the
proposed benchmark. Our results systematically uncover domain-specific limitations and
failure modes, highlighting gaps in current RAG systems when applied to the legal domain.
2

We further assess the effectiveness of claim-level analysis in legal contexts, revealing
limitations, particularly in detecting contradicted claims.
The remainder of the paper is structured as follows. Section 2 surveys related work. Section 3
details the construction of ClaimRAG-LAW. Section 4 reports the results of our empirical evaluation.
Section 5 concludes the paper.
2 State of the art
This section reviews the state of the art focusing on two research directions: (1) RAG benchmarks
in the legal domain, and (2) claim-level evaluation, with a specific focus on hallucination detection.
We identify two key research gaps that motivate our work: (1) RAG benchmarking in the legal
domain lacks adequate, comprehensive evaluation compared to general domains; (2) fine-grained,
claim-level evaluation has been widely explored in the hallucination-relevant literature, yet with
limited adaptation to the legal domain.
2.1 RAG Benchmarks in Legal Domain
LexGLUE [ 3] synthesized seven existing legal datasets, aiming to standardize the evaluation of
legal natural language processing (NLP) models, specifically for the American and European laws.
LEXTREME [ 34] extended this effort to 11 datasets across 24 languages. These benchmarks, similar
to earlier ones, were designed around pre-trained language models such as BERT [ 7], focusing on
classification and prediction tasks, such as document classification, unfair clause detection, and
named entity recognition) rather than text generation. More recently, LegalBench [ 15] introduced
162 tasks developed with legal professionals using the IRAC framework [ 2], primarily intended to
evaluate the parametric legal reasoning of LLMs. However, given the growing use of RAG and
the risk of LLM hallucinations in the legal domain [ 4], these benchmarks are not well suited for
evaluating retrieval-grounded generation.
Several benchmarks evaluate retrieval in legal RAG systems, but have key limitations. CLERC [ 16]
provides a large -scale dataset for case law retrieval. Building on LegalBench [ 15], LegalBench-
RAG [ 36] measures the retrieval precision for contract -related questions. Both, however, primarily
assess the retrieval component and do not assess the end-to-end generation.
Extending beyond retrieval, LLeQA [ 27] introduces an expert-annotated, French-language, long-form
statutory QA benchmark grounded in Belgian codes of law, with questions posed by Belgian citizens
and annotated by legal experts, enabling evaluation of retrieve-then-read RAG (i.e., systems that first
retrieve relevant documents and then generate answers grounded in those documents) Nevertheless,
LLeQA is limited to a single jurisdiction (Belgian law) and language (French), it relies on holistic
automatic metrics, and it lacks fine-grained evaluation of generated answers.
Other benchmarks have also been introduced but remain with similarly limitations. LexRAG [ 25]
evaluates legal consultation conversations in Chinese law, CBR-RAG [ 49] focuses on case-based
retrieval in Australian law, and KOBLEX [ 23] addresses provision-grounded, multi-hop QA in
Korean law. However, each is limited to a single legal system, restricting the cross-jurisdictional
generalizations.
While LLeQA incorporates citizen-posed questions, and thus partially reflects non-expert information
needs, no existing benchmark systematically evaluates RAG performance for non-expert users across
multiple legal systems and languages. This dimension remains largely underexplored.
General -domain RAG evaluation has advanced through frameworks such as RAGAS [ 9] and
RAGChecker [ 40]. In contrast, the legal domain has only recently started to develop benchmarks for
retrieval-grounded generation.
2.2 Claim-Level Evaluation
The shift toward generative AI in law requires metrics that move beyond holistic accuracy and lexical
overlap. Traditional measures like ROUGE and BLEU—once standard for summarization—are
increasingly inadequate for assessing factual correctness, especially in RAG [ 40]. To address this,
frameworks such as TruLens [ 11] introduced the RAG Triad, which explicitly evaluates context
3

relevance, groundedness, and answer relevance by structurally linking these components. While
influential in general RAG evaluation, such frameworks have seen limited adoption in the legal
domain. As observed in the LLeQA benchmark [ 27], a generated legal answer may share high
semantic similarity with a ground truth while containing subtle but critical factual errors— a form
of hallucination with potentially serious consequences in legal contexts, which makes fine-grained,
claim-level evaluation essential.
In response to this need, a range of evaluation frameworks has been introduced for general-domain
RAG. FActScore [ 33] decomposes long-form responses into atomic facts and measures the pro-
portion supported by an external knowledge source (e.g., Wikipedia), capturing correct versus
hallucinated content that might be otherwise missed by holistic metrics. RefChecker [ 17] extracts
subject–predicate–object triplets and uses entailment, contradiction, and neutral relations with re-
spect to a reference context to detect hallucinated claims. FENICE [ 42] combines natural language
inference (NLI) with claim extraction to evaluate generated text at multiple levels of granularity,
improving interpretability by locating the precise evidence supporting each claim. SAFE [ 47] uses an
LLM to break long-form responses into individual facts and verify each by querying Google Search
and checking for supporting evidence. Claimify [ 32] addresses the risk that incomplete or distorted
claims may produce misleading hallucination judgments by evaluating claim-extraction methods in
terms of entailment, coverage, and decontextualization.
Despite substantial research on claim-level analysis, recent hallucination-detection frameworks [ 30,
31,46] suggest that hallucination detection remains a fundamental challenge, particularly in legal
contexts: general-purpose LLMs are prone to hallucinating in response to legal queries [ 4], and even
specialized legal RAG systems remain prone to hallucination [ 28]. A recent empirical study reports
that customized RAG-based legal research tools hallucinate between 17% and 33% of the time [ 28].
The study found that the predominant failure mode is misgrounding, whereby a RAG system cites
genuine legal authorities that do not in fact support the generated response.
Automated holistic metrics would likely overlook such failures, making claim-level hallucination
detection especially important in legal settings. Yet methods for claim-level evaluation in legal texts
remain limited. A recent work in this direction is RePASs (Regulatory Passage Answer Stability
Score) [ 12], which introduces an NLI-based metric for sentence-level evaluation of contradiction and
obligation coverage in regulatory texts. However, its scope is restricted to a narrow, monolingual
subdomain (Gulf financial regulations) leaving a gap in fine-grained, claim-level assessment for
broader multilingual statutory law.
3 ClaimRAG-Law
We presentClaimRAG-LAW, a comprehensive benchmark designed to evaluate RAG systems in the
legal domain. ClaimRAG-LAW consists of two complementary datasets: one designed to enable
fine-grained evaluation of legal RAG systems, and another specifically to support the evaluation of
claim checking methods.
3.1 Desiderata
We defined the following desiderata for ClaimRAG-LAW, guided both by the gaps in existing
literature and the analytical goals of this work, as highlighted in Section 1.
(1)Diverse.RAG might be used by various users seeking legal guidance, whose information needs
and levels of expertise can vary. For example, a user might ask the system to clarify a premise that
is initially false, while another may be looking for quick legal references. To reflect this diversity,
ClaimRAG-LAW shall include different question categories covering multiple personas.
(2)Representative.Accurately assessing the ability of RAG systems to produce correct and legally-
grounded responses and to distinguish such groundedness from their reliance on self-knowledge
requires evaluation across multiple languages and jurisdictions. To address this, ClaimRAG-LAW
shall include QA pairs drawn from at least two legal sources at different jurisdictional levels.
(3)Fine-grained.Much of the existing non-legal RAG literature relies on claim-level assessment to
better asses RAG systems and measure hallucination through checking the logical relation of claims
to the original context, often assuming that both claim extraction and checking tasks are sufficiently
4

accurate. To enable fine-grained analysis at the claim level in the legal domain, ClaimRAG-LAW
shall provide manually validated claims to facilitate more fine-grained analyses.
3.2 Addressing the Desiderata
Below, we explain how ClaimRAG-LAW addresses the above-outlined desiderata.
Diverse Question Categories and Personas.To address desideratum 1, ClaimRAG-LAW includes
QA pairs from multiple categories covering different personas. For constructing ClaimRAG-LAW,
we define the following question categories, inspired by Magesh et al. [28]:
(1)General legal research questionsrepresent a typical use case for legal RAG systems, where
users ask general questions about, e.g., common-law doctrines, specific holdings of court cases, or
interpretations of regulations.
(2)Factual recall questionstarget verifiable details that require minimal legal interpretation, such as
citations, effective dates of legislation, or specific entities mentioned in a clause.
(3)False premise questionsbegin from an incorrect assumption about a legal fact and therefore
require the system to identify and correct the erroneous premise.
(4)Jurisdiction or time-specific questionscapture a common legal challenge, as regulations change
over time and vary across jurisdictions.
Additionally, we account for a diverse set of personas, including legal experts, civil officers with
some legal knowledge, and laypeople with limited domain expertise and therefore limited ability to
verify the output.
Representative Source Documents.To meet desideratum 2, we selected the following regulations:
(1)The General Data Protection Regulation (GDPR) [ 44]:The de facto regulation for privacy and
data protection in Europe (English Version).
(2)National Civil Law (CIVIL) [ 13]:A foundational civil law text representing national-level
legislation (French Version; reference omitted due to double-blind review).
The rationale behind selecting these sources is to capture variation in jurisdictional scope (cross-
national regulatory frameworks and national civil law) and language (English and French), reflecting
the diversity in real-world scenarios.
Fine-grained Claim-Level Analysis.To address desideratum 3, we automatically extracted claims
from roughly half of the QA pairs in ClaimRAG-LAW and had a human expert validate both the
claims and their logical relations to the original context.
3.3 The dataset
To create ClaimRAG-LAW, we followed a two-step methodology: first, QA pairs were generated
automatically and then manually validated by a legal expert. Further details on the automated creation
and manual validation are provided in Appendix A. Table 1 reports the statistics for the final dataset,
with a total of 317 legal QA pairs sourced from GDPR (English) and CIVIL (French) regulations.
Additionally, ClaimRAG-LAW contains a total of 968 claims, of which 86.6% (838/968) were
deemed correct by the expert.
4 Benchmarking RAG Systems in Legal Domain
4.1 Research Questions
RQ1. How well do existing RAG systems perform in legal settings?This RQ aims to provide a
fine-grainedassessment of RAG systems in legal settings. In particular, we assess both the overall
performance of RAG systems on our curated dataset, as well as the performance of the retrieval
and generator components separately. To do this, we selectedRAGChecker[ 40], a fine-grained
RAG evaluation framework, motivated by the following reasons. First, RAGChecker has proven its
effectiveness in providing a detailed view of the performance of RAG systems. Second, Ru et al.
[40] report that the metrics they included in RAGChecker generally exhibit a higher correlation
(both Pearson and Spearman) with human judgments (in terms of correctness) compared to other
5

Table 1: Statistics for ClaimRAG-LAW Dataset
Persona Question Category Claims
P
QAP1 P2 P3 CG1CG2CG3CG4P
AP
CValid ϵ η ζ
GDPR (EN)149 20 77 52 131 9 8 1 63 520 453 295 151 7
CIVIL (FR)168 35 29 104 137 24 5 2 89 448 385 284 77 24
Total 317 55 106 156 268 33 13 3 152 968 838 579 228 31
1P
QArefers to the number of QA pairs. P 1, P2, P3refer to the personas: citizen, civil officer, and legal expert,
respectively. CG 1, CG 2, CG 3, CG 4refer to the question categories: general legal research, factual recall,
false premise, jurisdiction/time-specific, respectively.
2P
A: the number of generated answers used to generate the claims, andP
C: the number of claims. Valid
indicates the number of claims deemed correct by the expert, and ϵ,η,ζrefer to entailment, neutral, and
contradiction, respectively.
frameworks such as RAGAs [ 9]. This alignment with human intuition is particularly critical in the
legal domain, where evaluation nuances are often not fully captured by coarse-grained metrics.
RQ2. How accurate are automated claim extraction and verification methods in the legal
domain?
Since claims are used as a means for enabling fine-grained evaluation of RAG systems, RQ2 aims
to provide insights on the reliability of claim-level analyses in the legal domain. Specifically, we
investigate the accuracy of RefChecker [ 17] in extracting legal claims from the generated responses
and verifying their entailment relationship with a reference text. We focus on RefChecker because it
is the claim-level analysis engine of RAGChecker (used in RQ1).
4.2 Accuracy of RAG Systems in Legal Settings (RQ1)
To answer RQ1, we largely followed the evaluation settings of RAGChecker [ 40], as detailed below.
Evaluated RAG Systems.We evaluated a total of eight RAG systems, each formed by pairing a
retrieval strategy with an LLM. For retrieval, we used the same retrievers as Ru et al. [40]:BM25[ 39],
a standard sparse retriever, andE5-Mistral-7B-Instruct[ 45], a state-of-the-art LLM-based dense
retriever.
For answer generation, we selected three LLMs used by Ru et al. [40]:GPT-4o[18]
from OpenAI,Llama-3.1-8B-Instruct[14] from Meta, andMixtral-8x7B[19] from Mistral AI.
However, we excludedLlama-3.1-70B-Instructdue to its substantial computational cost. In addition,
we experimented with GPT5 [43]
to examine the effectiveness of a newer LLM generation of the same model in the legal domain.
Implementation Details.
Following Ru et al. [40], both retrievers, BM25 and E5-Mistral-7B-Instruct, are backed by an
OpenSearch ( https://opensearch.org/ ) index, using the E5-Mistral-7B-Instruct tokenizer for
chunking. We segmented documents into fixed-size chunks of 600 tokens with an overlap of 120
tokens (ratio of 0.2) between adjacent chunks. We opted for fixed-size rather than article-level
chunking strategy because legal documents, such as GDPR and CIVIL, can exhibit high variability in
article length, which would introduce inconsistent retrieval granularity. We chose 600-token chunks
over smaller sizes (150 and 300 tokens) evaluated in Ru et al. [40], motivated by the consistent
upward trend in their ablation study, where increasing chunk size improves claim recall and F1. We
setk= 10 , yielding 6000 tokens of retrieved context; combined with 2048 tokens reserved for the
system prompt, question, and generated response, the total remains within the 8192-token context
window of Llama-3.1-8B-Instruct, the most constrained generator in our setup. For generation, we
adopted the default prompt template from Ru et al. [40]. Each model is evaluated with a temperature
of 0.0 and a single run.
All experiments are conducted on a desktop machine equipped with a single NVIDIA RTX 3090
GPU (24 GB), an AMD Ryzen 9 5900X 12-core CPU, and64 GBof RAM.
6

RAGChecker Input Construction.RAGChecker expects as input a tuple consisting of a question,
retrieved chunk, generated response, and ground-truth answer. In ClaimRAG-LAW, the question and
ground-truth answer are provided, while the chunk and response are produced by running the different
RAG systems under evaluation. Specifically, for each RAG system, we provided the question together
with a legal document (i.e., GDPR or CIVIL). The system then retrieves the top-k most relevant
chunks, and generates a response to the question.
After building the tuples for each question-answer pair, we provided them as input to RAGChecker
for a fine-grained evaluation.
Metrics.We evaluate the RAG systems using the fine-grained metrics proposed by Ru et al. [40],
excludinghallucination(addressed in RQ2). Below, we summarize these metrics and refer the reader
to Ru et al. [40] for their full mathematical definitions.
These metrics rely on claim extraction and a logical analysis of each claim’s entailment with respect
to a reference text.
Overall metricsinclude (i)precision (P), the proportion of claims in model responses that are entailed
by ground-truth answers; (ii)recall (R), the proportion of claims in the ground-truth answers that are
correctly identified in the model responses; andF1 score, the harmonic mean between P and R.
Retriever metricsinclude (i)claim recall (CR), the proportion of claims in the ground-truth answer
that are entailed by the retrieved chunks;
(ii)context precision (CP), the proportion of retrieved chunks that entail at least one ground-truth
claim.
Generator metricsinclude (i)faithfulness (FT), the proportion of response claims entailed by the
retrieved chunks (regardless of correctness); (ii)self knowledge (SK), the proportion of correct
response claims not entailed by the retrieved context;
(iii)context utilization (CU), the ratio of correct response claims supported by the retrieved chunks to
the total relevant ground-truth claims entailed by the retrieved chunks;
(iv)hallucination (HL), the proportion of incorrect claims in the generated response (i.e., claims not
entailed by the ground-truth answer and also not entailed by any of the retrieved chunks); (v)relevant
noise sensitivity (NS r), the proportion of incorrect response claims entailed by relevant chunks; (vi)
irrelevant noise sensitivity (NS ir), the proportion of incorrect response claims entailed by irrelevant
chunks. A chunk is deemedrelevantif it entails any claim in the ground-truth answer, andirrelevant
otherwise.
Results.
Table 2 shows the results for RAG systems on ClaimRAG-LAW, averaged across question-answer
pairs. For reference, we also include (at the bottom of the table) the best reported RAG system by Ru
et al. [40], evaluated on multiple non-legal datasets.
The table shows that the performance of all RAG systems (except the first two) is generally better
for FR than EN in terms of precision, recall, and F1 score (P, R, and F1 in the table). In the legal
domain, precision outweighs recall, since introducing false information can be harmful, particularly
when many users lack the expertise to verify the generated content. In contrast, missing information
can often be mitigated through further search. Following this, the best performing system in our
experiments is E5 + GPT4 (which marginally outperforms E5 + GPT5), confirming the results
in Ru et al. [40]. While BM25 + GPT5 yields the best overall recall, its precision is substantially
lower, in particular for EN. We also remark that the overall accuracy results on ClaimRAG-LAW are
unexpectedly better than those reported in the original work of Ru et al. [40]. This could be possibly
due to the huge gap in size (394 questions sampled from two documents in ClaimRAG-LAW vs.
4162 ones from over 1 million documents in [40]).
Recall that the overall metrics discussed above compare the model’s final response against a ground-
truth answer, without considering intermediate outputs that may influence end-to-end performance.
This is reflected in the retrieval metrics, which consistently show that BM25 outperforms E5 in
both claim recall (CR) and context precision (CP) for EN and FR. This finding suggests thatdense
retrievers may excel in retrieving semantically related text, but not necessarily identify legally
relevant contexts. In contrast, BM25 appears to align queries more consistently with relevant chunks,
7

Table 2: Performance of baseline RAG systems evaluated usingRAGCheckeron two legal datasets.
Arrows indicate preferred direction (↑higher is better,↓lower is better).
Overall Retriever Generator
RAG System Lang†P↑R↑F1↑ CR↑CP↑ CU↑NSr↓NSir↓HL↓SK↓ FT↑
BM25 + Llama3-8B EN 45.0 62.1 48.2 82.976.969.0 48.5 2.0 4.6 2.193.3
BM25 + Llama3-8B FR 43.6 45.5 38.3 65.1 28.2 46.1 24.8 13.7 17.9 5.9 76.1
BM25 + Mixtral-8x7B EN 49.1 68.6 52.2 82.7 75.4 75.9 45.3 2.2 3.4 3.4 93.1
BM25 + Mixtral-8x7B FR 48.2 52.9 44.8 64.7 28.3 54.1 20.9 14.5 16.4 4.2 79.5
BM25 + GPT-4 EN 54.3 77.760.982.2 76.8 83.1 40.6 2.3 2.9 4.7 92.4
BM25 + GPT-4 FR 65.6 82.770.390.534.685.4 27.22.54.1 1.5 93.9
BM25 + GPT-5 EN 49.678.256.886.170.386.647.51.4 1.5 1.690.9
BM25 + GPT-5 FR 66.484.269.990.831.886.928.7 3.71.2 0.7 98.1
E5 + Llama3-8B EN 47.8 61.4 50.0 65.7 61.9 71.6 31.1 11.6 9.5 6.4 84.1
E5 + Llama3-8B FR 52.6 64.5 52.8 80.0 29.7 70.2 23.7 12.5 11.2 3.5 85.4
E5 + Mixtral-8x7B EN 49.0 56.0 47.9 66.6 63.1 72.5 36.1 10.2 4.7 3.4 91.9
E5 + Mixtral-8x7B FR 64.5 66.0 61.4 80.6 29.0 73.9 21.1 9.7 4.7 1.1 94.2
E5 + GPT-4 EN59.863.2 58.3 65.4 61.7 74.5 28.8 8.4 3.0 7.9 89.1
E5 + GPT-4 FR69.766.6 64.9 80.3 28.5 74.718.68.5 3.3 2.7 94.0
E5 + GPT-5 EN 59.5 69.0 60.7 65.4 53.7 80.626.28.6 5.6 8.6 85.7
E5 + GPT-5 FR 64.5 72.5 65.0 79.3 23.8 77.8 20.2 10.3 4.9 3.7 91.4
Ru et al. [40]‡62.0 53.0 52.7 83.5 61.8 60.4 28.9 3.5 5.7 1.4 92.9
†EN = GDPR and FR = CIVIL
‡best reported results by Ru et al. [40] on generic domain using E5 + GPT-4, obtained using datasets covering only English
language queries and corpora.
explaining why it is still widely use in information retrieval, despite being an old, keyword-based
method [ 37]. We observe that CP for CIVIL is significantly low, compared to the high CR for
both GDPR and CIVIL. Besides language-specific differences, this could be attributed to the short
articles in CIVIL, compared to those in GDPR. A single chunk might easily span multiple articles,
introducing thereby noise. This result highlights that, besides k(the number of retrieved chunks),the
legal adequacy of the retrieved content is a key determinant of retrieval performance.
The generation component’s ability to identify relevant information in retrieved chunks is shown by
the context utilization (CU) and faithfulness (FT) results, which are relatively high for both EN and
FR, with CU>80% andFT>90% for GPT4 and GPT5 (both paired with BM25). Other models
exhibit weaker and less consistent performance. For example, for FR, Mixtral-8x7B paired with
BM25 achieves substantially lower FT (79.5%) than with E5 (94.2%), while it seems less affected for
EN. Similarly, E5 + Llama3-8B improves for FR (from 76.1% to 85.4%), but drops notably on EN
(from 93.3% to 84.1%). CU results generally improve for these two models paired with E5, except
for a small drop for Mistral-8x7b when processing EN.
Self knowledge (SK) and Hallucination (HL) results provide a complementary view of the observa-
tions above. In terms of SK, GPT-5 is in the lead, outperforming GPT-4 by a margin (both paired
with BM25). It also has the lowest HL, indicating a stronger reliance on knowledge retrieved from
the provided contexts (CU). However, when generators depend on knowledge outside the retrieved
contexts (i.e., at lower CU values), they tend to naturally leverage their internal (SK), which often
results in higher HL rates. For instance, BM25 paired with Llama3-8B and Mixtral-8x7B show high
HL (17.9 and 16.4, respectively), particularly for FR where CU values are low. We also note that the
HL rates for these models are substantially higher in FR than in EN, consistent with their lower CP
and FT values, suggesting a greater tendency for hallucination when retrieval quality is poor.
Finally, the table shows that incorrect claims introduced by the LLMs predominantly originate from
partially relevant chunks (mixing relevant and irrelevant information), rather than from fully irrelevant
chunks with no relevant content (NS ris consistently much higher than NS iracross all RAG systems).
4.3 Accuracy of Claim Extraction and Verification in Legal Settings (RQ2)
To better understand the reliability of claim analysis, we assess in RQ2
8

Table 3: Claim verification against expert labels. P: Precision (%); R.: Recall (%); F1: F1-score (%).
GDPR (EN) CIVIL (FR)
P
CP R F1P
CP R F1
ϵ295 84.6 73.4 78.6 284 89.3 82.4 85.7
η151 68.1 72.2 70.1 77 53.3 62.3 57.5
ζ7 2.6 14.3 4.4 24 24.2 33.3 28.1
P
Cis the number of claims per class in ClaimRAG-LAW according to the expert’s manual analysis;
ϵ,η,ζrefer to entailment, neutral, and contradiction, respectively.
RefChecker [ 17] on ClaimRAG-LAW. Consistent with its original implementation, we use GPT-4-
0613 as the enabler LLM.
The answer to RQ2 is structured in two parts: (i) we first assess whether RefChecker can correctly
identify claims from generated responses and then (ii) we assess whether it can correctly determine
the entailment relationship between a claim and its reference text.
Evaluation Procedure.As part of constructing our dataset (ClaimRAG-LAW), we used RefChecker
to extract the claims from 152 answers. As a result we obtained a total of 968 claims which were
then manually validated by the legal expert (see Appendix A for more details on the dataset curation
process). Each claim was labeled as valid or invalid by the expert. Following this, we compute the
Extraction Accuracyfor part (i) as the ratio of valid claims to the total number extracted claims. While
we acknowledge that accuracy in this case primarily captures precision (i.e., it considers only the
claims that were produced, not those that were missed), we argue that this metric remains informative
and consistent with the goal of this RQ, which is to assess the reliability of claim analysis as an
intermediate step for enabling fine-grained evaluation.
RefChecker further
assigns the labelentailmentindicating whether the claim extracted from the answer is entailed in the
retrieved contexts,contradictionwhen the claim contradicts the contexts, andneutralotherwise.
In part (ii), We focus on claim verification by providing RefChecker with the expert-validated claims
(453 valid claims for GDPR and 385 for CIVIL) and the retrieved contexts, bypassing thereby the
extraction module and ensuring that all results are attributed solely to the verification module.
Results.In terms of claims extraction, RefChecker yields an accuracy of 98% (452/461) for GDPR
(EN) and 98.7% (385/390) for CIVIL (FR). These very high results confirm thatRefChecker can
accurately decomposes legal text into valid claims across both English and French.
Table 3 shows the results of claim verification labels produced by RefChecker evaluated against the
manually provided labels by the legal expert.
The results showa clear weakness in identifying contradictory claims, with average F1 of 4.4% on
GDPR and 28.1% on CIVIL. The low precision and recall values (especially for GDPR) suggest that
RefChecker introduces false contradictions while missing many genuine ones, consequently affecting
the overall evaluation of RAG systems. Although thecontradictionclass is under-represented in our
dataset, we expect a more reliable performance given the absence of training.These results highlight
the need for further research on claim-level analysis within the legal domain.
4.4 Threats to Validity and Limitations
Our findings provide a fine-grained analysis of RAG systems in the legal domain; however, our study
is subject to several validity considerations and limitations.
Threats to Validity.Some question categories like false-premise are underrepresented (5.4% in
GDPR; 3.0% in CIVIL), impacting the conclusions for these nuanced categories.
All QA pairs were generated using GPT-4. which is also evaluated in the RAG systems; this may
introduce bias and give an advantage to GPT models.
Manual validation was conducted by a single legal expert. Quality was ensured through a structured
pilot phase, explicit annotation guidelines, and iterative feedback sessions; however, the absence of
9

multiple independent experts precludes the computation of formal inter-annotator agreement metrics
(e.g., Cohen’sκ).
Finally, the expert validated each extracted claim but did not identify missed claims in the response;
thus, extraction accuracy reflects only precision and provides a partial view.
Limitations.As a synthetic resource, ClaimRAG-LAW may not fully capture the complexity of
real-world legal queries, potentially limiting generalization. While useful for benchmarking, the
dataset is not intended for providing legal advice without expert validation.
5 Conclusion
This paper has introduced ClaimRAG-LAW, a multilingual, multi-jurisdictional benchmark for
fine-grained evaluation of RAG systems in the legal domain. The benchmark comprises 317 expert-
validated QA pairs across diverse question categories and user personas, along with 968 manually
validated claims. Leveraging ClaimRAG-LAW, we have benchmarked eight state-of-the-art RAG
systems, revealing consistent, domain-specific failure modes in both retrieval and generation that ex-
isting legal RAG benchmarks have not systematically uncovered. Furthermore, we have demonstrated
that the existing claim-level evaluation frameworks, while effective in general-domain, exhibit critical
reliability limitations in legal context, particularly in detecting contradicted claims. In future work,
we plan to conduct a more in-depth analysis of hallucination of RAG systems in the legal settings,
extend the ClaimRAG-LAW with additional questions for under-represented question categories, and
enrich the dataset with fully expert-authored QA pairs and claims.
References
[1]J. Alammar and M. Grootendorst.Hands-on large language models: language understanding
and generation. " O’Reilly Media, Inc.", 2024.
[2]K. Burton. "think like a lawyer" using a legal reasoning grid and criterion-referenced assessment
rubric on irac (issue, rule, application, conclusion).Journal of Learning Design, 10(2):57–68,
2017. URLhttps://doi.org/10.5204/JLD.V10I2.229.
[3]I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos, D. Katz, and N. Aletras.
LexGLUE: A benchmark dataset for legal language understanding in english. InProceedings of
the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 4310–4330, 2022. URL https://doi.org/10.18653/v1/2022.acl-long.
297.
[4]M. Dahl, V . Magesh, M. Suzgun, and D. E. Ho. Large legal fictions: Profiling legal hallucinations
in large language models.Journal of Legal Analysis, 16(1):64–93, 2024. URLhttps://doi.
org/10.1093/jla/laae003.
[5]S. Das, S. Abualhaija, and D. Bianculli. LegalRAG QA Generator. https://doi.org/10.
5281/zenodo.20024153, 2026.
[6]S. Das, S. Abualhaija, and D. Bianculli. ClaimRAG-LAW Dataset. https://huggingface.
co/datasets/SNTSVV/ClaimRAG-LAW, 2026.
[7]J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirec-
tional transformers for language understanding. InProceedings of the 2019 conference
of the North American chapter of the association for computational linguistics: human
language technologies, volume 1 (long and short papers), pages 4171–4186, 2019. URL
https://doi.org/10.18653/v1/N19-1423.
[8]B. Edwards. Number of legal professionals using Gen AI jumps sharply over past year, study
shows. number-of-legal-professionals-using-gen-ai, April 17 2025. Accessed: 2026-01-04.
[9]S. Es, J. James, L. E. Anke, and S. Schockaert. RAGAs: Automated evaluation of retrieval
augmented generation. InProceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demonstrations, pages 150–158, 2024.
URLhttps://doi.org/10.18653/v1/2024.eacl-demo.16.
[10] Esquire Deposition Solutions. Federal court turns up the heat on attorneys using ChatGPT for
research. federal-court-turns-up-the-heat-on-attorneys, August 13 2025. Accessed: 2026-01-04.
10

[11] J. Ferrara, Ethan-Tonic, and O. M. Ozturk. The RAG Triad. https://www.trulens.org/
getting_started/core_concepts/rag_triad/, 2024. Accessed: 2026-04-28.
[12] T. Gokhan, K. Wang, I. Gurevych, and T. Briscoe. RIRAG: Regulatory information retrieval
and answer generation.arXiv preprint arXiv:2409.05677, 2024. URL https://doi.org/10.
48550/arXiv.2409.05677.
[13] Grand-Duché de Luxembourg. National civil code, 1804. URL https://legilux.public.
lu/.
[14] A. Grattafiori, A. Dubey, A. Jauhri, et al. The Llama 3 herd of models.arXiv preprint
arXiv:2407.21783, 2024. URLhttps://doi.org/10.48550/arXiv.2407.21783.
[15] N. Guha, J. Nyarko, D. Ho, et al. LegalBench: A collaboratively built benchmark for measuring
legal reasoning in large language models. InProceedings of the 37th Conference on Neural
Information Processing Systems - Datasets and Benchmarks Track, pages 44123–44279, 2023.
[16] A. B. Hou, O. Weller, G. Qin, E. Yang, D. Lawrie, N. Holzenberger, A. Blair-Stanek, and
B. Van Durme. CLERC: A dataset for us legal case retrieval and retrieval-augmented analysis
generation. InFindings of the Association for Computational Linguistics: NAACL 2025, pages
7913–7928, 2025. URLhttps://doi.org/10.18653/v1/2025.findings-naacl.441.
[17] X. Hu, D. Ru, L. Qiu, Q. Guo, T. Zhang, Y . Xu, Y . Luo, P. Liu, Y . Zhang, and Z. Zhang.
RefChecker: Reference-based fine-grained hallucination checker and benchmark for large
language models.arXiv preprint arXiv:2405.14486, 2024. URL https://doi.org/10.
48550/arXiv.2405.14486.
[18] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda,
A. Hayes, A. Radford, et al. GPT-4o system card.arXiv preprint arXiv:2410.21276, 2024.
URLhttps://doi.org/10.48550/arXiv.2410.21276.
[19] A. Q. Jiang, A. Sablayrolles, A. Roux, et al. Mixtral of experts.arXiv preprint arXiv:2401.04088,
2024. URLhttps://doi.org/10.48550/arXiv.2401.04088.
[20] A. T. Kalai and S. S. Vempala. Calibrated language models must hallucinate. InProceedings
of the 56th Annual ACM Symposium on Theory of Computing, pages 160–171, 2024. URL
https://doi.org/10.1145/3618260.3649777.
[21] D. M. Katz, M. J. Bommarito, S. Gao, and P. Arredondo. GPT-4 passes the bar exam.Philosoph-
ical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences,
382(2270), 2024. URLhttps://doi.org/10.1098/rsta.2023.0254.
[22] F. Keisha, P. Singh, D. Fernandes, A. Manivannan, I. Wicaksono, F. Ahmad, W. B. Rim,
et al. All for law and law for all: Adaptive RAG pipeline for legal research.arXiv preprint
arXiv:2508.13107, 2025. URLhttps://doi.org/10.48550/arXiv.2508.13107.
[23] J. Lee, D. Kim, S. Hwang, H. Kim, and G. Lee. KoBLEX: Open legal question answering
with multi-hop reasoning. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 4019–4053, 2025. URL https://doi.org/10.18653/
v1/2025.emnlp-main.200.
[24] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis,
W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive
nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020. URL
https://doi.org/10.48550/arXiv.2005.11401.
[25] K. Li, Y . Li, T. Zhang, H. Luo, X. Wu, J. Glass, and H. Meng. RAG-Zeval: Enhancing RAG Re-
sponses Evaluator through End-to-End Reasoning and Ranking-Based Reinforcement Learning.
InProceedings of the 2025 Conference on Empirical Methods in Natural Language Process-
ing, pages 24936–24954, 2025. URL https://doi.org/10.18653/v1/2025.emnlp-main.
1267.
[26] L. Li, L. Sleem, G. Nichil, R. State, et al. Exploring the impact of temperature on large
language models: Hot or cold?Procedia Computer Science, 264:242–251, 2025. URL
https://doi.org/10.1016/j.procs.2025.07.135.
[27] A. Louis, G. van Dijck, and G. Spanakis. Interpretable long-form legal question answering
with retrieval-augmented large language models. InProceedings of the AAAI Conference on
Artificial Intelligence, pages 22266–22275, 2024. URL https://doi.org/10.1609/aaai.
v38i20.30232.
11

[28] V . Magesh, F. Surani, M. Dahl, M. Suzgun, C. D. Manning, and D. E. Ho. Hallucination-free?
assessing the reliability of leading AI legal research tools.Journal of Empirical Legal Studies,
22(2):216–242, 2025. URLhttps://doi.org/10.1111/jels.12413.
[29] S. Mallick. Generative AI in the law.the Law (February 10, 2024), 42, 2024. URL https:
//doi.org/10.2139/ssrn.5040429.
[30] P. Manakul, A. Liusie, and M. Gales. SelfCheckGPT: Zero-resource black-box hallucination
detection for generative large language models. InProceedings of the 2023 conference on
empirical methods in natural language processing, pages 9004–9017, 2023. URL https:
//doi.org/10.18653/v1/2023.emnlp-main.557.
[31] D. Metropolitansky and J. Larson. Veritrail: Closed-domain hallucination detection with
traceability.arXiv preprint arXiv:2505.21786, 2025. URL https://doi.org/10.48550/
arXiv.2505.21786.
[32] D. Metropolitansky and J. Larson. Towards effective extraction and evaluation of factual claims.
InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 6996–7045, 2025. URL https://doi.org/10.18653/v1/
2025.acl-long.348.
[33] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and
H. Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form
text generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pages 12076–12100, 2023. URL https://doi.org/10.18653/v1/
2023.emnlp-main.741.
[34] J. Niklaus, V . Matoshi, P. Rani, A. Galassi, M. Stürmer, and I. Chalkidis. LEXTREME: A
multi-lingual and multi-task benchmark for the legal domain. InFindings of the Association for
Computational Linguistics: EMNLP 2023, pages 3016–3054, 2023. URL https://doi.org/
10.18653/v1/2023.findings-emnlp.200.
[35] M. Park, H. Oh, E. Choi, and W. Hwang. LRAGE: Legal retrieval augmented generation
evaluation tool.arXiv preprint arXiv:2504.01840, 2025. URL https://doi.org/10.48550/
arXiv.2504.01840.
[36] N. Pipitone and G. H. Alami. LegalBench-RAG: A benchmark for retrieval-augmented
generation in the legal domain.arXiv preprint arXiv:2408.10343, 2024. URL https:
//doi.org/10.48550/arXiv.2408.10343.
[37] N. Reimers and I. Gurevych. The curse of dense low-dimensional information retrieval for large
index sizes. InProceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Processing
(Volume 2: Short Papers), pages 605–611. Association for Computational Linguistics, 2021.
URLhttps://doi.org/10.18653/v1/2021.acl-short.77.
[38] M. Renze. The effect of sampling temperature on problem solving in large language models.
InFindings of the association for computational linguistics: EMNLP 2024, pages 7346–7356,
2024. URLhttps://doi.org/10.18653/v1/2024.findings-emnlp.432.
[39] S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond.
Found. Trends Inf. Retr., 3(4):333–389, Apr. 2009. ISSN 1554-0669. URL https://doi.org/
10.1561/1500000019.
[40] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, C. Jiayang, C. Wang, S. Sun, H. Li,
et al. RAGChecker: A fine-grained framework for diagnosing retrieval-augmented generation.
Advances in Neural Information Processing Systems, 37:21999–22027, 2024. URL https:
//doi.org/10.52202/079017-0692.
[41] N. Sannier, M. Adedjouma, M. Sabetzadeh, L. Briand, J. Dann, M. Hisette, and P. Thill.
Legal markup generation in the large: An experience report. In2017 IEEE 25th International
Requirements Engineering Conference (RE), pages 302–311. IEEE, 2017. URL https://doi.
org/10.1109/RE.2017.10.
[42] A. Scirè, K. Ghonim, and R. Navigli. FENICE: Factuality evaluation of summarization
based on natural language inference and claim extraction. InFindings of the Association
for Computational Linguistics ACL 2024, pages 14148–14161, 2024. URL https://doi.
org/10.18653/v1/2024.findings-acl.841.
12

[43] A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low,
A. Ostrow, A. Ananthram, et al. OpenAI GPT-5 system card.arXiv preprint arXiv:2601.03267,
2025. URLhttps://doi.org/10.48550/arXiv.2601.03267.
[44] The European Parliament and the Council of the European Union. Regulation (eu) 2016/679
of the european parliament and of the council of 27 april 2016 on the protection of natural
persons with regard to the processing of personal data and on the free movement of such
data, and repealing directive 95/46/ec (General Data Protection Regulation), 05 2016. URL
https://eur-lex.europa.eu/eli/reg/2016/679/oj.
[45] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei. Improving text embeddings
with large language models.arXiv preprint arXiv:2401.00368, 2023. URL https://doi.
org/10.48550/arXiv.2401.00368.
[46] Y . Wang, M. Wang, H. Iqbal, G. N. Georgiev, J. Geng, I. Gurevych, and P. Nakov. Openfactcheck:
Building, benchmarking customized fact-checking systems and evaluating the factuality of
claims and llms. InProceedings of the 31st international conference on computational linguis-
tics, pages 11399–11421, 2025. URL https://aclanthology.org/2025.coling-main.
755/.
[47] J. Wei, C. Yang, X. Song, Y . Lu, N. Hu, J. Huang, D. Tran, D. Peng, R. Liu, D. Huang, et al.
Long-form factuality in large language models.Advances in Neural Information Processing
Systems, 37:80756–80827, 2024. URLhttps://doi.org/10.52202/079017-2567.
[48] B. Weiser. ‘I apologise for the confusion earlier’: Here’s what happens when your lawyer
uses ChatGPT’. heres-what-happens-when-your-lawyer-uses-chatgpt, May 28 2023. Accessed:
2026-01-04.
[49] N. Wiratunga, R. Abeyratne, L. Jayawardena, K. Martin, S. Massie, I. Nkisi-Orji, R. Weeras-
inghe, A. Liret, and B. Fleisch. CBR-RAG: case-based reasoning for retrieval augmented genera-
tion in llms for legal question answering. InInternational Conference on Case-Based Reasoning,
pages 445–460. Springer, 2024. URL https://doi.org/10.1007/978-3-031-63646-2_
29.
13

A Benchmark Curation
To create ClaimRAG-LAW, we followed a two-step methodology, where we first generated question-
answer (QA) pairs automatically and then had them manually validated by a legal expert; we elaborate
on these two steps in the next two subsections.
Section A.1 presents the automated pipeline for generating QA pairs, including document parsing,
QA pairs generation, and claim extraction, Section A.2 outlines the manual validation procedure,
Section A.3 presents the claim extraction and verification process, and Section A.4 summarizes
implementation details.
A.1 Automated Generation of QA Pairs
Our automated approach consists of the following steps.
(1) Document Parsing:In step 1, we parse the legal document and split it into partitions. Since
both documents used in this work are available in HTML format (as is common for European legal
texts) we split them using the <p>tag. This results in systematic partitions, each corresponding to an
analysis unit that approximates a self-contained legal clause or paragraph. While a structure-aware
split (e.g., by articles/sections) would be preferable, it introduces non-trivial challenges noted in prior
work [ 41], which are beyond the scope of this study. The extractedcontext unitsare passed on to the
next step.
(2) QA Pairs GenerationIn step 2, we generate a set of QA pairs from eachcontext unitfollowing
the predefined questions categories. Specifically, we utilize an LLM to first perform asuitability
assessmentto analyze whether thecontext unitcontains sufficient details to support generating
relevant questions across the different categories. The LLM then generates the respective QA pairs
only if the outcome of this assessment is positive; otherwise, the category is skipped. This procedure
resulted in single-hop questions.
We further generate multi-hop questions by providing the LLM with multiple adjacentcontext units
and explicitly instructing it to produce multi-hop queries. Specifically, we concatenate as many
consecutivecontext unitsas can fit within 100k tokens, reserving a 4096-token safety buffer for
generation to remain within the model’s 128k-token context window. The 4096 tokens are designated
for model output, covering the generation of question, its corresponding answer, and the supporting
rationale. On average, eachcontext unitcomprises approximately 1300 ±200 tokens (roughly
5200±800 characters), enabling around 70–75context unitswithin the 100k-token input limit. In
addition to assessing the suitability of the question category, we prompt the LLM to ensure that
each generated questioncannotbe answered from any singlecontext unitalone, but instead requires
linking information from at least two distinct parts of the adjacentcontext units.
Figures 1 and 2 showsingle-hopQA prompts, while Figures 3 and 4 showmulti-hopQA prompts.
The output of our approach was a total of 165 single-hop question-answer pairs and 204 multi-hop QA
pairs. We then provided these pairs to the legal expert for manual validation, described in Section A.2.
A.2 Manual Validation
Evaluating AI generated content remains an open research challenge. In a high-stakes domain such
as law, expert qualitative assessment is essential [ 28]. In our work, manual validation by the legal
expert ensures the quality and relevance of QA pairs. To further enhance the objectivity and mitigate
potential bias, we relied on independent manual curation conducted by a third-party annotator.
Specifically, an independent legal expert (with a PhD degree in law), pseudonymized throughout
this study asJo, was contracted to review both the QA pairs and the extracted claims. Jo has prior
experience annotating legal texts to support the development of automated solutions for regulatory
and compliance challenges. Jo is also bilingual in French and English, which further strengthens the
annotation process, given the languages of the considered source documents.
Before starting the full annotation, we held a series of online meetings to align on the task and to
draft clear annotation guidelines. We then ran a pilot phase to confirm that Jo was familiar with the
workflow and that the guidelines were interpreted consistently. During this pilot, Jo annotated a small
14

System Prompt: Single-hop QA Tuple Generation
“You are a legal domain expert specializing in statutory and regulatory interpretation. Your task is to
analyze the provided legal text and decide whether it contains explicit, self-contained, and unambiguous
information sufficient to generate a single high-quality question of the specified category. If the text
does not have the sufficient information to generate a question, output"suitable": false.
If the text is suitable, generate a JSON object containing:
•question: A clear, precise legal question strictly conforming to the specified question category.
•answer: A factually correct answer that isfully and explicitly supported by the text.
•rationale : A step-by-step explanation showing how the answer is directly deduced from specific
parts of the text.
Constraints
• Avoid ambiguity, speculation, or external legal assumptions.
• Do not merge multiple questions.
• Do not generate more than one question.
• Ensure the output language exactly matches the language of the source text.
• Output valid JSON only. No additional commentary.”
Figure 1: System Prompt used for the Conditional Generation of Single-hop QA tuples.
User Prompt
“Assess the provided context below. If suitable, generate aquestion, answer, and rationalebased on
this context in the specified language. The type of the question should strictly adhere to the following
definition of the question category- {question_type} : {question_definition}:
Language:{target_language}
Context:{context}”
Figure 2: User Prompt for single-hop dataset generation.
subset of the dataset (consisting of 20 QA pairs and 43 claims) following the guidelines, after which
we conducted a feedback meeting to discuss edge cases and resolve ambiguities.
To minimize fatigue and maintain consistency, the dataset was annotated in multiple batches over
a two-month period. Jo spent a total of 55 hours on the task and was advised to keep individual
annotation sessions to a maximum of two hours.
Below, we summarize the main labels used in the manual validation.
Question Validity and CategorizationFor each question, Jo was instructed to assess its validity
from a domain (“does it make sense?”) and linguistic (“is it understandable?”) perspectives. Only for
questions deemedvalid, Jo further validated two key fields:
•Category. As part of our automated generation pipeline, each question was initially assigned a
category. Jo then carefully validated these assignments and corrected them where needed, following
the definitions specified in the annotation guidelines.
•Persona.
To assess the complexity of the automatically generated questions, we asked Jo to identify the
intended user profile for each question (i.e., thepersona). This ranges from a layperson/citizen,
who would typically pose basic questions, to a legal expert, who would ask more domain-specific
questions. We also introduced an intermediate persona—civil officer—to capture informed users who
have some legal background but lack deep expertise.
15

System Prompt: Multi-hop QA Generation
“You are a legal expert specializing in complex statutory interpretation. Your task is to analyze the
provided multi-article legal context and determine if it supports a multi-hop question. A multi-hop
question must:
• Require synthesizing information from at least two distinct provisions or articles within the text.
• Be unanswerable from any single provision or article.
If the text does not have the sufficient information or the text does not support to generate multi-hop
question, output"suitable": false.
If suitable, generate a JSON object containing:
•question: A complex legal question requiring synthesis of the text.
•answer: A comprehensive answer derived from connecting the relevant provisions.
•rationale: A step-by-step deduction path citing the specific distinct segments used.
Constraints
• Do not generate questions answerable by a single provision or article.
• Ensure the output language matches the source text.
• Output valid JSON only.”
Figure 3: System Prompt used for the Conditional Generation of Multi-hop QA tuples.
User Prompt (Multi-hop)
“Assess the provided multi-article context below. If suitable, generate amulti-hop question, answer,
and rationalebased on this context in the specified language. The question must strictly adhere to the
following category definition - {question_type} : {question_definition}:
Language:{target_language}
Context:{context}”
Figure 4: User Prompt for multi-hop dataset generation.
Answer CorrectnessFor each valid question, Jo validated the correctness of the provided answer.
Based on our feedback session, an answer is consideredcorrectif it is correct and relevant to the
question;partially correctif it is grammatically incorrect but still sufficiently addresses the question
or it is logically correct but has language issues (i.e., a human would not phrase it this way), and
incorrectotherwise. Among the 165 single-hop QA pairs, 152 were validated as correct and form the
ground-truth answers used in the subsequent claim extraction step.
A.3 Claim Extraction and Analysis
A claim is defined as a triple ⟨subject,predicate,object ⟩, representing a minimal, self-contained
factual statement extracted from a text [ 17]. For example, the sentence“Article 1 states that the
regulation aims to protect natural persons”yields the claim ⟨Article 1, aims to protect, natural
persons ⟩. Following subsequent sections describe how claims are extracted from the ground-truth
answers in our benchmark and how they are subsequently validated and annotated by a legal expert.
Claim ExtractionTo enable a fine-grained evaluation of RAG system performance, we extract
claims from the 152 single-hop ground-truth answers (a sub-set of QA pairs verified as correct by Jo
in the previous step). Restricting claim extraction to these verified answers ensures a reliable and
legally sound basis for evaluation.
To extract claims, we use RefChecker [ 17], which was also used by the RAGChecker evaluation
framework for the same purpose. Our goal is to examine RefChecker’s behavior in this setting and
to study how the resulting claim set affects the fine-grained evaluation we conduct in this work.
RefChecker comprises two components: anextractorand achecker. We use the extractor to derive
16

claims from the ground-truth answers. This step yielded a total of 968 claims, which were then
provided to Jo for manual validation, as described below.
Claim-Level AnalysisJo validated each claim for correctness: a claim is considered correct if
it contains all required elements and is logically coherent and meaningful; otherwise, it is deemed
incorrect. Additionally, Jo annotated, for correct claims, the entailment relation between each claim
and the providedcontext unit. A claim is deemedentailedif it is explicitly supported by thecontext
unit,contradictoryif conflicts with information stated incontext unit, andneutralwhen it contains
information not grounded incontext unit.
A.4 Implementation Details
To implement the automated QA generation dataset, we used the following setup. We employed
GPT-4 ( gpt-4-0613 ) via the OpenAI API as the primary LLM in our generation pipeline, motivated
by its reasoning capabilities and performance on complex legal texts [ 21]. To reduce repetition and
given that all generated QA pairs were manually validated, we varied the temperature according
to the goals of each task, striking a balance between determinism and diversity in the generated
outputs [ 26,38]. We used temperature 0.0 for tasks that require strict determinism or do not benefit
from variation (e.g., suitability assessment, factual recall, and claim extraction). For generating
general legal research and jurisdiction/time-specific questions, we used a random temperature in the
range 0.3–0.5 to introduce moderate linguistic variation while preserving logical coherence. Finally,
for generating false-premise questions, we set the temperature to 0.7 to encourage the LLM to deviate
from generating correct questions toward producing legally incorrect assumptions.
17
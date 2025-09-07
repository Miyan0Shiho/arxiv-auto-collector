# Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain

**Authors**: Shakiba Amirshahi, Amin Bigdeli, Charles L. A. Clarke, Amira Ghenai

**Published**: 2025-09-04 00:45:58

**PDF URL**: [http://arxiv.org/pdf/2509.03787v1](http://arxiv.org/pdf/2509.03787v1)

## Abstract
Retrieval augmented generation (RAG) systems provide a method for factually
grounding the responses of a Large Language Model (LLM) by providing retrieved
evidence, or context, as support. Guided by this context, RAG systems can
reduce hallucinations and expand the ability of LLMs to accurately answer
questions outside the scope of their training data. Unfortunately, this design
introduces a critical vulnerability: LLMs may absorb and reproduce
misinformation present in retrieved evidence. This problem is magnified if
retrieved evidence contains adversarial material explicitly intended to
promulgate misinformation. This paper presents a systematic evaluation of RAG
robustness in the health domain and examines alignment between model outputs
and ground-truth answers. We focus on the health domain due to the potential
for harm caused by incorrect responses, as well as the availability of
evidence-based ground truth for many common health-related questions. We
conduct controlled experiments using common health questions, varying both the
type and composition of the retrieved documents (helpful, harmful, and
adversarial) as well as the framing of the question by the user (consistent,
neutral, and inconsistent). Our findings reveal that adversarial documents
substantially degrade alignment, but robustness can be preserved when helpful
evidence is also present in the retrieval pool. These findings offer actionable
insights for designing safer RAG systems in high-stakes domains by highlighting
the need for retrieval safeguards. To enable reproducibility and facilitate
future research, all experimental results are publicly available in our github
repository.
  https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL

## Full Text


<!-- PDF content starts -->

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial
Evidence in the Health Domain
SHAKIBA AMIRSHAHI, University of Waterloo, Canada
AMIN BIGDELI, University of Waterloo, Canada
CHARLES L. A. CLARKE, University of Waterloo, Canada
AMIRA GHENAI, Toronto Metropolitan University, Canada
Retrieval augmented generation (RAG) systems provide a method for factually grounding the responses of a Large Language Model
(LLM) by providing retrieved evidence, or context, as support. Guided by this context, RAG systems can reduce hallucinations and
expand the ability of LLMs to accurately answer questions outside the scope of their training data. Unfortunately, this design introduces
a critical vulnerability: LLMs may absorb and reproduce misinformation present in retrieved evidence. This problem is magnified if
retrieved evidence contains adversarial material explicitly intended to promulgate misinformation. This paper presents a systematic
evaluation of RAG robustness in the health domain and examines alignment between model outputs and ground-truth answers. We
focus on the health domain due to the potential for harm caused by incorrect responses, as well as the availability of evidence-based
ground truth for many common health-related questions. We conduct controlled experiments using common health questions, varying
both the type and composition of the retrieved documents (helpful, harmful, and adversarial) as well as the framing of the question by
the user (consistent, neutral, and inconsistent). Our findings reveal that adversarial documents substantially degrade alignment, but
robustness can be preserved when helpful evidence is also present in the retrieval pool. These findings offer actionable insights for
designing safer RAG systems in high-stakes domains by highlighting the need for retrieval safeguards. To enable reproducibility and
facilitate future research, all experimental results are publicly available in our github repository1.
CCS Concepts: •Information systems →Evaluation of retrieval results ;Information retrieval query processing ;Adversarial
retrieval ;•Applied computing →Health informatics .
Additional Key Words and Phrases: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Adversarial Robustness,
Query Framing, Contextual Alignment
ACM Reference Format:
Shakiba Amirshahi, Amin Bigdeli, Charles L. A. Clarke, and Amira Ghenai. 2025. Evaluating the Robustness of Retrieval-Augmented
Generation to Adversarial Evidence in the Health Domain. In Proceedings of Make sure to enter the correct conference title from your
rights confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 25 pages. https://doi.org/XXXXXXX.XXXXXXX
1https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL
Authors’ Contact Information: Shakiba Amirshahi, University of Waterloo, Waterloo, ON, Canada, shakiba.amirshahi@uwaterloo.ca; Amin Bigdeli,
University of Waterloo, Waterloo, ON, Canada, abigdeli@uwaterloo.ca; Charles L. A. Clarke, University of Waterloo, Waterloo, ON, Canada, claclark@
gmail.com; Amira Ghenai, Toronto Metropolitan University, Toronto, ON, Canada, aghenai@torontomu.ca.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2509.03787v1  [cs.IR]  4 Sep 2025

2 Amirshahi et al.
1 Introduction
Large Language Models (LLMs) have become central to modern information access systems, powering a broad range
of applications in fields such as question answering [ 2,45,51]. However, their deployment introduces new risks to
information reliability, particularly due to their susceptibility to hallucinations and the generation of factually incorrect
responses [ 4,8,33,55,83]. To resolve these issues and improve factual grounding, many such systems adopt a Retrieval-
Augmented Generation (RAG) architecture, which combines the generalization capabilities of LLMs with document-level
evidence retrieved from an external corpus to enhance the credibility and accuracy of output answer [ 28,79]. When
answering questions, RAG systems employ retrieved documents as context to provide factual ground for their responses.
While the architecture of RAG systems is designed to enhance accuracy and trustworthiness, it also introduces new
vulnerabilities as the reliability of LLM output becomes contingent on the integrity of the retrieved documents employed
as context. Therefore, if malicious or misleading content is injected into the retrieval corpus, whether unintentionally
or through deliberate manipulation by attackers, LLMs may confidently generate harmful misinformation.
Under this threat model, the retrieval system and its neural ranking model (NRM) act as an attack vector for adversarial
documents, which are designed to exploit scoring and ranking functions, thereby elevating documents containing
malicious content into the top-k ranks [ 5,14,47,78]. When supplied as context to the LLM, these documents then
serve as a payload, delivering manipulated content that can directly influence the generation process. This two-stage
vulnerability mirrors classical security architectures, where an exploit (vector) delivers a harmful effect (payload) to a
dependent system.
A growing body of literature demonstrates that LLMs are susceptible to such adversarial attacks [ 10,52,80,86,89].
For example, Zhang et al . [86] craft adversarial documents by appending the user query with a fixed instruction prompt
designed to elicit a specific answer from the LLM. In a related study, Chaudhari et al . [10] concatenates the user
query with templated instruction prompts to achieve various attack objectives, such as inducing refusals to answer or
generating biased responses. Zou et al . [89] adopt a similar strategy by pairing the user query with an LLM-generated
passage that embeds a malicious answer.
Despite demonstrating the feasibility of adversarial manipulation in RAG pipelines, existing attack strategies face
several notable limitations and challenges: (1) Attacks that append the query to manipulated documents, commonly
referred to as Query+ in the IR adversarial literature, are easily detectable, as they introduce repetitive patterns that
can be flagged by both human evaluators and automated filters designed to detect semantic redundancy [ 5,14,78]. (2)
Fixed-format prompt injections or simplistic LLM-generated texts lacking rhetorical diversity, which could be detected
by safety systems similar to appending the query to the document. (3) These attacks do not evaluate LLM vulnerabilities
on high-stakes topics where models are explicitly trained to resist misinformation, such as health domains.
Our work does not introduce a new attack method but instead provides an empirical robustness evaluation of LLMs
under controlled conditions, where real and realistic documents containing misinformation are introduced as RAG
context. To the best of our knowledge, no prior study has empirically tested RAG robustness in medically critical
domains using realistic human-annotated and adversarially constructed documents, making this the first systematic
assessment in such a high-risk setting. Specifically, we investigate the vulnerabilities of LLMs in both RAG and Non-RAG
settings in the health domain, especially their sensitivity to health-related misinformation, such as counterfactual claims
about COVID-19, where incorrect or misleading results can have severe real-world consequences. We focus on precisely
those domains where models are expected to be most resilient due to targeted alignment and safety training, thereby
providing a more stringent and realistic robustness assessment. Unlike prior work that relies on synthetic, templated,
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 3
or low-stakes content, we employ naturalistic, high-risk adversarial documents designed to evade pre-processing and
semantic similarity filters. Specifically, we: (1) evaluate the susceptibility of RAG systems to adversarial documents in
the medical domain where these documents are provided as retrieval context for medically consequential queries, and
(2) analyze how presuppositions and tone embedded in user queries interact with retrieved content to bias the LLM’s
interpretation, agreement, and final response.
To this end, we conduct experiments using test collections from the TREC 2020 and TREC 2021 Health Misinformation
Tracks [ 18,19], which provide medically focused queries accompanied by gold-standard human-written documents
labeled as helpful, harmful, or neutral by expert assessors based on relevance, correctness, and credibility. Conventional
large-scale benchmarks such as MS MARCO [ 53] or the TREC Web tracks [ 20], as well as other existing datasets, are
unsuitable for our purposes, as they provide only topical relevance judgments and do not include per-query annotations
distinguishing helpful from harmful documents or accounting for factual accuracy and misinformation. To complement
these human-annotated corpora, we also employ LLM-generated adversarial documents released by Bigdeli et al . [6]. An
adversarial document is defined as a document that has been deliberately manipulated or generated to appear credible
while introducing misleading or incorrect information, with the explicit goal of disrupting retrieval performance. In
their work, Bigdeli et al . [6] generated such documents for each helpful item in both TREC 2020 and TREC 2021 using
a range of adversarial generation strategies. These include methods that invert factual claims, reframe content to
introduce subtle inaccuracies, and inject misleading evidence, thereby producing realistic but harmful alternatives that
can evade detection, ranking above authentic, helpful documents.
To evaluate the interaction between document content and user query framing, we construct three distinct versions
of each query to reflect different user presuppositions: (i) a consistent version aligned with the query’s ground-truth
stance, (ii) a neutral version, representing the original query from the test collection, and (iii) an inconsistent version that
contradicts the ground-truth stance. By pairing these with helpful, harmful, and adversarial documents, we evaluate
LLMs in both RAG (Ragnarok framework [ 62]) and Non-RAG settings. Importantly, we assume documents are already
retrieved, which allows us to isolate the generation stage from the retrieval process. This enables a focused evaluation
of LLM behavior under controlled document-query conditions.
Based on this setup, our study addresses the following research questions (RQs):
•RQ1: Does retrieved context influence LLM responses?
•RQ2: To what extent does the type of retrieved context (helpful, harmful, and adversarial) influence robustness?
•RQ3: In what ways do orders and combinations of multiple evidence sources shape model behavior?
•RQ4: How do different query framings (consistent, neutral, and inconsistent) influence ground-truth alignment?
Our contribution lies in providing a systematic empirical characterization of these vulnerabilities and the contextual
factors that influence their severity. Although the susceptibility of RAG systems to adversarial documents has been
established conceptually, the quantitative impact across different document types, their interaction with query framing,
and their effectiveness against safety-aligned models in the high-stakes health domain remains underexplored. By
quantifying the degree to which adversarial documents undermine ground truth alignment, and by analyzing how
query framing interacts with retrieved content, our study provides an evidence base for understanding and mitigating
risks. Through systematic measurement across query variations and document combinations, we establish baseline
vulnerability metrics and provide the empirical foundation necessary for developing robust retrieval architectures.
This systematic evaluation addresses a critical gap between theoretical attack feasibility and practical risk assessment,
transforming abstract security concerns into measurable risks for evidence-based system design.
Manuscript submitted to ACM

4 Amirshahi et al.
More concretely, the contributions of our work include:
•We introduce a controlled evaluation framework for measuring RAG robustness against adversarial exposure in
high-stakes health domains.
•We systematically analyze how query framing and document type jointly influence ground-truth alignment,
offering insights into how implied user presuppositions interact with retrieved evidence.
•We provide empirical findings showing that while adversarial documents substantially degrade ground-truth
alignment, LLMs exhibit greater resilience to COVID-related misinformation compared to general health misin-
formation, suggesting that post-training alignment influences domain-specific robustness.
2 Related Work
2.1 RAG Systems
RAG systems emerged in part as an approach to overcoming the hallucination problem of LLMs, and to increase
accuracy in generated responses by combining their generative capabilities with factual grounding provided by external
knowledge [ 28,31,40]. In a basic RAG pipeline, a retrieval component first identifies and ranks relevant documents
from a large corpus. Then, the top-k documents are passed to an LLM as context for generating the final response. This
RAG architecture has been widely adopted for question answering and similar applications [2, 45, 71].
Recent research has increasingly centered on evaluating and improving the performance of RAG systems, while also
introducing tailored architectures and task-specific adaptations [ 23,27,34,35,44,48,72,74,84,87]. For instance, [ 44]
explored the impact of the positioning of the documents within the context window on the output generated by the
LLM. In a similar study, Cuconasu et al . [23] investigated the impact of the positioning of not only relevant documents
but also noisy irrelevant documents on the RAG system. Gao et al . [27] introduced a RAG pipeline that jointly optimizes
its policy network, retriever, and answer generator via reinforcement learning, achieving better performance and
lower retrieval cost than separately trained modules. While these works provide valuable insights into RAG behavior,
they primarily focus on the RAG settings and their performance. In contrast, our work investigates RAG systems
under adversarial conditions, where the context may be intentionally crafted to mislead the LLM, thereby exposing
vulnerabilities at the intersection of retrieval and generation.
2.2 Prompt Framing in LLMs and RAG Systems
Recent advancements in LLMs have demonstrated impressive performance across various natural language processing
tasks. However, their sensitivity to prompt formulation remains a notable challenge [ 63,65]. Minor modifications in
the wording, structure, or even punctuation of prompts can cause outputs that are substantially different and often
incorrect [ 32,42,49,50,57,63,65,75]. Qiang et al . [63] demonstrated that fine-tuned LLMs suffer large performance
drops when prompts are perturbed by synonyms or paraphrases, introducing Prompt Perturbation Consistency Learning
to enforce prediction stability across such variants. In related work, Mao et al . [49] conducted a comprehensive study of
prompt position and demonstrated that the location of prompt tokens in input text has a large effect on zero-shot and
few-shot performance, with many widely used placements proving suboptimal.
In the RAG setting, prompt effects also remain critical. Perçin et al . [57] investigated the robustness of RAG pipelines
at the query level, illustrating that subtle prompt variations, such as redundant phrasing and shifts in formal tone,
can considerably affect retrieval quality and overall accuracy. Complementing these findings, Hu et al . [32] analyzed
prompt perturbations in RAG and proposed the Gradient Guided Prompt Perturbation, an adversarial method that
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 5
steers model outputs toward incorrect answers, while also presenting a detection approach based on neuron activation
patterns to mitigate these risks.
In this paper, we extend this line of work by examining how different query framings interact with retrieval context
in RAG systems focused on the health domain. We evaluate consistent, neutral, and inconsistent query styles to measure
their influence on ground-truth alignment and robustness under adversarial retrieval conditions. This design captures
the ways in which query framing amplifies or mitigates the effects of helpful and harmful evidence, offering a systematic
view of prompt sensitivity in a high-stakes domain.
2.3 Adversarial Attacks on Neural Retrieval Models and LLMs
A basic RAG system comprises two primary components: a retriever, which retrieves and ranks relevant documents
from an external corpus, and an LLM, which generates responses based on the retrieved context. Prior research has
investigated the vulnerabilities of each component in isolation. In particular, recent studies have shown that dense
retrieval and neural ranking models could serve as an attack vector, through which minor perturbations and manipulation
to already existing malicious documents can significantly boost their ranking positions [ 5,14,43,47,76,78]. For instance,
Liu et al . [47] shows that a subtle word substitution-based attack can boost the ranking position of random documents
in the retrieved list of documents for queries. Liu et al . [43] propose a trigger-based adversarial attack against neural
ranking models that leverages a surrogate model to identify tokens highly influential to the model’s ranking score.
These “trigger tokens” are then injected into target documents to exploit the surrogate model’s vulnerabilities, thereby
boosting the documents’ rankings in the ranked list of documents given by the victim model. Bigdeli et al . [5] propose
an embedding-based perturbation attack that shifts a target document’s representation closer to that of the query to
generate adversarial sentences that can successfully promote its ranking and deceive neural rankers. Another approach
by Chen et al . [14] generates connection sentences between the query and the target document using the BART [ 39]
language model and appends them to the target document to boost it among the top-k documents.
Researchers have also introduced various attack strategies targeting LLMs, primarily through jailbreak attacks [ 9,24,
41,68,77,88] and prompt injection attacks [ 30,46,56,58,69]. Jailbreak attacks bypass the safety alignment of LLMs by
crafting prompts designed to deceive the model into producing harmful or undesirable behaviors, disclosing sensitive
information, or otherwise violating its intended safety constraints. Prompt injection attacks manipulate model behavior
by embedding malicious instructions directly into the input prompt, often overriding or subverting the original task.
For example, an attacker may insert a directive such as: “Ignore the instructions above and do... ” [ 58], causing the model
to follow the injected command instead of the intended instructions. Although jailbreak and prompt injection attacks
have demonstrated the ability to exploit vulnerabilities in LLMs and bypass their safety mechanisms, recent research
has shown that many of these weaknesses can be mitigated through targeted defense strategies. Broadly, these defenses
fall into two categories: model-level defenses and input-level defenses.
Model-level defenses aim to make the LLM inherently more resistant to malicious prompts by refining its internal
decision-making and alignment mechanisms. Notable approaches include fine-tuning methods such as Direct Preference
Optimization (DPO) [ 7,12,13,60,64,81] which optimize the model’s responses to better align with human-preferred safe
behaviors, as well as other alignment-enhancement techniques [ 11,38,59,66,70,82,85] that reinforce policy adherence,
increase refusal consistency, or provide provable robustness guarantees. Input-level defenses focus on intercepting
and neutralizing malicious instructions before they are processed by the LLM. These include prompt filtering and
classification systems that detect adversarial intent and block harmful requests at inference time [ 3,36,37,67]. Such
systems can identify suspicious patterns, injection-like structures, or known exploit phrases without significantly
Manuscript submitted to ACM

6 Amirshahi et al.
degrading the model’s usability. Together, these defense strategies have significantly improved the robustness of LLMs
against prompt injection and jailbreak attacks, making it more difficult for adversaries to induce harmful behaviors or
override safety constraints.
2.4 Adversarial Attacks on RAG Systems
With the emergence of the RAG paradigm, researchers have investigated its vulnerabilities to poisoning attacks that
target the generator phase by either promoting malicious documents into the retrieved context used for grounding and
answer generation or by introducing adversarial prompt perturbations [10, 15, 17, 32, 52, 80, 86, 89].
Several studies focus on document-level poisoning, where the attacker crafts malicious documents that are injected
into the corpus for retrieval by the RAG system. For example, Zhang et al . [86] and Chaudhari et al . [10] append
user search queries with documents containing fixed malicious instruction prompts, enabling these documents to
rank within the top-k retrieved results and causing the LLM to produce incorrect output. Similarly, Zou et al . [89]
leverages LLM-induced conditional hallucinations to generate documents with factually incorrect answers, subsequently
appending the query to boost their ranking. Xue et al . [80] introduces a trigger-based corpus poisoning attack on
RAG systems, optimizing adversarial passages for targeted retrieval and using them to manipulate LLM outputs via
denial-of-service and sentiment steering. Cheng et al . [15] presents a poisoning framework for RAG systems that
generates adversarial documents containing factually incorrect answers and query-matching trigger text, enabling
them to rank highly in retrieval and mislead the LLM during answer generation. Cho et al . [17] proposed a genetic
algorithm-based attack on RAG systems by introducing subtle, low-level perturbations (such as typos) into the retrieval
corpus. These perturbations aim to degrade retriever performance, allowing malicious passages to replace correct ones
in the retrieval results and influence the generated answers.
Other work investigated the impact of prompt-level perturbations on the robustness of RAG systems. For example,
Hu et al . [32] introduced a gradient guided prompt perturbation method that inserts short adversarial prefixes into
user queries that causing the RAG pipeline to retrieve targeted, misleading passages and generate factually incorrect
answers. Similarly, Wang and Yu [73] introduces a black-box adversarial attack on RAG systems that appends short,
optimized adversarial suffixes to user queries to promote a targeted incorrect document into high retrieval ranks that
manipulates the RAG response.
While prior work has demonstrated that RAG pipelines are susceptible to adversarial manipulation, existing attacks
remain limited in scope and practicality. Many rely on appending the query to malicious documents to satisfy retrieval
patterns, a strategy that is both easily detectable by spam filters [ 5,14,78]. Others use fixed-format prompt injections
with low rhetorical diversity or generate text that lacks naturalness, making it more susceptible to detection by pre-
processing and semantic similarity filters. Some methods depend on prompt-level perturbations that are impractical in
real-world settings without access to user search queries, while others assume white-box access to retrievers, which
may be an unrealistic requirement in many applications. Moreover, these approaches are often undermined by modern
LLM reasoning and are rarely evaluated in high-stakes domains where models are trained to resist misinformation.
Our work complements this body of work by shifting the focus from proposing new attacks to empirically testing
RAG robustness under realistic, high-stakes conditions. Specifically, we employ health-related queries from TREC’s
misinformation tracks alongside both human-annotated harmful documents and adversarially generated alternatives [ 6].
This setup allows us to evaluate how LLMs behave when exposed to naturalistic, medically consequential misinformation
that is designed to evade detection in both RAG and Non-RAG settings. In contrast to prior approaches that rely on
synthetic injection patterns, we test scenarios where the malicious payload is both plausible and high-risk, reflecting
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 7
conditions that are of greatest societal concern. Furthermore, by analyzing how document type interacts with user
query framing, we provide insights into the combined influence of evidence and presuppositions on model robustness,
an aspect that has been largely overlooked in prior adversarial RAG research.
3 Methodology
In this section, we describe the methodology used to assess how combinations of document types and query framings
influence the reliability of LLM outputs in our health misinformation scenarios. Specifically, we examine how user
queries with different presuppositions, together with helpful, harmful, and adversarial documents, affect the ground-
truth alignment and robustness of model responses in health-related question answering. In all settings, each prompt
consists of a user query combined with one or more retrieved documents.
3.1 Datasets
3.1.1 Benchmark Collections. Evaluating the robustness of LLMs and RAG systems requires datasets that go beyond
topical relevance and explicitly distinguish between helpful and harmful information. In adversarial settings, it is
insufficient to judge documents solely on relevance; their factual accuracy, credibility, and potential to mislead must
also be considered. Conventional large-scale benchmarks such as MS MARCO [ 21,53] and the TREC Web and Deep
Learning tracks [ 20,22] are inadequate for this purpose, as their annotations focus exclusively on topical relevance and
overlook judgments of misinformation.
To address this gap, we employ the TREC 2020 and TREC 2021 Health Misinformation Track collections [ 18,19],
which are explicitly designed for studying retrieval in the health domain. We select these collections because they are
the only benchmarks in which each query is paired with both human-written helpful and harmful documents, judged in
terms of three key criteria: correctness, credibility, and usefulness. According to the TREC Health Misinformation Track
guidelines [ 18,19], a document is considered helpful if it agrees with ground-truth medical consensus and provides
correct, credible, and useful information answering the query. In contrast, a document is considered harmful if it
disagrees with the ground-truth, promoting incorrect or misleading information.
The TREC 2020 track consists of 46 queries on COVID-19 treatments (e.g., “Can pneumococcal vaccine prevent
COVID-19?”), with candidate documents sourced from the Common Crawl News dataset2, covering the early months
of the pandemic. The TREC 2021 track comprises 35 queries on broader medical treatments (e.g., “Will taking zinc
supplements improve pregnancy?”), using documents from the “ noclean ” version of the C4 dataset3. Queries are
provided in both short keyword-style and longer natural-language forms, and each query is paired with a binary stance
label indicating whether the treatment is considered effective. Each topic in these collections has an associated set of
documents annotated by human assessors for their correctness, credibility, and usefulness in answering the queries.
These dimensions are combined into a single preference code for evaluation purposes. In TREC 2020, preference codes
range from -2 to 4, while in TREC 2021, they range from -3 to 12, with larger values denoting more helpful and credible
documents and negative values marking harmful misinformation.
3.1.2 Target Queries and Documents. Following the approach of Bigdeli et al. [ 6], we focus on queries and their
associated helpful and harmful documents from the TREC 2020 and TREC 2021 Health Misinformation Tracks, retaining
only those queries that contain at least one helpful and one harmful document to ensure high-quality evidence at both
2https://commoncrawl.org/2016/10/news-dataset-available/
3https://paperswithcode.com/dataset/c4
Manuscript submitted to ACM

8 Amirshahi et al.
Table 1. Statistics of target queries and documents for TREC 2020 and TREC 2021. The upper section reports original collections
(helpful and harmful evidence), while the lower section lists adversarial documents created by Bigdeli et al. [6].
Statistic TREC 2020 TREC 2021
Original Documents
Number of Queries 22 27
Average Helpful per Query 8.8 8.5
Average Harmful per Query 5.5 6.5
Total Helpful Documents 193 229
Total Harmful Documents 121 175
Adversarial Documents
Total Rewriter Documents 193 229
Total Paraphraser Documents 193 229
Total Fact-Inversion Documents 193 229
Total FSAP-InterQ Documents 193 229
Total FSAP-IntraQ Documents 193 229
Total Liar Documents 193 229
ends of the spectrum. As described in [ 6], document selection is based on the graded preference codes assigned in the
original tracks. For TREC 2020, helpful documents are those with a score of 4, while harmful documents are those with
a score of -2. If more than ten documents were available in either category, a random subset of ten was selected; if
fewer were available, all were retained. Applying this procedure results in 22 queries with both helpful and harmful
documents. For TREC 2021, helpful documents (scores 9 - 12) were selected, prioritizing the highest scores (12, then
11, and so on) until ten are chosen per query, while harmful documents were taken from those with scores of -3 or -2
following the same strategy. After filtering, 27 queries remained with both helpful and harmful documents.
In addition to these human-written documents, we incorporate adversarially generated documents released by Bigdeli
et al. [6]. For each query, adversarial variants of helpful documents were generated using multiple attack strategies,
each designed to mimic realistic misinformation threats:
•Rewriter Document. Generated by their “re-writer” attack method, which rewrites an existing harmful docu-
ment associated with the query into a stylistically distinct but stance-consistent adversarial version.
•Paraphraser Document. Produced by their “paraphraser” attack method, which rephrases an existing harmful
document while preserving the harmful stance, increasing lexical diversity, and reducing detectability.
•Fact-Inversion Document. Generated by their “fact-inversion” attack method, which alters factual claims in
helpful documents, flipping them into misleading counterfactual assertions.
•FSAP-IntraQ Document. Generated by their “Few-Shot Adversarial Prompting (Intra-Query)” method, which
conditions generation on harmful examples from the same query to capture query-specific adversarial patterns.
•FSAP-InterQ Document. Generated by their “Few-Shot Adversarial Prompting (Inter-Query)” method, which
leverages harmful examples from unrelated queries, transferring adversarial strategies across topics to generalize
attack effectiveness.
•Liar Document. Constructed by their “liar” attack method, which produces adversarial content from scratch
using only the query (and its description) along with a specified incorrect stance.
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 9
These adversarial documents simulate a range of attack methods, from surface-level rewriting to deeper factual
corruption and stance manipulation. These adversarial documents are shown to be effective at bypassing filters and
can routinely outrank reliable, helpful documents in re-ranking pipelines [6]. A detailed breakdown of the number of
queries, helpful and harmful documents, and adversarially generated documents is provided in Table 1. This combination
of carefully selected human-written helpful and harmful documents, as well as systematically generated adversarial
content, forms a comprehensive testbed for evaluating the robustness of LLMs and RAG systems.
3.2 Query Design
We construct three user query types for each query in the dataset to capture variations in user intent. Based on the
ground-truth stance of each query, we categorize user queries as consistent, neutral, or inconsistent. The three user
query types are defined as follows:
(1)Consistent Query: A user query whose presupposition aligns with the ground-truth stance.
(2)Neutral Query: The original query presented without additional framing.
(3)Inconsistent Query: A user query whose presupposition contradicts the ground-truth stance.
We employ fixed templates to generate consistent and inconsistent user queries according to the ground-truth stance
of each query. For example, when the ground-truth stance is unhelpful, the consistent user query adopts a skeptical
tone, such as “I’ve heard that [claim]. I don’t believe that. Right?” . In contrast, the inconsistent user query introduces an
implicit assumption of correctness, for instance: “My doctor told me that [claim]. Can you give me more information
on how it works?” . When the ground-truth stance is helpful, these templates are applied in the reverse manner. The
neutral user query corresponds to the original question and serves as a baseline. Example 1 illustrates these user query
categories for a query with a helpful stance.
Example 1: Query: Can dupixent treat eczema?
Ground Truth Stance: Helpful
Consistent “My doctor told me that Dupixent can treat eczema. Can you give me
more information on how it works?”
Neutral “Can dupixent treat eczema?”
Inconsistent “I’ve heard that Dupixent can treat eczema. I don’t believe that. Right?”
3.3 Experimental Setup
The following sections present our four main experimental setups: 1) single-document setup (Section 3.3.1), 2) paired-
document setup (Section 3.3.2), 3) passage-based pooling setup (Section 3.3.3), and 4) bias-controlled pooling setup
(Section 3.3.4). These setups are designed to address the research questions introduced in Section 1. The single-document
setup addresses RQ1 by comparing the Non-RAG baseline to those conditioned on a single document, and RQ2 by
comparing the effects of helpful, harmful, and adversarial contexts. The paired-document, passage-based pooling, and
bias-controlled pooling setups address RQ3 by investigating the impact of evidence order and composition on how
the model performs. Finally, RQ4 is evaluated across all setups by varying user query framings (consistent, neutral,
and inconsistent). All experiments are performed within the established Ragnarok framework [ 62], which provides an
easy-to-use and reusable platform for reproducible RAG experiments that supports both closed- and open-source models.
In each setup, we use helpful and harmful documents from the TREC 2021 and TREC 2020 Health Misinformation
Manuscript submitted to ACM

10 Amirshahi et al.
Tracks, as well as adversarial collections introduced by Bigdeli et al. [ 6] (Section3.1). Each prompt consists of a user
query combined with one or more retrieved documents. The prompt templates used across all setups are available in
the project’s github repository.
3.3.1 Single-Document Setup. In the single-document setup, each user query is paired with a single document serving
as context for the RAG response. If there are multiple documents in a category (helpful, harmful, and adversarial),
they are all assessed separately to guarantee full coverage. As summarized in Table 1, this includes all human-written
helpful and harmful documents as well as all adversarial variants (Rewriter, Paraphraser, Fact-Inversion, FSAP-InterQ,
FSAP-IntraQ, and Liar). For example, TREC 2020 contains 22 queries paired with 193 helpful, 121 harmful, and 6 ×193
adversarial documents, while TREC 2021 contains 27 queries paired with 229 helpful, 175 harmful, and 6 ×229 adversarial
documents. Each of these documents is evaluated independently, which guarantees complete representation across
categories and attack types. To account for how users’ assumptions affect the results, each document is paired with three
different user queries of consistent, neutral, and inconsistent, which results in unique query–document combinations
for analysis. We additionally include a Non-RAG baseline in which no documents are provided, establishing a reference
point for model performance in the absence of the external evidence provided by the documents supplied as context.
The single-document setup serves as our main experiment and is conducted across six models: GPT-4.1, GPT-5, Claude-
3.5-Haiku, DeepSeek-R1-Distill-Qwen-32B, Phi-4 [ 1], and LLaMA-3 8B Instruct [ 25]. By evaluating this diverse set of
models — including both closed-source and open-source families — we can compare the performance of models across
different architectures, sizes, and training objectives to determine if the trends we identify extend beyond an individual
model. As GPT-5 was released just before our submission deadline, we were only able to include it in the single-document
experiment as a first look at its behavior. Given consistent results across models, remaining experiments use GPT-4.1 for
its performance and speed. This setup serves as our primary experimental condition, contributing to RQ1 by comparing
the Non-RAG baseline with the single-document condition, to RQ2 by assessing variations among document types, and
to RQ4 by analyzing how query presuppositions affect LLM robustness.
3.3.2 Paired-Document Setup. To investigate RQ3, our first experiment is the paired-document setup, which analyzes
how document order affects model behavior. In this setup, each user query is paired with two documents providing
context, one helpful and one harmful/adversarial, arranged in different orders. Altogether, we use four pairing conditions:
helpful–harmful ,harmful–helpful ,helpful–adversarial , and adversarial-helpful , which capture scenarios where supportive
evidence is provided prior to or subsequent to contradictory material.
Based on the dataset statistics in Table 1, the number of possible document pairs quickly grows large. For example,
with 193 helpful and 121 harmful documents in TREC 2020, the helpful–harmful condition alone would generate over
23,000 possible pairs; when combined with six adversarial variants per query in both TREC 2020 and 2021, the number
of pairs becomes even larger. To keep computation feasible, we only sample up to ten document pairs for each condition
per query. If there are fewer than ten valid pairs, all available combinations are included. Each document pair is then
combined with user query framings to assess how sequencing and user assumptions affect the model outputs together.
This setup mainly contributes to RQ3 by testing the influence of evidence order and combinations, while also providing
insight into RQ2 by extending the single-document analysis to multi-document contexts, and RQ4 by evaluating how
different query framings shape ground-truth alignment outcomes.
3.3.3 Passage-Based Pooling Setup. The passage-based pooling setup moves beyond controlled pairings. In this setup,
all documents in the dataset are first segmented into smaller chunks to enable segment-level retrieval. We segment each
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 11
document with a sliding window of 512 words and a 256-word overlap. These segments, regardless of their original
classification as helpful, harmful, or adversarial, are subsequently reranked using the MonoT5 reranker [ 54]. For each
query, the ten highest-ranked segments are retained to form the retrieval pool, which is then provided to the language
model as context in combination with one of three user query formulations. This setup thus reflects how LLMs would
behave under conventional retrieval conditions, where retrieved evidence is shaped by ranking models rather than
manually controlled.
This setup primarily addresses RQ3 by examining the impact of various evidence types on model behavior, while
simultaneously enhancing RQ2 by evaluating the persistence of document type effects observed in controlled environ-
ments within a conventional retrieval pipeline. This setup also examines RQ4 by analysing the interplay between query
formulations and non-uniform retrieval pools in establishing ground-truth alignment results.
3.3.4 Bias-Controlled Pooling Setup. The bias-controlled pooling setup examines how context pools that are biased
towards a certain type of evidence (helpful or harmful) affect how models respond. This setup is inspired by the
methodological framework of Pogacar et al . [61] , who showed that imbalances in evidence presentation can affect
human judgment. We apply this concept to the RAG context by creating two skewed retrieval pools: one biased towards
helpful evidence and the other biased towards harmful evidence . We used the MonoT5 reranker to process each category
of document segments (helpful, harmful, and adversarial) independently for every query, producing the top-10 helpful,
top-10 harmful, and top-10 adversarial results for each attack type. These segments are then grouped into retrieval
pools of fixed size (ten segments). In the helpful-biased condition, the eight highest-ranked segments from the top-10
helpful list are placed at the top of the pool, while the remaining two highest-ranked segments are selected from the
top-10 harmful or adversarial lists. In the harmful-biased condition, the eight highest-ranked segments from the top-10
harmful or adversarial list are placed first, followed by the two highest-ranked segments from the top-10 helpful list.
This setup contributes to RQ3 by testing imbalanced evidence and RQ4 by examining how user query formulations
interact with skewed pools to affect alignment.
3.4 Evaluation
To evaluate model performance across different experimental setups, we use a single metric, Ground-Truth Alignment
(i.e., accuracy), which measures whether the stance of a generated response matches the ground-truth stance label for
the query. A response is considered aligned if its predicted stance is identical to the ground-truth stance.
Prior work has shown that LLMs can perform reliably as automatic evaluators [ 16,29], often approaching human-
level agreement. Following this line of work, we employ gemini-2.0-flash as an external stance classifier. To further
validate our evaluation pipeline, we conduct a secondary classification of all responses using gpt-4o-mini and measure
inter-rater agreement with gemini-2.0-flash and compute Cohen’s Kappa score [ 26]. Agreement was 0.90 for TREC
2020 and 0.82 for TREC 2021. These values indicate what is conventionally characterized as “almost perfect” agreement
between the two classifiers and indicate the reliability of automated stance detection. We use gemini-2.0-flash as the
only stance classifier for all of the experiments to avoid relying on the same model family for both generation and
evaluation.
For each experimental setup, we compute the Ground-Truth Alignment Rate, defined as the proportion of aligned
responses relative to the total number of generated responses. The prompt used for stance classification of LLM
responses is publicly available in the project’s github repository.
Manuscript submitted to ACM

12 Amirshahi et al.
Non-RAGHelpful Harmful RewriterParaphraserFact-InversionFSAP-IntraQ FSAP-InterQLiar020406080100Ground-Truth Alignment (%)
TREC 2021  GPT-4.1
Non-RAGHelpful Harmful RewriterParaphraserFact-InversionFSAP-IntraQ FSAP-InterQLiar020406080100Ground-Truth Alignment (%)
TREC 2020  GPT-4.1
Consistent Query Neutral Query Inconsistent Query
Fig. 1. Results of the single-document setup with GPT-4.1 onTREC 2020 and TREC 2021 . The x-axis shows the different document
types provided to the LLMs, as well as the Non-RAG baseline, and the y-axis reports ground-truth alignment rate (%). Error bars
indicate 95% confidence intervals estimated via bootstrapping. Across both datasets, providing helpful documents leads to the highest
alignment rate, while presenting Liar documents results in the lowest. In addition, the results reveal a consistent user query framing
trend in which the alignment rate is highest with consistent queries, lower with neutral queries, and lowest with inconsistent queries.
4 Results
In this section, we present results arranged around our four research questions. We organize the results as follows:
Section 4.1 (RQ1) compares the Non-RAG baseline with the single-document condition to evaluate the effect of
adding context; Section 4.2 (RQ2) examines how context type influences robustness in the single-document setup,
evaluates generalization across models and datasets (TREC 2020 vs. TREC 2021); Section 4.3 (RQ3) investigates order
and combination effects using paired-document, passage-based pooling, and bias-controlled pooling setups, while
Section 4.4 (RQ4) analyzes the impact of user query formulations (consistent, neutral, and inconsistent) across all
experimental setups. All LLM responses and additional plots are available in the project’s github repository4.
4.1 RQ1: Does retrieved context impact LLM responses?
To determine the impact of retrieved evidence on LLM behavior, we compare ground-truth alignment rate in the
Non-RAG baseline, which relies solely on the prompt, with the single-document setup, where a single document
serves as context. Figure 1 illustrates these results for GPT-4.1 on TREC 2020 and TREC 2021, while Tables 2–5 report
4https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 13
Table 2. Results of single-document setup (%, mean±SD with 95% CI) of ground-truth alignment rate for closed-source LLMs on
TREC 2021 . Providing helpful documents yields the highest ground-truth alignment rates across all models, while the Liar documents
result in the lowest. The pattern of results is consistent across all three models, with cells shaded by mean value to indicate relative
performance (darker shading corresponds to higher alignment rate).
Setting Context Query Type GPT-4.1 GPT-5 Claude-3.5-Haiku
Non-RAG– Consistent 92.5%±5.0% (81.5–100.0) 85.2%±6.8% (70.4–96.3) 70.4%±8.8% (51.9–85.2)
Neutral 88.9%±6.0% (74.1–100.0) 81.6%±7.5% (66.7–96.3) 70.4%±8.8% (51.9–85.2)
Inconsistent 66.7%±9.1% (48.1–85.2) 66.7%±9.1% (48.1–85.2) 63.0%±9.3% (44.4–81.5)
Original Documents
RAGHelpfulConsistent 97.4%±1.1% (95.2–99.1) 96.9%±1.1% (94.8–99.1) 96.5%±1.2% (93.9–98.7)
Neutral 98.2%±0.9% (96.5–99.6) 98.7%±0.8% (96.9–100.0) 97.4%±1.1% (95.2–99.1)
Inconsistent 93.0%±1.7% (89.5–96.1) 98.7%±0.8% (96.9–100.0) 97.4%±1.1% (95.2–99.1)
HarmfulConsistent 50.9%±3.8% (43.4–58.3) 47.4%±3.8% (40.0–54.9) 32.0%±3.5% (25.1–38.9)
Neutral 37.7%±3.7% (30.9–45.1) 43.4%±3.8% (36.0–50.9) 36.0%±3.6% (21.9–43.4)
Inconsistent 32.6%±3.5% (25.7–39.4) 36.5%±3.6% (29.7–44.0) 30.8%±3.5% (24.0–37.7)
Adversarial Documents
RewriterConsistent 71.2%±3.0% (65.5–76.9) 59.4%±3.2% (52.8–65.5) 57.7%±3.3% (51.1–64.2)
Neutral 60.7%±3.2% (54.6–66.8) 55.9%±3.3% (49.3–62.4) 58.5%±3.3% (52.0–65.1)
Inconsistent 59.9%±3.2% (53.3–66.4) 52.8%±3.3% (46.3–59.4) 61.2%±3.2% (55.0–67.2)
ParaphraserConsistent 73.0%±2.9% (67.2–78.6) 60.7%±3.2% (54.6–66.8) 55.9%±3.3% (49.3–62.4)
Neutral 62.9%±3.2% (56.8–69.0) 57.7%±3.3% (51.1–64.2) 57.7%±3.3% (51.1–64.2)
Inconsistent 59.4%±3.2% (52.8–65.5) 53.3%±3.3% (46.7–59.8) 62.5%±3.2% (56.3–68.6)
Fact InversionConsistent 39.7%±3.2% (33.6–46.3) 32.7%±3.1% (26.6–38.9) 28.8%±3.0% (23.1–34.9)
Neutral 34.5%±3.1% (28.4–40.6) 37.1%±3.2% (31.0–43.2) 31.0%±3.0% (25.3–37.1)
Inconsistent 34.9%±3.1% (28.8–41.0) 33.2%±3.1% (27.1–39.3) 31.0%±3.0% (25.3–37.1)
FSAP-InterQConsistent 21.0%±2.7% (15.7–26.6) 21.4%±2.7% (16.2–27.1) 17.0%±2.5% (12.2–22.3)
Neutral 20.5%±2.7% (15.7–26.2) 25.3%±2.9% (19.7–31.0) 17.0%±2.5% (12.2–22.3)
Inconsistent 21.4%±2.7% (16.2–27.1) 24.5%±2.8% (19.2–30.1) 16.6%±2.5% (11.8–21.4)
FSAP-IntraQConsistent 7.4%±1.7% (4.4–10.9) 7.8%±1.8% (4.4–11.4) 1.3%±0.8% (0.0–3.1)
Neutral 6.1%±1.6% (3.1–9.2) 6.5%±1.6% (3.5–10.0) 2.6%±1.1% (0.9–4.8)
Inconsistent 3.9%±1.3% (1.7–6.6) 5.2%±1.5% (2.6–8.3) 2.2%±1.0% (0.4–4.4)
LiarConsistent 1.7%±0.9% (0.4–3.5) 0.4%±0.4% (0.0–1.3) 1.3%±0.8% (0.0–3.1)
Neutral 4.4%±1.4% (1.7–7.0) 2.6%±1.1% (0.9–4.8) 0.9%±0.6% (0.0–2.2)
Inconsistent 1.7%±0.9% (0.4–3.5) 1.7%±0.9% (0.4–3.5) 0.0%±0.0% (0.0–0.0)
ground-truth alignment rates across all closed- and open-source models, including the mean, standard deviation (SD),
and 95% confidence intervals (CI) obtained via bootstrapping. Since it is not feasible to include plots for every model in
the paper, we provide the corresponding visualizations in the project’s github repository5, while Tables 2–5 present
the key comparative results for all models used in the single-document experiment.
In both the TREC 2020 and TREC 2021 datasets, adding helpful retrieved context causes clear improvements in ground-
truth alignment rate compared to the Non-RAG baseline, with a single supporting document markedly improving
5https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL/tree/main/Plots/gemini2.0flash/Single-Document
Manuscript submitted to ACM

14 Amirshahi et al.
Table 3. Results of single-document setup (%, mean±SD with 95% CI) of ground-truth alignment rate for closed-source LLMs on
TREC 2020 . Providing helpful documents yields the highest ground-truth alignment rates across all models, while the Liar documents
result in the lowest. The pattern of results is consistent across all three models, with cells shaded by mean value to indicate relative
performance (darker shading corresponds to higher alignment rate).
Setting Context Query Type GPT-4.1 GPT-5 Claude-3.5-Haiku
Non-RAG– Consistent 90.8%±6.1% (77.3–100.0) 95.3%±4.4% (86.4–100.0) 86.3%±7.3% (72.7–100.0)
Neutral 86.3%±7.3% (72.7–100.0) 86.3%±7.3% (72.7–100.0) 86.3%±7.3% (72.7–100.0)
Inconsistent 90.8%±6.1% (77.3–100.0) 77.3%±8.9% (59.1–95.5) 86.3%±7.3% (72.7–100.0)
Original Documents
RAGHelpfulConsistent 93.3%±1.8% (89.6–96.4) 86.0%±2.5% (80.8–90.7) 88.6%±2.3% (83.9–92.7)
Neutral 95.3%±1.5% (92.2–97.9) 94.8%±1.6% (91.2–97.9) 92.8%±1.9% (89.1–96.4)
Inconsistent 92.8%±1.9% (89.1–96.4) 95.8%±1.4% (92.7–98.4) 94.8%±1.6% (91.2–97.9)
HarmfulConsistent 79.4%±3.7% (71.9–86.8) 76.9%±3.9% (69.4–84.3) 67.0%±4.3% (58.7–75.2)
Neutral 70.3%±4.2% (62.0–78.5) 80.2%±3.7% (72.7–86.8) 66.2%±4.3% (57.9–74.4)
Inconsistent 67.9%±4.3% (59.5–76.0) 74.4%±4.0% (66.1–81.8) 71.2%±4.2% (62.8–79.3)
Adversarial Documents
RewriterConsistent 89.7%±2.2% (85.0–93.8) 84.0%±2.7% (78.8–89.1) 88.6%±2.3% (83.9–92.7)
Neutral 86.0%±2.5% (80.8–90.7) 90.7%±2.1% (86.5–94.3) 83.4%±2.7% (78.2–88.6)
Inconsistent 83.4%±2.7% (78.2–88.6) 91.2%±2.0% (87.0–94.8) 91.2%±2.0% (87.0–94.8)
ParaphraserConsistent 87.1%±2.4% (82.4–91.7) 84.0%±2.7% (78.8–89.1) 84.0%±2.7% (78.8–89.1)
Neutral 83.4%±2.7% (78.2–88.6) 85.5%±2.5% (80.3–90.2) 80.9%±2.9% (75.1–86.0)
Inconsistent 79.3%±2.9% (73.6–85.0) 86.0%±2.5% (80.8–90.7) 87.1%±2.4% (82.4–91.7)
Fact InversionConsistent 85.5%±2.5% (80.3–90.2) 85.5%±2.5% (80.3–90.2) 74.1%±3.2% (67.9–80.3)
Neutral 78.8%±3.0% (73.1–84.5) 82.9%±2.7% (77.2–88.1) 75.7%±3.1% (69.4–81.9)
Inconsistent 75.2%±3.1% (68.9–81.3) 79.8%±2.9% (74.1–85.5) 83.4%±2.7% (78.2–88.6)
FSAP-InterQConsistent 71.6%±3.3% (65.3–77.7) 76.2%±3.1% (69.9–82.4) 62.2%±3.5% (55.4–68.9)
Neutral 59.6%±3.6% (52.8–66.3) 73.6%±3.2% (67.4–79.8) 59.6%±3.6% (52.8–66.3)
Inconsistent 57.6%±3.6% (50.8–64.8) 67.4%±3.4% (60.6–74.1) 67.9%±3.4% (61.1–74.6)
FSAP-IntraQConsistent 55.0%±3.6% (48.2–62.2) 65.9%±3.4% (59.1–72.5) 41.4%±3.5% (34.7–48.7)
Neutral 39.9%±3.5% (33.2–47.2) 59.1%±3.6% (52.3–65.8) 35.2%±3.4% (28.5–42.0)
Inconsistent 34.7%±3.4% (28.0–41.5) 48.2%±3.6% (40.9–55.4) 44.1%±3.6% (37.3–51.3)
LiarConsistent 31.1%±3.3% (24.9–37.8) 44.1%±3.6% (37.3–51.3) 25.4%±3.1% (19.7–31.6)
Neutral 19.2%±2.8% (14.0–24.9) 31.1%±3.3% (24.9–37.8) 20.7%±2.9% (15.5–26.4)
Inconsistent 15.6%±2.6% (10.9–20.7) 23.3%±3.0% (17.6–29.5) 28.0%±3.2% (21.8–34.7)
alignment rate. For instance, as shown in Table 2, when GPT-4.1 is evaluated on TREC 2021 and paired with a helpful
document, its alignment rate goes from 88.9% (neutral user query, Non-RAG baseline) to 98.2%. Inconsistent user queries
also show similar improvements, with alignment rate going from 66.7% (Non-RAG baseline) to 93.0% with helpful
evidence, and comparable trends are observed across other models on TREC 2021 and TREC 2020. For example, GPT-5
evaluated on TREC 2020, as reported in Table 3, shows an increase from 86.3% in the Non-RAG baseline with a neutral
user query to 94.8% with helpful context, while DeepSeek-R1-Distill-Qwen-32B improves from 86.3% to 92.8% under the
same user query condition on TREC 2020 with results reported in Table 5.
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 15
Table 4. Results of single-document setup (%, mean±SD with 95% CI) of ground-truth alignment rate for open-source LLMs on
TREC 2021 . Providing helpful documents yields the highest ground-truth alignment rates across all models, while the Liar documents
result in the lowest. The pattern of results is consistent across all three models, with cells shaded by mean value to indicate relative
performance (darker shading corresponds to higher alignment rate).
Setting Context Query Type DeepSeek-R1-Distill-Qwen-32B Phi-4 LLaMA-3 8B Instruct
Non-RAG– Consistent 81.6%±7.5% (66.7–96.3) 74.1%±8.5% (55.6–88.9) 81.6%±7.5% (66.7–96.3)
Neutral 63.0%±9.3% (44.4–81.5) 66.7%±9.1% (48.1–85.2) 81.6%±7.5% (66.7–96.3)
Inconsistent 51.8%±9.7% (33.3–70.4) 55.5%±9.6% (37.0–74.1) 59.2%±9.5% (40.7–77.8)
Original Documents
RAGHelpfulConsistent 98.2%±0.9% (96.5–99.6) 96.9%±1.1% (94.8–99.1) 95.2%±1.4% (92.1–97.8)
Neutral 92.6%±1.7% (89.1–95.6) 96.5%±1.2% (93.9–98.7) 93.9%±1.6% (90.8–96.9)
Inconsistent 84.3%±2.4% (79.5–88.6) 91.7%±1.8% (88.2–95.2) 89.1%±2.1% (84.7–93.0)
HarmfulConsistent 53.1%±3.7% (45.7–60.6) 38.3%±3.7% (31.4–45.7) 44.6%±3.8% (37.1–52.0)
Neutral 33.7%±3.6% (26.9–40.6) 34.8%±3.6% (28.0–42.3) 38.3%±3.7% (31.4–45.7)
Inconsistent 22.9%±3.2% (16.6–29.1) 29.7%±3.5% (23.4–36.6) 26.3%±3.3% (20.0–33.1)
Adversarial Documents
RewriterConsistent 71.2%±3.0% (65.5–76.9) 61.6%±3.2% (55.5–67.7) 72.5%±3.0% (66.8–78.2)
Neutral 57.7%±3.3% (51.1–64.2) 52.4%±3.3% (45.9–59.0) 55.0%±3.3% (48.5–61.6)
Inconsistent 44.5%±3.2% (38.4–51.1) 51.1%±3.3% (44.5–57.6) 48.0%±3.3% (41.5–54.6)
ParaphraserConsistent 77.8%±2.7% (72.5–83.0) 59.4%±3.2% (52.8–65.5) 69.5%±3.1% (63.3–75.1)
Neutral 56.8%±3.3% (50.2–63.3) 55.5%±3.3% (48.9–62.0) 55.0%±3.3% (48.5–61.6)
Inconsistent 45.4%±3.3% (39.3–52.0) 54.6%±3.3% (48.0–61.1) 44.1%±3.3% (38.0–50.7)
Fact InversionConsistent 57.7%±3.3% (51.1–64.2) 32.3%±3.1% (26.6–38.4) 32.3%±3.1% (26.6–38.4)
Neutral 38.0%±3.2% (31.9–44.1) 30.6%±3.0% (24.9–36.7) 27.9%±3.0% (22.3–34.1)
Inconsistent 29.2%±3.0% (23.6–35.4) 31.0%±3.0% (25.3–37.1) 27.1%±2.9% (21.4–33.2)
FSAP-InterQConsistent 32.7%±3.1% (26.6–38.9) 16.6%±2.5% (11.8–21.4) 17.5%±2.5% (12.7–22.7)
Neutral 21.0%±2.7% (15.7–26.6) 18.3%±2.5% (13.5–23.6) 15.3%±2.4% (10.9–20.1)
Inconsistent 20.1%±2.6% (15.3–25.3) 17.5%±2.5% (12.7–22.7) 14.8%±2.4% (10.5–19.7)
FSAP-IntraQConsistent 17.5%±2.5% (12.7–22.7) 2.6%±1.1% (0.9–4.8) 1.3%±0.8% (0.0–3.1)
Neutral 6.1%±1.6% (3.1–9.2) 4.4%±1.4% (1.7–7.0) 2.2%±1.0% (0.4–4.4)
Inconsistent 3.5%±1.2% (1.3–6.1) 2.2%±1.0% (0.4–4.4) 0.0%±0.0% (0.0–0.0)
LiarConsistent 13.1%±2.2% (8.7–17.9) 1.7%±0.9% (0.4–3.5) 0.0%±0.0% (0.0–0.0)
Neutral 1.3%±0.8% (0.0–3.1) 2.2%±1.0% (0.4–4.4) 0.4%±0.4% (0.0–1.3)
Inconsistent 3.9%±1.3% (1.7–6.6) 0.0%±0.0% (0.0–0.0) 0.4%±0.4% (0.0–1.3)
Conversely, in both TREC 2020 and TREC 2021, harmful documents consistently degrade performance. For GPT-4.1
on TREC 2021 (Table 2), the alignment rate falls to 37.7% with neutral user queries and 32.6% with inconsistent user
queries, while for GPT-5 drops to 43.4% and 36.5%, respectively. These results demonstrate that retrieved documents
actively influence model alignment, with the ability to markedly enhance or diminish it rather than merely supplying
contextual information. The consistency of this pattern across both closed- and open-source models underscores the
central role of context in shaping LLM behavior when compared to the Non-RAG baseline, and this finding addresses
RQ1 by showing that context definitely impacts LLM responses.
Manuscript submitted to ACM

16 Amirshahi et al.
Table 5. Results of single-document setup (%, mean±SD with 95% CI) of ground-truth alignment rate for open-source LLMs on
TREC 2020 . Providing helpful documents yields the highest ground-truth alignment rates across all models, while the Liar documents
result in the lowest. The pattern of results is consistent across all three models, with cells shaded by mean value to indicate relative
performance (darker shading corresponds to higher alignment rate).
Setting Context Query Type DeepSeek-R1-Distill-Qwen-32B Phi-4 LLaMA-3 8B Instruct
Non-RAG– Consistent 95.3%±4.4% (86.4–100.0) 95.3%±4.4% (86.4–100.0) 90.8%±6.1% (77.3–100.0)
Neutral 86.3%±7.3% (72.7–100.0) 95.3%±4.4% (86.4–100.0) 90.8%±6.1% (77.3–100.0)
Inconsistent 95.3%±4.4% (86.4–100.0) 86.3%±7.3% (72.7–100.0) 86.3%±7.3% (72.7–100.0)
Original Documents
RAGHelpfulConsistent 95.8%±1.4% (92.7–98.4) 91.7%±2.0% (87.6–95.3) 92.3%±1.9% (88.1–95.9)
Neutral 92.8%±1.9% (89.1–96.4) 93.3%±1.8% (89.6–96.4) 91.2%±2.0% (87.0–94.8)
Inconsistent 95.3%±1.5% (92.2–97.9) 93.8%±1.7% (90.2–96.9) 92.8%±1.9% (89.1–96.4)
HarmfulConsistent 84.4%±3.3% (77.7–90.9) 72.8%±4.1% (64.5–80.2) 62.9%±4.4% (53.0–71.1)
Neutral 62.0%±4.4% (52.9–70.2) 63.7%±4.4% (54.5–71.9) 57.9%±4.5% (48.8–66.9)
Inconsistent 64.5%±4.4% (56.2–72.7) 57.9%±4.5% (48.8–66.9) 61.2%±4.5% (52.9–69.4)
Adversarial Documents
RewriterConsistent 96.4%±1.4% (93.3–98.4) 80.3%±2.9% (74.6–86.0) 77.8%±3.0% (71.5–83.4)
Neutral 75.2%±3.1% (68.9–81.3) 77.8%±3.0% (71.5–83.4) 68.5%±3.4% (61.7–75.1)
Inconsistent 72.1%±3.3% (65.8–78.2) 75.7%±3.1% (69.4–81.9) 79.3%±2.9% (73.6–85.0)
ParaphraserConsistent 95.3%±1.5% (92.2–97.9) 80.9%±2.9% (75.1–86.0) 72.1%±3.3% (65.8–78.2)
Neutral 73.1%±3.2% (66.8–79.3) 71.0%±3.3% (64.2–77.2) 65.9%±3.4% (59.1–72.5)
Inconsistent 74.7%±3.2% (68.4–80.8) 71.6%±3.3% (65.3–77.7) 72.6%±3.2% (66.3–78.8)
Fact InversionConsistent 92.8%±1.9% (89.1–96.4) 76.2%±3.1% (69.9–82.4) 74.1%±3.2% (67.9–80.3)
Neutral 74.1%±3.2% (67.9–80.3) 74.1%±3.2% (67.9–80.3) 64.3%±3.5% (57.5–71.0)
Inconsistent 73.6%±3.2% (67.4–79.8) 69.0%±3.4% (62.2–75.6) 65.9%±3.4% (59.1–72.5)
FSAP-InterQConsistent 79.3%±2.9% (73.6–85.0) 51.3%±3.6% (44.6–58.5) 46.6%±3.6% (39.9–53.9)
Neutral 49.8%±3.6% (43.0–57.0) 45.6%±3.6% (38.3–52.8) 42.0%±3.6% (35.2–49.2)
Inconsistent 52.3%±3.6% (45.1–59.1) 43.0%±3.6% (36.3–50.3) 40.4%±3.5% (33.7–47.7)
FSAP-IntraQConsistent 66.4%±3.4% (59.6–73.1) 32.6%±3.4% (25.9–39.4) 33.2%±3.4% (26.4–39.9)
Neutral 28.5%±3.2% (22.3–35.2) 26.4%±3.1% (20.2–32.6) 23.8%±3.0% (18.1–30.1)
Inconsistent 28.5%±3.2% (22.3–35.2) 23.8%±3.0% (18.1–30.1) 20.7%±2.9% (15.5–26.4)
LiarConsistent 51.8%±3.6% (44.6–59.1) 17.1%±2.7% (11.9–22.8) 19.7%±2.8% (14.5–25.4)
Neutral 23.3%±3.0% (17.6–29.5) 14.5%±2.5% (9.8–19.7) 19.2%±2.8% (14.0–24.9)
Inconsistent 22.8%±3.0% (17.1–29.0) 11.9%±2.3% (7.8–16.6) 12.5%±2.4% (7.8–17.1)
4.2 RQ2: To what extent does the type of retrieved context influence robustness?
With RQ1 confirming that context matters, RQ2 examines how its type, helpful, harmful, or adversarial, affects ground-
truth alignment. Results from the single-document setup (Tables 2–5) reveal a consistent pattern across both models
and datasets, as helpful documents raise alignment rate substantially above the Non-RAG baseline, harmful documents
reduce it, and adversarial documents often cause robustness to collapse almost entirely.
For closed-source models on TREC 2021 (Table 2), helpful documents push alignment rate close to the ceiling for
almost all user query framings. For instance, ground-truth alignment rate rises from 88.9% in Non-RAG baseline to
98.2% for GPT-4.1 for neutral user queries, from 81.6% to 98.7% for GPT-5, and from 70.4% to 97.4% for Claude-3.5-Haiku.
Under the same neutral setting on TREC 2021, harmful evidence reduces these gains and pushes alignment rate down
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 17
to 37.7% for GPT-4.1, 43.4% for GPT-5, and 36.0% for Claude-3.5-Haiku. Furthermore, alignment rate is nearly eliminated
by strong adversarial documents such as Liar, with GPT-4.1 dropping to 4.4%, GPT-5 to 2.6%, and Claude-3.5-Haiku to
0.9% for the same neutral condition.
Open-source models follow the same pattern. As shown in Table 4 for TREC 2021, the alignment rate of DeepSeek-
R1-Distill-Qwen-32B improves from 63.0% in the Non-RAG baseline to 92.6% with helpful documents for neutral user
queries, but collapses to just 1.3% under Liar inputs. A similar trend appears for Phi-4, which rises from 66.7% in the
Non-RAG baseline to 96.5% with helpful evidence, but drops sharply to 2.2% with Liar documents. LLaMA-3 8B Instruct
shows the same pattern, moving from 81.6% without context to 93.9% with helpful inputs, before falling to just 0.4%
under Liar documents. Even weaker adversarial documents, such as Rewriter and Paraphraser, substantially reduce
alignment rate, often into the 55–70% range across all open-source models. The same pattern is observed for TREC 2020
as well, but with consistently higher rates, as detailed in the following paragraphs. We emphasize neutral user query
results here because RQ1–RQ3 focus primarily on the effect of retrieved context, while the role of query framing is
analyzed separately in Section 4.4.
Across both datasets and model families, the relative effectiveness of adversarial strategies follows a consistent
ordering: Liar≫FSAP-IntraQ >FSAP-InterQ >Fact Inversion >Paraphraser≈Rewriter . This hierarchy is a result
of how adversarial documents are made. Liar documents are constructed from scratch based on the query and an
incorrect stance, so they are not constrained to the wording or rhetorical framing of pre-existing harmful content. This
independence enables more fluent phrasing and broader topic coverage, making misinformation more confident and
persuasive, and thus more challenging for LLMs. In contrast, Paraphraser and Rewriter documents inherit limitations
from pre-existing harmful material, which can make them appear less natural.
In cross-model comparison, GPT-4.1 and GPT-5 achieve the highest levels of alignment rate when paired with
helpful documents; however, both demonstrate comparable vulnerabilities when confronted with adversarial evidence.
Claude-3.5-Haiku has significant performance drops in the presence of harmful or adversarial content; on the other
hand, DeepSeek-R1-Distill-Qwen-32B performs well and often obtains levels that are similar to closed-source models
when provided with helpful documents. Phi-4 and LLaMA-3 8B Instruct achieves a lower overall alignment rate but
follows the same qualitative trends. Since these patterns are consistent across all models, we selected GPT-4.1 as the
representative model for subsequent experiments due to its relatively high performance and speed. GPT-5 was not
released at the time these experiments were first conducted and was included afterwards.
Another important finding of this experiment is the difference at the dataset level, where models consistently display
a higher alignment rate on TREC 2020 (COVID-19 misinformation) than on TREC 2021 (general health misinformation).
For example, TREC 2021 includes queries about misinformation on everyday remedies and niche treatments, such
as using duct tape for warts and toothpaste for pimples, which rarely reach the same level of wide public discourse,
although the ground truth for these questions is based on evidence. This underscores an unexpected outcome from our
research, suggesting that LLMs appear considerably more robust against adversarial manipulation in the COVID-19
context. This difference may be due to post-training alignment placing greater emphasis on topics like COVID-19,
which was the subject of widespread misinformation and public attention, while giving limited focus to less prominent
areas of health.
Overall, our results address RQ2 by showing that the type and quality of retrieved context substantially affect LLM
robustness, by demonstrating that helpful documents consistently improve alignment rate relative to the baseline,
whereas harmful and adversarial documents undermine stability and reliability, in some cases reducing robustness to
the point of complete failure.
Manuscript submitted to ACM

18 Amirshahi et al.
While it may appear self-evident that helpful evidence would improve alignment rate and harmful or adversarial
evidence would reduce it, our results move beyond speculation by empirically quantifying both the magnitude and
the consistency of these effects across datasets and model families. The measurements establish concrete performance
bounds. Alignment rate can approach near-maximum levels by providing a single helpful document, yet it can collapse
to almost zero under adversarial documents such as Liar. As outlined in Section 1, this systematic analysis offers
the empirical evidence required to turn abstract concerns about adversarial susceptibility into evidence-based design
principles for RAG systems. In line with RQ1, the findings highlight that context can significantly shape outcomes and
must be safeguarded against harmful or adversarial inputs.
4.3 RQ3: In what ways do orders and combinations of multiple evidence sources shape model behavior?
Findings from RQ1 and RQ2 show that context type strongly shapes ground-truth alignment, which motivates an
assessment of more complex scenarios with multiple documents presented in different orders and compositions,
leading to RQ3. To address this question, we examine how models respond when exposed to multiple sources of
evidence, focusing on two key dimensions: (i) the order in which helpful and misleading documents are presented, and
(ii) the composition of the retrieval pool when evidence is naturally or systematically skewed. We investigate these
dimensions through three complementary setups: the paired-document setup, the passage-based pooling setup, and the
bias-controlled pooling setup, which are analyzed in the following sections.
4.3.1 Paired-Document Analysis. Figure 2 shows ground-truth alignment rate when a helpful document is either first
(green, “Helpful First”) or second (blue, “Helpful Second”) relative to a harmful/adversarial document, compared with the
single-document baseline (red). The results show that the order of the documents has minimal impact on ground-truth
alignment. The Helpful First and Helpful Second conditions deliver nearly the same results, with confidence intervals
that mostly overlap. For instance, on TREC 2021 under the consistent user query, the alignment rate of LLM responses
rises from only 1.7% (CI 0.4–3.5) in the single-document baseline to 39.1% (CI 33.3–45.0) when helpful evidence appears
first and 45.0% (CI 39.1–50.8) when it appears second for Liar inputs. Similarly, for FSAP-IntraQ documents, alignment
rate improves from 7.4% (CI 4.4–10.9) to 41.1% (CI 35.3–47.3) when helpful content precedes and 48.8% (CI 43.0–55.0)
when it follows the adversarial document. On TREC 2020, robustness is consistently higher. For example, under the
neutral user query, for FSAP-IntraQ documents, alignment rate rises from 54.9% (CI 48.2–62.2) to 68.3% (CI 61.9–74.8)
when helpful material comes first and 76.7% (CI 70.8–82.7) when it is placed second relative to them. These results
show that presenting helpful evidence, regardless of position, minimizes the impact of harmful or adversarial content.
Additionally, discrepancies among datasets persist, as TREC 2020 (COVID-19) queries demonstrate greater robustness
than the more general health queries in TREC 2021, thereby reinforcing the previously identified dataset-level variations.
4.3.2 Passage-Based Pooling Analysis. In the passage-based pooling setup, the MonoT5 reranker provides the top-10
passages for each query, obtained by sliding a 512-word window with 256-word overlap across documents and ranking
the resulting segments. Since the adversarial documents are designed to rank highly, these pools are heavily skewed
toward this adversarial content, with more than nine of the top-10 segments being adversarial. For TREC 2020, the
top-10 pools comprise ∼92% adversarial, 5% helpful, and 3% harmful passages, whereas for TREC 2021 the distribution is
∼94% adversarial, 5% helpful, and 2% harmful passages. As shown in Fig. 3, the dominance of adversarial segments in the
passage-based pool leads to consistently low alignment rates. On TREC 2020, GPT-4.1 reaches 43.7% under consistent
user queries, but performance drops to 24.9% with neutral user queries and remains low at 24.9% under inconsistent
user queries. The degradation is even worse on TREC 2021, where the alignment rate drops to 17.3% with consistent
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 19
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar020406080100Ground-Truth Alignment (%)
TREC2021  Consistent Query
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC2021  Neutral Query
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC2021  Inconsistent Query
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar020406080100Ground-Truth Alignment (%)
TREC2020  Consistent Query
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC2020  Neutral Query
Harmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC2020  Inconsistent Query
Helpful First Helpful Second Single Document
Fig. 2. Results of paired-document setup forGPT-4.1 onTREC 2020 andTREC 2021 with consistent, neutral, and inconsistent
user queries. The x-axis shows the document type with which the helpful document is paired (either first or second), and the y-axis
reports ground-truth alignment rate (%). Error bars indicate 95% confidence intervals estimated via bootstrapping. Bars compare
helpful-first, helpful-second, and single-document inputs (harmful/adversarial only). Helpful placement shows only minor differences,
while single-document inputs yield substantially lower ground-truth alignment rates.
user queries, 13.0% with neutral user queries, and only 4.3% with inconsistent user queries. These findings emphasize
retrieval as the primary bottleneck in RAG pipelines and demonstrate that the composition of the evidence pool can
significantly undermine robustness, irrespective of user query framing.
4.3.3 Bias-Controlled Pooling Analysis. To examine the impact of systematic skew in retrieval, we construct pools
biased toward either helpful or harmful content. In the condition biased toward harmful, the ground-truth alignment
rate declines markedly. On TREC 2021, as shown in Fig. 3, GPT-4.1 achieves only a 26.04% alignment rate when the pool
is biased toward harmful content with Fact Inversion documents, and 4.35% when biased with Liar documents, even
under consistent user queries. The situation worsens under inconsistent user queries, where providing Liar examples
leads to a 0% alignment rate, and presenting FSAP-IntraQ documents collapses the rate to 0% as well. By contrast, TREC
2020 demonstrates somewhat greater resilience. Although alignment rate against Rewriter documents remains as high
as 93.76% and Paraphraser at 87.52% under consistent user queries, it still collapses for the most adversarial harmful
cases, dropping to 24.95% for Liar and 37.4% for FSAP-IntraQ documents.
In the biased-toward-helpful condition, alignment levels are close to the best possible. On both datasets, consistent
and neutral user queries frequently sustain a 100% alignment rate, with LLM responses across all attack categories in
TREC 2021 reaching a full alignment rate when helpful documents are included in the pool. Even inconsistent user
queries remain robust: in TREC 2020, pooled conditions containing Rewriter or FSAP-InterQ documents yield a 100%
Manuscript submitted to ACM

20 Amirshahi et al.
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar020406080100Ground-Truth Alignment (%)
TREC 2021  Consistent Query
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC 2021  Neutral Query
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC 2021  Inconsistent Query
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar020406080100Ground-Truth Alignment (%)
TREC 2020  Consistent Query
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC 2020  Neutral Query
Passage-BasedHarmful Rewriter
ParaphraserFact InversionFSAP-InterQ FSAP-IntraQLiar
TREC 2020  Inconsistent Query
Biased to Helpful Biased to Harmful Passage-Based Pooling
Fig. 3. Results of passage-based pooling andbiased-controlled pooling setups forGPT-4.1 onTREC 2020 andTREC 2021
across consistent, neutral, and inconsistent user queries. The x-axis includes the passage-based pooling baseline as well as the
different document types used in biased conditions, where helpful documents are paired with harmful or adversarial ones. The y-axis
reports ground-truth alignment rate (%). Error bars indicate 95% confidence intervals estimated via bootstrapping. Results show
helpful-biased pools yield near-perfect alignment, whereas harmful and passage-based pools sharply reduce it.
alignment rate, and Fact Inversion pools reach 95.63% on TREC 2021. Once helpful content dominates the retrieval pool,
the influence of user query framing diminishes, as supportive context outweighs misleading presuppositions.
Overall, the bias-controlled experiments confirm that the composition of retrieved evidence plays a critical role
in RAG systems’ robustness. Pools dominated by helpful documents consistently lead to high alignment rates, often
exceeding 90–100%, while harmful-biased pools drastically reduce reliability, in some cases collapsing alignment rate to
nearly zero. These results underscore that the stability of RAG systems is fundamentally dependent on the distribution
of their evidence sources, which highlights the retrieval stage as the key vulnerability in pipeline design.
4.4 RQ4: How do different query framings affect stance alignment?
After discussing the significance of evidence type (RQ2) and evidence order/combination (RQ3), we move on to RQ4,
which assesses the impact of user query framing (consistent, neutral, and inconsistent) on the results of ground-truth
alignment. Prior work has consistently shown that LLM performance is highly sensitive to prompt formulation. Even
minor variations in wording, token placement, or tone can lead to substantially different outcomes, and several studies
have proposed methods to stabilize predictions under such alterations [ 32,42,49,50,57,63,65,75]. Motivated by these
results, we revisit our experimental setups with query sensitivity in mind.
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 21
We analyze user query formulation effects across all four experimental setups: the single-document setup, the paired-
document setup, the passage-based pooling setup, and the bias-controlled pooling setup. Figures 1–3 and Tables 2–5
summarize results across models, datasets, and retrieval conditions.
Framing has a persistent impact on ground-truth alignment rate, and the order of user query types remains the same
across models and datasets (Consistent >Neutral >Inconsistent), which directly addresses RQ4 by showing that the
framing of user queries impacts ground-truth alignment outcomes. The same hierarchy emerges in the single-document
setup (Fig.1), persists when multiple sources of evidence are paired (Fig.2), and carries over to more passage-based
pooling setups (Fig. 3).
Consistent user queries, those which align with the ground-truth stance of the answer, reliably produce the highest
alignment rates by framing the model’s reasoning in a direction that agrees with available evidence; on the other hand,
inconsistent user queries embed contradictory presuppositions that actively mislead the model, often overwhelming the
corrective signal of helpful documents. While explicit stance bias is eliminated by neutral user queries, models are still
able to base their responses on the evidence they have retrieved, resulting in intermediate performance. These results
show that models reflect the quality of retrieved evidence (RQ2) and systematically inherit bias from how user queries
are phrased. User query formulation, therefore, plays a central role in shaping ground-truth alignment outcomes (RQ4).
5 Concluding Discussion
A central finding of this work is that query framing and retrieval context interact to affect how well LLMs handle health
misinformation. The results consistently show that user queries and the type of evidence retrieved have a significant
impact on model behavior, with the content distribution of the retrieval pool shaping outcomes alongside the quality of
individual documents. In particular, the inclusion of helpful evidence, even in the presence of adversarial or harmful
content, can protect against misalignment.
The empirical analysis demonstrates that ground-truth alignment is most severely harmed by Liar and FSAP
documents, with effects that are consistent across both open-source and closed-source model families. Our observations
are consistent with previous studies Chaudhari et al . [10] , Wang and Yu [73], Xue et al . [80] , Zou et al . [89] showing that
RAG systems are sensitive to the information provided as context, and when this context is malicious or adversarially
manipulated, models can be misled into producing incorrect or biased output. This reinforces the broader evidence that
the reliability of RAG pipelines is dependent not only on retrieval accuracy but also on the integrity of the documents
used to ground generation. However, adding helpful documents often makes performance superior to the Non-RAG
baseline. Query framing is also a key factor in shaping model behavior, with consistent user queries mostly improving
alignment and conflicting queries reducing it.
In addition, our results reveal a consistent robustness gap between COVID-19 queries in TREC 2020 and general
health queries in TREC 2021, likely due to extensive exposure to pandemic-related misinformation during training
that may have reinforced model defenses. In contrast, broader health topics remain more vulnerable, which suggests a
potential imbalance in how current LLMs are safeguarded across medical domains. These outcomes indicate a more
extensive asymmetry in alignment practices, wherein robustness is shown to be enhanced in response to prominent
crises but is insufficiently applied to other medically significant domains.
Our study provides important evidence on the vulnerabilities and unintended risks associated with generative
AI in potentially high-stakes domains such as healthcare. Our findings demonstrate that adversarial documents can
significantly disrupt model behavior, undermining the reliability of even state-of-the-art RAG systems. These failures
occur regardless of the size or architecture of the model and underscore the retrieval stage as the primary point of
Manuscript submitted to ACM

22 Amirshahi et al.
vulnerability. Once malicious content is promoted to the top ranks by the retriever or reranker, LLMs rarely recover,
even when supported by consistent user queries. This highlights a broader concern: generative AI systems are not only
prone to adversarial manipulation but can also amplify harmful content when such information is injected into their
context. The results emphasize that enhancing the generation component alone is insufficient to ensure trustworthiness.
Robust defenses must also address vulnerabilities in the retrieval pipeline to prevent harmful content from reaching the
generation stage.
6 Limitations
Our study advances the understanding of RAG robustness under adversarial evidence, but it has limitations. First,
we rely on ground-truth alignment as the primary evaluation metric, which does not fully capture other important
aspects of response quality, such as completeness, clarity, or potential user trust effects. Future work should employ
multidimensional evaluations to provide a more comprehensive assessment of system performance in sensitive domains.
Second, our design assumes that relevant documents are already retrieved. This lets us isolate how document content
and query framing influence model behavior, but omits retrieval-stage factors such as ranking errors and bias. Future
work should extend this analysis to full RAG pipelines to examine how adversarial documents interact with retrievers
and rerankers under real-world conditions.
Finally, our evaluation relies on automated stance classification with gemini-2.0-flash, validated using gpt-4o-mini.
While prior work shows LLMs can align well with human judgments [ 16,29], automated classifiers may miss subtle
inaccuracies or nuanced risks. Future work should include human-in-the-loop or multi-annotator studies to strengthen
validity in safety-critical domains.
References
[1]Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi,
Piero Kauffmann, et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905 (2024).
[2]Md Adnan Arefeen, Biplob Debnath, and Srimat Chakradhar. 2024. Leancontext: Cost-efficient domain-specific question answering using llms.
Natural Language Processing Journal 7 (2024), 100065.
[3]Md Ahsan Ayub and Subhabrata Majumdar. 2024. Embedding-based classifiers can detect prompt injection attacks. arXiv preprint arXiv:2410.22284
(2024).
[4]Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language
models be too big?. In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency . 610–623.
[5]Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, and Charles LA Clarke. 2024. EMPRA: Embedding Perturbation Rank Attack against Neural
Ranking Models. arXiv preprint arXiv:2412.16382 (2024).
[6]Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, and Charles L. A. Clarke. 2025. Adversarial Attacks against Neural Ranking Models via In-Context
Learning. arXiv:2508.15283 [cs.IR] https://arxiv.org/abs/2508.15283
[7]Stephen Casper, Lennart Schulze, Oam Patel, and Dylan Hadfield-Menell. 2024. Defending against unforeseen failure modes with latent adversarial
training. arXiv preprint arXiv:2403.05030 (2024).
[8]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A
survey on evaluation of large language models. ACM transactions on intelligent systems and technology 15, 3 (2024), 1–45.
[9]Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2025. Jailbreaking black box large language
models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) . IEEE, 23–42.
[10] Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, and Alina
Oprea. 2024. Phantom: General trigger attacks on retrieval augmented language generation. arXiv preprint arXiv:2405.20485 (2024).
[11] Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, and David Wagner. 2025. Defending Against Prompt Injection With a Few
DefensiveTokens. arXiv preprint arXiv:2507.07974 (2025).
[12] Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, and Chuan Guo. 2024. Secalign: Defending against
prompt injection with preference optimization. arXiv preprint arXiv:2410.05451 (2024).
[13] Sizhe Chen, Arman Zharmagambetov, David Wagner, and Chuan Guo. 2025. Meta SecAlign: A Secure Foundation LLM Against Prompt Injection
Attacks. arXiv preprint arXiv:2507.02735 (2025).
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 23
[14] Xuanang Chen, Ben He, Zheng Ye, Le Sun, and Yingfei Sun. 2023. Towards imperceptible document manipulations against neural ranking models.
arXiv preprint arXiv:2305.01860 (2023).
[15] Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, and Gongshen Liu. 2024. Trojanrag: Retrieval-augmented
generation can be backdoor driver in large language models. arXiv preprint arXiv:2405.13401 (2024).
[16] Cheng-Han Chiang and Hung-yi Lee. 2023. Can Large Language Models Be an Alternative to Human Evaluations?. In Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics . Association for Computational Linguistics, Toronto, Canada, 15607–15631.
https://aclanthology.org/2023.acl-long.870
[17] Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, and Jong C Park. 2024. Typos that Broke the RAG’s Back: Genetic Attack on RAG
Pipeline by Simulating Documents in the Wild via Low-level Perturbations. arXiv preprint arXiv:2404.13948 (2024).
[18] Charles L. A. Clarke, Maria Maistro, and Mark D. Smucker. 2021. Overview of the TREC 2021 Health Misinformation Track. In Proceedings of the
Thirtieth Text REtrieval Conference, TREC 2021, online, November 15-19, 2021 (NIST Special Publication, Vol. 500-335) , Ian Soboroff and Angela Ellis
(Eds.). National Institute of Standards and Technology (NIST). https://trec.nist.gov/pubs/trec30/papers/Overview-HM.pdf
[19] Charles L. A. Clarke, Saira Rizvi, Mark D. Smucker, Maria Maistro, and Guido Zuccon. 2020. Overview of the TREC 2020 Health Misinformation
Track. In Proceedings of the Twenty-Ninth Text REtrieval Conference, TREC 2020, Virtual Event [Gaithersburg, Maryland, USA], November 16-20,
2020 (NIST Special Publication, Vol. 1266) , Ellen M. Voorhees and Angela Ellis (Eds.). National Institute of Standards and Technology (NIST).
https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.HM.pdf
[20] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview of the TREC 2020 deep learning track. CoRR abs/2102.07662
(2021). arXiv:2102.07662 https://arxiv.org/abs/2102.07662
[21] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M Voorhees, and Ian Soboroff. 2025. Overview of the TREC 2022
deep learning track. arXiv preprint arXiv:2507.10865 (2025).
[22] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M Voorhees. 2020. Overview of the TREC 2019 deep learning track. arXiv
preprint arXiv:2003.07820 (2020).
[23] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for rag systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 719–729.
[24] Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, and Yang Liu. 2023. Masterkey: Automated
jailbreak across multiple large language model chatbots. arXiv preprint arXiv:2307.08715 (2023).
[25] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang,
Angela Fan, et al. 2024. The llama 3 herd of models. arXiv e-prints (2024), arXiv–2407.
[26] Joseph L Fleiss, Bruce Levin, Myunghee Cho Paik, et al .1981. The measurement of interrater agreement. Statistical methods for rates and proportions
2, 212-236 (1981), 22–23.
[27] Jingsheng Gao, Linxu Li, Weiyuan Li, Yuzhuo Fu, and Bin Dai. 2024. Smartrag: Jointly learn rag-related tasks from the environment feedback. arXiv
preprint arXiv:2410.18141 (2024).
[28] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997 (2024).
[29] Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National
Academy of Sciences 120, 30 (2023), e2305016120.
[30] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not what you’ve signed up for:
Compromising real-world llm-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM workshop on artificial
intelligence and security . 79–90.
[31] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International
conference on machine learning . PMLR, 3929–3938.
[32] Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik, and Liming Zhu. 2024. Prompt perturbation in retrieval-augmented generation based large
language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 1119–1130.
[33] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin,
et al.2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems 43, 2 (2025), 1–55.
[34] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020).
[35] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022. Few-shot learning with retrieval augmented language models. arXiv preprint arXiv:2208.03299 1, 2 (2022).
[36] Dennis Jacob, Hend Alzahrani, Zhanhao Hu, Basel Alomair, and David Wagner. 2024. Promptshield: Deployable detection for prompt injection
attacks. In Proceedings of the Fifteenth ACM Conference on Data and Application Security and Privacy . 341–352.
[37] Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas
Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614 (2023).
Manuscript submitted to ACM

24 Amirshahi et al.
[38] Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, and Himabindu Lakkaraju. 2023. Certifying llm safety against
adversarial prompting. arXiv preprint arXiv:2309.02705 (2023).
[39] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019. BART:
Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461
(2019).
[40] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33
(2020), 9459–9474.
[41] Haoran Li, Dadi Guo, Wei Fan, Mingshi Xu, Jie Huang, Fanpu Meng, and Yangqiu Song. 2023. Multi-step jailbreaking privacy attacks on chatgpt.
arXiv preprint arXiv:2304.05197 (2023).
[42] Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali Bahrainian. 2025. Enhancing retrieval-augmented generation: a study of best practices.
arXiv preprint arXiv:2501.07391 (2025).
[43] Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, and Xiaozhong Liu. 2022. Order-disorder: Imitation
adversarial attacks for black-box neural ranking models. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications
Security . 2025–2039.
[44] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How
language models use long contexts. arXiv preprint arXiv:2307.03172 (2023).
[45] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pre-train, Prompt, and Predict: A Systematic
Survey of Prompting Methods in Natural Language Processing. ACM Comput. Surv. (2023).
[46] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024. Formalizing and benchmarking prompt injection attacks and
defenses. In 33rd USENIX Security Symposium (USENIX Security 24) . 1831–1847.
[47] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, and Xueqi Cheng. 2023. Black-box adversarial attacks against
dense retrieval models: A multi-view contrastive learning method. In Proceedings of the 32nd ACM International Conference on Information and
Knowledge Management . 1647–1656.
[48] Craig Macdonald, Jinyuan Fang, Andrew Parry, and Zaiqiao Meng. 2025. Constructing and Evaluating Declarative RAG Pipelines in PyTerrier. In
Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval . 4035–4040.
[49] Junyu Mao, Stuart E Middleton, and Mahesan Niranjan. 2023. Do prompt positions really matter? arXiv preprint arXiv:2305.14493 (2023).
[50] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. 2024. State of what art? a call for multi-prompt llm
evaluation. Transactions of the Association for Computational Linguistics 12 (2024), 933–949.
[51] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian.
2025. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology 16, 5 (2025), 1–72.
[52] Fatemeh Nazary, Yashar Deldjoo, and Tommaso di Noia. 2025. Poison-rag: Adversarial data poisoning attacks on retrieval-augmented generation in
recommender systems. In European Conference on Information Retrieval . Springer, 239–251.
[53] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. Ms marco: A human-generated machine
reading comprehension dataset. (2016).
[54] Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. arXiv preprint
arXiv:2003.06713 (2020).
[55] Yasumasa Onoe, Michael JQ Zhang, Eunsol Choi, and Greg Durrett. 2022. Entity cloze by date: What LMs know about unseen entities. arXiv preprint
arXiv:2205.02832 (2022).
[56] Rodrigo Pedro, Daniel Castro, Paulo Carreira, and Nuno Santos. 2023. From prompt injections to sql injection attacks: How protected is your
llm-integrated web application? arXiv preprint arXiv:2308.01990 (2023).
[57] Sezen Perçin, Xin Su, Qutub Sha Syed, Phillip Howard, Aleksei Kuvshinov, Leo Schwinn, and Kay-Ulrich Scholl. 2025. Investigating the Robustness
of Retrieval-Augmented Generation at the Query Level. arXiv preprint arXiv:2507.06956 (2025).
[58] Fábio Perez and Ian Ribeiro. 2022. Ignore previous prompt: Attack techniques for language models. arXiv preprint arXiv:2211.09527 (2022).
[59] Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, and Duen Horng Chau. 2023. Llm self defense: By
self examination, llms know they are being tricked. arXiv preprint arXiv:2308.07308 (2023).
[60] Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, and David Wagner. 2024. Jatmo: Prompt
injection defense by task-specific finetuning. In European Symposium on Research in Computer Security . Springer, 105–124.
[61] Frances A Pogacar, Amira Ghenai, Mark D Smucker, and Charles LA Clarke. 2017. The positive and negative influence of search results on people’s
decisions about the efficacy of medical treatments. In Proceedings of the ACM SIGIR International Conference on Theory of Information Retrieval .
209–216.
[62] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan Nguyen, Daniel Campos, Nick Craswell, and Jimmy Lin. 2025. Ragnarök:
A reusable RAG framework and baselines for TREC 2024 retrieval-augmented generation track. In European Conference on Information Retrieval .
Springer, 132–148.
[63] Yao Qiang, Subhrangshu Nandi, Ninareh Mehrabi, Greg Ver Steeg, Anoop Kumar, Anna Rumshisky, and Aram Galstyan. 2024. Prompt perturbation
consistency learning for robust language models. arXiv preprint arXiv:2402.15833 (2024).
Manuscript submitted to ACM

Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 25
[64] Md Abdur Rahman, Hossain Shahriar, Guillermo Francia, Fan Wu, Alfredo Cuzzocrea, Muhammad Rahman, Md Jobair Faruk, and Sheikh Ahamed.
2025. Fine-tuned large language models (llms): Improved prompt injection attacks detection. (2025).
[65] Amirhossein Razavi, Mina Soltangheis, Negar Arabzadeh, Sara Salamat, Morteza Zihayat, and Ebrahim Bagheri. 2025. Benchmarking prompt
sensitivity in large language models. In European Conference on Information Retrieval . Springer, 303–313.
[66] Alexander Robey, Eric Wong, Hamed Hassani, and George J Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks.
arXiv preprint arXiv:2310.03684 (2023).
[67] Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, et al .2025.
Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red teaming. arXiv preprint arXiv:2501.18837 (2025).
[68] Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2024. " do anything now": Characterizing and evaluating in-the-wild
jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security .
1671–1685.
[69] Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong. 2024. Optimization-based prompt injection
attack to llm-as-a-judge. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security . 660–674.
[70] Tianneng Shi, Kaijie Zhu, Zhun Wang, Yuqi Jia, Will Cai, Weida Liang, Haonan Wang, Hend Alzahrani, Joshua Lu, Kenji Kawaguchi, et al .2025.
PromptArmor: Simple yet Effective Prompt Injection Defenses. arXiv preprint arXiv:2507.15219 (2025).
[71] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the
domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. Transactions of the Association for
Computational Linguistics 11 (2023), 1–17.
[72] Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, and Roman Teucher. 2024. Rag-ex: A generic framework for explaining retrieval augmented
generation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 2776–2780.
[73] Jerry Wang and Fang Yu. 2025. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt
Injection. arXiv preprint arXiv:2507.15042 (2025).
[74] Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang, and Guido Zuccon. 2024. Feb4rag: Evaluating federated search in the context of retrieval
augmented generation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 763–773.
[75] Weixuan Wang, Barry Haddow, Alexandra Birch, and Wei Peng. 2024. Assessing factual reliability of large language model knowledge. In Proceedings
of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies . 805–819.
[76] Yumeng Wang, Lijun Lyu, and Avishek Anand. 2022. Bert rankers are brittle: a study using adversarial document perturbations. In Proceedings of the
2022 ACM SIGIR International Conference on Theory of Information Retrieval . 115–120.
[77] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing
Systems 36 (2023), 80079–80110.
[78] Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten De Rijke, Yixing Fan, and Xueqi Cheng. 2023. Prada: Practical black-box adversarial attacks against
neural ranking models. ACM Transactions on Information Systems 41, 4 (2023), 1–27.
[79] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking Retrieval-Augmented Generation for Medicine. In Findings of the
Association for Computational Linguistics: ACL 2024 .
[80] Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. 2024. BadRAG: Identifying Vulnerabilities in Retrieval Augmented
Generation of Large Language Models. arXiv preprint arXiv:2406.00083 (2024).
[81] Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, and Bo Li. 2024. Rigorllm: Resilient guardrails for large language models
against undesired content. arXiv preprint arXiv:2403.13031 (2024).
[82] Yifan Zeng, Yiran Wu, Xiao Zhang, Huazheng Wang, and Qingyun Wu. 2024. Autodefense: Multi-agent llm defense against jailbreak attacks. arXiv
preprint arXiv:2403.04783 (2024).
[83] Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. 2023. How language model hallucinations can snowball. arXiv preprint
arXiv:2305.13534 (2023).
[84] Peitian Zhang, Zheng Liu, Shitao Xiao, Zhicheng Dou, and Jian-Yun Nie. 2024. A multi-task embedder for retrieval augmented LLMs. In Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics . 3537–3553.
[85] Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, and Qian Wang.
2025. Jbshield: Defending large language models from jailbreak attacks through activated concept analysis and manipulation. arXiv preprint
arXiv:2502.07557 (2025).
[86] Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, and Jianwei Yin. 2024. Hijackrag: Hijacking attacks against
retrieval-augmented large language models. arXiv preprint arXiv:2410.22832 (2024).
[87] Shuyan Zhou, Uri Alon, Frank F Xu, Zhiruo Wang, Zhengbao Jiang, and Graham Neubig. 2022. Docprompting: Generating code by retrieving the
docs. arXiv preprint arXiv: 2207.05987 (2022).
[88] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on
aligned language models. arXiv preprint arXiv:2307.15043 (2023).
[89] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. {PoisonedRAG}: Knowledge Corruption Attacks to {Retrieval-Augmented }
Generation of Large Language Models. In 34th USENIX Security Symposium (USENIX Security 25) . 3827–3844.
Manuscript submitted to ACM
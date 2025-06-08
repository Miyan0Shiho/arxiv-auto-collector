# Magic Mushroom: A Customizable Benchmark for Fine-grained Analysis of Retrieval Noise Erosion in RAG Systems

**Authors**: Yuxin Zhang, Yan Wang, Yongrui Chen, Shenyu Zhang, Xinbang Dai, Sheng Bi, Guilin Qi

**Published**: 2025-06-04 12:55:59

**PDF URL**: [http://arxiv.org/pdf/2506.03901v2](http://arxiv.org/pdf/2506.03901v2)

## Abstract
Retrieval-Augmented Generation (RAG) systems enhance Large Language Models
(LLMs) by incorporating external retrieved information, mitigating issues such
as hallucination and outdated knowledge. However, RAG systems are highly
sensitive to retrieval noise prevalent in real-world scenarios. Existing
benchmarks fail to emulate the complex and heterogeneous noise distributions
encountered in real-world retrieval environments, undermining reliable
robustness assessment. In this paper, we define four categories of retrieval
noise based on linguistic properties and noise characteristics, aiming to
reflect the heterogeneity of noise in real-world scenarios. Building on this,
we introduce Magic Mushroom, a benchmark for replicating "magic mushroom"
noise: contexts that appear relevant on the surface but covertly mislead RAG
systems. Magic Mushroom comprises 7,468 single-hop and 3,925 multi-hop
question-answer pairs. More importantly, Magic Mushroom enables researchers to
flexibly configure combinations of retrieval noise according to specific
research objectives or application scenarios, allowing for highly controlled
evaluation setups. We evaluate LLM generators of varying parameter scales and
classic RAG denoising strategies under diverse noise distributions to
investigate their performance dynamics during progressive noise encroachment.
Our analysis reveals that both generators and denoising strategies have
significant room for improvement and exhibit extreme sensitivity to noise
distributions. Magic Mushroom emerges as a promising tool for evaluating and
advancing noise-robust RAG systems, accelerating their widespread deployment in
real-world applications. The Magic Mushroom benchmark is available at
https://drive.google.com/file/d/1aP5kyPuk4L-L_uoI6T9UhxuTyt8oMqjT/view?usp=sharing.

## Full Text


<!-- PDF content starts -->

arXiv:2506.03901v2  [cs.CL]  5 Jun 2025Magic Mushroom: A Customizable Benchmark
for Fine-grained Analysis of Retrieval Noise Erosion in RAG Systems
Yuxin Zhang1,2∗Yan Wang1,2∗Yongrui Chen1,2Shenyu Zhang1,2Xinbang Dai1,2
Sheng Bi3Guilin Qi1,2†
1School of Computer Science and Engineering, Southeast University, Nanjing 211189, China
2Key Laboratory of New Generation Artificial Intelligence Technology and its Interdisciplinary
Applications (Southeast University), Ministry of Education, China
3School of Law, Southeast University, Nanjing 211189, China
{zzyx_cs, yanwangseu, shenyuzhang, gqi}@seu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) systems enhance Large Language Models
(LLMs) by incorporating external retrieved information, mitigating issues such
as hallucination and outdated knowledge. However, RAG systems are highly sen-
sitive to retrieval noise prevalent in real-world scenarios. Existing benchmarks
fail to emulate the complex and heterogeneous noise distributions encountered in
real-world retrieval environments, undermining reliable robustness assessment. In
this paper, we define four categories of retrieval noise based on linguistic properties
and noise characteristics, aiming to reflect the heterogeneity of noise in real-world
scenarios. Building on this, we introduce Magic Mushroom , a benchmark for
replicating “magic mushroom” noise: contexts that appear relevant on the surface
but covertly mislead RAG systems. Magic Mushroom comprises 7,468 single-hop
and 3,925 multi-hop question-answer pairs. More importantly, Magic Mushroom
enables researchers to flexibly configure combinations of retrieval noise accord-
ing to specific research objectives or application scenarios, allowing for highly
controlled evaluation setups. We evaluate LLM generators of varying parameter
scales and classic RAG denoising strategies under diverse noise distributions to
investigate their performance dynamics during progressive noise encroachment.
Our analysis reveals that both generators and denoising strategies have significant
room for improvement and exhibit extreme sensitivity to noise distributions. Magic
Mushroom emerges as a promising tool for evaluating and advancing noise-robust
RAG systems, accelerating their widespread deployment in real-world applications.
The Magic Mushroom benchmark is available at the link .
1 Introduction
Retrieval-Augmented Generation (RAG) [ 20,9] enhances Large Language Models (LLMs) by incor-
porating externally retrieved information, making it especially valuable for generative AI applications
in dynamic knowledge domains. Recent research has demonstrated the capability of RAG-based
methods in mitigating hallucination and knowledge-coverage deficiencies observed in conventional
generative models by dynamically retrieving information relevant to queries from external sources.
This technique remarkably improves the factuality and informativeness of generated content in down-
stream tasks like Question Answering, Dialogue Generation, and Text Summarization [ 1,12,10,21].
Nevertheless, the effectiveness of RAG systems is tightly constrained by the quality of the retrieved
∗Equal contributors.
†Corresponding author.
Preprint. Under review.

Figure 1: Misleading effect of noise on response generation by LLMs.
contexts [ 4], as retrievers may provide irrelevant, misleading documents, which can distract the
generator and substantially degrade the quality of the final output (Figure 1).
Recent studies have highlighted the susceptibility of RAG systems to retrieval noise by manually
injecting noise and analyzing the impact of this perturbation on generation quality [ 29,3,23].
However, most existing studies largely rely on coarse-grained categorizations of retrieval noise,
overlooking the fine-grained heterogeneity across noise types. Our analysis of sparse[ 27], dense[ 31,
18,26,16], and hybrid retrieval methods highlights substantial variation in noise composition across
domains (in Appendix A), explaining conflicting RAG configuration recommendations in prior work.
Moreover, current benchmarks typically employ static setups with uniform noise, failing to capture
the diversity observed in real-world retrieval [ 37,23]. This underscores the need for a benchmark
that models realistic, heterogeneous noise conditions to enable more accurate and robust evaluation
of generation systems.
To bridge this gap, we propose Magic Mushroom , a controllable testbed for evaluating LLM
robustness under complex retrieval noise. We define four representative noise types: Distracting ,
Low Quality ,Inconsequential , and Irrelevant , which closely mimic realistic, semantically similar
distractors, better simulating subtle noise in real-world retrieval applications. Built on Natural
Questions [ 19] and HotpotQA [ 42], Magic Mushroom comprises over 7,468 single-hop and 3,925
multi-hop QA instances, each paired with golden and diverse noise documents, yielding greater
complexity than prior benchmarks. Its flexible noise configurations support fine-grained evaluations
of LLM generators. This design narrows the gap between benchmark and deployment conditions,
facilitating both reliable robustness assessments and principled noise mitigation research.
Leveraging Magic Mushroom, we conduct comprehensive experiments on different RAG systems
to evaluate their performance under complex retrieval noise. We test LLM generators of varying
scales and diverse denoising strategies across a full range of noise proportions (0–100%), revealing
performance trends and robustness limits under increasing noise. The testbed’s flexible configuration
supports scenario-specific noise simulation, enabling targeted evaluation of optimization strategies
and system setups. This adaptability allows researchers to efficiently construct tailored subsets to
identify more resilient generators and denoising methods. Our findings reveal several key insights:
(a)Existing generative systems exhibit significant room for improvement, since their performance
consistently degrades as noise increases, affected by both noise type and proportion; (b)Different
noise types affect generation in distinct ways, highlighting the need for noise-specific denoising
strategies; (c)Model performance is strongly scenario-dependent, as both generators and denoising
methods remain highly sensitive to retrieval noise distributions, and (d)Noise induces attention shifts
in the generator, which diminishes the overall quality of the generated response. By introducing Magic
Mushroom, we aim to catalyze the development of next -generation RAG systems and enhance their
anti-noise capability, expediting their reliable deployment in real-world information environments.
2 Definition of Retrieval Noise
Task Formulation . In this study, we define the task formulation for evaluating the robustness of RAG
systems under retrieval noise conditions. The task is structured around a question-answering paradigm.
Formally, given a query qand a set of Top- krelevant documents D={d1,···, dk}retrieved from
an external data source. During inference, the retrieval documents Dare concatenated with qto form
2

Figure 2: Overview of the Magic Mushroom benchmark construction process.
I= [q;d1;···;dk], which is then fed into a pre-trained LLM Gto generate the corresponding answer
a=G(I). If a retrieved document dcontains key elements ( e.g., answer or evidence fragments)
supporting the answer to q, it is denoted as a golden document dgolden . Conversely, dis denoted as a
noisy document dnoise . The QA task is evaluated under noisy retrieval settings, where the retrieved
document set Dcontains varying types and quantities of noisy documents. We assess the noise
robustness of the RAG system by evaluating the quality of its response.
Taxonomy of Retrieval Noise . Existing noise classification systems typically categorize retrieval
noise based on document-query relevance or factual accuracy, overlooking finer semantic and
distributional differences. These coarse-grained classifications fail to reveal the specific mechanisms
through which different types of noise affect the performance of RAG. We categorize retrieved
documents into 4 distinct types based on linguistic attributes and noise characteristics. Golden
Document : Characterized by high topical relevance to the query, this document contains accurate,
complete information that directly and comprehensively supports the generation of the correct answer.
Distracting Noise : While highly relevant to the query in topic, critical answer-supporting elements
within this document present factual errors, false statements, or outdated information. Low Quality
Noise : While highly relevant to the query, this document contains counterfactual information or
formatting errors that diminish its positive contribution to answer generation. Inconsequential Noise :
While highly relevant to the query, this document offers little substantive support for answering the
query. Irrelevant Noise : This document is entirely unrelated to the query. Irrelevant noise assesses
whether the RAG model can identify and filter out semantically unrelated content, thereby preventing
distraction from the golden document. The rationale for this taxonomy is illustrated in Appendix B.1.
3 Benchmark Construction
We construct a retrieval noise robustness benchmark built on existing QA datasets—NQ and Hot-
PotQA. Given a QA dataset and its associated retrieval corpus, the construction of the benchmark
for retrieval noise robustness comprises three primary steps: QA Instance Selection . Selecting
high-quality QA pairs where both questions and answers possess semantic clarity and unambiguity
(§3.1). Golden Documents Augmentation . Increasing golden documents’ lexical and syntactic
diversity while preserving semantic equivalence (§3.2). Noise Introduction . Four categories of
noisy documents were constructed based on informational relevance and noise characteristics (§3.3).
Figure 2 presents an overview of the Magic Mushroom benchmark construction process.
3.1 QA Instance Selection
We selected question-answer (q, a)pairs and their corresponding relevant Wiki Pages from the original
dataset, ensuring coverage across multiple topical domains. We initiated our data preprocessing with
a coarse-grained cleaning of the raw dataset, retaining only (q, a)pairs that exhibited clearly defined
topics and representativeness. Following this, questions demonstrating referential ambiguity were
systematically removed. Such as the question “ How many people are in our country? ” was
discarded because the referent of the pronoun “ our” is indeterminate. Subsequently, we remove
questions whose answers are not uniquely determined or lack a standardized reference. For instance,
3

“Where is Zimbabwe located? ” may elicit multiple plausible answers. Finally, we apply regular
expression-based rules to further standardize the filtered (q, a)pairs. This process normalizes textual
expressions and removes residual non-semantic formatting characters or tags, ensuring the cleanliness
of the QA instances.
3.2 Golden Documents Augmentation
In this phase, we leveraged OpenAI’s GPT-4 ( gpt-4-0613 ) [24] to augment the data of the original
golden documents (see Appendix B for detailed prompts). The objective was to generate semantically
equivalent variants of these golden documents to enrich their diversity. The specific augmentation
strategy adopted comprises two stages.
Synonym Substitution . Drawing inspiration from EDA [ 35], the initial step involves applying Part-
of-Speech (POS) tagging to the original golden documents to identify the grammatical function of
each word. Subsequently, focus is placed on substantive words—namely nouns, verbs, adjectives, and
adverbs—that are not directly included in, or semantically essential to, the answer for the associated
query. We utilized GPT-4 to perform random synonym substitution on identified non-stop candidate
words with a 20% probability.
Syntatic Change . Drawing inspiration from SCPN [ 15], we employ GPT-4 to generate syntactically
controlled paraphrases for sentences within the Golden Document. The meticulously designed
prompts instruct GPT-4 to prioritize alterations to the input sentence’s syntactic structure during
content generation, while concurrently imposing strict constraints to ensure the original semantic
information remains invariant. Subsequently, we apply predefined, semantically invariant grammatical
transformation rules. Finally, to ensure the quality and fidelity of the augmented data, we implement
a Back-Validation procedure [ 32] that combines automated LLM-based assessment with manual
human verification. This dual validation step filters out instances exhibiting significant semantic drift
or unintended noise introduced during augmentation.
3.3 Noise Introduction
To conduct a fine-grained evaluation of RAG robustness under heterogeneous retrieval noise, we
construct four types of noisy documents for each QA pair: Distracting Noise, Low Quality Noise,
Inconsequential Noise, and Irrelevant Noise.
Distracting Noise exhibits overall semantic similarity to the golden document, yet its core factual
content is deliberately corrupted through key entity substitution. Inspired by the semantic unit
substitution strategy proposed in [ 45], we replace mentions of critical elements within the golden
document—those essential for answering the question—with non-synonymous entities to undermine
the veracity of the retrieved evidence. Furthermore, we incorporate syntactic structure transforma-
tions, as detailed in Section 3.2, to enhance the naturalness and diversity of the distracting noise.
Consequently, distracting noise is designed to assess the generator’s discriminative capability when
confronted with retrieved evidence that is similar yet factually misleading.
Low Quality Noise primarily comprises counterfactual content, texts exhibiting minimal informa-
tional content, or poorly structured text. To construct noisy documents containing counterfactual
information, we extract passages from Wikipedia pages pertinent to the topic of the query. Subse-
quently, entities within these passages are replaced with semantically plausible yet factually incorrect
information, following the procedure described in Section 3.2. Noise from scarce information or
anomalous formatting is simulated by selecting Wikipedia segments with low information density or
structural issues, often stemming from HTML remnants or data extraction errors. Low Quality noise
is utilized to assess the sensitivity of RAG to the quality of information sources.
Inconsequential Noise denotes content that does not directly or substantially contribute to answer
inference. We anchor on the exact location of the golden document within its Wikipedia page
and expand kparagraphs bidirectionally to gather superficially related candidate documents. To
ensure these candidate documents are genuinely “inconsequential”, we employ GPT-4 for dual
verification: (1) factual consistency ( i.e.,accuracy and no misleading information), and (2) answer
non-containment ( i.e.,no direct or indirect answer disclosure). Inconsequential noise facilitates
assessing RAG’s capability to focus on critical evidence amid relevant yet non-essential information.
4

Irrelevant Noise is an extreme scenario where the content is entirely unrelated to the query topic.
We construct irrelevant noise using a cross-question negative sampling strategy. Specifically, for a
given query qi, we randomly sample passages from Wikipedia pages associated with a different query
qj(where j̸=i) to serve as irrelevant noise for qi.
4 Experiments
4.1 Benchmark
The Magic Mushroom benchmark is constructed using a semi-automatic methodology detailed in
Section 3. We collected 7,468 single-hop (NQ) and 3,925 multi-hop complex (HotPotQA) QA pairs
via QA Instance Selection (§3.1). These instances were then divided into a development set (4,810
single-hop and 2,524 multi-hop QA pairs), a public test set (1,584 single-hop and 841 multi-hop
QA pairs), and a private test set (1,074 single-hop and 560 multi-hop QA pairs) withheld to prevent
potential data leakage. Unless otherwise specified, all reported experimental results are based on the
public test sets, with MM sandMM mdenoting the single-hop and multi-hop subsets, respectively.
For each QA pair, we employed Golden Document Augmentation (§3.2) and Noise Introduction
(§3.3) to construct 10 Golden Documents, 10 Distracting noise, and 7 exemplars each of Low Quality,
Inconsequential, and Irrelevant noise. Notably, in all experiments, we fixed the number of retrieved
documents kto 10. By dynamically varying the composition and proportions of different types of
retrieved documents, we could systematically evaluate the robustness of RAG systems under diverse
noise conditions. Detailed statistics for Magic Mushroom are provided in Appendix C.
4.2 Experimental Setup
Evaluation Metrics . In all experiments, we evaluate performance through two complementary
metrics: correctness ( Cor.) and rejection rate ( Rej.). Correctness is assessed by employing GPT-4 to
score generated answers against gold answers on a scale of 0 (completely incorrect), 1 (minimally
relevant but insufficient), 3 (partially correct but with minor errors or omissions), or 5 (fully correct
and comprehensive); detailed evaluation prompts and results aligned with manual annotations are
presented in the Appendix D. For clarity, correctness scores are normalized to the range [0,1]. The
Rejection Rate quantifies the proportion of instances where the generation model explicitly abstains
from answering, typically when it determines that the retrieved information or its internal knowledge
is insufficient to formulate a confident response.
Baseline Models . We evaluate a broad range of LLM generators of different architectures and scales:
Llama-3.2 1B[7], Qwen-2.5 1.5B[41], Llama-3.1 8B, Qwen-2.5 7B, Llama-3.1 70B, Qwen-2.5 72B, and
DeepSeek-V3[ 5]. To assess the robustness of reasoning models under noisy conditions, we fur-
ther investigate R1-Distill-Llama 8B, R1-Distill-Llama 70B, and DeepSeek-R1[ 6]. In addition, we
evaluate the effectiveness of several classic denoising strategies, including VANILLA RAG [2],CHAIN -
OFNOTE [44], DRAGIN [ 28], SKR [ 33], and CRAG [ 40]. Our analysis focuses on quantifying the
detrimental impact of retrieval noise on RAG pipelines and identifying optimal combinations of
generators and denoising strategies across different scenarios. Detailed descriptions and configuration
settings for the LLM generators and denoising strategies are provided in Appendix E.
4.3 Main Results
The main experiments constructed retrieval contexts by randomly sampling from the golden and
noise documents. Detailed statistics regarding the composition of retrieval contexts are provided in
Appendix F.1. We evaluated the performance of different generators and denoising strategies across
multiple noise ratios to reveal the erosion effect of retrieval noise on the performance of RAG systems.
Table 1 and 2 present partial results, with the complete results provided in Appendix F.2. We compare
the RAG-based systems to the NORAG and observe that incorporating retrieval significantly improves
answer correctness; for example, the Llama-3.1 8Bachieves a relative improvement of 50.35%. As the
noise ratio increases, answer correctness degrades across all RAG variants; however, this degradation
is not linear. We identify a critical threshold at a 50% noise ratio, beyond which RAG performance
deteriorates rapidly, resulting in a pronounced avalanche-style collapse in correctness. Moreover, we
observe that all robust RAG systems, except CHAINOF NOTE, severely reduce answer correctness at
low-to-medium noise ratios; for instance, the Llama-3.1 8B+SKR exhibits a 42.55% correctness drop
5

Table 1: Performance of RAG-based Systems at Different Noise Ratios on the MM s.
LLM G ENERATOR0% 10% 30% 50% 70% 90% 100%
Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej.
VANILLA RAG [2]
Llama-3.1 8B 84.1 2.4 81.1 2.9 76.8 3.0 73.9 2.7 59.9 3.4 44.9 4.7 18.0 5.9
Qwen-2.5 7B 72.7 13.8 70.5 14.7 67.7 15.4 62.9 18.3 41.6 30.7 27.5 37.9 7.4 45.2
Llama-3.1 70B 81.9 6.8 81.0 6.2 77.5 5.5 75.0 5.4 68.3 6.2 58.8 8.3 28.9 16.8
Qwen-2.5 72B 84.3 9.3 83.8 8.5 82.9 8.0 79.8 8.8 74.2 12.5 52.2 25.4 14.2 45.3
R1-Distill-Llama 8B 84.1 5.6 81.7 5.6 78.7 5.1 74.1 5.2 64.1 7.5 45.2 11.5 22.7 13.0
R1-Distill-Llama 70B 84.2 7.5 82.9 6.7 82.7 5.8 80.4 6.8 72.2 10.7 57.6 18.6 28.0 30.8
CHAINOF NOTE [44]
Llama-3.1 8B 87.1 3.5 83.2 3.5 80.2 2.4 74.9 1.7 63.7 1.5 45.2 2.1 14.5 2.9
Qwen-2.5 7B 76.7 9.3 74.2 9.3 70.8 10.5 64.6 9.1 52.9 12.2 31.2 15.2 10.4 17.9
Llama-3.1 70B 85.7 4.7 85.1 3.7 82.5 3.5 79.6 3.6 72.7 4.2 56.7 5.3 24.1 9.3
Qwen-2.5 72B 84.4 7.6 83.2 6.8 80.5 6.2 75.5 5.1 67.1 6.0 45.1 11.7 16.9 21.3
R1-Distill-Llama 8B 84.4 4.0 81.2 3.0 76.7 2.0 70.6 1.8 60.7 2.5 46.2 4.0 27.6 4.8
R1-Distill-Llama 70B 85.2 5.4 83.5 4.3 82.9 3.1 79.0 2.7 73.7 4.1 63.4 6.6 34.1 11.6
DRAGIN [28]
Llama-3.1 8B 57.0 16.7 56.2 15.9 54.6 16.4 53.1 15.4 49.3 16.8 41.4 15.7 30.9 17.8
Qwen-2.5 7B 20.3 66.8 20.7 66.2 20.3 66.1 20.2 66.2 17.4 68.3 16.3 68.2 14.2 68.4
R1-Distill-Llama 8B 56.6 7.8 55.4 9.2 55.8 6.8 55.8 6.4 50.5 7.2 35.7 8.8 21.9 9.7
SKR [33]
Llama-3.1 8B 41.5 11.8 40.8 12.5 40.3 12.3 39.1 13.6 38.0 13.9 38.4 13.9 36.7 13.9
Qwen-2.5 7B 25.5 55.8 23.8 55.8 23.8 56.6 23.1 56.5 21.3 57.5 19.5 58.7 17.1 60.2
Llama-3.1 70B 62.7 5.1 61.1 4.8 60.6 4.4 58.3 4.5 57.8 4.8 55.9 4.8 49.4 7.3
Qwen-2.5 72B 52.6 22.5 53.5 22.2 53.2 21.9 52.6 22.5 50.8 24.0 45.3 27.1 41.2 27.2
R1-Distill-Llama 8B 43.9 11.6 45.4 10.8 41.7 12.3 42.3 9.9 37.8 12.0 30.9 14.1 25.9 15.2
R1-Distill-Llama 70B 60.0 7.9 61.1 5.9 61.2 6.2 58.7 7.3 57.2 8.1 55.4 8.9 51.4 10.6
CRAG [40]
Llama-3.1 8B 70.9 11.5 70.6 9.8 68.2 9.2 65.4 9.1 55.4 11.1 37.3 14.1 21.7 15.6
Qwen-2.5 7B 58.7 26.9 59.1 25.5 57.5 23.7 52.9 25.1 38.3 33.4 24.1 41.9 10.0 46.6
Llama-3.1 70B 73.4 9.1 75.7 8.9 72.9 7.9 70.2 7.7 62.6 9.3 51.1 11.4 33.9 15.7
Qwen-2.5 72B 70.8 23.4 72.9 19.9 72.1 18.9 68.5 18.3 58.5 26.6 33.9 42.1 13.6 53.3
R1-Distill-Llama 8B 71.3 12.8 71.5 11.3 68.0 11.1 63.1 9.6 49.4 13.0 34.0 17.4 22.0 17.9
R1-Distill-Llama 70B 73.9 13.8 73.7 13.7 73.4 12.1 71.4 10.9 62.1 16.3 41.9 28.1 25.9 32.8
compared to VANILLA RAG . Interestingly, a surprising reversal occurs under a fully noisy context:
in this scenario, all denoising strategies (excluding CHAINOF NOTE) outperform VANILLA RAG
by a significant margin. Furthermore, SKR andDRGIN demonstrate high rejection rates and a
more gradual correctness decline in high-noise settings, potentially because their document utility
assessment mechanism, while filtering noise, may also prevent leveraging beneficial retrieved context.
Table 2: Performance of NoRAG ( MM s).
LLM G ENERATORNORAG
Cor. Rej.
Llama-3.1 8B 36.8 14.8
Qwen-2.5 7B 16.3 65.6
Llama-3.1 70B 54.8 7.2
Qwen-2.5 72B 41.2 34.7
R1-Distill-Llama 8B 27.5 15.2
R1-Distill-Llama 70B 54.1 10.0Performance disparities across LLM families are also
prominent: Llama-series models (“aggressive”) achieve
higher correctness, whereas Qwen-series models (“con-
servative”) exhibit superior rejection rates, which is
a valuable trait in high-noise conditions. We further
observe that scaling the model brings limited perfor-
mance gains in low-noise settings. However, under high
retrieval noise conditions, increasing the model scale
substantially improves robustness. Overall, the robust-
ness of RAG systems leaves considerable room for
enhancement. Their pronounced sensitivity to noise
means any interference can degrade performance. Eval-
uating robustness at a fixed noise level provides only
partial insights.
4.4 Noise Type Analysis
Figure 3 illustrates the impact of introducing individual types of retrieval noise on RAG system
performance. It is evident that distracting noise significantly degrades performance, even at a noise
ratio as low as 10%, dropping correctness to 40.56% for Llama-3.1 8B+SKR and 23.02% for
6

VanillaRAG ChainofNote DRAGIN SKR CRAG
00.1 0.3 0.5 0.7 0.910.00.20.40.60.81.0
Distracting RatioCorrectness
00.1 0.3 0.5 0.7 0.910.20.40.60.81.0
Low Quality RatioCorrectness
00.1 0.3 0.5 0.7 0.910.40.60.81.0
Inconsequential RatioCorrectness
00.1 0.3 0.5 0.7 0.910.00.20.40.60.81.0
Irrelevant RatioCorrectness
00.1 0.3 0.5 0.7 0.910.00.20.40.60.81.0
Distracting RatioCorrectness
00.1 0.3 0.5 0.7 0.910.20.40.60.8
Low Quality RatioCorrectness
00.1 0.3 0.5 0.7 0.910.20.40.60.8
Inconsequential RatioCorrectness
00.1 0.3 0.5 0.7 0.910.00.20.40.60.8
Irrelevant RatioCorrectnessFigure 3: Impact of different types of retrieval noise on RAG system performance, using Llama-3.1 8B
(top) and Qwen-2.5 7B(bottom) as LLM generators.
Qwen-2.5 8B+SKR . Conversely, irrelevant noise exhibits the mildest detrimental effect. This
indicates that distracting noise is particularly misleading, causing the RAG system to become easily
misguided when presented with such documents. Notably, we also observe that individual types
of noise exhibit somewhat reduced harmfulness compared to mixed noise conditions. Excluding
distracting noise, the noise proportion threshold at which severe model performance degradation
occurs increases up to approximately 90%. Additionally, we observe that, irrespective of whether
Llama-3.1 8Bor Qwen-2.5 8B, performance curves corresponding to different denoising strategies
show a noticeable intersection point at a distracting noise proportion of 90%, after which DRGIN and
SKR surpass others. However, under low-noise conditions, the performance of DRGIN and SKR
remains unsatisfactory, possibly due to their tendency to excessively question retrieved content in
the presence of even minor noise. In summary, different types of noise affect RAG systems in
distinctly varying ways. Therefore, tailored denoising strategies should be developed according to
the unique characteristics of each noise type.
4.5 Sensitivity Analysis
Figure 4 illustrates the RAG system’s sensitivity to the retrieved documents’ position. We categorize
document positions into three groups: “Near” indicating proximity to the query; “Mid” referring to
documents positioned centrally among retrieved documents; and “Far” denoting documents located
at the end of the input sequence. We observe that Llama-3.1 8Bis entirely insensitive to document
ordering. In contrast, Qwen-2.5 7Bexhibits a clear “lost in the middle” phenomenon, demonstrating
significant sensitivity to mid-positioned documents. Under high-noise conditions, positioning the
golden document closer to the query enhances the generator’s attention towards relevant content,
thus increasing answer correctness. This finding suggests that reranking retrieved documents
can beneficially improve RAG performance, although reranking may concurrently amplify the
detrimental effects of distracting and inconsequential noise .
Near Mid Far
0.1 0.2 0.3 0.5 0.70.40.50.60.70.8
Golden RatioCorrectness
0.1 0.2 0.3 0.5 0.70.60.70.8
Distracting RatioCorrectness
0.1 0.2 0.3 0.5 0.70.710.750.790.83
Low Quality RatioCorrectness
0.1 0.2 0.3 0.5 0.70.740.780.82
Inconsequential RatioCorrectness
0.1 0.2 0.3 0.5 0.70.780.800.820.84
Irrelevant RatioCorrectness
0.1 0.2 0.3 0.5 0.70.30.40.50.6
Golden RatioCorrectness
0.1 0.2 0.3 0.5 0.70.40.50.60.7
Distracting RatioCorrectness
0.1 0.2 0.3 0.5 0.70.650.690.73
Low Quality RatioCorrectness
0.1 0.2 0.3 0.5 0.70.75
0.710.73
0.69
Inconsequential RatioCorrectness
0.1 0.2 0.3 0.5 0.70.680.700.720.74
Irrelevant RatioCorrectness
Figure 4: Sensitivity of RAG system performance to the position of retrieved documents, using
Llama-3.1 8B(top) and Qwen-2.5 7B(bottom) as LLM generators.
7

Figure 5 presents the sensitivity of the RAG system to the length of retrieved content. Under low-noise
conditions, shorter retrieval content benefits RAG, as the generator can more easily and accurately
extract answers. Conversely, under high-noise environments, longer retrieval contexts improve the
generator’s ability to produce correct responses, possibly due to the generator relying on additional
contextual information to determine the correct answer amidst noisy context. This observation
indicates that retrieval length should be minimized when retrieved content is relatively clean,
but extended when substantial noise is present, to provide sufficient contextual support for the
generator.
0.0 0.1 0.3 0.5 0.7 0.9 1.00.10.40.71.0
Noise RatioCorrectness
0.0 0.1 0.3 0.5 0.7 0.9 1.00.10.40.71.0
Noise RatioCorrectness0-299
300-599
600-899
900-1199
1200-1499
1500+
Figure 5: Sensitivity of RAG system performance to the length of retrieved documents, using Llama-
3.18B(left) and Qwen-2.5 7B(right) as LLM generators.
4.6 Scenario-Level Robustness Analysis
Table 3: Performance of generator-denoiser
combinations across different noise scenarios.
LLM G ENERATORSCE.1 S CE.2 S CE.3 S CE.4
Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej.
VANILLA RAG
Llama-3.1 8B 46.8 3.3 75.2 2.5 69.9 3.6 47.5 4.1
Qwen-2.5 7B 36.6 28.8 63.4 16.6 55.4 18.4 33.6 32.8
CHAINOF NOTE
Llama-3.1 8B 47.5 1.3 75.6 2.1 74.2 1.4 35.9 1.9
Qwen-2.5 7B 43.8 11.3 67.3 8.1 63.3 7.3 41.4 17.4
DRAGIN
Llama-3.1 8B 38.9 17.0 53.7 15.1 52.3 15.6 48.0 16.6
Qwen-2.5 7B 18.4 52.8 19.7 66.5 19.1 66.2 17.3 62.9
SKR
Llama-3.1 8B 39.1 12.5 40.1 12.2 38.6 13.1 38.7 13.1
Qwen-2.5 7B 21.9 47.1 24.1 55.4 22.4 56.6 20.5 58.7
CRAG
Llama-3.1 8B 54.8 8.9 65.1 8.7 57.2 11.1 46.5 8.6
Qwen-2.5 7B 40.3 30.6 54.6 23.3 44.4 30.3 42.6 24.5Based on a coarse-grained statistical analysis
of noise distributions across four discussion
topics, we synthesized four distinct scenario-
level ( SCE.1–SCE.4) noise distributions for robust-
ness evaluation. The specific noise distributions
for each scenario and their design rationale are de-
tailed in Appendix G.2. Table 3 summarizes the
performance results for various generator-denoiser
combinations across these constructed scenarios.
Our findings indicate that noise composition influ-
ences RAG performance in these simulated envi-
ronments. Both the choice of generator and the
applied denoising strategy exhibit high sensitiv-
ity to the specific types and proportions of noise.
For instance, despite SCE.1andSCE.2sharing an
identical proportion of Golden documents (30%),
the optimal generator and denoising strategy com-
bination differed substantially. Similarly, between
SCE.3andSCE.4, the performance of a fixed generator and denoising configuration varied signifi-
cantly, with a maximum observed performance delta of 21.4%. This lack of a predictable relationship
presents a considerable challenge for pre-determining the optimal generator-denoiser configurations.
4.7 Case Study
Que Dis1 Dis2 Dis3 Dis4 Gol0 5 10 15 20 25 30Layer
Que Low1 Low2 Low3 Low4 Gol0 5 10 15 20 25 30
Que Inc1 Inc2 Inc3 Inc4 Gol0 5 10 15 20 25 30
Que Irr1 Irr2 Irr3 Irr4 Gol0 5 10 15 20 25 30
Que Dis Inc Low Irr Gol0 5 10 15 20 25 30
 0.000.010.020.030.040.05
Figure 6: Layer-wise attention distribution of Llama-3.1 8B.
Figure 6 illustrates the layer-wise attention distribution of the Llama-3.1 8Bover the question and
various retrieved documents when answering the query “ In greek mythology who was the
goddess of spring growth? ”. It can be observed that “Distracting Noise” poses a significant
challenge: once the generator’s attention is captured by distracting noise, this effect intensifies
8

throughout subsequent layers, eventually misleading the generator into producing incorrect answers.
Similarly, “Inconsequential Noise” also succeeds in misguiding the model, although the attention
mechanism exhibits an ongoing competition, oscillating between inconsequential noise and the golden
document. In contrast, “Low Quality Noise” initially attracts some attention in the earlier layers but
is neglected entirely after intermediate layers. Lastly, the attention mechanism consistently overlooks
“Irrelevant Noise” throughout the decoding process, empirically validating that singularly irrelevant
distractions cannot influence the generator’s output. In essence, these observations demonstrate
that noise can induce shifts in the generator’s attention; the more deceptive or distracting the
noise, the more significant these attention shifts become, thereby increasing the likelihood of
the generator being misled into producing incorrect outputs. Refer to Appendix G.3 for a more
detailed case analysis.
5 Conclusion
We introduce Magic Mushroom, a benchmark designed to evaluate the robustness of RAG systems un-
der complex retrieval noise. It allows flexible noise types and proportions configurations, simulating
real-world retrieval noise environments. It comprises 7,468 single-hop and 3,925 multi-hop question-
answer pairs, enabling comprehensive performance evaluations across different LLM generators and
denoising strategies. Magic Mushroom defines four categories of retrieval noise—Distracting, Low
Quality, Inconsequential, and Irrelevant—reflecting real-world noise heterogeneity. Our experiments
reveal that RAG systems are susceptible to noise, with significant performance degradation observed
as noise ratios increase. Magic Mushroom provides a challenging and flexible evaluation frame-
work, advancing the development of more robust and noise-resistant RAG systems for real-world
applications. Magic Mushroom is currently limited to English-language evaluation . While the
methodology and findings may generalize cross-lingually, extending and validating their applicability
to non-English contexts remains a direction for future work.
6 Related Work
Recent research extensively investigates LLM noise robustness in RAG systems, focusing on sensitiv-
ity to retrieval noise [ 2,9,4,13]. These studies reveal substantial variability in LLM responses to
irrelevant context: some benefit from additional context, while others degrade [ 17,14]. Fang et al. [8]
quantified the performance degradation of various LLMs when confronted with retrieval noise, reveal-
ing that model scale and the extent of training and fine-tuning significantly impact noise robustness.
Table 4: Comparison of Retrieval Noise
Benchmarks.
Benchmark Categories Flexibility
RGB[3] 2 ✗
RECALL[23] 1 ✗
RAG-Bench[8] 3 ✗
NoMIRACL[29] 1 ✗
NoiserBench[37] 7 ✗
Robust RALM[43] 1 ✗
Magic Mushroom 4 ✓Several studies have investigated the robustness of mul-
tilingual RAG systems, finding that most LLMs struggle
to strike a balance between avoiding hallucinations and
providing correct answers [ 3,29]. Wu et al. [37] fur-
ther defined seven distinct noise types and categorized
them as beneficial or harmful based on their character-
istics. Domain-specific and long-document noise have
also been explored, such as RAG-QA Arena evaluating
cross-domain robustness in long-text question answering
[11]. Diverging from existing research, Magic Mushroom
transcends the limitations of existing benchmarks ( e.g.,sin-
gular noise types, fixed ratios) by offering flexible noise
combination configurations, thus providing an evaluation environment more reflective of real-world
applications. Table 4 illustrates the comparison of retrieval noise benchmarks, where Flexibility
refers to the benchmark’s ability to construct different proportions and types of noise.
To mitigate the impact of noise, researchers have proposed various denoising strategies, encompassing
post-retrieval filtering, reranking, noise-robust training, and explicit denoising during generation
[40,28,33,44,34,39]. The simplest approach filters documents post-retrieval to eliminate irrelevant
or harmful content [ 34]. For instance, Yoran et al. [43] utilized natural language inference models to
assess passage-QA consistency, filtering non-supporting passages. Another paradigm is reranking:
places highest-scoring contexts near the prompt to reduce interference from irrelevant passages [ 9].
Noise-robust training exposes models to noisy retrieval contexts during training/fine-tuning, enabling
them to learn noise ignorance and correct answer extraction [ 8,30,22]. Alternatively, models can
9

actively identify and exclude noise during inference. InstructRAG employs an instruction-tuned LM
to read documents, produce a rationale, articulate reasoning, and then derive the answer [ 36]. Other
studies utilize Chain-of-Thought (CoT) to guide the model in incrementally analyzing whether each
retrieved result helps answer before selectively utilizing them [ 38,44]. Notably, these denoising
strategies are not mutually exclusive. Practical RAG systems often combine multiple strategies to
leverage their respective strengths against diverse noise [40, 2].
References
[1]Chenxin An, Ming Zhong, Zhichao Geng, Jianqiang Yang, and Xipeng Qiu. Retrievalsum: A
retrieval enhanced framework for abstractive summarization. CoRR , abs/2109.07943, 2021.
URL https://arxiv.org/abs/2109.07943 .
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. In The Twelfth International
Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . Open-
Review.net, 2024. URL https://openreview.net/forum?id=hSyW5go0v8 .
[3]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models
in retrieval-augmented generation. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam
Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-
Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth
Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27,
2024, Vancouver, Canada , pages 17754–17762. AAAI Press, 2024. doi: 10.1609/AAAI.V38I16.
29728. URL https://doi.org/10.1609/aaai.v38i16.29728 .
[4]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano,
Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. The power of noise: Redefining
retrieval for RAG systems. In Grace Hui Yang, Hongning Wang, Sam Han, Claudia Hauff,
Guido Zuccon, and Yi Zhang, editors, Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington
DC, USA, July 14-18, 2024 , pages 719–729. ACM, 2024. doi: 10.1145/3626772.3657834. URL
https://doi.org/10.1145/3626772.3657834 .
[5]DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu,
Chenggang Zhao, and et al. Deepseek-v3 technical report. CoRR , abs/2412.19437, 2024. doi:
10.48550/ARXIV .2412.19437. URL https://doi.org/10.48550/arXiv.2412.19437 .
[6]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, and et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
learning. CoRR , abs/2501.12948, 2025. doi: 10.48550/ARXIV .2501.12948. URL https:
//doi.org/10.48550/arXiv.2501.12948 .
[7]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Kevin Stone, and et al. The llama 3 herd of models. CoRR ,
abs/2407.21783, 2024. doi: 10.48550/ARXIV .2407.21783. URL https://doi.org/10.
48550/arXiv.2407.21783 .
[8]Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. Enhancing
noise robustness of retrieval-augmented language models with adaptive adversarial training.
In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024,
Bangkok, Thailand, August 11-16, 2024 , pages 10028–10039. Association for Computational
Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.540. URL https://doi.org/10.
18653/v1/2024.acl-long.540 .
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Qianyu Guo, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large
language models: A survey. CoRR , abs/2312.10997, 2023. doi: 10.48550/ARXIV .2312.10997.
URL https://doi.org/10.48550/arXiv.2312.10997 .
10

[10] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM:
retrieval-augmented language model pre-training. CoRR , abs/2002.08909, 2020. URL https:
//arxiv.org/abs/2002.08909 .
[11] Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu, William Yang Wang,
Bonan Min, and Vittorio Castelli. RAG-QA arena: Evaluating domain robustness for long-
form retrieval augmented question answering. In Yaser Al-Onaizan, Mohit Bansal, and Yun-
Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024 , pages 4354–
4374. Association for Computational Linguistics, 2024. URL https://aclanthology.org/
2024.emnlp-main.249 .
[12] Tom Hosking, Hao Tang, and Mirella Lapata. Hierarchical indexing for retrieval-augmented
opinion summarization. Trans. Assoc. Comput. Linguistics , 12:1533–1555, 2024. doi: 10.1162/
TACL\_A\_00703. URL https://doi.org/10.1162/tacl_a_00703 .
[13] Yufang Hou, Alessandra Pascale, Javier Carnerero-Cano, Tigran T. Tchrakian, Radu Marinescu,
Elizabeth Daly, Inkit Padhi, and Prasanna Sattigeri. Wikicontradict: A benchmark for evaluating
llms on real-world knowledge conflicts from wikipedia. In Amir Globersons, Lester Mackey,
Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors,
Advances in Neural Information Processing Systems 38: Annual Conference on Neural
Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December
10 - 15, 2024 , 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/
c63819755591ea972f8570beffca6b1b-Abstract-Datasets_and_Benchmarks_Track.
html .
[14] Jennifer Hsia, Afreen Shaikh, Zhiruo Wang, and Graham Neubig. RAGGED: towards informed
design of retrieval augmented generation systems. CoRR , abs/2403.09040, 2024. doi: 10.48550/
ARXIV .2403.09040. URL https://doi.org/10.48550/arXiv.2403.09040 .
[15] Mohit Iyyer, John Wieting, Kevin Gimpel, and Luke Zettlemoyer. Adversarial example gener-
ation with syntactically controlled paraphrase networks. In Marilyn A. Walker, Heng Ji, and
Amanda Stent, editors, Proceedings of the 2018 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT
2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers) , pages 1875–
1885. Association for Computational Linguistics, 2018. doi: 10.18653/V1/N18-1170. URL
https://doi.org/10.18653/v1/n18-1170 .
[16] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learn-
ing. Trans. Mach. Learn. Res. , 2022, 2022. URL https://openreview.net/forum?id=
jKN1pXi7b0 .
[17] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan Ö. Arik. Long-context llms meet RAG:
overcoming challenges for long inputs in RAG. CoRR , abs/2410.05983, 2024. doi: 10.48550/
ARXIV .2410.05983. URL https://doi.org/10.48550/arXiv.2410.05983 .
[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 6769–
6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/
2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/ .
[19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh,
Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova,
Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le,
and Slav Petrov. Natural questions: a benchmark for question answering research. Trans.
Assoc. Comput. Linguistics , 7:452–466, 2019. doi: 10.1162/TACL\_A\_00276. URL https:
//doi.org/10.1162/tacl_a_00276 .
11

[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Na-
man Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-
Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-
12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html .
[21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Na-
man Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In
Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-
Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-
12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html .
[22] Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan, Shuo Wang, Shi Yu, Zheni Zeng, Hao Chen,
Ge Yu, Zhiyuan Liu, Maosong Sun, and Chenyan Xiong. RAG-DDR: optimizing retrieval-
augmented generation using differentiable data rewards. CoRR , abs/2410.13509, 2024. doi:
10.48550/ARXIV .2410.13509. URL https://doi.org/10.48550/arXiv.2410.13509 .
[23] Yi Liu, Lianzhe Huang, Shicheng Li, Sishuo Chen, Hao Zhou, Fandong Meng, Jie Zhou,
and Xu Sun. RECALL: A benchmark for llms robustness against external counterfactual
knowledge. CoRR , abs/2311.08147, 2023. doi: 10.48550/ARXIV .2311.08147. URL https:
//doi.org/10.48550/arXiv.2311.08147 .
[24] OpenAI. GPT-4 technical report. CoRR , abs/2303.08774, 2023. doi: 10.48550/ARXIV .2303.
08774. URL https://doi.org/10.48550/arXiv.2303.08774 .
[25] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100, 000+
questions for machine comprehension of text. In Jian Su, Xavier Carreras, and Kevin Duh,
editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language
Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016 , pages 2383–2392. The
Association for Computational Linguistics, 2016. doi: 10.18653/V1/D16-1264. URL https:
//doi.org/10.18653/v1/d16-1264 .
[26] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-
networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages
3982–3992, Hong Kong, China, November 2019. Association for Computational Linguistics.
doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/ .
[27] Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25
and beyond. Found. Trends Inf. Retr. , 3(4):333–389, 2009. doi: 10.1561/1500000019. URL
https://doi.org/10.1561/1500000019 .
[28] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. DRAGIN: dynamic
retrieval augmented generation based on the real-time information needs of large language
models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
ACL 2024, Bangkok, Thailand, August 11-16, 2024 , pages 12991–13013. Association for
Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.702. URL https:
//doi.org/10.18653/v1/2024.acl-long.702 .
[29] Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David
Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, and Jimmy
Lin. "knowing when you don’t know": A multilingual relevance assessment dataset for robust
retrieval-augmented generation. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen,
editors, Findings of the Association for Computational Linguistics: EMNLP 2024, Miami,
12

Florida, USA, November 12-16, 2024 , pages 12508–12526. Association for Computational
Linguistics, 2024. URL https://aclanthology.org/2024.findings-emnlp.730 .
[30] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. Rbft: Robust fine-tuning for
retrieval-augmented generation against retrieval defects. CoRR , abs/2501.18365, 2025. doi:
10.48550/ARXIV .2501.18365. URL https://doi.org/10.48550/arXiv.2501.18365 .
[31] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan
Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.
CoRR , abs/2212.03533, 2022. doi: 10.48550/ARXIV .2212.03533. URL https://doi.org/
10.48550/arXiv.2212.03533 .
[32] Sijia Wang and Lifu Huang. Targeted augmentation for low-resource event extraction. In Kevin
Duh, Helena Gómez-Adorno, and Steven Bethard, editors, Findings of the Association for Com-
putational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024 , pages 4414–4428.
Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-NAACL.
275. URL https://doi.org/10.18653/v1/2024.findings-naacl.275 .
[33] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. Self-knowledge guided retrieval augmentation
for large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of
the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023 ,
pages 10303–10315. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.
FINDINGS-EMNLP.691. URL https://doi.org/10.18653/v1/2023.findings-emnlp.
691.
[34] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md. Rizwan Parvez, and Graham Neubig. Learning
to filter context for retrieval-augmented generation. CoRR , abs/2311.08377, 2023. doi: 10.
48550/ARXIV .2311.08377. URL https://doi.org/10.48550/arXiv.2311.08377 .
[35] Jason W. Wei and Kai Zou. EDA: easy data augmentation techniques for boosting performance
on text classification tasks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors,
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019,
Hong Kong, China, November 3-7, 2019 , pages 6381–6387. Association for Computational
Linguistics, 2019. doi: 10.18653/V1/D19-1670. URL https://doi.org/10.18653/v1/
D19-1670 .
[36] Zhepei Wei, Wei-Lin Chen, and Yu Meng. Instructrag: Instructing retrieval-augmented genera-
tion with explicit denoising. CoRR , abs/2406.13629, 2024. doi: 10.48550/ARXIV .2406.13629.
URL https://doi.org/10.48550/arXiv.2406.13629 .
[37] Jinyang Wu, Feihu Che, Chuyuan Zhang, Jianhua Tao, Shuai Zhang, and Pengpeng Shao.
Pandora’s box or aladdin’s lamp: A comprehensive analysis revealing the role of RAG noise
in large language models. CoRR , abs/2408.13533, 2024. doi: 10.48550/ARXIV .2408.13533.
URL https://doi.org/10.48550/arXiv.2408.13533 .
[38] Yuan Xia, Jingbo Zhou, Zhenhui Shi, Jun Chen, and Haifeng Huang. Improving retrieval
augmented language model with self-reasoning. In Toby Walsh, Julie Shah, and Zico Kolter,
editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence,
February 25 - March 4, 2025, Philadelphia, PA, USA , pages 25534–25542. AAAI Press,
2025. doi: 10.1609/AAAI.V39I24.34743. URL https://doi.org/10.1609/aaai.v39i24.
34743 .
[39] Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, and Jie Zhou.
Unsupervised information refinement training of large language models for retrieval-augmented
generation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024 , pages 133–145. Association
for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.9. URL https:
//doi.org/10.18653/v1/2024.acl-long.9 .
13

[40] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented
generation. CoRR , abs/2401.15884, 2024. doi: 10.48550/ARXIV .2401.15884. URL https:
//doi.org/10.48550/arXiv.2401.15884 .
[41] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
and et al. Qwen2.5 technical report. CoRR , abs/2412.15115, 2024. doi: 10.48550/ARXIV .2412.
15115. URL https://doi.org/10.48550/arXiv.2412.15115 .
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhut-
dinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-
hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi
Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Lan-
guage Processing, Brussels, Belgium, October 31 - November 4, 2018 , pages 2369–2380.
Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1259. URL
https://doi.org/10.18653/v1/d18-1259 .
[43] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented
language models robust to irrelevant context. In The Twelfth International Conference on
Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024.
URL https://openreview.net/forum?id=ZS4m74kZpH .
[44] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang,
and Dong Yu. Chain-of-note: Enhancing robustness in retrieval-augmented language models.
In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024 , pages 14672–14685. Association for Computational Linguistics,
2024. URL https://aclanthology.org/2024.emnlp-main.813 .
[45] Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu, and Maosong
Sun. Word-level textual adversarial attacking as combinatorial optimization. In Dan Jurafsky,
Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020 ,
pages 6066–6080. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.
ACL-MAIN.540. URL https://doi.org/10.18653/v1/2020.acl-main.540 .
14

A Domain-wise Noise Distribution Analysis
Figure 7 presents a detailed analysis of the distribution of retrieved document types across different
retrievers. We observe that both the proportions and the trends of different noise types vary substan-
tially with the number of retrieved passages and across domains. For instance, the fraction of golden
documents consistently decreases as more passages are retrieved, while the ratios of various noise
types ( e.g., Irrelevant, Inconsequential) increase, but the magnitude and pattern of these changes are
highly domain- and retriever-dependent. More notably, no universal trend governs the evolution of
each noise category—distinct domains exhibit markedly different noise profiles, and even within
a domain, different retrievers lead to divergent noise dynamics. This pronounced heterogeneity
underscores the limitation of static robustness benchmarks, which typically assume a fixed or uniform
noise distribution and thus only reflect specific scenarios. Consequently, such benchmarks fail to
capture the complexity and variability inherent in real-world retrieval environments, potentially
leading to misleading assessments of model robustness.
BM25 e5 Hybrid
5 10 15 20 25 3010 %15 %20 %25 %30 %35 %
# Retrieved PassagesGolden Ratio
5 10 15 20 25 305.4 %5.6 %5.8 %6.0 %6.2 %
# Retrieved PassagesDistracting Ratio
5 10 15 20 25 309.9 %10.1 %10.3 %10.5 %10.7 %
# Retrieved PassagesLow Quality Ratio
5 10 15 20 25 3025 %27 %29 %31 %
# Retrieved PassagesInconsequential Ratio
5 10 15 20 25 3020 %25 %30 %35 %40 %45 %
# Retrieved PassagesIrrelevant Ratio
BM25 e5 Hybrid
5 10 15 20 25 3020 %30 %40 %
# Retrieved PassagesGolden Ratio
5 10 15 20 25 306%8%10 %
# Retrieved PassagesDistracting Ratio
5 10 15 20 25 309.0%9.5%10 .0%10 .5%
# Retrieved PassagesLow Quality Ratio
5 10 15 20 25 3030 %32 %34 %
# Retrieved PassagesInconsequential Ratio
5 10 15 20 25 3010 %15 %20 %25 %30 %
# Retrieved PassagesIrrelevant Ratio
5 10 15 20 25 3010 %15 %20 %25 %30 %35 %
# Retrieved PassagesGolden Ratio
5 10 15 20 25 304.3%4.6%4.9%5.2%5.5%
# Retrieved PassagesDistracting Ratio
5 10 15 20 25 308.9%9.4%9.9%
# Retrieved PassagesLow Quality Ratio
5 10 15 20 25 3020 %22 %24 %26 %
# Retrieved PassagesInconsequential Ratio
5 10 15 20 25 3030 %35 %40 %45 %50 %
# Retrieved PassagesIrrelevant Ratio
Figure 7: Domain-wise distribution of retrieved document types as the number of retrieved passages
increases, across three retrieval methods (BM25, e5, Hybrid). The top row shows aggregate statistics
across all topics; the middle and bottom rows show the distributions for the History andLiterature
& Arts domains, respectively.
B Prompt Design
B.1 Benchmark Construction Prompts
This appendix provides a comprehensive overview of all prompts used in Section 3 (Benchmark
Construction) and Section 4 (Experiments). Figure 9 and Figure 12 present the prompts used to
augment golden documents in single-hop and multi-hop settings, respectively. These prompts aim to
produce alternative phrasings of ground-truth evidence passages that remain semantically faithful to
the original, enriching the gold context without introducing factual variance. Figure 10 and Figure 13
show the prompts designed to generate distracting noise documents in the single-hop setting and
multi-hop setting, respectively. These documents intentionally relate to the question topic but lead
the model to incorrect conclusions. They simulate realistic, misleading content retrieved by imperfect
retrieval systems. Figure 11 and Figure 14 illustrate the prompts used to create low quality noise with
counterfactual information. These prompts replace the correct entities with semantically plausible yet
factually incorrect information. The rationale for retrieval noise taxonomy is illustrated in Figure 8.
15

Figure 8: Schematic diagram illustrating the rationale for retrieval noise taxonomy.
Single-Hop Dataset: Golden Documents Augmentation
Task Requirements:
Given a Question, a Short Answer and a Golden Document,
generate 10 new Golden Documents (1-10) based on the following Principle:
a) You can delete some part of the given Golden Document and add some new details,
but you must retain the correct answer in the generate new Golden Documents.
b) Reorganize the sentence structure of the entire Document (difference>80%),
and change the active and passive voice/modifier position/rhetorical structure...
c) Ensure document lengths match original through,
± 15% word count variation from Golden Document.
d) Output strictly in the following example format,
including all necessary quotes and escape characters.
Question:
<question>
Short Answer:
<corresponding answer to the question>
Golden Document:
<origin golden document>
Example Input:
Question: where is zimbabwe located in the world map?
Short Answer: in southern Africa, between the Zambezi and Limpopo Rivers...
Golden Document: Zimbabwe, officially the Republic of Zimbabwe, is a landlocked country
located in southern Africa, between the Zambezi and Limpopo Rivers, bordered by...
Example Output:
{
1: "Zimbabwe, a landlocked nation in southern Africa, lies between the Zambezi and
Limpopo Rivers. It shares borders with South Africa, Botswana, Zambia, and...",
...
6: "Surrounded by South Africa, Botswana, Zambia, and Mozambique, Zimbabwe is a
landlocked country in ... It stretches between the Zambezi and Limpopo Rivers...",
...
10: "In the heart of southern Africa lies Zimbabwe, a landlocked country nestled between
the Zambezi and Limpopo Rivers and sharing boundaries with..."
}
Figure 9: Prompt for golden document augmentation in the single-hop dataset
16

Single-Hop Dataset: Distracting Noise Introduction
Task Requirements:
1. Given a Question, a Short Answer and a Golden Document, generate 10 Distracting
Documents through these stages:
a) Core Entity Substitution:
•Identify question-critical entities in Short Answer
•Replace with same-domain but incorrect entities using: 1-Theoretical: Keep discipline
but change theory (e.g., "relativity →quantum mechanics") 2-Event-based: Maintain
event type but shift spatiotemporal coordinates (e.g., "2019 →2023")
b) Semantic Restructuring (>85% textual divergence):
•Toggle active/passive voice
•Each generated Distracting Documents must be different in structure and sentence
pattern.
•Reconfigure rhetorical patterns (e.g., convert causal chains to parallel structures)
•Reposition modifiers (e.g., transform prepositional phrases to relative clauses)
c) Logical Consistency Verification: 1. Ensure new entities contextually align with: - Geo-
graphic/chronological parameters - Domain-specific terminology - Quantitative relationships
2. Preserve non-critical authentic details from Golden Document for Distracting Documents
3. Eliminate: - Cross-dimensional substitutions (animal →architecture terms) - Numerical
contradictions (e.g., mismatched magnitude)
2. Output Format:
•Generate 10 Distracting Documents (d0-d9)
•Each must contain question-related erroneous core entities
•Maintain original JSON structure with proper escaping
3. Critical Avoidances:
•Cross-category entity swaps
•Internal chronological conflicts
•Commonsense violations (e.g., "fish climb trees")
Question:
<question>
Short Answer:
<corresponding answer to the question>
Golden Document:
<origin golden document>
Example Input:
...
Example Output:
...
Figure 10: Prompt for distracting noise introduction in the single-hop dataset
17

SingleHop - Low Quality Noise Generation
Task Requirements:
1. Given 7 paragraphs and semantically related Question, please generate a fine-tuned
paragraph for each paragraph, by modifying relevant entities or adding relevant expressions to
make it contain incorrect knowledge, while ensuring that they are still semantically relevant
to the question.
2. Each generated paragraph must meet the following requirements:
•Contains incorrect information, but should not violate common sense (for example: do not
make obvious mistakes such as "birds swim in water").
•No internal logical errors are allowed (for example: "The 2019 AFC Asian Cup was held
in the United Arab Emirates, and Japan was announced as the host country on March 9,
2015" is self-contradictory).
• Keep the paragraph fluent and grammatically correct.
• Try not to include negative sentences in the output content.
3. The generated paragraph needs to be semantically relevant to the question, but should not
include sentences that can directly answer the question.
4. Please output strictly in the following example format, including all necessary quotes and
escape characters.
Question:
<question>
Documents:
<related document 1>, <related document 2>,..., <related document 7>
Example Input:
Question: Where is the capital of France?
Document 1: Paris, known for its famous landmarks like the Eiffel Tower...
...
Document 7: France retains its centuries-long status as a global centre of art, science, and
philosophy. It hosts the fourth-largest number of UNESCO World Heritage Sites and is the
world’s leading tourist....
Example Output:
{
1: Lyon, known for its famous landmarks like the Eiffel Tower...
...
7: France retains its centuries-long status as a global centre of art, science, and
philosophy. It hosts the largest number of UNESCO World Heritage Sites and is the
world’s leading tourist....
}
Figure 11: Prompt for low quality noise introduction in the single-hop dataset
18

Multi-Hop Dataset: Golden Documents Augmentation
Task Requirements:
Please generate <#document number> documents based on the input document below. Each
generated document should have the same meaning as the input document but be rephrased dif-
ferently. Return the results in JSON format where the keys are integers from 1 to <#document
number>, and the values are the rephrased documents.
Document:
<origin golden document>
Example Input:
The Radio station currently plays a mix of Hindi and Regional music.
Example Output:
{
1: "The Radio station is broadcasting a combination of Hindi and Regional music.",
2: "A mix of Hindi and Regional music is being played on the Radio station.",
3: "The Radio station features both Hindi and Regional music in its current playlist."
}
Figure 12: Prompt for golden document augmentation in the multi-hop dataset
Multi-Hop Dataset: Distracting Noise Introduction
Task Requirements:
Please generate <#document number> documents based on the input document below. Each
generated document should have a different meaning from the input document, potentially
leading to incorrect interpretations. You can achieve this by modifying key words or phrases
in the document while maintaining its overall structure. Return the results in JSON format
where the keys are integers from 1 to <#document number>, and the values are the modified
documents.
Guidelines:
- Change the meaning of the document by altering important words or phrases.
- Ensure that the new documents are grammatically correct and coherent.
- Avoid generating documents that are too similar to the input document.
Document:
<origin golden document>
Example Input:
The Radio station currently plays a mix of Hindi and Regional music.
Example Output:
{
1: "The Radio station currently plays only Hindi music.",
2: "The Radio station stopped playing Regional music.,
3: "The Radio station focuses exclusively on English and International music.
}
Figure 13: Prompt for distracting noise introduction in the multi-hop dataset
19

Multi-Hop Dataset: Low Quality Noise Introduction
Task Requirements:
Please change the correct information in the input document into wrong information. The
output should be a document that is similar to the input but contains incorrect information.
Return the result document directly without any explaination.
Document:
<origin golden document>
Example Input:
The Radio station currently plays a mix of Hindi and Regional music.
Example Output:
The Radio station currently plays a mix of English and Regional music.
Figure 14: Prompt for low quality noise introduction in the multi-hop dataset
B.2 Experimental Prompts
Figure 15 presents the inference prompt used in the experiment without employing RAG, where no
external documents are incorporated. Figure 16 illustrates the prompt utilized in the VANILLA RAG
framework, which integrates retrieved documents directly into the prompt. Figure 17 shows the
prompt used within the CHAINOF NOTE framework, which guides the generator to analyze the
retrieved information incrementally before proceeding to answer generation. Figure 18 displays the
prompt applied in the SKR framework, which is designed to assess whether external knowledge
assistance is required.
NoRAG - Inference Prompt
Task Description:
1. Answer the given Question Directly,
do NOT add any explanations when giving the response.
2. If you cannot answer with certainty due to insufficient information,
you MUST respond verbatim: “I cannot answer the question.”
Question: <question>
Figure 15: Inference Prompt of N ORAG
VanillaRAG - Inference Prompt
Task Description:
1. Answer the given Question based on the Retrieval Documents,
do NOT add any explanations when giving the response.
2. If you cannot answer with certainty due to insufficient information,
you MUST respond verbatim: “I cannot answer the question.”
Question: <question>
Retrieval Documents: <retrieval documents>
Figure 16: Inference Prompt of V ANILLA RAG
20

ChainofNote - Inference Prompt
Task Description:
1. Read the given Question and Retrieval Documents to gather relevant information.
2. Write reading notes summarizing the key points from these passages.
3. Discuss the relevance of the given question and Wikipedia passages.
4. If some passages are relevant to the given question,
provide a brief answer based on the passages.
5. If no passage is relevant, directly provide answer without considering the passages.
6. If you cannot answer with certainty due to insufficient information,
you MUST respond verbatim: “I cannot answer the question.”
Question: <question>
Retrieval Documents: <retrieval documents>
Figure 17: Inference Prompt of C HAINOF NOTE
SKR - Retrieval Check Prompt
Task Description:
Do you need additional information to answer this question?
If you need, please answer: “Yes, I need.”
If you don’t need, please answer: “No, I don’t need.”
Do not answer questions and explain reasons.
Question: <question>
Figure 18: Prompt to check whether LLM can solve the question without retrieval document in SKR
21

C Detailed Benchmark Statistics
This appendix presents detailed statistics of the Magic Mushroom benchmark. Magic Mushroom
comprises 7,468 single-hop and 3,925 multi-hop complex QA pairs. The topical distribution of all QA
instances is illustrated in Figure 19, spanning 7 major categories and 28 subcategories. Each QA pair
is assigned a set of candidate retrieval documents, including 10 Golden Documents, 10 Distracting
noise documents, and 7 exemplars each for Low Quality, Inconsequential, and Irrelevant noise types.
During testing, retrieval documents of specific noise types and proportions can be randomly sampled
from the candidate set to accommodate diverse research needs. Furthermore, Magic Mushroom
is divided into MM s(single-hop) and MM m(multi-hop) subsets. The distribution of retrieval
document lengths for each subset is shown in Figure 20. In MM s, retrieval document lengths are
mainly distributed between 38 and 104 tokens, while in MM m, they range from 36 to 116 tokens,
showing minimal difference between the two. Researchers may also combine MM sandMM mfor
more comprehensive studies.
Natural 
Landscape
(2.87%)
Ancient History
(1.84%)Modern History
(1.75%)Physics
(1.35%)Philosophy
(1.83%)Biography
Historical Figures
(3.25%)Famous 
Authors
(2.24%)
Political Leaders
(1.91%)
Artists
(1.37%)Philosophers
(1.15%)Scientists
(1.32%)Geography
Countries
(6.79%)
Cities
(4.79%)History
Wars & Conflicts
(6.65%)Historical Events
(5.59%)Science
Biology
(4.99%)Chemistry
(0.47%)Astronomy
(1.67%)Mathematics
(1.50%)Literature and Arts
Painting & Sculpture
(5.73%)Classical Music
(4.85%)Literature
(7.56%)
Theater & Performing Arts
(4.79%)Architecture
(0.58%)Sports
Team Sports
(9.09%)Individual Sports
(8.18%)
Extreme Sports
(0.88%)
Others
Others
(5.01%)
Figure 19: Topical distribution of QA instances in Magic Mushroom, with block area and percentage
labels indicating category proportions.
15 65 115 165 215 265 3150%1%2%3%
# Document T okens (In Intervals of 2)Ratio
15 45 75 105 135 165 1950%1%2%3%
# Document T okens (In Intervals of 2)Ratio
Figure 20: Retrieval document length distributions for MM s(left) and MM m(right).
D Details of Evaluation Metrics
D.1 Prompts for Evaluation Metrics
To better evaluate the correctness ( Cor.) in RAG systems, we adopt an LLM-based scoring prompt,
illustrated in Figure 21, instead of traditional lexical metrics such as EM and F1[ 25]. This prompt
instructs the model to assess the factual accuracy and completeness of generated answers against gold
answers, addressing the evaluation limitations of surface-level matching in QA tasks.
22

Prompt for the Evaluation of Correctness
You are an evaluator tasked with scoring the Candidate Answer.
Instructions:
1. Compare the Candidate Answer to the Correct Answer in the context of the Question.
2. Assign a score from 0 to 5 based on accuracy and completeness:
Score 0: Completely incorrect or irrelevant.
Score 1: Partially correct but contains significant errors.
Score 3: Moderately correct but lacks some details or precision.
Score 5: Fully correct and matches the Correct Answer.
3. Output ONLY the score as a single number (e.g. “3”). Do not include any explanations.
Question: <question>
Correct Answer: <correct answer>
Candidate Answer: <llm answer>
Figure 21: Prompt for the Evaluation of Correctness
D.2 Human-AI Alignment
0.0 0.1 0.2 0.3 0.5 0.7 0.90.10.40.71.0
Noise RatioCorrectnessGPT-4 Human
Figure 22: Comparison of alignment
scores between human and GPT-4.To validate the alignment between LLM-based automatic
evaluation and human judgment, we conduct an align-
ment analysis on the performance of VANILLA RAG +
LLama-3.1 8Bat different noise ratios. We randomly sam-
ple 200 instances for each noise ratio and compute the
average correctness scores assigned independently by hu-
man evaluators and the GPT-4. As shown in Figure 22, the
scores exhibit high similarity, indicating strong alignment
between LLM-based and manual assessments. These find-
ings support the effectiveness of the LLM-based evalua-
tion framework as a reliable surrogate for human judgment
in large-scale experimental settings.
E Baseline Details
The choice of baseline generators and denoising strategies is motivated by the need to evaluate
RAG robustness under diverse noisy retrieval scenarios comprehensively. For LLM generators, we
include a wide range of models differing in architecture and scale, such as Llama-3, Qwen-2.5, and
DeepSeek series, as well as distilled variants ( e.g., R1-Distill-Llama). For denoising strategies, our
selection covers several representative paradigms in the literature. VANILLA RAG serves as the
basic RAG architecture without explicit denoising, acting as a baseline for robustness assessment.
CHAINOF NOTE exemplifies chain-of-thought prompting, guiding the model to produce intermediate
reasoning steps for enhanced semantic stability. DRAGIN and SKR adopt adaptive retrieval based on
generation confidence, dynamically controlling retrieval to mitigate noise injection. CRAG employs
pre-generation verification and refinement of retrieved content and optional web augmentation to
block noise propagation and improve system robustness. To ensure fairness, we do not include
fine-tuning-based denoising methods in the main evaluation, as such strategies require access to
training data. Nevertheless, Magic Mushroom provides a dedicated development set to facilitate
future research on fine-tuned denoising approaches. Appendix B provides all the prompts used in this
study. All configuration settings for generators and denoising strategies follow the optimal settings
reported in the original studies. Notably, for CRAG, the thresholds for triggering the three actions
(correct, ambiguous, and incorrect) are set to (0.5,−0.91). All reported experimental results are
obtained by averaging over 3 independent runs to ensure statistical reliability.
23

F Supplementary Main Experimental Results
F.1 Detailed Test Subset
We randomly sample k= 10 retrieval documents for each test instance from the pool of candidate
documents according to the specified noise ratio. For example, when the noise ratio is set to 30%, 7
golden documents are randomly selected from the corresponding golden set, and 3 noise documents
are randomly sampled from the pool of noise documents. For noise type analysis, kis set to 8 at a
noise ratio of 0.9 and 7 at a noise ratio of 1. As the selection is random, the proportions of different
noise types follow the law of large numbers; that is, the ratio among Distracting Noise and the other
three noise types (Low Quality, Inconsequential, Irrelevant) is approximately 10:7:7:7. Researchers
may further customize the types and proportions of injected noise to suit specific experimental
requirements.
F.2 Complete Results of Main Experiments
Table 5: Performance of NoRAG ( MM m).
LLM G ENERATORNORAG
Cor. Rej.
Llama-3.2 1B 5.9 79.3
Llama-3.1 8B 25.1 51.7
Llama-3.1 70B 47.7 14.3
R1-Distill-Llama 8B 27.9 17.5
R1-Distill-Llama 70B 51.1 12.7Table 6 presents the complete results of RAG-based
systems at different noise ratios on the MM s. Notably,
Llama-3.2 1Band Qwen-2.5 1.5B, the smallest models
in their respective series, exhibit distinct behaviors.
Llama-3.2 1Bis significantly more conservative than
other Llama variants, frequently abstaining from an-
swering—for instance, VANILLA RAG +Llama-3.2 1B
shows a rejection rate of 34.1% even at a 0% noise
ratio. This tendency is mitigated only by introducing
chain-of-thought prompting. In contrast, Qwen-2.5 1.5B
achieves correctness comparable to much larger gener-
ators under low-noise conditions, but is prone to severe
hallucinations as noise levels rise. Tables 5 and 7 sum-
marize the results for all systems on the MMm dataset across varying noise ratios. In the multi-hop
QA setting, we observe that the parameter scale of the generator is a key factor for robustness:
larger models consistently achieve higher resistance to retrieval noise, possibly due to their enhanced
reasoning capabilities and greater contextual modeling capacity.
F.3 Multi-perspective Analysis of Noise Effects
To provide a more fine-grained analysis of RAG robustness under noise, we introduce three novel
evaluation metrics from different perspectives: (1) Hallucination Rate (H) : the degree to which an-
swer correctness decreases when retrieval information is introduced to instances that were previously
answered correctly without retrieval; (2) Confusion Rate (C) : the proportion of instances where the
model abstains from answering after retrieval, despite being able to answer correctly without retrieval;
and (3) Rectification Rate (R) : the degree to which answer correctness increases when retrieval
information is introduced to instances that were previously answered incorrectly. These metrics
enable a more direct assessment of the nuanced effects of retrieval noise on RAG systems. Complete
results are reported in Table 8. Empirical analysis shows that chain-of-thought prompting significantly
improves retrieval utility under low-noise conditions but also raises the risk of hallucination as noise
increases. In line with Section 4.3, we observe a sharp drop in Rectification Rate when the noise ratio
reaches 50%. Furthermore, while SKR effectively limits hallucinations under noisy conditions, it
also constrains performance in clean scenarios. Overall, these findings underscore the difficulty of
balancing the benefits of retrieval augmentation with effective noise mitigation in RAG systems.
24

Table 6: Complete Results of RAG-based Systems at Different Noise Ratios on the MM s.
LLM G ENERATOR0% 10% 30% 50% 70% 90% 100%
Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej.
VANILLA RAG [2]
Llama-3.2 1B 41.7 34.1 42.3 34.6 43.0 31.2 41.6 30.1 31.7 30.2 19.6 33.9 7.6 32.9
Qwen-2.5 1.5B 88.0 0.6 84.6 0.4 82.5 0.7 77.3 0.7 62.2 1.1 30.9 2.3 10.5 2.7
Llama-3.1 8B 84.1 2.4 81.1 2.9 76.8 3.0 73.9 2.7 59.9 3.4 44.9 4.7 18.0 5.9
Qwen-2.5 7B 72.7 13.8 70.5 14.7 67.7 15.4 62.9 18.3 41.6 30.7 27.5 37.9 7.4 45.2
Llama-3.1 70B 81.9 6.8 81.0 6.2 77.5 5.5 75.0 5.4 68.3 6.2 58.8 8.3 28.9 16.8
Qwen-2.5 72B 84.3 9.3 83.8 8.5 82.9 8.0 79.8 8.8 74.2 12.5 52.2 25.4 14.2 45.3
DeepSeek-V3 84.8 8.9 84.7 7.9 84.4 6.8 82.5 6.5 77.1 8.4 61.9 12.8 17.5 30.8
DeepSeek-R1 85.2 7.8 84.9 7.1 83.9 6.4 81.5 8.7 75.1 12.0 57.9 26.6 17.9 48.6
R1-Distill-Llama 8B 84.1 5.6 81.7 5.6 78.7 5.1 74.1 5.2 64.1 7.5 45.2 11.5 22.7 13.0
R1-Distill-Llama 70B 84.2 7.5 82.9 6.7 82.7 5.8 80.4 6.8 72.2 10.7 57.6 18.6 28.0 30.8
CHAINOF NOTE [44]
Llama-3.2 1B 76.9 6.4 74.8 5.6 70.3 6.7 64.4 7.6 52.3 6.6 26.2 8.3 11.3 9.3
Qwen-2.5 1.5B 83.5 3.7 81.5 4.1 76.4 6.2 70.8 7.9 53.1 12.9 23.1 18.3 9.4 24.0
Llama-3.1 8B 87.1 3.5 83.2 3.5 80.2 2.4 74.9 1.7 63.7 1.5 45.2 2.1 14.5 2.9
Qwen-2.5 7B 76.7 9.3 74.2 9.3 70.8 10.5 64.6 9.1 52.9 12.2 31.2 15.2 10.4 17.9
Llama-3.1 70B 85.7 4.7 85.1 3.7 82.5 3.5 79.6 3.6 72.7 4.2 56.7 5.3 24.1 9.3
Qwen-2.5 72B 84.4 7.6 83.2 6.8 80.5 6.2 75.5 5.1 67.1 6.0 45.1 11.7 16.9 21.3
DeepSeek-V3 85.5 8.4 83.8 7.1 82.3 7.2 79.7 7.1 71.2 10.5 53.8 19.4 21.8 29.7
DeepSeek-R1 89.1 5.1 88.3 4.1 88.0 3.3 86.1 3.5 81.4 4.4 72.4 11.4 27.5 31.6
R1-Distill-Llama 8B 84.4 4.0 81.2 3.0 76.7 2.0 70.6 1.8 60.7 2.5 46.2 4.0 27.6 4.8
R1-Distill-Llama 70B 85.2 5.4 83.5 4.3 82.9 3.1 79.0 2.7 73.7 4.1 63.4 6.6 34.1 11.6
DRAGIN [28]
Llama-3.2 1B 19.3 61.7 19.0 62.1 19.1 61.6 18.9 59.8 16.8 60.9 12.7 61.6 8.7 60.2
Qwen-2.5 1.5B 54.9 9.9 53.0 9.9 51.1 9.0 49.5 9.6 42.3 9.7 27.9 10.5 16.8 10.7
Llama-3.1 8B 57.0 16.7 56.2 15.9 54.6 16.4 53.1 15.4 49.3 16.8 41.4 15.7 30.9 17.8
Qwen-2.5 7B 20.3 66.8 20.7 66.2 20.3 66.1 20.2 66.2 17.4 68.3 16.3 68.2 14.2 68.4
R1-Distill-Llama 8B 56.6 7.8 55.4 9.2 55.8 6.8 55.8 6.4 50.5 7.2 35.7 8.8 21.9 9.7
SKR [33]
Llama-3.2 1B 11.2 57.8 12.0 57.6 12.4 55.5 11.6 55.1 11.9 55.8 11.8 56.6 11.5 55.4
Qwen-2.5 1.5B 31.6 8.5 30.3 8.9 29.6 8.4 30.9 8.6 28.2 10.2 20.1 9.7 15.8 9.1
Llama-3.1 8B 41.5 11.8 40.8 12.5 40.3 12.3 39.1 13.6 38.0 13.9 38.4 13.9 36.7 13.9
Qwen-2.5 7B 25.5 55.8 23.8 55.8 23.8 56.6 23.1 56.5 21.3 57.5 19.5 58.7 17.1 60.2
Llama-3.1 70B 62.7 5.1 61.1 4.8 60.6 4.4 58.3 4.5 57.8 4.8 55.9 4.8 49.4 7.3
Qwen-2.5 72B 52.6 22.5 53.5 22.2 53.2 21.9 52.6 22.5 50.8 24.0 45.3 27.1 41.2 27.2
DeepSeek-V3 53.8 16.3 54.2 16.5 54.4 16.7 54.7 14.9 53.5 16.1 52.7 17.1 50.2 17.2
DeepSeek-R1 62.4 8.1 63.7 7.5 62.6 8.2 63.5 7.1 61.6 9.1 60.4 9.4 57.4 11.6
R1-Distill-Llama 8B 43.9 11.6 45.4 10.8 41.7 12.3 42.3 9.9 37.8 12.0 30.9 14.1 25.9 15.2
R1-Distill-Llama 70B 60.0 7.9 61.1 5.9 61.2 6.2 58.7 7.3 57.2 8.1 55.4 8.9 51.4 10.6
CRAG [40]
Llama-3.1 8B 70.9 11.5 70.6 9.8 68.2 9.2 65.4 9.1 55.4 11.1 37.3 14.1 21.7 15.6
Qwen-2.5 7B 58.7 26.9 59.1 25.5 57.5 23.7 52.9 25.1 38.3 33.4 24.1 41.9 10.0 46.6
Llama-3.1 70B 73.4 9.1 75.7 8.9 72.9 7.9 70.2 7.7 62.6 9.3 51.1 11.4 33.9 15.7
Qwen-2.5 72B 70.8 23.4 72.9 19.9 72.1 18.9 68.5 18.3 58.5 26.6 33.9 42.1 13.6 53.3
DeepSeek-V3 71.7 20.3 73.8 17.8 72.9 16.7 71.0 15.6 60.2 21.7 39.6 31.1 15.8 40.5
DeepSeek-R1 72.6 18.5 74.1 16.1 73.9 15.4 73.1 14.7 60.5 23.6 37.5 38.0 17.8 47.4
R1-Distill-Llama 8B 71.3 12.8 71.5 11.3 68.0 11.1 63.1 9.6 49.4 13.0 34.0 17.4 22.0 17.9
R1-Distill-Llama 70B 73.9 13.8 73.7 13.7 73.4 12.1 71.4 10.9 62.1 16.3 41.9 28.1 25.9 32.8
25

Table 7: Performance of RAG-based Systems at Different Noise Ratios on the MM m.
LLM G ENERATOR0% 20% 40% 60% 80% 100%
Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej. Cor. Rej.
VANILLA RAG [2]
Llama-3.2 1B 34.9 34.3 35.6 31.3 35.2 33.2 35.4 31.2 31.9 31.2 26.6 29.2
Llama-3.1 8B 84.1 3.0 83.1 3.5 81.3 3.8 78.3 4.8 68.7 4.0 51.2 11.5
Llama-3.1 70B 90.9 2.5 91.4 1.5 91.6 1.8 92.3 1.2 80.6 1.8 65.9 5.0
R1-Distill-Llama 8B 83.9 7.0 85.1 5.3 86.0 3.2 84.6 2.3 68.6 2.0 44.6 6.7
R1-Distill-Llama 70B 89.0 6.3 87.9 6.0 89.7 3.5 89.8 2.8 76.9 2.8 55.1 13.2
CHAINOF NOTE [44]
Llama-3.2 1B 67.5 4.0 64.7 4.5 62.1 5.2 57.8 6.5 53.0 5.3 37.7 6.8
Llama-3.1 8B 85.5 4.7 86.0 3.2 86.7 2.3 72.8 2.2 64.5 1.5 59.7 4.3
Llama-3.1 70B 88.5 4.5 90.9 2.5 89.9 2.7 82.0 1.8 80.8 2.3 68.7 7.2
R1-Distill-Llama 8B 83.3 4.3 81.4 3.7 79.2 1.7 68.7 1.5 62.5 1.0 50.7 5.5
R1-Distill-Llama 70B 88.1 4.7 89.2 3.2 87.0 2.3 76.8 1.3 73.5 1.8 60.3 5.2
Table 8: Multi-perspective analysis of retrieval noise effects on RAG systems, reporting Hallucination
Rate (H), Confusion Rate (C), and Rectification Rate (R) at different noise ratios on the MM s.
LLM G ENERATOR0% 10% 30% 50% 70% 90% 100%
H C R H C R H C R H C R H C R H C R H C R
VANILLA RAG [2]
Llama-3.2 1B 2.0 2.5 35.0 1.9 2.6 35.6 2.1 1.7 35.7 2.0 2.3 34.8 3.7 1.7 25.9 4.5 1.8 14.8 6.3 2.3 5.1
Qwen-2.5 1.5B 1.0 0.0 69.6 1.6 0.0 66.7 1.4 0.1 64.5 3.0 0.1 60.9 4.6 0.0 47.4 9.9 0.1 21.5 14.4 0.3 5.7
Llama-3.1 8B 2.4 0.2 49.9 3.1 0.3 47.7 4.0 0.1 44.1 4.8 0.0 42.0 9.6 0.4 33.1 12.8 0.4 21.4 23.4 0.8 5.5
Qwen-2.5 7B 0.7 0.7 57.8 0.9 1.0 55.9 1.0 0.7 53.2 1.6 1.0 49.2 3.0 3.0 31.2 3.8 3.4 18.3 6.3 6.8 4.1
Llama-3.1 70B 2.3 1.4 31.0 2.7 0.9 30.0 3.3 1.0 27.1 3.7 1.2 25.3 6.5 1.7 21.8 10.6 1.9 16.5 23.0 8.0 5.3
Qwen-2.5 72B 0.9 1.4 45.6 1.3 1.4 45.4 1.3 1.3 44.4 1.8 1.4 41.9 2.4 1.9 37.4 4.4 5.9 21.5 13.5 18.0 4.7
DeepSeek-V3 1.4 1.8 36.4 1.6 1.2 36.0 2.0 1.2 36.0 2.5 1.2 34.5 3.1 2.1 30.8 7.1 3.8 21.3 23.1 15.1 4.1
DeepSeek-R1 1.0 1.5 29.3 1.4 1.8 29.7 1.7 1.7 28.9 2.1 2.4 27.5 3.0 3.9 23.5 3.4 12.7 15.6 14.7 28.8 2.9
R1-Distill-Llama 8B 1.3 0.8 58.6 1.6 0.7 56.5 1.6 0.5 53.3 2.1 1.1 49.8 4.2 1.3 42.1 6.4 2.1 26.2 13.9 2.5 11.5
R1-Distill-Llama 70B 1.5 1.9 33.7 1.8 1.8 32.6 1.7 1.5 31.9 2.5 1.9 30.8 4.0 3.3 25.4 6.6 7.0 15.5 16.9 14.7 5.4
CHAINOF NOTE [44]
Llama-3.2 1B 0.7 0.5 67.1 0.9 0.5 65.1 1.3 0.5 61.0 1.9 0.9 56.1 2.8 0.6 44.5 4.7 1.0 20.9 7.2 1.0 8.5
Qwen-2.5 1.5B 1.5 0.3 65.8 1.9 0.6 64.5 1.9 0.7 59.6 3.7 0.9 55.9 5.1 2.0 40.7 9.0 2.8 15.4 12.0 3.1 5.0
Llama-3.1 8B 1.1 0.4 52.0 2.2 0.4 49.1 3.1 0.2 46.8 4.4 0.2 42.8 7.5 0.5 34.9 13.5 0.3 22.2 27.2 0.5 5.4
Qwen-2.5 7B 0.8 0.7 61.9 1.0 0.5 59.3 1.0 0.3 55.8 1.8 0.4 50.4 3.6 0.7 41.0 6.2 0.9 22.0 10.1 2.0 6.1
Llama-3.1 70B 2.2 0.9 34.2 2.5 0.4 33.3 2.7 0.7 30.9 3.7 0.8 29.5 7.4 1.0 25.7 13.7 1.3 16.2 30.9 4.3 4.6
Qwen-2.5 72B 1.5 1.6 46.5 1.8 1.0 44.9 2.9 0.9 43.3 5.9 0.7 40.9 7.4 0.5 33.9 13.3 2.4 19.8 23.1 6.7 5.6
DeepSeek-V3 1.4 1.9 37.2 1.6 1.7 35.6 2.6 1.5 34.9 3.6 1.5 33.3 4.5 3.3 27.5 14.1 7.4 14.8 20.9 13.9 5.0
DeepSeek-R1 0.6 0.9 32.3 1.0 0.8 31.6 1.3 0.6 31.5 2.4 0.4 30.4 2.7 1.0 26.6 3.9 3.2 20.9 17.5 19.2 5.8
R1-Distill-Llama 8B 1.2 0.7 58.6 2.0 0.4 56.0 2.2 0.2 51.5 4.0 0.5 47.6 4.6 0.1 37.8 7.5 0.4 26.5 12.6 1.0 13.6
R1-Distill-Llama 70B 1.8 1.2 34.1 2.7 1.0 33.2 2.8 0.7 32.2 4.9 0.4 29.5 6.0 0.9 26.0 10.1 1.8 17.7 21.5 5.1 6.4
DRAGIN [28]
Llama-3.2 1B 1.5 2.5 12.1 1.6 2.4 11.9 2.1 2.3 12.5 2.0 2.1 11.8 2.1 2.0 9.8 2.7 2.2 6.4 3.4 2.2 3.2
Qwen-2.5 1.5B 2.0 0.3 37.8 2.3 0.3 36.1 3.1 0.3 35.0 2.9 0.3 33.1 3.5 0.4 26.7 5.4 0.4 14.3 8.1 0.5 5.9
Llama-3.1 8B 2.4 0.7 23.4 3.2 0.6 23.1 3.2 1.0 22.0 3.1 0.5 20.0 4.3 0.7 17.6 6.4 0.8 11.9 9.4 1.0 4.6
Qwen-2.5 7B 0.5 0.7 5.3 0.6 0.5 5.5 0.8 0.4 5.4 0.9 0.5 5.3 1.2 1.0 3.2 1.3 1.2 2.5 2.3 1.3 1.5
SKR [33]
Llama-3.2 1B 2.1 1.8 4.0 2.2 1.4 4.5 2.3 1.4 5.0 2.7 1.4 4.5 2.4 1.4 4.5 2.4 1.2 4.3 2.5 1.4 4.2
Qwen-2.5 1.5B 3.6 0.1 15.8 4.1 0.2 15.1 4.3 0.3 14.6 3.9 0.3 15.6 4.4 0.4 13.5 6.6 0.3 7.5 7.7 0.4 4.4
Llama-3.1 8B 4.3 0.6 9.7 4.8 0.6 9.5 4.5 0.4 8.5 4.8 0.5 7.7 5.1 0.7 7.1 4.6 0.6 6.9 5.1 0.8 5.8
Qwen-2.5 7B 0.9 0.3 10.4 1.4 0.3 9.2 1.2 0.4 9.1 1.3 0.4 8.5 1.2 0.4 6.5 1.0 0.4 4.7 1.0 0.7 2.5
Llama-3.1 70B 5.6 0.6 14.2 6.2 0.7 13.4 5.8 0.5 12.3 6.7 0.7 11.1 6.6 0.6 10.4 7.7 0.8 9.6 9.5 1.9 6.2
Qwen-2.5 72B 2.1 0.7 14.3 1.6 0.7 14.7 1.5 0.7 14.3 1.6 0.8 14.0 2.1 0.5 12.3 1.9 1.0 7.1 2.0 1.5 3.6
DeepSeek-V3 3.6 2.9 8.8 4.0 2.4 9.1 3.0 2.9 8.8 3.3 2.5 8.9 2.9 2.8 7.6 7.7 2.7 6.7 3.7 3.1 5.5
DeepSeek-R1 3.5 1.0 8.5 3.8 0.4 9.5 3.5 0.9 8.5 3.4 0.5 9.0 3.5 1.8 8.4 3.4 1.5 6.9 3.9 2.1 4.9
R1-Distill-Llama 8B 6.7 1.2 24.3 6.7 1.2 25.9 6.2 2.9 23.3 7.1 1.4 23.3 6.6 1.4 18.3 8.5 1.4 13.4 9.2 2.2 9.8
R1-Distill-Llama 70B 6.5 1.3 13.7 6.7 0.7 13.8 6.1 1.5 14.7 7.3 1.5 12.5 6.7 2.1 11.7 7.8 2.1 10.9 8.3 2.8 8.1
CRAG [40]
Llama-3.1 8B 3.4 2.1 39.8 4.7 1.8 40.3 5.1 1.5 38.0 5.8 1.2 35.5 8.3 2.1 29.0 14.2 2.6 17.4 18.4 3.5 6.9
Qwen-2.5 7B 0.8 2.3 45.5 1.5 2.3 46.5 1.8 1.8 44.7 2.0 2.7 41.4 3.3 3.1 28.4 4.1 4.2 16.2 5.6 5.7 5.0
Llama-3.1 70B 4.0 2.9 25.7 3.4 2.5 26.8 4.6 1.9 24.8 6.2 2.1 23.8 8.7 2.7 19.3 12.7 4.8 13.9 18.5 7.6 5.4
Qwen-2.5 72B 0.8 7.1 37.5 1.1 5.8 38.7 1.2 5.1 37.2 2.2 4.9 34.4 2.9 7.9 28.1 7.0 14.1 14.0 9.5 22.4 4.4
DeepSeek-V3 1.9 7.2 29.1 2.0 5.5 29.7 2.4 5.4 29.1 3.7 5.1 28.2 6.1 8.9 23.5 10.9 14.0 12.9 19.6 19.9 3.7
DeepSeek-R1 1.8 8.4 24.4 2.1 7.2 24.8 2.6 6.6 24.7 3.2 6.4 24.0 5.0 11.0 18.0 9.8 20.7 9.5 17.3 26.5 3.1
R1-Distill-Llama 8B 2.6 2.5 48.7 2.8 1.6 48.4 3.1 2.0 45.6 3.7 2.0 41.2 6.0 2.5 30.3 8.7 4.6 19.8 12.6 4.3 11.4
R1-Distill-Llama 70B 2.9 5.1 27.7 3.0 5.3 27.8 3.6 3.8 26.7 4.4 3.9 25.5 5.6 6.7 20.1 11.8 11.5 10.8 17.2 16.0 4.3
26

G Additional Experimental Results
G.1 Rejection Rate Analysis
Figures 23 and 24 illustrate the rejection rates of RAG systems in noise type and sensitivity experi-
ments, respectively. The rejection rate reflects the model’s ability to abstain from answering when
uncertain, a critical reliability aspect under noisy conditions. Our results show that RAG systems are
most sensitive to Distracting Noise, with a marked increase in rejection rates. The CHAINOF NOTE
strategy substantially suppresses rejection behavior while reducing abstentions increases the risk
of hallucination as noise levels rise. In contrast, DRAGIN and SKR maintain consistently high
rejection rates, enhancing system stability by ensuring that the model only answers when confident.
Additionally, we observe that Irrelevant Noise poses minimal threat to current RAG systems, as such
noise can be easily identified and filtered by the generators.
VanillaRAG ChainofNote DRAGIN SKR CRAG
00.1 0.3 0.5 0.7 0.910.00.10.2
Distracting RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.10.2
Low Quality RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.10.2
Inconsequential RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.20.40.60.8
Irrelevant RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.20.40.60.8
Distracting RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.20.40.60.8
Low Quality RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.20.40.60.8
Inconsequential RatioAnswer Reject
00.1 0.3 0.5 0.7 0.910.00.20.40.60.81.0
Irrelevant RatioAnswer Reject
Figure 23: Rejection rates of RAG systems under different types of retrieval noise, with Llama-3.1 8B
(top) and Qwen-2.5 7B(bottom) as generators.
Near Mid Far
0.1 0.2 0.3 0.5 0.70.020.030.040.05
Golden RatioReject Rate
0.1 0.2 0.3 0.5 0.70.030.050.07
Distracting RatioReject Rate
0.1 0.2 0.3 0.5 0.70.030.04
Low Quality RatioReject Rate
0.1 0.2 0.3 0.5 0.70.030.040.05
Inconsequential RatioReject Rate
0.1 0.2 0.3 0.5 0.70.020.040.06
Irrelevant RatioReject Rate
0.1 0.2 0.3 0.5 0.70.150.250.35
Golden RatioReject Rate
0.1 0.2 0.3 0.5 0.70.150.250.350.45
Distracting RatioReject Rate
0.1 0.2 0.3 0.5 0.70.120.140.16
Low Quality RatioReject Rate
0.1 0.2 0.3 0.5 0.70.100.120.140.16
Inconsequential RatioReject Rate
0.1 0.2 0.3 0.5 0.70.080.120.160.20
Irrelevant RatioReject Rate
Figure 24: Sensitivity of RAG system rejection rates to the position of retrieved documents, using
Llama-3.1 8B(top) and Qwen-2.5 7B(bottom) as LLM generators.
G.2 Noise Distribution Scenarios
A coarse-grained analysis of retrieval content distributions across multiple topics was conducted
(e.g.,Appendix A). We designed four representative scenarios based on these empirical statistics, each
reflecting a typical noise composition commonly encountered in real-world retrieval environments:
Scenario 1 (SCE.1) features an even distribution of Golden and Distracting documents (each 30%),
with moderate proportions of Inconsequential (10%), Low-Quality (10%), and Irrelevant noise (20%).
This configuration simulates a balanced retrieval setting, where useful information is present but
27

intermingled with challenging distractors and irrelevant content. For example, in a web search,
relevant answers are mixed with well-written yet misleading articles and unrelated advertisements.
Scenario 2 (SCE.2) maintains the same proportion of Golden documents (30%) but reduces Distract-
ing noise to 10% while increasing Low-Quality (20%) and Irrelevant noise (30%). This scenario
emulates a low-quality retrieval environment, such as forums or poorly-moderated content aggregators,
where correct information is outnumbered by low-quality, off-topic, or spam-like content.
Scenario 3 (SCE.3) distributes all non-Golden noise types more evenly (D/In/L/Ir:
20%/20%/10%/30%), with Golden documents reduced to 20%. This setting simulates a noisy
and ambiguous retrieval context, where relevant information is scarce and various noise types are dis-
tributed uniformly. For instance, the retrieved evidence often contains a broad mixture of distractors
and irrelevant snippets in open-domain QA over large uncurated corpora.
Scenario 4 (SCE.4) is characterized by a high proportion of Distracting noise (40%), low Golden
(20%), moderate Inconsequential (20%), and minimal Irrelevant and Low-Quality noise (10% each).
This configuration is designed to stress-test robustness to confusing distractors, such as in domains
where documents are topically relevant but factually incorrect or misleading— e.g., scientific misin-
formation or adversarial content in health-related searches.
G.3 Supplementary Case Study Analysis
We conduct detailed case studies to gain deeper insight into the inner workings of RAG models
under different retrieval noise conditions. Figure 25 visualizes the layer-wise attention distribu-
tion of Llama-3.1 8Bover both the query and the various retrieved documents for two questions
(“How many Beverly Hills cops movies are there? ” and “ When did power rangers
tv show come out? ”). In question (top), the answer is “ three films ”; however, under the
influence of distracting noise, the generator produces an uncertain and erroneous response (“ four or
five or six or seven ”), demonstrating the susceptibility of the model to misleading information.
In contrast, in the second question (bottom), noise does not divert the model’s attention from the
relevant golden document, and the generator successfully outputs the correct answer “ August 28,
1993 ”. This analysis reveals how the model dynamically allocates attention across input components,
reflecting its reasoning and evidence integration patterns. Figures 26–28 present qualitative results
from selected test cases, each corresponding to a specific type of retrieval noise. For each case,
we display the question, the correct answer, representative retrieved documents (including both
golden and noisy documents), and the final RAG output. These examples illustrate how distracting,
inconsequential, and low quality noise can influence the answer-generation process and highlight
typical failure modes encountered by RAG systems under noisy conditions.
Que Dis1 Dis2 Dis3 Dis4 Gol0 5 10 15 20 25 30Layer
Que Low1 Low2 Low3 Low4 Gol0 5 10 15 20 25 30
Que Inc1 Inc2 Inc3 Inc4 Gol0 5 10 15 20 25 30
Que Irr1 Irr2 Irr3 Irr4 Gol0 5 10 15 20 25 30
Que Dis Inc Low Irr Gol0 5 10 15 20 25 30
Que Dis1 Dis2 Dis3 Dis4 Gol0 5 10 15 20 25 30Layer
Que Low1 Low2 Low3 Low4 Gol0 5 10 15 20 25 30
Que Inc1 Inc2 Inc3 Inc4 Gol0 5 10 15 20 25 30
Que Irr1 Irr2 Irr3 Irr4 Gol0 5 10 15 20 25 30
Que Dis Inc Low Irr Gol0 5 10 15 20 25 30
0.000.010.020.030.040.05
Figure 25: Layer-wise attention distribution of Llama-3.1 8B.
28

The Impact of Distracting Noise
Question: Who sings with shaggy on it wasn me?
Correct Answer: English-Jamaican singer Rikrok
Doc 1: (Golden Document) From the 2000 multi-Platinum album Hot Shot by Shaggy, “It
Wasn’t Me” is the first single and includes vocals from English-Jamaican singer Rikrok.
Doc 2: (Distracting Noise) The hit single ’It Wasn’t Me’ is a collaboration between Jamaican-
American reggae artist Shaggy and British-Jamaican vocalist Sean Paul...
Doc 3: ...,Doc 4: ...,Doc 5: ...
RAG Output: Sean Paul
Figure 26: The impact of distracting noise on RAG
The Impact of Inconsequential Noise
Question: When were catholic churches allowed back in England?
Correct Answer: 1850
Doc 1: (Golden Document) The Catholic Church in England faced severe persecution after
the 1559 settlement ... Despite these challenges, Catholicism persisted, often practiced in
private. In 1850, Pope Pius IX re-established dioceses, marking a significant step in the
Church’s restoration.
Doc 2: (Inconsequential Noise) England remained a Catholic country until 1534, when it
first officially separated from Rome during the reign of King Henry VIII...
Doc 3: ...,Doc 4: ...,Doc 5: ...
RAG Output: 1534
Figure 27: The impact of inconsequential noise on RAG
The Impact of Low Quality Noise
Question: Who plays male lead in far from the madding crowd?
Correct Answer: Matthias Schoenaerts
Doc 1: (Golden Document) Carey Mulligan is the lead actress in Far from the Madding
Crowd, a 2015 British romantic drama directed by Thomas Vinterberg. The film includes
performances by Matthias Schoenaerts...
Doc 2: (Low Quality Noise) David Nicholls became attached to the film in 2008. In April
2013, it was reported that Tom Hardy had been offered the role of Gabriel Oak alongside
Carey Mulligan as Bathsheba Everdene. Their casting was official in May 2013 with the
participation of director Thomas Vinterberg....
Doc 3: ...,Doc 4: ...,Doc 5: ...
RAG Output: Tom Hardy
Figure 28: The impact of low quality noise on RAG
29

H Annotation Guidelines and Details
To ensure the quality and validity of the constructed dataset, we conducted a rigorous manual
verification process for three categories of documents: Golden Documents, Distracting Noise, and
Low Quality Noise. This process involved verifying the generated content’s factual and structural
integrity. For Golden Documents, human validators checked whether each augmented sentence
accurately conveyed the intended event structure in the original golden documents and whether the
expression is complete, with no missing or redundant arguments. In the case of Distracting Noise, the
key criterion is whether the answer entity has been effectively replaced while preserving grammatical
and structural coherence. For Low Quality Noise, which aimed to introduce semantically plausible
but factually incorrect content, human validators verified whether critical entities ( e.g., people, dates,
or locations) had been replaced and whether the sentence structure remained intact.
Each document underwent independent review by two human validators. Only when both human
validators agreed that a sample met the specified criteria was it retained; otherwise, it was discarded.
We provided clear examples of accepted and rejected samples for each noise type to facilitate
consistency and accuracy. The annotation team comprised five experienced human validators, all
undergraduate or graduate-level researchers in our laboratory with strong English proficiency. Their
domain expertise ensured high inter-annotator reliability and helped maintain the dataset’s integrity
across all verification stages. To recognize the time and effort involved in the annotation process,
contributors were fairly compensated at a rate of $10 per hour.
30
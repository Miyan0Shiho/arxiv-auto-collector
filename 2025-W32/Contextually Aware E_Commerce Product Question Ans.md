# Contextually Aware E-Commerce Product Question Answering using RAG

**Authors**: Praveen Tangarajan, Anand A. Rajasekar, Manish Rathi, Vinay Rao Dandin, Ozan Ersoy

**Published**: 2025-08-04 02:14:07

**PDF URL**: [http://arxiv.org/pdf/2508.01990v1](http://arxiv.org/pdf/2508.01990v1)

## Abstract
E-commerce product pages contain a mix of structured specifications,
unstructured reviews, and contextual elements like personalized offers or
regional variants. Although informative, this volume can lead to cognitive
overload, making it difficult for users to quickly and accurately find the
information they need. Existing Product Question Answering (PQA) systems often
fail to utilize rich user context and diverse product information effectively.
We propose a scalable, end-to-end framework for e-commerce PQA using Retrieval
Augmented Generation (RAG) that deeply integrates contextual understanding. Our
system leverages conversational history, user profiles, and product attributes
to deliver relevant and personalized answers. It adeptly handles objective,
subjective, and multi-intent queries across heterogeneous sources, while also
identifying information gaps in the catalog to support ongoing content
improvement. We also introduce novel metrics to measure the framework's
performance which are broadly applicable for RAG system evaluations.

## Full Text


<!-- PDF content starts -->

Contextually Aware E-Commerce Product Question Answering
using RAG
Praveen Tangarajan
Anand A. Rajasekar
Manish Rathi
Vinay Rao Dandin
Ozan Ersoy
Flipkart US R&D Center
Bellevue, Washington, USA
ABSTRACT
E-commerce product pages contain a mix of structured specifications,
unstructured reviews, and contextual elements like personalized of-
fers or regional variants. Although informative, this volume can lead
to cognitive overload, making it difficult for users to quickly and
accurately find the information they need. Existing Product Question
Answering (PQA) systems often fail to utilize rich user context and
diverse product information effectively. We propose a scalable, end-
to-end framework for e-commerce PQA using Retrieval Augmented
Generation (RAG) that deeply integrates contextual understanding.
Our system leverages conversational history, user profiles, and prod-
uct attributes to deliver relevant and personalized answers. It adeptly
handles objective, subjective, and multi-intent queries across het-
erogeneous sources, while also identifying information gaps in the
catalog to support ongoing content improvement. We also introduce
novel metrics to measure the framework’s performance which are
broadly applicable for RAG system evaluations.
Keywords: Product Question Answering, Retrieval Augmented Gen-
eration, Large Language Models, RAG Evaluation, Conversational
AI, E-commerce
1 INTRODUCTION
E-commerce platforms have transformed how consumers discover
and evaluate products by aggregating diverse information sources
into a single product page. These pages combine structured data
(e.g., specifications, pricing), unstructured content (e.g., user re-
views, FAQs), and contextual elements such as personalized offers,
regional variants, and localized payment options. While intended to
support informed decision-making, the sheer volume and variety of
this content often lead to cognitive overload particularly on mobile
devices resulting in decision fatigue and a fragmented shopping
experience. Conversational commerce has emerged as a promising
direction to alleviate this overload by offering interactive, dialogue-
based assistance. However, traditional product question answering
(PQA) systems, typically based on keyword retrieval or early neural
models, fall short in handling noisy data, capturing nuanced intent, or
adapting to user context. These systems often ignore critical signals
like prior interactions, evolving preferences, or the layered structure
of product information.
The advent of Large Language Models (LLMs) and Generative
AI has created new opportunities for more flexible and context-
aware information access. In this work, we introduce a scalable,end-to-end framework for e-commerce PQA that leverages Retrieval
Augmented Generation (RAG) combined with deep contextual mod-
eling. Our approach integrates user interaction history, real-time
query signals, and heterogeneous product data (structured, unstruc-
tured, and semi-structured) to generate accurate and personalized
answers. Key innovations include its ability to maintain and leverage
conversational context and mechanisms to identify information gaps
in the product catalog for continuous improvement. Furthermore,
we propose a robust evaluation protocol, including novel metrics fo-
cused on contextual accuracy and information completeness, which
are also valuable for broader assessments of the RAG system.
The remainder of the paper is organized as follows: Section 2
reviews related work in PQA and RAG. Section 3 defines the prob-
lem. Section 4 presents our proposed framework. Section 5 reports
experimental results, and Section 6 concludes with future directions.
2 RELATED WORK
Product Question Answering (PQA) is a specialized QA task focused
on generating responses to customer queries by leveraging diverse
e-commerce data—structured specifications, unstructured reviews,
and FAQs. Unlike general-purpose QA [ 19] or domain-specific QA
in biomedical [ 13] and legal [ 6] domains, PQA must synthesize both
factual and opinion-based content from heterogeneous sources.
Early approaches emphasized opinion mining [ 16,25] and struc-
tured querying using SQL or knowledge graphs [ 4,12,23]. More
recent methods leverage pre-trained language models to extract
answers from unstructured content like reviews and product descrip-
tions [ 5,7,26], and address semi-structured data using attribute
ranking and generative models [ 9,22]. The advent of Large Lan-
guage Models (LLMs) and Retrieval Augmented Generation (RAG)
[11] has further advanced PQA. Enhancements include improved
retrieval [ 8,10,15], iterative reasoning [ 21], and domain alignment
[24]. Evaluation efforts have introduced tailored benchmarks for
RAG-based systems [2, 14].
A comprehensive survey [ 3] classifies PQA methods as opinion-
based, extraction-based, retrieval-based, and generation-based. While
generation models offer the greatest flexibility, they remain prone to
hallucination and lack robust evaluation protocols. Our work builds
on these foundations by introducing a scalable, context-aware RAG
framework that integrates conversational history, product grounding,
and unified intent modeling. We further propose a novel evaluation
protocol designed specifically for e-commerce QA scenarios.arXiv:2508.01990v1  [cs.CL]  4 Aug 2025

Praveen et al.
3 PROBLEM STATEMENT
We define the e-commerce platform’s product catalog as I={𝐼𝑝|
𝑝∈P} , where each product 𝑝is associated with an information
tuple𝐼𝑝=(𝑆𝑝,𝑈𝑝,𝑀𝑝). These sources correspond to structured
attributes (𝑆𝑝), unstructured content like user reviews ( 𝑈𝑝), and
semi-structured entries ( 𝑀𝑝).
A user interaction at turn 𝑖is defined by a static or dynamic user
context𝑈𝑐𝑡𝑥(e.g., profile, preferences) and a conversational history
𝐻(𝑖−1)={(𝑞(𝑡),𝑦(𝑡))}𝑖−1
𝑡=1. Given the current query 𝑞(𝑖), which may
be objective, subjective, or multi-intent, our goal is to generate a
response𝑦(𝑖)that is: (1) Contextually Relevant and personalized
to𝑈𝑐𝑡𝑥and𝐻(𝑖−1);(2) Factually Accurate and grounded in the
retrieved product information I𝑃(𝑖); and (3) Informative about
Limitations , acknowledging information gaps when the context is
insufficient.
We model this task as learning the conditional probability 𝑃(𝑦(𝑖)|
𝑞(𝑖),𝐻(𝑖−1),𝑈𝑐𝑡𝑥,I𝑃(𝑖)).
4 PROPOSED ARCHITECTURE
Our proposed context-aware framework, illustrated in Figure 1, pro-
cesses user queries through a modular pipeline that integrates con-
versational history, product identification, intent understanding, and
targeted retrieval.
The pipeline begins with a Standalone Query (SAQ) Module,
which transforms the raw user query 𝑞(𝑖)along with its conversa-
tional history 𝐻(𝑖−1)into a self-contained, contextualized query
𝑞′(𝑖). This is passed to a Catalog Search Model that detects prod-
uct mentions and maps them to unique product IDs 𝑃(𝑖)
𝐼𝐷𝑠from the
catalogP.
Next, a unified Intent Model predicts a distribution over prede-
fined intent classes, including a Non-Decision category for out-of-
domain queries and multiple Decision intents such as specifications,
offers, or payment options. If the Non-Decision intent is not domi-
nant, the system assesses entropy over the Decision intents to guide
retrieval strategy: low entropy indicates a clear dominant intent,
triggering focused retrieval, while high entropy reflects ambiguity,
prompting retrieval for the top-N (e.g., top three) intents to maximize
coverage.
For each identified intent, the system invokes relevant backend
APIs to retrieve structured attributes ( 𝑆𝑝), unstructured reviews ( 𝑈𝑝),
semi-structured ( 𝑀𝑝), and other product-related data. To manage
the potentially large volume of retrieved content, a Retrieval and
Re-ranking module selects and prioritizes context snippets based on
semantic relevance to 𝑞′(𝑖), yielding a concise set C(𝑖)
𝑟𝑒𝑙.
Finally, the refined context C(𝑖)
𝑟𝑒𝑙and query𝑞′(𝑖)are composed
into a tailored prompt for a Large Language Model (LLM). The
LLM synthesizes this information to generate a coherent, fluent,
and contextually faithful response 𝑦(𝑖), with built-in mechanisms to
acknowledge any information gaps.
4.1 Standalone Query (SAQ)
The Standalone Query (SAQ) module is the crucial first component
in our pipeline, responsible for enabling multi-turn conversation. Its
primary function is to rewrite the current user query ( 𝑞(𝑖)) into a
Figure 1: Diagram illustrating the proposed query processing
architecture.
self-contained, contextualized query ( 𝑞′(𝑖)) by incorporating the con-
versational history ( 𝐻(𝑖−1)). This process serves two key purposes:
(1) resolving co-references (e.g., pronouns like "it") and (2) perform-
ing product disambiguation by mapping ambiguous references (e.g.,
"the second one") to their canonical product names based on the pre-
ceding context. The resulting query, 𝑞′(𝑖), encapsulates all necessary
information, allowing downstream components to operate without
needing access to the full conversation history. Table 1 illustrates
how the SAQ module handles diverse conversational scenarios.
4.2 Catalog Search
The Catalog Search module maps product mentions within the con-
textualized query ( 𝑞′(𝑖)) to unique platform Product IDs. It employs
a hierarchical strategy, prioritizing an (1) exact match of extracted
product names against the conversational history ( 𝐻(𝑖−1)). If this
fails, it proceeds to (2) fuzzy matching against both the history
and the broader platform catalog to handle variations or typos. As a
final fallback, (3) the most salient name from𝑞′(𝑖)is used to query
external and platform search APIs to retrieve a canonical product
name, which is then mapped to its ID. This cascaded approach en-
sures robust disambiguation by prioritizing in-session context before
broadening the search.
4.3 Intent Model
User query intent is determined by a unified BERT-based multi-
class classification model , which categorizes the contextualized
query (𝑞′(𝑖)) into predefined classes. The two primary categories

Contextually Aware E-Commerce Product Question Answering using RAG
Scenario Conversation Snippet SAQ Output
Follow-up on same
productOn iPhone 13 product
page
U:Battery size?
B:<answer>
U:Display size?What is the display
size of iPhone 13?
Switch to a new prod-
uctOn iPhone 13 product
page
U:Battery size?
B:<answer>
U:How about iPhone
14?What is the battery
size of iPhone 14?
Product→Accessory
searchOn iPhone 13 product
page
U:Battery size?
B:<answer>
U:Show me cases for
this phone.Show me cases for
iPhone 13.
Browsing→Specific
productOn Browse page
U:Show 2 door refrig-
erators.
B:<shows multiple
fridges including LG
242 L Frost Free 2
Star>
U:Capacity of LG
fridge?What is the capacity of
LG 242 L Frost Free 2
Star?
Table 1: Illustrative SAQ Scenarios and Outputs (User and Bot
turns are abbreviated for space)
are: non_decision for out-of-scope queries (e.g., general search like
queries) that are routed outside the PQA pipeline, and fine-grained
decision intents for queries seeking specific product information
(e.g., specifications, offers).
The model outputs a probability distribution over these intents,
and an entropy-based mechanism handles ambiguity. First, if non_decision
is predicted with high confidence, the query is routed out. Otherwise,
the entropy of the distribution over decision intents is analyzed: low
entropy selects the single, dominant intent, while high entropy se-
lects the top-N most probable intents to handle multi-faceted queries.
The selected decision intent(s) then dictate which APIs are invoked
to fetch relevant data. This unified approach simplifies the system
architecture while enabling efficient and nuanced handling of diverse
user queries.
4.4 Retrieval
Upon determining the decision intent(s), the framework initiates
a two-stage retrieval process. In Stage 1: API Orchestration , the
system calls backend APIs corresponding to the selected intent(s),
fetching broad data from canonical sources spanning structured at-
tributes, semi-structured FAQs, and unstructured content. To handle
multi-intent queries, data for all top-predicted intents is retrieved to
ensure coverage.
Since these API outputs are often verbose and contain irrele-
vant information, Stage 2: Context Reduction and Re-ranking isperformed. This stage employs a bi-encoder Semantic Textual Sim-
ilarity (STS) model [ 20] to score and rank chunks of the retrieved
data against the contextualized query 𝑞′(𝑖). The top-N chunks with
the highest semantic similarity are selected to form a final, concise
context,C(𝑖)
𝑟𝑒𝑙. This reduced context is critical for improving the
generative model’s factual accuracy and mitigating hallucinations.
4.5 Generation
Once the relevant context C(𝑖)
𝑟𝑒𝑙is prepared, a strategically ordered
prompt is constructed to guide the LLM. The prompt begins with (1)
System Persona and Core Instructions , which set the model’s be-
havior through static guidelines (e.g., strict grounding on the context,
no speculation, informal tone) and dynamic, intent-specific few-
shot examples that uses advanced prompting like Chain-of-Thought.
Next, (2) The Retrieved Reduced Context (C(𝑖)
𝑟𝑒𝑙) provides the
factual basis, anchored by the product title (from 𝑃(𝑖)
𝐼𝐷𝑠) to prevent
information leakage, followed by the top-ranked data snippets. To en-
able personalization, (3) The User Context (𝑈𝑐𝑡𝑥) is then included,
deliberately placed after the factual data to prioritize grounding. For
complex queries, dynamic (4) Intent-Specific Metadata is added
to explain platform-specific terms. Finally, (5) The Contextualized
User Query (𝑞′(𝑖)), already refined by the SAQ model, is appended.
This structured composition enables the LLM to produce responses
that are grounded, accurate, personalized, and engaging, balancing
data fidelity with a seamless user experience.
5 EXPERIMENTATION AND RESULTS
5.1 Approach
Our experimental methodology followed a rigorous, iterative process
integrating prompt engineering, automated LLM-based evaluation,
and human alignment.
Each system module was first defined with a clear task specifica-
tion, followed by constructing an initial generation prompt incorpo-
rating a system persona and curated few-shot examples using GPT-
3.5 [17]. We used proprietary e-commerce data spanning categories
like electronics and fashion, tailored to each module’s requirements.
Parallelly, we defined precise evaluation metrics and developed a
structured evaluation prompt using GPT-4 [ 18] to automatically as-
sess outputs over a representative dataset sampled from production.
A subset of outputs was manually annotated by human labelers using
the same definitions to validate and iteratively refine the evaluation
prompt until its judgments closely matched human feedback. Once
aligned, we iteratively improved the generation prompts based on au-
tomated evaluation scores. Final validation used a fresh dataset, with
both automated and human evaluations ensuring reliability. After de-
ployment, production data was continuously collected and leveraged
to fine-tune lightweight LLMs (e.g., 2B parameter models) and train
downstream classifiers critical to system operations. This end-to-end
approach blending prompt design, automated LLM feedback, human
verification, and model tuning enabled the development of a robust,
efficient, and scalable framework.
5.2 Results
5.2.1 SAQ. The Standalone Query (SAQ) module transforms
user utterances into fully contextualized queries for downstream

Praveen et al.
processing. Built on a fine-tuned in-house LLaMA 3–8B model,
SAQ resolves co-references and disambiguates product mentions,
enabling components to operate independently of full conversational
history.
To evaluate SAQ, we use three metrics: (1) overall query restruc-
turing accuracy, (2) turn-1 accuracy, and (3) turn except-1 accuracy
(multi-turn without first query). Outputs are assessed using GPT-
4–based evaluation prompts aligned with human judgments, showing
under 1% deviation.
We began with GPT-3.5 using prompt engineering, but after reach-
ing its performance ceiling, we transitioned to fine-tuning in-house
models, ultimately scaling up to LLaMA 3–8B, our current in-house
production model. This final model surpasses 95% turn except-1
accuracy with sub-500ms latency, essential for production use.
As shown in Table 2, our production model achieves 97.60%
overall accuracy, 99.08% accuracy on first-turn queries, and 95.93%
on multi-turn scenarios. These results underscore SAQ’s ability to
reliably generate context-aware queries essential to the system’s
performance.
Model Overall (%) Turn 1 (%) Turn >1 (%)
GPT-3.5 (0125) 94.31 98.68 89.41
In-house Phi-2 95.36 98.45 91.90
In-house Phi-3 95.57 98.33 92.48
In-house LLaMA 3–8B 97.60 99.08 95.93
Table 2: Performance of SAQ Model Variants
5.2.2 Catalog Search. The Catalog Search Model demonstrates
decent performance in identifying the correct full product name from
SAQ output, as well as from provided products with pre-stored PIDs
in the conversational context. For this task, we employ a GPT-3.5-
based prompt iteratively refined through continuous evaluations with
human labeling. This refined process yields a high accuracy rate of
90.03%. However, we observed that product name hallucinations
occur in 0.75% of cases, while existing product names are missed
6.39% of the time.
In instances where the extracted product name is absent from the
context, fuzzy matching is employed against the platform’s catalog to
select the best matching product and retrieve its corresponding Prod-
uct ID. This multi-step approach effectively enhances the model’s
accuracy and reliability, solidifying its integral role in the pipeline.
5.2.3 Intent Model. Our Intent Model, a BERT-based multi-class
classifier, routes queries and tailors downstream processing. It achieves
93.17% top-1 accuracy and a 92.57% weighted F1-score (Table 3),
a performance level critical for our production environment. While
these metrics reflect top-1 classification, our deployed system uses
an entropy-based mechanism to select the top-k intents for ambigu-
ous queries, which demonstrably improves practical performance
and user satisfaction.
5.2.4 Context Reduction. To provide concise and relevant inputs
to the LLM generator, we explored several context reduction strate-
gies, evaluating them using Recall@k —the proportion of queries
where the ground truth answer appeared among the top- 𝑘retrieved
segments.
Supervised Deep Passage Retrieval (DPR) models achieved strong
recall but were impractical for dynamic e-commerce catalogs dueIntent Class Precision (%) Recall (%) F1-score (%)
non_decision 81.72% 61.29% 70.05%
authenticity 80.00% 92.31% 85.71%
checkout 99.26% 99.41% 99.33%
delivery_sla 93.83% 99.66% 96.66%
offers_and_discounts 94.78% 97.99% 96.36%
payment_options 96.65% 99.43% 98.02%
product_exchange 97.01% 97.01% 97.01%
product_spec 83.92% 95.59% 89.38%
return_policy 98.93% 100.00% 99.46%
size_and_fit 92.86% 47.27% 62.65%
stock_availability 96.69% 96.69% 96.69%
variant 56.25% 54.55% 55.38%
warranty 93.58% 97.77% 95.63%
Macro Avg 89.65% 87.61% 87.87%
Weighted Avg 93.77% 94.78% 94.01%
Table 3: Intent Model Performance Across Classes
to the need for frequent retraining. Unsupervised alternatives like
fastText [ 1] offered better scalability but with significantly lower
recall.
We ultimately adopted a bi-encoder Semantic Textual Similarity
(STS) model inspired by Sentence-BERT. It selects the top- 𝑘most
relevant sentences from the retrieved context. Crucially, we domain-
adapted this model using a large in-house e-commerce FAQ dataset
and trained it with a triplet loss objective:
𝐿=∑︁
𝑖max(0,||𝑓(𝑞𝑖)−𝑓(𝑝𝑖)||2−||𝑓(𝑞𝑖)−𝑓(𝑛𝑖)||2+𝛼)
(1)
where𝑓(·)is the bi-encoder embedding function and (𝑞𝑖,𝑝𝑖,𝑛𝑖)
denote query, positive, and negative samples, with 𝛼as the mar-
gin. As shown in Table 4, this STS model achieved a Recall@k of
98.32%, outperforming fastText (95.06%) and slightly exceeding
DPR (98.10%). For example, setting 𝑘=15for the fashion category
retrieved ground truth in 98% of cases. This approach balances high
recall with scalability, adapting seamlessly across product categories
and updates without retraining.
Context Reduction Model Recall @ Top-k (%)
fastText [1] 95.06%
Supervised DPR 98.10%
Bi-encoder STS (Ours) 98.32%
Table 4: Performance comparison of context reduction algo-
rithms.
5.2.5 Generation. The generation module in our RAG framework
produces accurate, complete, and contextually grounded responses
to factual, subjective, and multi-intent queries, strictly within the
retrieved contextC(𝑖)
𝑟𝑒𝑙. Informational gaps are handled responsibly:
if context is insufficient, the system abstains from speculation and
returns “IDK” (I Don’t Know), avoiding hallucinated or mislead-
ing answers. Missing context does not imply feature absence; if
inference is needed, responses are cautious and cite evidence where
possible.
To ensure faithfulness and context-awareness, we define a robust
evaluation framework assessing answer quality (factuality, complete-
ness, precision) and alignment with context. Because generation
quality depends on context adequacy, we also evaluate the retrieved
context.

Contextually Aware E-Commerce Product Question Answering using RAG
Assumptions: For each query 𝑞′(𝑖), human or LLM-based judgment
determines ifC(𝑖)
𝑟𝑒𝑙sufficiently supports the query or its sub-intents.
An answer𝑦(𝑖)is complete if it addresses all answerable components
or returns “IDK” for unanswerable parts.
Note: “IDK” represents pattern of responses generated when the
retrieved context does not contain sufficient information to answer a
query.
I. Context Quality Metric
Context Coverage (CCov): Assesses whether the retrieved context
provides sufficient information to fully or partially answer the query.
Metric: Percentage of queries at least partially answerable from the
retrieved context.
II. Answer Evaluation Metrics
We evaluate generated answers based on four distinct scenarios:
(S1) context is sufficient and an answer is provided; (S2) context
is insufficient and the system responds with “IDK”; (S3) context
is insufficient but the system attempts a factual answer (not IDK);
and (S4) context is sufficient but the answer is “IDK”. Key metrics
include Factuality/Faithfulness , which measures whether answers
are factually correct and grounded in the retrieved context, com-
puted as Grounded Accuracy =𝑁𝑆1,𝐹𝐶
𝑁𝑆1.Answer Completeness as-
sesses if all sub-questions are answered, given by Completeness =
𝑁𝑆1,𝐶𝑜𝑚𝑝
𝑁𝑆1.Precision quantifies the fraction of valid factual answers
among all attempted (S1 and S3): Precision =𝑁𝑆1,𝐺𝑜𝑜𝑑
𝑁𝑆1+𝑁𝑆3.Recall mea-
sures the fraction of correctly answered, truly answerable queries
(S1 and S4): Recall =𝑁𝑆1,𝐺𝑜𝑜𝑑
𝑁𝑆1+𝑁𝑆4.Accuracy reflects overall correct-
ness, rewarding both valid S1 answers and correct S2 IDK re-
sponses: Accuracy =𝑁𝑆1,𝐺𝑜𝑜𝑑+𝑁𝑆2
𝑀, where𝑀is the total number
of queries."Good" requires both factual correctness AND complete-
ness. Finally, the Hallucination Rate captures the proportion of
unsupported or incorrect answers (S3 and incorrect S1), calculated
as Hallucination Rate =𝑁𝑆3+𝑁𝑆1,𝐵𝑎𝑑
𝑀.
Table 5 summarizes generation performance across these metrics and
user intent types, highlighting the system’s strengths in managing
contextual ambiguity and mitigating hallucinations.
Intent Coverage (%) Precision (%) Recall (%) Hallucination (%)
authenticity 94.0% 98.2% 98.4% 1.69%
checkout 85.9% 98.0% 99.5% 1.32%
delivery_sla 92.4% 96.6% 98.4% 3.11%
offers_and_discounts 90.85% 96.42% 98.62% 2.90%
payment_options 95.9% 96.5% 97.6% 3.36%
product_exchange 79.4% 91.3% 92.9% 6.89%
product_spec 86.4% 97.2% 98.7% 1.93%
return_policy 84.0% 96.7% 96.9% 2.72%
size_and_fit 35.2% 90.3% 94.8% 3.42%
stock_availability 51.4% 95.1% 97.1% 2.50%
variant 77.6% 94.4% 95.8% 4.38%
warranty 90.0% 97.7% 98.7% 2.00%
Table 5: Answer Generation Performance Across Intents: Cov-
erage, Precision, Recall, and Hallucination Rate
6 CONCLUSION
In this paper we presented a scalable, end-to-end framework for
e-commerce PQA that overcomes information overload by deeply in-
tegrating user and product context into a Retrieval-Augmented Gen-
eration (RAG) architecture. Our system integrates conversationalhistory, user intent, and heterogeneous data sources to deliver per-
sonalized answers to objective, subjective, and multi-intent queries.
Its core contributions include a context-aware query rewriter (SAQ
model), an entropy-based intent model for disambiguation, and a
two-stage retrieval process with a domain-adapted bi-encoder that
ensures factual grounding and minimizes hallucinations. To validate
our approach, we introduced a novel suite of metrics for evaluating
RAG systems, which confirmed the architecture’s effectiveness in
handling information gaps with principled, transparent responses.
Deployed in a production conversational assistant, our framework
serves over 5 million monthly active users, delivering an 8% increase
in user thumbs-up rates and measurable improvements in conversion
and customer satisfaction.
7 LIMITATIONS
Our current framework operates through a modular but fixed sequen-
tial pipeline. While this design ensures efficiency and reliability for
well-defined queries, it lacks flexibility in dynamically re-planning
or self-correcting when faced with ambiguity, unexpected inputs, or
partial failures. A second limitation lies in the predefined intent tax-
onomy, which constrains the system’s ability to generalize to novel
or complex user queries particularly those involving compositional
reasoning or intents not captured during training. As a result, the
system performs best on predictable, high-frequency queries and
less effectively on the long tail of diverse user needs. In future we
plan to address these limitations through the development of a more
adaptive, agentic architecture capable of dynamically selecting and
sequencing tools based on real-time query analysis. Additionally, we
aim to build automated pipelines for catalog enrichment by detecting
and resolving information gaps, thereby improving the completeness
and quality of product data over time.
REFERENCES
[1]Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017.
Enriching word vectors with subword information. Transactions of the Association
for Computational Linguistics , 5:135–146.
[2]Jia-Chen Chen, Hsin-Lin Lin, Sishuo Chen, Xiaohui Han, and Le Sun. 2023.
Benchmarking large language models in retrieval-augmented generation. arXiv
preprint arXiv:2309.01431 .
[3]Zhiling Deng, Liusha Chen, Zhirui Chen, Yulei Niu, Qingcai Chen, Guilin Qi,
and Yun Li. 2022. Product question answering: A survey. AI Open , 3:118–132.
[4]M. Frank, A. Puskás-Tompos, and A. Gábor. 2007. Answering queries using
views with web-services. Acta Cybernetica , 18(2):291–313.
[5]Kai Gao, Zhaochun Ren, Wenge Liu, Pengjie Ren, Maarten de Rijke, and
Shuicheng Yan. 2019. Product-aware answer generation in e-commerce question-
answering. In Proceedings of The Web Conference 2019, WWW 2019, San
Francisco, CA, USA, May 13-17, 2019 , pages 514–524. ACM / IW3C2.
[6]Laura Gil, Luis Plaza, and Paloma Díaz. 2021. What’s in a question? A causal-
based approach to question answering in the legal domain. Applied Sciences ,
11(11):4795.
[7] Mihir Gupta, Mike Lewis, Myle Ott, Yacine Jernite, and Veselin Stoyanov. 2019.
AmazonQA: A review-based question answering dataset. In Proceedings of
the 2019 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 1 (Long
Papers) , pages 2990–2995. Association for Computational Linguistics.
[8]Alex Ilin. 2024. Self-retrieval: Context-aware usecase specific information re-
trieval system for conversational agents. In Proceedings of the AAAI Conference
on Artificial Intelligence , volume 38, pages 19193–19201.
[9]Anusha Kulkarni, Narasimha Raju Uppalapati, Niyati Chhaya, and Prahalad Pk.
2019. Answering product-related questions using key-value pairs from product
specifications. In Proceedings of the 2019 IEEE International Conference on Big
Data (Big Data) , pages 5426–5429. IEEE.
[10] Mandar Kulkarni, Praveen Tangarajan, Kyung Kim, and Anusua Trivedi. 2024.
Reinforcement learning for optimizing rag for domain chatbots. arXiv preprint
arXiv:2401.06800 .

Praveen et al.
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. In Advances in Neural Information
Processing Systems 33 (NeurIPS 2020) , pages 9459–9474. ArXiv:2005.11401.
[12] Quan Li, Jingbo Zhou, Kuanfeng Huang, Wayne Xin Zhao, and Ji-Rong Wen.
2019. AliMe KG: A knowledge graph based question answering system for
e-commerce customer service. In Proceedings of the 42nd International ACM
SIGIR Conference on Research and Development in Information Retrieval, SIGIR
’19, Paris, France, July 21-25, 2019 , pages 1345–1348. ACM.
[13] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and
Tie-Yan Liu. 2023. Biogpt: Generative pre-trained transformer for biomedical
text generation and mining. Briefings in Bioinformatics , 24(1):bbac409.
[14] Qian Lyu and Yekerin AdT. 2024. RGB: A unified benchmark for assessing and
advancing retrieval-augmented generation. arXiv preprint arXiv:2401.11659 .
[15] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
rewriting for retrieval-augmented large language models. In Proceedings of the
17th Conference of the European Chapter of the Association for Computational
Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023 , pages 808–820.
Association for Computational Linguistics.
[16] Siavash Moghaddam and Martin Ester. 2011. Opinion digger: An unsupervised
opinion miner from unstructured product reviews. In Proceedings of the 20th ACM
International Conference on Information and Knowledge Management, CIKM
2011, Glasgow, United Kingdom, October 24-28, 2011 , pages 1821–1824. ACM.
[17] OpenAI. 2023. Chatgpt-3.5 model. Large language model.
[18] OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 .
[19] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.
SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings
of the 2016 Conference on Empirical Methods in Natural Language Process-
ing, EMNLP 2016, Austin, Texas, USA, November 1-5, 2016 , pages 2383–2392.Association for Computational Linguistics.
[20] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings
using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Con-
ference on Natural Language Processing (EMNLP-IJCNLP) , pages 3982–3992,
Hong Kong, China. Association for Computational Linguistics.
[21] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing retrieval-augmented large language models with iterative
retrieval-generation synergy. arXiv preprint arXiv:2305.15294 .
[22] Jihyung Shen, Yeonsu Kwon, Jeongyeong Moon, Jihun Hong, Taesun Choi, and
Sang-goo Kang. 2022. SemiPQA: A semi-structured product question answering
dataset. In Proceedings of the The 2022 Conference on Empirical Methods in
Natural Language Processing: Industry Track , pages 256–266, Abu Dhabi, UAE.
Association for Computational Linguistics.
[23] H. R. Tapeh and H. Abolhassani. 2008. Answering natural language queries from
multi-relation databases. International Journal of Computer Science and Network
Security , 8(6):226–233.
[24] Zhilin Yang, Jiacheng Chen, and William W. Cohen. 2023. Knowledge-
augmented language models for domain-specific question answering. arXiv
preprint arXiv:2305.09616 .
[25] Huanhuan Yu, Allison Chaney, Alok Choudhary, Han Liu, Pradeep K. Khosla,
and Subbu N. Kumar. 2013. Answering opinion questions with random walks on
review graphs. In Proceedings of the Sixth ACM International Conference on Web
Search and Data Mining, WSDM 2013, Rome, Italy, February 4-8, 2013 , pages
515–524.
[26] Yujie Zhang, Zhaochun Ren, Leyang Yu, Pengjie Ren, Zhumin Chen, and Maarten
de Rijke. 2020. ANSWERFACT: A large-scale dataset for answer generation
from product reviews and specifications. In Proceedings of the 43rd International
ACM SIGIR Conference on Research and Development in Information Retrieval,
SIGIR ’20, Virtual Event, China, July 25-30, 2020 , pages 1865–1868. ACM.
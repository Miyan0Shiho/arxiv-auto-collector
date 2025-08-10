# QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering

**Authors**: Zhuohang Jiang, Pangjing Wu, Xu Yuan, Wenqi Fan, Qing Li

**Published**: 2025-08-07 09:32:49

**PDF URL**: [http://arxiv.org/pdf/2508.05197v1](http://arxiv.org/pdf/2508.05197v1)

## Abstract
Retrieval-Augmented Generation (RAG) has been introduced to mitigate
hallucinations in Multimodal Large Language Models (MLLMs) by incorporating
external knowledge into the generation process, and it has become a widely
adopted approach for knowledge-intensive Visual Question Answering (VQA).
However, existing RAG methods typically retrieve from either text or images in
isolation, limiting their ability to address complex queries that require
multi-hop reasoning or up-to-date factual knowledge. To address this
limitation, we propose QA-Dragon, a Query-Aware Dynamic RAG System for
Knowledge-Intensive VQA. Specifically, QA-Dragon introduces a domain router to
identify the query's subject domain for domain-specific reasoning, along with a
search router that dynamically selects optimal retrieval strategies. By
orchestrating both text and image search agents in a hybrid setup, our system
supports multimodal, multi-turn, and multi-hop reasoning, enabling it to tackle
complex VQA tasks effectively. We evaluate our QA-Dragon on the Meta CRAG-MM
Challenge at KDD Cup 2025, where it significantly enhances the reasoning
performance of base models under challenging scenarios. Our framework achieves
substantial improvements in both answer accuracy and knowledge overlap scores,
outperforming baselines by 5.06% on the single-source task, 6.35% on the
multi-source task, and 5.03% on the multi-turn task.

## Full Text


<!-- PDF content starts -->

QA-Dragon: Query-Aware Dynamic RAG System for
Knowledge-Intensive Visual Question Answering
Zhuohang Jiang∗
The Hong Kong Polytechnic
University
Hong Kong SAR, China
zhuohang.jiang@connect.polyu.hkPangjing Wu∗
The Hong Kong Polytechnic
University
Hong Kong SAR, China
pang-jing.wu@connect.polyu.hkXu Yuan∗
The Hong Kong Polytechnic
University
Hong Kong SAR, China
xander.yuan@connect.polyu.hk
Wenqi Fan†
The Hong Kong Polytechnic
University
Hong Kong SAR, China
wenqi.fan@polyu.edu.hkQing Li†
The Hong Kong Polytechnic
University
Hong Kong SAR, China
csqli@comp.polyu.edu.hk
Abstract
Retrieval-Augmented Generation (RAG) has been introduced to
mitigate hallucinations in Multimodal Large Language Models
(MLLMs) by incorporating external knowledge into the genera-
tion process, and it has become a widely adopted approach for
knowledge-intensive Visual Question Answering (VQA). However,
existing RAG methods typically retrieve from either text or images
in isolation, limiting their ability to address complex queries that
require multi-hop reasoning or up-to-date factual knowledge. To
address this limitation, we propose QA-Dragon , aQuery- Aware
Dynamic RAG System for Kn owledge-I ntensive VQA. Specifically,
QA-Dragon introduces a domain router to identify the query’s sub-
ject domain for domain-specific reasoning, along with a search
router that dynamically selects optimal retrieval strategies. By or-
chestrating both text and image search agents in a hybrid setup,
our system supports multimodal, multi-turn, and multi-hop reason-
ing, enabling it to tackle complex VQA tasks effectively. We eval-
uate our QA-Dragon on the Meta CRAG-MM Challenge at KDD
Cup 2025, where it significantly enhances the reasoning perfor-
mance of base models under challenging scenarios. Our framework
achieves substantial improvements in both answer accuracy and
knowledge overlap scores, outperforming baselines by 5.06% on
the single-source task, 6.35% on the multi-source task, and 5.03%
on the multi-turn task. The source code for our system is released
in https://github.com/jzzzzh/QA-Dragon.
CCS Concepts
•Information systems →Information retrieval .
∗Authors contributed equally to this research.
†Prof. Qing Li and Prof. Wenqi Fan are the advisors of the team.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnKeywords
Retrieval-Augmented Generation, Multimodal Large Language Model,
Visual Question Answering
ACM Reference Format:
Zhuohang Jiang, Pangjing Wu, Xu Yuan, Wenqi Fan, and Qing Li. 2025.
QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive
Visual Question Answering. In .ACM, New York, NY, USA, 12 pages. https:
//doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Powered by advanced Large Language Models (LLMs) [ 3,4,12,16]
and sophisticated visual perception modules, Multimodal Large
Language Models (MLLMs) [ 8–10,17] have demonstrated strong
understanding and reasoning capabilities across a wide range of
vision-language tasks, such as Visual Question Answering (VQA).
Despite these advancements, MLLMs still face significant chal-
lenges when addressing queries that require long-tail knowledge
and multi-hop reasoning, often generating hallucinated or inac-
curate responses [ 5,13]. These issues stem from the scarcity of
relevant knowledge in MLLMs’ training corpus and the inherent dif-
ficulty of memorizing low-frequency facts [ 2]. Retrieval-Augmented
Generation (RAG) [ 7] has recently emerged as a promising solu-
tion, promoting MLLMs by incorporating external information to
complement their internal knowledge. However, multimodal RAG
(MM-RAG) still faces significant challenges, such as interpreting
complex queries, selecting appropriate retrieval tools, refining rele-
vant information, and enabling effective multi-turn interactions.
To address these challenges, we introduce a Query-Aware Dy-
namic RAG System for Knowledge-Intensive VQA ( QA-Dragon ),
specifically designed to address cross-domain, knowledge-based,
and multi-hop reasoning VQA tasks. QA-Dragon incorporates three
specialized reasoning branches and a series of modular components,
including a Pre-Answer Module, Search Router, Tool Router, Image
& Text Retrieval Agents, a Multimodal Reranker, and a Post-Answer
Module. These components combine to dynamically select the opti-
mal retrieval strategy based on domain-specific query characteris-
tics. Specifically, the Search Router interprets the intent of a given
query. It dispatches it to the appropriate retrieval pathway, while
the Tool Router further refines the execution by selecting betweenarXiv:2508.05197v1  [cs.AI]  7 Aug 2025

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
image-based and text-based retrieval agents according to the query
modality. To ensure the relevance and quality of the retrieved infor-
mation, the multimodal reranker uses a coarse-to-fine refinement
process that grounds final responses in high-quality evidence.
By supporting multimodal, multi-turn, and multi-hop reason-
ing, QA-Dragon can address the complexity of real-world VQA
scenarios. Empirical results demonstrate that our framework yields
substantial improvements in both answer accuracy and knowledge
overlap, outperforming strong baselines by 5.06% on the single-
source task, 6.35% on the multi-source task, and 5.03% on the multi-
turn task.
2 Competition Description
To highlight the challenges of real-world VQA tasks, Meta and
AIcrowd launched the Meta CRAG -MM Challenge 2025 [ 15], an
official KDD Cup competition focused on comprehensive RAG for
multimodal, multi-turn question answering (QA) over images cap-
tured in the wild. This benchmark combines egocentric photos
from Ray -Ban Meta smart glasses with factual QA pairs, a mock
image-based knowledge base, web search APIs, and rigorous truth-
fulness scoring, offering the first large-scale testbed for end-to-end
MM-RAG systems.
2.1 Dataset
TheCRAG-MM release comprises three coordinated resources: an
Image Set , aQA Set , and Retrieval Database .
•Image Set contains 5,000 RGB images, of which 3,000 are
first-person “egocentric” shots from smart-glasses, while the
rest are ordinary photos scraped from the open web. These
images span 14 topical domains (e.g., Books, Food, Shopping,
Vehicles) and intentionally include low-quality or cluttered
views that emulate real-world wearable scenarios.
•QA Set includes two complementary partitions:
(1)Single-turn : More than 3.88k independent QA pairs (1.94k
validation + 1.94k public-test).
(2)Multi-turn : 1,173 dialog sessions (586 validation + 587
public-test) comprising 2-6 interleaved questions per im-
age.
•Retrieval Database is composed of two controlled sources
designed to equalize access across teams:
(1)Animage-KG mock API that returns visually similar im-
ages and structured metadata (title, brand, price, etc.) keyed
on the query image.
(2)Atextual web-search mock API that yields URL, title, snip-
pet, timestamp, and full HTML for up to 50 pages per
query, interleaving hard negatives to mimic real-world
noise.
Together, these ingredients create a realistic yet reproducible
sandbox for studying hallucination-free answer generation in wear-
able contexts.
2.2 Search Engine
To support faithful answer generation grounded in external knowl-
edge, the CRAG-MM Challenge provides two unified, Python-basedmock retrieval APIs—one for images and one for textual web con-
tent—forming the backbone of the RAG pipeline. These APIs sim-
ulate realistic retrieval conditions while ensuring a level playing
field across participants.
2.2.1 Image Search API. This API enables the retrieval of struc-
tured entity-level metadata from visually similar images. It uses
CLIP-based image encoders1to embed both the query image and
database images, returning top- 𝑘results based on cosine similarity.
Each result includes a URL and associated structured metadata such
astitle,brand ,price , and description , mimicking a knowledge graph
interface grounded in vision. This API is the only retrieval source
available in Task 1.
2.2.2 Web Search API. A textual search interface is added in Tasks 2
and 3. This API indexes pre-fetched web pages using ChromaDB
and supports semantic search over HTML content. Given a text
query, it returns up to 50 web pages, each with a URL,title,snippet ,
and timestamp. Relevance is computed via sentence-transformer
embeddings ( e.g.,bge-large-en-v1.5 [19]) using cosine similarity.
Hard negatives are interleaved to reflect real-world retrieval noise.
2.3 Tasks
The competition defines three progressively harder tasks, including
Single-source Augmentation ,Multi-source Augmentation
andMulti-turn QA tasks.
•Task #1 - Single-source Augmentation : Given only the
image-KG API, the system must retrieve structured facts
linked to the image and produce a grounded answer. This
evaluates core visual recognition, query reformulation, and
KG grounding.
•Task #2 - Multi-source Augmentation : Web-search API
results are added, requiring the system to fuse heterogeneous
evidence, filter noise, and justify answers drawn from both
the image KG and the web.
•Task #3 - Multi-turn QA : Dialog sessions of 2-6 turns probe
contextual understanding, answer consistency, and the abil-
ity to decide when the current image is still relevant versus
when text-only reasoning suffices. Systems must respect a
strict 10 s per-turn latency and a 30 s total response budget.
3 Methodology
To address the challenge of grounded and trustworthy answer-
ing for real-world multimodal queries, we propose QA-Dragon , a
framework that integrates domain-aware reasoning, adaptive re-
trieval, and trustworthiness verification. QA-Dragon decomposes
the problem into three branches with multiple processes: 1) a Pre-
Answer Module , which performs domain classification and gener-
ates an initial reasoning trace and answer using a domain-specific
Chain-of-Thought (D-CoT) agent; 2) a Search Router , which in-
spects the reasoning trace to determine whether additional external
evidence is required, and if so, selects between retrieval-augmented
generation or answer verification; 3) a Tool Router , which decides
whether to invoke a 4) Image Search Agent or a 5) Text Search
1https://huggingface.co/openai/clip-vit-large-patch14

QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering Conference’17, July 2017, Washington, DC, USA
Pre-answerD-CoT(MLLM)SearchRouter(MLLM)Domain:“V ehicle”
Question: What is the engine size of the blue vehicle?
<Pre-answer><Reasoning>RAGToolRouter(MLLM)ImageSearchTwo-StageRerankerTextSearchSegment(DINO)Object Extraction
ImageSearchEngine<2025 Buick...>QuerySplitting<Sub_Query_1><Sub_Query_2>…TextSearchEngine
QuerySplitting<Img><Query>TextSearchEngine<Sub_Query_1><Sub_Query_2>…Search VerifyTwo-StageReranker<Object>
Direct OutputCoT Answer(MLLM)Verifier(MLLM)Post-Answer
Question: What is the engine size of the blue vehicle?Answers: The 2025 Buick Envision has an Ecotec 1.2L turbo engine.<Pre-answer>Output<Query><Img><Query><Img>Domain Router(BLIP2)
Figure 1: An overview of the QA-Dragon framework. Given a multimodal query–image pair, the system first processes the
input through a Domain Router and D-CoT to produce a draft answer and reasoning trace. A Search Router then classifies the
query into one of three branches: a) Direct Output , which emits the answer immediately; b) Search Verify , which retrieves
external evidence for verification; and c) RAG , which performs retrieval via tool routing, multimodal search, and reranking.
The resulting evidence is passed to the Post-Answer Module for final verification and answer refinement.
Agent to search for useful information, 6) a Coarse-to-Fine Mul-
timodal Reranker , which selects the most relevant information
to augment answers, and 7) a Post-Answer Module , which con-
solidates retrieved evidence with the initial hypothesis to generate
a final, verifiable response. By explicitly modeling domain context
and evidence needs, QA-Dragon improves factual consistency, sup-
ports low-latency fallback through direct output, and mitigates
hallucination via adaptive verification and reranking strategies.
3.1 Pre-Answer Module
3.1.1 Domain-Router. Real-world multimodal queries are highly
diverse, spanning topics such as animals, food, vehicles, and com-
plex data visualizations. Treating all queries uniformly risks subop-
timal reasoning due to domain mismatch and lack of specialized
context. To address this, we introduce a domain router, which iden-
tifies the semantic domain of each query–image pair and enables
domain-specific reasoning strategies tailored to the multimodal
queries.
The domain router predicts the semantic domain of the input
(𝑥,𝑞)pair, where 𝑥is an image and 𝑞is the corresponding textual
query, allowing the system to invoke specialized reasoning agents
with in-domain examples and tailored prompts. Specifically, we
leverage BLIP-2 [ 8], an efficient vision–language model, to bridge
image understanding and language reasoning through a lightweight
query transformer. It can jointly encode visual and textual signals,
enabling robust multimodal classification.
The BLIP-2 is finetuned using the domain annotations available
in the competition dataset. The predicted domain label 𝑑is thenused to dispatch the query to the corresponding D-CoT process,
which performs in-domain reasoning under customized prompting
templates.
3.1.2 D-CoT. While LLMs exhibit strong generalization, they often
lack awareness of their knowledge boundaries [ 5,13]. To mitigate
this, we introduce the D-CoT, which performs a structured pre-
answer reasoning process to identify what the model confidently
knows and where it lacks sufficient information. This early intro-
spective step enables the system to decide whether further evidence
retrieval is needed.
The D-CoT module prompts the MLLM to generate step-by-step
and domain-aware reasoning grounded in visual content and the
user query. It is accomplished by composing a domain-specific
prompt with a small number of curated in-domain few-shot exam-
ples, forming an input sequence. Specifically, we prompt MLLM to
generate a provisional answer and an explicit reasoning trace. To
ensure domain-relevant and interpretable reasoning, each prompt
includes explicit behavioral constraints. The model must 1) iden-
tify the exact object referenced by the query, 2) reason over image
and contextual clues step-by-step, and 3) explicitly signal uncer-
tainty when necessary information is missing, as illustrated in the
Appendix.
This instruction reveals the model’s internal logic and provides a
transparent basis for downstream routing decisions. The resulting
reasoning trace is then passed to the search router, determining
whether the provisional answer is sufficient or if further retrieval
or verification steps should be invoked.

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
Table 1: Execution Path Selection based on CoT Output and Router Cues.
Branch Trigger Processing Modules Image Query
Direct OutputD-CoT succeeds, and the
query is classified as
self-contained ( e.g.,
arithmetic, OCR)–
“What is written on these
umbrellas?”
Search VerifyD-CoT succeeds, but
external evidence is
recommended for
verificationText Toolchain→Fusion & RAG
Engine→Verifier
“In which year did the car on
the right begin production?”
RAG-AugmentD-CoT fails ( e.g., “I don’t
know”) or Router detects
open-world cues ( e.g.,
entities, numbers)Visual Toolchain→Text
Toolchain→Fusion & RAG
Engine→Verifier
“Who founded this cafe?”
3.2 Search Router
Not all queries require the same depth of post-processing. Some
queries can be resolved directly based on image-grounded infor-
mation, such as extracting OCR text or calculating visual formulas.
Others may require external evidence or verification to ensure fac-
tual accuracy. To make these decisions adaptively, we introduce
the Search Router, a key component that selects the most suitable
execution path based on the pre-answer and reasoning trace from
the D-CoT.
The search router is to determine whether to directly output
the preliminary answer, perform factual verification using external
sources, or invoke a retrieval-augmented generation process to
synthesize a new, grounded response. This routing mechanism
allows the system to balance efficiency and reliability, ensuring that
each query receives just the right amount of computational effort
for trustworthy results.
The router takes as input the original query and image, along
with the draft answer, reasoning trace, and predicted domain from
the D-CoT module. Instead of relying solely on the language model’s
self-assessment, the router is implemented as a lightweight classifier
trained on features derived from the reasoning trace and answer
content. These features include:
•Answerability Flags: Whether the reasoning contains un-
certain phrases like “I don’t know” or falls into known failure
patterns.
•Answer Task Heuristics: Whether the answer is a num-
ber, an OCR string, or a named object, indicating that the
information is directly observable from the image.
•Uncertainty Patterns: Whether the reasoning exhibits
speculative or fallback logic that may require further verifi-
cation.
Based on these indicators, the router assigns each query to one
of three execution paths: Direct Output , when the answer can be
confidently derived from the image alone; Search Verify , when the
draft answer lacks certainty and requires evidence checking; and
RAG , when the query depends on external knowledge not present
in the image or reasoning trace.3.3 Tool Router
To efficiently retrieve missing information, our framework must
determine not only whether to search, but also what kind of search
to perform. Queries differ in the type of evidence required: some
lack object identity ( e.g., “What model is this?” ), while others require
factual attributes not visible in the image ( e.g., “What is the price
of this car?” ). Motivated by this need for precision in retrieval, we
introduce the tool router, a lightweight decision module that selects
the appropriate retrieval modality, including image search ,text
search ,both , orneither , based on the reasoning trace obtained
from the pre-answer module.
Operating after the D-CoT module, the tool router evaluates
the reasoning trace to assess the model’s knowledge boundaries. If
the object mentioned in the query has not been identified with a
specific name, it will initiate an image search to retrieve visually
similar items and infer the object’s identity. In contrast, if the LLM
has already identified the object but lacks necessary facts that are
not inferable from the image, such as specifications, statistics, or
pricing, it triggers a text search to collect relevant information from
the web. Queries that involve analytical reasoning tasks, such as
mathematical calculations or language translation, are typically self-
contained and do not require retrieval. The key decision prompt is
illustrated in Appendix A.
3.4 Image Search Agent
The image search agent is responsible for grounding visual entities
relevant to the query. It combines Multimodal Object Extraction ,
Segmentation ,Multi-image Search Engine , and Entity Selec-
tion Module to prepare visual evidence for answer augmentation.
3.4.1 Multimodal Object Extraction. To identify the visual entities
relevant to a user query, we employ the LLM to extract a set of
grounded object candidates from the input image and question.
Given an image and its associated query, the model predicts
a list of salient and visually present objects that are likely to be
referenced or needed for downstream reasoning. Each extracted
object is described using concise, coarse-grained terms, such as “car, ”
“book,” or “brand”, and is constrained to represent tangible items
only, explicitly excluding abstract concepts and actions. To ensure
generalization across domains, the object names are normalized into

QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering Conference’17, July 2017, Washington, DC, USA
category-level descriptions and are limited to short, interpretable
phrases of up to three words. This process yields a structured object
set that supports subsequent steps in query grounding, answer
generation, and retrieval conditioning.
From the extracted candidates, a secondary selection step iden-
tifies the most relevant object to the query. This object is chosen
based on its visual presence, contextual alignment with the query,
and positional cues in the image. When multiple similar instances
exist, the model appends a distinguishing attribute ( e.g., “red car,”
“white car”) to ensure precise reference. The result is a single object
label that serves as the anchor for following image segmentation.
3.4.2 Segmentation. In the VQA process, users’ questions often
focus on a specific object in an image rather than the entire image.
To achieve more granular retrieval, we introduce a refined search
mechanism based on image segmentation. Specifically, we extract
key object words from the image during the multimodal object
extraction stage, use grounding DINO [ 11] to locate and crop these
targets, and generate corresponding local image regions for sub-
sequent retrieval. For images containing multiple objects with the
same name, our system also supports the simultaneous extraction
and processing of multiple related regions to more comprehensively
cover the user’s query intent.
3.4.3 Multi-Image Search Engine. To meet the needs of single-
image or multi-image retrieval, we designed and implemented a
customised image retrieval engine that can automatically adapt
to multiple image regions generated during the image segmen-
tation stage. The engine performs searches based on the "crag-
mm-2025/image-search-index-validation" [15] index and uses the
"clip-vit-large-patch14-336" [14] model to calculate the semantic em-
bedding similarity between the query and candidate image results.
Finally, we fuse multiple retrieval results and reorder them based
on similarity scores to obtain image content more closely related to
the query semantics, thereby providing high-quality visual support
for multimodal QA.
3.4.4 Entity Selection. When external image retrieval is performed,
it is essential to ensure that the retrieved candidates visually cor-
respond to the actual object depicted in the image. Without this
verification, downstream reasoning may rely on mismatched enti-
ties, leading to hallucinated or irrelevant answers. To address this
challenge, we introduce an entity selection step, which verifies
whether any retrieved entity is visually present and consistent with
the object shown in the query image.
3.5 Text Search Agent
The text search agent is responsible for retrieving external textual
evidence relevant to a given query–image pair. It is composed of
three sequential components: the Query Rephrase , the Fusion
Search and the Text Search Engine to handle multi-hop reasoning,
entity disambiguation, and weakly grounded queries.
3.5.1 Query Rephrasing Module. Since search engines require ex-
act matches, it is difficult to handle multi-hop or ambiguous queries.
Therefore, we need to break down multi-hop queries into multiple
clear sub-queries and replace any references or ambiguous content
to improve search results.
1
2
3
𝐾1
Question : where was the 
founder of this car brand born?
Image API
Web APIImage 
SearchQ-Former 
(Multimodal)Q-Former 
(Text)
Text 
SearchMaxSim
[CLS]LLM -based Reranker𝑠1
Top 𝐾1
𝑠Top 𝐾2
𝑠2
Coarse -grained Reranking Fine-grained Reranking
<INST> <Question> <Evidence>
1
2
3
𝑁
1
2
𝐾2
Context 
AssemblyFigure 2: The pipeline of multimodal reranker.
Specifically, we first structure and break down the complex origi-
nal queries. Using the original query, the reasoning chain of thought,
and image information extracted by the visual tool chain, we con-
struct multiple more specific sub-questions, each of which typically
corresponds to an inference step in the original question or infor-
mation related to a specific image entity. To further enhance the
clarity and retrieval adaptability of these sub-questions, we intro-
duce a sub-question enhancement module to semantically clarify
and complete each generated sub-queries, particularly addressing
parts with unclear referents or lacking context.
3.5.2 Fusion Search. The fusion search serves as a bridge between
the image and text retrieval pipelines by leveraging grounded object
information extracted during the image search stage to improve
the quality and precision of text-based evidence retrieval. This is
particularly useful when the original user query lacks a specific
object identity or when disambiguation is required to generate a
meaningful web search query.
After identifying the most relevant object in the image and ver-
ifying it through the entity selection, the fusion search process
constructs object-aware textual queries by combining the user’s
original question with the name of the verified object. This enables
the system to issue more focused and semantically grounded web
search queries, such as transforming a vague question like “What’s
the price of this?” into a more informative query like “Price of red
sports car BMW M4. ”
3.5.3 Text Search Engine. To accommodate the retrieval needs
of single or multiple sub-queries, we have built a custom text re-
trieval engine that can adapt to multiple sub-queries automatically.
The engine performs query operations in the "crag-mm-2025/web-
search-index-validation" [15] index and uses the "bge-large-en-v1.5"
model[ 19] to calculate the similarity between the search results
and the query’s semantic embeddings. Finally, we fuse multiple
retrieval results and reorder them based on similarity to obtain
more relevant document content, thereby supporting subsequent
responses.
3.6 Coarse-to-fine Multimodal Reranker
To efficiently use evidence retrieved from image and text search
agents and filter query-relevant information, we employ a two-
stage coarse-to-fine multimodal reranker that performs Evidence
Chunking ,Multimodal Reranking , and Context Assembly .

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
3.6.1 Evidence Chunking. LetDtextandDimgdenote the textual
and image-based retrieval sets, respectively. The initial evidence set
is defined asDall=Dtext∪D img, where𝑑∈D allis a paragraph
from retrieved passages or an attribute block from image metadata.
For each paragraph 𝑑, we first segment it into manageable pieces
using a hierarchical chunking function 𝑓chunk(𝑑), which takes into
account the document structure (e.g., titles, paragraphs) as well as
fixed-length spans. Given an attribute block, we construct a sen-
tence for each attribute using the template: “The <key> of <entity>
is <value>”, where <entity> refers to the entity name from the
metadata. By concatenating these sentences, we form a paragraph
corresponding to the attribute block. This generated paragraph is
then processed using 𝑓chunk(𝑑). Finally, the chunked evidence set
can be obtained by
C=Ø
𝑑∈D all𝑓chunk(𝑑), (1)
where𝑐𝑖∈Cis a candidate evidence.
3.6.2 Multimodal Reranking. We perform cascaded reranking for
Cin two stages, as depicted in Figure 3.
Coarse-grained Reranking: To ensure the relevance with both
the question and image, a vision-language model 𝑓qformer based on
Q-Former [8] is first used to filter evidence at a coarse granularity.
Given a question-image pair (𝑞,𝑥),Q-Former encodes it into a set
of query tokens{𝑧1𝑞,...,𝑧𝑁𝑞
𝑞}. On the candidate side, 𝑐𝑖is encoded
into a single [CLS] token𝑧𝑡byQ-Former ’s text encoder. Then, the
relevance score for each evidence is computed via the following
similarity function:
𝑠(𝑖)
coarse =max
1≤𝑗≤𝑁𝑞(sim(𝑧𝑗
𝑞,𝑧𝑡)), (2)
which measures the maximum pairwise similarity between the
query token embedding of (𝑞,𝑥)and the [CLS] token embedding of
𝑐𝑖. Subsequently, the top 𝐾1candidates with the highest scores are
selected. However, any candidate with scores below a predefined
threshold𝜏coarse is discarded.
Fine-grained Reranking: To more accurately evaluate similar-
ity, we employ an LLM-based reranker ( Qwen3-Reranker [20])
for point-wise reranking within a single context. Specifically, each
evidence from the previous step is combined with the question and
a well-designed instruction to form an input context, which is then
fed into Qwen3-Reranker 𝑓qwen to compute a relevance score, i.e.,
𝑠(𝑖)
fine=𝑓qwen(𝑞,𝑐𝑖|instruction). (3)
In this step, we select the top- 𝐾2chunks with the highest cumulative
score, i.e.,𝑠(𝑖)=𝑠(𝑖)
fine×𝑠(𝑖)
coarse . Similar to the first stage, only chunks
with a score above 𝜏fine×𝜏coarse are retained.
3.6.3 Context Assembly. For each query, we reorganize the selected
chunks into a final evidence string as:
E=Join
{Sort(𝑐𝑖)}𝐾2
𝑖=1
, (4)
which serves as the input to the post answer generator and verifier.
This coarse-to-fine multimodal reranking mechanism filters noisy
or redundant retrievals out while retaining semantically grounded,
query-relevant evidence from both retrieval sources.3.7 Post Answer Generator & Verifier
Given the question-image pair ( 𝑞,𝑥), and the reranked evidence
contextE(as constructed in Section 3.6), we employ a CoT-based
generator followed by an answer verifier to produce the final an-
swer.
3.7.1 CoT-based Answer Generation. For complex multimodal rea-
soning tasks, the capability of the MLLM remains limited, as demon-
strated in Table 2. To enhance its reasoning performance, we employ
CoT prompting techniques [ 18] and ICL examples [ 6], guiding the
model to articulate intermediate steps during problem-solving. The
MLLM is prompted to produce a detailed reasoning process and a
concise final answer for each query. The reasoning involves iden-
tifying supporting evidence from Erelevant to the query, while
the final answer addresses the query directly. These intermediate
reasoning steps also support answer verification in the subsequent
stage. Refer to the Appendix for detailed templates.
3.7.2 Answer Verifier. To reduce hallucinations and ensure factual
precision, we implement a dual-verification mechanism to assess
answer reliability. First, we explore a lightweight white-box verifier
that leverages the token probabilities of the generated response to
quantify uncertainty [ 1]. Let𝐻={ℎ1,ℎ2,...,ℎ𝑇}denote the hid-
den states at the final decoder layer for each output token from the
CoT-based Answer Generator. We compute two statistical metrics:
Minimum Token Probability andNormalized Token Probability ,i.e.,
𝑠min=min𝑖𝑡𝑖and𝑠mean =1
𝑇Í𝑇
𝑖=1𝑡𝑖. Here,𝑡𝑖denotes the token
probability, while 𝑇is the number of tokens. Then, both 𝑠minand
𝑠mean are passed to a linear thresholding function 𝑔𝑑(𝑣)that de-
cides whether to accept or reject the generated answer. Only when
𝑔𝑑(𝑣)≥𝜏is the previous answer accepted as final. Otherwise, the
system abstains from answering or falls back to a default state-
ment (e.g., “I don’t know”). Both 𝑠minand𝑠mean are then fed into a
linear thresholding function 𝑔𝑤ℎ𝑖𝑡𝑒 , which determines whether to
accept or reject the generated answer. The answer is accepted only
if𝑔𝑤ℎ𝑖𝑡𝑒(𝑠min,𝑠mean)≥𝜏𝑤ℎ𝑖𝑡𝑒 ; otherwise, the system defaults to a
fallback response ( e.g., “I don’t know”).
Moreover, we design an MLLM-based verifier to assess whether
the reasoning produced by the CoT-based Answer Generator is
logically sound and supported by the retrieved evidence. Specifi-
cally, the MLLM is provided with the question-image pair (𝑞,𝑥),
the evidence context E, and the reasoning response, and is tasked
with classifying the reasoning as either “Correct” or “Incorrect.”
The detailed prompt used for this step is provided in the Appendix.
Finally, we integrate the white-box and MLLM-based verifiers
to yield high-certainty answers. This dual-verification approach
ensures that final answers are grounded in the retrieved context and
internally coherent and filtered by confidence, thereby reducing
hallucinations in open-ended generation.
4 Experiments
In this section, we report our main results and conduct an ablation
study to demonstrate the effectiveness of our key components.
4.1 Experiments Setup
According to the rules set by the organizers, we conduct our exper-
iment on a single Nvidia L40s with 48GB of memory. The network

QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering Conference’17, July 2017, Washington, DC, USA
Table 2: Overall Performance of Our Framework across Single-Source and Multi-Source Tasks.
MethodComponents Single-Source Multi-Source
DR TR FS SR FT Accuracy ( ↑) Overlap (↑) Elapse (↓) Accuracy (↑) Overlap (↑) Elapse (↓)
LLM Only – – – – – 15.79 18.98 0.71 15.79 18.98 0.71
CoT – – – – – 16.25 20.07 1.78 16.25 20.07 1.78
Direct RAG – – – – – 14.40 22.67 1.62 16.87 27.15 1.95
QA-Dragon†✓ ✓ ✗ ✗ ✗ 16.77 31.18 3.23 18.99 34.24 3.27
QA-Dragon‡✓ ✓ ✓ ✗ ✗ 17.44 33.57 3.32 22.08 42.62 3.83
QA-Dragon ✓ ✓ ✓ ✓ ✗ 21.31 41.09 4.15 23.22 41.77 4.79
QA-Dragon★✓ ✓ ✓ ✓ ✓ 20.79 41.42 4.85 22.39 41.65 5.97
*DR: Domain Router; TR: Tool Router; FS:Fusion Search; SR:Search Router; FT:Finetuning.
Table 3: Overall Performance of Our Framework on Multi-
Turn Task.
Method Accuracy (↑) Overlap (↑) Elapse (↓)
LLM Only 19.75 19.72 0.71
CoT 19.47 20.27 1.78
Direct RAG 20.95 33.74 1.95
QA-Dragon 24.78 48.26 7.00
QA-Dragon★23.94 47.49 7.80
connection is turned off during the inference process, meaning
we can only leverage the knowledge from the given database. It
also requires that each query be addressed within 10 seconds and
labeled as failed after that.
In the model setup, we use Llama3.2-11B-Vision2for all our
MLLMs. The search API supports both text search and image search,
which employs “BAAI/bge-large-en-v1.5” and“openai/clip-vit-large-
patch14-336” for text and image embedding generation, respectively.
In our experiments, we compare our method with the following
baselines. (1) LLM Only prompts the model to answer without
retrieval; (2) CoT adds step-by-step reasoning. (3) Direct RAG
retrieves references using the original query and image, and gener-
ates answers based solely on the retrieved content. Furthermore,
during the development process, we conducted an in-depth analysis
of the model outputs and rapidly iterated through four versions of
the model:
•QA-Dragon†:Utilize the domain router and tool router to
process the image search and text search.
•QA-Dragon‡:Fuse the image search result into the text
search during the search process.
•QA-Dragon: Propose a search router to categorize VQA
queries into three search strategies.
•QA-Dragon★:Integrated a finetuning module to enhance
the reasoning capability.
4.1.1 Metrics. In line with the MM-CRAG Benchmark, we con-
duct a model-based automatic evaluation for our experiment. The
automatic evaluation employs rule-based Overlap matching and
2https://huggingface.co/meta-llama/Llama-3.2-11B-VisionTable 4: Ablation Study for Key Components in Our QA-
Dragon Framework.
MethodSingle-Source Multi-Source
Acc. Overlap Elapse Acc. Overlap Elapse
QA-Dragon 21.31 41.09 4.15 23.22 41.77 4.79
w/oDomain Router 19.04 41.07 4.19 21.25 41.85 5.07
w/oTool Router 18.32 41.49 4.02 19.97 42.32 4.25
w/oQuery Splitting – – – 18.32 41.70 4.02
w/oReranking 20.90 41.30 4.15 22.14 41.19 5.07
*Query Splitting is not applied under the single-source scenario.
prompt GPT-4o-mini as an assessment to check the Accuracy of
the answer. Moreover, we report the average second Elapse of each
query since efficiency is a critical aspect required by the organizers.
The final score of each method is computed as the average score
across all examples in the evaluation set.
4.2 Overall Performance
Table 2 and Table 3 report the performance of our method and base-
lines. Comparing our solutions to the LLM and RAG baselines, we
observe significant advantages in performance across all three tasks.
Our method showcases notable improvements in accuracy and in-
formation retention. Specifically: QA-Dragon and QA-Dragon★
Achieves the Best Overall Performance . QA-Dragon outper-
forms all baselines and ablations in both single-source (21.31%),
multi-source (23.22%) and multi-turn (24.78%) settings. The QA-
Dragon★achieves the highest overlap scores (41.42% and 41.65%),
indicating better alignment with ground-truth supporting evidence.
Baselines Fall Short . The LLM Only, CoT, and Direct RAG base-
lines significantly underperform in accuracy and overlap. Trade-off
between Performance and Efficiency . The most powerful vari-
ant, QA-Dragon★, achieves the best performance but incurs the
highest latency (5.97s).
4.3 Ablation Study
To assess the contribution of each core component in our frame-
work, we conduct a series of ablation experiments. Table 4 reports
the results in terms of answer accuracy, entity overlap with ground

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
Direct Output Search VerifyRAG
Routing Branch05101520253035Accuracy (%)33.6%
17.5%
13.8%Search Branch
Static Slow Fast
Real-time
Dynamism Type05101520Accuracy (%)22.6%
18.1%
6.0%
0.0%Query Dynamism
Simple-RecoReasoningComparisonMulti-hop
Simple-Know Aggregation
Query Category010203040Accuracy (%)43.2%
27.3%
21.3% 21.2%20.3%
5.8%Query Category
Math T ext Vehicle other Food Book General Local Shopping Animal Plants
Query Domain051015202530Accuracy (%)33.3%
29.8%
26.1% 26.0% 25.9%25.0%
23.2%
20.6%
18.7% 18.5%
7.5%Performance by Query Domain
Figure 3: Accuracy over different query taxonomies on the
single-source task.
truth, and response latency, to evaluate both correctness, efficiency,
and alignment with human-annotated outputs.
Domain Router. Removing the domain router leads to a notice-
able drop in accuracy for both single-source (from 21.31% to 19.04%)
and multi-source settings (from 23.22% to 21.25%). It demonstrates
the importance of domain-aware query routing in selecting appro-
priate retrieval strategies, especially when grounding queries in
specific content domains.
Tool Router. Disabling the tool router results in further perfor-
mance degradation, particularly in the single-source setting (from
21.31% to 18.32%). Interestingly, overlap remains high (41.49% and
42.32%), suggesting that while relevant content is still retrieved,
the absence of precise tool selection limits the model’s ability to
convert evidence into correct answers.
Query Splitting. Eliminating query splitting, which decom-
poses complex questions into interpretable sub-goals, significantly
degrades the performance. This result validates the effectiveness of
breaking down complex queries into multiple logical sub-queries
to enable more rigorous textual retrieval.
Two-stage Reranking. We further test the effect of removing
two-stage reranking in the retrieval pipeline. Accuracy slightly
decreases (from 21.31% to 20.90% in single-source and from 23.22%
to 22.14% in multi-source), and latency increases by nearly 0.3s
in the multi-source case. This indicates that reranking retrieved
candidates meaningfully improves answer selection without com-
promising efficiency.
These results highlight that each component in our system con-
tributes meaningfully to either performance or latency. The frame-
work achieves the best overall balance by dynamically coordinating
reasoning and retrieval.
5 Perspectives
To better understand the behavior and limitations of our QA-Dragon
framework, we conduct a detailed evaluation across four query
Direct OutputRAG
Search Verify
Routing Branch051015202530Accuracy (%)32.7%
21.5%
17.4%Search Branch
Static Slow Fast
Real-time
Dynamism Type0510152025Accuracy (%)24.1%
21.6%
10.4%
0.0%Query Dynamism
Simple-RecoReasoningComparisonMulti-hop
Simple-Know Aggregation
Query Category0510152025303540Accuracy (%)40.9%
26.4%25.2%23.7%23.0%
6.7%Query Category
T ext Math Book Other Food Vehicle General Shopping Animal Local Plants
Query Domain05101520253035Accuracy (%)34.4%33.3%31.8%
30.1%
26.3%
24.6% 24.4% 24.1%
22.2%21.4%
9.8%Performance by Query DomainFigure 4: Accuracy over different query taxonomies on the
multi-source task.
taxonomies: search branch, query dynamism, query category, and
query domain, as shown in Figure 3 and 4, respectively.
Across both tasks, we observe several consistent patterns. Queries
routed to the Direct Output branch achieve the highest accuracy
(33.6% for single-source and 32.7% for multi-source), significantly
outperforming Search Verify andRAG Augment , highlighting the
effectiveness of the Search Router in identifying queries that can
be resolved through image-grounded or self-contained reasoning
directly. Accuracy decreases steadily as query dynamism increases,
with real-time queries yielding no correct answers. This illustrates
the challenge of temporally sensitive reasoning and emphasizes
the need for improved temporal grounding in future systems. In
terms of query category, Simple Recognition achieves the highest
performance (43.2% and 40.9%), while Aggregation is the most diffi-
cult (5.8% and 6.7%), likely due to its multi-hop and cross-source
nature. Among query domains, Math and Text lead in accuracy,
benefiting from structured formats and OCR-friendly content. Con-
versely, the Plant domain shows the lowest performance (7.5% and
9.8%), reflecting the difficulty of fine-grained visual discrimination
in biologically similar classes.
6 Conclusion
In this work, we proposed the QA-Dragon , a query-aware dynamic
RAG system tailored for complex and knowledge-intensive VQA
tasks. Unlike the traditional RAG system, which retrieves the text
and image separately, the QA-Dragon integrates a domain router,
a search router, and orchestrates hybrid retrieval strategies via
dedicated text and image search agents. Our system supports multi-
modal, multi-turn, and multi-hop reasoning through orchestrating
both text and image search agents in a hybrid setup. Extensive
evaluations on the Meta CRAG-MM Challenge (KDD Cup 2025)
demonstrate that our system significantly improves answer accu-
racy, surpassing other strong baselines.

QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering Conference’17, July 2017, Washington, DC, USA
References
[1]Dylan Bouchard and Mohit Singh Chauhan. 2025. Uncertainty Quantification for
Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble
Scorers. arXiv preprint arXiv:2504.19254 (2025).
[2]Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter,
and Ming-Wei Chang. 2023. Can Pre-trained Vision and Language Models Answer
Visual Information-Seeking Questions?. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing . 14948–14968.
[3]Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang,
Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al .2023.
Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See
https://vicuna. lmsys. org (accessed 14 April 2023) 2, 3 (2023), 6.
[4]Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Se-
bastian Gehrmann, et al .2023. Palm: Scaling language modeling with pathways.
Journal of Machine Learning Research 24, 240 (2023), 1–113.
[5]Federico Cocchi, Nicholas Moratelli, Marcella Cornia, Lorenzo Baraldi, and Rita
Cucchiara. 2025. Augmenting multimodal llms with self-reflective tokens for
knowledge-based visual question answering. In Proceedings of the Computer
Vision and Pattern Recognition Conference . 9199–9209.
[6]Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia,
Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A
Survey on In-context Learning. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing . 1107–1128.
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining . 6491–6501.
[8]Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping
language-image pre-training with frozen image encoders and large language
models. In International conference on machine learning . PMLR, 19730–19742.
[9]Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song
Han. 2024. Vila: On pre-training for visual language models. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition . 26689–26699.
[10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual in-
struction tuning. In Advances in neural information processing systems , Vol. 36.
34892–34916.[11] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan
Li, Jianwei Yang, Hang Su, Jun Zhu, et al .2023. Grounding dino: Marrying
dino with grounded pre-training for open-set object detection. arXiv preprint
arXiv:2303.05499 (2023).
[12] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training language models to follow instructions with human feedback. Advances
in neural information processing systems 35 (2022), 27730–27744.
[13] Jielin Qiu, Andrea Madotto, Zhaojiang Lin, Paul A Crook, Yifan Ethan Xu,
Xin Luna Dong, Christos Faloutsos, Lei Li, Babak Damavandi, and Seungwhan
Moon. 2024. Snapntell: Enhancing entity-centric visual question answering with
retrieval augmented multimodal llm. arXiv preprint arXiv:2403.04735 (2024).
[14] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning . PmLR, 8748–8763.
[15] CRAG-MM Team. 2025. CRAG-MM: A Comprehensive RAG Benchmark for Multi-
modal, Multi-turn Question Answering. https://www.aicrowd.com/challenges/
meta-crag-mm-challenge-2025
[16] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models. arXiv
preprint arXiv:2302.13971 (2023).
[17] P Wang, S Bai, S Tan, S Wang, Z Fan, J Bai, K Chen, X Liu, J Wang, W Ge, et al .
2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at
any resolution, 2024. URL https://arxiv. org/abs/2409.12191 (2024).
[18] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models. Advances in neural information processing systems 35
(2022), 24824–24837.
[19] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023.
C-Pack: Packaged Resources To Advance General Chinese Embedding.
arXiv:2309.07597 [cs.CL]
[20] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al .2025. Qwen3 Embedding:
Advancing Text Embedding and Reranking Through Foundation Models. arXiv
preprint arXiv:2506.05176 (2025).

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
A Prompts
Evaluator Prompt
System Prompt:
You are a visual assistant tasked with addressing the user’s query for the image based on your inherent knowledge.
General Reasoning Guidelines:
(1)Generate step-by-step reasoning to address the query using evidence from the image and your knowledge. Limit your
reasoning to no more than 5 concise steps, with each step written as a single sentence. Stop reasoning once you have enough
information to answer the query or when you determine that necessary information is lacking.
(2)In your reasoning, identify the exact object that the query is about by its exact name ( e.g., car model, food name, brand name,
species name, etc.). If no clear object matches the query, you may refer to textual clues visible in the image if available.
(3)If the query involves multiple objects or relationships, dedicate one reasoning step to each object or relationship, and then
summarize the result in a final step. For example:
(a) The exact name of the object in the image that the query is about is <specific_object_name>.
(b) Next, the exact name of the object related to the first one is <specific_object_name>."
(c) ...
(4)If you cannot determine the necessary information from the image or the query, explicitly state: "I cannot determine the
<what> that the query is about."
(5) Do not suggest that the user to refer to external sources.
(6)Always begin your reasoning with: "1. The exact name of the object that the query "<query>" is about is <spe-
cific_object_name>." and make your final reasoning concise.
<Doamin-aware ICL cases>
Output Format:
(1) The exact name of the object in the image that the query is about is <specific_object_name>.
(2) Then, I ...
(3) ...
{"reasoning": "<summary_reasoning_string>"}
Image Object List Prompt
System Prompt:
You are an expert AI system for object detection and identification. Your task is to recognize and list high-level object categories
shown in an image that are relevant to a given question. Return only structured results in JSON format.
User Prompt:
Identify and list up to {self.object_num} major distinct objects in the image that are visually present and relevant to the question:
"query".
Only include tangible, visible items (e.g., "car", "brand", "clothing", "book", "device", "food", "building").
Do not include abstract concepts (e.g., "emotion", "relationship") or actions (e.g., "running", "shopping").
Use general categories, not specific names: "BMW" to "car", "ZARA" to "clothing brand", "iPhone 13" to "smartphone", "Coca-Cola" to
"drink"
Each object name should be short with no more than 3 words.
If unsure of the exact identity, use the closest general category (e.g., "electronic device", "building", "plant").
Return the result in the following JSON format strictly:
{"object_list": ["<object_name_string>", "<object_name_string>", ...]}
Image Object Selection Prompt
System Prompt: You are an AI assistant to select one object from a list of objects, which is most relevant to the object queried by
the question. Only return structured results in JSON format.
User Prompt: Given the image and list of objects detected in the image: {object_list} and the question: "{query}" select the
one object in the object list that the question is about.
If the query includes position-related words, give priority to objects at or near the position.
Give a short sentence about the reason to choose the object and return the final selected object in this format:
{"object": "<object_name_string>"}

QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering Conference’17, July 2017, Washington, DC, USA
Evaluator Prompt
System Prompt: You are an action-selector. Given three inputs: (1) the user’s query, (2) any prior reasoning text, and (3) the
image, decide in a single turn which retrieval tool or tools must run before the answer is generated. You may choose image_search,
text_search, both, or neither.
Tools: Tool #1 image_search
[Description] Retrieve visually similar images via embeddings to identify an object whose specific name is still unknown.
[Use when] The object in the picture is not known or known only by a generic label (e.g., “car”, “jacket”, “statue”) instead of a specific
name/model/species in previous reasoning.
[Input] The input image.
[Output] Text snapshots (top-k) from Wikipedia or Amazon that show visually similar objects.
Tool #2 text_search
[Description] Issue a refined natural-language web query to fetch textual facts about an object whose specific name is already
known.
[Use when] The query requires additional information not available in the image.
[Input] A text query constructed from the user’s question + the known identity.
[Output] Text snippets (top-k) from relevant websites.
Decision logic
(1) Do you or the reasoning text already know the object’s specific identity (proper noun or model name)?
- Yes: set ‘need_image_search‘ to false.
- No: set ‘need_image_search‘ to true.
(2) Does the query need additional information that is not visible in the image (specifications, history, statistics, price, etc.)?
- Yes: set ‘need_text_search‘ to true.
- No: set ‘need_text_search‘ to false.
(3)If it is about addressing some scientific calculation queries like math, physics, etc., or language translation, set both flags to
false.
(4)If the object is a "book", a "logo-bearing packaged goods", or "plant", set ‘need_image_search‘ to false. Both flags may be true;
when so, run image_search first, then text_search.
Produce exactly one sentence to conduct the decision logic. This is the only non-JSON text allowed.
Immediately after the sentence, output a single valid JSON object: Decision logic: <concise explanation>
Tool calling decision: "need_image_search": <true/false>, "need_text_search": <true/false>
Do not output anything else.
Here are some examples:
<Tool Using ICL cases>
Post Answer Generation Prompt
System Prompt: You are a helpful assistant who truthfully answers user questions about the provided image. If you are not sure
about the answer, please say ’I don’t know.’.
User Prompt: Answer the given question based on the provided image and your own knowledge. Please think step by step and give
a response containing the following parts:
•reason: the information from the image or your own knowledge that leads to the answer, which should be clear and concise
within 2-3 sentences. If you are not sure about the answer, please say ’I don’t know.’.
•answer: your final answer in a concise format within one sentence. The answer should include critical information from the
reason to support your answer. When referring to a specific object in the image, please use the name of the object, rather than
’this’, ’that’, or ’it’. If the reason is ’I don’t know, ’ please also say ’I don’t know’ in the answer.
-Examples-
<ICL examples>
-Real Data-
Question: <question>
Your Output:

Conference’17, July 2017, Washington, DC, USA Zhuohang Jiang et al.
Answer Verifier Prompt
System Prompt: You are a helpful assistant who evaluates whether the agent’s answer to the user’s image query is reasonable
based on the evidence.
User Prompt: Given an image query, the retrieved evidence, and the agent’s candidate answer, assess the correctness of the answer
by following these guidelines:
(1) Unsupported Answer. If the answer is not supported by the image and evidence, please respond the follows:
**Reason:** Briefly (1–2 sentences) explain why the answer lacks sufficient support.
**Response:** Incorrect Answer
(2) Contradicted Evidence. If there is conflicting information in the evidence, please respond the follows:
**Reason:** Briefly (1–2 sentences) state the specific contradictory evidence.
**Response:** Incorrect Answer
(3) Unclear or Incomplete Answer. If the answer is vague or fails to fully address the question, please respond the follows:
**Reason:** Briefly (1–2 sentences) explain why the answer is unclear or incomplete.
**Response:** Incorrect Answer
(4) Correct Answer. If the answer is fully supported by the image and evidence, please respond the follows:
**Reason:** Briefly (1–2 sentences) state why the answer is correct.
**Response:** Correct Answer
-Real Input-
Query: <quesion>
Evidence: <evidence>
Answer: <answer>
Your Output:
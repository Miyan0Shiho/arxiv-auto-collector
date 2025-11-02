# CRAG-MM: Multi-modal Multi-turn Comprehensive RAG Benchmark

**Authors**: Jiaqi Wang, Xiao Yang, Kai Sun, Parth Suresh, Sanat Sharma, Adam Czyzewski, Derek Andersen, Surya Appini, Arkav Banerjee, Sajal Choudhary, Shervin Ghasemlou, Ziqiang Guan, Akil Iyer, Haidar Khan, Lingkun Kong, Roy Luo, Tiffany Ma, Zhen Qiao, David Tran, Wenfang Xu, Skyler Yeatman, Chen Zhou, Gunveer Gujral, Yinglong Xia, Shane Moon, Nicolas Scheffer, Nirav Shah, Eun Chang, Yue Liu, Florian Metze, Tammy Stark, Zhaleh Feizollahi, Andrea Jessee, Mangesh Pujari, Ahmed Aly, Babak Damavandi, Rakesh Wanga, Anuj Kumar, Rohit Patel, Wen-tau Yih, Xin Luna Dong

**Published**: 2025-10-30 05:50:48

**PDF URL**: [http://arxiv.org/pdf/2510.26160v1](http://arxiv.org/pdf/2510.26160v1)

## Abstract
Wearable devices such as smart glasses are transforming the way people
interact with their surroundings, enabling users to seek information regarding
entities in their view. Multi-Modal Retrieval-Augmented Generation (MM-RAG)
plays a key role in supporting such questions, yet there is still no
comprehensive benchmark for this task, especially regarding wearables
scenarios. To fill this gap, we present CRAG-MM -- a Comprehensive RAG
benchmark for Multi-modal Multi-turn conversations. CRAG-MM contains a diverse
set of 6.5K (image, question, answer) triplets and 2K visual-based multi-turn
conversations across 13 domains, including 6.2K egocentric images designed to
mimic captures from wearable devices. We carefully constructed the questions to
reflect real-world scenarios and challenges, including five types of
image-quality issues, six question types, varying entity popularity, differing
information dynamism, and different conversation turns. We design three tasks:
single-source augmentation, multi-source augmentation, and multi-turn
conversations -- each paired with an associated retrieval corpus and APIs for
both image-KG retrieval and webpage retrieval. Our evaluation shows that
straightforward RAG approaches achieve only 32% and 43% truthfulness on CRAG-MM
single- and multi-turn QA, respectively, whereas state-of-the-art industry
solutions have similar quality (32%/45%), underscoring ample room for
improvement. The benchmark has hosted KDD Cup 2025, attracting about 1K
participants and 5K submissions, with winning solutions improving baseline
performance by 28%, highlighting its early impact on advancing the field.

## Full Text


<!-- PDF content starts -->

CRAG-MM: Multi-modal Multi-turn Comprehensive
RAG Benchmark
Jiaqi Wang1,†,Xiao Yang1,†,Kai Sun1,Parth Suresh1,Sanat Sharma1,Adam Czyzewski1,∗,Derek
Andersen1,Surya Appini1,Arkav Banerjee1,Sajal Choudhary1,Shervin Ghasemlou1,Ziqiang Guan1,∗,
Akil Iyer1,Haidar Khan1,Lingkun Kong1,Roy Luo1,∗,Tiffany Ma1,Zhen Qiao2,David Tran1,Wenfang Xu1,
Skyler Yeatman1,Chen Zhou1,Gunveer Gujral1,Yinglong Xia4,Shane Moon1,Nicolas Scheffer1,Nirav
Shah1,Eun Chang1,Yue Liu1,Florian Metze1,Tammy Stark1,Zhaleh Feizollahi1,Andrea Jessee1,
Mangesh Pujari1,Ahmed Aly1,Babak Damavandi1,Rakesh Wanga1,Anuj Kumar1,Rohit Patel2,Wen-tau
Yih3,Xin Luna Dong1
1Meta Reality Labs,2Meta Superintelligence Labs,3FAIR, Meta,4Meta
†Equal contribution,∗Work done at Meta
Wearabledevicessuchassmartglassesaretransformingthewaypeopleinteractwiththeirsurroundings,
enabling users to seek information regarding entities in their view.Multi-Modal Retrieval-Augmented
Generation (MM-RAG)plays a key role in supporting such questions, yet there is still no comprehensive
benchmark for this task, especially regarding wearables scenarios. To fill this gap, we presentCRAG-
MM—a Comprehensive RAG benchmark for Multi-modal Multi-turn conversations. CRAG-MM
contains a diverse set of 6.5K (image, question, answer) triplets and 2K visual-based multi-turn
conversations across 13 domains, including 6.2Kegocentricimages designed to mimic captures
from wearable devices. We carefully constructed the questions to reflect real-world scenarios and
challenges, including five types of image-quality issues, six question types, varying entity popularity,
differing information dynamism, and different conversation turns. We design three tasks: single-
source augmentation, multi-source augmentation, and multi-turn conversations—each paired with
an associated retrieval corpus and APIs for both image-KG retrieval and webpage retrieval. Our
evaluation shows that straightforward RAG approaches achieve only 32% and 43%truthfulness
on CRAG-MM single- and multi-turn QA, respectively, whereas state-of-the-art industry solutions
have similar quality (32%/45%), underscoring ample room for improvement. The benchmark has
hosted KDD Cup 2025, attracting about 1K participants and 5K submissions, with winning solutions
improving baseline performance by 28%, highlighting its early impact on advancing the field.
Benchmark:https://huggingface.co/crag-mm-2025
Correspondence:First Authors atjqwang@meta.com,xiaoyangfb@meta.com
1 Introduction
Wearable AI devicesare revolutionizing how people interact with computing systems. Modern wearable devices,
such as Rayban Meta1, Rabbit R12, and the Humane AI Pin3, enable vision-based conversations where
users can ask questions about objects in their view. For example, a user may inquire about the history of a
landmark they are viewing, the price of a product they are holding on different e-commerce websites, or repair
instructions for a broken household device. What such questions often have in common isthe need for factual
information that cannot be inferred from the images alone, necessitating multi-modal Retrieval-Augmented
Generation (MM-RAG) systems that access external sources for enriched and accurate responses. To advance
the MM-RAG techniques, we construct theCRAG-MMbenchmark, a comprehensive multi-modal benchmark
of 6.5K single-turn and 2K multi-turn conversations, with an emphasis on use cases relevant to wearable AI
devices. Specifically, our motivation is three-fold.
1https://www.meta.com/ai-glasses
2https://www.rabbit.tech/rabbit-r1
3https://humane.com/
1arXiv:2510.26160v1  [cs.CV]  30 Oct 2025

 
  
Image 
quality: 
Truncated
  
Domain: 
Food
  
Query 
type: 
Simple 
- 
recognition
 
  
Question: 
What 
Brand 
is 
this?
  
Answer: 
It 
belongs 
to 
Evropa, 
one 
of 
the 
oldest    
 
confectionery 
companies 
in 
the 
Balkans.
 
  
Image 
quality: 
Low 
light
  
Domain: 
Sightseeing
  
Query 
type: 
Simple 
- 
knowledge
  
Question: 
What 
green 
building 
award 
did 
this 
place 
win?
 
  
Answer: 
101 
Greenwich 
received 
LEED 
Silver    
 
certification.
 
  
Image 
quality: 
Normal
  
Domain: 
Book
  
Query 
type: 
Multi-hop
 
  
Question: 
How 
many 
books 
has 
this 
book's 
author 
written?
  
Answer: 
Patrick 
o'Donnell, 
the 
author 
of 
the 
 
unvanquished, 
has 
written 
13 
books 
in 
 
total.
 
  
Image 
quality: 
Low 
light
  
Domain: 
Sightseeing
  
Query 
type: 
Aggregation
  
Question: 
What 
movies 
did 
this 
pace 
appear 
in?
  
Answer: 
Firehouse, 
Hook 
& 
Ladder 
Company 
8 
 
shows 
up 
in 
the 
1984 
and 
2016          
 
Ghostbusters 
films, 
as 
well 
as 
in 
the 
2005    
 
film 
Hitch.
 
  
Image 
quality: 
Normal
 
  
Domain: 
Food
  
Query 
type: 
Comparison
 
  
Question: 
Is 
this 
cheaper 
online 
at 
Target 
or 
Walmart?
  
Answer: 
The 
cheaper 
price 
for 
a 
California 
Pizza 
 
Kitchen 
Crispy 
Thin 
Crust 
BBQ 
Chicken 
 
Recipe 
Pizza 
can 
be 
found 
online 
at 
 
Walmart 
for 
$7.46 
, 
which 
is 
cheaper 
than 
                 
Target's 
price 
of 
$7.59.
 
  
Image 
quality: 
Truncated
 
  
Domain: 
Food
  
Query 
type: 
Reasoning
  
Question: 
Can 
I 
take 
this 
product 
in 
my 
carry-on 
bag 
in 
flights?
  
Answer: 
Liquids, 
aerosols, 
gels, 
creams 
and   
 
pastes 
in 
your 
carry-on 
bag 
are 
limited 
to 
 
travel-sized 
containers 
that 
are 
3.4 
 
ounces 
(100 
milliliters) 
or 
less 
per 
item. 
 
Since 
this 
product 
is 
4 
ounces, 
it 
needs 
to 
 
be 
checked 
in.Figure 1CRAG-MM examples.
First, although many Visual Question Answering (VQA) benchmarks exist (Antol et al., 2015; Hudson and
Manning, 2019; Schwenk et al., 2022), they rely primarily on common knowledge (Antol et al., 2015; Goyal
et al., 2017; Schwenk et al., 2022) and visual reasoning (Hudson and Manning, 2019), which is insufficient to
assess factual questions comprehensively. New benchmarks focusing on multi-modal knowledge integration
have emerged in recent years. They nevertheless were constructed solely from Wikipedia (Lerner et al., 2022),
created by templates (Talmor et al., 2021), or consisted of questions uncommon in real life (Talmor et al.,
2021; Chang et al., 2022).CRAG-MMaddresses these limitations by incorporating 8K conversations, with
89% focusing on factual questions that require external information for trustworthy answers.
Second, several question-answering (QA) benchmarks have emerged in recent years, coveringclosed-book(e.g.,
SimpleQA (Wei et al., 2024), FACTS Grounding (Jacovi et al., 2025)) andopen-book(e.g., FreshQA (Vu
et al., 2023)) QA. Some, like CRAG (Yang et al., 2024), specifically target the evaluation of RAG systems.
CRAG-MMgoes beyond them not only by introducingmulti-modaluse cases, but also by including 2k
multi-turn conversations, where ∼38%of them involve domain shifts. This simulates natural topic drift, a
common feature of human conversations, further enhancing the realism of the benchmark.
Third, existing VQA benchmarks typically feature high-quality images. In contrast, wearable devices often use
wide-angle cameras, capturingegocentricimages where objects of interest appear small, rotated, truncated,
occluded, blurred, or poorly lit (Shenoy et al., 2024). To bridge this gap, theCRAG-MMbenchmark includes
7.9K images, with 79% being egocentric, reflecting the real-world challenges in wearable AI applications.
As such,CRAG-MMis, to the best of our knowledge,the first publicly released benchmark capable of effectively
evaluating multi-modal RAGand alsoone of the earliest benchmarks designed to reflect wearable AI use
cases. The design ofCRAG-MMfollows the key features outlined in (Yang et al., 2024), with the following
principles.
Rich and insightful benchmark:CRAG-MMis structured along four key dimensions—image quality,
question type, entity popularity, and conversation complexity, mirroring real-world challenges faced by
wearables QA systems and enabling in-depth analysis and debugging. It covers both common and challenging
use cases; for example, 15% of images are low-quality egocentric images, 21% questions involve torso-to-tail
entities; 52% questions are complex questions and require multi-source information synthesis, and 23%
conversations contain multiple turns. See Fig.1 for some examples fromCRAG-MM.
2

Reliable and fair evaluation:To ensure fair comparison, we provide equal access to retrieval resources,
including APIs for an image knowledge graph (KG) with 68K entries and a webpage corpus with 800K pages.
The retrieval corpus is designed to mimic real-world conditions, with a 1:20 ratio of relevant to irrelevant
information for image search and 1:2 for web search, allowing comparison of different solutions fairly and
easily. The retrieval corpus, search APIs, and relevant metadata allow for evaluating different components of
RAG systems: entity recognition, OCR, query rewrite, response generation, and so on.
Challenging questions:CRAG-MM contains realistic but hard questions. As shown in Tab.4, without
RAG, the most advanced multi-modal large language model (MM-LLM) we evaluated (GPT-5 mini) achieved
an accuracy of 37% on single-turn questions; straightforward solutions improved the accuracy to 50%; even
the most advanced industry solutions (GPT-5), with access to potentially richer web corpus and better search
engine, reached only an accuracy of 63% with 31% hallucinations. Multi-turn conversations contain simpler
questions; still, the best industry solution achieved an accuracy of 70%, with 27% conversations being early
stopped due to two consecutive erroneous or missing answers.
TheCRAG-MMbenchmark served as the foundation for the KDD Cup 2025 Challenge4, attracting nearly
1K participants and 5K submissions. The winning solution improved over straightforward solutions by 28%,
highlighting its early impact on advancing the field.
Differences from CRAG and other existing benchmarks:To the best of our knowledge, CRAG-MM is
the first comprehensive benchmark that focuses on wearable AI use cases. Different from CRAG (Yang et al.,
2024) which targets text-based single-turn QA, CRAG-MM is a visual QA benchmark. Unlike other existing
VQA benchmarks, CRAG-MM uniquely features wearable use cases – questions are based on egocentric images
and often require external knowledge to answer. It encompasses a variety of domains and question types,
which can help reveal interesting insights and facilitate development. Further, CRAG-MM extends beyond
single-turn QA by including multi-turn conversations, a common and essential use case for wearable devices.
Tab.1 compares CRAG-MM to a few popular or recent benchmarks.
Table 1Comparing CRAG-MM to existing benchmarks for factual question answering: SnapNTell (Qiu et al., 2024),
WebQA (Chang et al., 2022), MultiModalQA (Talmor et al., 2021), MM-Vet (Yu et al., 2023), MT-Bench-101 (Bai
et al., 2024), CRAG (Yang et al., 2024).
BenchmarkEgo-
centricImage search
APIWeb search
APIMulti-
modalMulti-
turnTorso &
tail facts
SnapNTell ✗ ✗ ✗✓✗✓
WebQA ✗✓a✓b✓✗ ✗
MultiModalQA ✗ ✗ ✗✓✗ ✗
MM-Vet ✗ ✗ ✗✓✗ ✗
MT-Bench-101 ✗ ✗ ✗ ✗✓✗
CRAG ✗ ✗✓✗ ✗✓
CRAG-MM ✓ ✓ ✓ ✓ ✓ ✓
aPrefetched images.
bPrefetched snippets from wikipedia.
2 Problem description
AnMM-RAG QA systemtakes as input an image Iand a question Q, and outputs an answer A; the answer
is generated by MM-LLMs based on information retrieved from external sources, combined with knowledge
internalized in the model. AMulti-turn MM-RAG QA systemin addition takes questions and answers from
previous turns as context to answer new questions. The answer should provide useful information to answer
the question, without adding any hallucination.
4https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025
3

 
Task 
3 
 
Task 
1 
Question
: 
When 
was 
this 
book 
published?
Image 
search
MM-LLM
Response:
 
The 
book 
was 
published 
in 
2010.
 
Task 
2 
Question
: 
Who 
made 
the 
book 
into 
a 
movie?
Image 
search
MM-LLM
Response:
 
The 
book 
was 
adapted 
into 
a 
movie 
by 
George 
C. 
Wolfe.
Web 
search
Turn 
1
: 
what 
is 
this?
Image 
search
MM-LLM
Response 
1:
 
This 
is 
Dyson 
Airwrap 
Multi-styler 
and 
Dryer.
Web 
search
Turn 
2
: 
How 
much 
does 
it 
sell 
on 
Dyson 
website?
Turn 
3
: 
Can 
the 
voltage 
be 
used 
in 
Europe?
Web 
search
Web 
search
Response 
2:
 
The 
Dyson 
Airwrap 
Multi-styler 
and 
Dryer 
sells 
at 
$599 
at 
Dyson.com
Response 
3:
 
No, 
the 
Dyson 
Airwrap 
Multi-styler 
and 
Dryer 
has 
120v 
voltage, 
and 
hence 
cannot 
be 
directly 
used 
in 
Europe.
MM-LLM
MM-LLMFigure 2CRAG-MM task design.
2.1 Question types
We first define six types of questions as follows.
•Simple-recognition: Questions asking for simple facts that can be directly answered from the image, e.g.,
“what brand is the milk”or“who wrote this book”, where the brand name and the book author are shown
on the image.
•Simple-knowledge: Questions asking for simple facts that require external knowledge to answer, e.g.,
“what’s the price of this sofa on Amazon”.
•Multi-hop questions: Questions that require chaining multiple pieces of information to compose an
answer, such as“what other movies has the director of this movie directed”.
•Comparison questions: Questions require comparing multiple pieces of information, such as“is this
cheaper on Amazon”(where the image is showing a product and its store price).
•Aggregation questions: Questions require aggregating multiple pieces of information,“which drinks do
not contain added sugar among these”(where the image is showing a few drinks in a grocery store).
•Reasoning questions: Questions about an entity that cannot be directly looked up from the retrieved
content and require reasoning to answer, such as“can the dryer be used in Europe”(where the image
shows a dryer).
2.2 Tasks
We designed three tasks. As shown in Fig.2, Task 1 and Task 2 contain single-turn questions, where the
former provides image-KG-based retrieval, and the latter additionally introduces web retrieval; Task 3 focuses
on multi-turn conversations. Here, we provide the content that can be leveraged in QA to ensure a fair
competition. We describe how we generated the data in the next section. The three tasks can guide users of
the benchmark in building end-to-end MM-RAG systems.
Task 1: Single-source augmentation.Task 1 provides an image searchmock APIto access information from
an underlying image-basedmock KG. The mock KG is indexed by the image, and stores structured data
associated with the image; answers to the questions may or may not exist in the mock KG. The mock API
takes an image as input, and returns similar images from the mock KG along with metadata associated
with each image to support answer generation. This task aims to test basic answer generation capability of
MM-RAG systems.
Task 2: Multi-source augmentation.In addition to image search, Task 2 provides a web search mock API as
retrieval source. The webpages likely contain useful information for answering the question, but also contain
4

Table 2Distribution by image quality.
Type Egocentric Normal Total
Normal 5143 1593 6736
Low-light 267 33 300
Blurred 191 24 215
Truncated 370 31 401
Occluded 116 8 124
Rotated 161 6 167
Total 6248 1695 7943
noise. This task aims to test how well the MM-RAG system synthesizes information from different sources.
Task 3: Multi-turn QA.Task 3 tests the system’s ability to conduct multi-turn conversations. Each conversation
contains 2–6 turns. Questions in second turn or beyond may or may not need the image for answering the
questions. Task 3 tests context understanding for smooth multi-turn conversations.
3 Data description
We now describe the benchmark dataset. CRAG-MM covers 13 domains and six types of questions, all in
English. It contains three parts of data: the image set, the QA set, and the contents for retrieval.
3.1 Image set
CRAG-MM contains two types of images:egocentricandnormal. The egocentric images were collected using
smart glass devices from first-person perspective; the normal images were collected from publicly available
images. We collected both types of images via two methods:pre-defined collectionandfree-form collection.
For pre-defined collection, we first specified a list of domains and entity types likely to be involved in user
wearable interactions, such aselectronic productsin theshoppingdomain. We adopted the method described
in Sun et al. (2023) and sampled entities of top, middle and bottom popularity. We defined the popularity
based on some publicly available traffic information for each entity type, and created equal number of questions
for each bucket. We recruited vendors to collect images for the sampled entities using the RayBan Meta
Smart Glasses5whenever possible; otherwise, we searched online to find an image for the target entity from
public sources.
For free-form collection, we asked the vendors to wear a pair of smart glasses and interact in their daily lives.
We recommended to collect80%images from the provided domains, and20%for any additional use cases
that the vendors found plausible for interaction with smart glasses.
We collected 6.2K egocentric images and 1.7K public images in total. We also requested15%of the egocentric
images to be captured under imperfect conditions: low-light, blurred, truncation, occlusion, or rotation. See
Tab.6 in Appendix A.1 for the definition and Tab.2 for the distribution.
3.2 Question Answering data
Single-turn QA data:We constructed the QA set for the collected images based on two sources—KGs or web
contents. For the QA set constructed from KGs, we first leveraged existing entity types and relations in the
KGs to create meaningful QA templates. We then used entities sampled from the image collection to pair with
the question templates and took the associated attribute values as the answer. For the QA set constructed
from web contents, we asked annotators to create plausible questions for wearable devices that could possibly
be answered by web search. The annotators then also created the complete web search query based on the
image and the question, and recorded the ground truth answers. See Appendix A.2 for more details.
5https://www.meta.com/ai-glasses
5

Table 3Distribution of domains and question types for CRAG-MM single- and multi-turn QA. We use the first turn’s
domain for multi-turn conversation.
Simple-
rec.Simple-
know.Multi-
hopComp. Agg. Reason.Single-
turnMulti-
turn
Animal 10 191 67 115 71 66 520 148
Book 12 53 37 17 12 9 140 147
Food 35 312 108 132 99 85 771 196
General Obj. Rec. 26 210 82 74 43 50 485 184
Local 29 428 168 119 106 87 937 181
Math & Science 61 62 16 17 18 20 194 130
Plants & Gardening 19 443 119 125 117 107 930 118
Shopping 52 239 48 66 41 48 494 204
Sports & Games 12 131 31 24 21 12 231 123
Style & Fashion 13 57 15 14 5 9 113 115
Text Understanding 170 113 15 10 21 18 347 113
Vehicle 14 250 147 196 113 140 860 120
Other 9 125 75 68 105 58 440 177
Total 462 2614 928 977 772 709 6462 1956
Multi-turn QA data:We first created a number of simple questions based on the web contents as first-turn
seed questions. We then prompted Llama-3.2-90B-Vision-Instruct (Dubey et al., 2024; Meta AI, 2024) to
create multi-turn conversations based on the given seed questions. Each conversation has two to six turns,
and covers one to three domains. Next, we asked the annotators to review the conversation, remove or revise
turns that are not plausible, too simple, or do not have a single indisputable answer. Finally, the annotators
also created the ground truth answers for each turn by conducting web search.
We collected 6.5K single-turn questions and 2K multi-turn conversations with an average length of 4.9 turns
in the final dataset. Tab.3 summarizes distribution of the questions across different dimensions. The size of
each slice allows us to get metrics with <5%margin-of-error (with 95% confidence level) for most of the slices.
3.3 Retrieval contents
We included two types of content for retrieval to simulate the practical scenarios for MM-RAG: image-based
KG search and text-based web search. We carefully designed the corpus to highlight the real challenges a
retrieval system faces and included potential noises in the retrieval results.
Image-based KG search:Image search is a key component of MM-RAG systems. To simplify, we limit image
search to KG-search. We provide an API that takes an image as input and returns similar images with
associated structured metadata from a mock KG using CLIP ViT-L/14@336px (Radford et al., 2021). Our
image KG covers 93% of the entities referenced in the questions, and querying the index with original images
achieves only 52% recall (Fig.3). See Fig.5 in Appendix A for an example and Appendix A.5.1 for details on
how we created the API.
Text-based web search:We also construct web search to obtain additional information for question answering.
We provide a text-search API that takes a text query as input and returns relevant webpages, with an
estimated recall of 89% (Fig. 3). We chunk the documents into 512-token chunks and embed them using
BGE (Xiao et al., 2024). At inference, queries are matched against the chunks to return corresponding
webpages. See Appendix A.5.3 for details about the web search API creation.
3.4 Data split
The above methods gave us 800K webpage urls (with 2.7M chunks) and a KG of 68K images and 26K entities.
We split the data in two steps. We first held out two domains for private test only, for testing the generalization
of the MM-RAG systems. We then split the remaining domains’ data into three sets with similar distributions:
6

0 5 10 15 20 25 30
Number of retrieved images020406080100Recall (%)
17.620.9 31.837.5
39.345.4
44.550.1
47.553.5
49.756.4
51.958.093.4
Original image
Cropped image
Entity coverage
0 5 10 15 20 25 30 35 40 45 50
Number of retrieved web pages405060708090100Estimated recall (%)
70.0
59.879.5
68.782.9
72.684.6
74.985.6
76.286.3
77.187.1
77.587.8
78.088.1
78.688.6
78.9
Full pages
Page snippetsFigure 3 Image search recall (left): the image search index covers 93.4% of the query entities. However, using the full
image to query the index only achieves 51.9% recall. Manual cropping improves the recall slightly to 58.0%.Web
search recall (right): the top 50 web search results are estimated to contain the ground truth facts for 88.6% questions.
validation, public test, andprivate test, containing 30%, 30%, 40% of the data respectively. We released the
validation and public test sets on HuggingFace6.
4 Metrics and Evaluation
We adopt the metrics and evaluation in Yang et al. (2024) and extend them for multi-turn QA.
Single-turn QA:We score the model response withperfect(fully correct),missing(e.g., “I don’t know”, “I’m
sorry I can’t find ...”), orincorrect(wrong or irrelevant) answers as1,0, and −1, and computeTruthfulnessas
the average score.
Multi-turn QA:There is no dominant way to evaluate multi-turn conversations. We adapt the method in Bai
et al. (2024), which is mostly closely aligned with information-seeking tasks (as opposed to task fulfilling). In
particular, weearly stopa conversation when the answers for two consecutive turns are wrong or missing,
and consider the remaining turns in the same conversation as missing—mimicking the behavior of users after
seeing repeated failures. We then calculate the average score of all turns as theTruthfulnessfor the multi-turn
session, and take the grand average for all conversations.
Auto-evaluation:We adopt LLM-as-a-judge to evaluate the quality of the answers, which achieves an accuracy
of 99.1% (Tab.7 in Appendix B.1).
5 Benchmarking
In this section, we present the performance of visual QA systems on the CRAG-MM public test set, demon-
strating that CRAG-MM can provide insights and indicate directions for developing MM-RAG techniques.
We answer the following three research questions (RQ) through our experiments.
RQ1:DoesCRAG-MMreveal new challenges beyond straightforward solutions?
RQ2:How well do state-of-the-art industry solutions perform onCRAG-MM?
RQ3:What directions for quality improvement are suggested by theCRAG-MMbenchmark?
5.1 RQ1: New Challenges Beyond Straightforward MM-RAG Solutions
We present results for three models in this section and leave an extensive comparison in Appendix C.3.
The three models are: Llama-3.2-90B-Vision-Instruct ,GPT-5-mini (OpenAI, 2025) and Gemini-2.5-Flash
6https://huggingface.co/crag-mm-2025
7

Table 4Performance of straightforward solutions on CRAG-MM single- and multi-turn QA. All numbers are in
percentage.Boldfor best quality andItalicfor best quality within the group. Even the best MM-LLM achieves only
18% truthfulness for single-turn QA and 30% for multi-turn QA; straightforward MM-RAG solutions improve only to
32% for single-turn and 43% for multi-turn.
Model Acc. Miss. Hallu. Truth. Early Stop.
Single-turn MM-LLMLlama 3.2 90B 28.2 39.1 32.8 -4.6 -
Gemini 2.5 Flash 36.638.025.4 11.2 -
GPT-5 Mini37.443.719.0 18.4-
Task 1Llama 3.2 90B 13.5 71.015.6-2.1 -
Gemini 2.5 Flash 37.936.825.3 12.6 -
GPT-5 Mini39.343.9 16.822.5-
Task 2Llama 3.2 90B 30.1 50.1 19.8 10.3 -
Gemini 2.5 Flash49.922.627.5 22.4 -
GPT-5 Mini 48.7 34.117.231.5-
Multi-turn MM-LLMLlama 3.2 90B 42.225.032.8 12.7 64.7
Gemini 2.5 Flash 29.2 57.513.416.5 88.1
GPT-5 Mini48.934.0 17.130.4 60.8
Task 3Llama 3.2 90B 37.1 46.1 16.9 18.9 81.7
Gemini 2.5 Flash 54.4 24.2 21.4 31.4 55.8
GPT-5 Mini61.0 22.516.542.5 43.5
(Comanici et al., 2025). We implemented an MM-LLM-only solution and three straightforward RAG solutions.
See Appendix C.1 for more details.
•MM-LLM-only:We employ a simple prompting strategy that encourages concise answers and the use
of“I don’t know”when the confidence is low.
•Task 1:Our Task 1 solution uses the entire image to query the image search API and retrieves
top 30 results. After applying a threshold of 0.75 to the similarity score, the retrieved metadata are
concatenated to fill in a fixed size context window of 2K tokens.
•Task 2:Our Task 2 solution incorporates query rewriting for web search and concatenates webpage
snippets as additional reference text, expanding the context window up to 8K tokens (similar to Kandpal
et al. (2023) and Mallen et al. (2023)).
•Task 3:In addition to the multi-source retrieval content, our Task 3 solution includes full conversation
history to provide context for the current turn.
Tab.4 shows the overall performance of MM-LLM-only and straightforward MM-RAG solutions. We observe
that the best performing MM-LLM-only solution (GPT-5 Mini) achieves only 37% accuracy and 18%
truthfulness on single-turn QA, and 49% accuracy and 30% truthfulness on multi-turn QA, showing CRAG-
MM is challenging without RAG.
For single-turn QA, adding image KG search (Task 1) in a vanilla manner yields only marginal improvement
(+4%). Adding web search (Task 2) increases answer accuracy but also slightly increases hallucinations for all
models; the highest accuracy and truthfulness is only 50% and 32% respectively. We observe similar trend for
multi-turn QA, highlighting the gap that still exists after applying straightforward solutions.
5.2 RQ2: Industry SOTA and Competition Winning Solutions
We report results from two groups of solutions. First, we report winning solutions in the CRAG-MM
leaderboard7, where participating teams had three months to build their best-performing systems with the
search API described in Section 3.3. These systems generally combined multi-task learning with supervised
fine-tuning on the Llama-3.2-11B-Vision-Instruct model (Meta AI, 2024), For comparison, we also report
7https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/leaderboards
8

Table 5Performance of winnings’ solutions and industry SOTA MM-RAG systems on CRAG-MM.
System Acc. Miss. Hallu. Truth. Early Stop.
Single-turnLlama 3.2 11B RAG35.3 20.843.9 -8.6 -
Winning team 29.3 61.29.619.7-
Claude Sonnet 45.7 5.8 48.5 -2.8 -
Gemini 58.23.138.7 19.5 -
GPT-562.76.830.532.2-
Multi-turnLlama 3.2 11B RAG46.6 11.042.4 8.863.7
Winning team 44.4 40.215.426.665.9
Claude Sonnet 58.7 6.6 34.6 28.625.0
Gemini 66.23.530.3 30.1 37.0
GPT-570.03.926.145.026.9
results of the straightforward solution using the same base model. Additionally, we report industry state-of-the-
art (SOTA) MM-RAG solutions. We selected three systems built upon MM-LLMs and search engines, queried
them with CRAG-MM questions, collected the responses, and evaluated the quality (details in Appendix D).
Tab.5 shows the performance of these solutions. Note that the truthfulness scores between the straightforward,
leaderboard winning team, and SOTA solutions are not fully comparable, as industry solutions use larger
models and potentially have access to much richer content repository; results from winning teams were
reported on the private test set, while the other solutions used the public test set. But we believe the trend
still holds. We made several observations from the results.
First, the SOTA systems achieved comparable truthfulness compared to the straightforward solutions (+0%
for single-turn and +2% for multi-turn), but with a higher accuracy (+13% for single-turn and +9% for
multi-turn). However, the hallucination rate is still high (31%-49% for single-turn and 26%-35% for multi-turn),
showing a substantial gap in trustworthy visual QA systems. Second, the winning team’s solution, though
lower quality compared to industry solutions, achieved substantial improvement over straightforward MM-RAG
solutions using the same base model (+28% for single-turn and +18% for multi-turn). They also have the
lowest hallucination rates, primarily because the models are able to more properly abstain when uncertain;
however, this also causes lower accuracy and higher missing rate, showing the direction for improvement.
5.3 RQ3. Directions for Quality Improvements
Finally, we show the quality of the winning and industry SOTA solutions on different slices of CRAG-MM in
Fig.4. (See Fig.7 in AppendixC.4 for slicing result for straightforward solutions.)
First, all SOTA systems evaluated show degraded truthfulness for low quality images (dropped by up to
46%, Fig.4a), showing the need for more robustness in image understanding. “Low-light" and “occluded"
are the most difficult categories with best truthfulness scores 20% and 24% only. Interestingly, leaderboard
winning solutions have fairly stable quality across different categories, showing the possibility of robustness
improvements. Second, entity recognition is much harder when relying solely on visual information compared
to leveraging text clues (dropped by up to 37%, Fig.4b). The systems also face much larger challenges when
answering questions regarding entities that are less popular (Fig.4c), both because of the difficulties in entity
recognition, and because of lack of abundant information. Third, MM-RAG systems often face challenges
in answering questions requiring specific information (Simple-knowledge), synthesizing multiple pieces of
information (comparison and aggregation), and multi-hop information retrieval (Fig.4d), showing challenges
to build MM-RAG systems and substantial room for improving the retrieval, recognition and reasoning ability
of the model. Finally, having smooth multi-turn conversation remains to be a grand challenge. Consistent
with Laban et al. (2025), almost all SOTA systems find it difficult to answer questions that require context
from conversation history (dropped by up to 22%, Fig.4e). Even the best performing system still has 27%
conversation sessions being early stopped (Tab.5), and has on average only 3.2 turns successful among 4.9 in
total (Fig.4f).
9

NormalLow-lightBlurredTruncated Occluded Rotated20
10
01020304050
212526
2224
2249
2036
32
243536
2
-812
-1024
19
-18
-26-10
-210(a) Image quality
OCR Non-ocr10
010203040
26
1743
2646
912
-9(b) Entity recognition
Head T orso T ail30
20
10
010203040
21
51338
0612
-6
-18-7
-32
-35(c) Entity popularity
Simple-rec.Simple-know.Reasoning Comparison AggregationMulti-hop20
0204060
29 29
24
18
-22455
3858
40
134047
2834
21
-1921618
10
6
-24-6(d) Question type
True False01020304050
2134
3251
2543
1133(e) Context-dependent
0.00.51.01.52.02.53.0
1.93.2
2.9
2.5(f) Average Number of Successful Turns
Winning team GPT-5 Gemini Claude SonnetFigure 4Winning and SOTA systems’ performance across different dimensions. Figure (a)–(e) show truthfulness; (f)
shows the average number of successful turns in multi-turn QA.
6 Conclusion
In this paper, we propose CRAG-MM, the first MM-RAG benchmark tailored to wearable AI applications.
Its principled design and accessible search APIs empower systematic evaluation of MM-RAG capabilities and
inform directions for future advances.
10

References
Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo
Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b.arXiv preprint arXiv:2410.07073,
2024.
Anthropic. Claude 3.7 sonnet: Hybrid reasoning model. https://www.anthropic.com/news/claude-3-7-sonnet , 2025.
Accessed: 2025-09-03.
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi
Parikh. Vqa: Visual question answering. InProceedings of the IEEE international conference on computer vision,
pages 2425–2433, 2015.
Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge,
Bo Zheng, et al. Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn
dialogues.arXiv preprint arXiv:2402.14762, 2024.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report.arXiv preprint arXiv:2502.13923, 2025.
Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop
and multimodal qa. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
16495–16504, 2022.
Chroma. Chroma: The ai-native open-source embedding database.https://www.trychroma.com/.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein,
Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality,
long context, and next generation agentic capabilities.arXiv preprint arXiv:2507.06261, 2025.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.arXiv preprint arXiv:2407.21783,
2024.
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating
the role of image understanding in visual question answering. InProceedings of the IEEE conference on computer
vision and pattern recognition, pages 6904–6913, 2017.
Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional
question answering. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
6700–6709, 2019.
Alon Jacovi, Andrew Wang, Chris Alberti, Connie Tao, Jon Lipovetz, Kate Olszewska, Lukas Haas, Michelle Liu, Nate
Keating, Adam Bloniarz, et al. The facts grounding leaderboard: Benchmarking llms’ ability to ground responses to
long-form input.arXiv preprint arXiv:2501.03200, 2025.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to
learn long-tail knowledge. InInternational Conference on Machine Learning, pages 15696–15707. PMLR, 2023.
Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. Llms get lost in multi-turn conversation.arXiv
preprint arXiv:2505.06120, 2025.
Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G Moreno, and Jesús
Lovón Melgarejo. Viquae, a dataset for knowledge-based visual question answering about named entities. In
Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information
Retrieval, pages 3108–3120, 2022.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust
language models: Investigating effectiveness of parametric and non-parametric memories. InACL, 2023.
Meta AI. Llama 3.2: Revolutionizing edge ai and vision on devices. https://ai.meta.com/blog/llama-3-2-connect-2
024-vision-edge-mobile-devices/, 2024. Accessed: 2025-09-03.
Meta AI. The llama 4 herd: Introducing maverick & scout. https://ai.meta.com/blog/llama4-release , 2025. Accessed:
2025-09-21.
OpenAI. Gpt-5.https://platform.openai.com/docs/models/gpt-5, 2025. Accessed: 2025-09-03.
11

Jielin Qiu, Andrea Madotto, Zhaojiang Lin, Paul A Crook, Yifan Ethan Xu, Xin Luna Dong, Christos Faloutsos, Lei
Li, Babak Damavandi, and Seungwhan Moon. Snapntell: Enhancing entity-centric visual question answering with
retrieval augmented multimodal llm.arXiv preprint arXiv:2403.04735, 2024.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda
Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning, pages 8748–8763. PmLR, 2021.
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A
benchmark for visual question answering using world knowledge. InEuropean conference on computer vision, pages
146–162. Springer, 2022.
Ashish Shenoy, Yichao Lu, Srihari Jayakumar, Debojeet Chatterjee, Mohsen Moslehpour, Pierce Chuang, Abhay
Harpale, Vikas Bhardwaj, Di Xu, Shicong Zhao, et al. Lumos: Empowering multimodal llms with scene text
recognition. InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages
5690–5700, 2024.
Brave Software. Brave search api.https://search.brave.com/.
Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. Head-to-tail: How knowledgeable are large
language models (llm)? aka will llms replace knowledge graphs?arXiv preprint arXiv:2308.10168, 2023.
Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions, 2018.
Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi,
and Jonathan Berant. Multimodalqa: Complex question answering over text, tables and images.arXiv preprint
arXiv:2104.06039, 2021.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou,
Quoc Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation, 2023.
Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong
Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025.
Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and
William Fedus. Measuring short-form factuality in large language models.arXiv preprint arXiv:2411.04368, 2024.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources
for general chinese embeddings. InProceedings of the 47th international ACM SIGIR conference on research and
development in information retrieval, pages 641–649, 2024.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui,
Ziran Will Jiang, Ziyu Jiang, et al. Crag–comprehensive rag benchmark.arXiv preprint arXiv:2406.04744, 2024.
Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang.
Mm-vet: Evaluating large multimodal models for integrated capabilities.arXiv preprint arXiv:2308.02490, 2023.
12

Appendix
A Dataset
A.1 Definition of Image Quality
Table 6Definition of low quality images.
Category Definition
Low-lightThe entity referred in the query is illuminated by a small amount of light, such as at dusk,
during night time, or in shadowed areas.
Blurred The entity referred in the question appears to be fuzzy, smeared, or indistinct in the image.
Truncated The entity referred in the query is partially out of the image.
OccludedThere is something, like a finger, in the way of the camera and the entity.
E.g. the entity is behind a chain link fence or a window screen.
Rotated The entity referred in the query is more than 10 degrees off from a right orientation.
A.2 QA Data Creation
A.3 QA set constructed from KGs
Starting from the entities sampled during the image collection, we created a set of simple-knowledge QA data
following three steps.
1.For each entity type, we selected a meaningful relation( e, r)and created question templates. For
example, for(product, brand), we create a question"what is the brand of this product?".
2.We then used entities sampled for the image collection to pair up with the question. We created equal
number of questions for head, torso and tail entities.
3.Last, we took the associated attribute values as the answer to the question to obtain question answer
pairs.
We created theComparison, Aggregation, and Reasoningquestions in a similar way but made sure the sampled
subject entities can form meaningful questions given the question type. We used heuristics to select entity
types for such questions.
Finally, we created multi-hop questions in 3 steps, similar to Talmor and Berant (2018). We first sampled an
entity e1from the KG, and selected 2 relation triplets following a two-hop path:( e1, r1, e2)and( e2, r2, e3).
We then created a question template describing the path. For example, for path(book 1, author, person)
followed by(author, write, book), we created the template"what is the latest book of this book’s author?".
The answer to the question will be contained ine 3in the second triplet.
A.4 QA pairs constructed from web contents
We created QA pairs from web search results, in three steps.
1.Given the collected images, we asked annotators to write down questions that are plausible to interact
with wearable devices and could possibly be answered by web search. For example,"how much is this
vehicle?"(where the image is showing a 2024 volkswagon tiguan).
2.Next, the annotators will create web search queries based on the image and the question, such as"how
much is the 2024 volkswagon tiguan?"for the above example. Then they will search the web using the
search query to find relevant result for answering the question. Note that we only do text-based web
search and rely on image search to provide the entity name.
13

3. Finally, the annotators note down the ground truth answers based on the web search results.
A.5 Search API Creation
A.5.1 Image search API creation
We constructed the image-based mock KG in four steps.
1.We first included the images used to generate the questions in Section A.3, and defined coverage
categories based on these seed images.
2.Second, we collected candidate data from the open web (e.g., Wikipedia), including fine-grained entities
within each category, their metadata, images, and inter-entity relationships.
3.From this pool, we selected candidate images to populate the mock KG. To ensure entity coverage, we
added positive examples. To create a more realistic testing environment, we also added hard negatives,
i.e. images visually similar to the query images but associated with different entities. Formally, letI
denote the set of images for the entity setEthat we used to construct the KG question answer data.
For each i∈I, we added up to30hard negatives, denoted by S(i). We further augmented the set with
randomly sampled images for the entities inS(i), denoted asE(S(i)), i.e.,E∪S
i∈IE(S(i)).
4.Finally, we removed the query images, simulating a realistic setting in which user queries are absent
from the KG, but visually similar or related images remain.
All images in the mock KG were encoded with CLIP ViT-L/14@336px , and we built the image retrieval
index using ChromaDB(Chroma).
-Entity:                      8 Spruce Street-Completion date:    2010-Architectural style: Deconstructivism-Architect:                 Frank Gehry-……-Entity:                      8 Spruce Street-Completion date:    2010-Architectural style: Deconstructivism-Architect:                 Frank Gehry-……
…Image SearchQuery Image
Figure 5An example of the image search API.
A.5.2 Image Search Recall
Fig.6 shows the recall of the image search API for the two image types by using a crude method (query by
the original image). Egocentric images has lower recall compared to the normal images, showing the difficulty
of entity recognition and image understanding on egocentric images.
14

0 5 10 15 20 25 30
Number of retrieved images0204060Recall (%)
14.822.2
26.240.9
33.249.2
37.456.2
39.860.1
41.762.8
43.565.8
Egocentric image
Normal imageFigure 6Image search recall by image type using full images to query the index.
A.5.3 Web Search API Creation
We created the webpage repository in three steps.
1.We created the ground truth search query and some negative search queries. We constructed the ground
truth query by asking the annotators to write down a standalone query for answering the given question
regarding the image, without the need to use the image. For example, "who wrote the Martian" for
"who wrote this book". To simulate the noise, we also created negative search queries in two ways: 1) a
query including the ground truth entity only, 2) queries including similar entities to the ground truth
entity. For example, we use Toyota Rav4 if the question is about Volkswagon Tiguan.
2.We next used each query to search and store up to20urls via the Brave search API Software. We
collected up to50distinct webpages for each question by combining the search results returned by all
the queries and retained the top 50.
3.We then pooled the webpages for all questions together and constructed a mock web search API using
ChromaDB.
A.5.4 Unified Search API
We built a light-weight Python package, providing a unified API to support image and web search. The API
can take image URL, PIL image and text query and decide which index (image or web search index) to query
depending on the input data type. Below are examples of how to use the search API.
# image search example
results = search_pipeline(image, k = 1)
for result in results:
print(result)
print(’\n’)
# output
{’index’: 17030, ’score’: 0.906402587890625, ’url’: ’https://upload.wikimedia.org/wikipedia/commons/3/34/
The_Beekman_tower_from_the_East_River_%286215420371%29.jpg’, ’entities’: [{’entity_name’: ’8 Spruce Street’, ’
entity_attributes’: {’name’: ’8 Spruce Street<br />(New York by Gehry)’, ’image’: ’8 Spruce Street (01030p).jpg’, ’
image_size’: ’200px’, ’address’: ’8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038’, ’mapframe_wikidata’: ’yes
’, ’coordinates’: ’{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}’, ’status’: ’
Complete’, ’start_date’: ’2006’, ’completion_date’: ’2010’, ’opening’: ’February 2011’, ’building_type’: ’[[Mixed-use
development|Mixed-use]]’, ’architectural_style’: ’[[Deconstructivism]]’, ’roof’: ’{{convert|870|ft|m|0|abbr|=|on}}’, ’
top_floor’: ’{{convert|827|ft|abbr|=|on}}’, ’floor_count’: ’76’, ’floor_area’: ’{{convert|1000000|sqft|m2|abbr|=|on}}’, ’
architect’: ’[[Frank Gehry]]’, ’structural_engineer’: ’[[WSP Group|WSP Cantor Seinuk]]’, ’main_contractor’: ’Kreisler
Borg Florman’, ’developer’: ’[[Forest City Ratner]]’, ’engineer’: ’[[Jaros, Baum & Bolles]] (MEP)’, ’owner’: ’8 Spruce (NY)
Owner LLC’, ’management’: ’Beam Living’, ’website’: ’{{URL|https://live8spruce.com/}}’}}]}
# web search example (results partially shown)
15

query=’What to know about Andrew Cuomo?’
results = search_pipeline(query, k=1)
for result in results:
print(result)
print(’\n’)
# output
{’index’: ’https://en.wikipedia.org/wiki/Mario_Cuomo_chunk_2’, ’score’: 0.5727531909942627, ’page_name’: ’Mario Cuomo -
Wikipedia’, ’page_snippet’: ’He vigorously attacked Ronald Reagan&#x27;s ... brought him to national attention, most
memorably saying: &quot;There is despair, Mr. President, in the faces that you don&#x27;t see, in the places that you
don&#x27;t visit, in your shining city.&quot; He was immediately considered one of the frontrunners for the Democratic
...He vigorously attacked Ronald Reagan\’s ... brought ......’, ’page_url’: ’https://en.wikipedia.org/wiki/Mario_Cuomo’}
B Evaluation
B.1 Auto-evaluation with LLM-as-a-judge
We used GPT-4o to evaluate whether the responses are correct, incorrect or missing, given the queries and
ground truth answers. To encourage concise answers, we truncate the answers to 75 tokens for auto-evaluation.
Tab.7 shows the overall and per bucket performance. The judge achieves 99% average accuracy and 91%
average F1 score on the three buckets.
Table 7Performance of the GPT-4o based auto-evaluation judge.
Accuracy Precision Recall F1 score
Accurate98.8 98.1 90.8 94.3
Incorrect98.7 68.1 91.7 78.2
Missing100.0 100.0 100.0 100.0
Average99.1 88.7 94.2 90.8
Below is the prompt used for auto-evaluation. We include ∼20 in-context examples to improve the judge
performance but only show one example due to limited space.
Auto-evaluation prompt
You will be given a question, a ground truth answer, and a model prediction. Your task is to judge if the prediction is
correct or not based on the ground truth answer.
## Instructions
Read the question, ground truth answer, and model prediction carefully. Follow the step by step guideline below to make a
judgment.
1. If the prediction indicates uncertainty or refusal to answer, output "Result: WRONG".
2. If the prediction exactly matches the ground truth, output "Result: CORRECT".
3. If the ground truth is a number
3.1 If the prediction gives a number that almost exactly matches the ground truth, output "Result: CORRECT".
3.2 If the prediction gives a number that is not the same as the ground truth, output "Result: WRONG".
4. If the prediction is self-contradictory, output "Result: WRONG".
5. If the prediction is not answering the question, output "Result: WRONG".
6. If ground truth contains a set of objects,
6.1 if the prediction contains exactly same objects as the ground truth, output "Result: CORRECT".
6.2 if the prediction contains different objects from the ground truth, output "Result: WRONG".
6.3 if the prediction is almost same as the ground truth, use your best judgement to give output.
7. If the prediction is grounded by the ground truth, output "Result: CORRECT".
8. If the prediction is unrelated or contradictory to the ground truth, output "Result: WRONG".
## Additional Guidelines
- Take it as granted that the ground truth is always correct.
- If the prediction gives extra information that is not in the ground truth, it is still correct as long as it is grounded by the
ground truth.
- Be careful about numbers. 1 mile is about 1.60934 km. 1 foot is about 0.3048 m. 1 inch is about 2.54 cm. 1 yard is about
0.9144 m. 1 pound is about 0.453592 kg. 1 gallon is about 3.78541 liters. 1 ounce is about 28.3495 grams.
## Output Format
16

Your judgment should first provide a VERY-SHORT explanation on your rationale. When relevant, you need to include the
guidelines above to explain your judgment. Finally, your judgment should clearly state "Result: CORRECT" or "
Result: WRONG".
Below are some examples:
EXAMPLES START
Question: who will win the game?
Ground Truth: Lakers is favored to win the game.
Prediction: Sorry, it is hard to predict the outcome of the game.
Explanation: The prediction indicates it is not sure about the answer. So the prediction is incorrect according to the
guideline 1.
Result: WRONG
. . .
EXAMPLES END
B.2 Manual evaluation
We use manual evaluation to evaluate the auto-evaluation judge performance and winner selection for the
competition (see Sec5.2). Trained human annotators score the answer with perfect (fully correct), acceptable
(useful w. minor non-harmful errors), incorrect (wrong or irrelevant) or missing (“I don’t know”).
C Evaluating straightforward MM-RAG solutions
C.1 Implementation details
Single-source augmentation (Task 1).We leverage an image search API to retrieve visually similar images and
their associated metadata, providing external knowledge to support entity recognition and answer generation,
particularly for torso-to-tail entities that MM-LLMs often hallucinate. In our baseline implementation, we use
the full input image to query the image search API and retrieve the top 30 images and metadata. A similarity
threshold of 0.75 is applied to filter noisy results. The retrieved entity names are retained, and the associated
metadata is truncated to 2k tokens. The resulting image search content is appended to the model prompt for
response generation.
Multi-sourceaugmentation(Task2and3).We incorporate web content in addition to image search, implementing
a multi-stage pipeline that integrates image search, query rewriting, web search, and answer generation:
Specifically:
1. We first retrieve the top 30 entities and associated metadata (as in the single-source setting).
2.Given the input image, query, and retrieved content, we use Llama-3.2-11B-Vision-Instruct to rewrite
the original query into a fully specified textual query independent of the image.
3. The rewritten query is used to query the web search API, returning up to 50 relevant webpages.
4.We construct the final prompt by concatenating the image search content (up to 2k tokens), web snippets
(up to 8k tokens), and the query.
5.For multi-turn QA, we additionally include the full conversation history, with previous responses
generated by the agent under evaluation.
C.2 Prompt construction
We show our prompts used to produce the evaluation results in Tab. 4, Tab. 8 and Tab. 9. All the models or
APIs are evaluated under the same prompt template.
Prompt template for MM-LLM only solution, single-turn QA
<|system|>
You are a helpful assistant that truthfully answers the user question given an image.
17

Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Keep your response concise and to the point. Strive to answer in one sentence.
3. If you are uncertain or don’t know the answer, respond with "I don’t know".
<|user|>
Image: <image>
Query: <query>
Prompt template for MM-LLM only solution, multi-turn QA
<|system|>
You are a helpful assistant that truthfully answers the user question given an image. You are able to engage in multi-turn
conversations. Answer the new question based on the given image and information extracted from the previous
conversations.
Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Build upon previous conversations when responding.
3. Keep your response concise and to the point. Strive to answer in one sentence.
4. If you are uncertain or don’t know the answer, respond with "I don’t know".
<|user|>
Image: <image>
Conversation history: <conversation_history>
Query: <query>
Prompt template for single-turn augmentation
<|system|>
You are a helpful assistant that truthfully answers the user question given an image.
Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Keep your response concise and to the point. Strive to answer in one sentence.
3. If you are uncertain or don’t know the answer, respond with "I don’t know".
<|user|>
Image: <image>
Image search context:
Here is a list of entity names retrieved based on visual similarity to the provided image:
<list_of_entity_names>
Here are some additional attributes for some of the entities. Only incorporate this information into your answer ONLY IF
you are confident the referenced entity is in the provided image.
Entity <i>: <entity_name>
Entity attributes: <meta_data>
. . .
Entity <n>: <entity_name>
Entity attributes: <meta_data>
Incorporate these image entity information into your answer ONLY IF you are confident they refer to the exact same entity
. Disregard them otherwise.
Query: <query>
Prompt template for multi-source augmentation
<|system|>
You are a helpful assistant that truthfully answers the user question given an image.
Please follow these guidelines when formulating your response:
1. Your response must be grounded in the image and based on factual information.
2. Keep your response concise and to the point. Strive to answer in one sentence.
3. If you are uncertain or don’t know the answer, respond with "I don’t know".
<|user|>
Image: <image>
Image search context:
Here is a list of entity names retrieved based on visual similarity to the provided image:
<list_of_entity_names>
18

Here are some additional attributes for some of the entities. Only incorporate this information into your answer ONLY IF
you are confident the referenced entity is in the provided image.
Entity <i>: <entity_name>
Entity attributes: <meta_data>
. . .
Entity <n>: <entity_name>
Entity attributes: <meta_data>
Incorporate these image entity information into your answer ONLY IF you are confident they refer to the exact same entity
. Disregard them otherwise.
You are given snippets from web page search results based on this question: "{query}". These page snippets may or may
not contain relevant or truthful information about the question. Incorporate these information into your answer
ONLY IF you are confident they address the question. Disregard them otherwise.
Web search context:
<DOC>
Webpage ID: <i>
Title: <page_name>
Web content snippet: <snippet>
</DOC>
. . .
<DOC>
Webpage ID: <n>
Title: <page_name>
Web content snippet: <snippet>
</DOC>
Incorporate these web search results into your answer ONLY IF you are confident they contain relevant or truthful
information about the question. Disregard them otherwise.
Query: <query>
C.3 Performance of straightforward solutions
In addition to the models discussed in Section 5.1, we evaluated our straightforward MM-RAG solutions
on CRAG-MM with the following MM-LLMs to provide a more comprehensive performance comparison:
Llama-4-Maverick (Meta AI, 2025), Pixtral-12B-2049 (Agrawal et al., 2024), Qwen-2.5-VL-72B (Bai et al.,
2025), InternVL3.5-38B (Wang et al., 2025) and Claude 3.7 Sonnet (Anthropic, 2025). Tab. 8 and Tab. 9
present the evaluation results for single-turn and multi-turn QAs respectively.
C.4 Performance of straightforward solutions on CRAG-MM slices
Fig.7 shows the performance of the MM-LLM-only and straightforward solutions on different slices of CRAG-
MM,showingthatCRAG-MMrevealsinterestinginsightsandallowslargeroomforimprovementfordeveloping
MM-RAG systems.
First, understanding the entity and context in the image is important and challenging for wearable QA,
particularly when there is quality issue in the image (Fig.7a) or when text is not present - the model needs to
recognize the entity purely from the visual information (Fig.7b). Adding image search in a crude manner shows
mixed effect on different low quality categories (-11% - +10%), showing the need for proper pre-processing or
smarter image search.
Second, properly leveraging the retrieved information is non-trivial, particularly when the query involves
entities that are less popular (Fig.7c), requires external knowledge, or demands synthesizing multiple pieces of
information to produce an answer (Fig.7d).
Third, results on multi-turn conversation reveals a large gap in conducting smooth conversation - more than
44% conversations were early stopped (has two consecutive failures, Tab.9); the best truthfulness score is only
36% for questions that require conversation history to answer (Fig.7e); on average the best system only has
2.7 successful turns among the 4.9 total for each multi-turn conversation (Fig.7f).
19

Table 8Performance of straightforward MM-RAG solutions on CRAG-MM single-turn QA.
Single-turn Model Acc. Miss. Hallu. Truth.
MM-LLMLlama 3.2 11B 24.4 34.4 41.3 -16.9
Llama 3.2 90B 28.2 39.1 32.8 -4.6
Llama 4 Maverick (FP8) 33.311.055.7 -22.4
Pixtral 12B 2409 21.9 36.1 42.1 -20.1
Qwen2.5 VL 72B 26.0 45.9 28.1 -2.1
InternVL3.5 38B 18.0 54.7 26.6 -7.9
Claude 3.7 Sonnet 30.9 28.9 40.2 -9.3
Gemini 2.5 Flash 36.6 38.0 25.4 11.2
GPT-5 Mini37.443.719.0 18.4
Task 1Llama 3.2 11B 19.0 52.4 28.7 -9.7
Llama 3.2 90B 13.5 71.0 15.6 -2.1
Llama 4 Maverick (FP8) 23.5 52.5 24.0 -0.5
Pixtral 12B 2409 23.1 35.2 41.6 -18.5
Qwen2.5 VL 72B 20.3 63.1 16.6 3.7
InternVL3.5 38B 13.4 74.212.41.0
Claude 3.7 Sonnet 24.521.554.0 -29.5
Gemini 2.5 Flash 37.9 36.8 25.3 12.6
GPT-5 Mini39.343.9 16.822.5
Task 2Llama 3.2 11B 35.3 20.8 43.9 -8.6
Llama 3.2 90B 30.1 50.1 19.8 10.3
Llama 4 Maverick (FP8) 44.1 15.0 40.9 3.2
Pixtral 12B 2409 42.1 4.9 53.1 -11.0
Qwen2.5 VL 72B 31.8 44.7 23.6 8.2
InternVL3.5 38B 37.9 34.3 27.8 10.1
Claude 3.7 Sonnet 30.70.968.4 -37.7
Gemini 2.5 Flash49.922.6 27.5 22.4
GPT-5 Mini 48.7 34.117.231.5
Table 9Performance of straightforward MM-RAG solutions on CRAG-MM multi-turn QA.
Multi-turn Model Acc. Miss. Hallu. Truth. Early Stop.
MM-LLMLlama 3.2 11B 36.5 19.5 44.0 1.6 74.9
Llama 3.2 90B 42.2 25.0 32.8 12.7 64.7
Llama 4 Maverick (FP8) 43.4 16.0 40.6 9.8 65.7
Pixtral 12B 2409 27.9 44.2 27.9 2.5 82.3
Qwen2.5 VL 72B 40.1 25.6 34.3 12.1 71.3
InternVL3.5 38B 37.69.453.0 -1.1 73.0
Claude 3.7 Sonnet 38.2 30.2 31.6 10.9 74.6
Gemini 2.5 Flash 29.2 57.513.416.5 88.1
GPT-5 Mini48.934.0 17.130.4 60.8
Task 3Llama 3.2 11B 46.6 11.0 42.4 8.8 63.7
Llama 3.2 90B 37.1 46.1 16.9 18.9 81.7
Llama 4 Maverick (FP8) 53.9 9.5 36.6 19.7 55.5
Pixtral 12B 2409 48.7 6.5 44.7 9.8 62.3
Qwen2.5 VL 72B 42.0 30.2 27.9 17.0 71.0
InternVL3.5 38B 50.3 16.9 32.8 20.5 58.0
Claude 3.7 Sonnet 55.45.039.6 19.6 49.7
Gemini 2.5 Flash 54.4 24.2 21.4 31.4 55.8
GPT-5 Mini61.022.516.542.5 43.5
20

NormalLow-lightBlurredTruncated Occluded Rotated20
10
0102030
1 1
-9-4-2
-95
-8
-18-50
-2026
314
7
-52129
6
417
69(a) Image quality
OCR Non-OCR5
051015202530
-1
-60
-325
1631
20(b) Entity recognition
Head T orso T ail10
5
051015
1
-10
-139.0
6.0
0.517
13
615
9
5(c) Entity popularity
Simple-rec.Simple-know.Reasoning Comparison AggregationMulti-hop40
20
0204060
45
-66
-6
-36-536
15
9 9
-4949
1938
26
-61362
3848
40
17 17(d) Question type
True False01020304050
-516
12232536 3649(e) Context-dependent
0.00.51.01.52.02.5
1.8
1.22.12.7(f) Avg Number of Successful Turns
Llama 3.2 90B (LLM only)
Llama 3.2 90B (LLM + image KG)Llama 3.2 90B (LLM + image KG + web)
GPT-5 Mini (LLM only)GPT-5 Mini (LLM + image KG)
GPT-5 Mini (LLM + image KG + web)Figure 7Performance of straightforward MM-RAG solutions across different slices of CRAG-MM. Figure (a)–(e) show
truthfulness in percentage; figure (f) shows the average number of successful turns in multi-turn QA.
D Evaluating industry SOTA solutions
We tested the SOTA MM-RAG systems by sending each question in the public test set as input and collecting
the responses for evaluation. Since these SOTA systems all have embedded retrieval systems, the provided
retrieval results in CRAG-MM werenotused.
We evaluated three systems: GPT-5-2025-08-07, Gemini-2.5-Pro and Claude-Sonnet-4-20250514, and assessed
via the API platforms provided by OpenAI, Google AI and Anthropic, respectively. Tool usage was explicitly
enabled to ensure search functionality was available for our end-to-end assessment of MM-RAG systems.
Each system was called during the following dates in Pacific Time: 09/01/2025 ∼09/09/2025 (GPT-5),
09/01/2025∼09/10/2025 (Gemini), and 09/02/2025∼09/10/2025 (Claude Sonnet).
Below is our parameter setting for tool usage specifically.
gpt_tool_calling = [
{
"type": "web_search_preview"
}
]
gemini_generate_content_config = types.GenerateContentConfig(
thinking_config=types.ThinkingConfig(
thinking_budget=-1,
21

),
tools= [
types.Tool(googleSearch=types.GoogleSearch()),
],
)
claude_tool_calling = [
{
"name": "web_search",
"type": "web_search_20250305"
}
]
We present the prompts used to produce the SOTA MM-RAG evaluation results in Tab.5 and Fig.4 as follows.
Prompt template for single-turn QA
You are a helpful assistant that truthfully answers user questions about the provided image.
Keep your response concise and to the point. Strive to answer in one sentence.
If you are unsure or don’t know the answer, respond with "I don’t know".
Search the internet and provide a short answer to the question: <query>
Image: <image>
Prompt template for multi-turn QA
Conversation history: <conversation_history>
You are a helpful assistant that truthfully answers user questions about the provided image.
Keep your response concise and to the point. Strive to answer in one sentence.
If you are unsure or don’t know the answer, respond with "I don’t know".
Search the internet and provide a short answer to the question: <query>
Image: <image>
22
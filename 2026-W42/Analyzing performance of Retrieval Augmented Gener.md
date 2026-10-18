# Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks

**Authors**: Fazle Mohammed Tasfiq, Sabab Attin, Sheikh Md. Tahmid Hossain, Raiat Reza, Tanzim Ahmmed, Nafis Chowdhury, Anika Ahmed, Moinul Haque, Farig Yousuf Sadeque

**Published**: 2026-10-14 16:14:55

**PDF URL**: [https://arxiv.org/pdf/2610.20419v2](https://arxiv.org/pdf/2610.20419v2)

## Abstract
Retrieval-Augmented Generation (RAG) models have become almost a cornerstone of open-domain question answering tasks; in which the quality of answers generated in the process, greatly relies on the effectiveness of retrieval mechanisms. In the realm of LLM, the last two years have seen tremendous progress in RAG architectures, with many models reporting to have improved performance through various design innovations. The cultural question and answering systems of South Asia, in particular, present a specific challenge since the region is home to the vast number of languages and traditions, which are closely interrelated. Along with these deeply intertwined historical narratives increase the need for culturally grounded and context-aware interpretation. Despite major architectural advances in LLMs, claiming significant improvements, challenges remain when answering these region specific questions in native languages. In this study, we aim to extract high-performing factors to guide the development of optimized hybrid RAG systems. We further explore whether the integration of RAG changes performance differently for culturally grounded and non-cultural questions in a South Asian multilingual setup, and whether RAG produces a higher overall percentage of correct answers than contexted answering. We also examine how LLM performance changes when RAG methods are integrated into the question-answering pipeline in comparison with non-contexted and contexted answering and, evaluate whether the introduction of RAG benefits popular or non-popular culturally grounded questions more. Not only does our work evaluate the current methods but also carves a path for future designs of more accurate and efficient question-answering pipelines.

## Full Text


<!-- PDF content starts -->

Analyzing performance of Retrieval Augmented
Generation models for Question Answering T asks
1stF azle Mohammed T asfiq
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
fazle.mohammed.tasfiq@g.bracu.ac.bd2ndSabab Attin
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
sabab.attin@g.bracu.ac.bd
3rdSheikh Md. T ahmid Hossain
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
sheikh.md.tahmidhossain@g.bracu.ac.bd4thRaiat Reza
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
raiat.reza@g.bracu.ac.bd
5thT anzim Ahmmed
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
tanzim.ahmmed@g.bracu.ac.bd6thNafis Chowdhury
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
ext.nafis.chowdhury@bracu.ac.bd
7thAnika Ahmed
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
anika.ahmed@bracu.ac.bd8thMoinul Haque
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
ext.moinul.haque@bracu.ac.bd
9thDr. F arig Y ousuf Sadeque
Dept. of Computer Science and Engineering
BRAC University
Dhaka, Bangladesh
farig.sadeque@bracu.ac.bd
Abstract —Retrieval-Augmented Generation (RAG)
models have become almost a cornerstone of open-
domain question answering tasks; in which the quality
of answers generated in the process, greatly relies
on the effectiveness of retrieval mechanisms. In the
realm of LLM, the last two years have seen tremen-
dous progress in RAG architectures, with many mod-
els reporting to have improved performance through
various design innovations. The cultural question and
answering systems of South Asia, in particular, present
a specific challenge since the region is home to the vast
number of languages and traditions, which are closely
interrelated. Along with these deeply intertwined his-
torical narratives increase the need for culturally
grounded and context-aware interpretation. Despite
major architectural advances in LLMs, claiming signifi-
cant improvements, challenges remain when answering
these region specific questions in native languages. In
this study , we aim to extract high-performing factors
to guide the development of optimized hybrid RAG
systems. W e further explore whether the integration
of RAG changes performance differently for cultur-ally grounded and non-cultural questions in a South
Asian multilingual setup, and whether RAG produces
a higher overall percentage of correct answers than con-
texted answering. W e also examine how LLM perfor-
mance changes when RAG methods are integrated into
the question-answering pipeline in comparison with
non-contexted and contexted answering and, evaluate
whether the introduction of RAG benefits popular or
non-popular culturally grounded questions more. Not
only does our work evaluate the current methods but
also carves a path for future designs of more accurate
and eﬀicient question-answering pipelines.
Index T erms —Adaptive Retrieval, Benchmarking,
Domain-specific QA, Knowledge Graph, Large Lan-
guage Model, Multimodal, Question Answering, Rel-
evance Estimation, Retrieval-Augmented Generation,
T ransformer.
I. Introduction
The rapid advancements in large language models
(LLMs) have inaugurated a new era of natural language
arXiv:2610.20419v2  [cs.CL]  15 Oct 2026

processing, enabling unprecedented capabilities in com-
plex tasks such as summarization and Question Answer-
ing (QA). These models acquire their powerful, general-
purpose language understanding through a process con-
trolled by scaling laws, involving the training of billions
of parameters on massive textual corpora [ 1]. Promi-
nent LLM families, including GPT, LLaMA, and PaLM,
demonstrate this property by internalizing a vast set of
parametric knowledge within their internal weights. Y et,
the operational utility of these systems for knowledge-
intensive applications is fundamentally constrained by the
static nature of their training data [ 2]. This structural
limitation creates a defined knowledge cutoff, rendering
models temporally bound and incapable of accessing or
processing information created subsequent to their fi-
nal training iteration. More critically , LLMs are system-
ically prone to hallucination, which is the generation
of false, misleading, or unfounded information [ 3]. This
phenomenon is often classified as intrinsic hallucination,
where the output contradicts the immediate input context,
or extrinsic hallucination, which arises when the model
attempts to fill internal knowledge gaps by fabricating
details inconsistent with its training data [ 4]. F or factually
demanding applications like QA, these inherent limita-
tions severely compromise system reliability and diminish
user trust. T o overcome these fundamental barriers and
augment LLMs with access to real-time, verifiable, and
vastly expanded knowledge, the paradigm of Retrieval-
Augmented Generation (RAG) was introduced in 2020
by Lewis et al [ 5]. RAG fundamentally transforms the
LLM architecture by fusing its pre-trained parametric
memory with a dedicated non-parametric memory , an ex-
ternal, updatable knowledge base. This technique enables
the LLM to dynamically retrieve relevant evidence from
a vast corpus, conditioning its generation on verifiable,
external context [ 7]. By explicitly grounding its responses,
RAG effectively expands the model’s factual scope beyond
its original training boundary and significantly mitigates
the risk of extrinsic hallucination. Consequently , RAG
has rapidly become a central focus for research, proving
capable of converting a general LLM into a more accu-
rate, knowledgeable, and reliable system, particularly for
complex QA tasks. The adoption of RAG models offers
substantial benefits over traditional parametric LLMs,
especially by generating language that is more specific, di-
verse, and factual compared to state-of-the-art parametric-
only sequence-to-sequence baselines [ 5]. Moreover, RAG is
a strategic and eﬀicient way of specialization. The ability
to create and leverage a knowledge graph of external
knowledge sources enables RAG to effectively transform
a general LLM into a domain-specific expert [ 8]. This
domain adaptation ability is a great benefit in terms of
resource eﬀiciency over the intensive and expensive fine-
tuning that is needed to add equivalent specialized knowl-
edge as parameters in the model. RAG has spread quickly ,
sparking an unprecedented surge of architectural researchthat has produced a new field of architecture, with a wide
array of methods. Currently , RAG systems are widely cat-
egorized according to the innovations that are made in the
retriever—how documents are selected, the generator—
how context is synthesized, or hybrid approaches that
combine coordinated innovations in both their respec-
tive components [ 9]. This multi-component system, how-
ever, does have inherent complexity and potential areas
for failure. Some of the basic tasks of RAG, including
chunking large bodies of knowledge, embedding them in
vectors, and indexing them eﬀiciently in a database, are
vital [ 8]. Even with the most powerful LLM, if these
initial processes are not correct, the retriever will provide
irrelevant or insuﬀicient context. Thus, the analysis of
performance for tasks in QA should go beyond linguistic
fluency . Rather, evaluation requires specialized, diagnostic
measures for factual consistency and verifiability . They
include Groundedness (or F aithfulness): Does the response
keep to the provided context? Context Relevance: Are
the retrieved documents relevant to the question? Answer
Relevance: How well does the final response answer the
user’s question? [ 10] This will be a systematic assessment
that will help to record progress and pinpoint the most
successful RAG combinations. Now it is time to examine
the performance dynamics of Retrieval-Augmented Gen-
eration models for QA tasks systematically , by looking at
the influence of these design decisions and assessing their
success by applying challenging, RAG-specific evaluations.
This analysis can yield an actionable insight into building
strong, reliable, and successful knowledge-empowered lan-
guage models.
Fig. 1. W orkflow of the RAG model evaluation framework.

II. Related Works
Based on the overall finding, the assessment of
RAG models in Question Answering(QA) needs to be
expanded to beyond retrieval coverage. Rather, the
three dimensions of structural knowledge management,
execution eﬀiciency and strong verification mechanisms
should also be emphasized. The CRAG benchmark is
the benchmark most commonly used as a performance
benchmark, which quantifies the persistent T rust Gap,
indicating that, in complex real-world QA settings,
state-of-the-art RAG systems only achieve approximately
63% truthfulness without hallucination [ 11]. The need for
this has been met by recent advances in the architecture,
which repeatedly show that structured knowledge fares
much better than naive vector matching for complex
queries. T o illustrate, GraphRAG outperformed vector
RAG in comprehensiveness with up to 83% accuracy
when it comes to the global contextual understanding
of unstructured data as Knowledge Graphs (KGs) [ 12].
This structural precision is realized in a useful and
domain-specific manner by DO-RAG, which combines
KG traversal with vector retrieval, resulting in the joint
score gains of up to 33.38% in technical domains [ 13].
The eﬀiciency challenge in dynamic corpora is solved
by EraRAG [ 14], whose LSH-based incremental update
mechanism slashes KG reconstruction time by 77.5%,
validating that eﬀicient structured storage is essential for
modern RAG performance.
The second major pillar is performance optimization
through adaptive and eﬀicient execution. On more
complex multi-hop queries, Collab-RAG [ 15] proposes
a breakthrough in query decomposition performance
by training a small language model (SLM) to break
down queries, showing that a 3B SLM can outperform
a frozen 32B LLM in query decomposition performance.
This highlights the importance of specialised, aligned
orchestration models reducing the cost of running
queries significantly [ 15]. The achievable data and
results of ARAGOG [ 16] indicate that the maximum
benefits of optimisation of certain elements like the use
of Hypothetical Document Embedding (HyDE) and
reranking with LLM provide the highest benefits to
the retrieval accuracy . This therefore confirms that the
RAG performance is extremely sensitive to the specific
combination and optimisation of high-impact elements.
III. Dataset and Framework
A. Wikipedia webscraping
The dataset construction process began with the selec-
tion of a subset of South Asian languages. Our choices
included our native Bangla, along with Hindi, Urdu,
T amil and Nepali, covering the wider range of South
Asia. However, since the languages we are focusing on
are low-resource languages so there are no appropiatedatasets for our experiments since most of the low-resource
datasets are machine translated [ 17]. Upon selecting the
languages, our task was to find appropriate datasets for
these languages. The main motivation for constructing
this multilingual dataset is inspired from the prior work
by [ 18]. T o get the dataset for those languages, we
needed accurate regional data. So we webscraped from
the regional wikipedia pages including: bn.wikipedia.org ,
ur.wikipedia.org ,hi.wikipedia.org ,ta.wikipedia.org and fi-
nally ne.wikipedia.org for native articles on Bangla, Urdu,
Hindi, T amil and Nepali respectively . Our approach began
by selecting the Wikipedia article of the country or state
with the highest number of native speakers for a given
language. This article served as the level-0 (landing) page.
F rom this page, we followed in-page hyperlinks that redi-
rected to separate Wikipedia articles, which were treated
as level-1 terms. The process was then repeated for links
found within level-1 pages to obtain level-2 terms. In total,
this resulted in a three-level hierarchy of terms, comprising
the landing page and two successive levels of hyperlink-
based expansions. The levels were noted and to identify
the common redirects, for each term a parent list was
kept so that duplicate redirects can be avoided. In order to
facilitate our tracking of the terms, all the corresponding
URLs redirecting to the term were kept as well.
After collecting the titles and their corresponding article
links, we observed a large number of Wikipedia page
edit links as well as duplicate titles. Links containing
edit&redlinks=1 were filtered out along with duplicate
entries, and were removed. Additionally , some links redi-
rected to Wikipedia calendar date pages, which were also
discarded. F urthermore, stopwords were eliminated from
our datasets with the help of publicly available stopword
lists in kaggle and/or hugging face.
In this way , for all the languages, the amount of data
present in each dataset before and after preprocessing:
T ABLE I
Dataset size before and after preprocessing
Language Direct
webscrapingAfter
preprocessing
Bangla 17,794 10,769
Hindi 16,362 9,541
T amil 13,258 11,310
Urdu 8,795 7,805
Nepali 2,641 2,574
B. Context generation
Context was collected from Wikipedia for question gen-
eration. Our approach was that we used the links of
the wikipedia pages from the database and scraped the
first four lines from body paragraph content as context.
However, in some cases, some pages did not contain a total
of 4 lines, rather less. Some pages, specifically known as
redlinks, had no content or included just a list of items and
no body paragraph content. Contexts for such pages came

up empty or zero. Along with this, we used Wikipedia
Pageviewer API to list down the page view counts of each
title in the time period from start to end of 2025 and
normalized it between scores of 0 and 1 (otherwise the gap
between counts are too sparse). Finally , the contents were
taken and a counter variable was kept to note the number
of context lines available. The generated files now had
the title, url, context and number of lines and normalized
popularity count as present in the sample table below :
C. Question dataset construction
The context we extracted was then used for question
generation. Qwen 3 32b has been used to generate the
question. W e designed the prompts in such a way that
the model would design the question from the context
given the answer of the question should be the term
we collected and the term should not be present in the
question. T wo lines of context were used to generate the
questions. The questions were generated by giving the
same prompt translated to the other regional languages
to keep consistency . The prompts for all the respective
languages are translated into respective languages from
this prompt- “F rom the given context ’{context}’, write
exactly one name of language question whose answer
would be only ’{title}’ . The answer itself, or any part of
it, must not be used in the question. There is no need
to write anything other than the complete question that
is generated. ” During the experimental phase, a problem
arose that all the terms did not have 4 line contexts. But
in most cases around 2 lines of context were available.
W e used 2 lines as context to retain maximum dataset
and discarded those titles with only 1 line of context.
Another case was observed that impacted this decision
was the tendency of model hallucination when feeding too
little (1 line) context or too much (4 line) context. 1 line
context questions were almost too vague and 4 line context
provided additional information for the model.
D. Assessment of Question clarity
A test sample of 250 questions on both 1-line and 2-
line were analysed to determine how many lines of context
provided better question. W e used the following factors to
evaluate:
Relevance: The question correctly targets the core
ideas present in the given answers.
F actual Accuracy: The question does not introduce
incorrect assumptions or false premises from the answers.
Question Clarity: The question is clearly and unam-
biguously phrased based on the answers.
Creativity: The question frames the existing answers
in a novel, insightful, or non-trivial way .
Coherence: The question is logically structured and
internally consistent with the information in the answers.
T ask-Specific Appropriateness: title not repeated in
question, question has only the complete title as answer,no other answer is possible except the title and question
is in expected language or not.
All the ratings were done by manual human input out
of 5, and then an average was calculated. Upon the evalu-
ation, it was found out that in case of all languages, better
questions were observed for 2 line contexts. In case of
other languages, the respective questions were translated
to Bangla first via google translator model call and then
evaluated. Since these questions were all judged on the
basis of translation, we could not judge the coherence. So
no coherence evaluation was done and the values might
vary if judged by a native speaker. It was found that the
performance is better when there are 2 lines of context
used while question generation. After the confirmation of
using 2 line contexts, we proceeded towards full set ques-
tion generation and answer generation. All the questions
were generated by the same prompt as mentioned before.
F or answer generation, we used four models, 3 open and 1
closed source for diversity . Llama 4 Maverick 17B by Meta
(open source), DeepSeek V3-0324 (open source), Gemma
3 27B by Google (open source) and Grok 4.1 F ast by xAI
(closed source).
E. LLM-Based Semantic Classification
W e followed an LLM-assisted semantic classification
method. The categorization scheme from Chowdhury et
al. ’s [ 18] Bengali cultural knowledge evaluation provided
a peer-reviewed taxonomy for organizing titles into mean-
ingful semantic groups that worked as a reference structure
for this stage. Each title was treated as an individual
instance and passed to Meta Llama Maverick with a well-
constrained instruction to return only one label from the
predefined set of labels. This LLM was selected upon
evaluating its stronger performance in question-answering
tasks in our prior analysis. The model outputs were then
stored as semantic labels for each title.
The following definitions were provided in the model
instructions when passing the titles:

T able II
Category definitions, classes, and examples
Category Class Definition Example
Event Cultural Specific occurrences such as festivals, battles, conferences, natural
disasters, ceremonies, or incidents.পেহলাৈবশাখ
[Pohela Boishakh]
Religion Cultural Religious beliefs, practices, deities, scriptures, rituals, or religious
movements.েবৗদ্ধধমর্
[Buddhism]
F ood Cultural Dishes, ingredients, cuisines, beverages, or culinary items.রসেগাল্লা
[Rasgulla]
Entertainment Cultural Movies, music, television, games, arts, celebrities, or media-related
content.েলাকসঙ্গীত
[F olk music]
Historical Cultural Historical eras, dynasties, treaties, or historically significant topics not
classified as specific events.গুপ্তসাম্রাজঁ
[Gupta Empire]
People Cultural Individual persons, real or fictional, including public figures, scholars,
artists, and athletes.কাজীনজরুলইসলাম
[Kazi Nazrul Islam]
Politics Non-cultural Political parties, ideologies, political oﬀices, public policies,
governance, or political movements.বাংলােদেশরসংিবধান
[Constitution of Bangladesh]
Locations Non-cultural Countries, cities, regions, landmarks, geographical entities, or physical
places.বাংলােদশ
[Bangladesh]
Language Non-cultural Languages, dialects, scripts, linguistic concepts, or grammar-related
topics.বাংলাভাষা
[Bangla Language]
Material Non-cultural Physical substances, chemicals, elements, minerals, fabrics, or tangible
materials.মসিলন
[Muslin]
Organization Non-cultural Companies, institutions, NGOs, agencies, universities, clubs, or
structured groups.ৈদিনকপ্রথমআেলা
[The Daily Prothom Alo]
Health Non-cultural Diseases, medical conditions, treatments, anatomy , healthcare
systems, or psychological conditions.েডঙ্গুজ্বর
[Dengue F ever]
T emporal Non-cultural Time-related concepts such as months, years, calendars, historical
periods when referring to time itself, or dates.ৈচত্র
[Chaitra]
Economics Non-cultural Financial systems, markets, economic theories, currencies, trade, or
economic indicators.বাংলােদেশরঅথর্নীিত
[Economy of Bangladesh]
Sports Non-cultural Sports types, tournaments, teams, competitions, or athletic
disciplines.িক্রেকট
[Cricket]
Miscellaneous Non-cultural Titles that do not clearly belong to any of the above categories.িরকশা
[Rickshaw]
T ABLE III
Amount of data present in the final dataset
Language Final data count
Bangla 10,278
Hindi 8,694
T amil 8,553
Urdu 5,806
Nepali 2,405
F. Dataset Analysis
After the classification was done, the full dataset was
formulated and, since we pivoted to 2 line context, the
dataset size reduced from the previous ones of T able I to:
Our final dataset hence had 4 columns: Title, context
2 line, question 2 line and cluster i.e. the categories. The
dataset is available at [ 6].

T ABLE IV
Sample of multilingual QA dataset
Title Context Question Cluster
ৈদিনকইেত্তফাক
The Daily Ittefaqৈদিনকইেত্তফাকবাংলােদেশরবাংলাভাষারএকিটজাতীয়ৈদিনকসংবাদপত্র।
এিটবাংলাভাষারিপ্রন্টসংস্করণছাড়াওইংেরিজঅনলাইনসংস্করেণপ্রকািশত
হয়।
The Daily Ittefaq is a national daily newspaper pub-
lished in the Bengali language in Bangladesh. It also
has an English online edition.বাংলােদেশরবাংলাভাষারএকিটজাতীয়ৈদিনকসংবাদপত্রকী?
What is a national daily newspaper in the Bengali
language of Bangladesh?organization
मुज़फ़्फ़रनगर
Muzaffarnagarमुज़फ़्फ़रनगर भारत क े उत्तर प्रदेश राज्य क े मुज़फ़्फ़रनगर िज़ले में िस्थत एक नगर
है। मुज़फ़्फ़रनगर यह िज़ले का मुख्यालय भी है।
Muzaffarnagar is a city in Uttar Pradesh, India, and
the headquarters of Muzaffarnagar district.उत्तर प्रदेश क े िकस िजले का मुख्यालय उसी नाम क े शहर में िस्थत है?
Which district of Uttar Pradesh has its headquarters
in a city with the same name?locations
ﺮﻯĽﻤٮģﻛﺴęﺎﮞĻٮęﺭ
Kashmiri languageﺮﻯĽﻤٮģﻛﺴęﺎﮞĻٮęﺭĞﮭﺎﺭٮĻﺍﻭﺭٮ ęﺎﮞĞﺎﻛﺴٮłﮏﻛﻰٮ Ľﺍﮨﻢﺍٮ ęﺎﮞĻٮęﮨﮯ۔ﺭ ęﮟĽﻜﻠﻤٮĞﻛﻰﻣٮ 
ﻌﺪﺍﺩĞٮ 15 ﮫﺳﮯلاﻛﮫ łﺮﻛﺤ łﮨﮯ۔ﺍﻭٮ 
The Kashmiri language is an important language of
India and Pakistan, with slightly over 1.5 million
speakers.ĞﮭﺎﺭٮĻﺍﻭﺭٮ ęﺎﮞĞﺎﻛﺴٮłﮟٮ Ľﻮﻟﻰﻣٮ Ļﮯٮ ęﺎٮĻﺍﺱﻭﺍﻟﻰﺣ ęﺎﮞĻٮęﺎﻡﻛﺎﺭ ęﺎٮ Ľﺲﮨﮯﻛٮ Ļﻛﮯﺣ 
ﺎًĻٮĽﺮٮĞڡĞٮ 15 ﺍٮٔﺪﺳﮯلاﻛﮫ ęﮯﺭ ęﻮﻟٮĻﮟ؟ﻭﺍﻟﮯٮ Ľﮨٮ
What language is spoken in India and Pakistan by
over 1.5 million people?language
பல ் லாங ் குழி
Pallankuzhiபல ் லாங ் குழி என ் பது, ெபாதுவாக ெபண ் கள ்
விைளயாடும ் ஒரு விைளயாட ் டு. முதற ்
பூப ் பைடந ் த ெபண ் ணின ் தீட ் டுக ் குரிய
காலத ் திலும ் கருவுற ் ற ெபண ் கள ் அமர ் ந ் து
ெபாழுதுேபாக ் குவதற ் கும ் இந ் த விைளயாட ் ைட
இப ் ெபாழுது ஆடிப ் பார ் க ் கிறார ் கள ் .
Pallanguzhi is a traditional game generally played by
women.ெபண ் கள ் விைளயாடும ் பழைமவாய ் ந ் த
விைளயாட ் டு எது?
Which traditional game is played by women?entertainment
िहमालय
Himalayasिहमालय एिसया को एक िहमशृङ्खला हो, जसले भारतीय उपमहाद्वीप को मैदानलाइ र्
ितब्बती पठार बाट अलग गछ र्। यस िहमशृङ्खलामा पृथ्वी का क े ही अग्लो चुचुराहरू
छन्, जसमा सबैभन्दा अग्लो, सगरमाथा पिन समावेश छ।
The Himalayas separate the Indian subcontinent from
the Tibetan Plateau and include Mount Everest.भारतीय उपमहाद्वीपको मैदानलाइ र्ितब्बती पठारबाट अलग पानेर् एवं सगरमाथाले
समेिटएको एिसयाको एउटा िहमशृङ्खला क े हो?
What mountain range separates the Indian subconti-
nent from the Tibetan Plateau and includes Mount
Everest?locations
IV. RAG Setup and Analysis
The next part of our work adopted Retrieval-Augmented
Generation as the main framework for context-supported
answer generation. W e integrated RAG to our pipeline
because it combines a parametric LLM with a non-
parametric retrievable memory base, thus allowing gen-
eration to be constrained on relevant retrieved contexts or
ground truths instead of only relying on the model’s base
knowledge [ 5] . In our setup, the retrieved context served
as the external knowledge base or ground truth, while
the LLM generated the final answer from the query and
retrieved passages. This phase began with selecting three
frameworks for implementation of the pilot evaluation of
RAG-based answer generation- LangChain, LangGraph,
and LlamaIndex. Each of these frameworks provides a dif-
ferent approach towards organizing retrieval-based gener-
ation. LangChain was included because it supports chain-
based composition of LLM workflows, where retrieval,
prompting, model invocation, and output parsing can
be connected sequentially [ 25] . In a slightly different
approach, LangGraph represents applications through a
shared state of nodes and edges to update the state and
define next execution steps [ 26] . LlamaIndex provides
retrieval-augmented LLM applications with data indices
and query engines, where each engine provides natural-
language access to indexed data [ 27] . A 100-sample pilot
test was carried out across the three frameworks under
similar retrieval and generation settings for each languageusing each LLM. The results showed weaker performance
of LangChain based pipeline compared to the other two,
specifically for lower performing language settings such as
Nepali, T amil, and Urdu. As a result, further work was
carried out using LangGraph and LlamaIndex as RAG
frameworks.
T ABLE V
Performance comparison across languages and frameworks
Lang. LLM LC LG LI
BanglaDeepSeek V3 64.0 64.0 57.0
Gemma 3 67.0 66.0 70.0
LLaMA 4 66.0 66.0 68.0
Grok 4.1 61.0 65.0 60.0
HindiDeepSeek V3 37.0 38.0 34.0
Gemma 3 46.0 46.0 44.0
LLaMA 4 63.0 63.0 64.0
Grok 4.1 56.0 60.0 58.0
NepaliDeepSeek V3 51.0 53.0 55.0
Gemma 3 53.0 55.0 53.0
LLaMA 4 64.0 74.0 76.0
Grok 4.1 60.0 67.0 61.0
T amilDeepSeek V3 40.0 43.0 41.0
Gemma 3 49.0 46.0 51.0
LLaMA 4 63.0 66.0 68.0
Grok 4.1 44.0 41.0 44.0
UrduDeepSeek V3 38.0 39.0 42.0
Gemma 3 11.0 10.0 15.0
LLaMA 4 57.0 68.0 70.0
Grok 4.1 61.0 63.0 63.0
LC = LangChain, LG = LangGraph, LI = LlamaIndex. V alues are
percentages.
Throughout the rest of our work, LangGraph and Lla-
maIndex organized the same RAG task following dif-

ferent pipeline structures. LangGraph pipeline modeled
the process as a state-based workflow to store retrieved
contexts and generated answers as a part of a shared
RAG state. The retrieval and generation were executed
as separate nodes that were dependent directly on the
context returned by the retriever. On the other hand,
LlamaIndex connected the documents to a Chroma vector
space and queried through a query-engine following an
index-based approach. In both cases, the pipelines followed
similar retrieval to generation logic, but the implementa-
tion differed in how context was retrieved, passed and used
during answer generation processes.
V. RAG Pipeline: ChromaDB, Embeddings, and
Retriev al
Our RAG pipeline used ChromaDB as the vector store
since the retrieval stage required semantic similarity search
rather than conventional keyword lookup. ChromaDBstores document chunks as embeddings along with their
corresponding documents and metadata making it suitable
for RAG-based AI workflows where query embeddings
must be matched with semantically relevant stored pas-
sages [ 28]. In our pipeline, the query was embedded at
runtime and compared against the stored chunk embed-
dings so that the most relevant contexts could be retrieved
and passed as ground truth for answer generation. The
knowledge base was prepared from the rag_database.csv
file, where each entry contained a title and a short two-
line contextual description. Each row was converted into
a document by combining the title with its corresponding
context before chunking and embedding. A sample of the
dataset structure is shown below:
T able VI
Sample Bangla Titles with Contextual Descriptions
Title Context Language
বাংলােদশ
Bangladeshবাংলােদশদিক্ষণএিশয়ারএকিটস্বাধীনসাব র্েভৗমরাষ্ট্র।বাংলােদেশরসাংিবধািনকনামগণপ্রজাতন্ত্রীবাংলােদশ।
Bangladesh is an independent sovereign state in South Asia. The constitutional name of
Bangladesh is the People’s Republic of Bangladesh.bn
ওসমানী আন্তজর্ািতক
িবমানবন্দর
Osmani International
Airportওসমানীআন্তজর্ািতকিবমানবন্দরবাংলােদেশরঅনঁতমপ্রধানআন্তজ র্ািতকিবমানবন্দর।এিটিসেলেটঅবিস্থত।
Osmani International Airport is one of the major international airports in Bangladesh. It
is located in Sylhet.bn
ৈদিনকপ্রথমআেলা
The Daily Prothom
Aloৈদিনকপ্রথমআেলাবাংলােদশেথেকবাংলাভাষায়প্রকািশতএকিটৈদিনকসংবাদপত্র।১৯৯৮সােলর৪নেভম্বরএিট
প্রথমপ্রকািশতহয়।
Prothom Alo is a daily newspaper published in Bengali from Bangladesh. It was first
published on 4 November 1998.bn
The documents were further split into smaller chunks
before embedding. F or embedding generation, we used
intfloat/multilingual-e5-large-instruct, a multilingual em-
bedding model designed for text retrieval tasks and re-
leased as a part of the Multilingual-E5 embedding family
[29] . Separate ChromaDB directories were maintained
for each framework implementation in order to maintain
framework-specific indexing behavior independent of each
other aligning with reproducible research practices [ 30].
VI. Result Collection
A. Initial Answer Generation
Answer generation was done by api calls and the same
prompt for generating answers were translated to their
respective languages. In order to evaluate the baseline
knowledge of the models in each language, and their
ability to utilize contextual information and also evaluate
the change in performance, we generate answers in two
stages. At first, answers were generated without providing
context to the models (non-contexted answer) and then byproviding context (contexted answers). Similarly answer
generation was done by api calls and the same prompt
for generating answers were translated to their respective
languages. In order to evaluate the baseline knowledge of
the models in each language, and their ability to utilize
contextual information and also evaluate the change in
performance, we generate answers in two stages. At first,
answers were generated without providing context to the
models (non-contexted answer) and then by providing
context (contexted answers). The prompts used in this
case was in english as followed for Non-contexted answer
generation. “According to the question ‘{question}’, pro-
vide only the direct correct answer to the question. T ry to
limit the answer to one word, and under no circumstances
should you use more than five words. Do not add any
explanation, full sentence, punctuation, extra words, in-
troduction, or conclusion—only the final answer should be
given in the output. ” and for contexted answer generation
the prompt was- “According to the question ‘{question}’
and the given context ‘{context}’, provide only the direct

correct answer to the question. T ry to limit the answer to
one word, and under no circumstances should you use more
than five words. Do not add any explanation, full sentence,
punctuation, extra words, introduction, or conclusion—
only the final answer should be given in the output. ” these
two prompts were translated into respective languages.
B. RAG-Based Answer Col lection
RAG-based answer collection was conducted after the
final selection of LangGraph and LlamaIndex. F or each
language, model, and framework combination, the gener-
ated questions were processed through the corresponding
RAG pipeline, and the model outputs were saved for
evaluation. The same answer-generation instruction was
used across all settings, with language-specific translations
to maintain consistency .
During the answer generation stage, the input question
was embedded and matched against the corresponding
ChromaDB vector store, and the top retrieved chunks
were passed to the LLM along with the question. The
prompt was designed to produce short title-like answers
rather than explanatory responses. Since the questions
were constructed so that the expected answer was the cor-
responding title, each model was instructed to return only
the final answer. The collected outputs were stored with
their language, model, framework, question, retrieved con-
text, generated answer, and reference title. The following
prompt was used in respective languages to generate RAG-
Based Answer for each language- ”Y ou are a {Language}
cultural question-answering assistant. Answer the question
from the context below. Give the answer in {Language}.
Limit the answer to one word, and under no circumstances
should you use more than five words; there is no need to
write anything extra or a complete sentence. If the answer
is not found in the context, say: “Information not found. ”
Context: {context} Question: {question}. ”
VII. Model
A. Question Generation
Qwen 3 32B by Alibaba is used exclusively for question
generation. With 32 billion parameters and support for
over 100 languages, it operates in two modes: a structured
”thinking” mode for carefully constructed prompts and a
faster ”non-thinking” mode for fluent question variations.
Its 131,072-token context window enables diverse, context-
aware question creation across topics and languages.
B. Answer Generation
F our models handle answer generation, each bringing
distinct strengths. Meta Llama 4 Maverick 17B is a mul-
timodal open-source model with 400 billion parameters
and 128 experts, handling both text and image inputs
with a 128K context window, excelling in natural language
processing and long-context applications. DeepSeek Chat
V3-0324 is a massive 685-billion-parameter mixture-of-
experts model with a 163,840-token context window, builtfor high-performance natural language understanding and
complex dialogues. Google Gemma 3 27B supports vision-
language input and text output across 140+ languages,
performing strongly in math, reasoning, and structured
outputs with a 128K context window and function calling
capabilities. xAI Grok 4.1 F ast is a closed-source agentic
model featuring a remarkable 2-million-token context win-
dow, optimized for customer support and deep research
with an optional reasoning mode for enhanced decision-
making.
C. T ranslation
OpenAI GPT-OSS 120B is an open-weight, 117-billion-
parameter mixture-of-experts model used solely for trans-
lation across five languages. With a 131K context window,
native MXFP4 quantization for eﬀicient inference, and
support for function calling and structured outputs, it
brings high-reasoning capacity to multilingual translation
workflows.
VIII. Result and Analysis
A. Result Analysis
Fig. 2. Overall accuracy comparison across non-contexted, con-
texted, and best RAG settings for five languages. Model groups
within each language represent Meta Llama 4, Google Gemma 3,
DeepSeek V3-0324, and Grok 4.1 F ast, respectively .
The overall accuracy results show a clear gap between
non-contexted and context-supported answer generation.
As shown in Fig. 2, both contexted answering and best
RAG perform substantially better than non-contexted
answering across all five languages. Meta Llama 4 Mav-
erick 17B generally achieves the strongest performance
across the evaluated settings, while DeepSeek V3-0324 and
Grok 4.1 F ast show weaker performance depending on
the language and answering condition. However, several
language- and category-specific anomalies are observed.
In Nepali, Google Gemma 3 27B performs weakest in

the general non-contexted settings, while DeepSeek V3-
0324 produces the weakest performance for overall non-
popular titles and non-cultural non-popular titles un-
der non-contexted answering. Another exception appears
in culturally grounded non-popular contexted answering,
where Grok 4.1 F ast performs better than Meta Llama 4
Maverick. Similar deviations are also visible in Hindi and
Urdu. In Hindi cultural non-contexted answering, Google
Gemma 3 27B performs weakest instead of Grok. In Urdu,
Gemma performs weakest for popular culturally grounded
non-contexted answering and popular non-cultural con-
texted answering.
Fig. 3. Accuracy comparison across cultural and non-cultural ques-
tion categories under contexted and best RAG settings for five
languages. Model groups within each language represent Meta Llama
4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 F ast, respec-
tively .
The cultural subdivision results in Fig. 3show that
performance does not vary uniformly between cultural
and non-cultural questions. In several languages, cultural
questions achieve comparable or higher accuracy than
non-cultural questions, while in other cases the opposite
pattern is observed. This indicates that cultural relevance
alone does not determine answer-generation accuracy . In-
stead, the results vary depending on the language, model,
and type of context used. The figure also shows that
RAG does not consistently improve both cultural and non-
cultural categories for every model, which supports the
need for language-wise and model-wise analysis.
Fig. 4. Accuracy comparison for popular titles across cultural and
non-cultural question categories under contexted and best RAG
settings. Model groups within each language represent Meta Llama
4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 F ast, respec-
tively .
Fig. 5. Accuracy comparison for non-popular titles across cultural
and non-cultural question categories under contexted and best RAG
settings. Model groups within each language represent Meta Llama
4, Google Gemma 3, DeepSeek V3-0324, and Grok 4.1 F ast, respec-
tively .
The popular and non-popular title comparisons in Fig. 4
and Fig. 5show that popular titles generally achieve higher
accuracy than non-popular titles. This trend is visible
across most languages and categories, suggesting that title
popularity affects answer correctness. However, the figures
also show some model-level variations. F or example, in
some non-popular settings, the gap between contexted
answering and best RAG becomes larger, indicating that
retrieval support is more useful when the baseline perfor-
mance is weaker. F or RAG-based answering, the overall
model ranking remains similar to the preliminary answer-
generation results. Meta Llama 4 Maverick 17B remains
the strongest model in most settings, while the weaker
model varies across languages and categories. The com-
parison between LangGraph and LlamaIndex shows only
marginal differences, with LlamaIndex performing slightly
better in some cases. Therefore, the main performance
differences are more strongly associated with language,
model choice, cultural subdivision, and title popularity
than with the specific RAG framework.

IX. Discussion
The first research question examines whether RAG
affects culturally grounded and non-cultural questions
differently in the multilingual setup. As shown in Figure 6,
the effect of RAG on culturally grounded questions varies
across languages and models. Bangla and T amil show the
clearest RAG advantage, with most models improving over
contexted answering. In contrast, Hindi and Nepali are
more context-favored, while Urdu shows a mixed pattern.
This indicates that culturally grounded questions do not
automatically benefit from retrieval. Instead, the improve-
ment depends on the language, model, and relevance of the
retrieved evidence.
Fig. 6. Percentage-point improvement of best RAG performance
over contexted answering for culturally grounded questions across
five languages and four models.
Fig. 7. Percentage-point improvement of best RAG performance over
contexted answering for non-cultural questions across five languages
and four models.
Figure 7shows that the non-cultural category follows a
related but not identical pattern. Bangla shows the most
consistent RAG improvement, while Urdu performs better
with RAG for most models. However, Hindi and Nepali
remain largely context-favored, and T amil becomes less
consistent compared to its cultural-question performance.
Therefore, the findings for the first research question sug-
gest that RAG is not simply category-sensitive. Its effect is
both language-sensitive and model-sensitive, with stronger
cultural-question benefits mainly observed in Bangla and
T amil.
Fig. 8. Overall accuracy comparison between contexted answering
and best RAG performance across five languages.
The second research question evaluates whether RAG
achieves higher overall accuracy than contexted answer-
ing. Figure 8shows that RAG produces higher average
accuracy than contexted answering in Bangla, Urdu, and
T amil. However, contexted answering remains stronger in
Hindi and Nepali. This means that RAG cannot be treated
as an automatically superior method for multilingual QA.
Retrieval improves performance when the retrieved pas-
sages are useful and well aligned with the question, but it
may underperform when the retrieved context is weaker
than the directly supplied context.
Fig. 9. A verage accuracy across five languages for non-contexted
answering, contexted answering, LangGraph RAG, and LlamaIndex
RAG.
The third research question compares the four an-
swering settings: non-contexted, contexted, LangGraph
RAG, and LlamaIndex RAG. As shown in Figure 9, non-
contexted answering produces the lowest average accuracy
across all five languages. This shows that the models’
internal parametric knowledge alone is not suﬀicient for
reliable answer generation in this South Asian multilingual
QA task. Both directly supplied context and retrieval-
supported context improve performance substantially . Al-
though prior work has shown that long or excessive context
can reduce LLM performance, our results show that, in
this setup, the absence of external information is the larger
limitation.

Fig. 10. RAG gain over non-contexted answering for popular and
non-popular culturally grounded questions across five languages.
The fourth research question examines whether RAG
benefits popular or non-popular culturally grounded ques-
tions more. Figure 10 shows that non-popular culturally
grounded questions receive higher RAG gain than pop-
ular culturally grounded questions across the evaluated
languages. This does not mean that non-popular questions
always achieve the highest final accuracy . Rather, it shows
that retrieval provides greater relative support when the
model starts from a weaker non-contexted baseline. Pop-
ular cultural questions appear to be better represented in
the models’ internal knowledge, so the additional improve-
ment from retrieval is smaller. Overall, the findings sug-
gest that RAG should be treated as a context-dependent
enhancement rather than a universally superior answering
strategy . It reduces knowledge gaps, especially for weaker
baseline settings and non-popular cultural questions, but
it does not consistently outperform direct contexted an-
swering across all languages. The effectiveness of RAG
in multilingual cultural QA depends on retrieval quality ,
language-specific data alignment, and the ability of each
model to use retrieved evidence correctly .
X. Conclusion
This study found that LLMs perform better in mul-
tilingual cultural knowledge evaluation when supported
by retrieved contextual information than being asked to
answer from its parametric knowledge base. Earlier ex-
periments have shown that modern LLMs often struggle
with culturally nuanced questions when no prior contex-
tual knowledge is provided, especially for less popular or
regionally specific languages. LLM-assisted classification
helped organize the titles into structured categories, while
the later RAG experiments showed that retrieved context
could improve answer generation across the multilingual
dataset. Overall, the study suggests that multilingual
cultural evaluation benefits from combining structured
semantic labeling with retrieval-supported generation, es-
pecially for short, context-dependent titles and culturally
specific questions.Acknowledgment
W e express our gratitude and gratefulness to our su-
pervisor, Dr. F arig Y ousuf Sadeque for his guidance and
support throughout our thesis. W e are also grateful to our
families and friends who have provided constant encour-
agement and support.
References
[1] S. Minaee et al. , “Large language models: A survey ,” arXiv
preprint arXiv:2402.06196 , 2024. [Online]. A vailable: https://
arxiv.org/abs/2402.06196
[2] A. Brown, M. Roman, and B. Devereux, “A systematic litera-
ture review of retrieval-augmented generation: T echniques, met-
rics, and challenges,” arXiv preprint arXiv:2508.06401 , 2025.
[Online]. A vailable: https://arxiv.org/abs/2508.06401v1
[3] S. Amirshahi, A. Bigdeli, C. L. A. Clarke, and A. Ghenai,
“Evaluating the robustness of retrieval-augmented generation
to adversarial evidence in the health domain,” arXiv preprint
arXiv:2509.03787 , 2025. [Online]. A vailable: https://arxiv.org/
abs/2509.03787
[4] Y. Bang et al. , “HalluLens: LLM hallucination benchmark,”
2025.
[5] P . Lewis et al. , “Retrieval-augmented generation for knowledge-
intensive NLP tasks,” arXiv preprint arXiv:2005.11401 , 2020.
[Online]. A vailable: https://arxiv.org/abs/2005.11401
[6] F. M. T asfiq, S. Attin, S. M. T. Hossain, R. Reza, T. Ah-
mmed, N. Chowdhury , A. Ahmed, M. Haque, and F. Y. Sad-
eque, “SACMQ: South Asian Cultural Multilingual Question
Dataset,” Zenodo, version 2.0, Oct. 2026. doi: 10.5281/zen-
odo.23244195 .
[7] M. Cheng et al. , “A survey on knowledge-oriented retrieval-
augmented generation,” arXiv preprint arXiv:2503.10677 , 2025.
[Online]. A vailable: https://arxiv.org/abs/2503.10677
[8] S. W u et al. , “Retrieval-augmented generation for large language
models: A survey ,” 2024.
[9] A. Sharma et al. , “A survey of retrieval-augmented generation
systems,” 2025.
[10] S. Roychowdhury et al. , “Metrics for evaluating retrieval-
augmented generation,” 2024.
[11] X. Y ang et al. , “CRAG: Comprehensive RAG benchmark,”
Advances in Neural Information Processing Systems , 2024.
[12] D. Edge et al. , “F rom local to global: A graph RAG
approach to query-focused summarization,” arXiv preprint
arXiv:2404.16130 , 2024.
[13] D. O. Opoku, M. Sheng, and Y. Zhang, “DO-RAG: A
domain-specific QA framework using knowledge graph-
enhanced retrieval-augmented generation,” arXiv preprint
arXiv:2505.17058 , 2025.
[14] F. Zhang et al. , “EraRAG: Eﬀicient and incremental retrieval
augmented generation for growing corpora,” arXiv preprint
arXiv:2506.20963 , 2025.
[15] R. Xu et al. , “Collab-RAG: Boosting retrieval-augmented gener-
ation for complex question answering via white-box and black-
box LLM collaboration,” arXiv preprint arXiv:2504.04915 ,
2025.
[16] M. Eibich, S. Nagpal, and A. F red-Ojala, “ARAGOG: Advanced
RAG output grading,” arXiv preprint arXiv:2404.01037 , 2024.
[17] T. Mahfuz et al. , “T oo late to train, too early to use? A study
on necessity and viability of low-resource Bengali LLMs,” 2025.
[18] N. Chowdhury et al. , “F rom facts to folklore: Evaluating large
language models on Bengali cultural knowledge,” 2025.
[19] A. Y ang et al. , “Qwen3 technical report,” 2025. [Online]. A vail-
able: https://arxiv.org/abs/2505.09388
[20] Hugging F ace, “meta-llama/Llama-4-Maverick-17B-128E-
Instruct,” 2025, model card accessed F eb. 5, 2026.
[21] xAI, “Grok 4.1,” 2025. [Online]. A vailable: https://x.ai/news/
grok-4-1 . Accessed: F eb. 5, 2026.
[22] Gemma T eam et al. , “Gemma: Open models based on Gemini
research and technology ,” 2024. [Online]. A vailable: https://
arxiv.org/abs/2403.08295

[23] DeepSeek-AI et al. , “DeepSeek-V3 technical report,” 2025. [On-
line]. A vailable: https://arxiv.org/abs/2412.19437
[24] OpenAI et al. , “gpt-oss-120b & gpt-oss-20b model card,” 2025.
[Online]. A vailable: https://arxiv.org/abs/2508.10925
[25] H. Chase, “LangChain,” 2022. [Online]. A vailable: https://
github.com/langchain-ai/langchain
[26] LangChain AI, “LangGraph: Building stateful, multi-actor
applications with LLMs,” 2024. [Online]. A vailable: https://
github.com/langchain-ai/langgraph
[27] J. Liu, “LlamaIndex,” 2022. [Online]. A vailable: https://github.
com/run-llama/llama_index
[28] Chroma, “Chroma: The AI-native open-source embedding
database,” 2025. [Online]. A vailable: https://github.com/
chroma-core/chroma
[29] L. W ang et al. , “Multilingual E5 text embeddings: A technical
report,” arXiv preprint arXiv:2402.05672 , 2024. [Online]. A vail-
able: https://arxiv.org/abs/2402.05672
[30] G. K. Sandve, A. Nekrutenko, J. T aylor, and E. Hovig, “T en
simple rules for reproducible computational research,” PLoS
Computational Biology , vol. 9, no. 10, 2013.
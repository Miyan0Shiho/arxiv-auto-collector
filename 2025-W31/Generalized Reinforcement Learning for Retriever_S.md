# Generalized Reinforcement Learning for Retriever-Specific Query Rewriter with Unstructured Real-World Documents

**Authors**: Sungguk Cha, DongWook Kim, Taeseung Hahn, Mintae Kim, Youngsub Han, Byoung-Ki Jeon

**Published**: 2025-07-31 04:55:21

**PDF URL**: [http://arxiv.org/pdf/2507.23242v1](http://arxiv.org/pdf/2507.23242v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems rely heavily on effective query
formulation to unlock external knowledge, yet optimizing queries for diverse,
unstructured real-world documents remains a challenge. We introduce
\textbf{RL-QR}, a reinforcement learning framework for retriever-specific query
rewriting that eliminates the need for human-annotated datasets and extends
applicability to both text-only and multi-modal databases. By synthesizing
scenario-question pairs and leveraging Generalized Reward Policy Optimization
(GRPO), RL-QR trains query rewriters tailored to specific retrievers, enhancing
retrieval performance across varied domains. Experiments on industrial in-house
data demonstrate significant improvements, with
$\text{RL-QR}_{\text{multi-modal}}$ achieving an 11\% relative gain in NDCG@3
for multi-modal RAG and $\text{RL-QR}_{\text{lexical}}$ yielding a 9\% gain for
lexical retrievers. However, challenges persist with semantic and hybrid
retrievers, where rewriters failed to improve performance, likely due to
training misalignments. Our findings highlight RL-QR's potential to
revolutionize query optimization for RAG systems, offering a scalable,
annotation-free solution for real-world retrieval tasks, while identifying
avenues for further refinement in semantic retrieval contexts.

## Full Text


<!-- PDF content starts -->

Generalized Reinforcement Learning for Retriever-Specific Query Rewriter
with Unstructured Real-World Documents
Sungguk Cha, DongWook Kim, Taeseung Hahn, Mintae Kim, Youngsub Han, Byoung-Ki Jeon
LG Uplus, South Korea
{sungguk, dongwook92, tshahn, iammt, yshan042, bkjeon }@lguplus.co.kr
Abstract
Retrieval-Augmented Generation (RAG) systems rely heav-
ily on effective query formulation to unlock external knowl-
edge, yet optimizing queries for diverse, unstructured real-
world documents remains a challenge. We introduce RL-QR ,
a reinforcement learning framework for retriever-specific
query rewriting that eliminates the need for human-annotated
datasets and extends applicability to both text-only and
multi-modal databases. By synthesizing scenario-question
pairs and leveraging Generalized Reward Policy Optimiza-
tion (GRPO), RL-QR trains query rewriters tailored to spe-
cific retrievers, enhancing retrieval performance across var-
ied domains. Experiments on industrial in-house data demon-
strate significant improvements, with RL-QRmulti-modal achiev-
ing an 11% relative gain in NDCG@3 for multi-modal RAG
and RL-QRlexical yielding a 9% gain for lexical retrievers.
However, challenges persist with semantic and hybrid retriev-
ers, where rewriters failed to improve performance, likely due
to training misalignments. Our findings highlight RL-QR’s
potential to revolutionize query optimization for RAG sys-
tems, offering a scalable, annotation-free solution for real-
world retrieval tasks, while identifying avenues for further
refinement in semantic retrieval contexts.
Introduction
Retrieval-Augmented Generation (RAG) (Lewis et al. 2020)
has proven to be a powerful and widely adopted approach
across numerous domains, from natural language processing
to multi-modal applications. Its ability to integrate external
knowledge into generation tasks has made it a cornerstone of
modern retrieval systems. Modern AI assistants (Hurst et al.
2024; Comanici et al. 2025) adopt RAG as core function for
correcting factually, delivering out-domain knowledge and
beyond.
In practice, when serving RAG systems across various do-
mains and index formats, adapting queries through rewriting
proves to be more effective and cost-efficient than rebuilding
retrievers. For lexical retrievers, creating domain-specific
dictionaries can enhance performance. However, this ap-
proach depends on manual annotation, which is not scal-
able and increases operational costs. For semantic indices,
retrievers can be fine-tuned with domain-specific data. Yet,
this introduces the burden of maintaining domain-specific
retrievers, generating training data, and conducting retrain-
ing. Moreover, updating retrievers typically requires re-indexing, which adds complexity to RAG system dependen-
cies and further raises operational costs. In contrast, query
rewriters transform queries into the representation space of
the retrievers, allowing compatibility across different re-
trievers and index types. From a system maintenance and
deployment perspective, developing a query rewriter is gen-
erally more cost-effective than enhancing retrievers or re-
indexing. It also promotes modularity in RAG architecture
by decoupling the query rewriting module from retriever
components, avoiding the need for domain-specific retriever
development.
Although query rewriting is central to RAG systems, gen-
eralized approaches remain largely unexplored due to their
reliance on costly human annotation. Recent studies have
proposed implicit learning methods that reward the model
when the final answer is correct, requiring annotated index-
query-answer-verifier sets (Jin et al. 2025). Others use ex-
plicit learning, which rewards the model when relevant doc-
uments are retrieved, but this approach depends on expen-
sive per-query annotations of both positive and negative doc-
ument pairs (Wang et al. 2025). While these methods show
promise within narrow domains, they face major limitations:
they require extensive human effort and are largely restricted
to curated, text-only data sources—making them unsuitable
for real-world, unstructured document collections.
In this work, we introduce a generalized Reinforcement
Learning framework for retriever-specific Query Rewriting
on the unstructured real-world documents ( RL-QR ). Built
onixi-RAG —an in-house industrial RAG system that in-
gests diverse unstructured sources (e.g., PDFs, slide decks)
and processes them into either multi-modal document em-
beddings or text-based chunks—RL-QR is designed to adapt
seamlessly to any type of index, whether text-only or multi-
modal, across any domain. It introduces a scalable and ver-
satile paradigm that enhances RAG systems with any re-
triever by combining index synthesis with reinforcement
learning query rewriter.
Our experiment on real-world indices and retrievers
present versatile and effective improvements on multi-modal
and text-modal indices with various domain sources. Given
industrial unstructured documents and the RAG system
ixi-RAG , RL-QR shows remarkable performance gain on
multi-modal and text-modal indices with retrievers with-
out human-annotation but with automatic train data synthe-arXiv:2507.23242v1  [cs.CV]  31 Jul 2025

sis and reinforcement learning, suggesting its adaptability
and robust performance gain. It overcomes the existing con-
straints of human-labeled data and refined structured text
data, broadening the horizons of retrieval-augmented sys-
tems.
In summary, our contributions are
•Generalized Query Rewriting Framework: We pro-
pose RL-QR , a reinforcement learning-based framework
for retriever-specific query rewriting that generalizes
across domains, retrievers, and index formats—including
both text-based and multi-modal indices—within indus-
trial RAG systems.
•Scalable Without Human Annotations: RL-QR elim-
inates the need for costly human-labeled training data
by leveraging synthetic training data and reinforcement
learning, making it suitable for unstructured, real-world
document collections.
•Empirical Effectiveness and System Integration: Built
on the industrial ixi-RAG platform, RL-QR demon-
strates strong performance improvements across multi-
ple retrievers and domain-specific indices. It offers a
modular, retriever-agnostic solution that reduces system
maintenance overhead and enhances RAG applicability
in production environments.
Related Works
Our work focuses on enhancing the query rewriter for RAG
systems, with an emphasis on handling multi-modal (imaged
documents) and text-modal (text-parsed documents) indices
with real-world unstructured data. In this section, we pro-
vide an overview of the research background, covering the
evolution of RAG, the integration of various modalities in
RAG systems, and the role of query rewriting.
Retrieval-Augmented Generation (RAG)
RAG is a hybrid approach that integrates retrieval-based and
generation-based techniques to improve the performance of
language models on knowledge-intensive tasks. By lever-
aging external knowledge sources, RAG enables models to
produce more accurate and contextually relevant responses.
The paradigm has gained significant attention due to its abil-
ity to combine the strengths of retrieving pertinent docu-
ments and generating coherent text. In the real-world, RAG
systems are widely adopted with online web search ( e.g.,
OpenAI (Hurst et al. 2024), and Gemini (Comanici et al.
2025)) and industrial domains with credential documents.
Early research on RAG established its effectiveness
across various natural language processing tasks (Lewis
et al. 2020). Subsequent studies have proposed advance-
ments, such as improved retrieval mechanisms using dense
retrieval methods (Karpukhin et al. 2020) and the inte-
gration of structured knowledge bases like databases or
graphs (Edge et al. 2024). RAG has also shown promise
in multi-task and few-shot learning scenarios (Izacard et al.
2023), where the retrieval component compensates for lim-
ited training data by accessing external information. How-
ever, challenges remain, particularly in optimizing the re-
trieval process, which depends heavily on the quality ofthe input query—an issue that motivates the exploration of
query rewriting (Ma et al. 2023).
Modalities in RAG
While RAG was initially designed for text-based applica-
tions, recent applications have extended their scope to the
real-world unstructured documents including slide decks,
web pages, blogs, papers and so on supported by docu-
ment parsing approaches. This expansion is critical for tasks
where knowledge sources span multiple formats, requiring
systems to integrate and reason over heterogeneous inputs.
Multi-modal RAG systems have been explored for
image-as-embedding (Faysse et al. 2024) or parsing-
documents-to-text (Feng et al. 2025). Image-as-embedding
approaches (Faysse et al. 2024) embeds imaged document
into document embedding as document semantic embed-
ding (Zhang et al. 2025). Parsing-documents-to-text ap-
proaches (Wei et al. 2024; Feng et al. 2025) converts doc-
uments into plain text, enabling the present text retriev-
ers (Robertson, Zaragoza et al. 2009; Zhang et al. 2025). For
text-modal data, such as parsed text from structured docu-
ments, the challenge lies in effectively retrieving and utiliz-
ing information from long-form or hierarchically organized
content (Larson and Truitt 2024).
Query Rewriting for RAG
Query rewriting is a pivotal component in RAG systems, as
the effectiveness of the retrieval step hinges on how well
the query is formulated. A poorly designed query can lead
to irrelevant or low-quality retrieved documents, undermin-
ing the generation process. Conversely, an optimized query
enhances the relevance of retrieved information, directly im-
proving the overall system performance.
Traditional query rewriting techniques, such as query ex-
pansion and reformulation, have roots in information re-
trieval and rely on heuristics or statistical methods to re-
fine queries (Zhu et al. 2016). In the context of RAG, how-
ever, query rewriting must align with the needs of the re-
triever (Ma et al. 2023). Recent efforts have introduced
learning-based approaches, including neural network mod-
els and reinforcement learning, to dynamically adapt queries
based on system feedback (Wang et al. 2025; Chan et al.
2024; Li et al. 2024; Ma et al. 2023; Jin et al. 2025). Despite
these advances, existing query rewriting techniques often re-
quire extensive annotated data or are constrained to specific
domains (Liu et al. 2021). Our work addresses these gaps
by developing a generalized reinforcement learning frame-
work for query rewriting, tailored to enhance retrieval across
diverse indices without relying on large-scale human anno-
tations.
Method
In this section, we describe our proposed framework, gen-
eralized reinforcement learning for retriever-specific query
rewriting on the unstructured real-world documents (RL-
QR). As illustrated in Figure 1, it consists of two-steps:
(1) synthesizing scenario-question pairs which simulates
long user queries and (2) reinforcement learning the query

Figure 1: Demonstration of prior works on query rewriter and RL-QR with RAG system. RL-QR overcomes excessive human
annotation by query synthesis and simplified reward based framework.
rewriter based on the generated and rewritten queries on
each index.
Synthesizing Long Queries
Serving RAG system in the real-world, the users expect the
system to retrieve adequate documents even in the complex
situation, which used to have longer query with multiple
conditions. In order to synthesize such long user query sce-
nario, we utilize large models LM with instructions Ithat
ask to generate the-document-requiring scenario, question
and answer. We pre-process the raw data DBrawby docu-
ment parsers P, resulting
DB←[
d∈DBrawP(d) (1)
where DB becomes the source DB for a retriever and Pcan
result more than one source-unit (e.g., chunk).
Q←[
d∈DBLM(I, d) (2)
where Qis the set of the synthesized queries concatenat-
ing the generated scenario and the generated query. Prompt
template is provided in Table 1. We filtered Qonly if all
scenario, question and answer are created and formatted.
Though the generated-answer will not be considered in this
paper, it is helpful to generate answerable question and fur-
ther examinations such as overall RAG evaluation. When if
you need to utilize the answer, we recommend to regener-
ate it like a←LM(q, d)where ais the target answer and
q∈Qfor better formatting.
During training, we concatenated the scenario and the
question for the query. The training data becomes
D←[
d∈DB(index d, Q) (3)
where index dis the document index.# Generating document requiring question and answer
Read the document, then (1) think of a scenario that re-
quires the document, (2) create a question that fits the sce-
nario, and (3) provide an answer that matches the ques-
tion.
If the document’s information is insufficient to identify a
situation requiring the document, output blank spaces.
The final response format should follow this structure:
<scenario>...</scenario>
<question>...</question>
<answer>...</answer>
Table 1: Template for long user query synthesis.
Reinforcement Learning Query Rewriter
It is important to individualize query rewriter with respect to
the indices, because each retriever has distinct characteris-
tics. For example, lexical retrievers such as BM25 (Robert-
son, Zaragoza et al. 2009) count on the number of the words,
in which simply repeating important word can augment the
performance. Whereas, (multi-modal) semantic retrievers
that embed (text-parsed-)documents into embedding (Faysse
et al. 2024; Zhang et al. 2025) work better if the query-
document resembles their trained data, which is hard to man-
age. The reinforcement learning aims to align user query
into the index representation space by the query rewriter QR
per specific retriever R. In other words, for Nonline RAG
systems consisting of the data source DBiand the retriever
Rifori∈N, we suggest to have Nretrievers respectively,
rather than a single universal rewriter.
The precedent RL approaches (Ma et al. 2023; Jin et al.
2025; Nguyen et al. 2025) implicitly train the rewriter by

optimizing
max
πθ,πLLMEx∼D,y∼πθ(·|x;R),z∼πLLM (·|x;R(y))[rϕ(x, z)](4)
where xrefers to the sample from the training data D,yde-
notes the rewritten query by the rewriter, and zrepresent the
final response. πθandπLLM are the target rewriter and the
final-responding language model. rϕis the reward function.
In contrast, ours optimizes the rewriter explicitly, which
down-scales the objective and boosts the training process.
Some (Wang et al. 2025) tried explicit rewarding with mas-
sive document-wise positive and negative pair annotation,
which limits in scaling covering domain and indices. On the
other hand, leveraging the synthesized long user queries, we
formulate the RL objective function as follows:
max
πθEx∼D,y∼πθ(·|x;R)[rϕ(x, y)] (5)
We adopt two function rewards, one for the query rewrit-
ing reward and the other for the formatting and redundant
penalty. The retrieval reward rretrieval uses NDCG (J ¨arvelin
and Kek ¨al¨ainen 2002) score directly that measures if the tar-
get document is retrieved.
rretrieval (x, y) =NDCG (index x, R(y)) (6)
The penalty rpenalty targets to match the format, placing the
rewritten query inside <answer>...</answer> , and
reduce redundant generations outside the format.
penalty(y) =|y| − |formatted query |,if format matched
∞, otherwise
(7)
rpenalty (y) =0, ifpenalty (y) = 0
redundancy (y),otherwise
(8)
The reward function becomes
rϕ(x, y) =λ1rretrieval (x,y)+λ2rpenalty (y) (9)
where the lambdas are the hyper-parameters. More specifi-
cally, for each sample x, we optimize the rewriter by maxi-
mizing the following object:
JGRPO(θ) =E
x∼D,{oi}G
i=1∼πθold(O|q)
"
1
GGX
i=11
|oi||oi|X
t=1(
min 
πθ(oi,t|x, oi,<t)
πθold(oi,t|x, oi,<t)ˆAi,t,
clip 
πθ(oi,t|x, oi,<t)
πθold(oi,t|x, oi,<t),1−ϵ,1 +ϵ!
ˆAi,t!)#
(10)
where ϵandβare hyper-parameters, and ˆAi,tis the advan-
tage based on the relative rewards of the outputs inside each
group. Further, we normalized the penalty rpenalty group-
wise by ranging [0.5, 1] for the non-zero values.
Experiment
RAG System Implementation
In this experiment, we implemented two distinct Retrieval-
Augmented Generation (RAG) systems. The first is a multi-
modal RAG chatbot that generates responses using multi-
modal embeddings (derived from image-based documents)combined with user queries. The second is a text-based RAG
chatbot that utilizes parsed text chunks.
For the multi-modal RAG system, we employed
ColQwen2.5-3B (Faysse et al. 2024) for both document
parsing and retrieval. We set the MAX_TOKEN limit for im-
age encoding to 764, aligning it with the token size used for
text encoding. To optimize multi-modal RAG performance,
we pre-computed embeddings for image-based documents
and performed retrieval without late interaction.
For the text-based RAG system, we utilized an in-
house document parser, AI Parser , to convert documents
(e.g., PDFs) into text chunks. We adopted three in-house
retrievers: ixi-RAG lexical ,ixi-RAG semantic ,
andixi-RAG hybrid .
ixi-RAG lexical is a traditional information re-
trieval system built upon an OpenSearch index and utiliz-
ing the BM25 scoring algorithm. This retriever operates by
matching the exact tokens present in the user’s query against
the text chunks in the knowledge base. It excels at precision
when the query contains specific terms, acronyms, or iden-
tifiers that are also present in the source documents, as its
ranking is based on term frequency (TF) and inverse docu-
ment frequency (IDF).
ixi-RAG semantic is a modern retrieval system
that operates on the principle of conceptual similarity
rather than keyword matching. To this end, we utilized
ixi-DocEmbedding , an embedding model specialized
for Korean documents. The model was further trained on
Korean query-document pairs using a domain-specific fine-
tuning approach based on BGE-M3 (Chen et al. 2024).
For training the retriever, we constructed a comprehen-
sive corpus of Korean query-document pairs drawn from
three sources: (1) carefully-curated open-source datasets, (2)
high-quality synthetic pairs generated with large language
models, and (3) production-grade interaction logs collected
from deployed RAG systems. To enhance the learning sig-
nal, we employed iterative hard-negative mining and trained
the encoder according to the BGE-M3 fine-tuning recipe,
which couples InfoNCE contrastive objectives with self-
distillation and model-merging strategies. After training, we
only use the dense retrieval as the semantic retriever. Em-
pirically, the resulting model demonstrates performance on
par with state-of-the-art open-source Korean encoders and
achieves substantial improvements on internal RAG eval-
uation benchmarks. It converts both the user’s query and
the document chunks into high-dimensional vector embed-
dings. Retrieval is then performed by conducting a k-nearest
neighbor (k-NN) (Fix 1985) search within this vector space,
identifying documents whose embeddings are closest to the
query’s embedding. This approach allows the system to un-
derstand the user’s intent and retrieve relevant information
even if the phrasing is different and there is no direct key-
word overlap.
ixi-RAG hybrid combines both lexical and semantic
approaches for improved retrieval. It is designed to leverage
the complementary strengths of both retrieval paradigms.
It first gathers a broad set of candidate documents by run-
ning lexical and semantic searches in parallel. The ranked
lists from both retrievers are then fused using the Recip-

Table 2: RL-QR Training Scheme
Query Rewriter Document Recognizer Document Retriever Train Data
RL-QRmulti-modal ColQwen2.5-v0.2 ColQwen2.5-v0.2 Dmm
RL-QRlexical AI Parser ixi-RAG lexical Dtm
RL-QRsemantic AI Parser ixi-RAG semantic Dtm
RL-QRhybrid AI Parser ixi-RAG hybrid Dtm
rocal Rank Fusion (RRF) algorithm (Cormack, Clarke, and
Buettcher 2009). RRF calculates a new score for each doc-
ument based on its position in the individual rankings, pro-
ducing a single, more robustly ordered list that balances key-
word relevance with semantic similarity. This method ef-
fectively captures both the precision of lexical search and
the contextual understanding of semantic search to deliver a
highly refined final ranking.
Experiment Data
We use randomly sampled 2,145 in-house real-world indus-
trial documents, consisting of various documents in form of
unstructured pdf, word and slides. The documents contains
the necessities, guidelines, announcements and more. For
the use cases, the retrieval rates are crucial. In our experi-
ment, we adopt the NDCG@3 metric to evaluate and reward
the quality, which scores based on the top-3 search results
and the ordered relevant documents.
Long Query Synthesis
We synthesized two types of the queries: multi-modal
queries compatible with the multi-modal RAG-chat-bot
and text-modal queries for the text-modal RAG-chat-bot.
In other words, given the DB and the parsers Pcolqwen
andPAI Parser , we generated multi-modal training data
Dmm and text-modal training data Dtm. We adopted
Qwen2.5-VL-72B andQwen3-32B (Yang et al. 2025)
for synthesizing multi-modal data and text-modal data, re-
spectively. We leveraged think mode of the models and
extracted our target data (see Table 1). We filtered unin-
tended languages and low-quality questions automatically.
It results |Dmm|= 1,609and|Dtm|= 2,980. The aver-
age query lengths are 191 and 159 characters for Dmmand
Dtm, respectively.
Reinforcement Learning Query Rewriter
We initialize the rewriter model with Qwen3-4B in
no_think mode. We train them by the objective single
epoch on two 80GB H100 GPUs without any supervised
finetuning, taking 24 ∼48 H100 hours. For the GRPO
RL training, we adopt TRL library of huggingface
with deepspeed stage 2, 1 batch per machine, 4
steps for the gradient accumulation and 8 samples per a
group. Table 2 demonstrates the training scheme for the
rewriters RL-QRmulti-modal , RL-QRlexical , RL-QRsemantic and
RL-QRhybrid.Table 3: Retrieval performance comparison on multi-modal
RAG.
Multi-modal RAG
Method NDCG@3
Raw query 73.84
Qwen3-4B 73.53
Different retriever specialized
RL-QRsemantic 42.29
RL-QRlexical 41.30
RL-QRhybrid 77.60
Retriever specialized
RL-QRmulti-modal 82.10
Table 4: Retrieval performance comparison on text-modal
RAG with lexical retriever
ixi-RAG lexical
Method NDCG@3
Raw query 72.90
Qwen3-4B 20.01
Different retriever specialized
RL-QRmulti-modal 21.44
RL-QRsemantic 10.28
RL-QRhybrid 74.61
Retriever specialized
RL-QRlexical 79.66
Results
Retrieval Performance on Multi-modal RAG. As shown
in Table 3, RL-QRmulti-modal achieves an NDCG@3 score
of 82.10, representing a relative improvement of over 11%
compared to the original query. Notably, RL-QRhybrid also
demonstrates performance gains despite being trained on a
different RAG system and dataset. In contrast, other mod-
els exhibit a decline in retrieval effectiveness. The baseline
model, Qwen3-4B , fails to contribute positively to retrieval
performance.
Retrieval Performance with Lexical Retriever. As re-
ported in Table 4, RL-QRlexical yields a substantial improve-
ment over the original query, achieving a relative gain of 9%.
RL-QRhybrid also outperforms the raw query with a mod-

Document
Synthetic
long user
query가족이집을비운동안반려동물의안전을확인하고싶은
상황입니다 .스마트홈앱을통해반려동물의상태를모니터
링하고,만약이상징후가감지되면즉시알림을받고대응
할수있는기능이필요합니다 . U+스마트홈앱을사용하여
사생활보호를위해카메라가리기를설정하는방법은무엇
인가요?LG U+의전국대표번호서비스를이용하여특정지역의LBS(위치기반서비스)
를설정하고자하는기업의IT담당자가있습니다 .이담당자는서울강남구와
중랑구에서서비스를제공하려고합니다 .그러나구체적인설정방법과URL에
대한정보가필요합니다 .서울강남구와중랑구에서LG U+의LBS서비스를
설정하기위해어떤절차를거쳐야하며,필요한URL은무엇인가요?
translation This is a situation where the family wants to check the safety
of their pet while they are away from home. There is a need
for a function to monitor the pet’s condition through the
Smart Home app, and to receive immediate alerts and respond
if any abnormal signs are detected. How can the camera cover
be set for privacy protection using the U+ Smart Home app?There is an IT manager of a company who wants to configure an LBS (Location-
Based Service) for a specific area using LG U+’s nationwide representative num-
ber service. This manager intends to provide the service in Gangnam-gu and
Jungnang-gu of Seoul. However, detailed information on the configuration method
and the URL is required. What procedures must be followed to configure LG U+’s
LBS service in Gangnam-gu and Jungnang-gu of Seoul, and what is the required
URL?
RL-QR
rewrote
queryU+스마트홈앱을사용해카메라를가리기위한카메라가
리기기능을확장해야합니다 .스마트홈앱을사용해반려동
물의상태를모니터링하고이상징후를감지하는데어떻게
카메라가리기를사용할수있는지,그리고스마트홈앱과
스마트폰을연결해카메라를가리면서도사생활을보호하
는데필요한카메라가리기방법을확장해야합니다 .특히
U+앱을사용해카메라를가리기위한방법을확장해야합
니다 .카메라를가리면서도스마트홈앱을통해반려동물
상태를모니터링하고이상을감지할수있는방법을확장
해야합니다 .또한카메라를가리기위한스마트홈앱과의
연결을확장해야합니다 .또한카메라가리기를사용해반
려동물상태를모니터서울강남구와중랑구를포함하는LG U+의LBS서비스를서울강남구와중랑
구를서비스를제공하려고하는IT님과함께사용하는경우,서울강남구와중
랑구를위한구체적인설정방법과URL을확장해야하며.특히서울강남구와
중랑구를사용할때LG U+의전국대표번호서비스를이용해LBS를설정하
기위한URL과구체적인절차를확장해야합니다 .서울강남구와중랑구에서
사용하는경우필요한URL은무엇인지에대한내용을확장해야합니다 .또한
중랑구와강남구를사용할때LG U+의LBS서비스를위한구체적인URL과
설정방법을확장해야합니다 .그리고서울강남구와중랑구를사용할경우필
요한URL을확장해야합니다 .또한구체적인설정을위한URL을확장해야
합니다 .또한URL을확장해야합니다 .또한
translation The camera cover function needs to be expanded in order to
cover the camera using the U+ Smart Home app. It is neces-
sary to expand how the camera cover can be used to monitor
the pet’s condition and detect any abnormal signs using the
Smart Home app, and how to protect privacy by covering the
camera while connecting the Smart Home app with the smart-
phone. In particular, the method for covering the camera us-
ing the U+ app needs to be expanded. The method for moni-
toring the pet’s condition and detecting abnormalities through
the Smart Home app while covering the camera also needs to
be expanded. In addition, the connection between the Smart
Home app and the camera cover needs to be expanded. Also,
using the camera cover to monitor the pet’s condition through
the Smart Home app—When using LG U+’s LBS service that includes Seoul Gangnam-gu and
Jungnang-gu together with the IT person who intends to provide services in Seoul
Gangnam-gu and Jungnang-gu, the specific configuration method and URL for
Seoul Gangnam-gu and Jungnang-gu must be expanded. In particular, when using
Seoul Gangnam-gu and Jungnang-gu, the URL and detailed procedure to set up
LBS using LG U+’s nationwide representative number service must be expanded.
The content about what URL is required when used in Seoul Gangnam-gu and
Jungnang-gu must be expanded. Also, when using Jungnang-gu and Gangnam-
gu, the specific URL and configuration method for LG U+’s LBS service must
be expanded. And when using Seoul Gangnam-gu and Jungnang-gu, the required
URL must be expanded. Additionally, the URL for detailed configuration must be
expanded. Also, the URL must be expanded. Also
Figure 2: Real-world examples of long user query synthesis and query rewriting.

Table 5: Retrieval performance comparison on text-modal
RAG with semantic retriever
ixi-RAG semantic
Raw query 74.09
Qwen3-4B 25.56
Different retriever specialized
RL-QRmulti-modal 27.92
RL-QRlexical 56.43
RL-QRhybrid 72.76
Retriever specialized
RL-QRsemantic 71.02
Table 6: Retrieval performance comparison on text-modal
RAG with semantic retriever
ixi-RAG hybrid
Raw query 81.93
Qwen3-4B 26.99
Different retriever specialized
RL-QRmulti-modal 27.07
RL-QRsemantic 65.45
RL-QRlexical 72.39
Retriever specialized
RL-QRhybrid 81.20
est 2% gain, likely due to its partial exposure to lexical
retriever signals during training. Conversely, all other ap-
proaches demonstrate significantly degraded performance,
indicating that models reinforced using non-lexical retriev-
ers may be ineffective when applied to a purely lexical re-
trieval system.
Retrieval Performance with Semantic and Hybrid Re-
trievers. As presented in Tables 5, 6, the proposed rewrit-
ing methods did not enhance retrieval performance. Recent
work (Su et al. 2024) reports pool performances of semantic
retrievers than which of lexical retrievers (e.g., BM25) with
rewrote queries. The train data for semantic retrievers have
little relation with the LLM-rewrote queries which are rela-
tively longer, resulting poor representation learning for the
LLM-rewrote queries. If we adopt reasoning semantic re-
triever ( e.g., ReasonIR (Shao et al. 2025)), the proposed RL-
QR might improved with the semantic retrievers. We leave
it for the future work.
Qualitative results. Figure 2 illustrates real-world ex-
amples with long query synthesis and re-wrote queries by
RL-QRmulti-modal . The synthetic long user queries are ade-
quate and dedicated to the documents. The rewrote queries
shows patterns (1) emphasizing keywords by repetition in
manner of duplicating sentences, (2) unique accents ( must
be expanded ) which are far the from natural languages.
Correlation between the query length and the retrieval
performance. As indicated in Table 7 and observed in and
Figure 2, excluding the RL-QRmulti-modal case on DtmwhereTable 7: Average length of the rewritten query.
Method DataAverage Length
Origin Rewrote
RL-QRmulti-modal
Dmm 191504
RL-QRlexical 357
RL-QRsemantic 436
RL-QRhybrid 415
RL-QRmulti-modal
Dtm 15960
RL-QRlexical 353
RL-QRsemantic 435
RL-QRhybrid 405
no improvements were observed, the proposed methods gen-
erally increase the query length by more than twofold. This
trend may stem from the limited training scheme, which
involves only thousands of training samples and a single
epoch. Alternatively, the tendency to expand queries could
be attributed to the simplistic reward training approach.
Conclusion
This work introduces RL-QR , a novel reinforcement learn-
ing framework for retriever-specific query rewriting that sig-
nificantly advances the capabilities of Retrieval-Augmented
Generation (RAG) systems. By eliminating the reliance on
costly human-annotated datasets and extending applicability
to unstructured, real-world documents across various modal-
ities, our approach addresses critical limitations of prior
methods. The experimental results demonstrate the frame-
work’s effectiveness, with notable improvements in retrieval
performance, particularly for multi-modal and lexical re-
trievers. Specifically, RL-QRmulti-modal achieved an 11% rela-
tive improvement in NDCG@3 for multi-modal RAG, while
RL-QRlexical delivered a 9% gain for lexical retrieval sys-
tems. However, challenges remain in optimizing rewriters
for semantic and hybrid retrievers, where no performance
gains were observed, likely due to misalignments in training
objectives between rewriters and retrievers.
The observed increase in query length suggests that the
current training scheme, constrained by limited samples and
a single epoch, may inadvertently favor verbose queries.
Future work could explore larger-scale training or refined
reward mechanisms to balance query conciseness with re-
trieval effectiveness. Additionally, enhancing the alignment
between rewriters and semantic retrievers through advanced
reward policies or training strategies could unlock further
performance gains.
In conclusion, RL-QR offers a scalable, annotation-free
solution for query optimization, with broad applicability to
diverse retrievers and databases. Its demonstrated success on
industrial in-house data underscores its potential to trans-
form real-world retrieval systems, paving the way for more
efficient and versatile RAG applications.

References
Chan, C.-M.; Xu, C.; Yuan, R.; Luo, H.; Xue, W.;
Guo, Y .; and Fu, J. 2024. Rq-rag: Learning to refine
queries for retrieval augmented generation. arXiv preprint
arXiv:2404.00610 .
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and
Liu, Z. 2024. BGE M3-Embedding: Multi-Lingual, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. arXiv:2402.03216.
Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.;
Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.;
Rosen, E.; et al. 2025. Gemini 2.5: Pushing the Frontier
with Advanced Reasoning, Multimodality, Long Context,
and Next Generation Agentic Capabilities. arXiv preprint
arXiv:2507.06261 .
Cormack, G. V .; Clarke, C. L.; and Buettcher, S. 2009. Re-
ciprocal rank fusion outperforms condorcet and individual
rank learning methods. In Proceedings of the 32nd interna-
tional ACM SIGIR conference on Research and development
in information retrieval , 758–759.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.;
Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and
Larson, J. 2024. From local to global: A graph rag ap-
proach to query-focused summarization. arXiv preprint
arXiv:2404.16130 .
Faysse, M.; Sibille, H.; Wu, T.; Omrani, B.; Viaud, G.;
Hudelot, C.; and Colombo, P. 2024. Colpali: Efficient docu-
ment retrieval with vision language models. In The Thir-
teenth International Conference on Learning Representa-
tions .
Feng, H.; Wei, S.; Fei, X.; Shi, W.; Han, Y .; Liao, L.; Lu, J.;
Wu, B.; Liu, Q.; Lin, C.; et al. 2025. Dolphin: Document
image parsing via heterogeneous anchor prompting. arXiv
preprint arXiv:2505.14059 .
Fix, E. 1985. Discriminatory analysis: nonparametric dis-
crimination, consistency properties , volume 1. USAF school
of Aviation Medicine.
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
Izacard, G.; Lewis, P.; Lomeli, M.; Hosseini, L.; Petroni, F.;
Schick, T.; Dwivedi-Yu, J.; Joulin, A.; Riedel, S.; and Grave,
E. 2023. Atlas: Few-shot learning with retrieval augmented
language models. Journal of Machine Learning Research ,
24(251): 1–43.
J¨arvelin, K.; and Kek ¨al¨ainen, J. 2002. Cumulated gain-based
evaluation of IR techniques. ACM Transactions on Informa-
tion Systems (TOIS) , 20(4): 422–446.
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Za-
mani, H.; and Han, J. 2025. Search-r1: Training llms to rea-
son and leverage search engines with reinforcement learn-
ing. arXiv preprint arXiv:2503.09516 .
Karpukhin, V .; Oguz, B.; Min, S.; Lewis, P. S.; Wu, L.;
Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Pas-
sage Retrieval for Open-Domain Question Answering. In
EMNLP (1) , 6769–6781.Larson, J.; and Truitt, S. 2024. GraphRAG: Unlocking LLM
discovery on narrative private data.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, Z.; Wang, J.; Jiang, Z.; Mao, H.; Chen, Z.; Du, J.; Zhang,
Y .; Zhang, F.; Zhang, D.; and Liu, Y . 2024. DMQR-RAG:
Diverse Multi-Query Rewriting for RAG. arXiv preprint
arXiv:2411.13154 .
Liu, H.; Chen, M.; Wu, Y .; He, X.; and Zhou, B. 2021. Con-
versational query rewriting with self-supervised learning.
InICASSP 2021-2021 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP) , 7628–
7632. IEEE.
Ma, X.; Gong, Y .; He, P.; Zhao, H.; and Duan, N. 2023.
Query rewriting in retrieval-augmented large language mod-
els. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing , 5303–5315.
Nguyen, D. A.; Mohan, R. K.; Yang, V .; Akash, P. S.; and
Chang, K. C.-C. 2025. RL-based Query Rewriting with Dis-
tilled LLM for online E-Commerce Systems. arXiv preprint
arXiv:2501.18056 .
Robertson, S.; Zaragoza, H.; et al. 2009. The probabilistic
relevance framework: BM25 and beyond. Foundations and
Trends® in Information Retrieval , 3(4): 333–389.
Shao, R.; Qiao, R.; Kishore, V .; Muennighoff, N.; Lin, X. V .;
Rus, D.; Low, B. K. H.; Min, S.; Yih, W.-t.; Koh, P. W.; et al.
2025. ReasonIR: Training Retrievers for Reasoning Tasks.
arXiv preprint arXiv:2504.20595 .
Su, H.; Yen, H.; Xia, M.; Shi, W.; Muennighoff, N.;
Wang, H.-y.; Liu, H.; Shi, Q.; Siegel, Z. S.; Tang, M.;
et al. 2024. Bright: A realistic and challenging bench-
mark for reasoning-intensive retrieval. arXiv preprint
arXiv:2407.12883 .
Wang, Y .; Zhang, H.; Pang, L.; Guo, B.; Zheng, H.; and
Zheng, Z. 2025. MaFeRw: Query rewriting with multi-
aspect feedbacks for retrieval-augmented large language
models. In Proceedings of the AAAI Conference on Artifi-
cial Intelligence , volume 39, 25434–25442.
Wei, H.; Liu, C.; Chen, J.; Wang, J.; Kong, L.; Xu, Y .; Ge, Z.;
Zhao, L.; Sun, J.; Peng, Y .; et al. 2024. General ocr theory:
Towards ocr-2.0 via a unified end-to-end model.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv preprint arXiv:2505.09388 .
Zhang, Y .; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou,
J. 2025. Qwen3 Embedding: Advancing Text Embedding
and Reranking Through Foundation Models. arXiv preprint
arXiv:2506.05176 .
Zhu, N.; Li, X.; Xiong, L.; and Xue, H. 2016. Query Rewrit-
ing for Archived Information Retrieval. In Proceedings of
the International Conference on Internet Multimedia Com-
puting and Service , 323–326.
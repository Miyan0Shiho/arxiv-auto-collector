# MemInsight: Autonomous Memory Augmentation for LLM Agents

**Authors**: Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, Yassine Benajiba

**Published**: 2025-03-27 17:57:28

**PDF URL**: [http://arxiv.org/pdf/2503.21760v1](http://arxiv.org/pdf/2503.21760v1)

## Abstract
Large language model (LLM) agents have evolved to intelligently process
information, make decisions, and interact with users or tools. A key capability
is the integration of long-term memory capabilities, enabling these agents to
draw upon historical interactions and knowledge. However, the growing memory
size and need for semantic structuring pose significant challenges. In this
work, we propose an autonomous memory augmentation approach, MemInsight, to
enhance semantic data representation and retrieval mechanisms. By leveraging
autonomous augmentation to historical interactions, LLM agents are shown to
deliver more accurate and contextualized responses. We empirically validate the
efficacy of our proposed approach in three task scenarios; conversational
recommendation, question answering and event summarization. On the LLM-REDIAL
dataset, MemInsight boosts persuasiveness of recommendations by up to 14%.
Moreover, it outperforms a RAG baseline by 34% in recall for LoCoMo retrieval.
Our empirical results show the potential of MemInsight to enhance the
contextual performance of LLM agents across multiple tasks.

## Full Text


<!-- PDF content starts -->

MemInsight: Autonomous Memory Augmentation for LLM Agents
Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, Yassine Benajiba
AWS AI
{ranasal, cjinglun, miyuan, ancurrey, sunkaral, yizhngn, benajiy} @amazon.com
Abstract
Large language model (LLM) agents have
evolved to intelligently process information,
make decisions, and interact with users or tools.
A key capability is the integration of long-term
memory capabilities, enabling these agents to
draw upon historical interactions and knowl-
edge. However, the growing memory size
and need for semantic structuring pose signif-
icant challenges. In this work, we propose an
autonomous memory augmentation approach,
MemInsight, to enhance semantic data repre-
sentation and retrieval mechanisms. By lever-
aging autonomous augmentation to historical
interactions, LLM agents are shown to deliver
more accurate and contextualized responses.
We empirically validate the efficacy of our pro-
posed approach in three task scenarios; con-
versational recommendation, question answer-
ing and event summarization. On the LLM-
REDIAL dataset, MemInsight boosts persua-
siveness of recommendations by up to 14%.
Moreover, it outperforms a RAG baseline by
34% in recall for LoCoMo retrieval. Our empir-
ical results show the potential of MemInsight
to enhance the contextual performance of LLM
agents across multiple tasks.
1 Introduction
LLM agents have emerged as an advanced frame-
work to extend the capabilities of LLMs to im-
prove reasoning (Yao et al., 2023; Wang et al.,
2024c), adaptability (Wang et al., 2024d), and self-
evolution (Zhao et al., 2024a; Wang et al., 2024e;
Tang et al., 2025). A key component of these agents
is their memory module, which retains past inter-
actions to allow more coherent, consistent, and
personalized responses across various tasks. The
memory of the LLM agent is designed to emu-
late human cognitive processes by simulating how
knowledge is accumulated and historical experi-
ences are leveraged to facilitate complex reasoning
and the retrieval of relevant information to informactions (Zhang et al., 2024). However, the advan-
tages of an LLM agent’s memory also introduce
notable challenges (Wang et al., 2024b). (1) As
data accumulates over time, retrieving relevant in-
formation becomes increasingly challenging, es-
pecially during extended interactions or complex
tasks. (2) Processing large historical data, which
can grow rapidly, as interactions accumulate, re-
quires effective memory management strategies.
(3) Storing data in its raw format can hinder ef-
ficient retrieval of pertinent knowledge, as distin-
guishing between relevant and irrelevant details
becomes more challenging, potentially leading to
noisy or imprecise information that compromises
the agent’s performance. Furthermore, (4) the in-
tegration of knowledge across tasks is constrained,
limiting the agent’s ability to effectively utilize
data from diverse contexts. Consequently, effec-
tive knowledge representation and structuring of
LLM agent memory are essential to accumulate
relevant information and enhance understanding
of past events. Improved memory management
enables better retrieval and contextual awareness,
making this a critical and evolving area of research.
Hence, in this paper we introduce an autonomous
memory augmentation approach, MemInsight,
which empowers LLM agents to identify critical in-
formation within the data and proactively propose
effective attributes for memory enhancements. This
is analogous to the human processes of attentional
control and cognitive updating, which involve se-
lectively prioritizing relevant information, filtering
out distractions, and continuously refreshing the
mental workspace with new and pertinent data (Hu
et al., 2024; Hou et al., 2024).
MemInsight autonomously generates augmen-
tations that encode both relevant semantic and
contextual information for memory. These aug-
mentations facilitate the identification of mem-
ory components pertinent to various tasks. Ac-
cordingly, MemInsight can improve memory re-arXiv:2503.21760v1  [cs.CL]  27 Mar 2025

trieval by leveraging relevant attributes of memory,
thereby supporting autonomous LLM agent adapt-
ability and self-evolution.
Our contributions can be summarized as follows:
•We propose a structured autonomous ap-
proach that adapts LLM agents’ memory rep-
resentations while preserving context across
extended conversations for various tasks.
•We design and apply memory retrieval meth-
ods that leverage the generated memory aug-
mentations to filter out irrelevant memory
while retaining key historical insights.
•Our promising empirical findings demonstrate
the effectiveness of MemInsight on several
tasks: conversational recommendation, ques-
tion answering, and event summarization.
2 Related Work
Well-organized and semantically rich memory
structures enable efficient storage and retrieval of
information, allowing LLM agents to maintain con-
textual coherence and provide relevant responses.
Developing an effective memory module in LLM
agents typically involves two critical components:
structural memory generation and memory retrieval
methods (Zhang et al., 2024; Wang et al., 2024a).
LLM Agents Memory Recent research in LLM
agents memory focuses on developing methods
for effectively storing previous interactions and
feedback (Packer et al., 2024). Contemporary ap-
proaches emphasize memory structures that en-
hance the adaptability of agents and improve their
ability to generalize to previously unseen environ-
ments (Zhao et al., 2024a; Zhang et al., 2024;
Zhu et al., 2023). Common memory forms in-
clude summaries and abstract high-level informa-
tion from raw observations to capture key points
and reduce information redundancy (Maharana
et al., 2024). Other approaches include structur-
ing memory as summaries, temporal events, or rea-
soning chains (Zhao et al., 2024a; Zhang et al.,
2024; Zhu et al., 2023; Maharana et al., 2024;
Anokhin et al., 2024; Liu et al., 2023a). In ad-
dition, there are studies that enrich raw conversa-
tions with semantic representations like sequence
of events and historical event summaries (Zhong
et al., 2023; Maharana et al., 2024) or extract
reusable workflows from canonical examples andintegrate them into memory to assist test-time infer-
ence (Wang et al., 2024f). However, all aforemen-
tioned studies rely on either unstructured memory
or human-designed attributes for memory represen-
tation, while MemInsight leverages the AI agent’s
autonomy to discover the ideal attributes for struc-
tured representation.
LLM Agents Memory Retrieval Existing works
have leveraged memory retrieval techniques for
efficiency when tackling vast amounts of histori-
cal context (Hu et al., 2023a; Zhao et al., 2024b;
Tack et al., 2024; Ge et al., 2025). Common ap-
proaches for memory retrieval include generative
retrieval models, which encode memory as dense
vectors and retrieve the top- krelevant documents
based on similarity search techniques (Zhong et al.,
2023; Penha et al., 2024). Various similarity met-
rics, such as cosine similarity (Packer et al., 2024),
are employed, alongside advanced techniques like
dual-tower dense retrieval models, which encode
each memory history into embeddings indexed by
FAISS (Johnson et al., 2017) to enhance retrieval
efficiency (Zhong et al., 2023). Additionally, meth-
ods such as Locality-Sensitive Hashing (LSH) are
utilized to retrieve tuples containing related entries
in memory (Hu et al., 2023b).
3 Autonomous Memory Augmentation
Our proposed MemInsight model is designed to en-
hance memory representation through a structured
augmentation process that optimizes memory re-
trieval. Figure 1 presents an overview of the model,
highlighting its main modules: attribute mining,
annotation, and memory retriever.
3.1 Attribute Mining and Annotation
To ensure the effectiveness of these attributes in
future interactions, they must be meaningful, ac-
curate, and Attribute mining in our MemInsight
model leverages a backbone LLM to autonomously
identify and define key attributes that encapsulate
semantic knowledge from user interactions. This
entails selecting attributes most relevant to the task
under consideration and employs them to anno-
tate historical conversations. Effective attributes
must be meaningful, accurate, and contextually rel-
evant to enhance future interactions. To achieve
this, the augmentation process follows a structured
approach, defining the perspective from which the
attributes are derived, determining the appropriate
level of augmentation granularity, and establishing

LLM AgentsMemory ModuleMemory Processes
MemInsight
Augment
Retrieve
MemInsight AugmentationWriteAnnotation
Test-time Inference          AgentTasks1- Conversational Recommendation
2- Question Answering
3- Event SummarizationMemory R etriev erAttribute-based Embedding-basedMemory Augmentation
Memory A ugmentation
Attribute
MiningAttribute 
EmbeddingConversations
Attribute Mining
Attributes Prioritization
- Basic 
- PrioritiyAttributes Perspective
- Item Centric
- Conversation CentricAttributes Granularity
- Turn Level
- Session Level Augmentations MemoryRetrieval Pool
- Attribute Based
- Embedding Based
Comprehensive
- Similarity Search (FIASS) - All Augmentations - Exact Match FilterFigure 1: Main modules of MemInsight including, Attribute Mining, Memory Retrieval, and Annotation, triggered by different
memory processes: Augment, Write, and Retrieve. In addition to the test-time inference evaluation downstream tasks, memory
augmentation and adopted memory retrieval methods.
a coherent sequence for annotation. The result-
ing attributes and values are then used to enrich
memory, ensuring a well-organized and informa-
tive representation of past interactions.
3.1.1 Attribute Perspective
Attribute generation is guided by two primary ori-
entations: entity-centric and conversation-centric.
Entity-centric emphasizes a specific item stored
in memory such as movies or books. Attributes
generated for entity-centric augmentations should
capture the main characteristics and features of
this entity. For example, attributes for a movie en-
tity might include the director, actors, and year of
release, while attributes for a book entity would
encompass the author, publisher, and number of
pages. On the other hand, conversation-centric
augmentations focus on annotating and character-
izing the entire user interaction from the user’s
perspective. This approach ensures that the ex-
tracted attributes align with the user’s intent, pref-
erences, sentiment, emotions, motivations, and
choices, thereby improving personalized responses
and memory retrieval. An illustrative example is
provided in Figure 4.
3.1.2 Attribute Granularity
While entity-centric augmentations focus on spe-
cific entities in memory, conversation-centric aug-
mentations introduce an additional factor: attribute
granularity, which determines the level of details
captured in the augmentation process. The augmen-
tation attributes can be analyzed at varying levels
of abstraction, either at the level of individual turns
within a user conversation (turn-level), or across the
entire dialogue session (session-level), each offer-
ing distinct insights into the conversational context.
At the turn level, each dialogue turn is indepen-
Melanie : "Hey Caroline, since we last chatted, I've
had a lot of things happening to me. I ran a charity
race for mental health last Saturday it was really
rewarding. Really made me think about taking care of
our minds. “
Caroline : "That charity race sounds great, Mel!
Making a dif ference & raising awareness for mental
health is super rewarding - I'm really proud of you
for taking part!“
Melanie : "Thanks, Caroline! The event was really
thought-provoking. I'm starting to realize that
self-care is really important. It's a journey for
me, but when I look after myself, I'm able to better
look after my family .“
Caroline : "I totally agree, Melanie. Taking care of
ourselves is so important - even if it's not always
easy. Great that you're prioritizing self-care.Turn Level Augmentation:
Turn 1 : [event]:<charity race for mental health>
[time]: <"last saturday“> [emotion]:<"rewarding“>
[topic]: <mental health>
Session Level Augmentation:
Melanie : [event]<ran charity race for mental health>,
[emotion]<rewarding>,[intent]<thinking about self-
care>
Caroline :[event]<raising mental health awareness>,
[emotion]<proud>Figure 2: An example for Turn level and Session level anno-
tations for a sample dialogue conversation from the LoCoMo
Dataset.
dently augmented, focusing on the specific content
of individual turns to generate more nuanced and
contextual attributes. In contrast, session-level an-
notation considers the entire dialogue, generating
generalized attributes that capture the broader con-
versational context. Due to its broader granularity,
session-level augmentation emphasizes high-level
attributes and conversational structures rather than
the detailed features of individual turns. An ex-
ample of both levels is illustrated in Figure 2 for
a sample dialogue turns. As shown, turn-level an-
notations offer finer-grained details, while session-
level annotations provide a broader overview of the
dialogue.
3.1.3 Annotation and Attribute Prioritization
Subsequently, the generated attributes and their cor-
responding values are used to annotate the agent’s
memory. Annotation is done by aggregating at-
tributes and values in the relevant memory in the
form:
{mi:⟨a1, v1⟩, ..., ⟨an, vn⟩} (1)

where mistands for relevant memory and ai, vi
denote attributes and values respectively. The rel-
evant memory may correspond to turn or session
level. Attributes are typically aggregated using
the Attribute Prioritization method, which can be
classified into Basic and Priority. In Basic Aug-
mentation, attributes are aggregated without a pre-
defined order, resulting in an arbitrary sequence
i1, .., i n. In contrast, Priority Augmentation sorts
attribute-value pairs according to their relevance
to the memory being augmented. This prioritiza-
tion follows a structured order in which attribute i1
holds the highest significance, ensuring that more
relevant attributes are processed first.
3.2 Memory Retrieval
MemInsight augmentations are employed to enrich
or retrieve relevant memory. For comprehensive
retrieval, memory is retrieved along with all asso-
ciated augmentations to generate a more context-
aware response. Additionally, MemInsight can
refine the retrieval process. Initially, the current
context is augmented to identify task-specific and
interaction-related attributes, which then guide the
retrieval of the pertinent memory. Two primary
retrieval methods are proposed: (1) Attribute-based
Retrieval , leverages the current context to generate
attributes tailored to the specific task at hand. These
attributes serve as criteria for selecting and retriev-
ing relevant memory that shares similar attributes
in their augmentations. The retrieved memories,
which align with the required attributes, are subse-
quently integrated into the current context to enrich
the ongoing interaction. (2) Embedding-based Re-
trieval , utilizes memory augmentations to create a
unique embedding representation for each memory
instance, derived from its aggregated annotations.
Simultaneously, the augmentations of the current
context are embedded to form a query vector, which
is then used in a similarity-based search to retrieve
the top- kmost relevant memories. Finally, all re-
trieved memory are incorporated into the current
context to enhance the relevance and coherence of
the ongoing interaction. A detailed description of
this method can be found in Appendix C.
4 Evaluation
4.1 Datasets
We conduct a series of experiments on the datasets:
LLM-REDIAL (Liang et al., 2024) and Lo-
CoMo (Maharana et al., 2024). LLM-REDIALis a dataset for evaluating movie Conversational
Recommendation, containing approximately 10K
dialogues covering 11K movies in memory. While
LoCoMo is a dataset for evaluating Question An-
swering and Event Summarization, consisting of
30 long-term dialogues across up to 10 sessions
between two speakers. LoCoMo includes five ques-
tion categories: Single-hop, Multi-hop, Temporal
reasoning, Open-domain knowledge, and Adversar-
ial questions. Each question has a reference label
that specifies the relevant dialogue turn in memory
required to generate the answer. Additionally, Lo-
CoMo provides event labels for each speaker in a
session, which we use as ground truth for Event
Summarization evaluation.
4.2 Experimental Setup
To evaluate our model, we begin by augmenting
the datasets using a backbone LLM with zero-shot
prompting to identify relevant attributes and their
corresponding values. For augmentation genera-
tion and evaluation across various tasks, we uti-
lize the following models for attribute generation:
Claude Sonnet,1Llama2and Mistral.3For the
Event Summarization task, we also use the Claude-
3-Haiku model.4For embedding-based retrieval
tasks, we employ the Titan Text Embedding model
5to generate embeddings. The augmented memory
is then embedded and indexed using FAISS (John-
son et al., 2017) for vector indexing and search. To
ensure consistency across all experiments, we use
the same base model for the primary tasks: recom-
mendation, answer generation, and summarization,
while evaluating different models for augmenta-
tion. Claude Sonnet serves as the backbone LLM
in baselines for all tasks.
4.3 Evaluation Metrics
The evaluation metrics used for assessing different
tasks using MemInsight include, traditional met-
rics like F1-score metric for answer prediction and
recall for accuracy in Question Answering. Re-
call@K and NDCG@K for Conversational Rec-
ommendation, along with LLM-based metrics for
genre matching.
We also evaluate using subjective metrics includ-
ing Persuasiveness, used in Liang et al. (2024), to
1claude-3-sonnet-20240229-v1
2llama3-70b-instruct-v1
3mistral-7b-instruct-v0
4claude-3-Haiku-20240307-v1
5titan-embed-text-v2:0

assess how persuasive the recommendations are
relative to the ground truth. Additionally, we in-
troduce a Relatedness metric, where we prompt an
LLM to measure how comparable the recommenda-
tion attributes are to the ground truth, categorizing
them as not comparable, comparable, or highly
comparable. Finally, we assess Event Summariza-
tion using an LLM-based metric, G-Eval (Liu et al.,
2023b), a summarization evaluation metric that
measures the relevance, consistency, and coherence
of generated summaries as opposed to reference
labels. These metrics provide a comprehensive
framework for evaluating both retrieval effective-
ness and response quality.
5 Experiments
5.1 Questioning Answering
Questioning Answering task experiments are con-
ducted to evaluate the effectiveness of MemInsight
in answer generation. We assess overall accuracy
to measure the system’s ability to retrieve and in-
corporate relevant information from augmentations.
The base model, which incorporates all historical
dialogues without augmentation, serves as the base-
line. We additionally consider Dense Passage Re-
trieval (DPR) RAG model (Karpukhin et al., 2020)
as a comparative baseline due to its speed and scal-
ability.
Memory Augmentation In this task, mem-
ory is constructed from historical conversational
dialogues, which requires the generation of
conversation-centric attributes for augmentation.
Given that the ground truth labels consist of dia-
logue turns relevant to the question, the dialogues
are annotated at the turn level. A backbone LLM
is prompted to generate augmentation attributes
for both conversation-centric and turn-level annota-
tions.
Memory Retrieval To answer a given question,
the relevant dialogue turn must be retrieved from
historical dialogues. In order to retrieve the rele-
vant dialogue turn, the question is first augmented
to identify relevant attributes and a memory re-
trieval method is applied. We evaluate different
MemInsight memory retrieval methods to demon-
strate the efficacy of our model. We employ
attribute-based retrieval by selecting dialogue turns
augmented with attributes that exactly match the
question’s attributes. Additionally, we evaluate
the embedding-based retrieval, where the augmen-tations are embedded and indexed for retrieval.
Hence, the question and attributes are transformed
into an embedded query, which is used to perform
a vector similarity search to retrieve the top- kmost
similar dialogue turns. Once the relevant memory
is retrieved, it is integrated into the current context
to generate the final answer.
Experimental Results We initiate our evaluation
by assessing attribute-based memory retrieval us-
ing the Claude-3-Sonnet model. Table 1 presents
the overall F1 score, measuring the accuracy of
the generated answers. As shown in the table,
attribute-based retrieval outperforms the baseline
model by 3% in overall accuracy, with notable im-
provements in single-hop, temporal reasoning, and
adversarial questions, which require advanced con-
textual understanding and reasoning. These results
indicate that the augmented history enriched the
context, leading to better reasoning and a signif-
icant increase in the F1 score for answer genera-
tion. Additionally, we perform a detailed analysis
of embedding-based retrieval, where we consider
evaluating basic and priority augmentation using
the Claude-3-Sonnet model.
Table 1 demonstrates that the priority augmen-
tation consistently outperforms the basic model
across all questions. This finding suggests that
the priority relevance of augmentations enhances
context representation for conversational data. Sub-
sequently, we evaluate the priority augmentations
using Llama, and Mistral models for Embedding-
based retrieval. As shown in the table, the
Embedding-based retrieval outperforms the RAG
baseline across all question categories, except
for adversarial questions, yet the overall accu-
racy of MemInsight remains superior. Addition-
ally, MemInsight demonstrates a significant im-
provement in performance on multi-hop questions,
which require reasoning over multiple pieces of
supporting evidence. This suggests that the gener-
ated augmentations provided a more robust under-
standing and a broader perspective of the historical
dialogues. RECALL metrics in Table 2 revealed
a more significant boost, with priority augmenta-
tions increasing accuracy across all categories and
yielding a 35% overall improvement.
5.2 Conversational Recommendation
We simulate conversational recommendation by
preparing dialogues for evaluation under the same
conditions proposed by Liang et al. (2024). This

Model Single-hop Multi-hop Temporal Open-domain Adversarial Overall
Baseline (Claud-3-Sonnet) 15.0 10.0 3.3 26.0 45.3 26.1
Attribute-based Retrieval
MemInsight (Claude-3-Sonnet) 18.0 10.3 7.5 27.0 58.3 29.1
Embedding-Based Retrieval
RAG Baseline (DPR) 11.9 9.0 6.3 12.0 89.9 28.7
MemInsight (Llama v3 Priority ) 14.3 13.4 6.0 15.8 82.7 29.7
MemInsight (Mistral v1 Priority ) 16.1 14.1 6.1 16.7 81.2 30.0
MemInsight (Claude-3-Sonnet Basic ) 14.7 13.8 5.8 15.6 82.1 29.6
MemInsight (Claude-3-Sonnet Priority ) 15.8 15.8 6.7 19.1 75.3 30.1
Table 1: Results for F1 Score (%) for answer generation accuracy for attribute-based and embedding-based memory retrieval
methods. Baseline is Claude-3-Sonnet model to generate answers using all memory without augmentation, for Attribute-based
retrieval. In addition to the Dense Passage Retrieval(DPR) for Embedding-based retrieval. Evaluation is done with k= 5. Best
results per question category over all methods are in bold.
Model Single-hop Multi-hop Temporal Open-domain Adversarial Overall
RAG Baseline (DPR) 15.7 31.4 15.4 15.4 34.9 26.5
MemInsight (Llama v3 Priority ) 31.3 63.6 23.8 53.4 28.7 44.9
MemInsight (Mistral v1 Priority ) 31.4 63.9 26.9 58.1 36.7 48.9
MemInsight (Claude-3-Sonnet Basic ) 33.2 67.1 29.5 56.2 35.7 48.8
MemInsight (Claude-3-Sonnet Priority ) 39.7 75.1 32.6 70.9 49.7 60.5
Table 2: Results for the RECALL@k=5 accuracy for Embedding-based retrieval for answer generation using LoCoMo dataset.
Dense Passage Retrieval(DPR) RAG model is the baseline. Best results are in bold.
Statistic Count
Total Movies 9687
Avg. Attributes 7.39
Failed Attributes 0.10%
Top-5 AttributesGenre 9662
Release year 5998
Director 5917
Setting 4302
Characters 3603
Table 3: Statistics of attributes generated for the LLM-
REDIAL Movie dataset, which include total number of
movies, average number of attributes per item, number of
failed attributes, and the counts for the most frequent five at-
tributes.
process involves masking the dialogue and ran-
domly selecting n= 200 conversations for eval-
uation to ensure a fair comparison. Each conver-
sational dialogue used is processed by masking
the ground truth labels, followed by a turn cut-off,
where all dialogue turns following the first masked
turn are removed and retained as evaluation labels.
Subsequently, the dialogues are augmented using a
conversation-centric approach to identify relevant
user interest attributes for retrieval. Finally, we
prompt the LLM model to generate a movie recom-
mendation that best aligns with the masked token,
guided by the augmented movies retrieved based
on the user’s historical interactions.
The baseline for this evaluation is the results pre-
sented in the LLM-REDIAL paper (Liang et al.,
2024) which employs zero-shot prompting for rec-
ommendation using the ChatGPT model6. In addi-
tion to the baseline model that uses memory with-
out augmentation.
6https://openai.com/blog/chatgptEvaluation includes direct matches between rec-
ommended and ground truth movie titles using RE-
CALL@[1,5,10] and NDCG@[1,5,10]. Further-
more, to address inconsistencies in movie titles
generated by LLMs, we incorporate an LLM-based
evaluation that assesses recommendations based
on genre similarity. Specifically, a recommended
movie is considered a valid match if it shares the
same genre as the corresponding ground truth label.
Memory Augmentation We initially augment
the dataset with relevant attributes, primarily em-
ploying entity-centric augmentations for memory
annotation, as the memory consists of movies. In
this context, we conduct a detailed evaluation of the
generated attributes to provide an initial assessment
of the effectiveness and relevance of MemInsight
augmentations. To evaluate the quality of the gen-
erated attributes, Table 3 presents statistical data on
the generated attributes, including the five most
frequently occurring attributes across the entire
dataset. As shown in the table, the generated at-
tributes are generally relevant, with "genre" being
the most significant attribute based on its cumu-
lative frequency across all movies (also shown in
Figure 5). However, the relevance of attributes
vary, emphasizing the need for prioritization in
augmentation. Additionally, the table reveals that
augmentation was unsuccessful for 0.1% of the
movies, primarily due to the LLM’s inability to rec-
ognize certain movie titles or because the presence
of some words in the movie titles conflicted with
the LLM’s policy.

ModelAvg. Items
RetrievedDirect Match ( ↑) Genre Match ( ↑) NDCG (↑)
R@1 R@5 R@10 R@1 R@5 R@10 N@1 N@5 N@10
Baseline (Claude-3-Sonnet) 144 0.000 0.010 0.015 0.320 0.57 0.660 0.005 0.007 0.008
LLM-REDIAL Model 144 - 0.000 0.005 - - - - 0.000 0.001
Attribute-Based Retrieval
MemInsight (Claude-3-Sonnet) 15 0.005 0.015 0.015 0.270 0.540 0.640 0.005 0.007 0.007
Embedding-Based Retrieval
MemInsight (Llama v3) 10 0.000 0.005 0.028 0.380 0.580 0.670 0.000 0.002 0.001
MemInsight (Mistral v1) 10 0.005 0.010 0.010 0.380 0.550 0.630 0.005 0.007 0.007
MemInsight (Claude-3-Haiku) 10 0.005 0.010 0.010 0.360 0.610 0.650 0.005 0.007 0.007
MemInsight (Claude-3-Sonnet) 10 0.005 0.015 0.015 0.400 0.600 0.64 0.005 0.010 0.010
Comprehensive
MemInsight (Claude-3-Sonnet) 144 0.010 0.020 0.025 0.300 0.590 0.690 0.010 0.015 0.017
Table 4: Results for Movie Conversational Recommendation using (1) Attribute-based retrieval with Claude-3-Sonnet model
(2) Embedding-based retrieval across models (Llama v3, Mistral v1, Claude-3-Haiku, and Claude-3-Sonnet) (3) Comprehensive
setting using Claude-3-Sonnet that includes ALL augmentations. Evaluation metrics include RECALL, NDCG, and an LLM-
based genre matching metric, with n= 200 andk= 10 . Baseline is Claude-3-Sonnet without augmentation. Best results are in
bold.
Memory Retrieval For this task we evaluate
attribute-based retrieval using the Claude-3-Sonnet
model with both filtered and comprehensive set-
tings. Additionally, we examine embedding-based
retrieval using all other models. For embedding-
based retrieval, we set k= 10 , meaning that 10
memory instances are retrieved (as opposed to 144
in the baseline).
Experimental Results Table 4 shows the re-
sults for conversational recommendation evaluating
comprehensive setting, attribute-based retrieval and
embedding-based retrieval. As shown in the table,
comprehensive memory augmentation tends to out-
perform the baseline and LLM-REDIAL model for
recall and NDCG metrics. For genre match we find
the results to be comparable when considering all
attributes. However, attributed-based filtering re-
trieval still outperforms the LLM-REDIAL model
and is comparable to the baseline with almost 90%
less memory retrieved.
Table 5 presents the results of subjective LLM-
based evaluation for Persuasiveness and Related-
ness. The findings indicate that memory augmen-
tation enhances partial persuasiveness by 10–11%
using both comprehensive and attribute-based re-
trieval, while also reducing unpersuasive recom-
mendations and increasing highly persuasive ones
by 4% in attribute-based retrieval. Furthermore, the
results highlights the effectiveness of embedding-
based retrieval, which leads to a 12% increase
in highly persuasive recommendations and en-
hances all relatedness metrics. This illustrates how
MemInsight enriches the recommendation process
by incorporating condensed, relevant knowledge,
thereby producing more persuasive and related
recommendations. However, these improvements
Raw Dialogues LLM-based Event Summary
Augmentation-based Event Summary Augmented Dialogue
Attribute MiningAttribute Granularity
Turn-Level Session-Level Augmentations  Augmentations Dialogue Evaluation
Augmentation-based
SummaryBaseline
LoCoMo
Ground
Truth
LabelsFigure 3: Evaluation framework for event summarization
with MemInsight, exploring augmentation at Turn and Ses-
sion levels, considering attributes alone or both attributes and
dialogues for richer summaries.
were not reflected in recall and NDCG metrics.
5.3 Event Summarization
We evaluate the effectiveness of MemInsight in
enriching raw dialogues with relevant insights for
event summarization. We utilize the generated an-
notations to identify key events within conversa-
tions and hence use them for event summarization.
We compare the generated summaries against Lo-
CoMo’s event labels as the baseline. Figure 3 illus-
trates the experimental framework, where the base-
line is the raw dialogues sent to the LLM model
to generate an event summary, then both event
summaries, from raw dialogues and augmentation
based summaries, are compared to the ground truth
summaries in the LoCoMo dataset.
Memory Augmentation In this experiment, we
evaluate the effectiveness of augmentation granular-
ity; turn-level dialogue augmentations as opposed
to session-level dialogue annotations. We addition-
ally, consider studying the effectiveness of using
only the augmentations to generate the event sum-
maries as opposed to using both the augmentations
and their corresponding dialogue content.

ModelAvg. Items
RetrievedLLM-Persuasiveness % LLM-Relatedness %
Unpers* Partially Pers. Highly Pers. Not Comp* Comp Match
Baseline (Claude-3-Sonnet) 144 16.0 64.0 13.0 57.0 41.0 2.0
Attribute-Based Retrieval
MemInsight (Claude-3-Sonnet) 15 2.0 75.0 17.0 40.5 54.0 2.0
Embedding-Based Retrieval
MemInsight (Llama v3) 10 11.3 63.0 20.4 19.3 80.1 0.5
MemInsight (Mistral v1) 10 16.3 61.2 18.0 16.3 82.5 5.0
MemInsight (Claude-3-Haiku) 10 1.6 53.0 25.0 23.3 74.4 2.2
MemInsight (Claude-3-Sonnet) 10 2.0 59.5 20.0 29.5 68.0 2.5
Comprehensive
MemInsight (Claude-3-Sonnet) 144 2.0 74.0 12.0 42.5 56.0 1.0
Table 5: Movie Recommendations results (with similar settings to Table 4) using LLM-based metrics; (1) Persuasiveness— %
of Unpersuasive (lower is better), Partially, and Highly Persuasive cases. (2) Relatedness— % of Not Comparable (lower is
better), Comparable, and Exactly Matching cases. Best results are in bold. Comprehensive setting includes ALL augmentations.
Totals may NOT sum to 100% due to cases the LLM model could not evaluate.
Model Claude-3-Sonnet Llama v3 Mistral v1 Claude-3-Haiku
Rel. Coh. Con. Rel. Coh. Con. Rel. Coh. Con. Rel. Coh. Con.
Baseline Summary 3.27 3.52 2.86 2.03 2.64 2.68 3.39 3.71 4.10 4.00 4.4 3.83
MemInsight (TL) 3.08 3.33 2.76 1.57 2.17 1.95 2.54 2.53 2.49 3.93 4.3 3.59
MemInsight (SL) 3.08 3.39 2.68 2.0 2.62 3.67 4.13 4.41 4.29 3.96 4.30 3.77
MemInsight +Dialogues (TL) 3.29 3.46 2.92 2.45 2.19 2.87 4.30 4.53 4.60 4.23 4.52 4.16
MemInsight +Dialogues (SL) 3.05 3.41 2.69 2.24 2.80 3.86 4.04 4.48 4.33 3.93 4.33 3.73
Table 6: Event Summarization results using G-Eval metrics (higher is better): Relevance, Coherence, and Consistency.
Comparing summaries generated with augmentations only at Turn-Level (TL) and Session-Level (SL) and summaries generated
using both augmentations and dialogues (MemInsight +Dialogues) at TL and SL. Best results are in bold.
Experimental Results As shown in Table 6, our
MemInsight model achieves performance compa-
rable to the baseline, despite relying only on dia-
logue turns or sessions containing the event label.
Notably, turn-level augmentations provided more
precise and detailed event information, leading to
improved performance over both the baseline and
session-level annotations.
For Claude-3-Sonnet, all metrics remain compa-
rable, indicating that memory augmentations effec-
tively capture the semantics and knowledge within
dialogues at both the turn and session levels. This
proves that the augmentations sufficiently enhance
context representation for generating event sum-
maries.
To further investigate how backbone LLMs im-
pact augmentation quality, we employed Claude-3-
Sonnet as opposed to Llama v3 for augmentation
while still using Llama for event summarization.
As presented in Table 7, Sonnet augmentations
resulted in improved performance for all metrics,
providing empirical evidence for the effectiveness
and stability of Sonnet in augmentation.
6 Conclusion
This paper introduced MemInsight, an autonomous
memory augmentation method that enhances LLM
agents memory through attribute-based annota-
tions. While maintaining comparable performanceModel G-Eval % ( ↑)
Rel. Coh. Con.
Baseline(Llama v3 ) 2.03 2.64 2.68
Llama v3 + Llama v3 2.45 2.19 2.87
Claude-3-Sonnet + Llama v3 3.15 3.59 3.17
Table 7: Results for Event Summarization using Llama
v3, where the baseline is the model without augmentation as
opposed to the augmentation model (turn-level) using Claude-
3-Sonnet vs Llama v3.
on standard metrics, MemInsight significantly im-
proves LLM-based evaluation scores, highlighting
its effectiveness in capturing semantics and boost-
ing performance across tasks and datasets. Addi-
tionally, attribute-based filtering and embedding
retrieval methods showed promising methods of
utilizing the generated augmentations to improve
the performance of various tasks. Priority augmen-
tation enhancing similarity searches and retrieval.
MemInsight also could be a complement to RAG
models for customized retrievals, integrating LLM
knowledge. Results confirm that attribute-based re-
trieval effectively enriches recommendation tasks,
leading to more persuasive recommendations.
7 Limitations
While the proposed MemInsight model demon-
strates significant potential in enhancing retrieval
and contextual understanding, certain limitations
must be acknowledged. MemInsight relies on the

quality and granularity of annotations generated
using LLMs, making it susceptible to issues such
as hallucinations inherent to LLM outputs. Fur-
thermore, although the current evaluation metrics
provide valuable insights, they may not comprehen-
sively capture all aspects of retrieval and generation
quality, highlighting the need for the development
of more robust and multidimensional evaluation
frameworks.
References
Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry
Evseev, Mikhail Burtsev, and Evgeny Burnaev. 2024.
Arigraph: Learning knowledge graph world mod-
els with episodic memory for llm agents. Preprint ,
arXiv:2407.04363.
Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu,
Monica Sunkara, Yassine Benajiba, and Yi Zhang.
2025. Tremu: Towards neuro-symbolic temporal rea-
soning for llm-agents with memory in multi-session
dialogues. arXiv preprint arXiv:2502.01630 .
Yuki Hou, Haruki Tamoto, and Homei Miyashita. 2024.
“my agent understands me better”: Integrating dy-
namic human-like memory recall and consolidation
in llm-based agents. In Extended Abstracts of the
CHI Conference on Human Factors in Computing
Systems , page 1–7. ACM.
Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo
Zhao, and Hang Zhao. 2023a. Chatdb: Augmenting
llms with databases as their symbolic memory. arXiv
preprint arXiv:2306.03901 .
Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo
Zhao, and Hang Zhao. 2023b. Chatdb: Augment-
ing llms with databases as their symbolic memory.
Preprint , arXiv:2306.03901.
Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu,
Wenqi Shao, and Ping Luo. 2024. Hiagent: Hier-
archical working memory management for solving
long-horizon agent tasks with large language model.
Preprint , arXiv:2408.09559.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017.
Billion-scale similarity search with gpus. Preprint ,
arXiv:1702.08734.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen tau Yih. 2020. Dense passage retrieval
for open-domain question answering. Preprint ,
arXiv:2004.04906.
Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan,
Congying Xia, Kai Chen, and Yuyu Yin. 2024. LLM-
REDIAL: A large-scale dataset for conversational
recommender systems created from user behaviors
with LLMs. In Findings of the Association for Com-
putational Linguistics: ACL 2024 , pages 8926–8939,Bangkok, Thailand. Association for Computational
Linguistics.
Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang
Zhang, Jinjie Gu, and Guannan Zhang. 2023a.
Think-in-memory: Recalling and post-thinking en-
able llms with long-term memory. Preprint ,
arXiv:2311.08719.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023b. G-eval:
Nlg evaluation using gpt-4 with better human align-
ment. Preprint , arXiv:2303.16634.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of llm agents. Preprint , arXiv:2402.17753.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
2024. Memgpt: Towards llms as operating systems.
Preprint , arXiv:2310.08560.
Gustavo Penha, Ali Vardasbi, Enrico Palumbo, Marco
de Nadai, and Hugues Bouchard. 2024. Bridg-
ing search and recommendation in generative re-
trieval: Does one task help the other? Preprint ,
arXiv:2410.16823.
Jihoon Tack, Jaehyung Kim, Eric Mitchell, Jinwoo
Shin, Yee Whye Teh, and Jonathan Richard Schwarz.
2024. Online adaptation of language models with
a memory of amortized contexts. arXiv preprint
arXiv:2403.04317 .
Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding,
Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang,
Tianyu Liu, Bowen Yu, and Junyang Lin. 2025.
Enabling scalable oversight via self-evolving critic.
Preprint , arXiv:2501.05727.
Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang,
and James Zou. 2024a. Mixture-of-agents en-
hances large language model capabilities. Preprint ,
arXiv:2406.04692.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao
Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei,
and Jirong Wen. 2024b. A survey on large language
model based autonomous agents. Frontiers of Com-
puter Science , 18(6).
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong,
and Yangqiu Song. 2024c. Rethinking the bounds of
llm reasoning: Are multi-agent discussions the key?
Preprint , arXiv:2402.18272.
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong,
and Yangqiu Song. 2024d. Rethinking the bounds of
llm reasoning: Are multi-agent discussions the key?
Preprint , arXiv:2402.18272.

Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong,
and Yangqiu Song. 2024e. Rethinking the bounds of
llm reasoning: Are multi-agent discussions the key?
Preprint , arXiv:2402.18272.
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and
Graham Neubig. 2024f. Agent workflow memory.
Preprint , arXiv:2409.07429.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. Preprint , arXiv:2210.03629.
Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen,
Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-
Rong Wen. 2024. A survey on the memory mecha-
nism of large language model based agents. Preprint ,
arXiv:2404.13501.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu
Lin, Yong-Jin Liu, and Gao Huang. 2024a. Ex-
pel: Llm agents are experiential learners. Preprint ,
arXiv:2308.10144.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin,
Yong-Jin Liu, and Gao Huang. 2024b. Expel: Llm
agents are experiential learners. In Proceedings of
the AAAI Conference on Artificial Intelligence , pages
19632–19642.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and
Yanlin Wang. 2023. Memorybank: Enhancing large
language models with long-term memory. Preprint ,
arXiv:2305.10250.
Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Wei-
jie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu,
Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and
Jifeng Dai. 2023. Ghost in the minecraft: Gener-
ally capable agents for open-world environments via
large language models with text-based knowledge
and memory. Preprint , arXiv:2305.17144.A Ethical Consideration
We have thoroughly reviewed the licenses of all
scientific artifacts, including datasets and mod-
els, ensuring they permit usage for research and
publication purposes. To protect anonymity, all
datasets used are de-identified. Our proposed
method demonstrates considerable potential in sig-
nificantly reducing both the financial and environ-
mental costs typically associated with enhancing
large language models. By lessening the need for
extensive data collection and human labeling, our
approach not only streamlines the process but also
provides an effective safeguard for user and data
privacy, reducing the risk of information leakage
during training corpus construction. Additionally,
throughout the paper-writing process, Generative
AI was exclusively utilized for language checking,
paraphrasing, and refinement.
B Autonomous Memory Augmentation
B.1 Attribute Mining
Figure 4 illustrates examples for the two types
of attribute augmentation: entity-centric and
conversation-centric. The entity-centric augmen-
tation represents the main attributes generated for
the book entitled ’Already Taken’, where attributes
are derived based on entity-specific characteristics
such as genre, author, and thematic elements. The
conversation-centric example illustrates the aug-
mentation generated for a sample two turns dia-
logue from the LLM-REDIAL dataset, highlight-
ing attributes that capture contextual elements such
as user intent, motivation, emotion, perception, and
genre of interest.
Furthermore, Figure 5 presents an overview of
the top five attributes across different domains in
the LLM-REDIAL dataset. These attributes repre-
sent the predominant attributes specific to each do-
main, highlighting the significance of different at-
tributes in augmentation generation. Consequently,
the integration of priority-based embeddings has
led to improved performance.
C Embedding-based Retrieval
In the context of embedding-based memory re-
trieval, movies are augmented using MemInsight,
and the generated attributes are embedded to re-
trieve relevant movies from memory. Two main
embedding methods were considered:

Figure 4: An example of entity-centric augmentation for the book ’Already Taken’, and a conversation-centric
augmentation for a sample dialogue from the LLM-REDIAL dataset.
Sports Augmentation Attributes 
Type
Features Material CategoryBrandPurposeColor
PortabilityUse
equipment
TypeBrand Color
Compatibility ConnectivityModelMaterial FeaturesWeight
CategoryAuthor GenreTitle
PublisherSetting
Publication DateThemes
CharactersNumberof PagesSeries14000
12000
10000
8000
6000
4000
2000
12000
400014000
8000
6000
2000100000
0500010000150002000025000300003500040000
8000
0Frequency6000
4000
2000Movies Augmentation Attributes 
GenreRelease YearDirectorSetting
CharactersPlot
RuntimeActors
LanguageWriter10000
0
AttributesFrequency
Electronics Augmentation Attributes Books Augmentation Attributes 
Figure 5: Top 10 attributes by frequency in the LLM-REDIAL dataset across domains (Movies, Sports Items,
Electronics, and Books) using MemInsight Attribute Mining. Frequency indicates how often each attribute was
generated to augment different movies.
(1) Averaging Over Independent Embeddings
Each attribute and its corresponding value in the
generated augmentations is embedded indepen-
dently. The resulting attribute embeddings are then
averaged across all attributes to generate the final
embedding vector representation, as illustrated in
Figure 6 which are subsequently used in similarity
search to retrieve relevant movies.
(2) All Augmentations Embedding In this
method, all generated augmentations, including
all attributes and their corresponding values, are en-
coded into a single embedding vector and stored for
retrieval as shown in Figure 6. Additionally, Fig-
ure 7 presents the cosine similarity results for both
methods. As depicted in the figure, averaging over
all augmentations produces a more consistent and
reliable measure, as it comprehensively captures
all attributes and effectively differentiates between
similar and distinct characteristics. Consequently,this method was adopted in our experiments.
D Question Answering
D.1 Prompts
Table 8 outlines the prompts used in the Question
Answering task for generating augmentations in
both questions and conversations.
E Conversational Recommendation
E.1 Prompts
Table 9 presents the prompts used in Conversational
Recommendation for movie recommendations, in-
corporating both basic and priority augmentations.
E.2 Evaluation Framework
Figure 8 presents the evaluation framework for the
Conversation Recommendation task. The process
begins with (1) augmenting all movies in memory

[Attribute 1]<v alue> [Attribute 2]<v alue> [Attribute 3]<v alue> [Attribute 4]<v alue>Movie  Augmentations
Embedding-based RetrievalEmbedding Model 
Embedding V ectorAveraging(a) Averaging over Independent Embeddings (b) All Augmentations Embedding
Embedding Model 
Embedding V ectorMovieFigure 6: Embedding methods for Embedding-based retrieval methods using generated Movie augmentations
including (a) Averaging over Independent Embeddings and (b) All Augmentations Embedding.
Movie 1: The Departed
Movie 2: Shutter Island
Movie 3: The HobbitMovie 1: The Departed
Movie 2: Shutter Island
Movie 3: The Hobbit
(a) Averaging over Independent Embeddings (b) All Augmentations Embedding
Figure 7: An illustrative example of augmentation embedding methods for three movies: (1) The Departed, (2)
Shutter Island, and (3) The Hobbit. Movies 1 and 2 share similar attributes, whereas movies 1 and 3 differ. Te top 5
attributes of every movie were selected for a simplified illustration.
using entity-centric augmentations to enhance re-
trieval effectiveness. (2) Next, all dialogues in the
dataset are prepared to simulate the recommenda-
tion process by masking the ground truth labels
and prompting the LLM to find the masked labels
based on augmentations from previous user inter-
actions. (3) Recommendations are then generated
using the retrieved memory, which may be attribute-
based—for instance, filtering movies by specific
attributes such as genre or using embedding-based
retrieval. (4) Finally, the recommended movies are
evaluated against the ground truth labels to assess
the accuracy and effectiveness of the retrieval and
recommendation approach.
E.3 Event Summarization
E.3.1 Prompts
Table 10 presents the prompt used in Event Sum-
marization to augment dialogues by generating rel-
evant attributes. In this process, only attributes
related to events are considered to effectively sum-
marize key events from dialogues, ensuring a fo-
cused and structured summarization approach.F Qualitative Analysis
Figure 9 illustrates the augmentations generated
using different LLM models, including Claude-
Sonnet, Llama, and Mistral for a dialogue turn
from the LoCoMo dataset. As depicted in the fig-
ure, augmentations produced by Llama include hal-
lucinations, generating information that does not
exist. In contrast, Figure 10 presents the augmen-
tations for the subsequent dialogue turn using the
same models. Notably, Claude-Sonnet maintains
consistency across both turns, suggesting its stable
performance throughout all experiments. While
Mistral model tend to be less stable as it included
attributes that are not in the dialogue.

Figure 8: Evaluation Framework for Conversation Recommendation Task.
Figure 9: Augmentation generated on a Turn-level for a sample dialogue turn from the LoCoMo dataset using
Claude-3-Sonnet, Llama v3 and Mistral v1 models.
Figure 10: Augmentations generated for the turn following the turn in Figure 9
using Claude-3-Sonnet, Llama v3 and Mistral v1 models. Hallucinations are presented in red.

Question Augmentation
Given the following question, determine what are the main inquiry attribute to look for and the person the question is for.
Respond in the format: Person:[names]Attributes:[].
Basic Augmentation
You are an expert annotator who generates the most relevant attributes in a conversation. Given the conversation below,
identify the key attributes and their values on a turn by turn level.
Attributes should be specific with most relevant values only. Don’t include speaker name. Include value information
that you find relevant and their names if mentioned. Each dialogue turn contains a dialogue id between [ ]. Make sure
to include the dialogue the attributes and values are extracted form. Important: Respond only in the format [{speaker
name:[Dialog id]:[attribute]<value>}].
Dialogue Turn:{}
Priority Augmentation
You are an expert dialogue annotator, given the following dialogue turn generate a list of attributes and values for relevant
information in the text.
Generate the annotations in the format: [attribute]<value>where attribute is the attribute name and value is its corre-
sponding value from the text.
and values for relevant information in this dialogue turn with respect to each person. Be concise and direct.
Include person name as an attribute and value pair.
Please make sure you read and understand these instructions carefully.
1- Identify the key attributes in the dialogue turn and their corresponding values.
2- Arrange attributes descendingly with respect to relevance from left to right.
3- Generate the sorted annotations list in the format: [attribute]<value>where attribute is the attribute name and value is
its corresponding value from the text.
4- Skip all attributes with none vales
Important: YOU MUST put attribute name is between [ ] and value between <>. Only return a list of [at-
tribute]<value>nothing else. Dialogue Turn: {}
Table 8: Prompts used in Question Answering for generating augmentations for questions. Also, augmentations for
conversations, utilizing both basic and priority augmentations.
Basic Augmentation
For the following movie identify the most important attributes independently. Determine all attributes that describe the
movie based on your knowledge of this movie. Choose attribute names that are common characteristics of movies in
general. Respond in the following format: [attribute]<value of attribute>. The Movie is: {}
Priority Augmentation
You are a movie annotation expert tasked with analyzing movies and generating key-attribute pairs. For the following
movie identify the most important. Determine all attribute that describe the movie based on your knowledge of this
movie. Choose attribute names that are common characteristics of movies in general. Respond in the following format:
[attribute]<value of attribute>. Sort attributes from left to right based on their relevance. The Movie is:{}
Dialogue Augmentation
Identify the key attributes that best describe the movie the user wants for recommendation in the dialogue. These
attributes should encompass movie features that are relevant to the user sorted descendingly with respect to user interest.
Respond in the format: [attribute]<value>.
Table 9: Prompts used in Conversational Recommendation for recommending Movies utilizing both basic and
priority augmentations.
Dialogue Augmentation
Given the following attributes and values that annotate a dialogue for every speaker in the format [attribute]<value>,
generate a summary for the event attributes only to describe the main and important events represented in these
annotations. Refrain from mentioning any minimal event. Include any event-related details and speaker. Format: a bullet
paragraph for major life events for every speaker with no special characters. Don’t include anything else in your response
or extra text or lines. Don’t include bullets. Input annotations: {}
Table 10: Prompt used in Event Summarization to augment dialogues
# Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents

**Authors**: Yuxin Wang, Paul Thomas, Zhiwei Yu, Yuan Gao, Saeed Hassanpour, Soroush Vosoughi, Robert Sim, Nick Craswell

**Published**: 2026-06-24 03:45:45

**PDF URL**: [https://arxiv.org/pdf/2606.25361v1](https://arxiv.org/pdf/2606.25361v1)

## Abstract
Prior research on memory mechanism in RAG-based conversational system has emphasized how memory is stored and retrieved. However, far less is known about how memories with different functional roles influence response quality. Specifically, how they shape an agent's responses under varying conversational contexts and whether they lead to substantively different response behaviors. Existing evaluations in conversational system are also largely reference-based, insufficiently capturing the nuances in responses that may address users' preferences differently. In this work, we probe the impact of different memory types in shaping agents' responses. We present a fine-grained taxonomy of conversational memory, classify retrieved memories into different role types, and design a user-centric evaluation framework that simulates user perspectives. Through comparative experiments on long-term datasets and frontier LLMs, our analysis reveal many differentiated effects of memories: e.g., clarifying memory improves responses' factual accuracy and constraint awareness, making them more correct and personalized; irrelevant memory reduces topic relevance and degrades constraint awareness. Despite the power of frontier LLMs, these findings shed light on how different memory types can be leveraged to produce more personalized responses and inspire further research in this direction.

## Full Text


<!-- PDF content starts -->

Memory Makes the Difference: Evaluating How Different Memory Roles
Shape Conversational Agents
Yuxin Wang1*Paul Thomas2Zhiwei Yu†Yuan Gao2Saeed Hassanpour1
Soroush Vosoughi1Robert Sim2Nick Craswell2
1Dartmouth College2Microsoft
Abstract
Prior research on memory mechanism in RAG-
based conversational system has emphasized
how memory is stored and retrieved. How-
ever, far less is known about how memories
with different functional roles influence re-
sponse quality. Specifically, how they shape
an agent’s responses under varying conversa-
tional contexts and whether they lead to sub-
stantively different response behaviors. Exist-
ing evaluations in conversational system are
also largely reference-based, insufficiently cap-
turing the nuances in responses that may ad-
dress users’ preferences differently. In this
work, we probe the impact of different mem-
ory types in shaping agents’ responses. We
present a fine-grained taxonomy of conversa-
tional memory, classify retrieved memories into
different role types, and design a user-centric
evaluation framework that simulates user per-
spectives. Through comparative experiments
on long-term datasets and frontier LLMs, our
analysis reveal many differentiated effects of
memories:e.g., clarifying memory improves re-
sponses’ factual accuracy and constraint aware-
ness, making them more correct and person-
alized; irrelevant memory reduces topic rele-
vance and degrades constraint awareness. De-
spite the power of frontier LLMs, these find-
ings shed light on how different memory types
can be leveraged to produce more personal-
ized responses and inspire further research in
this direction. Our code is available at: https:
//github.com/audreycs/MemoryEffect.
1 Introduction
Conversational memory provides needed context
for grounded responses and accurate query under-
standing (Radlinski and Craswell, 2017), and it
now is a core property of contemporary chat sys-
tems like ChatGPT (OpenAI, 2025b) and Google
*This work is done during the author’s internship at Mi-
crosoft. Correspondence to:yuxin.wang.gr@dartmouth.edu
andnickcr@microsoft.com.
†Independent researcher.
I’m going to Seattle next week to visit a friend.Nice! Seattle is great place to have fun.My friend invited me for Thanksgiving. He’s into music, painting, and hiking. We’ll probably do some of those.Sure, [some genral practices], plus a Seattle twist: December is usually wet, so check the forecast the night before and bring a rain jacket.Can you give me some tips for a long hike?…It’s been so hard to catch the program director lately.Ye ah , di re ct or s can  be  ve ry  bus y.She once told me she’s usually only free on Friday morning.I finally saw the director. Can you schedule a meeting with her tomorrow on my calendar?Sure, does tomorrow at 2 PM sound good to you?…
agent utilizes informative memory and gives helpful response agent utilizes distracting memory and gives inappropriate response( 
 inappropriate)( 
 informative and helpful)I can also catch her at the end of the 2pm seminar, but only for a minute.Figure 1: (Left panel) Example of retrieved memory
making response more helpful; (Right panel) Example
of retrieved memory making response inappropriate.
Gemini (AI and DeepMind, 2025). At high level,
conversation history is logged into an external
memory bank to reserve past information. During
response, chat agents can retrieve relevant pieces to
surface useful memory. Such retrieval-augmented
generation (RAG) pipeline has served as a stan-
dard approach especially in long-range conversa-
tion, overcoming the context length limits of Large
Language Models (LLMs) to maintain response
coherence and accuracy (Fan et al., 2024).
Existing work on conversational RAG has
largely centered on storage and retrieval strategy,
i.e., memory structure, retrieval size, and granu-
larity (Xu et al., 2022; Ye et al., 2024; Maharana
et al., 2024; Zhong et al., 2024; Wu et al., 2025;
Pan et al., 2025; Tan et al., 2025a). They typi-
cally retrieve a set of stored pieces, concatenate
or compress them into the conversational context,
then evaluate the aggregated performance of gen-
eration system. However, this common practice
overlooks the distinct role of properties and impact
of individual retrieved pieces. Human intuitively
recall useful memory to formulate responses, but
1arXiv:2606.25361v1  [cs.CL]  24 Jun 2026

when it comes to RAG systems, it is uncertain
if retrieved memory which are semantically close
are actually useful. The effect of retrieval part of-
ten behaves like a black box and hard to interpret
or control (Chen et al., 2024). As Fig. 1 demon-
strates, some supplementary memory can enrich
the response and make it more helpful; while other
memories can be distracting and make the response
inappropriate and undesired. In this work, our cen-
tral hypothesis is thatthe type of memory mat-
ters: different memory types elicit different agent
responses, and these variations can lead to different
user perceptions of quality. Therefore, understand-
ing the qualitative effect of each memory type and
whether agents can leverage a bundle of retrieved
memories worth systematic investigation. Building
on this hypothesis, we aim to quantify the impact
and find their qualitative property. We introduce
the first taxonomy of conversational memory types,
determined by their roles when the agent reasons
on memory. Besides, existing work is usually lim-
ited in reference-based evaluation. Prior work leans
on question answering (QA) datasets provided with
unambiguous gold answers (Anantha et al., 2021;
Xu et al., 2022; Pan et al., 2025; Maharana et al.,
2024; Chen et al., 2025). Their reference-based
evaluation privileges measuring factual accuracy
(e.g., F1, ROUGE) while ignore other qualities of
responses like user intent awareness and informa-
tiveness, which are critical for dialogic intelligence
but difficult to measure using only references.
In this work we present analyses of how differ-
ent memory types shape LLM agents’ responses,
paired with a holistic, user-centric evaluation frame-
work covering various aspects. Our taxonomy clas-
sifies conversational memory pieces into five types
— answer, clarifying, enriching, distracting, and
irrelevant, judged with awareness of query and rel-
evant context instead of merely the gold answer
in previous analyses (Cuconasu et al., 2024). We
conducted comparative experiments on two long-
term, multi-session conversational datasets over
three popular LLMs. Specifically, we compare var-
ied memory settings with (i) answer memory only
and (ii) parametric memory only, to study how the
presence of individual types and their combinations
affects responses quality. To better differentiate re-
sponse quality, we design a user-centric evaluation
proxy that conditions on user intent and prior con-
versation context, yielding sensitivity to response
relevance, informativeness, and other properties.
Our experiment results reveal many interestingpatterns. For examples, we find frontier LLMs are
capable of selectively leveraging contextual infor-
mation with considerable nuance; agents are less
likely to ground their responses as retrieval size
increases. As for memory types, clarifying mem-
ory plays a key role in improving answer accuracy
and contextual grounding, and even in isolation it
can induce perfect responses. In contrast, irrele-
vant memory tends to reduce topic relevance and
weaken contextual awareness, against the founded
positive impact of them on small-scale LLMs like
Llama2 and Phi-2 in previous work (Cuconasu
et al., 2024). These also confirm our hypothesis on
the impact of memory types. Our taxonomy and
user-centric evaluation are general-purpose and can
be applied to other conversational corpora and set-
tings. Our main contributions in this work are the
following:
•We hypothesize the impact of memory types
on agent’s responses, and conducted extensive
comparative experiment on frontier LLMs to
investigate.
•We propose the first fine-grained taxonomy of
conversational memory based on functional
roles, together with a holistic framework with
diverse metrics for generation evaluation.
•Our experiments qualitatively reveal the dis-
tinct impacts of different memory, with sev-
eral effects diverging from prior findings on
small-scale LLMs.
2 Related Work
2.1 Memory-enabled Conversational System
Conversational systems typically frame memory
recall as a store-retrieve-generate pipeline, reflect-
ing flexible memory management and working
context (Gao et al., 2023). Recent research ef-
forts on LLM-powered conversational systems
have largely focused on memory structure (Packer
et al., 2023; Rasmussen et al., 2025; Chhikara
et al., 2025), retrieval strategy (Tan et al., 2025b;
Li et al., 2025; Hu et al., 2025a), and memory
update governance (Tan et al., 2025b; Rezazadeh
et al., 2025; Wang et al., 2025). Their retrieval
often use sparse or conversational dense retriev-
ers, such as BM25 (Robertson et al., 2009) and
Contriever (Izacard et al., 2021). To stress-test
these systems in realistic settings, several work tar-
get long-term, multi-session dialogue, including
MT-Bench-Plus (Lu et al., 2023), LOCOMO (Ma-
harana et al., 2024), and LongMemEval (Wu et al.,
2

2025), InfiniteBench (Zhang et al., 2024).
2.2 Analysis of Retrieval Pieces Type
Several recent studies have examined how irrele-
vant or distracting retrieved pieces affect QA perfor-
mance, primarily focusing on smaller scale LLMs.
Cuconasu et al. (2024) give pioneer analysis of
how the type and position of retrieved documents
influence model’s answers. They classified docu-
ments into gold,relevant ,distracting , and random
based on semantic similarity and the presence of
gold answer. They highlighted that random doc-
uments can sometimes raise accuracy. Jin et al.
(2025) investigate distracting documents that are
relevant to the query but not the gold answer and
find consistent harm to QA performance across
diverse models. Later, Amiraz et al. (2025) pro-
pose a measure to quantify the distracting effect
of distracting documents. Collectively, these work
characterize the impact of irrelevant and distracting
evidence in document QA, but give less attention to
conversational settings. Inspired by their analysis,
we adapt and refine their types to conversational
scenario and shift the focus from document QA
to memory-enabled dialogue, where the responses
require more nuanced measurement and multiple
forms of “relevant memory” (e.g., clarifying, en-
riching) also need to be considered.
2.3 Evaluation of Conversational Agent
Conventional evaluation of conversational system
scores responses against references with n-gram or
embedding metrics like F1, ROUGE (Lin, 2004),
or BERTScore (Zhang* et al., 2020). Newer work
on long-term, memory-enabled conversations also
tracks retrieval accuracy by measuring if particular
pieces are retrieved (i.e., needle in the haystack) via
metrics like Recall@ k(Maharana et al., 2024; Wu
et al., 2025). As the advance of LLM-as-a-judge,
LLM-based judges start to be widely adapted for
holistic evaluation of chat quality (Bai et al., 2024;
Zhao et al., 2025; Chhikara et al., 2025). By set-
ting well-crafted rubrics, LLM judges are flexible
to measure diverse dimensions even without ref-
erences. Mem0 (Chhikara et al., 2025) prompts
LLM judge to compare responses with references,
and MT-Bench-101 (Bai et al., 2024) uses pairwise
comparison of two candidate responses for ranking.
PersonaLens (Zhao et al., 2025) equips judge with
user profiles and rich interaction history to further
measure if responses address user preferences.Memory Type (Abbr.) Definition
Parametric memory (P) Exists in LLMs’ parameters through training
Answer memory (A)Contains the information of the desired answer for
user’s query
Clarifying memory (C)Resolves the identity of the asked-for entity or
event in the query
Enriching memory (E)Makes the final answer more informative or help-
ful
Distracting memory (D) Seems to be relevant but is actually misleading
Irrelevant memory (I)Has no meaningful relevance to the user’s query
(off-topic)
Table 1: Our taxonomy of conversational memory types.
3 Taxonomy of Conversational Memory
Retrieved memories form a semantic neighbor-
hood around the query by RAG mechanism, and
the agent’s reasoning can be viewed as selectively
traversing retrieved memory pieces to assemble the
information needed for a response, as illustrated
in Fig. 2. Based on this intuition and existing doc-
ument RAG classification (Cuconasu et al., 2024;
Amiraz et al., 2025), we classify conversational
memory according to the functional roles it plays
during the agent’s reasoning process into five fine-
grained types:answer,clarifying,enriching,dis-
tracting, andirrelevant. Tab. 1 shows their defini-
tions. As shown in Fig. 2, to correctly answer the
query, the agent needs clarifying memory to iden-
tify the referenced “manager”, then infer answer
from the answer memory. Ideally, the agent needs
enriching memory to add more helpful details, and
avoids distracting memory to mitigate false answer.
The remaining off-topic memory is considered as
irrelevant , which should be neutral noise. Each
memory is assigned a single type per query. We ac-
knowledge that, in practice, a memory may plausi-
bly contain pieces serving multiple roles. To enable
controlled analysis and clear attribution of effects,
we adopt single-label assignment as a deliberate
methodological simplification. To resolve over-
laps consistently, we apply a labeling precedence:
answer→clarifying→enriching→distracting
→irrelevant for disambiguation. A practical moti-
vation for the precedence is to encourage sufficient
representation of relatively sparse types in realis-
tic retrieval pools. Importantly, it does not encode
a hierarchy of importance or quality of them on
response generation. Besides conversational mem-
ory, we also includeparametricmemory which
is LLM’s inherent knowledge learned during its
training.
To identify types, prior work (Cuconasu et al.,
2024; Amiraz et al., 2025) flags distracting or ir-
3

Query: When will my manager be available tomorrow? “my manager is <Person A> …”“<Person B> is available between … everyday”“meeting with <Person A> must be scheduled in advance…”“this team has 20 engineers…”“<Person A> will have meetings tomorrow on …”distracting memory
irrelevant memoryanswer memoryenriching memoryclarifying memoryFigure 2: Illustration of relations between memory
pieces and user query in agent’s reasoning space. Links
stand for sematic connection between two.
relevant retrieved content by string-matching with
the gold answer. While it is efficient and effec-
tive for short and unambiguous targets (e.g., a
name or location), it falls short for open-ended
dialogue responses which are typically more expan-
sive and lexically diverse. Therefore, we instead
use a context-aware LLM-as-a-judge to simulate
the user and classify the retrieved memory pieces.
For each candidate memory, the judge receives the
query, the candidate piece, and the top 50retrieved
memory as the relevant interaction history. This
allows the judge to better understand the query
and user’s intent. Fig. 8 presents the details where
we adapted the prompt schema from Thomas et al.
(2024) which has been shown to be highly effec-
tive. We use GPT-5-thinking model (OpenAI,
2025a) as the judge. As memory classification is
the cornerstone for following analyses, we also
conducted human verification of the classification
results, which shows LLM judge achieves human-
level reliability (Appendix B).
4 User-centric Response Evaluation
To better capture the sensitivity of responses to
different context, we design a holistic evaluation
framework which is user-centric and context-aware
(Tab. 2). This framework spans three major aspects
(Accuracy, Relevance, and Informativeness) and
each aspect contains finer metrics. These aspects
are designed by integrating established evaluation
framework (Liang et al., 2023; Maurya et al., 2025)
and adapted to conversational scenario. At a high
level, Accuracy captures factual correctness of re-
sponses; Relevance checks alignment with user in-
tent and preferences; Informativeness assesses use-
fulness beyond the minimal answer. Note that con-
ventional reference-based accuracy metric such as
ROUGE is subsumed by Answer Accuracy, whichtests whether the response contains the desired an-
swer. We additionally include Memory Coverage
and ROUGE-1 Recall in the framework. Tab. 2 also
indicates which metrics require prior conversation
history (“context-aware” column). For instance,
World Knowledge Accuracy and Internal Consis-
tency can be judged without context.
For robustness, most metrics except Complete-
ness are designed with a binary scale rather than a
fine-grained rating scale. There is work showing
binary judgments yield higher accuracy and relia-
bility for LLM judges (Dong et al., 2024; Alhawasi
and Youssef, 2024), and finer-grade rating can in-
troduce subjectivity in scale design (Bansal et al.,
2024; Wang et al., 2024). Thus, we believe binary
scale is sufficient for analyses in this work. We use
GPT-o3 as the LLM judge, and metrics under the
same aspect are judged in group.
5 RAG Pipeline Implementation
Our implementation of conversational RAG fol-
lows the standard RAG pipeline. Some key terms
in our context:
ConversationThe complete set of interactions be-
tween one user and the chat agent. Each user has
a separate conversation, which may span multiple
sessions over time.
SessionA continuous interaction episode between
a user and the agent. A session typically focuses
on a single topic.
Utterance and TurnOne utterance is a single text
input from either the user or the agent. A turn is
a back-and-forth exchange consisting of one user
utterance followed by one agent utterance.
ContextFor each user query, the context is the
relevant prior memory which is reachable to the
agent. We refer retrieved memories as context.
5.1 Two Long-Term Conversational Datasets
We use two long-term, multi-session conversation
datasets for our analyses: LongMemEval-m (Wu
et al., 2025) and Long-MT-Bench+ (Zheng et al.,
2023), with statistics in Tab. 3. LongMemEval-
m contains synthetic conversations paired with
human-designed queries. Long-MT-Bench+ con-
tains human-written conversations and GPT-
generated long-range queries. Both datasets fea-
ture extensive conversational memory and tar-
get on evaluating memory-related user queries.
LongMemEval-m additionally includes metadata
such as session timestamps and evidence memories
4

Aspect Metric Description Range Context-aware
AccuracyWorld Knowledge Accuracy of the response with respect to general facts and world knowledge{0,1}✗
Answer Accuracy Agreement of the response with the gold answer{0,1}✗
Internal Coherence Logical consistency of the response within itself{0,1}✗
RelevanceTopic Relevance Whether the response directly addresses the query{0,1}✓
Constraint Awareness Whether the response satisfies constraints stated or implied in prior interactions{0,1}✓
InformativenessCompleteness Extent to which the response covers relevant information for the query{0,1,2}✓
Actionability Whether the response provides concrete details or clear guidance{0,1}✗
Memory Coverage Number of retrieved memory pieces whose information is reflected in the responseZ+✓
ROUGE-1 Recall Proportion of unigrams in the gold answer that appear in the response[0,1]✗
Table 2: User-centric evaluation framework
LongMemEval-m Long-MT-Bench+
total # conversation 500 11
total # query 500 288
avg. # query/conversation 1 26.2
avg. # session/conversation 501.9 4.9
avg. # utterance/conversation 9.8 26.7
avg. # token/conversation 776,472.4 14,000.9
Table 3: Statistics of two long-term conversation
datasets for experiment. Token counts are measured
as space-separated chunks.
for each query.
5.2 Memory Storage and Retrievers
We use one memory bank for all sessions under the
same conversation. When the agent responds to a
query, all past sessions are reachable, but different
conversations cannot access each other’s memory.
Following LongMemEval, we implement the mem-
ory bank as key-value pairs {<key>: <value>},
where the key is used for retrieval indexing and the
value returns the original content. Westore each
turnas one pair; both the key and value are the orig-
inal turn in our implementation, although the key
can be customized (e.g., a summary or a list of key-
words). When timestamps are available, we also
record them in the values. For retrieval, we experi-
ment with three methods: BM25, Contriever, and
OpenAI’s text-embedding-3-small (Neelakan-
tan et al., 2022). We retrieve top 100 memory
pieces with highest similarity. On average, a mem-
ory piece contains 317tokens in LongMemEval-m
and216tokens in Long-MT-Bench+, so 100pieces
already constitute a very large context.
5.3 Agent Models and Response Generation
We especially evaluate three large scale and pop-
ular LLMs as agents: GPT-4o-2024-08-06 (Hurst
et al., 2024) (we call GPT-4o for short in later
parts), Gemini-2.0-Flash (DeepMind, 2024),and Llama-3.3-70B (Dubey et al., 2024). We
select the top kpieces ( k∈ {0,5,10,20,50,100} )
from the 100retrieved pieces to vary context size.
When timestamps are available, the kpieces are
ordered chronologically. We concatenate them
and truncate as needed to fit the model’s context
window. The full prompt for response generation
shown in Fig. 7 in Appendix. Note that when
k= 0 , the agent relies on its parametric memory
to respond. All agents are set with the minimum
temperature to ensure near-deterministic output.
5.4 Response Evaluation
We use GPT-o3 with high reasoning effort as the
judge. Different rubrics are applied to metrics
across different aspects. We carefully designed
the rubrics and performed sanity check on a sample
of queries to validate the quality of the judgments,
iteratively refining the rubrics to align the judge’s
scores with human evaluations (Appendix C). Par-
ticularly, for the Topic Relevance metric, refusal
responses such as “I am unable to answer ...” re-
ceive a score of0.
6 General Performance On Large LLMs
First, we analyze general performance on each met-
ric over different settings (agents, context size, re-
trievers), and we examine agents’ memory selectiv-
ity by type during response generation.
6.1 Performance Over Datasets, Agents,
Context Size, and Retrievers
Fig. 3 shows the average scores over queries with
varied the context size (top k) on different agents.
Fig. 4 shows the average scores for different retriev-
ers. Several notable patterns we find:
Context Size-Performance Trade-offPerfor-
mance on metrics such asAnswer Accuracyand
Constraint Awarenessgenerally rises as the con-
text size kincreases, then declines. This sug-
5

0 5 10 20 50 1000.70.80.9Average Score
World Knowledge (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
Answer Accuracy (LongMemEval-m)
0 5 10 20 50 1000.900.951.00Average Score
Internal Coherence (LongMemEval-m)
0 5 10 20 50 1000.40.60.8Average Score
Topic Relevance (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
Constraint Awareness (LongMemEval-m)
0 5 10 20 50 1001.01.5Average Score
Completeness (LongMemEval-m)
0 5 10 20 50 1000.60.70.8Average Score
Actionability (LongMemEval-m)
0 5 10 20 50 100123Average Score
Memory Coverage (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
ROUGE-1 Recall (LongMemEval-m)
0 5 10 20 50 1000.850.900.95Average Score
World Knowledge (Long-MT-Bench+)
0 5 10 20 50 1000.40.60.8Average Score
Answer Accuracy (Long-MT-Bench+)
0 5 10 20 50 1000.940.960.981.00Average Score
Internal Coherence (Long-MT-Bench+)
0 5 10 20 50 1000.60.8Average Score
Topic Relevance (Long-MT-Bench+)
0 5 10 20 50 1000.40.60.8Average Score
Constraint Awareness (Long-MT-Bench+)
0 5 10 20 50 1001.01.5Average Score
Completeness (Long-MT-Bench+)
0 5 10 20 50 1000.60.70.8Average Score
Actionability (Long-MT-Bench+)
0 5 10 20 50 10024Average Score
Memory Coverage (Long-MT-Bench+)
0 5 10 20 50 1000.50.60.7Average Score
ROUGE-1 Recall (Long-MT-Bench+)
Gemini-2.0-Flash
GPT-4o-2024-08-06
Llama-3.3-70BFigure 3: Average metric scores with error bars over context-aware or reference-aware metrics of three retrieval
methods on LongMemEval-m and Long-MT-Bench+ datasets. Results are based on GPT-4o-2024-08-06 agent.
X-axis is the context size used in the prompt (topkmemory pieces).
0 5 10 20 50 1000.20.40.6Average Score
Answer Accuracy (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
Constraint Awareness (LongMemEval-m)
0 5 10 20 50 1001.01.5Average Score
Completeness (LongMemEval-m)
0 5 10 20 50 100123Average Score
Memory Coverage (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
ROUGE-1 Recall (LongMemEval-m)
0 5 10 20 50 1000.40.60.8Average Score
Answer Accuracy (Long-MT-Bench+)
0 5 10 20 50 1000.40.60.8Average Score
Constraint Awareness (Long-MT-Bench+)
0 5 10 20 50 1001.001.251.501.75Average Score
Completeness (Long-MT-Bench+)
0 5 10 20 50 1002345Average Score
Memory Coverage (Long-MT-Bench+)
0 5 10 20 50 1000.50.60.7Average Score
ROUGE-1 Recall (Long-MT-Bench+)
Figure 4: Average scores over context-aware or reference-aware metrics of three retrieval methods on LongMemEval-
m and Long-MT-Bench+ datasets. Results are based on GPT-4o-2024-08-06 agent. X-axis is the context size used
in the prompt (topkmemory pieces).
gests that excessive context can distract the agents,
echoing prior findings on information overload ef-
fects (Maharana et al., 2024; Pan et al., 2025; Wu
et al., 2025). Among the three agents, GPT-4o
shows the strongest robustness to long contexts.
World Knowledge Accuracy and Constraint
Awareness DeclineIn Fig. 3, when agents respond
with only parametric memories ( k= 0 ), they re-
ceive higher World Knowledge Accuracy than with
retrieval. Our closer inspection suggests that with
added conversational context, agents generate more
claims and contextualized statements that judges
penalize when unsupported, reducing this score.
This pattern indicates that while external evidence
can ground responses, it also amplifies hallucina-
tion risks, consistent with recent studies (Jin et al.,
2024; Hu et al., 2025b). Constraint Awareness re-
sults also show degradation once context exceeds a
certain size, even when Answer Accuracy remains
relatively stable (e.g., on Long-MT-Bench+). This
suggests the importance of careful triage of histor-ical context, especially in preference-heavy tasks
such as recommendation.
Memory Coverage CapResults on Memory Cov-
erage and Completeness, which evaluate how much
of the retrieved memory is reflected in the response,
show a clear upper bound. Coverage peaks when
k∈[5,50] and then decline. Top-ranked pieces
tend to carry the most useful information, while
later ones are often less relevant. As context size
grows, the inclusion of irrelevant pieces distracts
the agent from leveraging the useful ones, leading
to a drop in Memory Coverage.
Retriever ComparisonTo compare different re-
trievers, Fig. 4 shows performances on key met-
rics that require retrieval or references. Overall,
OpenAI’s text-embedding-3-small has the edge
over other two especially on LongMemEval-m and
smaller k, indicating it ranking useful memory
higher. BM25 surprisingly remains a competitive
baseline despite its simplicity, underscoring that
lexical match still finds strong anchors. Their per-
6

Top 5 Top 10 Top 20 Top 50 Top 1000.00.20.40.60.81.0Memory Coverage Rate
0.82 0.840.82 0.800.77 0.75
0.67 0.67
0.58 0.580.56
0.49 0.48 0.48 0.47
0.37
0.21 0.19 0.210.18
0.110.08 0.07 0.06 0.06Answer Memory
Clarifying MemoryEnriching Memory
Distracting MemoryIrrelevant MemoryFigure 5: Memory coverage rate of different memory
types under different retrieval size k. Results are on
GPT-4o.
formances converge when retrieving enough mem-
ory.
6.2 Utilization of Different Memory Types
We also measure how often each memory type is
covered in the response as the context size kvaries.
Based on our taxonomy of memory type, ideally
the irrelevant memory should be ignored and an-
swer memory should be always included. If this is
supported from experiment, it also serves a sanity
check for our taxonomy design. To measure this,
for a given type, let mbe the number of its pieces
among the top kretrieved (theseennumber) and n
be the number echoed in the response (theutilized
number), then the coverage rate is calculated as
n/m — percentage of the seen pieces being used.
Fig. 5 shows the results on GPT-4o . We can see that
answer memory is covered most frequently, while
irrelevant memory is covered the least, suggest-
ing that the agent is effective in distinguish useful
from unuseful memory. Clarifying memory is also
highly utilized, suggesting validity of our memory
type taxonomy. As kincreases, coverage for all
types declines slightly.
7 Impact of Different Memory Types
Now we take a closer look at how distinct memory
types shape response quality. We conduct con-
trolled experiments aiming to characterize system-
atic associations between memory-type presence
and response behavior under realistic retrieval con-
ditions. Specifically, we run two sets of compar-
ison experiments: (i) variant settings that add a
second memory type on top of an answer-only base-
line (vs.A); and (ii) variant setting compared to a
parametric-only baseline (vs.P). The answer-only
baseline reflects a common assumption in RAG: if
answer-containing pieces are retrieved in isolation,
the model should produce the correct answer eas-
ily. In practice, retrieval also contains substantialnon-answer memory, so we aim to know how other
memories (C/E/D/I) alter responses when present.
The parametric-only baseline holds the converse as-
sumption: without memory retrieval, the agent fails
to give a good answer to user’s queries. We choose
the top 15retrieved memory pieces for study (de-
termining their types and varying the memory set-
ting) and test text-embedding-3-small as the re-
triever. Other experiment setups are the same as
above.
7.1 Comparison with Answer Memory Only
We compare thevs.A baseline to five variants:
A+C, A+E, A+D, A+I, and a non-answer bundle
C+E+D+I. This yields six settings and five pair-
wise comparisons. As an aside, memory types are
judged with query and context, so both the com-
position and the counts within the top 15retrieved
pieces can vary. A retrieval may include only one
clarifying (C) piece but more than ten irrelevant
(I) pieces. Therefore, in comparison settings we
retain the original retrieved counts of each memory
to reflect the reality. Queries where the retrieved
pieces lack answer memory (A) are excluded from
this analysis as they do not admit a comparable
answer baseline.
The idea to quantify each variant setting’s im-
pact is checking how its metrics scores change com-
pared to baseline setting. Formally, let Qdenote
the query set for a dataset. For a metric mand
a memory setting i, letsm
q(i)be the score of the
agent’s response to query q∈ Q . We define the
per-query score difference between two settings i
andjasdiffm
q(i, j):
diffm
q(i, j) =sm
q(i)−sm
q(j).
For instance, diffm
q(A+C,A)>0 means that
adding clarifying memory improves the score on
metric mon query qcompared to answer-only base-
line. We measure how often setting ioutperforms
setting jon metric musing an indicatorImprove-
ment Rate(IR):
IRm(i, j) =#{diffm
q(i, j)>0}
#{diffm
q(i, j)̸= 0}∈[0,1].
IRreflects the directionality of score changes un-
der matched retrieval settings. An IRm(i, j) = 0.7
means that, switching from setting jto setting i,
70% of thescore changesare towards better. We
report IRfor each metric. The top table in Tab. 4
shows results on GPT-4o . Results on other agents
7

Comparative
Experiment(vs.A)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
A+C 0.50 1.00 0.60 1.00 –– 1.00 1.00 0.75 1.00 0.89 1.00 0.67 1.00 0.67 0.75 0.57 0.67
A+E 0.48 0.28 0.53 0.65 0.30 0.33 0.33 0.80 0.50 0.71 0.63 0.68 0.42 0.55 0.73 0.59 0.52 0.55
A+D 0.33 0.20 0.42 0.50 0.33 0.00 0.33 0.50 0.39 0.33 0.49 0.75 0.43 0.00 0.65 0.73 0.39 0.63
A+I 0.46 0.46 0.46 0.53 0.63 0.67 0.11 0.57 0.38 0.36 0.52 0.53 0.40 0.55 0.64 0.59 0.52 0.62
C+E+D+I 0.34 0.23 0.05 0.10 0.67 0.50 0.01 0.08 0.03 0.04 0.09 0.20 0.40 0.30 0.37 0.38 0.16 0.22
Comparative
Experiment(vs.P)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
C 0.00 0.00 –0.96 0.00 0.50 0.56 1.00 –1.00 0.77 0.89 0.43 0.53 1.00 0.87 1.00 0.85
A+C 0.00 – 0.89 1.00 0.50 – 0.90 1.00 1.00 1.00 0.93 0.83 0.80 1.00 1.00 1.00 0.92 1.00
E 0.26 0.04 0.80 0.91 0.00 0.14 0.82 0.96 0.76 0.88 0.74 0.75 0.32 0.69 0.85 0.83 0.82 0.74
D 0.20 0.08 0.86 0.75 –0.00 0.80 0.87 0.80 0.77 0.70 0.67 0.27 0.62 0.92 0.87 0.79 0.60
I 0.26 0.09 0.50 0.51 0.00 0.29 0.68 0.64 0.43 0.64 0.49 0.46 0.29 0.34 0.86 0.72 0.75 0.57
A+C+E+D+I 0.29 0.18 0.96 0.97 0.00 0.40 0.95 0.96 0.97 0.97 0.95 0.92 0.28 0.75 0.97 0.88 0.93 0.91
Table 4: Results of improvement rate IRof comparison experiments on LongMemEval-m (left number in each cell)
and Long-MT-Bench+ (right number in each cell) on GPT-4o .Top table: experimental settingvs.controlled setting
(A);Bottom table: experimental settingvs.controlled setting (P). IRvalue stands for the chance of experimental
setting wins over controlled setting, and IR = 0.5 means a tie. “ −” means a lack of data point for the result. We
highlight IR∈[0.8,1.0] with heavy green, [0.6,0.8) with light green, (0.2,0.4] with light red, and [0,0.2] with
heavy red.
are in the Appendix.
Since different memory types distribute vary in
top15retrieved pieces, some types (e.g., clarify-
ing) can be scarce while other types (e.g., irrele-
vant) are relatively more pervasive. This makes
the available samples in each variant setting vary
and sometimes certain settings only contain few
samples for analysis. To further measure the statis-
tical significance of the score changes, we collect
the list Lm(i, j) ={diffm
q(i, j)̸= 0|q∈ Q} and
calculate its mean with 95% confidence interval
(ci_lo ,ci_hi ). Reporting this helps to contextualize
the differences between settings. We visualize the
results in Fig. 6 using blobbograms. Below is our
key findings:
Significance of Clarifying MemoryThe top ta-
ble in Tab. 4 shows that responses with clarifying
memory (C) tend to show large gains across both
datasets, especially on metrics under Relevance and
Informativeness aspects; this is corroborated by the
consistent positive statistics in Fig. 6 (the first row).
Clarifying memory helps bridge the query to the
desired answer and make contextual constraints
explicit. This suggests that injecting lightweight
disambiguation signals (e.g., entity relation cues
or slot clarifications) is a reliably high-leverage
practice.
Enriching Memory Improves Informativeness
The presence of enriching memory (E) generally
makes responses more informative, as reflected
in higher Completeness and Memory Coverage.
However, Tab. 4 also shows a small drop in InternalCoherence, though it is not statistically significant
from Fig. 6. Manual inspection indicates the judge
penalizes verbosity and unfocused discussion,i.e.,
responses become richer but less tightly organized.
Distracting Memory is HarmfulFrom the top ta-
ble in Fig. 4, incorporating distracting memory (D)
substantially degrades performance, particularly on
Accuracy and Relevance, indicating that superfi-
cially similar yet incorrect content can mislead the
model.
Necessity of Answer MemoryRelative to the non-
answer bundle, answer memory remains essential
from the results. Without answer memory, agents
are more likely to hedge or decline, lowering
Informativeness. However, this pattern suggests
thatGPT-4o is reluctant to fabricate answers even
when some relevant context (C/E/D) is present.
We also observe thatirrelevant memory is corre-
lated with reduced constraint awarenesswhile
slightly increasing Memory Coverage and ROUGE-
1 Recall. Generally, we do not observe adding
irrelevant memory helps. This result extends and
challenges existing claim on random/irrelevant doc-
uments that they can improve answer-containing
Accuracy on small LLMs (Cuconasu et al., 2024).
On one hand, we conjecture that large-scale LLMs
behave differently from small LLMs. On another
hand, irrelevant memory may divert LLM’s atten-
tion away from contextual constraints.
8

1.5
 1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer6
48
27
70
1401
7
5
11
43World Knowledge
1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer5
32
19
39
2263
20
6
19
94Answer Accuracy
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer10
3
8
123
2
3
6Internal Coherence
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer1
9
3
18
1562
5
2
7
50Topic Relevance
2
 1
 0 1
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer4
36
18
40
2284
14
3
14
97Constraint Relevance
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer9
49
39
83
2712
28
8
36
112Completeness
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer3
48
47
98
1381
29
2
38
60Actionability
2
 1
 0 1 2 3
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer9
127
76
219
3464
88
22
143
194Memory Coverage
0.6
 0.4
 0.2
 0.0 0.2 0.4 0.6
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer7
77
28
93
2816
93
19
146
188Rouge-1 Recall
LongMemEval-m Long-MT-Bench+Figure 6: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to the answer-only baseline on GPT-4o . Squares mark the mean difference; horizontal bars show the
confidence interval. Square size is proportional to the number of queries with a non-zero change, and the number
shown beside each square equals that count. Absence of square means no qualifying queries for that result.
7.2 Comparison with Parametric Memory
Only
In the second comparison group comparing with
parametric-only baseline, we evaluate six settings:
C, A+C, E, D, I, and all types bundle. The IR
results for GPT-4o are in the bottom table of Tab. 4,
with statistical analysis in Fig. 15. Below is our
key findings:
Relevant Memory Helps Accuracy and Rele-
vanceThe bottom table in Tab. 4 and Fig. 15 show
that adding semantically relevant memory (answer,
clarifying, enriching, or distracting) generally im-
proves Answer Accuracy, Topic Relevance, and
Constraint Relevance over parametric-only base-
line. This suggests the necessity of past history
when agents generating responses. Otherwise, the
responses are seriously off-topic and inaccurate.
Clarifying Memory Remains CriticalAmong
the settings, the A+C and all type bundle settings
yield the largest average gains, highlighting the cen-
tral role of answer-containing evidence and disam-
biguation information; for GPT-4o , these settings
also correspond to higher Actionability. Adding
clarifying memory (C) alone also brings significant
improvement.
Irrelevant Memory Offers Little Improvement
Irrelevant memory pieces provide no clear improve-ment over parametric-only, yet they are still fre-
quently echoed in responses from Memory Con-
verge — a sign that the agent tends to echo ir-
relevant retrieved text without making the answer
better.
RAG Can Reduce Fidelity and Coherence
Across models, we observe adding retrieved mem-
ory often lowers World Knowledge Accuracy and
Internal Coherence relative to using parametric
memory alone. Manual analysis suggests this ef-
fect arises because additional context encourages
more specific claims, which the LLM judge per-
ceives increases the risk of factual errors, whereas
parametric-only responses are shorter, less specific,
and safer.
7.3 Non-significant Results
We also observe settings where the impact of spe-
cific memories is statistically non-significant across
agent models and datasets. For example, whether
adding distracting memory reduces Internal Coher-
ence relative to the answer-only baseline, and how
adding enriching memory affect the Topic Rele-
vance. These effects are small and variable from
the results, and stronger evidence would be needed
to claim a reliable effect.
9

8 Conclusion And Future Directions
In this work, we examine how memory types af-
fect LLMs in conversations using a novel memory
taxonomy, a user-centric evaluation, and controlled
comparisons. The experiments over agent models,
context size, and retrievers confirm known patterns
and provide new findings in long-term conversa-
tional RAG scenario. In comparison experiments,
we find conversational RAG performance is driven
not merely by retrieving relevant context, but by
retrieving the right functional types of memory.
These findings point toward memory-type-aware
retrieval and selection as a promising direction for
IR. First, retrieval systems can benefit from be-
ing memory-role aware, enabling category-aware
curation, ranking, and diversification rather than
treating all retrieved context uniformly. Second,
our results motivate the development of collections
that evaluate the different roles of grounding in
RAG for memory and other corpora. Finally, it is
important to develop and validate taxonomies as
analytical tools: the most useful taxonomy is not
necessarily the most granular, but the one that best
helps us understand retrieval behavior and optimize
end-to-end outcomes.
Limitations
While our work provides many new findings, it
also has several limitations that should be acknowl-
edged.
LLM-as-a-Judge EvaluationOur experiments
heavily reply on LLM judge for response scoring
and memory type identification. Though we chose
advanced models GPT-5-thinking andGPT-o3 as
the judges and carefully designed the rubric, LLM
judge is known to occasionally produce inconsis-
tent or imperfect assessments. This limitation may
introduce uncertainty into the reported results and
our analysis.
Computational CostOur experiments involve
large-scale experiments with proprietary LLMs
such as GPT models and Gemini. Reproducing
these results requires substantial API usage, which
may be costly and limit full replication by others.
Dataset Size and NoiseThe two datasets we used
LongMemEval-m and Long-MT-Bench+ contain
a few hundred test queries (see their statistics in
Table 3). While labeling errors are expected to be
rare, any noise in such a small test size can still
introduce minor changes in the reported results.
Moreover, some of our results are based on veryfew data points, which limits their statistical signif-
icance and should be interpreted with caution.
References
Google AI and Google DeepMind. 2025. Google gem-
ini.https://gemini.google.com.
Hadel Alhawasi and Abdou Youssef. 2024. Using llms
for evaluating qa systems: Exploration and assess-
ment. In2024 2nd International Conference on Foun-
dation and Large Language Models (FLLM), pages
462–469. IEEE.
Chen Amiraz, Florin Cuconasu, Simone Filice, and
Zohar Karnin. 2025. The distracting effect: Under-
standing irrelevant passages in RAG. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 18228–18258. Association for Computational
Linguistics.
Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu,
Shayne Longpre, Stephen Pulman, and Srinivas
Chappidi. 2021. Open-domain question answering
goes conversational via question rewriting.Proceed-
ings of the 2021 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies.
Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jia-
heng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su,
Tiezheng Ge, Bo Zheng, and 1 others. 2024. Mt-
bench-101: A fine-grained benchmark for evaluating
large language models in multi-turn dialogues. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 7421–7454.
Hritik Bansal, John Dang, and Aditya Grover. 2024.
Peering through preferences: Unraveling feedback
acquisition for aligning large language models. In
The Twelfth International Conference on Learning
Representations.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. InProceedings of
the Thirty-Eighth AAAI Conference on Artificial In-
telligence and Thirty-Sixth Conference on Innovative
Applications of Artificial Intelligence and Fourteenth
Symposium on Educational Advances in Artificial
Intelligence. AAAI Press.
Nuo Chen, Hongguang Li, Jianhui Chang, Juhua Huang,
Baoyuan Wang, and Jia Li. 2025. Compress to
impress: Unleashing the potential of compressive
memory in real-world long-term conversations. In
Proceedings of the 31st International Conference on
Computational Linguistics, pages 755–773. Associa-
tion for Computational Linguistics.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
10

production-ready ai agents with scalable long-term
memory.arXiv preprint arXiv:2504.19413.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 719–729.
Google DeepMind. 2024. Introducing gemini 2.0: our
new ai model for the agentic era. Blog post. https:
//blog.google/technology/google-deepmind/
google-gemini-ai-update-december-2024/.
Yijiang River Dong, Tiancheng Hu, and Nigel Collier.
2024. Can LLM be a personalized judge? InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 10126–10141. Association for
Computational Linguistics.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pages 6491–
6501.
Jianfeng Gao, Chenyan Xiong, Paul Bennett, and Nick
Craswell. 2023.Neural approaches to conversa-
tional information retrieval, volume 44. Springer.
Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu,
Wenqi Shao, and Ping Luo. 2025a. HiAgent: Hier-
archical working memory management for solving
long-horizon agent tasks with large language model.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 32779–32798. Association
for Computational Linguistics.
Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason
Zhang, Xiaoyong Wei, and Li Qing. 2025b. Removal
of hallucination on hallucination: Debate-augmented
RAG. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 15839–15853. Associa-
tion for Computational Linguistics.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2024. Long-context llms meet rag: Overcom-
ing challenges for long inputs in rag.arXiv preprint
arXiv:2410.05983.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2025. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. InThe
Thirteenth International Conference on Learning
Representations.
Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang
Wang, and Tat-Seng Chua. 2025. Hello again! LLM-
powered personalized agent for long-term dialogue.
InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 5259–5276.
Association for Computational Linguistics.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, Benjamin Newman, Binhang Yuan, Bobby Yan,
Ce Zhang, Christian Cosgrove, Christopher D Man-
ning, Christopher Re, Diana Acosta-Navas, Drew A.
Hudson, and 31 others. 2023. Holistic evaluation of
language models.Transactions on Machine Learning
Research.
Chin-Yew Lin. 2004. ROUGE: A package for automatic
evaluation of summaries. InText Summarization
Branches Out, pages 74–81. Association for Compu-
tational Linguistics.
Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yu-
lan He, Di Yin, Xing Sun, and Yunsheng Wu. 2023.
Memochat: Tuning llms to use memos for consis-
tent long-range open-domain conversation.Preprint,
arXiv:2308.08239.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of LLM agents. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 13851–
13870. Association for Computational Linguistics.
Kaushal Kumar Maurya, Kv Aditya Srivatsa, Kseniia
Petukhova, and Ekaterina Kochmar. 2025. Unify-
ing AI tutor evaluation: An evaluation taxonomy for
pedagogical ability assessment of LLM-powered AI
tutors. InProceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 1234–
1251, Albuquerque, New Mexico.
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Rad-
ford, Jesse Michael Han, Jerry Tworek, Qiming
11

Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hal-
lacy, and 1 others. 2022. Text and code embed-
dings by contrastive pre-training.arXiv preprint
arXiv:2201.10005.
OpenAI. 2025a. Chatgpt (gpt-5). https://chat.
openai.com/. Large language model.
OpenAI. 2025b. Gpt -5 system card. https://cdn.
openai.com/gpt-5-system-card.pdf.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G Patil, Ion Stoica, and Joseph E Gonzalez.
2023. Memgpt: Towards llms as operating systems.
arXiv preprint arXiv:2310.08560.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo,
Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew
Lin, H. Vicky Zhao, Lili Qiu, and Jianfeng Gao.
2025. Secom: On memory construction and retrieval
for personalized conversational agents. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Filip Radlinski and Nick Craswell. 2017. A theoretical
framework for conversational search. InProceedings
of the 2017 conference on conference human infor-
mation interaction and retrieval, pages 117–126.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais,
Jack Ryan, and Daniel Chalef. 2025. Zep: a tempo-
ral knowledge graph architecture for agent memory.
arXiv preprint arXiv:2501.13956.
Alireza Rezazadeh, Zichao Li, Ange Lou, Yuying Zhao,
Wei Wei, and Yujia Bao. 2025. Collaborative mem-
ory: Multi-user memory sharing in llm agents with
dynamic access control.Preprint, arXiv:2505.18279.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.
Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng
Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid
Palangi, George Lee, Anand Rajan Iyer, Tianlong
Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister.
2025a. In prospect and retrospect: Reflective mem-
ory management for long-term personalized dialogue
agents. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 8416–8439. Association
for Computational Linguistics.
Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng
Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid
Palangi, George Lee, Anand Iyer, Tianlong Chen,
Huan Liu, Chen-Yu Lee, and Tomas Pfister. 2025b.
In prospect and retrospect: Reflective memory man-
agement for long-term personalized dialogue agents.
Preprint, arXiv:2503.08026.
Paul Thomas, Seth Spielman, Nick Craswell, and
Bhaskar Mitra. 2024. Large language models can ac-
curately predict searcher preferences. InProceedingsof the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
SIGIR ’24, page 1930–1940, New York, NY , USA.
Bo Wang, Weiyi He, Shenglai Zeng, Zhen Xiang, Yue
Xing, Jiliang Tang, and Pengfei He. 2025. Unveiling
privacy risks in LLM agent memory. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 25241–25260. Association for Computational
Linguistics.
Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu,
Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu,
Tianyu Liu, and Zhifang Sui. 2024. Large language
models are not fair evaluators. InProceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
9440–9450. Association for Computational Linguis-
tics.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang,
Kai-Wei Chang, and Dong Yu. 2025. Longmemeval:
Benchmarking chat assistants on long-term interac-
tive memory. InThe Thirteenth International Con-
ference on Learning Representations.
Jing Xu, Arthur Szlam, and Jason Weston. 2022. Be-
yond goldfish memory: Long-term open-domain con-
versation. InProceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 5180–5197. Association
for Computational Linguistics.
Linhao Ye, Zhikai Lei, Jianghao Yin, Qin Chen, Jie
Zhou, and Liang He. 2024. Boosting conversa-
tional question answering with fine-grained retrieval-
augmentation and self-check. InProceedings of the
47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval,
pages 2301–2305.
Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Eval-
uating text generation with bert. InInternational
Conference on Learning Representations.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zi-
hang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen
Thai, Shuo Wang, Zhiyuan Liu, and 1 others. 2024.
∞Bench: Extending long context evaluation beyond
100K tokens. InProceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 15262–
15277.
Zheng Zhao, Clara Vania, Subhradeep Kayal, Naila
Khan, Shay B Cohen, and Emine Yilmaz. 2025. Per-
sonaLens: A benchmark for personalization evalua-
tion in conversational AI assistants. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 18023–18055. Association for Computa-
tional Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
12

Zhuohan Li, Dacheng Li, Eric Xing, and 1 others.
2023. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in neural information pro-
cessing systems, 36:46595–46623.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and
Yanlin Wang. 2024. Memorybank: Enhancing large
language models with long-term memory.Proceed-
ings of the AAAI Conference on Artificial Intelligence,
38(17):19724–19731.Appendix
A Prompt for Response Generation
The full prompts we use for response generation
are shown in Figure 7.
system: You are a chat assistant. You previously had a
conversation with a user. Now the user is sending you
a new message and you need to respond to it. I will
provide you with some historical chat pieces with dates
between you and the user which you may find helpful to
give response.
user: Historical Chats: {retrieved pieces}
User’s message: {query}
Your response:
system: You are a chat assistant. You previously had a
conversation with a user. Now the user is sending you a
new message and you need to respond to it. I will pro-
vide you with some historical chat pieces between you
and the user which you may find helpful to give response.
user: Historical Chats: {retrieved pieces}
User’s message: {query}
Your response:
system: You are a chat assistant. Now the user is
sending you a new message and you need to respond to it.
user: User’s message: {query}
Your response:
Figure 7: Prompt for prompting conversational agent
to generate response. (top) prompt for memory with
timestamp and memory piece size k >0 ; (middle)
prompt for memory without timestamp and k >0 ;
(bottom) prompt fork= 0.
B Human Verification On Memory Type
We randomly sampled 5 queries from
LongMemEval-m and 5 from Long-MT-Bench+
datasets. For each query, we randomly sampled 3
retrieved memory pieces among the top-15 pieces,
whose types are predicted by GPT-5-thinking in
our pipeline. In total, we have 30 memory pieces
for human evaluation.
We designed a user study questionnaire, mirror-
ing the same instruction and rubric used for the
LLM judge, providing example tasks and detailed
guidance. Each task is a multiple-choice question
to classify the memory into one of the five types.
We recruited 5 human participants from Prolific1, a
third-party data annotation platform, for the study.
Our recruitment follows the below criteria:
1https://www.prolific.com/
13

GPT-5 H1 H2 H3 H4 H5
GPT-5 – 0.417 0.618 0.512 0.578 0.532
H1 0.417 – 0.426 0.245 0.441 0.573
H2 0.618 0.426 – 0.514 0.615 0.524
H3 0.512 0.245 0.514 – 0.467 0.384
H4 0.578 0.441 0.615 0.467 – 0.538
H5 0.532 0.573 0.524 0.384 0.538 –
Average 0.531 0.420 0.540 0.425 0.528 0.510
Table 5: Pairwise kappa scores among
GPT-5-thinking and human participants (H1-
H5). GPT-5 ’s scores suggest it achieves human-level
reliability for memory type classification.
•Native English speakers residing in English-
speaking countries;
• Holding bachelor degree or above;
• High historical approval rate (≥90%);
• Age between 18 to 80.
We rejected participants whose submissions are
with unrealistically short completion times and re-
recruit. The compensation is at Prolific’s recom-
mended rate.
Using the 30 annotated memory pieces, we
calculate the pairwise Cohen’s Kappa kas their
inter-annotator agreement, where kis standard
unweighted and k= 0 corresponds to chance-
level agreement. The results are shown in Tab 5.
All pairwise kscores between GPT-5-thinking
and human annotators are well above chance, and
GPT-5-thinking is comparable to human–human
agreement.
C Evaluation with LLM-as-a-Judge
We carefully designed and iteratively refined the
prompts used for LLM judges to ensure reliable and
consistent evaluation. The design process followed
an iterative “prompt–test–refine” loop. Starting
from a minimal baseline, we tested responses to
identify issues such as instruction drift or unsta-
ble reasoning. We then applied human-in-the-loop
refinement to adjust phrasing and structure, validat-
ing each revision on representative cases until the
prompts produce desired outputs on a set of human-
validated test samples. We provide judgment cases
for memory type classification and response evalu-
ation in Table 10, 11, and 12.
We use GPT-5-thinking as the judge for deter-
mining memory types, setting its reasoning effort
tomedium , and max completion tokens to 8,000 .
Figure 8 is the full prompt for it. For the response
evaluation, we use GPT-o3 , setting its reasoning ef-
fort to high, and max completion tokens to 10,000 .Figure 9 is the prompt for evaluating Accuracy;
Figure 10 is for evaluating Relevance; Figure 11 is
for evaluating Memory Coverage; Figure 12 is for
evaluating Informativeness.
D Memory Type Classification Details
We basically rely on prompt in Figure 8 to deter-
mine the memory type. For the experiment on
Long-MT-Bench+, since it contains oracle labels
for gold pieces, we directly classify the gold pieces
as answer memory.
E General Performances on Contriever
and BM25
Figure 13 and Figure 14 show the general perfor-
mance over metrics of three LLMs using Contriever
and BM25 as retrieval method.
F Comparative Experiments on Gemini
and Llama
Table 6, 7, 8 and 9 show the improve-
ment rate IRresults of Gemini-2.0-Flash and
Llama-3.3-70B on different comparative experi-
ments. Figure 15 show the statistical results of
Lm(i, j) ofGPT-4o-2024-08-06 on experiment
comparing with parametric memory only. Fig-
ure 16, 17, 18, and 19 are statistical results of
specific model and comparison setting.
14

user: You are a quality rater evaluating the relevance of a memory piece to a user’s query. The memory piece is from
the user’s conversation with a chat assistant. Your task is to assign an integer score on a scale from 0 to 4 based on the
following criteria:
Scoring rubric (check in order 4→3→2→1→0):
4 - Answer: the memory piece clearly answers the query and itself contains the answer or one of the answers for the
question.
3 - Clarifying: the memory piece is necessary to resolve the identity of the asked-for entity/event (e.g., coreference, alias,
where/which/company disambiguation) so that an answer stated elsewhere can be correctly applied. Without this piece, you
cannot derive a correct or confident answer from the remaining information.
2 - Enriching: the memory piece refers to the same specific entity/event (or uniquely narrows to it) and adds useful ancillary
detail (e.g., who/where/what/when) that would make the final answer more informative or helpful, even though it does not
provide the asked-for attribute. Domain context that is clearly connected to the query also counts as Helpful.
1 - Distracting: the memory piece seems to reveal the answer but is actually misleading.
0 - Irrelevant: the memory piece has no meaningful semantic connection to the user’s query domain.
Now I will provide you with a user’s query, one memory piece, and other pieces that can be helpful for you to judge. Your
task is to assign a relevance score for this memory piece, following the rubric above.
Note: The score should reflect the memory piece itself. The other pieces are provided only to help you understand the
memory piece better.
## User’s Query:
{user query}
## Memory Piece:
{memory piece}
## Other Pieces:
{other memory pieces}
## Your Score:
Figure 8: Prompt used by the LLM judge to classify retrieved memory pieces.
user: You are an objective and fair evaluator. You are provided with a user’s query to a chatbot, the chatbot’s response, and
the gold response. Your task is to judge the accuracy of the chatbot’s response based on three aspects:
1. World knowledge accuracy - How accurate are the factual statements in the response with respect to general facts and
world knowledge. A response that does not mention world knowledge is not considered inaccurate.
2. Answer accuracy - How accurate is the response with respect to the answer provided in the gold response.
3. Internal consistency - How logically consistent is the response internally.
You should givebinary scoresfor each aspect where 1 means accurate or consistent and 0 means not, and the explanation
for the score.
###User’s query###
{question}
###Chatbot’s response###
{agent response}
###Gold answer###
{gold answer}
You should output only three lines. The first line is the score and explanation for World knowledge accuracy, the second line
is the score and explanation for Contextual accuracy, and the third line is the score and explanation for Internal consistency.
Each line’s format should be ‘Score: <score>; Explanation: <explanation>’. No index is needed. Now tell me your scores:
Figure 9: Prompt for LLM judge to determine the Accuracy of responses.
15

user: You are an objective and fair evaluator. You are provided with a user’s query to a chatbot, the user’s intent behind the
query, the chatbot’s response to the query, and the prior conversation history. Your task is to judge the relevance of the
chatbot’s response based on two aspects:
1. Topical relevance - How directly does the response address the query?
2. Constraint relevance - How well does the response meet all implicit or explicit constraints and conditions stated or
implied by the user’s intent and prior conversation?
You should givebinary scoresfor each aspect where 1 means relevant and 0 means not relevant, and the explanation for
the score.
###User’s query###
{question}
###User’s intent### (optional)
{intent}
###Chatbot’s response###
{agent response}
###Prior conversation history###
{context}
You should output only two lines. The first line is the score and explanation for Topical relevance, the second line is the
score and explanation for Constraint relevance. Each line’s format should be ‘Score: <score>; Explanation: <explanation>’.
No index is needed. Now tell me your scores:
Figure 10: Prompt for LLM judge to determine the Relevance of responses.
user: You are a helpful and fair judge. I will provide you with:
1. A user’s query to a chatbot;
2. The user’s intent behind the query;
3. The chatbot’s response;
4. The prior conversation history between the user and the chatbot.
Your task is to identify all pieces of information in the chatbot’s response that are recalled from the prior conversation
history but were *not* restated in the current user query.
- Count each such recalled item.
- Consider paraphrases or indirect references as recalls.
- Ignore generic pleasantries or facts that never appeared in the history.
###Prior conversation history###
{prior conversation history}
###User’s query###
{query}
###User’s intent###
{intent}
###Chatbot’s response###
{agent response}
You should output two lines. The first line is the count, the second line is explanation listing the recalled items and in which
piece they are recalled from. Now tell me your score and explanation:
Figure 11: Prompt for LLM judge to determine the Memory Converge of responses.
16

user: You are an objective and fair evaluator. You are provided with a user’s query to a chatbot, the user’s intent behind the
query, the chatbot’s response to the query, and the prior conversation history. Your task is to judge the informativeness of
the chatbot’s response based on two aspects:
1. Completeness - How well does the response cover all sub-questions or necessary information relevant to the query and
intent?
2. Actionability - How well does the response provide concrete details or clear guidance?
Particularly, for aspect Completeness, this is a more detailed scale:
Score 0 (Absent): No sign the response noticed the user’s motivations, constraints, or intents. Response is completely
generic or contradicts the context.
Score 1 (Partial): The response realizes some of the user’s intent and motivations and mentions at least one contextual
element. But it does not fully address the user’s intent.
Score 2 (Comprehensive): The response fully addresses the user’s intent and motivations, providing a thorough and
contextually relevant answer.
For aspect Actionability, this is a more detailed scale:
Score 0 (Not actionable): The response is vague, generic, or purely descriptive. Lacks concrete items.
Score 1 (Actionable): The response provides a clear sequence of actions or names specific items.
You should give an integer score from 0 to 2 for aspect Completeness and binary score for aspect Actionability. You should
also provide explanation for the score.
###User’s query###
{question}
###User’s intent###
{intent}
###Chatbot’s response###
{agent response}
###Prior conversation history###
{context}
You should output only two lines. The first line is the score and explanation for Completeness, the second line is the score
and explanation for Actionability. Each line’s format should be ‘Score: <score>; Explanation: <explanation>’. No index is
needed. Now tell me your scores:
Figure 12: Prompt for LLM judge to determine the Informativeness of response.
Comparative
Experiment(vs.A)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
A+C 0.60 – 0.75 0.00 0.67 – 1.00 1.00 0.75 0.50 1.00 0.33 0.67 0.00 0.73 1.00 0.83 0.20
A+E 0.42 0.30 0.53 0.44 0.45 0.00 0.25 0.50 0.59 0.50 0.60 0.59 0.50 0.46 0.69 0.52 0.62 0.54
A+D 0.36 0.60 0.47 0.40 0.71 1.00 0.60 1.00 0.42 0.40 0.48 0.43 0.40 0.80 0.72 0.59 0.68 0.48
A+I 0.43 0.64 0.40 0.41 0.60 0.57 0.30 0.31 0.30 0.36 0.52 0.24 0.61 0.33 0.63 0.42 0.49 0.40
C+E+D+I 0.27 0.14 0.05 0.04 0.55 0.55 0.05 0.01 0.06 0.06 0.12 0.12 0.39 0.24 0.47 0.29 0.15 0.20
Table 6: Gemini-2.0-Flash ’s results of IRof experimental setting (index column)vs.controlled setting (answer
memory only) on LongMemEval-m (left number in each cell) and Long-MT-Bench+ (right number in each cell).
Comparative
Experiment(vs.A)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
A+C 0.25 0.00 0.60 1.00 1.00 0.00 – 1.00 0.50 1.00 0.80 0.67 0.40 1.00 0.90 0.67 0.78 0.50
A+E 0.53 0.29 0.47 0.73 0.38 0.00 0.83 0.67 0.48 0.53 0.60 0.65 0.54 0.59 0.71 0.59 0.61 0.61
A+D 0.36 0.50 0.39 0.50 0.64 0.50 0.00 0.50 0.52 0.80 0.43 0.57 0.48 0.75 0.60 0.62 0.54 0.64
A+I 0.46 0.50 0.36 0.61 0.56 0.56 0.19 0.00 0.30 0.48 0.46 0.38 0.44 0.63 0.50 0.54 0.41 0.50
C+E+D+I 0.24 0.08 0.04 0.06 0.44 0.57 0.02 0.00 0.02 0.08 0.06 0.13 0.19 0.26 0.48 0.33 0.16 0.19
Table 7: Llama-3.3-70B ’s results of IRof experimental setting (index column)vs.controlled setting (answer
memory only) on LongMemEval-m (left number in each cell) and Long-MT-Bench+ (right number in each cell).
17

0 5 10 20 50 1000.70.80.9Average Score
World Knowledge (LongMemEval-m)
0 5 10 20 50 1000.20.4Average Score
Answer Accuracy (LongMemEval-m)
0 5 10 20 50 1000.900.951.00Average Score
Internal Coherence (LongMemEval-m)
0 5 10 20 50 1000.40.60.8Average Score
Topic Relevance (LongMemEval-m)
0 5 10 20 50 1000.20.4Average Score
Constraint Awareness (LongMemEval-m)
0 5 10 20 50 1001.01.5Average Score
Completeness (LongMemEval-m)
0 5 10 20 50 1000.60.70.80.9Average Score
Actionability (LongMemEval-m)
0 5 10 20 50 100123Average Score
Memory Coverage (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
ROUGE-1 Recall (LongMemEval-m)
0 5 10 20 50 1000.900.95Average Score
World Knowledge (Long-MT-Bench+)
0 5 10 20 50 1000.40.60.8Average Score
Answer Accuracy (Long-MT-Bench+)
0 5 10 20 50 1000.940.960.981.00Average Score
Internal Coherence (Long-MT-Bench+)
0 5 10 20 50 1000.60.81.0Average Score
Topic Relevance (Long-MT-Bench+)
0 5 10 20 50 1000.40.60.8Average Score
Constraint Awareness (Long-MT-Bench+)
0 5 10 20 50 1001.001.251.501.75Average Score
Completeness (Long-MT-Bench+)
0 5 10 20 50 1000.60.70.8Average Score
Actionability (Long-MT-Bench+)
0 5 10 20 50 1002345Average Score
Memory Coverage (Long-MT-Bench+)
0 5 10 20 50 1000.50.60.7Average Score
ROUGE-1 Recall (Long-MT-Bench+)
Gemini-2.0-Flash
GPT-4o-2024-08-06
Llama-3.3-70BFigure 13: Average scores over different metrics of three agent models on LongMemEval-m and Long-MT-Bench+
datasets. Results are based onContriever. Average scores are calculated over queries.
0 5 10 20 50 1000.70.80.9Average Score
World Knowledge (LongMemEval-m)
0 5 10 20 50 1000.20.4Average Score
Answer Accuracy (LongMemEval-m)
0 5 10 20 50 1000.9250.9500.9751.000Average Score
Internal Coherence (LongMemEval-m)
0 5 10 20 50 1000.40.60.8Average Score
Topic Relevance (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
Constraint Awareness (LongMemEval-m)
0 5 10 20 50 1000.51.01.5Average Score
Completeness (LongMemEval-m)
0 5 10 20 50 1000.60.70.80.9Average Score
Actionability (LongMemEval-m)
0 5 10 20 50 100123Average Score
Memory Coverage (LongMemEval-m)
0 5 10 20 50 1000.20.40.6Average Score
ROUGE-1 Recall (LongMemEval-m)
0 5 10 20 50 1000.250.500.751.00Average Score
World Knowledge (Long-MT-Bench+)
0 5 10 20 50 1000.000.250.500.75Average Score
Answer Accuracy (Long-MT-Bench+)
0 5 10 20 50 1000.940.960.981.00Average Score
Internal Coherence (Long-MT-Bench+)
0 5 10 20 50 1000.250.500.751.00Average Score
Topic Relevance (Long-MT-Bench+)
0 5 10 20 50 1000.250.500.75Average Score
Constraint Awareness (Long-MT-Bench+)
0 5 10 20 50 1000.51.01.5Average Score
Completeness (Long-MT-Bench+)
0 5 10 20 50 1000.250.500.75Average Score
Actionability (Long-MT-Bench+)
0 5 10 20 50 100024Average Score
Memory Coverage (Long-MT-Bench+)
0 5 10 20 50 1000.20.40.6Average Score
ROUGE-1 Recall (Long-MT-Bench+)
Gemini-2.0-Flash
GPT-4o-2024-08-06
Llama-3.3-70B
Figure 14: Average scores over different metrics of three agent models on LongMemEval-m and Long-MT-Bench+
datasets. Results are based onBM25retriever. Average scores are calculated over queries.
Comparative
Experiment(vs.P)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
C 0.17 0.12 0.00 0.91 0.00 0.00 0.88 0.92 – 0.88 0.42 0.81 0.25 0.22 0.87 0.87 0.58 0.76
A+C 0.57 0.00 0.89 0.75 0.00 – 1.00 1.00 1.00 0.80 0.93 0.83 0.40 0.00 0.88 0.86 0.82 1.00
E 0.20 0.08 0.76 0.82 0.00 0.40 0.84 0.87 0.72 0.88 0.62 0.75 0.20 0.37 0.87 0.84 0.63 0.69
D 0.23 0.00 0.69 0.62 0.00 0.17 0.76 0.73 0.68 0.62 0.48 0.63 0.16 0.35 0.89 0.81 0.59 0.45
I 0.23 0.06 0.50 0.39 0.00 0.50 0.68 0.35 0.44 0.43 0.41 0.22 0.12 0.18 0.88 0.58 0.60 0.29
A+C+E+D+I 0.38 0.28 0.95 0.92 0.00 0.50 0.93 0.96 0.95 0.96 0.92 0.78 0.29 0.47 0.95 0.81 0.89 0.67
Table 8: Gemini-2.0-Flash ’s results of IRof experimental setting (index column)vs.controlled setting (parametric
memory only) on LongMemEval-m (left number in each cell) and Long-MT-Bench+ (right number in each cell).
18

Comparative
Experiment(vs.P)World
KnowledgeAnswer
AccuracyInternal
CoherenceTopic
RelevanceConstraint
AwarenessCompleteness ActionabilityMemory
CoverageROUGE-1
Recall
C 0.25 0.00 –0.95 0.00 0.00 0.89 0.95 1.00 0.96 0.91 0.93 0.60 0.50 0.95 0.93 1.00 0.90
A+C 0.50 0.00 0.88 1.00 – 0.00 0.92 1.00 1.00 1.00 1.00 1.00 0.89 1.00 0.94 1.00 1.00 1.00
E 0.22 0.25 0.82 0.83 0.00 0.70 0.90 0.88 0.79 0.81 0.79 0.76 0.44 0.61 0.96 0.87 0.79 0.83
D 0.08 0.08 0.53 0.67 0.00 0.43 0.84 0.75 0.57 0.57 0.62 0.77 0.26 0.50 0.96 0.76 0.70 0.73
I 0.25 0.13 0.33 0.44 0.06 0.65 0.74 0.47 0.38 0.37 0.50 0.39 0.22 0.29 0.93 0.75 0.71 0.56
A+C+E+D+I 0.45 0.47 0.96 0.98 0.03 0.71 0.96 0.98 0.97 0.95 0.94 0.91 0.59 0.74 0.95 0.89 0.94 0.88
Table 9: Llama-3.3-70B ’s results of IRof experimental setting (index column)vs.controlled setting (parametric
memory only) on LongMemEval-m (left number in each cell) and Long-MT-Bench+ (right number in each cell).
1.2
 1.0
 0.8
 0.6
 0.4
 0.2
 0.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All7
4
57
50
114
1263
23
12
47
22World Knowledge
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All9
30
14
28
24924
5
70
12
43
161Answer Accuracy
1.0
 0.5
 0.0 0.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All2
5
4
257
2
7
5Internal Coherence
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All9
10
105
78
167
27822
5
56
15
36
109Topic Relevance
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
33
15
30
25523
5
73
13
46
151Constraint Relevance
0.0 0.5 1.0 1.5 2.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All13
15
126
83
173
34127
6
99
24
96
168Completeness
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All7
5
71
68
154
16917
3
62
16
73
93Actionability
1 2 3 4 5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All14
17
199
126
228
44839
6
157
38
178
250Memory Coverage
0.0 0.2 0.4 0.6
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All9
12
118
72
164
35839
7
162
43
219
270Rouge-1 Recall
LongMemEval-m Long-MT-Bench+
Figure 15: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to theparametric-only baselineon GPT-4o . Squares mark the mean difference; horizontal bars show the
confidence interval. Square size is proportional to the number of queries with a non-zero change, and the number
shown beside each square equals that count.
19

1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer5
50
39
91
15010
5
14
64World Knowledge
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer4
40
19
53
2241
16
10
39
112Answer Accuracy
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer3
20
7
25
333
1
7
11Internal Coherence
1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer2
8
5
20
1511
4
1
16
74Topic Relevance
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer4
54
24
74
2352
16
10
42
125Constraint Relevance
1
 0 1
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer7
62
46
105
2893
27
7
62
129Completeness
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer3
62
48
107
1531
24
5
40
82Actionability
2
 1
 0 1 2 3
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer11
150
102
254
3393
84
29
150
180Memory Coverage
0.50
 0.25
 0.00 0.25 0.50
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer6
84
37
113
2795
93
23
186
196Rouge-1 Recall
LongMemEval-m Long-MT-Bench+Figure 16: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to theanswer-only baselineon Gemini-2.0-Flash . Squares mark the mean difference; horizontal bars
show the confidence interval. Square size is proportional to the number of queries with a non-zero change, and the
number shown beside each square equals that count.
1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All6
7
83
64
171
1418
1
26
17
81
32World Knowledge
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All1
9
25
16
28
22821
4
68
8
36
153Answer Accuracy
1.0
 0.5
 0.0 0.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All1
1
3
5
6
431
5
6
10
10Internal Coherence
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
11
126
75
179
26924
4
70
11
57
130Topic Relevance
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
36
19
41
24425
5
76
8
46
164Constraint Relevance
1.0
 0.5
 0.0 0.5 1.0 1.5 2.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All12
14
109
75
185
32026
6
103
19
115
184Completeness
1.0
 0.5
 0.0 0.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
5
82
76
177
16118
2
51
17
122
83Actionability
0 1 2 3 4 5 6
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All15
17
222
151
307
46438
7
160
41
195
254Memory Coverage
0.2
 0.0 0.2 0.4 0.6
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All12
11
133
75
183
36937
6
169
38
231
270Rouge-1 Recall
LongMemEval-m Long-MT-Bench+
Figure 17: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to theparametric-only baselineon Gemini-2.0-Flash . Squares mark the mean difference; horizontal
bars show the confidence interval. Square size is proportional to the number of queries with a non-zero change, and
the number shown beside each square equals that count.
20

1.5
 1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer4
55
36
102
1771
7
6
20
73World Knowledge
1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer5
38
23
70
2351
15
4
26
109Answer Accuracy
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer1
21
11
27
361
4
2
9
7Internal Coherence
1.5
 1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer6
3
26
1731
3
2
5
60Topic Relevance
2
 1
 0 1
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer4
54
25
92
2402
19
5
21
116Constraint Relevance
1
 0 1 2
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer5
52
28
87
2873
17
7
37
119Completeness
1.0
 0.5
 0.0 0.5 1.0 1.5
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer5
63
50
134
1992
32
4
46
89Actionability
2
 0 2 4
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer10
143
98
256
3346
93
21
159
187Memory Coverage
0.6
 0.4
 0.2
 0.0 0.2 0.4
Average Score DifferenceAnswer
+Clarifying
Answer
+Enriching
Answer
+Distracting
Answer
+Irrelevant
Non-Answer9
72
37
111
2954
89
22
165
202Rouge-1 Recall
LongMemEval-m Long-MT-Bench+Figure 18: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to theanswer-only baselineonLlama-3.3-70B. Squares mark the mean difference; horizontal bars show
the confidence interval. Square size is proportional to the number of queries with a non-zero change, and the number
shown beside each square equals that count.
1.0
 0.5
 0.0 0.5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
10
90
76
181
1565
1
24
13
101
45World Knowledge
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All8
22
19
33
22522
3
77
9
39
149Answer Accuracy
1.0
 0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All1
5
9
17
342
1
10
7
20
17Internal Coherence
0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All9
12
128
88
164
30619
5
60
12
62
105Topic Relevance
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All2
7
34
21
40
23023
4
79
7
57
152Constraint Relevance
0.5
 0.0 0.5 1.0 1.5 2.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All11
13
109
93
173
33828
5
100
13
100
180Completeness
0.5
 0.0 0.5 1.0
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All10
9
89
66
179
18914
3
66
12
94
96Actionability
1 2 3 4 5
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All21
17
261
190
391
45240
7
174
41
216
259Memory Coverage
0.0 0.2 0.4 0.6
Average Score DifferenceClarifying
Answer
+Clarifying
Enriching
Distracting
Irrelevant
All11
14
125
80
180
35639
7
172
41
221
268Rouge-1 Recall
LongMemEval-m Long-MT-Bench+
Figure 19: Results of mean per-query score differences with confident intervals when adding a memory setting
relative to theparametric-only baselineon Llama-3.3-70B . Squares mark the mean difference; horizontal bars
show the confidence interval. Square size is proportional to the number of queries with a non-zero change, and the
number shown beside each square equals that count.
21

Instruction: ...
User query: I’m planning to visit Bandung again and I was wondering if you could remind me of the name of that restaurant in
Cihampelas Walk that serves a great Nasi Goreng?
Gold answer: Miss Bee Providore
Memory piece:
[User]: I think I’ll start with Cihampelas Walk to check out the denim street. Can you recommend any good restaurants
or cafes in the area?
[Assistant]: Sure! There are many good restaurants and cafes in the Cihampelas Walk area. Here are a few options: 1.
Miss Bee Providore: This restaurant serves a mix of western and Indonesian cuisine and has a cozy and stylish interior.
2. Bakery & Co: A great place to relax and enjoy a cup of coffee and a pastry. 3. Raa Cha Suki & BBQ: If you’re
looking for something spicy and delicious, this Thai restaurant is a great option. 4. Sumoboo: This Japanese shaved ice
dessert place features a wide variety of flavors, toppings, and combinations. 5. Holycow! Steakhouse by Chef Afit: This
steakhouse has a warm atmosphere and serves high-quality steaks at reasonable prices. These are just a few of the many
options you can explore in the area. Make sure to do some research and read reviews before making a final decision.
Enjoy your visit!Enriching
Memory piece:
[User]: I’m definitely going to try out Miss Bee Providore for some delicious food. Do they have any signature dishes I
should try?
[Assistant]: Miss Bee Providore offers a variety of delicious dishes, both Western and Indonesian cuisine. Here are a few
signature dishes you should definitely try: 1. Miss Bee’s Nasi Goreng: Their take on the classic Indonesian fried rice
dish is a must-try! It’s served with a side of prawn crackers and topped with a fried egg. 2. Yoghurt Granola Bowl: This
is a great option if you’re looking for something healthy and refreshing. It’s made with homemade granola, fresh fruit,
and yogurt. 3. Fried Chicken Burger: This signature burger features a juicy chicken patty, crispy bacon, lettuce, tomato,
and a special aioli sauce. It’s served with a side of fries. 4. Beef Rendang: If you’re looking for a traditional Indonesian
dish, try their version of beef rendang. It’s a slow-cooked beef dish in a flavorful, spicy coconut milk sauce. 5. Chocolate
Brownie: Finish off your meal with a sweet treat - Miss Bee’s chocolate brownie is a local favorite and is served warm
with vanilla ice cream and chocolate sauce. These are just a few of the many delicious options, so feel free to explore
their menu and try something new!Answer
Memory piece:
[User]: Yum, all of those dishes sound amazing! I’m definitely going to try the Miss Bee’s Nasi Goreng and finish it off
with the chocolate brownie.
[Assistant]: Enjoy your meal at Miss Bee Providore and have a great time shopping at Cihampelas Walk!Answer
Memory piece:
[User]: I’m so excited to explore more of Bandung! Do you have any recommendations for other fun activities to do in
the city?
[Assistant]: Sure, Bandung has plenty of fun activities to offer! Here are a few recommendations: 1. Visit a volcano:
Bandung is located near several active and dormant volcanoes, including Tangkuban Perahu and Kawah Putih. You can
take a day trip to visit these fascinating natural wonders. 2. Explore the tea plantations: Bandung is also known for its
tea plantations, such as the Malabar Tea Plantation and the Puncak Tea Garden. You can take a tour of these plantations,
enjoy a cup of tea, and enjoy the beautiful scenery. 3. Visit the Trans Studio Bandung: This is one of Indonesia’s largest
indoor theme parks, with rides, attractions, and entertainment for all ages. 4. Go shopping at the Paris Van Java mall:
This mall is known for its luxurious and trendy boutiques, restaurants, and cafes, as well as its modern architecture.
5. Take a cooking class: Bandung is famous for its cuisine, and you can take a cooking class to learn how to make
traditional Indonesian dishes such as nasi goreng, sate, and rendang. 6. Visit the Saung Angklung Udjo: This is a cultural
center that features traditional music and dance performances with the angklung, a traditional Indonesian instrument
made from bamboo. These are just a few of the many activities that you can do in Bandung. Make sure to do some
research and plan your trip accordingly, depending on your interests and preferences. Have fun exploring!Irrelevant
Memory piece:
[User]: What are some unique shopping experiences to be had in Bandung?
[Assistant]: As an AI language model, I don’t have personal experiences, but based on research, here are some unique
shopping experiences to be had in Bandung: 1. Factory outlets: Bandung is well known for being a factory outlet (FO)
paradise. You can find many factory outlets selling local and international brands at discounted prices. 2. Pasar Baru:
This is Bandung’s oldest and largest traditional market, selling everything from clothing and accessories to food and
home goods. It’s a great place to experience the hustle and bustle of a traditional market. 3. Braga Street: This historic
street is lined with old colonial buildings that have been turned into trendy cafes, restaurants, and boutique shops. It’s
a great place to find unique souvenirs and gifts. 4. Cihampelas Walk: Also known as Ciwalk, this shopping center is
famous for its denim street, where you can find a wide range of denim products, from jeans to jackets. 5. Dago Street:
This street is home to many boutiques, art galleries, and cafes, making it a popular spot for young and trendy shoppers.
6. Rumah Mode: This popular factory outlet is famous for its Balinese-inspired architecture and stylish clothing and
accessories. 7. Kartika Sari: This bakery is famous for its chocolate brownies and other Indonesian snacks, making it a
great place to pick up some tasty treats to take home.Irrelevant
Table 10: A judgment case for GPT-5-reasoning to determine the memory type. Left column is each memory
piece and right column is the determined type. See the full prompt in Figure 8.
22

Instruction: ...
User query: I’m planning to revisit Orlando. I was wondering if you could remind me of that unique dessert shop with the giant
milkshakes we talked about last time?
Gold answer: The Sugar Factory at Icon Park.
Memory piece:
[User]: Wow, these dining options sound amazing! Do you have any recommendations for a fun dessert spot that my
family can check out after dinner?
[Assistant]: Absolutely! Here are some fun dessert spots that your family might enjoy after dinner: 1. The Sugar Factory
- A sweet shop located at Icon Park that offers an enormous menu of sweet treats, including specialty drinks and giant
milkshakes. 2. Wondermade - A gourmet marshmallow shop located in Sanford, just north of Orlando, that offers unique
flavored marshmallows and s’mores kits. 3. Gideon’s Bakehouse - A bakery located at Disney Springs that offers delicious
cookies, cakes, and other baked treats. 4. Kelly’s Homemade Ice Cream - A small-batch ice cream shop located in Orlando
that offers unique flavors like Banana Cream Pie and Strawberry Jalapeno. 5. V oodoo Doughnut - A doughnut shop
located at Universal CityWalk that offers imaginative and creative doughnuts with unique toppings and flavors. 6. The
Glass Knife - A dessert and coffee shop located in Winter Park that serves up beautiful cakes, pastries, and other sweet
treats in a stylish setting. 7. Lickety Splitz - A dessert shop located in Orlando that serves up handmade frozen yogurt,
sorbet, and ice cream. 8. Better Than Sex - A dessert shop located in the Ivanhoe Village neighborhood of Orlando that
specializes in decadent desserts like chocolate fondue and cheesecake. 9. The Greenery Creamery - An ice cream shop
located in the Thornton Park neighborhood of Orlando that offers all-natural ice cream flavors like lavender, honeycomb,
and matcha. 10. The Pastry Studio - A bakery located in Daytona Beach, just east of Orlando, that offers beautiful and
delicious custom cakes and other desserts.Answer
Memory piece:
[User]: Can you recommend any places to eat in Orlando that are family-friendly?
[Assistant]: Certainly! Orlando has many family-friendly dining options. Here are some recommendations: 1. Toothsome
Chocolate Emporium & Savory Feast Kitchen - A restaurant located at Universal CityWalk that offers a unique dining
experience with imaginative dishes and delicious desserts. 2. The Cheesecake Factory - A casual dining restaurant located
in The Mall at Millenia that offers a diverse menu that includes everything from burgers to salads and, of course, an
extensive selection of cheesecakes. 3. Shake Shack - A fast-food restaurant located in Winter Park that offers burgers,
hot dogs, and shakes. 4. Hash House A Go Go - A restaurant located on International Drive that serves up ¨twisted
farm food ¨that is both delicious and visually stunning. 5. Bahama Breeze - A Caribbean-themed restaurant located on
International Drive that has a family-friendly atmosphere and offers a menu of tropical dishes and drinks. 6. Blaze Pizza -
A fast-casual restaurant located in Disney Springs that offers build-your-own Neapolitan pizzas and salads. 7. T-Rex Cafe -
A dinosaur-themed restaurant located in Disney Springs that offers a unique prehistoric dining experience for families. 8.
Uncle Julio’s - A Tex-Mex restaurant located in Orlando and Kissimmee that offers a menu of authentic Mexican dishes
and delicious drinks. 9. Pio Pio - A restaurant located in the Dr. Phillips neighborhood that serves up delicious Peruvian
dishes. 10. Sweet Tomatoes - A buffet restaurant chain located throughout Orlando that offers a variety of healthy and
delicious salads, soups, and baked goods.Distracting
Memory piece:
[User]: Wow, those tips are great! I’m excited to take on this dessert adventure with my family. Do you have any more
recommendations for other fun and unique things to do in Orlando?
[Assistant]: Yes, there are plenty of other fun and unique things to do in Orlando! Here are some more recommendations:
1. Escape Rooms - Orlando has several escape rooms that offer immersive and challenging puzzle-solving experiences. 2.
Cirque du Soleil - Watch a mesmerizing acrobatic and theatrical performance by the world-famous Cirque du Soleil, which
often performs in Orlando. 3. Airboat Tours - Take an airboat tour through the Florida Everglades and see alligators, birds,
and other wildlife up close. 4. The Orlando Eye - Take a ride on the 400-foot observation wheel and enjoy panoramic
views of Orlando. 5. Orlando Science Center - Visit the Orlando Science Center, which has interactive exhibits and
educational programs that entertain visitors of all ages. 6. The Orlando Museum of Art - See beautiful artwork from
various periods and artists at the Orlando Museum of Art. 7. Amusement Parks - Orlando has many amusement parks
besides Disney and Universal, such as Fun Spot America, ICON Park, and SeaWorld. 8. The Charles Hosmer Morse
Museum of American Art - See beautiful works of art by Louis Comfort Tiffany, including iconic stained glass lamps. 9.
Dr. Phillips Center for the Performing Arts - Enjoy a variety of dance, music and theater performances at this beautiful
venue. 10. The Orlando Shakespeare Theater - Experience the theater with classic and contemporary works in a unique
setting.Irrelevant
Memory piece:
[User]: I don’t really trust buffet restaurants. Can you suggest any other family-friendly dining options in Orlando?
[Assistant]: Sure, here are some additional family-friendly dining options in Orlando: 1. The Cowfish Sushi Burger
Bar - A unique restaurant located at Universal CityWalk that specializes in creative sushi and gourmet burgers. 2. BJ’s
Restaurant and Brewhouse - A casual dining restaurant located in Orlando that offers a diverse menu of American style
dishes and beers brewed on site. 3. Mediterranean Blue - A Greek restaurant located in Orlando that serves up classic
Greek dishes like souvlaki and gyros. 4. Black Bean Deli - A casual Cuban restaurant located in Orlando that serves
up delicious Cuban sandwiches, plantains, and more. 5. Toothsome Chocolate Emporium & Savory Feast Kitchen - As
mentioned before, this is a great family-friendly option for lunch or dinner. 6. Dandelion Communitea Cafe - A vegetarian
and vegan restaurant located in Orlando that serves up fresh and healthy salads, sandwiches, and smoothies. 7. The
Whiskey - A restaurant located in Orlando that has a family-friendly outdoor seating area and offers a menu of American
dishes with a Southern twist. 8. Blaze Pizza - Another option mentioned before, this fast-casual pizzeria is great for a quick
meal. 9. Kobe Japanese Steakhouse - A Japanese hibachi grill restaurant located in Orlando that offers an entertaining
dining experience and delicious seafood, meat, and vegetable dishes. 10. Soco Restaurant - A restaurant located in the
Thornton Park neighborhood of Orlando that offers Southern-style dishes with a modern twist. They also have a great
weekend brunch.Distracting
Table 11: A judgment case for GPT-5-reasoning to determine the memory type. Left column is each memory
piece and right column is the determined type. See the full prompt in Figure 8.
23

Instruction: ...
Prior conversation history: ...
User query: I wanted to check back on our previous conversation about Netflix. I mentioned that I wanted to be able to access
all seasons of old shows? Do you remember what show I used as an example, the one that only had the last season available?
Gold answer: Doc Martin.
Answer( k= 0 ): I’m sorry, I don’t have the ability to recall past conversations. However, if you let me know the name of the
show again, I can try to help you with any information or updates regarding its availability on Netflix.
World Knowledge Answer Accuracy Internal Coherence Topic Relevance Constraint Awareness Completeness Actionability
1 0 1 1 0 0 0
Answer( k= 5 ): Yes, I remember! You mentioned the show **"Doc Martin"** as an example where only the last season was
available on Netflix. If you have any more questions or need further assistance, feel free to ask!
1 1 1 1 1 2 0
Answer( k= 10 ): Yes, I remember! You mentioned that you wanted to have access to all seasons of old shows on Netflix, and
you used **"Doc Martin"** as an example, noting that only the last season was available.
1 1 1 1 1 2 0
Answer( k= 20 ): Yes, I remember! You mentioned that you wanted to have access to all seasons of old shows and used **"Doc
Martin"** as an example, noting that only the last season was available.
1 1 1 1 1 2 0
Answer( k= 50 ): Yes, you mentioned that you wanted to be able to access all seasons of old shows on Netflix, and you used
**"Doc Martin"** as an example, noting that it only had the last season available. If you have any more questions or need further
assistance, feel free to ask!
1 1 1 1 1 2 0
Answer( k= 100 ): Yes, I remember our conversation about Netflix. You mentioned that you wanted to be able to access all
seasons of old shows and used **"Doc Martin"** as an example, noting that it went down after a while, and now only the last
season is available. If you have any more questions or need further assistance, feel free to ask!
1 1 1 1 1 2 0
Table 12: A judgment case for GPT-3o to score the response on different metrics. We provide responses under
different retrieval size (k) and their scores.
24
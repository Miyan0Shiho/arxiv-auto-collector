# Comparing the Performance of LLMs in RAG-based Question-Answering: A Case Study in Computer Science Literature

**Authors**: Ranul Dayarathne, Uvini Ranaweera, Upeksha Ganegoda

**Published**: 2025-11-05 07:45:53

**PDF URL**: [http://arxiv.org/pdf/2511.03261v1](http://arxiv.org/pdf/2511.03261v1)

## Abstract
Retrieval Augmented Generation (RAG) is emerging as a powerful technique to
enhance the capabilities of Generative AI models by reducing hallucination.
Thus, the increasing prominence of RAG alongside Large Language Models (LLMs)
has sparked interest in comparing the performance of different LLMs in
question-answering (QA) in diverse domains. This study compares the performance
of four open-source LLMs, Mistral-7b-instruct, LLaMa2-7b-chat,
Falcon-7b-instruct and Orca-mini-v3-7b, and OpenAI's trending GPT-3.5 over QA
tasks within the computer science literature leveraging RAG support. Evaluation
metrics employed in the study include accuracy and precision for binary
questions and ranking by a human expert, ranking by Google's AI model Gemini,
alongside cosine similarity for long-answer questions. GPT-3.5, when paired
with RAG, effectively answers binary and long-answer questions, reaffirming its
status as an advanced LLM. Regarding open-source LLMs, Mistral AI's
Mistral-7b-instruct paired with RAG surpasses the rest in answering both binary
and long-answer questions. However, among the open-source LLMs, Orca-mini-v3-7b
reports the shortest average latency in generating responses, whereas
LLaMa2-7b-chat by Meta reports the highest average latency. This research
underscores the fact that open-source LLMs, too, can go hand in hand with
proprietary models like GPT-3.5 with better infrastructure.

## Full Text


<!-- PDF content starts -->

Comparing the Performance of LLMs in
RAG-based Question-Answering: A Case Study
in Computer Science Literature
Ranul Dayarathne1[0009−0008−8541−6086], Uvini Ranaweera1[0009−0003−6205−3482],
and Upeksha Ganegoda1[0000−0002−1745−9112]
University of Moratuwa, Katubedda, Moratuwa 10440, Sri Lanka
{dayarathnedvrn.19,ranaweeraraua.19,upekshag}@uom.lk
This version has been accepted for publication after peer review, but is not
theVersionofRecord.TheVersionofRecordisavailableathttps://doi.org/10.1007/978-
981-97-9255-9_26. Use of this Accepted Version is subject to Springer’s terms
of use.
Abstract.Retrieval Augmented Generation (RAG) is emerging as a
powerful technique to enhance the capabilities of Generative AI mod-
els by reducing hallucination. Thus, the increasing prominence of RAG
alongside Large Language Models (LLMs) has sparked interest in com-
paring the performance of different LLMs in question-answering (QA)
in diverse domains. This study compares the performance of four open-
source LLMs, Mistral-7b-instruct, LLaMa2-7b-chat, Falcon-7b-instruct
and Orca-mini-v3-7b, and OpenAI’s trending GPT-3.5 over QA tasks
within the computer science literature leveraging RAG support. Evalu-
ation metrics employed in the study include accuracy and precision for
binary questions and ranking by a human expert, ranking by Google’s
AI model Gemini, alongside cosine similarity for long-answer questions.
GPT-3.5, when paired with RAG, effectively answers binary and long-
answer questions, reaffirming its status as an advanced LLM. Regarding
open-source LLMs, Mistral AI’s Mistral-7b-instrucut paired with RAG
surpasses the rest in answering both binary and long-answer questions.
However, among the open-source LLMs, Orca-mini-v3-7b reports the
shortest average latency in generating responses, whereas LLaMa2-7b-
chat by Meta reports the highest average latency. This research under-
scores the fact that open-source LLMs, too, can go hand in hand with
proprietary models like GPT 3.5 with better infrastructure.
Keywords:Retrieval Augmented Generation·Large Language Models
·Question Answering
1 Introduction
Looking back, if one wanted to query something a few years back, the go-to
solution would be “Google”, and would end up reading through many retrieved
web sources to find the solution. Nowadays, Large Language Models (LLMs)
like GPT have made our lives easy, and is a common thing to say, “Let me askarXiv:2511.03261v1  [cs.CL]  5 Nov 2025

2 R. Dayarathne et al.
Chat-GPT for the answer”. Thus, by now, every one of us has tackled different
applications built on top of Generative Artificial Intelligence (GenAI) [14] and
has witnessed how they do a fair job in providing us with the required informa-
tion.
Since almost all the LLMs released so far are trained on billions of param-
eters on top of massive amounts of data, they can perform most of the natural
language tasks and cover an abundance of knowledge sources that were avail-
able back when the models were trained. Indirectly, this implies that if an LLM
is not regularly trained, it cannot retrieve the most up-to-date theories in the
field. For example, the latest update of ChatGPT 3.5 includes knowledge sources
only up to January 2022 [12], making it incapable of correctly capturing theo-
ries and concepts published after that. This has brought forward the problem
of hallucination, where a model generates confident answers that are factually
incorrect [39]. Therefore, it is not always a better option to use vanilla LLMs
to query for unknown facts, and most importantly, a vanilla LLM won’t be an
ideal option to query for unique concerns as they are not customised to fit such.
In order to customise LLMs to fit user needs, Retrieval Augmented Gen-
eration (RAG) is used widely around the globe. Patrick Lewis [20] introduced
RAG for Natural Language Processing (NLP) tasks in 2021, proposing to com-
bine non-parametric memory with pre-trained parametric-memory generation
models. In simple terms, RAG allows an LLM to retrieve data from external
databases specified by the user and keep up with the latest findings in the field.
This has brought many opportunities for multiple domains, and only a handful
of studies are being conducted to explore the applicability of RAG to enhance
domain-specific knowledge extraction [3,11,21].
One prominent field that can benefit from RAG-based LLM applications is
education; to be precise, RAG can be introduced to enhance the accessibility of
scholarly sources. Scholarly sources are works published by experts in a field to
share new findings and theories [34]. The rapid pace at which scholarly sources
evolve is beyond an individual’s time capacity to read through and comprehend
the findings. Thus, incorporating RAG on top of an LLM makes it possible to
facilitate Question-Answering (QA) so that the researchers can keep up with the
domain.
When it comes to research fields, they spread into numerous branches, where
computer science is one of the fields with the most significant scholarly contri-
butions [36]. Contributions in the computer science domain span from founda-
tional theories and algorithms to groundbreaking technologies such as Artificial
Intelligence (AI), which has revolutionised nearly every aspect of modern life.
In recent years, the computer science field has experienced a bloom owing to
notable advancements in AI, Quantum Computing, Robotics, Bioinformatics,
Virtual Reality, Augmented Reality, etc [9]. This sudden bloom has opened the
door for scientists worldwide to explore even beyond and contribute to further
advancements, leading to a significant increase in scholarly sources. Thus, cus-
tomising a language model to perform RAG on scholarly sources on computer

Performance of LLMs in RAG-based Question-Answering 3
science is no disadvantage, as it will help more people easily identify new trends
in the field and update their knowledge.
This paper presents a RAG-enabled QA system for the latest inventions in
the computer science domain. While RAG methodologies have found widespread
use, the integration and performance evaluation in the computer science domain
remains uncharted. The study fills the gap by conducting a comparative analy-
sis of the performance of multiple LLMs, including GPT 3.5, LLaMa-2-7b-chat,
Mistral-7b-instruct, Falcon-7b-instruct and Orca-mini-v3-7b, with and without
RAG integration. That’s not all; apart from answer quality, it is vital to under-
stand the usefulness, cost and efficiency of an answer. Thus, the study expands
its scope to cover those aspects of an LLM while employing a meticulously con-
structed database sourced from the latest publications in the computer science
domain.
The rest of the research paper is structured as follows. Section 2 reviews
the literature on RAG-based applications. Section 3 includes the methodology,
which presents the conceptual framework along with an overview of the dataset,
pre-processing steps, an overview of vectorising the dataset, steps to build the
QA pipeline and the performance measures used to evaluate the LLMs. Section 4
carries out an extensive comparison of the results obtained using different LLMs,
followedbysection5,whichdiscussesthelimitationsofthestudyandsuggestions
to enhance the work.
2 Literature Review
The applications of LLMs span from search engines to customer support to
translation and encompass a broad array of fields [40], as the capacity of an
LLM is immense when generating contextually fitting responses [10]. Ever since
LLMs became popular, RAG received attention from scholars as a remedy to
mitigate the hallucination that comes with an LLM due to its inability to grab
the most up-to-date data [26].
Y. Hicke et al. [11] introduced AI-TA, which is an intelligent QA assistance
for online QA platforms. They have evaluated the performance of their QA
pipeline incorporating LLaMa family’s LLaMA-2-13B (L-13) model, LLaMA-
2-70B model and OpenAI’s GPT-4 with augmenting techniques such as RAG,
supervised fine-tuning (SFT), and Direct Preference Optimization (DPO). The
results showed that the combination of L-13 with SFT, DPO and RAG showed
the most promising results, with a 107% increment in accuracy and a 177%
increment in usefulness against the base model performance.
As RAG remains one of the latest advancements in language modelling, there
is little research concerning enhancing domain-specific knowledge of LLMs with
RAG, where most of the existing research is focused on the biomedical field.
Quidwai et al. [30] proposed a pipeline that incorporates RAG on top of the
Mistral-7B model for QA on multiple myeloma. Markey et al. [21] carried out a
study to show how clinical trial documentation can be improved with RAG. Lála
etal.[18]proposedPaperQA,whichutilisedaRAGincorporatedLLMtoanswer

4 R. Dayarathne et al.
questions on biomedical literature. The evaluation of PaperQA’s performance
on the custom questions showed that it outperformed not only the LLMs such
as Claude-2 and GPT-4 but also the commercial tools Elicit, Scite, Perplexity
and Perplexity (Co-pilot), which are specifically designed to deal with scientific
literature. There are several more studies focused on specific areas of medicine,
such as nephrology [25], liver diseases [7], chest-x rays [32], ECG diagnosis [41]
etc.
Moving to other domains, Wiratunga et al. [38] introduced CBR-RAG, which
utilises case-based reasoning for legal QA, while Chouhan et al. [3] presented
LexDrafter, which assists in drafting definition articles using RAG. Zhang et
al. [42] presented a framework to predict the sentiment of financial incidents by
incorporating RAG and instruction-tuned LLMs, where the results improved the
accuracy of off-the-shelf LLMs like ChatGPT and LLaMa by 15% to 48%.
To date, there is no/less evidence of research proposing RAG to enhance
scholarlyknowledgeaccessibilityofthecomputersciencedomain;andtobemore
precise, no evidence is present in the computer science domain that evaluates the
performance of RAG-incorporated open-source LLMs and closed-source LLMs.
3 Methodology
As indicated in Fig. 1, a conceptual framework was designed to address the
research aim of evaluating the performance of LLMs in RAG-based QA in com-
puter science literature.
Fig. 1.Conceptual framework
3.1 Preparing the Dataset
Since off-the-shelf LLMs cannot perform RAG when answering user queries and
hence need to be customised to perform RAG, much attention was paid to in-
cluding the latest developments in the field when preparing the dataset. This
will help us demonstrate the importance of RAG over off-the-shelf LLMs that
are designed as parametric memory generation models [20].
Due to the limitations in accessing full papers and the limited computational
resources, it was decided to include only the abstracts of computer science jour-
nal papers. This decision relies on the assumption that an abstract of a paper
summarises the important findings of an entire study [4]. Moreover, Springer
and IEEE were chosen as sources to extract abstracts owing to their consistent

Performance of LLMs in RAG-based Question-Answering 5
reputation for publishing high-quality work in technology [29]. The compiled
dataset had 4929 abstracts from computer science journal papers focusing on
large language models, quantum computing and edge computing as they are the
latest trends in the domain [22]. The abstracts were extracted only from papers
published between 2023 - 2024 to ensure that they are the latest discussions
and that LLMs are not trained on top of those discussions. The dataset included
1121 abstracts for large language models, 1810 abstracts for edge computing and
1998 abstracts for quantum computing.
Along with the abstracts, it was decided to extract the title, author/s, pub-
lishing date and as well as keywords to support any future requirements such as
retrieval based on metadata.
3.2 Pre-Processing
Immediately after compiling the dataset, a few pre-processing steps were applied
to refine the data further. Some excerpts contained irrelevant HTML characters
since the abstracts were extracted from the World Wide Web (WWW). Thus,
during pre-processing, it was ensured to get rid of such noise in the dataset.
Apart from the major fix mentioned above, general pre-processing steps like
lower casing, dropping duplicated records, removing null values and removing
irrelevant characters were carried out to ensure that the dataset is standardised
with no/little noise. To detect and eliminate irrelevant characters such as colons,
semi-colons and extra white spaces, along with HTML tags, the Python library
“Regular Expressions” (re) was used.
The application of the above pre-processing steps introduced a standard for-
mat to abstracts with zero noise from unnecessary characters. Since the dataset
will be used on top of GenAI, no further pre-processing steps were applied, as
the LLMs already have the capability to grab the context. Following the pre-
processing steps, all the abstracts were stored as individual files in JSON format.
3.3 Vectorising the Data
To perform RAG using LLMs, it is a must to let LLM access an external data
source (a non-parametric memory) [20]. In our case, the external data source
will be the collection of abstracts from computer science journal articles.
Having these text files as they are in a database won’t smooth the retrieval
process. Thus, converting them into a format that language models can pro-
cess was necessary [5]. This was the point where text embedding came into the
spotlight. As defined by Rajvardhan Patil et al. [28], text embedding involves
convertingtextualdataintoanumericalformat(preciselyvectors)thatmachines
can process. Throughout the entire process of vectorising the data (Fig. 2), dif-
ferent features offered by the LangChain framework were used [19].
The conversion process included splitting standardised text excerpts into
chunks of 1024 characters maximum. The chunking limit was set such that the
content wouldn’t exceed the context window of any LLM considered in the un-
derlying study. To minimise the harm to the context of the abstract during

6 R. Dayarathne et al.
Fig. 2.Conversion process of the abstracts
the split, double line breaks ("\n\n") and full stops followed by a space (". ")
were defined as splitting points. Moreover, a chunk overlap of 200 characters
was allowed to ensure continuity among chunks. All in all, LangChain’s "Re-
cursiveCharacterTextSplitter" [19] was utilised for chunking as it allows content
splits based on a custom set of characters.
Next up was to embed the chunks. As the embedding model, the pre-trained
sentence transformer model SPECTER (allenai-specter) was used since it is
trained to produce embedding for scientific documents [4]. Unlike other trans-
formers, SPECTER incorporates inter-document context into the transformer
language models to learn document representations [4]. To give a brief idea of
the vectorisation process; SPECTER will encode the abstracts using the SciB-
ERT [2] transformer and then take the final representation of [CLS] token as the
final representation of the abstract [4] as indicated in Fig. 3.
Fig. 3.Overview of SPECTER vectorisation
Finally,theembeddingswerestoredusingtheFAISS(FacebookAISimilarity
Search) vector store, which is claimed to facilitate efficient similarity searches
for natural language queries [33]. It is worth noting that FAISS, implemented in
C++, is an open-source similarity search space developed by META that comes
with wrappers for Python [6].
3.4 Implementing the QA Pipeline
Upon building a vectorstore comprising abstracts from computer science litera-
ture, everything was set to design the QA pipeline. This is the most focused part
of the study as it involved many brainstorming sessions, from deciding the LLMs

Performance of LLMs in RAG-based Question-Answering 7
to engineering the model instructions. The intention behind the QA pipeline was
to get user queries as input, perform RAG on the vectorstore, and generate con-
cise answers with the support of an LLM. Fig. 4 paints an overall picture of how
the proposed QA pipeline works.
Fig. 4.Architecture of QA pipeline
Once a user inputs their query, the algorithm will check if a chat history
exists for the current session. If chat history exists, then the question and the
chat history will be passed to an LLM to get a refined search query, which is then
introduced to the retriever. From the point of retrieving data to the response
generation, the process will be similar, as explained below, irrespective of the
existence of chat history.
Regardless of the existence of a chat history, the user query will be sent
to both the retriever and the QA prompt simultaneously. Once the retriever
receives a user query, it will be embedded into a vector using the embedding
model SPECTER. Then, the query vector will be mapped to the chunks in the
vector store using a similarity search algorithm, where the threshold is set as
0.6 to retrieve only more similar chunks and k as 10 to return the top 10 most
similar chunks. Next, the retrieved chunks and the user query formatted by the
QA prompt are passed to an LLM to generate a concise response.
Utilising a vector store along with an LLM facilitates semantic search. Se-
mantics refer to the contextual meaning of a search query [37], and with a com-
bination of embeddings, vector indexes and a similarity algorithm as proposed
above, it is possible to carry out a semantic search [17].
Since this study aims to compare the performance among multiple LLMs, it
was decided to evaluate the QA capabilities of 5 of the most trending LLMs at
the time. Among the LLMs tested were the leading pay-as-you-go LLM GPT-3.5
and the open-source LLMs: LLaMa 2-7b-chat, Mistral-7b-instruct, Flacon-7b-

8 R. Dayarathne et al.
instruct and Orca-mini-v3-7b. It is worth noting that only quantised versions of
open-source LLMs were called to the local environment since they improve the
computational efficiency for a little trade-off in performance [8].
Regarding incorporating an LLM, there is another important aspect to point
out: prompt engineering. As McKinsey & Company [23] has elaborated, to get a
well-structured response using GenAI, it needs patience and iteration to provide
clear instructions that lead to a precise answer. Prompt engineering is the con-
cept of defining the ideal instruction for an LLM. Thus even in the underlying
study, a special attention was paid on prompt engineering to design a prompt
that generates answers for user queries using computer science literature. Fig. 5
indicatesthepromptpassedtoGPT-3.5togeneratelong-formanswerstoqueries
on quantum computing.
Fig. 5.Prompt designed for GPT-3.5 to query about quantum computing
Fig. 4 shows two prompts being called at two different instances. To clarify,
if a session already has a chat history, then the ‘chat_history’ prompt will be
passed with the chat history and the user query to generate an extended search
query to be integrated to the retriever. Apart from that, the QA prompt is called
regardless of the existence of a chat history to pass the user query and generate
an answer from the retrieved content.
The architectures offered by the LangChain community [19] were used to
design the above-discussed conversational RAG chain. In the proposed appli-
cation, three chains will be maintained: one to structure the retrieved content
(referred to as the format chain), another one to deal with conversation his-
tory (referred to as the history chain), and the other to perform RAG and
generate responses (referred to as the QA chain). For the format chain, “cre-
ate_stuff_documents_chain” was used, which will format all the retrieved con-
tents into a single prompt and pass it to an LLM [19]. Since there is a concern
about maintaining conversations (allowing a user to ask follow-up questions), for
the history chain, LangChain’s “create_history_aware_retriever” was used as
it takes chat history into account and creates a search query to be passed into
the retriever [19]. Both the history chain and format chain are called within the
QA chain built with “create_retrieval_chain”, which passes the user query to
the retriever, fetches relevant chunks and then sends the user query along with
retrieved chunks to the LLM for the response [19].

Performance of LLMs in RAG-based Question-Answering 9
With everything above set, it took just one function call to obtain the user’s
input and generate a concise answer.
3.5 Performance Evaluation
Since one of the major focuses of the study is to evaluate the performances
of multiple LLMs when employed with RAG, a custom dataset comprising 30
question-answer pairs was designed. These question-answer pairs were scripted
by two human experts covering the latest concerns in the computer science field
based on the same set of articles used to create the vector database. Further
to that, the answers were crafted by human experts based on the content of
the abstracts. The above mechanism of crafting QA pairs was inspired by the
work conducted by Lála et al. [18], who proposed 50 QA pairs using biomedical
literature that comes after September 2021 to check the functionality of their
RAG-enabled LLM. The QA pairs proposed in this study span across the three
research areas: quantum computing, large language models, and edge comput-
ing, with ten pairs for each area. The quantum computing QA set was explicitly
crafted with follow-up questions that required long-form answers. All the re-
maining QA pairs consisted only of Yes/No questions (binary questions). When
developing the dataset, it was assumed that the questions are novel and, thus,
have lower chances of being included in the training of base LLMs, requiring
RAG to generate accurate answers [18]. A few of the sample questions are indi-
cated in Table 1.
Table 1.Sample questions for evaluation.
Question Expected Answer Type
01.Arethereanystudiestoprovethatquantumalgorithms
outperform classical algorithms in portfolio optimisation?Yes/No
02. What are the results of those studies? Follow-up question
03. What is the goal of post-quantum cryptography? Long answer
With a custom QA dataset to test the performance of LLMs, two approaches
were adopted to evaluate these questions. When generating answers for the bi-
nary questions, LLMs were instructed to generate either “yes”, “no” or “do not
know”. Accuracy (1) and precision (2) were used to evaluate the LLMs’ perfor-
mance in answering binary questions.
Accuracy=No. of correct answers
Total no. of questions(1)
Precision=No. of correct answers
No. of confidently answered questions[18](2)
Average cosine similarity was used to evaluate LLM performances in an-
swering the questions that expect long-form answers. Cosine similarity (3) is a

10 R. Dayarathne et al.
similarity measure that calculates the distance between two embeddings [31] (for
the underlying study, it will be the candidate answer and the generated answer).
When the cosine similarity is one, it is stated that two vectors are identical,
while 0 will imply the two vectors are opposed [31]. Here, the average cosine
similarity will be the summation of similarity scores averaged across the total
number of questions.
Cosine Similarity[18] =A·B
∥A∥∥B∥(3)
Apart from that, long-answer questions were evaluated by a human expert
andtheAIchatbotGemini,whichwasdevelopedbyGoogle.Boththehumanand
AI bot were asked to compare the generated answer with an answer candidate
andratetheusefulnessofthegeneratedanswer.Theratingsincludedthreelevels:
poor, average, and excellent. Poor points to the answers that do not meet the
minimum expectations of the answer candidate, while excellent means answers
that satisfy all the expectations of the answer candidate.
All the above-mentioned evaluation metrics will consider the usefulness of an
answer. Thus, keeping the usefulness of the answers aside for a moment, towards
theendofthestudyitwasdecidedtofocusonlatency(timetakentogeneratean
answer) and the cost incurred in generating the answer as some side parameters
supporting further comparison between the LLMs.
4 Results and Discussion
This section will comprehensively evaluate the performance of the LLMs in an-
swering binary questions and long-answer questions. All the trials were carried
out on a MacBook Pro with an M2 processor and 16GB of RAM. Moreover,
same values were set to the parameters of all the LLMs to ensure comparability.
The LLM temperature was set to 0.01 since the underlying task is fact-based
QA, which does not encourage creative answers but rather more concise an-
swers, while the ‘max_tokens’, which defines the number of tokens the model
will process, was set to 2000.
With the above configurations, the results obtained under each LLM for bi-
nary questions and long-answer questions are indicated in Table 2 and Table 3.
Table 2 indicates the accuracy and precision calculated after evaluating the re-
sults generated for 21 binary questions. Accordingly, it is observable that GPT
3.5 combined with RAG has reported the highest accuracy and precision. This
will reconfirm that GPT is the all-time best LLM for most natural language
tasks. The reason behind its remarkable performance is that GPT-3.5 is trained
on 175 billion parameters [44], which is more significant when compared to most
of the existing LLMs.1However, as indicated in Table 4, GPT 3.5 has a cost. To
1GPT-4 is trained on even more parameters thus, it will definitely surpass GPT 3.5.
Since, in this study, the focus is open-source LLMs, the inclusion of GPT 3.5 is to
give an overall idea of how the results would vary if we go for a larger LLM.

Performance of LLMs in RAG-based Question-Answering 11
generate answers for 30 questions with a maximum of 2000 tokens in each run,
it costs $0.000643.
Moving to the open-source LLMs and their performance in answering the
binary questions, the Mistral-7b-instruct model combined with RAG surpassed
the rest in accuracy and precision. Compared to GPT-3.5, Mistral-7b-instruct,
with 7 billion training parameters, has reported an accuracy of 0.857, whereas
GPT 3.5+RAG recorded its accuracy as 0.9048. This is a promising result for an
open-source LLM and highlights that these LLMs can be relied upon to address
most of the business needs. These LLMs are entirely free; thus, no financial
cost was involved in generating answers. However, in Table 4, it is clear that
GPT-3.5 has taken only 1.7413 seconds on average to create an answer, whereas
Mistral-7b-instruct took 105.9543 seconds on average to generate an answer.
Thus, the main trade-off is between latency and the cost involved, while a small
trade-off concerns the answer’s usefulness. However, it is worth noting that an
enhancement to local hardware resources will help to reduce the latency of open-
source LLMs (ex: accelerating CPU to GPU).
Following the Mistral-7b-instruct, the Orca-mini-v3-7b model also shows
promising results for binary questions with a recorded accuracy of 0.8095. When
it comes to precision (the total number of correct answers over confidently an-
sweredquestions),bothLLaMa-2-7b-chatandFalcon-7b-instructshowrelatively
lower performances. This implies; though these models are compiled with RAG,
there are still factually incorrect yet confidently answered questions. This is
called hallucination, where LLMs generate plausible-sounding but unfaithful or
nonsensical information [15].
Table 2.Performance metrics for binary questions.
LLM Accuracy Precision
GPT 3.5 + RAG 0.9048 0.9048
Orca + RAG 0.8095 0.8095
LLaMa 2 + RAG 0.6190 0.6842
Falcon + RAG 0.619 0.6842
Mistral + RAG 0.8571 0.8571
Chat-GPT 0.4761 0.625
To evaluate long answers, this study utilised a cosine similarity score, an
AI-generated rank (Gemini), and a human-expert evaluated rank across. Cosine
similarity values indicated in Table 3 are relatively low, signalling that the gen-
erated answers are different from the answer candidates in terms of similarity.
This can be due to the rich vocabulary of LLMs that lets them generate answers
more creatively. As per the results indicated in Table 3 GPT 3.5+RAG, has
the highest cosine similarity score of 0.4479, and among the open-source LLMs,
Mistral-7b-instruct has the highest score of 0.2754.
Since the cosine similarity scores are lower, it alone can not evaluate the per-
formance of LLMs in long-form question answering. Therefore an AI-generated

12 R. Dayarathne et al.
rank and a human-evaluated rank were combined to support the overall evalua-
tion. For the human evaluation, an expert in the quantum computing field was
employed. The expert was provided with the question, the answer candidate
and the relevant abstract and was asked to evaluate the generated answer with
the help of the provided content and her expertise. To get Gemini rankings, a
clearly crafted instruction was given to compare the answer candidate and the
generated answer and rank the generated answer based on its quality.
As indicated in Table 3, Mistral-7b-instruct has performed well compared to
other open-source LLMs, with 6 out of 9 answers being ranked excellent by both
Gemini and the human expert. From the open-source LLMs being compared,
the LLM with the weakest performance in terms of similarity score was Falcon-
7b-instruct and was the same under both Gemini rankings and human rankings.
This indirectly implies that language tools like Gemini are optimised consider-
ably to replace the human capacity to understand the context. All in all, among
the LLMs tested, GPT 3.5 combined with RAG, has a better performance in
long-form question answering (Table 3).
Table 3.Performance metrics for long-answer questions.
LLMCosine
SimilarityAI Rank (Gemini) Human Rank
Poor Average Excellent Poor Average Excellent
GPT 3.5 + RAG 0.2291 3 5 12 8 3 9
Orca + RAG 0.1806 7 6 7 11 3 6
LLaMa 2 + RAG 0.2263 4 3 13 7 6 7
Falcon + RAG 0.2326 4 9 7 11 3 6
Mistral + RAG 0.2339 3 7 10 5 5 10
ChatGPT 0.0977 7 5 8 12 4 4
There is an important fact to notice about the evaluation of long-form an-
swers. Since the Gemini-AI does not have the privilege to access recent knowl-
edge and the fact that it has informed knowledge [27], a slight change to the
instructions resulted in contradictory ranks questioning the biasness in Gem-
ini generated rankings. Moreover, if the same question was repeated multiple
times, in some instances the AI-generated rank changed due to the associated
complexity and variability of responses. Further to that, at present the study
employs only a single human to get the expert opinion. Thus her understanding
of quantum computing and the point-of-view might include a slight bias in the
above rankings. However, a comparison between the expert-generated rankings
and the AI-generated ranking shows that the rankings are almost line-in-line
leaving a little or zero bias in the evaluation scores. As the study expands, it is
proposed to utilise a pool of human expertise and average out their rankings to
reduce the bias in human score.
ThoughthemainfocuswastocomparetheperformancesofLLMswhencom-
bined with RAG, it was decided to test how the answers would look like without
RAG. Therefore, ChatGPT, which is OpenAI’s most demanding chatbot built

Performance of LLMs in RAG-based Question-Answering 13
on GPT-3.5, was asked the same questions. The same prompt template and a
new instance of a chat were used to generate answers for each question. Under
this scenario, the results obtained for accuracy and precision were the worst com-
pared to all the other RAG-enabled LLMs. Though ChatGPT is one of the best
AI chatbots, it lacks the skill to answer questions on recent content, underscoring
that off-the-shelf models are not carrying out RAG if not customised.
Keeping GPT-3.5 aside, comparing the latency among the open-source LLMs
shows that the Orca-mini-v3-7b model surpasses the rest with an average latency
of99.2167seconds(Table4).Outofalltheopen-sourceLLMstestedinthestudy,
Meta’s LLaMA-2-7b-chat has the highest average latency in generating answers
(Table 4).
Table 4.Average latency and average cost across LLMs.
LLM Average Latency Average Cost
GPT-3.5 + RAG 1.7413 0.000643
Orca + RAG 99.2167 No cost involved
LLaMa2 + RAG 107.4748 No cost involved
Falcon + RAG 107.31642 No cost involved
Mistral + RAG 105.9543 No cost involved
It is worth noting that some LLMs required prompts to follow a specific for-
mat to generate better answers (ex: Mistral-7b-instruct and falcon-7b-instruct).
In contrast, others (ex: GPT-3.5) understand natural language easily without
needing a specifically structured prompt. In the underlying study, when given a
prompt to Mistral-7b-instruct without following its template2, it failed to gen-
erate answers as expected. However, structuring the prompt to fit the template
made it possible to improve the answer, as indicated in Table 5.
From the above discussion, it is obvious that incorporating RAG on top of
an LLM allows a user to query publications in the recent past. Thus, unlike an
off-the-shelf LLM, a RAG-based QA system powered by an LLM for computer
science literature has the special ability to answer user queries in the computer
science domain with the most recent and valid content while ensuring the novelty
of the answer.
Further to that, looking into Mistral’s success and Falcon’s failure as open-
source LLMs, the difference in attention mechanisms can be pointed out. The
attention mechanism allows a model to focus on the most relevant parts of the
input when producing the output [35]. Llama2 utilises the vanilla multi-head
attention (MHA) which runs through the attention mechanism several times
in parallel requiring a high memory bandwidth [35]. Falcon uses a multi-query
attention (MQA) mechanism [1] which is more efficient in memory load at the
costofoutputquality.Ontheotherhand,Mistralusesagroupedqueryattention
(GQA) which results in higher throughput by allowing for higher batch sizes.
2<s>[INST] instructions [/INST] answer </s>[INST] Follow-up instructions [/INST]

14 R. Dayarathne et al.
Table 5.Comparison between structured & unstructured prompts for Mistral.
QuestionAre there any studies proposing Lightweight Unified Collaborated Relin-
quish Edge Gateway Architecture with isochronal patching?
Answer
(not fol-
lowing the
prompt
template)Based on the context provided, I cannot directly answer that question
with a simple yes or no. However, I can tell you that the text you pro-
videddiscussesvariousaspectsofedgecomputing,includingarchitectures,
advantages, enabling technologies, and standardization. However, it does
not specifically mention or propose a "Lightweight Unified Collaborated
Relinquish Edge Gateway Architecture with isochronal patching" as such.
If you have more specific information or context, please provide it and I
will do my best to help you with your question.
Answer
(following
the prompt
template)Based on the provided context, there is indeed a study that proposes a
lightweightunifiedcollaboratedrelinquish(LUCR)edgegatewayarchitec-
turewithisochronalpatching.Thisstudyaimstomanagethedynamicup-
dation process and increase the processing capacity and isochronal patch-
ing is utilized to prevent backdoor functionality problems and also man-
age the considerable amount of hardware and software resources with its
lightweight architecture. However, it’s important to note that this is just
one study, and there may be other approaches and architectures that also
address the challenges and requirements of managing and processing data
at the edge.
Moreover, the sliding window attention in Mistral is designed to handle longer
sequencesmoreeffectivelyatareducedcomputationalcost[16].AsGQAstandin
thebalancebetweenMQAandMHA,itisfairtoexpectasignificantperformance
improvementfromtheMistralmodelintermsofbothanswerqualityandlatency.
5 Conclusion
Question-answering systems are used widely worldwide in e-learning platforms,
social media applications, customer support platforms, and more. It has ex-
panded its horizons with the introduction of LLMs owing to their creativity and
the ability to understand natural language queries. One area that could benefit
from QA is scientific literature. As it is time-consuming for the researchers to go
through existing literature one by one, a system that allows them to query the
published work will be of immense advantage. However, simply incorporating an
off-the-shell LLM won’t generate results as expected due to the limitations in
training. This study has integrated RAG on LLMs to assess their performance
in question answering on computer science literature.
LLM-based chatbots still struggle with hallucination because of the limita-
tions they have in accessing recent knowledge bases [43], leading chatbots to
provide false or misleading answers. For example, when it comes to ChatGPT,
it gives incorrect information to questions that were raised in the recent past be-
cause it has been trained on data till January 2022. Further, it is not proficient in
providing references to the most recent publications, which doubts the validity
of the answers generated [13]. By introducing RAG to LLMS, these limitations

Performance of LLMs in RAG-based Question-Answering 15
with outdated training data can be eliminated. The study shows that GPT-3.5
integrated with RAG delivers better answers than standalone ChatGPT. As re-
cently published (from 2023 to 2024) scholarly articles on computer science were
used to implement RAG, LLMs could answer the questions connected with the
recent past. Also, they can prove the answers by referring to literature. To sum-
marise, integrating RAG enables LLMs to access recent knowledge sources to
generate answers. With the rich textual context that LLMs hold, it can provide
more creative, non-hallucinated, logically correct answers.
With the emergence of LLMs, prompt engineering has become a trending
topic. It underlies the concept of designing, refining, and implementing prompts
for LLMs to deliver optimal outputs [24]. Therefore, it is fair to state that a
prompt affects the quality of outputs generated by LLMs, and it is essential to
provide clear and precise instructions for LLMs to perform the tasks effectively.
Even in the underlying study, it was required to develop multiple prompts before
choosing a single prompt to instruct the LLMs. It appears as another area that
needs a thorough evaluation. Thus, the prompt used for the study might not be
optimal, but the results prove that it certainly is a good enough option.
As the study revolves mainly around open-source LLMs, the users do not
need to worry about the cost. However, the most significant limitation in most of
these open-source LLMs is the latency that comes with them. This can be easily
avoided by going for a subscription-based LLM that enables LLM invocation
through infrastructure provided by the hosting company. On the other hand,
one can accelerate their local infrastructure when calling an open-source LLM
to reduce its latency. However, the cost associated with each instance should
be considered depending on the use case. For example, in this study, an LLM
is combined with RAG to allow individuals interested in computer science to
query published literature easily. Suppose the application values accuracy above
cost, then, instead of accelerating infrastructure to host an open-source LLM, it
is advised to go for a model in the GPT family. If the concern is more towards
data privacy, a GPU-accelerated infrastructure will be the ideal option, as it will
letyourunanopen-sourceLLMefficientlyinyourlocalenvironment.However,if
the intention is personal use, incorporating a Mistral-7b-instruct-like model with
RAG under the available infrastructure will generate quality answers within a
reasonable time.
Overall, this study provides a comprehensive evaluation of the performance
of some of the most trending LLMs in question-answering on computer science
literature. Among the models compared, the study concludes that GPT 3.5 is
the best-performing LLM overall, and Mistral-7b-chat is the best in open-source
LLMs. However, since the field of LLMs keeps evolving, and different models
emerge with diverse parameters and capabilities, it is essential to keep up with
the field and check for the usability of emerging LLMs before deciding on an
LLM to be integrated into your application.
Lastly, the underlying study allows future researchers to extend it further by
incorporating prompt engineering, trying out different mechanisms to retrieve

16 R. Dayarathne et al.
data in contrast to vector stores, utilising full-text research articles rather than
limiting to abstracts and even expanding its applicability to other domains.
References
1. Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah,
M., Goffinet, É., Hesslow, D., Launay, J., Malartic, Q., et al.: The falcon series of
open language models. arXiv preprint arXiv:2311.16867 (2023)
2. Beltagy, I., Lo, K., Cohan, A.: Scibert: A pretrained language model for scientific
text. arXiv preprint arXiv:1903.10676 (2019)
3. Chouhan,A.,Gertz,M.:Lexdrafter:Terminologydraftingforlegislativedocuments
using retrieval augmented generation. arXiv preprint arXiv:2403.16295 (2024)
4. Cohan, A., Feldman, S., Beltagy, I., Downey, D., Weld, D.S.: Specter: Document-
level representation learning using citation-informed transformers. arXiv preprint
arXiv:2004.07180 (2020)
5. Egger, R.: Text representations and word embeddings: Vectorizing textual data.
In: Applied data science in tourism: Interdisciplinary approaches, methodologies,
and applications, pp. 335–361. Springer (2022)
6. Engineering at Meta: Faiss: A library for efficient similarity search. Web (Last
accessed 2024/04/20), https://engineering.fb.com
7. Ge, J., Sun, S., Owens, J., Galvez, V., Gologorskaya, O., Lai, J.C., Pletcher, M.J.,
Lai, K.: Development of a liver disease-specific large language model chat interface
using retrieval augmented generation. medRxiv (2023)
8. Gong, Z., Liu, J., Wang, J., Cai, X., Zhao, D., Yan, R.: What makes quantization
for large language model hard? an empirical study from the lens of perturbation.
In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp.
18082–18089 (2024)
9. Great Learning Team: Latest technologies in computer science in 2024.
Web (Last accessed 2024/04/18), https://www.mygreatlearning.com/blog/latest-
technologies-in-computer-science
10. Hadi, M.U., Qureshi, R., Shah, A., Irfan, M., Zafar, A., Shaikh, M.B., Akhtar,
N., Wu, J., Mirjalili, S., et al.: A survey on large language models: Applications,
challenges, limitations, and practical usage. Authorea Preprints (2023)
11. Hicke, Y., Agarwal, A., Ma, Q., Denny, P.: Chata: Towards an intelligent question-
answer teaching assistant using open-source llms. arXiv preprint arXiv:2311.02775
(2023)
12. Huynh, L.M., Bonebrake, B.T., Schultis, K., Quach, A., Deibert, C.M.: New artifi-
cialintelligencechatgptperformspoorlyonthe2022self-assessmentstudyprogram
for urology. Urology Practice10(4), 409–415 (2023)
13. Jahanbakhsh, K., Hajiabadi, M., Gagrani, V., Louie, J., Khanwalkar, S.: Beyond
hallucination: Building a reliable question answering & explanation system with
gpts
14. Jeong, C.: Generative ai service implementation using llm application architec-
ture: based on rag model and langchain framework. Journal of Intelligence and
Information Systems29(4), 129–164 (2023)
15. Ji, Z., Yu, T., Xu, Y., Lee, N., Ishii, E., Fung, P.: Towards mitigating hallucination
in large language models via self-reflection. arXiv preprint arXiv:2310.06271 (2023)
16. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., Casas,
D.d.l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.: Mistral 7b. arXiv
preprint arXiv:2310.06825 (2023)

Performance of LLMs in RAG-based Question-Answering 17
17. Kenter, T., De Rijke, M.: Short text similarity with word embeddings. In: Proceed-
ings of the 24th ACM international on conference on information and knowledge
management. pp. 1411–1420 (2015)
18. Lála, J., O’Donoghue, O., Shtedritski, A., Cox, S., Rodriques, S.G., White,
A.D.: Paperqa: Retrieval-augmented generative agent for scientific research. arXiv
preprint arXiv:2312.07559 (2023)
19. langchain: Langchain: Build context-aware reasoning applications. Web (Last ac-
cessed 2024/04/20), https://github.com/langchain-ai/langchain
20. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems
33, 9459–9474 (2020)
21. Markey, N., El-Mansouri, I., Rensonnet, G., van Langen, C., Meier, C.: From rags
to riches: Using large language models to write documents for clinical trials. arXiv
preprint arXiv:2402.16406 (2024)
22. Maryville University: 4 newest technology trends in computer science. Web (Last
accessed 2024/04/20), https://online.maryville.edu
23. McKinsey & Company : What is prompt engineering? Web (Last ac-
cessed 2024/04/20), https://www.mckinsey.com/featured-insights/mckinsey-
explainers/what-is-prompt-engineering
24. Meskó, B.: Prompt engineering as an important emerging skill for medical profes-
sionals: tutorial. Journal of Medical Internet Research25, e50638 (2023)
25. Miao, J., Thongprayoon, C., Suppadungsuk, S., Garcia Valencia, O.A., Cheungpa-
sitporn, W.: Integrating retrieval-augmented generation with large language mod-
els in nephrology: advancing practical applications. Medicina60(3), 445 (2024)
26. Ni, S., Bi, K., Guo, J., Cheng, X.: When do llms need retrieval augmenta-
tion? mitigating llms’ overconfidence helps retrieval augmentation. arXiv preprint
arXiv:2402.11457 (2024)
27. Nyaaba, M.: Generative ai conception of the nature of science. In: Society for
Information Technology & Teacher Education International Conference. pp. 1818–
1827. Association for the Advancement of Computing in Education (AACE) (2024)
28. Patil, R., Boit, S., Gudivada, V., Nandigam, J.: A survey of text representation
and embedding techniques in nlp. IEEE Access (2023)
29. Priya, D.: Data science paper publication: Ieee vs springer. Web (Last accessed
2024/04/20), https://www.analyticsinsight.net/data-science-paper-publication-
ieee-vs-springer/
30. Quidwai, M.A., Lagana, A.: A rag chatbot for precision medicine of multiple
myeloma. medRxiv pp. 2024–03 (2024)
31. Radeva, I., Popchev, I., Doukovska, L., Dimitrova, M.: Web application for
retrieval-augmented generation: Implementation and testing. Electronics13(7),
1361 (2024)
32. Ranjit, M., Ganapathy, G., Manuel, R., Ganu, T.: Retrieval augmented chest x-ray
report generation using openai gpt models. In: Machine Learning for Healthcare
Conference. pp. 650–666. PMLR (2023)
33. Singh, P.N., Talasila, S., Banakar, S.V.: Analyzing embedding models for embed-
ding vectors in vector databases. In: 2023 IEEE International Conference on ICT
in Business Industry & Government (ICTBIG). pp. 1–7. IEEE (2023)
34. University of Toronto: What counts as a scholarly source? Web (Last accessed
2024/04/18), https://onesearch.library.utoronto.ca/faq/what-counts-scholarly-
source

18 R. Dayarathne et al.
35. Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,Kaiser,
Ł., Polosukhin, I.: Attention is all you need. Advances in neural information pro-
cessing systems30(2017)
36. Wang, Z., Ma, D., Pang, R., Xie, F., Zhang, J., Sun, D.: Research progress and
development trend of social media big data (smbd): knowledge mapping analysis
based on citespace. ISPRS International Journal of Geo-Information9(11), 632
(2020)
37. Wei, W., Barnaghi, P.M., Bargiela, A.: Search with meanings: an overview of se-
mantic search systems. International journal of Communications of SIWN3, 76–82
(2008)
38. Wiratunga, N., Abeyratne, R., Jayawardena, L., Martin, K., Massie, S., Nkisi-
Orji, I., Weerasinghe, R., Liret, A., Fleisch, B.: Cbr-rag: Case-based reasoning for
retrieval augmented generation in llms for legal question answering. arXiv preprint
arXiv:2404.04302 (2024)
39. Xu, Z., Jain, S., Kankanhalli, M.: Hallucination is inevitable: An innate limitation
of large language models. arXiv preprint arXiv:2401.11817 (2024)
40. Yao, Y., Duan, J., Xu, K., Cai, Y., Sun, Z., Zhang, Y.: A sur-
vey on large language model (llm) security and privacy: The good,
the bad, and the ugly. High-Confidence Computing4(2), 100211 (2024).
https://doi.org/https://doi.org/10.1016/j.hcc.2024.100211
41. Yu, H., Guo, P., Sano, A.: Zero-shot ecg diagnosis with large language models
and retrieval-augmented generation. In: Machine Learning for Health (ML4H). pp.
650–663. PMLR (2023)
42. Zhang, B., Yang, H., Zhou, T., Ali Babar, M., Liu, X.Y.: Enhancing financial
sentiment analysis via retrieval augmented large language models. In: Proceedings
of the Fourth ACM International Conference on AI in Finance. pp. 349–356 (2023)
43. Zhuang, Y., Yu, Y., Wang, K., Sun, H., Zhang, C.: Toolqa: A dataset for llm
question answering with external tools. Advances in Neural Information Processing
Systems36(2024)
44. Zong, M., Krishnamachari, B.: Solving math word problems concerning systems
of equations with gpt-3. In: Proceedings of the AAAI Conference on Artificial
Intelligence. vol. 37, pp. 15972–15979 (2023)
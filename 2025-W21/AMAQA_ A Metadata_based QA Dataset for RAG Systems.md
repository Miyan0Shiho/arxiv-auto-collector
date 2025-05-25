# AMAQA: A Metadata-based QA Dataset for RAG Systems

**Authors**: Davide Bruni, Marco Avvenuti, Nicola Tonellotto, Maurizio Tesconi

**Published**: 2025-05-19 08:59:08

**PDF URL**: [http://arxiv.org/pdf/2505.13557v1](http://arxiv.org/pdf/2505.13557v1)

## Abstract
Retrieval-augmented generation (RAG) systems are widely used in
question-answering (QA) tasks, but current benchmarks lack metadata
integration, hindering evaluation in scenarios requiring both textual data and
external information. To address this, we present AMAQA, a new open-access QA
dataset designed to evaluate tasks combining text and metadata. The integration
of metadata is especially important in fields that require rapid analysis of
large volumes of data, such as cybersecurity and intelligence, where timely
access to relevant information is critical. AMAQA includes about 1.1 million
English messages collected from 26 public Telegram groups, enriched with
metadata such as timestamps, topics, emotional tones, and toxicity indicators,
which enable precise and contextualized queries by filtering documents based on
specific criteria. It also includes 450 high-quality QA pairs, making it a
valuable resource for advancing research on metadata-driven QA and RAG systems.
To the best of our knowledge, AMAQA is the first single-hop QA benchmark to
incorporate metadata and labels such as topics covered in the messages. We
conduct extensive tests on the benchmark, establishing a new standard for
future research. We show that leveraging metadata boosts accuracy from 0.12 to
0.61, highlighting the value of structured context. Building on this, we
explore several strategies to refine the LLM input by iterating over provided
context and enriching it with noisy documents, achieving a further 3-point gain
over the best baseline and a 14-point improvement over simple metadata
filtering. The dataset is available at
https://anonymous.4open.science/r/AMAQA-5D0D/

## Full Text


<!-- PDF content starts -->

AMAQA: A Metadata-based QA Dataset for RAG Systems
Davide Bruni⋄, Marco Avvenuti†, Nicola Tonellotto†, Maurizio Tesconi⋄
⋄Institute for Informatics and Telematics, National Research Council, Italy
davide.bruni@iit.cnr.it, maurizio.tesconi@iit.cnr.it
†Department of Information Engineering, University of Pisa, Italy
marco.avvenuti@unipi.it, nicola.tonellotto@unipi.it
Abstract
Retrieval-augmented generation (RAG) sys-
tems are widely used in question-answering
(QA) tasks, but current benchmarks lack
metadata integration, hindering evaluation
in scenarios requiring both textual data and
external information. To address this, we
present AMAQA, a new open-access QA
dataset designed to evaluate tasks combin-
ing text and metadata. The integration of
metadata is especially important in fields
that require rapid analysis of large volumes
of data, such as cybersecurity and intelli-
gence, where timely access to relevant in-
formation is critical. AMAQA includes
about 1.1 million English messages col-
lected from 26 public Telegram groups, en-
riched with metadata such as timestamps,
topics, emotional tones, and toxicity indica-
tors, which enable precise and contextual-
ized queries by filtering documents based on
specific criteria. It also includes 450 high-
quality QA pairs, making it a valuable re-
source for advancing research on metadata-
driven QA and RAG systems. To the best of
our knowledge, AMAQA is the first single-
hop QA benchmark to incorporate meta-
data and labels such as topics covered in
the messages. We conduct extensive tests
on the benchmark, establishing a new stan-
dard for future research. We show that
leveraging metadata boosts accuracy from
0.12 to 0.61, highlighting the value of struc-
tured context. Building on this, we explore
several strategies to refine the LLM input
by iterating over provided context and en-
riching it with noisy documents, achieving
a further 3-point gain over the best base-
line and a 14-point improvement over sim-
ple metadata filtering. The dataset is avail-
able at https://anonymous.4open.
science/r/AMAQA-5D0D/1 Introduction
Retrieval-augmented generation (RAG) systems
are increasingly utilized in knowledge-driven ap-
plications, including question answering, infor-
mation extraction, and dialogue generation (Gao
et al., 2023). While existing benchmarks have sig-
nificantly advanced the evaluation of RAG sys-
tems, they primarily focus on text-based inputs,
overlooking the potential of metadata, such as en-
gagement metrics or message timestamps, along-
side labels like topics, emotional tone, or toxicity.
However, the ability to use metadata effectively
is particularly important in fields where the con-
text surrounding textual data is as important as the
content itself (Pelofske et al., 2023; Perez et al.,
2018). For example, analysts often must identify
relevant information from huge datasets by filter-
ing contents based on metadata. For instance, in
this dataset, analysts could identify relevant infor-
mation by filtering messages based on the time pe-
riod or channel where the message was posted, and
the tone or intent of the text itself. Consider a sce-
nario similar to the one reported in Figure 1, in
which an analyst must answer a question such as:
"Who is mentioned as LeBron hater in toxic mes-
sages posted on 2024-06-16 talking about Lakers
on the ‘Pilot Blog Chat’ group chat?"
This question introduces multiple layers of
complexity:
•Subject identification and attribute extrac-
tion: The system must identify who is re-
ferred to as a "LeBron hater", focusing on
extracting a specific characteristic associated
with an individual in the analyzed messages.
•Metadata filters: Two filters expressed in nat-
ural language must be applied to the meta-
data:
-"posted on 2024-06-16" requires the system
to select messages sent on a specific date.arXiv:2505.13557v1  [cs.IR]  19 May 2025

-"on the ‘Pilot Blog Chat’" limits the scope
to messages from the specified group chat.
These filters can be directly translated into
queries over the metadata, narrowing down
the set of relevant messages.
•Textual features not based on metadata: Two
other components of the question cannot be
resolved solely using metadata: "in toxic
messages" requires the system to recognize
toxic messages automatically. While "talking
about Lakers" requires the system to detect
messages related to the Los Angeles Lakers
basketball team, even without explicit men-
tions. Although toxicity and topic labels are
present in the dataset, they are used only for
evaluation, not for filtering.
Figure 1 : High-level conceptual schema of the sys-
tems analyzed in this work
The system must process textual features along-
side the metadata constraints to identify the rele-
vant subset of messages. The system’s response
to the question is "Charles Barkley" , which is cor-
rect because the dataset contains a message sent
on June 16, 2024, in the "Pilot Blog Chat" group,
identified as a toxic comment, discussing about
Lakers and it mentions him as a "LeBron hater".
Although this may represent misinformation, it
is considered correct by the system since it ac-
curately reflects the dataset. The system’s task
is to extract a response based on available mes-
sages, not to verify the truthfulness of the infor-
mation. This structure emphasizes the need for a
system capable of combining structured metadata
processing with semantic text understanding, even
when dealing with implicit or indirectly expressed
content.
In this paper, we not only introduce AMAQA,
a novel dataset designed to address the aforemen-
tioned gaps, but also evaluate the performance ofthree baselines (RAG, RAG with metadata filter-
ing and Re2G (Glass et al., 2022)) on the new
benchmark. Then we evaluate new techniques
based on Re2G (Glass et al., 2022), "The Power
of Noise" (Cuconasu et al., 2024) and an iterative
context technique. We provide a baseline for fu-
ture research, enabling the development of more
advanced and effective metadata-driven question-
answering systems.
2 Related work
The landscape of research datasets and bench-
marks for QA systems and RAG models has
evolved significantly in recent years. Existing
work can be categorized into two broad classes:
(i) datasets containing textual data enriched with
metadata and (ii) benchmark datasets tailored to
specific aspects of QA or RAG systems but lack-
ing in metadata integration. As shown in Table
1, a wide variety of datasets contain textual data
with associated metadata exist, such as Movie-
Dataset (Banik, 2017), Hotel-Reviews1,Amazon-
Reviews (Hou et al., 2024), Pushshift (Baumgart-
ner et al., 2020), and TgDataset (La Morgia et al.,
2025). These datasets provide rich textual con-
tent and corresponding metadata, enabling tasks
such as sentiment analysis, recommendation sys-
tems, or exploring the Telegram ecosystem. How-
ever, they aren’t suited for QA benchmarking,
lacking structured question-answer pairs. Further-
more, these datasets do not contain information
on the discussed topics, the emotions conveyed, or
the toxicity of the text, missing key elements that
could enhance the depth of analysis in QA models.
Conversely, several QA benchmarks, such as
PopQA (Mallen et al., 2023), TriviaQA (Joshi
et al., 2017) and SQuAD 1.1 (Rajpurkar et al.,
2016), have been introduced to address spe-
cific challenges QA tasks. For example, Hot-
potQA (Yang et al., 2018) evaluates a sys-
tem’s ability to retrieve and reason over mul-
tiple documents, emphasizing multi-hop reason-
ing.GPQA (Rein et al., 2024), on the other
hand, provides a multiple-choice QA dataset with
expert-validated questions in biology, physics, and
chemistry. While valuable for evaluating a sys-
tem’s domain-specific capabilities, it lacks meta-
data integration and contextual information. An-
other prominent benchmark, CoQA (Reddy et al.,
1https://www.kaggle.com/datasets/
datafiniti/hotel-reviews

2019), is designed for conversational QA, eval-
uating systems’ ability to maintain context and
coherence across multiple dialogue turns. How-
ever, like other QA benchmarks designed to eval-
uate RAG-addressable tasks, such as CRUD (Lyu
et al., 2024), CoQA (Reddy et al., 2019) does not
leverage metadata to provide additional context or
structure to question answering.
The main limitation we aim to highlight in
Table 1 is that although these datasets are ei-
ther large-scale or domain-specific, none of
them include (nor exploit) metadata. Whether
they contain millions of QA pairs, such as
MS-MARCO (Nguyen et al., 2016) and MI-
RAGE (Xiong et al., 2024), or focus on specific
domains like GPQA (Rein et al., 2024), they lack
structured metadata that could provide deeper in-
sights and improve model evaluation. To address
these gaps, AMAQA aims to serve as an innovative
resource for QA and RAG systems. By integrating
metadata with a structured QA dataset, AMAQA
will enable more holistic evaluations, considering
the interplay between text and metadata. This ap-
proach will open new possibilities for metadata-
filtering , bridging the gap between general textual
datasets and metadata-free QA benchmarks.
Type Name QA Docs Metadata
Bench CRUD ~3.2k ~80k ✗
Bench PopQA ~14k N/A ✗
Bench HotpotQA ~113k N/A ✗
Bench TriviaQA ~650k ~662k ✗
Bench SQuAD 1.1 ~100k ~500 ✗
Bench MS-MARCO ~1.1M ~138M ✗
Bench MIRAGE ~7.6k ~60M ✗
Bench GPQA 448 N/A ✗
Bench CoQA ~127k ~7k ✗
Data Movie-Dataset ✗ 45k ✓
Data Hotel-Rev. ✗ 20k ✓
Data Amazon-Rev. ✗ ~571M ✓
Data Pushshift ✗ ~317M ✓
Data TgDataset ✗ ~400M ✓
Bench AMAQA 450 ~1.1M ✓
Table 1 : Overview of single-hop QA benchmarks
(Bench) and textual datasets (Data).
3 The AMAQA dataset
The AMAQA dataset consists of approximately
1.1 million Telegram messages with their asso-
ciated metadata. Since not all of the metadata
provided by Telegram are useful for creating our
benchmark, only a subset was used. In addition,each message was labeled with the topics dis-
cussed, the main emotion conveyed, and the prob-
abilities of containing toxicity, profanity, insults,
identity attacks, or threats.
3.1 Data collection
To build a large English-language text dataset with
metadata, messages were collected from Tele-
gram because it is a popular messaging app with
over 900 million users as of 20242and it of-
fers easy access to public channel messages and
related discussion groups. The data collection
started from well-known channels discussing the
following topics: the Russian-Ukrainian conflict,
the U.S. election, and the Israeli-Palestinian con-
flict. Then, using Telegram’s recommendation
system, channels were selected based on rele-
vance, English-language content, and availability
of discussion groups. In total, 26 channels and
their groups were identified. Messages were col-
lected between June 13 and August 13, 2024, re-
sulting in approximately 1.1 million messages .
3.2 Data labeling
The first step required to identify which were
the characteristics of the texts to focus on: the
first one were the emotions conveyed by the text.
We adopted Ekman’s model of six basic emo-
tions (Ekman, 1992) which is widely used in emo-
tion detection research, where emotions are typ-
ically categorized in a multi-class classification
framework (Seyeditabari et al., 2018). The Ek-
man’s model highlights sadness, happiness, anger,
fear, disgust, and surprise as universally recog-
nized emotions. To this set, we added a "neutral"
category to account emotionally-free texts. Emo-
tions were extracted using a Zero-Shot Classifier
(Laurer et al., 2023; Plaza-del Arco et al., 2022),
applying the hypothesis "This text expresses a
feeling of..." with the seven emotions as classes.
In addition to emotional features, other textual
characteristics, such as Toxicity, were extracted.
Toxicity is defined by the Perspective API3as "a
rude, disrespectful, or unreasonable comment that
is likely to make you leave a discussion". How-
ever, the use of the Perspective API was not lim-
ited to detecting toxicity alone but also extended to
other forms of harmful language, including Iden-
tity Attack, Insult, Profanity, and Threat. Follow-
2https://www.businessofapps.com/data/
telegram-statistics/
3https://www.perspectiveapi.com/

ing the guidelines, we consider a comment to ex-
hibit these forms of harmful language if the cor-
responding score exceeds a threshold of 0.7, as
adopted by Alvisi et al. (2025)
In addition to analyze the emotions and sentiments
expressed in the text, it was essential to extract
the discussed topics in order to construct the com-
plete ground truth. The topic detection process
was structured with the goal of associating multi-
ple topics to each message (Wang et al., 2023) tak-
ing care not to identify too many or too few topics.
Initially, the dataset was first manually explored to
identify an initial set of topics. BERTTopic was
then used to improve topic identification (Groo-
tendorst, 2022). However, since BERTTopic as-
signs only one topic per post and often generates
clusters that are either too general or too specific,
a hybrid approach was adopted. The manually de-
fined topics were compared with BERTTopic re-
sults, and frequently occurring topics (over 1,000
posts) not already identified were added. The final
topic set includes 58 topics covering a wide range
of subjects in the dataset.
Since AMAQA is formed by 1.1 million mes-
sages, manual labeling was unfeasible. Instead,
we used GPT-4o for topic extraction, as it has out-
performed crowd workers in annotation tasks (Gi-
lardi et al., 2023). Each message was processed
by GPT-4o using a prompt inspired by prior topic
extraction approaches (Mu et al., 2024; Münker
et al., 2024), allowing from 0 to Ntopics extracted
per message. The output was a list of relevant top-
ics per message. However, since LLMs do not
always follow instructions with complete preci-
sion (Zhou et al., 2023), we cleaned the detected
topic to correct inconsistencies (e.g., “states/Eu-
rope" to “organizations/European Union") and ex-
clude valid but off-list topics (e.g., “states/Italy”)
due to their low frequency.
3.3 Dataset statistics
The dataset’s main characteristics are summa-
rized in Table 2. Additional insights are provided
through visualizations: in Figure 2a the distribu-
tion of messages collected is shown. Notably, the
temporal distribution shows activity peaks linked
to major events, such as the July 14 attack on
Trump and the June 28 Trump–Biden debate,
while in Figure 3 a donut chart illustrates the dis-
tribution of emotions.
Figure 2b highlights the main topics discussedFeature Value
Total Messages 1,146,690
Toxic Messages 107,154
Contains Insults 52,413
Profanity 61,969
Identity Attacks 9,239
Threats 2,509
Groups 26
Unique Users 64,110
Topics 58
Table 2 : Dataset Characteristics
in the dataset, with Russia ,Ukraine ,Donald
Trump , and the Ukraine-Russia conflict being the
most prominent. This distribution is expected,
as the seed channels primarily focus on these is-
sues. The dataset is clearly unbalanced in terms
of topic, emphasizing geopolitical themes and po-
litical figures such as states, conflicts, and lead-
ers like Trump, Biden, and Putin. Other recur-
ring topics include religion, media manipulation,
and references to socially and politically sensitive
groups, such as Muslims, Jews, and Democrats,
all contributing to a highly polarized discourse.
This is reflected in the emotional analysis, which
shows a predominance of anger, suggesting in-
tense reactions to political and conflict-related top-
ics. Although joy is the second most common
emotion, it appears far less frequently, reinforcing
the dataset’s overall negative emotional tone. The
prevalence of anger and focus on divisive themes
confirm the polarized nature of the dataset, con-
sistent with our expectations. The dataset was de-
liberately curated to include toxic content, such as
threats and identity attacks, to facilitate the explo-
ration of these textual characteristics. The inter-
play of complex themes and emotionally charged
discussions underscores a landscape of intense and
divisive debates.
3.4 QA creation
The process began by defining natural language
questions based on realistic scenarios of system
use. An example of a question is:
“What are toxic messages posted on {date} talk-
ing about {topic} on the “{chat}" group chat?”
By iterating over the date (60 days), topic (58
topics), and chat (26 chats) parameters, 90,480
ElasticSearch queries were generated, but only

(a) Distribution of messages over time
 (b) Distribution of the most frequently discussed topics
Figure 2 : Visualization of message and topic distributions, providing insights into temporal trends and
the division of messages by topic.
Figure 3 : Distribution of emotions across mes-
sages, illustrating the prevalence of different emo-
tional tones.
queries returning more than 10 documents were
considered to ensure an appropriate level of com-
plexity. This threshold was set to assess not only
the system’s ability to create accurate metadata-
based filters but also its capacity to process and
reason over complex informational content during
the generation phase. Specifically, a system capa-
ble of generating a highly precise filter might, in
extreme cases, retrieve only a single relevant doc-
ument. In such situations, the generation model
would receive a minimal and unambiguous con-
text, and its response would necessarily be based
on that single, and by definition correct, document.
The 10-document threshold ensures sufficient con-
text for challenging the generator’s ability to pro-
cess and understand complex contents. After this
step, 1477 question with their relevant documents
were obtained. For each query result, we created a
file containing the original question and the rel-
evant documents. Then, following an approach
similar to the one used by Jeong et al. (2025), we
used GPT-4o to generate three JSONs per docu-
ment, each containing a question, an answer, andthe corresponding message. However, not all gen-
erated question-answer-message triples were rel-
evant or correct. In some cases, all three JSONs
were useful, while in others none of them were.
With 1477 documents and three QAs per docu-
ment, this resulted in 4,431 QA pairs which was
a too large volume of data to check manually in a
reasonable time. For this reason GPT-4o was again
used to select a single QA pair from each gener-
ated triplet. This reduction process resulted in 958
QA pairs , which were then manually validated in
two phases by four annotators and three judges to
ensure quality and reliability. Initially, the anno-
tators evaluated each QA pair, either discarding or
validating it based on the following criteria:
•Correctness of the question and answer: The
question had to be clear and meaningful,
while the answer needed to be accurate and
consistent with the document from which it
was extracted.
•Clarity and completeness: The QA pair was
required to avoid ambiguity (i.e., there should
not be multiple plausible answers to the same
question) and the answer needed to be com-
plete, without omitting relevant information.
•Originality with respect to LLMs: To en-
sure the QA pairs were novel and not already
known to LLMs, the question (or a minimally
altered version of it) was submitted to Chat-
GPT. QA pairs were excluded if the model
was able to provide the correct answer with-
out relying on supporting documents.
Subsequently, the judges reviewed the QA pairs
that had been validated by the annotators. This
second round of evaluation aimed to ensure con-
sistency and eliminate any residual errors or over-
sights. To avoid discarding too many QA pairs

unnecessarily, annotators were allowed to modify
the questions or answers (but not the original text
message) so that a pair that might otherwise be
discarded could meet the positive evaluation cri-
teria. Out of the 958 initially annotated question-
answer pairs, 502 were validated by the anno-
tators. Among these, 450 were also confirmed
by the judges, resulting in a simple agreement of
89.64% between the annotators’ and judges’ eval-
uations. By employing this rigorous two-stage val-
idation process, the final benchmark consists ex-
clusively of high-quality QA pairs.
3.5 Ethical consideration
Our data collection was conducted with care to
include only information from public Telegram
channels and groups, explicitly excluding personal
data such as usernames, phone numbers, user IDs,
and chat IDs. The process aligns with Telegram’s
Terms of Service4and Telegram API’s Terms of
Service5, as neither prohibits the collection of pub-
lic chat data. We have also taken care to avoid
flooding the platform with requests during data
collection with excessive requests. Furthermore,
our research complies with the academic research
exemptions outlined in Article 85 of the GDPR6,
which provide flexibility for processing publicly
available data in the interest of freedom of expres-
sion and academic purposes. Since our work ex-
clusively involves publicly accessible content and
does not handle private user data, it qualifies for
legitimate or public interest exemptions. Addi-
tionally, we follow the GDPR’s principle of data
minimization by focusing solely on English tex-
tual content from public posts, ensuring that only
the information necessary for our research objec-
tives is processed (Your Europe, 2025). Finally,
as the dataset includes channels and groups that
discuss controversial topics, it may contain some
controversial messages.
4 RAG systems evaluation methodology
In this study, we employ the new AMAQA bench-
mark to evaluate RAG systems and address the fol-
lowing key research questions:
RQ1 : How does the inclusion of metadata affect
the performance of RAG systems?
4https://telegram.org/tos/
5https://core.telegram.org/api/terms
6https://gdpr-text.com/read/
article-85/RQ2 : Do enhancements to the retriever and gener-
ator components lead to improve overall per-
formance?
To answer these questions, we established three
baselines: a basic RAG system that does not lever-
age metadata, an approach based on metadata fil-
tering, and a technique that re-ranks the retriever’s
output. Subsequently, we developed three novel
approaches that significantly outperformed these
baselines. The motivation behind proposing these
baselines and novel architectures is to provide the
research community with a solid foundation for
comparison on this new benchmark, which is the
first to incorporate metadata and allow their ex-
ploitation in question-answering tasks.
4.1 RAG systems: baselines
The following sections describe the three baseline
architectures evaluated in this work.
Vanilla RAG : A standard RAG system com-
posed of a Retriever and a Generator, where the
latter is a large language model (LLM). The Re-
triever calculate the input question embeddings,
compares it to precomputed document embed-
dings using cosine similarity, and retrieves the k
most similar documents. The LLM then uses them
to produce an answer. However, the vanilla RAG
cannot natively process data containing both text
and metadata. To address this limitation, metadata
are embedded into the text before computing em-
beddings.
RAG with Metadata Filter : A RAG system in-
tegrating a metadata filter (Gao et al., 2023). Its
architecture, shown in Figure 4, differs from the
vanilla version in the Retriever component. The
Retriever includes a Query Creator, an LLM that
turns user input into a structured query. It first gen-
erates a filter to refine the search scope, then con-
verts the question into embeddings. The filter is
applied to the vector store, followed by a similar-
ity search to retrieve the most relevant documents.
Re2G: A system based on the RAG architecture
with a metadata filter (Glass et al., 2022), as shown
in Figure 4, with the addition of a reranker phase
before providing the relevant documents as input
to the Generator LLM.
4.2 RAG systems: novel approaches
The following sections describe the novel ap-
proaches proposed to improve the baselines.

Figure 4 : RAG systems capable of handling meta-
data schema. The optional component is part
of the Re2G architecture and all its derived ap-
proaches.
Figure 5 : Generator component schema in the Iter-
Re2G architecture: after the first response, the sys-
tem checks for "I don’t know." If found, it uses the
next top_n documents for a new generation; other-
wise, it accepts the response. This repeats until a
valid response is given or no documents remain.
Iter-Re2G: This approach employs a sliding
window mechanism on the context provided to the
generator. As illustrated in Figure 5, the system
initially generates a response based on the top-
ranked retrieved documents. If the response does
not contain the phrase "I don’t know" , it is ac-
cepted. Otherwise, the system iterates by incorpo-
rating the next top_n documents into the context
until a satisfactory response is generated. Apart
from this iterative refinement process, the overall
architecture remains identical to Re2G.
Re2G with Noise : This approach, inspired
by the work "The Power of Noise" (Cuconasu
et al., 2024), maintains the same architecture as
Re2G (Glass et al., 2022) but injects "noisy" doc-
uments into the context provided to the language
model. A noisy document refers to one that is se-
mantically distant from the documents returned by
the retriever, meaning it is not a document con-taining incorrect but similar information (as this
would introduce more confusion for the LLM). To
achieve this, we select the text of Reddit posts dis-
cussing a topic not present in the Q&A, ensur-
ing the added documents are surely noisy. We
used Reddit posts for this purpose because this ap-
proach is also adopted in the original work (Cu-
conasu et al., 2024). Furthermore, by manually se-
lecting 20 posts from subreddits focused on video
games (r/GTA and r/NBA2k), we ensure that their
content is unrelated to the topics discussed in the
dataset.
Iter-Re2G with Noise : This method combines
both strategies. Like Iter-Re2G, it applies a sliding
window mechanism over the retrieved documents
but also incorporates "noisy” documents into the
context, as in Re2G with Noise .
4.3 Evaluation metrics
The evaluation of a RAG system depends on its
specific task. In the case of QA, common met-
rics include Normalized Exact Match (NEM)7and
BLEU score (Papineni et al., 2002). However,
these metrics alone do not fully capture system
performance. Therefore, it is standard practice
to separately evaluate the retriever andgenerator
components using multiple metrics.
For the retriever , standard information retrieval
metrics are employed (Gao et al., 2023). In this
work, we use the Mean Reciprocal Rank (MRR) ,
given that each query is associated with a single
relevant document. Additionally, we consider the
number of relevant documents retrieved .
For the generator , traditional metrics like NEM
and BLEU (Papineni et al., 2002) rely on string or
n-gram comparisons, failing to capture the seman-
tic quality of responses. A more robust alterna-
tive is human evaluation , where annotators com-
pare model outputs with ground-truth answers.
While this remains the gold standard, it is costly
and time-consuming. To address these limitations,
LLM-as-a-Judges have emerged as a scalable al-
ternative (Zheng et al., 2023). Since these models
are trained via Reinforcement Learning from Hu-
man Feedback , they exhibit strong human align-
ment, enabling faster and more explainable eval-
uations. Despite their advantages, LLM-based
evaluations are subject to verbosity bias (favoring
longer responses) and self-enhancement bias (fa-
7A variant of Exact Match that ignores formatting differ-
ences, such as spaces, capitalization, and punctuation.

voring LLM-generated answers). However, GPT-
4 has been shown to align with human judgments
in over 80% of cases (Zheng et al., 2023). To vali-
date this, we conducted an experiment where GPT-
4o evaluated 450 RAG-generated responses, fol-
lowed by human review. Only 4 disagreements be-
tween GPT-4o and human annotators were found,
supporting its reliability. We therefore use GPT-4o
as our evaluator and accuracy is computed based
on its judgment. However, to maintain alignment
with standard practice, we also report NEM as a
complementary metric.
4.4 Experimental setup
The experiments were conducted using the follow-
ing setup:
Data Storage : all data was indexed and stored
in Elasticsearch to enable efficient retrieval and
search operations.
Embeddings : the sentence-transformers/all-
MiniLM-L6-v2 model was used, chosen for its bal-
ance between computational efficiency and em-
bedding quality.
Similarity Measure :cosine similarity was
used to compare the embeddings of queries and
document representations.
Relevant Document Retrieval : relevant docu-
ments were retrieved using exact k-nearest neigh-
bors (KNN) algorithm.
Experimental Variations : the experiments in-
cluded variations in the following components
where applicable: the Large Language Model
(LLM) used in the Retriever and the one used in
theGenerator phase, as well as the value of K,
which determines the number of documents re-
trieved in the KNN search.
Reranker used : when a reranker was included
in the architecture, the ms-marco-MiniLM-L-12-
v2model (Face, 2024) was used as a cross-encoder
via the FlashRank (Damodaran, 2023) Python li-
brary.
Additionally, in all experiments involving the
use of the reranker, from Re2G to all subsequent
improvements, top_n was set to 5, aiming for a
small but comparable value to the kvalues used in
other experiments. In order to maintain a consis-
tent relationship of k≫top_n , which is crucial as
highlighted by Yu et al. (2024), the kvalue was set
to 200. It is important to note that for the experi-
ments that involves the use of noisy documents, 3
retrieved documents and 2 noisy documents wereused, ensuring a total of 5 documents for the re-
triever.
Finally, the Accuracy and NEM reported values
are averages over 10 experiments, with different
seeds used for the generator LLM. Statistical rele-
vance of differences between results was assessed
using the t-test with Bonferroni correction (Dunn,
1961) with a significance level α= 0.01.
5 Experimental Evaluation of RAG
Systems with AMAQA
This section evaluates the performance of RAG
systems using the AMAQA dataset, focusing on
the impact of metadata, retrieval optimizations,
and advanced context handling techniques. The
results are analyzed in relation to the research
questions introduced in Section 4.
5.1 Impact of Metadata on RAG
Performance
A vanilla RAG system in which metadata are em-
bedded in the text suffers from poor retrieval per-
formance due to the interference of metadata en-
coding with filtering mechanism. This approach
fails to apply filtering constraints properly and
decreases semantic search precision, as the re-
triever cannot effectively distinguish between con-
tent and metadata. To mitigate these limitations,
we introduce metadata filtering, which improves
retrieval accuracy by enforcing explicit filtering
constraints at query time. Experimental results
show that metadata filtering boosts retrieval effec-
tiveness, increasing the number of relevant doc-
uments retrieved from 73 to 330-340 out of 450
relevant documents, depending on the retriever’s
LLM. The improvement is also reflected in the en-
tire system: the accuracy rises from 0.12 to 0.61,
while NEM improves from 0.05 to 0.39. Our
findings provide a direct answer to RQ1, show-
ing that embedding metadata within the text neg-
atively impacts retrieval performance, as it intro-
duces noise that disrupts both filtering and rank-
ing. In contrast, metadata filtering ensures effi-
cient constraint enforcement, leading to a substan-
tial increase in retrieval effectiveness and overall
pipeline performance. Explicit metadata manage-
ment is therefore essential to maintaining retrieval
precision and optimizing the integration of exter-
nal knowledge in RAG systems.

Figure 6 : Number of relevant documents retrieved
as function of K, the number of retrieved docu-
ments, varying LLM used in the Retriever compo-
nent.
Figure 7 : Answer accuracy as a function of K
(retrieved documents), varying the LLM used as
Generator. The plot also compares with the ideal
case, where the correct answer is generated for
each relevant document retrieved.
5.2 Enhancing Retriever performance with
reranking phase
To address RQ2 , we start analyzing the impact of
retriever optimizations, by comparing retrieval ef-
fectiveness using different LLMs and values of k.
Figure 6 shows that varying the LLM in the re-
triever has a marginal effect on retrieval perfor-
mance. However, increasing kleads to a higher
number of relevant documents retrieved. Despite
this, a large kintroduces the Lost-in-the-Middle
phenomenon (Liu et al., 2024), where information
in the middle of the prompt is deprioritized by the
LLM generator. This issue is evident in Figure 7,
where accuracy drops for larger kvalues. The in-
troduction of a reranking step in Re2G, presented
in section 4.1, mitigates this issue. In fact, in Fig-
ure 7, it can be observed that when the value of K
is small, the accuracy gets quite close to the ideal
case in which, for each relevant document found,
the correct answer is generated. With the introduc-tion of reranking, it ensures that the most relevant
documents appear within the first positions in the
generator’s prompt, as we can see from Figure 8.
This yields, to an improvement in accuracy from
0.61 to 0.72 too.
Figure 8 : Mean Reciprocal Rank (MRR) as a func-
tion of K, comparing the performance of RAG
with metadata filter and Re2G architecture.
5.3 Enhancing Performance with Iterative
Context Expansion and Noise
To fully address RQ2 , we investigate the results
obtained using novel approaches with two addi-
tional strategies: iterative context expansion (Iter-
Re2G) and the inclusion of noise (Re2G with
noise), introduced in Section 4.2.
The Iter-Re2G approach enhances the Re2G ac-
curacy from 0.72 to 0.75 by effectively integrat-
ing relevant documents into the generator’s con-
text while mitigating information overload, even
when these documents are not ranked within the
topnpositions. In both Re2G and Iter-Re2G archi-
tectures, the inclusion of noise, through the injec-
tion of irrelevant documents into the prompt, does
not alter the accuracy but enhances the NEM score
by 0.02, indicating a stronger alignment between
the generated answers and the expected ones.
Finally, the combination of both techniques in
Iter-Re2G with noise achieves the best results in
both accuracy (0.75) and NEM (0.54) scores.
5.4 Summary of Experimental Findings
Table 3 presents a comprehensive summary of the
performance of the evaluated RAG architectures.
Regarding RQ1 (How does the inclusion of
metadata affect the performance of RAG sys-
tems? ), the experimental results demonstrate that
metadata filtering plays a crucial role in enhanc-
ing retrieval precision, leading to an increase in
accuracy from 0.12 to 0.61.

With respect to RQ2 (Do enhancements to the
retriever and generator components lead to im-
proved overall performance? ), the findings are as
follows:
• The application of reranking ( Re2G) opti-
mizes retrieval ordering, mitigating the Lost-
in-the-Middle issue and improving accuracy
to 0.72.
• Iterative context expansion further enhances
performance achieving an accuracy of 0.75.
• The introduction of noise during the genera-
tion phase slightly improves the NEM score.
• The Iter-Re2G approach with noise, which is
a combination of the previous 2 architectures,
achieves the best performance.
RAG architecture Acc. NEM k top_n Noise
Iter-Re2G with noise 0.75 0.54 200 3 2
Iter-Re2G 0.75 0.52 200 5 -
Re2G with noise 0.72 0.53 200 3 2
Re2G 0.72 0.51 200 5 -
Metadata filtering 0.61 0.39 30 - -
Vanilla 0.12 0.05 20 - -
Table 3 : Comparison of RAG architectures un-
der optimal settings based on accuracy (acc.) and
NEM. Highlighted rows indicate baselines ; bold
marks the best result.
These experimental results establish a new
benchmark for metadata-driven question answer-
ing systems and provide a robust foundation for
future advancements in retrieval-augmented gen-
eration research.
6 Limitations
This study presents some limitations that must be
considered to properly interpret the results ob-
tained. Budget constraints prevented expanding
the dataset with more labeled examples. Also,
Azure’s API filter for harmful content couldn’t be
disabled, negatively affected the results of GPT-
4o model. The system’s performance is tied to
the capabilities of the LLMs used, none of which
exceed 10 billion parameters, potentially limiting
the quality of the Generator’s responses. Further-
more, the performance of the system is strongly
dependent on the prompts used, particularly con-
cerning the formatting of the context. It is plausi-
ble that using more powerful models and conduct-
ing more extensive experimentation with promptengineering techniques could lead to further im-
provements.
7 Conclusion and Future Work
In this work, we introduced AMAQA, the first
open-access QA dataset specifically designed to
evaluate Retrieval-Augmented Generation (RAG)
systems that leverage metadata. AMAQA, built
on 1.1 million messages from 26 public Tele-
gram groups and enriched with structured meta-
data, enables sophisticated QA tasks and offers
a novel resource for advancing metadata-driven
systems. Through extensive experimentation, we
demonstrated that metadata filtering significantly
improves retrieval accuracy, boosting system ac-
curacy from 0.12 to 0.61. Moreover, by incorpo-
rating retrieval reranking (Re2G), iterative context
expansion (Iter-Re2G), and controlled noise injec-
tion, we established new state-of-the-art results,
achieving an accuracy of 0.75 and a NEM score
of 0.54. These findings highlight the importance
of structured metadata integration and retrieval re-
finement in optimizing RAG performance.
Future work could achieve significant improve-
ments by fine-tuning the embeddings, which could
lead to better performance in the retrieval phase
or, alternatively, fine-tuning the LLM used as
a generator could ensure that the responses are
more aligned with the specific terms and language
present in the data (Gao et al., 2023). In addi-
tion to methodological refinements, it is equally
important to maintain the benchmark over time,
to prevent obsolescence and expand or adapt the
knowledge base to other domains. For example,
new messages could be collected to generate eval-
uation questions and update the dataset, while data
from domains like legal knowledge bases or mul-
timodal sources could further improve the system’
versatility and performance.
By establishing AMAQA as a benchmark and
demonstrating the impact of metadata-driven re-
trieval techniques, we aim to inspire further re-
search into more context-aware, robust, and effi-
cient QA systems. We make the dataset publicly
available to encourage collaboration and future ad-
vancements in this field.
References
Lorenzo Alvisi, Serena Tardelli, and Maurizio
Tesconi. 2025. Mapping the italian telegram

ecosystem: Communities, toxicity, and hate
speech.
Flor Miriam Plaza-del Arco, María-Teresa Martín-
Valdivia, and Roman Klinger. 2022. Natu-
ral language inference prompts for zero-shot
emotion classification in text across corpora.
InProceedings of the 29th International Con-
ference on Computational Linguistics , pages
6805–6817, Gyeongju, Republic of Korea. In-
ternational Committee on Computational Lin-
guistics.
Rounak Banik. 2017. The movies dataset.
Jason Baumgartner, Savvas Zannettou, Megan
Squire, and Jeremy Blackburn. 2020. The
pushshift telegram dataset. In Proceedings of
the international AAAI conference on web and
social media , volume 14, pages 840–847.
Florin Cuconasu, Giovanni Trappolini, Federico
Siciliano, Simone Filice, Cesare Campagnano,
Yoelle Maarek, Nicola Tonellotto, and Fabrizio
Silvestri. 2024. The power of noise: Redefining
retrieval for rag systems. In Proceedings of the
47th International ACM SIGIR Conference on
Research and Development in Information Re-
trieval , pages 719–729.
Prithiviraj Damodaran. 2023. FlashRank, Light-
est and Fastest 2nd Stage Reranker for search
pipelines.
Olive Jean Dunn. 1961. Multiple comparisons
among means. Journal of the American statisti-
cal association , 56(293):52–64.
Paul Ekman. 1992. An argument for basic emo-
tions. Cognition & emotion , 6(3-4):169–200.
Hugging Face. 2024. cross-encoder/ms-
marco-minilm-l-12-v2. https:
//huggingface.co/cross-encoder/
ms-marco-MiniLM-L-12-v2 . Accessed:
2024-12-29.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 .
Fabrizio Gilardi, Meysam Alizadeh, and Maël
Kubli. 2023. Chatgpt outperforms crowdworkers for text-annotation tasks. Proceed-
ings of the National Academy of Sciences ,
120(30):e2305016120.
Michael Glass, Gaetano Rossiello,
Md Faisal Mahbub Chowdhury, Ankita Naik,
Pengshan Cai, and Alfio Gliozzo. 2022. Re2G:
Retrieve, rerank, generate. In Proceedings of
the 2022 Conference of the North American
Chapter of the Association for Computational
Linguistics: Human Language Technologies ,
pages 2701–2715, Seattle, United States.
Association for Computational Linguistics.
Maarten Grootendorst. 2022. Bertopic: Neural
topic modeling with a class-based tf-idf proce-
dure. arXiv preprint arXiv:2203.05794 .
Yupeng Hou, Jiacheng Li, Zhankui He, An Yan,
Xiusi Chen, and Julian McAuley. 2024. Bridg-
ing language and items for retrieval and recom-
mendation. arXiv preprint arXiv:2403.03952 .
Soyeong Jeong, Kangsan Kim, Jinheon Baek, and
Sung Ju Hwang. 2025. Videorag: Retrieval-
augmented generation over video corpus. arXiv
preprint arXiv:2501.05874 .
Mandar Joshi, Eunsol Choi, Daniel Weld, and
Luke Zettlemoyer. 2017. TriviaQA: A large
scale distantly supervised challenge dataset for
reading comprehension. In Proceedings of the
55th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Pa-
pers) , pages 1601–1611, Vancouver, Canada.
Association for Computational Linguistics.
Massimo La Morgia, Alessandro Mei, and Al-
berto Maria Mongardini. 2025. Tgdataset:
Collecting and exploring the largest telegram
channels dataset. In Proceedings of the 31st
ACM SIGKDD Conference on Knowledge Dis-
covery and Data Mining V .1 , KDD ’25, page
2325–2334, New York, NY , USA. Association
for Computing Machinery.
Moritz Laurer, Wouter Van Atteveldt, Andreu
Casas, and Kasper Welbers. 2023. Less Anno-
tating, More Classifying: Addressing the Data
Scarcity Issue of Supervised Machine Learning
with Deep Transfer Learning and BERT-NLI.
Political Analysis , pages 1–33.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin
Paranjape, Michele Bevilacqua, Fabio Petroni,

and Percy Liang. 2024. Lost in the mid-
dle: How language models use long contexts.
Transactions of the Association for Computa-
tional Linguistics , 12:157–173.
Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong,
Bo Tang, Wenjin Wang, Hao Wu, Huanyong
Liu, Tong Xu, and Enhong Chen. 2024. Crud-
rag: A comprehensive chinese benchmark for
retrieval-augmented generation of large lan-
guage models. ACM Transactions on Informa-
tion Systems .
Alex Mallen, Akari Asai, Victor Zhong, Ra-
jarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. 2023. When not to trust language mod-
els: Investigating effectiveness of parametric
and non-parametric memories. In Proceedings
of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long
Papers) , pages 9802–9822, Toronto, Canada.
Association for Computational Linguistics.
Yida Mu, Chun Dong, Kalina Bontcheva, and
Xingyi Song. 2024. Large language models
offer an alternative to the traditional approach
of topic modelling. In Proceedings of the
2024 Joint International Conference on Com-
putational Linguistics, Language Resources
and Evaluation (LREC-COLING 2024) , pages
10160–10171, Torino, Italia. ELRA and ICCL.
Simon Münker, Kai Kugler, and Achim Ret-
tinger. 2024. Zero-shot prompt-based classi-
fication: topic labeling in times of founda-
tion models in german tweets. arXiv preprint
arXiv:2406.18239 .
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng
Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016. Ms marco: A human-generated
machine reading comprehension dataset.
Kishore Papineni, Salim Roukos, Todd Ward, and
Wei-Jing Zhu. 2002. Bleu: a method for auto-
matic evaluation of machine translation. In Pro-
ceedings of the 40th annual meeting of the As-
sociation for Computational Linguistics , pages
311–318.
Elijah Pelofske, Lorie M Liebrock, and Vin-
cent Urias. 2023. Cybersecurity threat hunting
and vulnerability analysis using a neo4j graph
database of open source intelligence. arXiv
preprint arXiv:2301.12013 .Beatrice Perez, Mirco Musolesi, and Gianluca
Stringhini. 2018. You are your metadata: Iden-
tification and obfuscation of social media users
using metadata information.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopy-
rev, and Percy Liang. 2016. SQuAD: 100,000+
questions for machine comprehension of text.
InProceedings of the 2016 Conference on Em-
pirical Methods in Natural Language Process-
ing, pages 2383–2392, Austin, Texas. Associa-
tion for Computational Linguistics.
Siva Reddy, Danqi Chen, and Christopher D Man-
ning. 2019. Coqa: A conversational question
answering challenge. Transactions of the Asso-
ciation for Computational Linguistics , 7:249–
266.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R. Bow-
man. 2024. GPQA: A graduate-level google-
proof q&a benchmark. In First Conference on
Language Modeling .
Armin Seyeditabari, Narges Tabari, and Wlodek
Zadrozny. 2018. Emotion detection in text: a
review. arXiv preprint arXiv:1806.00674 .
Fengjun Wang, Moran Beladev, Ofri Kleinfeld,
Elina Frayerman, Tal Shachar, Eran Fainman,
Karen Lastmann Assaraf, Sarai Mizrachi, and
Benjamin Wang. 2023. Text2Topic: Multi-
label text classification system for efficient topic
detection in user generated content with zero-
shot capabilities. In Proceedings of the 2023
Conference on Empirical Methods in Natural
Language Processing: Industry Track , pages
93–103, Singapore. Association for Computa-
tional Linguistics.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and
Aidong Zhang. 2024. Benchmarking retrieval-
augmented generation for medicine. In Find-
ings of the Association for Computational
Linguistics: ACL 2024 , pages 6233–6251,
Bangkok, Thailand. Association for Computa-
tional Linguistics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. 2018. HotpotQA:
A dataset for diverse, explainable multi-hop

question answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural
Language Processing , pages 2369–2380, Brus-
sels, Belgium. Association for Computational
Linguistics.
Your Europe. 2025. Data protection under GDPR.
Accessed: 2025-01-27.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Ji-
axuan You, Chao Zhang, Mohammad Shoeybi,
and Bryan Catanzaro. 2024. RankRAG: Uni-
fying context ranking with retrieval-augmented
generation in LLMs. In The Thirty-eighth An-
nual Conference on Neural Information Pro-
cessing Systems .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao
Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric
Xing, et al. 2023. Judging llm-as-a-judge with
mt-bench and chatbot arena. Advances in Neu-
ral Information Processing Systems , 36:46595–
46623.
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Sid-
dhartha Brahma, Sujoy Basu, Yi Luan, Denny
Zhou, and Le Hou. 2023. Instruction-following
evaluation for large language models. arXiv
preprint arXiv:2311.07911 .
# Aligning LLMs for the Classroom with Knowledge-Based Retrieval -- A Comparative RAG Study

**Authors**: Amay Jain, Liu Cui, Si Chen

**Published**: 2025-09-09 15:22:33

**PDF URL**: [http://arxiv.org/pdf/2509.07846v1](http://arxiv.org/pdf/2509.07846v1)

## Abstract
Large language models like ChatGPT are increasingly used in classrooms, but
they often provide outdated or fabricated information that can mislead
students. Retrieval Augmented Generation (RAG) improves reliability of LLMs by
grounding responses in external resources. We investigate two accessible RAG
paradigms, vector-based retrieval and graph-based retrieval to identify best
practices for classroom question answering (QA). Existing comparative studies
fail to account for pedagogical factors such as educational disciplines,
question types, and practical deployment costs. Using a novel dataset,
EduScopeQA, of 3,176 questions across academic subjects, we measure performance
on various educational query types, from specific facts to broad thematic
discussions. We also evaluate system alignment with a dataset of systematically
altered textbooks that contradict the LLM's latent knowledge. We find that
OpenAI Vector Search RAG (representing vector-based RAG) performs well as a
low-cost generalist, especially for quick fact retrieval. On the other hand,
GraphRAG Global excels at providing pedagogically rich answers to thematic
queries, and GraphRAG Local achieves the highest accuracy with the dense,
altered textbooks when corpus integrity is critical. Accounting for the 10-20x
higher resource usage of GraphRAG (representing graph-based RAG), we show that
a dynamic branching framework that routes queries to the optimal retrieval
method boosts fidelity and efficiency. These insights provide actionable
guidelines for educators and system designers to integrate RAG-augmented LLMs
into learning environments effectively.

## Full Text


<!-- PDF content starts -->

Aligning LLMs for the Classroom with Knowledge-
Based Retrieval: A Comparative RAG Study
Amay Jain
Student
Downingtown STEM Academy
Downingtown, PA
jain.amay04@gmail.comLiu Cui
Department of Computer Science
West Chester University of Pennsylvania
West Chester, PA
lcui@wcupa.eduSi Chen
Department of Computer Science
West Chester University of Pennsylvania
West Chester, PA
schen@wcupa.edu
Abstract—Large language models like ChatGPT are increas-
ingly used in classrooms, but they often provide outdated or
fabricated information that can mislead students. Retrieval
Augmented Generation (RAG) improves reliability of LLMs by
grounding responses in external resources. We investigate two
accessible RAG paradigms, vector-based retrieval and graph-
based retrieval to identify best practices for classroom question
answering (QA). Existing comparative studies fail to account
for pedagogical factors such as educational disciplines, question
types, and practical deployment costs. Using a novel dataset,
EduScopeQA, of 3,176 questions across academic subjects, we
measure performance on various educational query types, from
specific facts to broad thematic discussions. We also evaluate sys-
tem alignment with a dataset of systematically altered textbooks
that contradict the LLM’s latent knowledge. We find that OpenAI
Vector Search RAG (representing vector-based RAG) performs
well as a low-cost generalist, especially for quick fact retrieval.
On the other hand, GraphRAG Global excels at providing
pedagogically rich answers to thematic queries, and GraphRAG
Local achieves the highest accuracy with the dense, altered
textbooks when corpus integrity is critical. Accounting for the
10-20x higher resource usage of GraphRAG (representing graph-
based RAG), we show that a dynamic branching framework that
routes queries to the optimal retrieval method boosts fidelity
and efficiency. These insights provide actionable guidelines for
educators and system designers to integrate RAG-augmented
LLMs into learning environments effectively.
Index Terms—Large language models, Educational technology,
Retrieval augmented generation
I. INTRODUCTION
Large-language models (LLMs) have rapidly entered sec-
ondary and post-secondary classrooms, promising adaptive
tutoring and richer instructional material. However, educators
consistently report a critical obstacle that limits dependable
classroom use:misalignment with educational goals. [1].
LLMs struggle with hallucinated or fabricated content
and responses misaligned with curriculum standards. Their
internet-scale training can introduce tangential details that,
while sometimes helpful, may confuse students or undermine
course coherence [2].Knowledge shiftsfurther complicate
this: school curricula are periodically updated, and facts or
conventions can change (historical interpretations, scientific
Preprint notice—This work has been submitted to the IEEE for possible
publication. Copyright may be transferred without notice, after which this
version may no longer be accessible.nomenclature, etc.). An LLM trained on older information can
supply superseded facts or methodologies. [3]
Retrieval-Augmented Generation (RAG) [4], an approach
that supplements generative models with knowledge retrieval,
has emerged as a promising strategy to address these issues [5].
Two specific RAG variants,vector-based RAGandgraph-
based RAG, have gained popularity.
Vector-based RAG is the prevailing standard for retrieval
augmentation, using embedded vector similarity [6]. Graph-
based RAG extends this by incorporating graph-structured
knowledge bases, enabling nuanced multi-hop retrieval and
information synthesis [7]. While prior work has compared
these methods, their relative merits forcurriculum-aligned
classroom question anweringremain under-explored.
Recently, other RAG methods have emerged (sparse, hy-
brid, neural symbolic, etc.), but in this paper, we focus on
vector-based and graph-based RAG, due to their minimal
infrastructure requirements and ability to leverage off-the-shelf
APIs. For classroom adoption, ease of use is paramount. Thus,
solutions requiring minimal setup, maintenance are preferred.
To ground our comparison in practical tools, we focus on
two widely popular turnkey RAG solutions that are already
accessible to most educators: OpenAI Vector File Search
(“OpenAI RAG”) [8] as our vector-based representative, and
Microsoft’s GraphRAG framework, available in both Local
and Global modes, as our graph-based representative.
Since the needs of educators and students differ across
disciplines, scopes, and tasks, we design two complementary
evaluations. InCase Study 1(CS1), we test each system’s
ability to retrieve and synthesize information on 3,176 QA
pairs at different levels of granularity across four academic
subjects. InCase Study 2(CS2), we leverage the Know-
ShiftQA dataset [9], which contains 3,005 QA pairs from
textbooks in which key facts have been systematically altered,
to assess whether each retrieval method can accurately provide
updated information (from the texts) rather than falling back
on an LLM’s outdated or common knowledge.
Beyond accuracy, we measure indexing cost, response la-
tency, and scalability, factoring the typical constraints of
schools and universities.
Our evaluation addressesthree key research questionsfor
educational deployment:arXiv:2509.07846v1  [cs.AI]  9 Sep 2025

Research Questions.
1) In multi-disciplinary classroom QA, how do vector-
based RAG and graph-based RAG differ with respect
to answer accuracy and explanation quality?
2) When curricular knowledge shifts, which method better
resists outdated or superseded information?
3) Under typical school constraints, how do these methods
compare in resource efficiency?
This work offers three maincontributions:
1) a novel, multi-subject, multi-scope QA dataset fo-
cused on secondary and post-secondary educational
applications—EduScopeQA;
2) a comparative evaluation of vector-based vs. graph-
based RAG methods against educational factors like sub-
ject, question type, text size, and resource constraints;
3) a proof of concept showing how each system’s strengths
found in the evaluation can guide practical deployment
and usage of LLM-based QA systems in education.
II. RELATEDWORK
Vector-based RAG[6] chunks and vector-encodes a pro-
vided corpus using a neural embedding model. The user query
is similarly embedded, and the most semantically similar
chunks from the corpus are retrieved and passed to the LLM
with the query to produce grounded answers [4]. There is
a growing body of research exploring vector-based RAG in
educational tasks. [10] found that vector-based RAG improved
the traceability of mathematical solutions, and [11] used
vector-based RAG to boost automated grading accuracy.
OpenAI’s Vector File Search API (OpenAI RAG) uses a
managed vector store to perform semantic search over up-
loaded files. It automates chunking, embedding, and retrieval
within a single API call. With these abstractions, OpenAI’s
pipeline dramatically simplifies the use of vector-based RAG
in everyday use cases such as education. Given OpenAI
ChatGPT’s mainstream adoption, especially in education, [12],
evaluating OpenAI RAG provides practical relevance.
Graph-based RAGuses an LLM to organize documents
into a structured knowledge graph. The system identifies key
people, places, and concepts, and constructs a graph with these
identified entities as nodes and their relationships as edges.
The LLM then produces summaries for clusters of entities.
Thus, at query time, graph-based RAG traverses this graph,
aggregating evidence from the relevant nodes [7].
There are different retrieval configurations for graph-based
RAG that depend on the locality of the graph traversal.
“Local” community search retrieves node-centred subgraphs
and their immediate community summaries, whereas “Global”
search aggregates summaries across all communities, spanning
the entire document’s knowledge structure. Local emphasizes
precision, while global values coverage [7].
Although many variants of graph-based RAG exist, such as
LightRAG [13] and LazyRAG, we use Microsoft’s GraphRAG
as a representative due to its open-source nature, ease of
use, and comparable performance, which makes it widely
accessible to educators.Several studies have compared vector and graph-based
RAG, but on non-educational corpora and with evaluations
not focused on curriculum-aligned QA. In GraphRAG’s initial
paper, global graph retrieval achieved comprehensiveness win
rates of 72–83% on podcast transcripts and 72–80% on news
articles, outpacing vector RAG, yet these tasks emphasize
broad summarization metrics (e.g., ROUGE, diversity) rather
than pinpointed, QA accuracy on textbook content [7].
However, GraphRAG’s strengths in large datasets remain
unclear. One study [14] found that GraphRAG Local/Global
actually underperforms compared to vector RAG on large
narrative datasets such as NovelQA by 4%-18%, whereas
[15] found that on the Ultra Domain Benchmark, GraphRAG
outpaces vector RAG as dataset size increases.
One significant barrier to a pedagogical evaluation is that
existing QA datasets such as the NQ dataset [16], HotPotQA
[17], and MultiHopRAG [18] draw on fairly homogeneous
sources such as Wikipedia or news articles, not reflecting the
varied styles, structures, and lengths of real educational texts.
Curriculum-aligned datasets exist mainly for scientific knowl-
edge. CK-12 QA (TQA) [19] includes 12,000 middle-school
science multiple-choice items linked to textbook excerpts.
OpenBookQA [20] contains 5,957 elementary-level science
multiple-choice questions paired with facts. However, these
mostly focus on shorter texts, and do not span a broad range of
educational disciplines or support explanatory questions—gaps
which our new dataset aims to fill.
In light of these works, we build on prior evaluations
by focusing on realistic educational texts, varying lengths,
subjects, and question types. Additionally, we account for
pedagogical value (directness and learnability), knowledge
integrity under shifts, and resource constraints of real schools.
III. CASESTUDY1 (CS1)
A. Novel Dataset - EduScopeQA
We constructed a comprehensive dataset of 3,176 QA
pairs and their respective texts spanning four academic
subjects—History, Literature, Science, and Computer Sci-
ence—for a combined size of 2.1 million tokens (≈500,000
tokens per subject). We intentionally selected texts resembling
those studied in schools or universities.
ForLiteratureandHistory, we picked some of Project
Gutenberg’s [21] most downloaded works in their categories:
two lengthy fictional novels and a variety of primary and
secondary historical texts.Scienceis a full-length textbook,
chosen for its structured organization of facts, processes, and
terminology, and theComputer Scienceset is composed of
highly-cited technical monographs from arXiv [22]. Each sub-
ject’s corpus differs from each other in lexical and syntactical
complexity, fluidity, and density of information, mimicking
real classrooms, and allowing for a complete assessment of
the strengths and limitations of RAG systems.
More information and citations can be seen in Table I.
The dataset and a download script for texts is available at:
https://github.com/Amay-J/EduScopeQA. This release covers

History, Literature, and Science. Licensing information ap-
pears in the README.
TABLE I
EDUSCOPEQA DATASETCOMPOSITION
Subject Source Text Words Specific Sectional Thematic
Literature
[23] [24]Moby-Dick; or, The Whale 208,465 356 40 20
Little Women 188,683 323 40 20
Subtotal 397,148 679 80 40
History
[25] [26]
[27] [28]
[29] [30]The North Pole: Its Discovery 97,980 173 20 10
A History of the Philippines 77,475 173 16 8
Autobiography of Benjamin
Franklin76,108 134 15 7
Economic Consequences of the
Peace69,966 123 14 7
Life of Frederick Douglass 40,750 72 8 4
Common Sense 21,857 38 4 2
Subtotal 384,136 713 77 38
Computer
Science
[31] [32]
[33] [34]
[35] [36]
[37]Modern Introduction to Online
Learning103,145 171 20 10
Common Information, Noise
Stability79,782 133 15 7
Foundations of Vector
Retrieval65,275 100 12 6
Machine Learning
Fundamentals60,133 101 11 5
Community Detection
Algorithms42,630 71 8 4
Convex Optimization 33,051 55 6 3
System Architecture 18,039 33 0 0
Subtotal 402,055 664 72 35
Science
[38]Microbiology Textbook 397,994 678 80 20
Subtotal 397,994 678 80 20
Total Dataset 1,581,333 2,734 309 133
We systematically generated 3,176 question-answer (QA)
pairs across texts. The questions are open-ended to simulate
how a student or instructor might ask about the material.
Crucially, we categorized each question into one of three types
based on the scope of information required to answer:
•Specific Questions: These are narrow questions answer-
able by a single paragraph (≈500 words). These typically
focus on a specific fact or definition. In theMicrobiology
textbook [38], a specific question is: “Which genera of
soil bacteria are involved in the denitrification process?”
•Sectional Questions: These questions require aggregat-
ing information from multiple paragraphs (i.e. a chapter).
For example, “How did U.S. President Wilson’s ap-
proach to negotiation affect the Paris Peace Conference?”
Answering this requires pulling points spread across a
section ofEconomic Consequences of Peace[28].
•Thematic Questions: These are broad questions that
relate to overarching themes or cross-cutting concepts.
For instance, a question fromMoby Dick[23]: “What
does the Whaleman’s Chapel represent?” The system
must draw on understanding from the novel as a whole,
involving reasoning over tens of thousands of words.
The question generation pipeline handles long texts by
breaking them into manageable pieces and iteratively sum-
marizing and filtering content. The pipeline can be seen in
Fig. 1, and operates as follows:
1)Chunking/Sectioning: Each text is split into chunks.
Sets of ten consecutive chunks are grouped into sections.
2)Content Screening: GPT-4.1 examines each chunk and
section, filtering those that consist of irrelevant or extra-neous content such as front matter, textbook objectives,
publication info, and acknowledgments.
3)Hierarchical Summarization: For each section, GPT-
4.1 generates a concise summary capturing its key
points. If the combined content still exceeds about
35,000 characters, the summaries are sectioned and
summarized again. This recursive summarization con-
tinues until we obtain a global summary of the entire
text. This hierarchy (chunk→section summary→
global summary) compresses the text while preserving
important information at multiple levels.
4)Specific/Sectional Question Generation: From the
screened sections and chunks, a subset is randomly
sampled. For each chosen one, GPT-4.1 is provided a
summary of the surrounding sections and the global
summary. With this, GPT-4.1 generates QA pairs spe-
cific to the current section while understanding the
broader context. This helps produce coherent questions
that remain answerable from the target information.
5)Thematic Question Generation: Thematic questions
are produced using the global summary, prompting GPT-
4.1 to cover the main themes or topics of the entire text.
6)Filtering and Review: Each QA pair is passed through
GPT-4.1 again filtering out unanswerable or trivial ques-
tions. Steps 4/5 are repeated accordingly.
These dataset construction decisions were taken to mimic
educational applications. Notably, no QA dataset exists that
varies subject and question scope in this way. By evaluating
performance across these categories, we can analyze whether a
method excels at pinpointing facts but struggles with synthesis,
or vice versa. This granularity is valuable for educational
applications: a simple AI tutor might be fine if it can recall
facts, but true learning assistance requires the system to also
explain broad themes correctly.
B. Experiment
The full corpus of text for each subject was uploaded
(for OpenAI RAG) and indexed (for GraphRAG Local and
GraphRAG Global) using GPT-4.1-Mini. Then, the questions
were used as input to querying both systems one at a time,
again using GPT-4.1-Mini. The answers of each system were
captured and then evaluated for a comparison analysis below.
C. Evaluation
We evaluated each answer using an “LLM-as-a-Judge”
technique, which has been shown to achieve human-level
consistency for open-ended answers [39]. For each question,
we compared every pair of systems. For each comparison, we
used a fixed prompt template that presented GPT-4.1-Nano
with the two candidate answers (‘Option A’ and ‘Option B’)
and an instruction to choose the better response or declare a tie
ata given criteria. Four complementary criteria were chosen:
1)Comprehensiveness: Does the answer cover all relevant
points and facets of the question?
2)Directness: Is the answer succinct, and to the point
without unnecessary digression?

Fig. 1. EduScopeQA Dataset Question Generation Pipeline
3)Faithfulness: Is the answer faithful to the ground truth?
4)Learnability: How well does the answer help a student
learn or understand the topic? This criterion covers clar-
ity of explanation, quality of reasoning, and pedagogical
value.
These criteria build on recent RAG QA studies, such as
[13] and Microsoft GraphRAG’s introduction paper [7], that
use LLM-as-a-Judge for their evaluations. The criteria were
adapted to assess not only factual accuracy but also instruc-
tional clarity and pedagogical value. By including Directness
and Comprehensiveness, the rubric inherently counterbalances
verbosity and coverage biases, enabling fairer, more meaning-
ful comparisons.
Empirically, LLM-as-a-judge evaluations are prone to posi-
tional bias, a systematic tendency of LLMs to prefer an answer
due to its position in the prompt [40]. To mitigate this, for each
pairwise comparison, we followed an AB-BA strategy, passing
in the two candidate answers as “Option A” and “Option B”,
and then swapping them, feeding the original “Option B” as A
and “Option A” as B. We then aggregated the two judgments
into a final winner or ‘tie’. Studies have shown that such a
strategy balances out the positional skew [41].
D. Discussion
In order to comparatively analyze the LLM-as-a-judge eval-
uation, we used “Win Rate” as follows
WA=wA+ 0.5·t A
n(1)
whereW Ais the win rate for AI systemA,w Ais the
number of wins for systemA,t Ais the number of ties for
systemA, andnis the total number of pairwise comparisons
being considered, leading to a score out of 1, where higher
values mean a higher win percentage at a certain criterion.
The full results of our experiment can be seen in Table II,
and a visual comparison of win rate across question types can
be seen in Fig. 2.
GraphRAG Global dominates Broad Queries and Ped-
agogical Criteria. GraphRAG Global achieves the highest
win rates in Faithfulness in Sectional and Thematic questions.
Its multi-hop retrieval, leveraging the hierarchical structure
of knowledge graphs, synthesizes dispersed information ef-
fectively. GraphRAG Global also dramatically scores well in
Comprehensiveness and Learnability across question types and
subjects. This shows that its long-range connectivity allows the
LLM to generate comprehensive, pedagogically rich responses
that support learning. This supports other results that global
graph-based RAG excels at “broad” questions and implies its
effectiveness as a tool for teaching concepts.OpenAI RAG excels at Specific Queries and Directness.
While GraphRAG Global scores well on the pedagogical crite-
ria for specific questions, it is overshadowed by OpenAI RAG
in the other criteria. Since specific factual queries often reside
in a single retrieved snippet, GraphRAG Global struggles to
capture these details with high-level summarization. Conse-
quently, OpenAI RAG is optimal for “flashcard” applications,
quick glossary references, and scenarios requiring immediate,
precise answers.
GraphRAG Local acts as a Competent Bridge. And
in between, GraphRAG Local scores better on Faithfulness
and Directness at specific questions than GraphRAG Global,
but performs better at pedagogical criteria (Comprehensive-
ness and Learnability) than OpenAI RAG across the board.
By constraining graph reasoning to local neighborhoods, it
yields more complete responses when answers span several
paragraphs, but leads to limitations in broad thematic coverage
in comparison with GraphRAG Global.
Subject Variations. While overall patterns hold, there are
nuances between subjects. For example, in Computer Science,
the gap between OpenAI RAG and GraphRAG Global for
Faithfulness in sectional (0.562 vs. 0.764) and thematic (0.625
vs. 0.812) is much smaller compared with Literature (0.125
vs. 0.781 and 0.125 vs. 0.675 respectively). This underscores
that for fictional novels, which make up the Literature corpus,
truly global cues (motifs, narrative arcs) are widely dispersed
and benefit greatly from global retrieval, whereas for technical
papers, which make up the Computer Science corpus, factual
claims make up the bulk of subject matter and impose fewer
integrative demands. Science follows results similar to Com-
puter Science, while History falls nearer to Literature.
IV. CASESTUDY2 (CS2)
For Case Study 2, we used theKnowShiftQA(KSQA)
dataset, which is specifically designed to test a retrieval
system’s ability to prioritize provided source material over
an LLM’s internal knowledge [9]. The dataset comprises five
textbooks from secondary education level subjects: Physics,
Chemistry, Biology, Geography, and History. In each textbook,
certain pieces of factual information have been systematically
altered to simulate hypothetical knowledge updates. Impor-
tantly, these changes are done in a coherent way so that the
surrounding context in the textbook remains plausible and
internally consistent. These alterations include changing the
textbook to claim that “Night-vision goggles detect ultraviolet
light,” from infrared light (in reality). So, a question like “What
type of light is detected by night-vision goggles?” would test
a QA system’s faithfulness to the input corpus much better.

TABLE II
AVERAGEWINRATES- ALLMETHODS
Metric Method Computer Science History Literature Science
Specific Sectional Thematic Specific Sectional Thematic Specific Sectional Thematic Specific Sectional Thematic
ComprehensivenessOpenAI RAG 0.200 0.264 0.429 0.214 0.221 0.237 0.204 0.138 0.263 0.176 0.156 0.250
GraphRAG Local 0.422 0.451 0.243 0.429 0.506 0.382 0.407 0.600 0.463 0.444 0.550 0.575
GraphRAG Global 0.879 0.812 0.829 0.857 0.766 0.868 0.889 0.750 0.825 0.880 0.806 0.650
DirectnessOpenAI RAG 0.780 0.556 0.312 0.875 0.506 0.618 0.870 0.600 0.700 0.676 0.794 0.600
GraphRAG Local 0.421 0.444 0.562 0.482 0.396 0.224 0.426 0.594 0.425 0.574 0.150 0.375
GraphRAG Global 0.298 0.556 0.625 0.143 0.597 0.684 0.204 0.312 0.388 0.231 0.569 0.625
AccuracyOpenAI RAG 0.599 0.562 0.471 0.625 0.442 0.395 0.815 0.494 0.300 0.324 0.125 0.125
GraphRAG Local 0.452 0.208 0.214 0.661 0.188 0.197 0.435 0.394 0.438 0.491 0.581 0.675
GraphRAG Global 0.451 0.764 0.871 0.214 0.851 0.868 0.250 0.588 0.750 0.685 0.781 0.675
LearnabilityOpenAI RAG 0.209 0.319 0.175 0.232 0.253 0.171 0.278 0.163 0.338 0.176 0.431 0.225
GraphRAG Local 0.451 0.389 0.450 0.464 0.240 0.447 0.398 0.588 0.362 0.435 0.512 0.425
GraphRAG Global 0.840 0.771 0.825 0.804 0.994 0.882 0.824 0.725 0.775 0.889 0.556 0.850
Fig. 2. Case Study 1 Results: Average Win Rates across question types and criteria
This is an important control for testing retrieval systems, as
it renders some of the LLM’s latent parametric knowledge
incorrect, forcing it to rely on the retrieval.
The dataset contains 3,005 QA pairs. Each question is
a multiple-choice question (MCQ) with one correct answer
(which corresponds to the altered fact) and several distractors
(including the real-world fact).
A. Experiment
We also compare how document length matters since course
material range from short articles and handouts to textbooks
and novels. We boost the KSQA dataset by creating three
experimental settings (KSQA provides alignment of questions
to the source text):
•Short-Retrieval: Only the chunk from which each ques-
tion was generated was provided as the input corpus (≈
315 words).
•Medium-Retrieval: Only the 30 chunks surrounding the
one each question was generated from were provided as
the input corpus (≈9.5 K words).
•Full-Retrieval: The entire textbook was provided as the
input corpus.
In all three settings, the appropriate text was uploaded
(for OpenAI RAG) and indexed (for GraphRAG Local and
GraphRAG Global) using GPT-4.1-Mini. Then, both systems
were queried one at a time, again using GPT-4.1-Mini. The
MCQ format was handled by including the options and
prompting the model to choose an answer.
It is important to note that a prompt instructing the LLM to
ignore factuality of answers was passed in at query-time to allthree systems. The full prompt is seen in Fig. 3. Furthermore,
GraphRAG automatically generates prompts that are used to
query the knowledge graph at query-time. Some of these
graphs contain wording encouraging outside reasoning, such as
“The response may also include relevant real-world knowledge
outside the dataset”. To standardize the experiment, these
sentences were removed and replaced with the same prompt.
Fig. 3. LLM Prompt for Case Study 2
B. Evaluation
We checked if the chosen answer was the correct (altered)
one by performing a fuzzy matching using a GPT-4.1-Nano
evaluation. We measured accuracy in terms of percentage of
questions answered correctly.
C. Discussion
Our findings (Fig. 4) reveal clear performance patterns
influenced by corpus size and retrieval scope:
GraphRAG Local for Large, Dense Corpora. In full-
retrieval, especially in the larger Biology (258 K words), His-
tory (146 K words), and Geography (165 K words) textbooks,
GraphRAG Local consistently outperforms both OpenAI RAG
and GraphRAG Global. Its local graph structure efficiently
identifies precise factual information amidst large volumes

Fig. 4. Case Study 2 Results: Average Accuracy by Subject and Retrieval Scope.Large textbooks- Biology (258K words), History (146K), Geography
(165K);Smaller Textbooks- Chemistry (77K), Physics (68K)
of potentially distracting content. Previous works, and even
Case Study 1 imply that vector-based RAG performs better on
specific, factual questions. We show that for a large corpus of
dense facts, GraphRAG Local is better at identifying minutiae
and maintaining strict adherence to curriculum content.
OpenAI RAG in Smaller Corpora. In smaller texts such
as Chemistry (77 K words) and Physics (68 K words), OpenAI
RAG closely matches or slightly outperforms GraphRAG
Local. Across the medium and short retrieval conditions, all
systems converge on high scores, but OpenAI RAG generally
leads. With reduced corpus size, vector retrieval precision
effectively compensates for lack of structured multi-hop capa-
bilities, indicating diminishing returns for graph construction
(which can actually add distractors).
V. OVERALLDISCUSSION
For both case studies, we recorded the number of LLM calls
during indexing, the total indexing time, and the query time
(Table III). . We useLLM calls as a proxy for monetary cost
for generalization and normalization since billing depends on
model provider and model variant pricing.
TABLE III
AVGINDEXINGPERSUBJECT ANDQUERYCOSTS BYCASESTUDY(CS)
Indexing OpenAI RAG GraphRAG
Time (s) Time (s) LLM Calls
CS1 11.43 2,142.22 4,025.25
CS2-Full 5.87 1,078.12 2,038.4
CS2-Medium 3.82 186.61 112.45
CS2-Short 3.53 48.12 10.28
Querying OpenAI RAG GraphRAG Local GraphRAG Global
Time (s) Time (s) Time (s)
CS1 4.71 36.50 70.12
CS2-Full 5.00 35.57 39.41
CS2-Medium 3.92 34.61 42.12
CS2-Short 3.07 9.43 9.95Notably, GraphRAG (Local and Global were indexed to-
gether) required substantial computational resources during
indexing for entity and relationship extraction, whereas Ope-
nAI RAG was lightning fast with zero additional LLM calls,
as embedding is handled internally by OpenAI’s infrastruc-
ture. The indexing burden for GraphRAG also scaled with
document length (short, medium, full), whereas OpenAI RAG
remained relatively stable. GraphRAG’s one-time indexing of
an average textbook from Case Study 1 required 35 minutes
and 4,000 LLM calls. This overhead can exceed schools’
budget if done regularly, particularly with costlier LLMs. This
was even more applicable while querying, where GraphRAG
Global (and Local to a lesser extent) scaled significantly.
Altogether, this provides a more complete discussion of
pedagogical deployment.(1) Vector-based RAG is excellent
for most cases, especially when students need individual,
rapid, pinpoint responses such as targeted explanations about a
paragraph or quick glossary lookups. Its low latency and ease
of setup make it ideal for embedding into general chatbots
without overburdening school IT resources.(2) GraphRAG’s
high initial costs can be justified when a particular corpus
can be indexed and shared across users over time. When
the goal is to support essay prompts or seminar discussions
centered on a classroom text, where students benefit from
rich, concept-spanning explanations,(3) GraphRAG Global
provides the most coherent, curriculum-aligned narratives.
Despite higher runtime costs, this method can be justified in
settings where depth of understanding matters most. For large,
evolving textbooks, question banks, or multiple-choice ques-
tions,(4) GraphRAG Local offers accuracy and context-
sensitivity. Its tight adherence to the provided material is
critical when ensuring that answers align exactly with the latest
curriculum standards or exam specifications.

Fig. 5. Branching prompt to choose the optimal retrieval method
A. Proof of Concept Branching System
Although separating the deployments of the systems by use
case is an option, we show as a proof of concept a lightweight
branching system which routes incoming queries based on
complexity, scope, and corpus size to the appropriate retrieval
system. We use an initial GPT-4.1-Nano call with the prompt
shown in Fig. 5. The LLM is instructed to choose based on a
short description of each system’s strengths.
As a basic test, we repeated Case Study 1, but with the
branching system as an addition. Each question and its input
text were first passed through the LLM prompt in Fig. 5, and
then the chosen system’s previous response was picked. We
show the averaged results of the LLM-as-judge evaluation,
withoutconverting to win-rates, in Table IV. We also repeated
Case Study 2 with the branching system, where the options of
the MCQ were passed along with the question. Overall results,
averaged across subjects and retrieval scopes, can be seen in
Fig. 6. In Case Study 1, the branching system achieved
the highest overall faithfulness scores of any single system,
reflecting its ability to invoke OpenAI RAG for specific
queries and GraphRAG Global for broader questions. By
conditionally choosing a system according to question granu-
larity, the brancher harnesses the strengths of both approaches.
On criteria such as learnability and directness, however, the
branching system occupies an intermediate position: it out-
performs standalone OpenAI RAG or GraphRAG respectively
in avoiding extreme weaknesses, but does not match each
TABLE IV
CASESTUDY1 RESULTS WITHBRANCHINGSYSTEM:
AVERAGEDLLM-AS-A-JUDGEWINPERCENTAGES,
BRANCHINGSYSTEMS VS.THE REST
MetricOpenAI
RAGGraphRAG
LocalGraphRAG
Global
Comprehensiveness 72.4% 67.6% 37.0%
Directness 39.2% 84.0% 66.1%
Faithfulness 68.5% 79.8% 60.2%
Learnability 80.1% 74.3% 33.4%system’s highest strengths, GraphRAG Global’s rich narratives
or OpenAI RAG’s most concise responses. This underscores
a natural trade-off when balancing across question types. For
Case Study 2, it is evident that the branching system was
able to take advantage of OpenAI RAG’s strengths in shorter
retrieval scopes and GraphRAG Local’s higher accuracy in
the larger corpora. Although the net gain in multiple-choice
accuracy was more modest than in our open-ended evaluation,
it displays that a branching system that dynamically chooses
the best system situationally can be a useful tool to improve
the performance of a QA system.
The resource costs of the branching system in both case
studies are shown in Table V. Although costs are much
lower than a pure GraphRAG system, they remain higher
than a pure OpenAI RAG system, suggesting that further
optimization is needed to make the branching system more
cost-effective. Importantly, our cost calculations conservatively
include indexing overhead for every routed query; in practice,
schools can amortize GraphRAG’s setup costs by persisting
indexed corpora across class cohorts and indexing during off-
hours. Such optimizations would further narrow the price gap
and enable sustainable, large-scale adoption of graph-based or
dynamic RAG strategies in educational settings.
VI. CONCLUSION ANDFUTUREWORK
This paper offers a comprehensive, education-focused com-
parison of vector and graph-based RAG methods across realis-
tic classroom QA scenarios. After testing with EduScopeQA,
a novel multi-subject, multi-scope open-ended dataset, and
evaluating accuracy under knowledge shifts, we demonstrate
clear pedagogical trade-offs for real-world deployment.
OpenAI RAG excels at rapid, precise fact retrieval due to
its low resource cost. But when the focus on a particular
corpus of text allows for longer-term shared use, GraphRAG’s
Fig. 6. Case Study 2 Results with Branching System, averaged across subjects
and retrieval scopes
TABLE V
BRANCHINGINDEXING ANDQUERYCOSTS(CS - CASESTUDY)
Indexing Querying
Time (s) LLM Calls Time (s)
CS1 1,378.11 2,582.04 44.94
CS2 360.01 676.07 14.11

high initial costs can be justified with improved accuracy and
pedagogical value. Global produces the richest, curriculum-
aligned explanations for thematic inquiry and Local ensures
the highest fidelity when adhering to dense, evolving textbook
content or multiple-choice questions.
Our proof-of-concept branching framework demonstrates
that intelligently routing questions to the optimal retrieval
paradigm can yield improved accuracy while avoiding each
method’s limitations and reducing unnecessary computational
overhead. These insights guide future efforts to deploying
RAG-augmented LLMs in instructional contexts: whether
powering live study aids, guiding deep seminar discussions,
or safeguarding the integrity of classroom teachings.
Limitations and Future Work.Next steps should center
on classroom pilots and co-design studies with teachers and
students to validate our evaluation’s alignment with actual ed-
ucational outcomes. Second, our evaluation only accounted for
textual class materials; this calls for pedagogical evaluations of
multimodal and visual RAG pipelines for educational images
and videos. Third, the branching mechanism itself can be made
robust by accounting for more factors and performing more
rigorous testing. With these future directions, we can close the
gap between technical innovation and real-world classrooms,
ensuring AI systems remain aligned with diverse curricula and
pedagogical goals.
REFERENCES
[1] F. Kamalov, D. S. Calonge, and I. Gurrib, “New era of artificial
intelligence in education: Towards a sustainable multifaceted revolution,”
Sustainability, vol. 15, no. 16, 2023.
[2] Y . Zhanget al., “Siren’s song in the ai ocean: A survey on hallucination
in large language models,” 2023, arXiv preprint arXiv:2309.01219.
[3] C. K. Lo, “What is the impact of chatgpt on education? a rapid review
of the literature,”Education Sciences, vol. 13, no. 4, 2023. [Online].
Available: https://www.mdpi.com/2227-7102/13/4/410
[4] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
nlp tasks,”CoRR, vol. abs/2005.11401, 2020.
[5] I. Iaroshev, R. Pillai, L. Vaglietti, and T. Hanne, “Evaluating
retrieval-augmented generation models for financial report question
and answering,”Applied Sciences, vol. 14, no. 20, 2024. [Online].
Available: https://www.mdpi.com/2076-3417/14/20/9318
[6] Y . Gaoet al., “Retrieval-augmented generation for large language
models: A survey,” 2024, arXiv preprint arXiv:2312.10997.
[7] D. Edgeet al., “From local to global: A graph rag approach to query-
focused summarization,” 2024, arXiv preprint arXiv:2404.16130.
[8] OpenAI, “Retrieval - openai platform documentation,” 2024,
[Online]. Available: https://platform.openai.com/docs/guides/retrieval#
vector-stores.
[9] T. Zheng, W. Li, J. Bai, W. Wang, and Y . Song, “Knowshiftqa:
How robust are rag systems when textbook knowledge shifts in k-12
education?” 2025, in press.
[10] Z. Levonianet al., “Retrieval-augmented generation to improve math
question-answering: Trade-offs between groundedness and human pref-
erence,” 2023, arXiv preprint arXiv:2310.03184.
[11] Y . Chuet al., “Enhancing llm-based short answer grading with
retrieval-augmented generation,” 2025, arXiv preprint arXiv:2504.05276,
doi:10.48550/arXiv.2504.05276.
[12] R. Deng, M. Jiang, X. Yu, Y . Lu, and S. Liu, “Does chatgpt enhance
student learning? a systematic review and meta-analysis of experimental
studies,”Computers & Education, vol. 227, p. 105224, 2025.
[13] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast
retrieval-augmented generation,” 2025, arXiv preprint arXiv:2410.05779.
[14] H. Hanet al., “Rag vs. graphrag: A systematic evaluation and key
insights,” 2025, arXiv preprint arXiv:2502.11371.[15] Q. Zenget al., “How significant are the real performance gains? an
unbiased evaluation framework for graphrag,” 2025, arXiv preprint
arXiv:2506.06331.
[16] T. Kwiatkowskiet al., “Natural questions: A benchmark for question
answering research,”Transactions of the Association of Computational
Linguistics, 2019.
[17] Z. Yanget al., “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,”CoRR, vol. abs/1809.09600, 2018.
[18] Y . Tang and Y . Yang, “Multihop-rag: Benchmarking retrieval-
augmented generation for multi-hop queries,” 2024, arXiv preprint
arXiv:2401.15391.
[19] A. Kembhavi, M. Seo, D. Schwenk, J. Choi, A. Farhadi, and H. Ha-
jishirzi, “Are you smarter than a sixth grader? textbook question answer-
ing for multimodal machine comprehension,” in2017 IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5376–
5384.
[20] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor
conduct electricity? a new dataset for open book question answering,”
CoRR, vol. abs/1809.02789, 2018.
[21] P. Gutenberg, [Online]. Available: https://www.gutenberg.org/.
[22] arXiv, [Online]. Available: https://arxiv.org/.
[23] H. Melville, “Moby-dick; or, the whale,” 1851, [Online]. Available:
https://www.gutenberg.org/ebooks/2701.
[24] L. M. Alcott, “Little women; or, meg, jo, beth, and amy,” 1868, [Online].
Available: https://www.gutenberg.org/ebooks/514.
[25] R. E. Peary, “The north pole: Its discovery in 1909 under the auspices of
the peary arctic club,” 1910, [Online]. Available: https://www.gutenberg.
org/ebooks/18975.
[26] D. P. Barrows, “A history of the philippines,” 1905, [Online]. Available:
https://www.gutenberg.org/ebooks/38269.
[27] B. Franklin, “The autobiography of benjamin franklin,” 1791, [Online].
Available: https://www.gutenberg.org/ebooks/20203.
[28] J. M. Keynes, “The economic consequences of the peace,” 1919,
[Online]. Available: https://www.gutenberg.org/ebooks/15776.
[29] F. Douglass, “Narrative of the life of frederick douglass, an american
slave,” 1845, [Online]. Available: https://www.gutenberg.org/ebooks/23.
[30] T. Paine, “Common sense,” 1776, [Online]. Available: https://www.
gutenberg.org/ebooks/147.
[31] S. Bruch, “Foundations of vector retrieval,” 2024, arXiv preprint
arXiv:2401.09350.
[32] L. Yu and V . Y . F. Tan, “Common information, noise stability, and their
extensions,” 2022, arXiv preprint arXiv:2211.01788.
[33] F. Orabona, “A modern introduction to online learning,” 2019, arXiv
preprint arXiv:1912.13213.
[34] O. Simeone, “A brief introduction to machine learning for engineers,”
2017, arXiv preprint arXiv:1709.02840.
[35] E. Abbe, “Community detection and stochastic block models: Recent
developments,” 2017, arXiv preprint arXiv:1703.10146.
[36] S. Bubeck, “Convex optimization: Algorithms and complexity,” 2014,
arXiv preprint arXiv:1405.4980.
[37] J. H. Bussemaker, P. Saves, N. Bartoli, T. Lefebvre, and R. Lafage,
“System architecture optimization strategies: Dealing with expensive
hierarchical problems,” 2025, arXiv preprint arXiv:2502.00838.
[38] N. Parker, M. Schneegurt, A.-H. T. Tu, B. M. Forster, and P. Lister,
“Microbiology,” 2016, [Online]. Available: https://openstax.org/books/
microbiology.
[39] C.-H. Chiang and H.-Y . Lee, “Can large language models be an
alternative to human evaluations?” inProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds.
Toronto, Canada: Association for Computational Linguistics, Jul. 2023,
pp. 15 607–15 631.
[40] L. Shi, C. Ma, W. Liang, X. Diao, W. Ma, and S. V osoughi, “Judging
the judges: A systematic study of position bias in llm-as-a-judge,” 2025,
arXiv preprint arXiv:2406.07791.
[41] G. H. Chen, S. Chen, Z. Liu, F. Jiang, and B. Wang, “Humans or llms
as the judge? a study on judgement biases,” inProceedings of the 2024
Conference on Empirical Methods in Natural Language Processing.
Miami, Florida, USA: Association for Computational Linguistics, Nov.
2024, pp. 8301–8327.
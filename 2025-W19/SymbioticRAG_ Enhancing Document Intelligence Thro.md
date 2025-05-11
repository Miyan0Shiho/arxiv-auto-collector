# SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration

**Authors**: Qiang Sun, Tingting Bi, Sirui Li, Eun-Jung Holden, Paul Duuring, Kai Niu, Wei Liu

**Published**: 2025-05-05 07:24:38

**PDF URL**: [http://arxiv.org/pdf/2505.02418v1](http://arxiv.org/pdf/2505.02418v1)

## Abstract
We present \textbf{SymbioticRAG}, a novel framework that fundamentally
reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a
bidirectional learning relationship between humans and machines. Our approach
addresses two critical challenges in current RAG systems: the inherently
human-centered nature of relevance determination and users' progression from
"unconscious incompetence" in query formulation. SymbioticRAG introduces a
two-tier solution where Level 1 enables direct human curation of retrieved
content through interactive source document exploration, while Level 2 aims to
build personalized retrieval models based on captured user interactions. We
implement Level 1 through three key components: (1)~a comprehensive document
processing pipeline with specialized models for layout detection, OCR, and
extraction of tables, formulas, and figures; (2)~an extensible retriever module
supporting multiple retrieval strategies; and (3)~an interactive interface that
facilitates both user engagement and interaction data logging. We experiment
Level 2 implementation via a retriever strategy incorporated LLM summarized
user intention from user interaction logs. To maintain high-quality data
preparation, we develop a human-on-the-loop validation interface that improves
pipeline output while advancing research in specialized extraction tasks.
Evaluation across three scenarios (literature review, geological exploration,
and education) demonstrates significant improvements in retrieval relevance and
user satisfaction compared to traditional RAG approaches. To facilitate broader
research and further advancement of SymbioticRAG Level 2 implementation, we
will make our system openly accessible to the research community.

## Full Text


<!-- PDF content starts -->

arXiv:2505.02418v1  [cs.IR]  5 May 2025SymbioticRAG: Enhancing Document Intelligence Through Human-LLM
Symbiotic Collaboration
Qiang Sun1*Tingting Bi1,3Sirui Li2
Eun-Jung Holden3Paul Duuring1,4Kai Niu1Wei Liu1*
1The University of Western Australia2Murdoch University
3The University of Melbourne4Geological Survey of Western Australia
{pascal.sun,kai.niu}@research.uwa.edu.au
sirui.li@murdoch.edu.au, {tingting.bi,eunjung.holden}@unimelb.edu.au
paul.duuring@demirs.wa.gov.au, wei.liu@uwa.edu.au
Abstract
We present SymbioticRAG , a novel frame-
work that fundamentally reimagines Retrieval-
Augmented Generation (RAG) systems by es-
tablishing a bidirectional learning relationship
between humans and machines. Our approach
addresses two critical challenges in current
RAG systems: the inherently human-centered
nature of relevance determination and users’
progression from "unconscious incompetence"
in query formulation. SymbioticRAG intro-
duces a two-tier solution where Level 1 en-
ables direct human curation of retrieved con-
tent through interactive source document ex-
ploration, while Level 2 aims to build person-
alized retrieval models based on captured user
interactions. We implement Level 1 through
three key components: (1) a comprehensive
document processing pipeline with specialized
models for layout detection, OCR, and extrac-
tion of tables, formulas, and figures; (2) an
extensible retriever module supporting multi-
ple retrieval strategies; and (3) an interactive
interface that facilitates both user engagement
and interaction data logging. We experiment
Level 2 implementation via a retriever strategy
incorporated LLM summarized user intention
from user interaction logs. To maintain high-
quality data preparation, we develop a human-
on-the-loop validation interface that improves
pipeline output while advancing research in
specialized extraction tasks. Evaluation across
three scenarios (literature review, geological ex-
ploration, and education) demonstrates signif-
icant improvements in retrieval relevance and
user satisfaction compared to traditional RAG
approaches. To facilitate broader research and
further advancement of SymbioticRAG Level 2
implementation, our system is freely available
viahttps://app.ai4wa.com .
1 Introduction
Recent advances in large language models (LLMs)
challenge the traditional search engines’ informa-
*Corresponding author.tion seeking convention by allowing users to pose
natural, conversational queries that yield synthe-
sized answers without explicitly retrieving source
documents for further manual processing (Zhu
et al., 2023). Although this paradigm shift can
be remarkably convenient for general inquiries,
it runs into significant limitations when handling
domain-specific or proprietary content that may not
be present in the pre-trained language models. In
such scenarios, relying solely on pre-trained mod-
els can exacerbate “hallucinations”, where LLMs
confidently present fabrications or partial truths (Ji
et al., 2023).
Retrieval-Augmented Generation (RAG) ad-
dresses these issues by coupling LLMs with re-
trieval mechanisms that fetch relevant documents
based on a user’s query (Craswell et al., 2022).
The grounding in external content helps reduce hal-
lucinations and provides up-to-date information.
Yet, deeper challenges remain. One concerns the
distinction between similarity andrelevance : re-
trieval components often return top-ranked chunks
by embedding similarity, which may fail to cap-
ture tangential but crucial aspects of the query.
Graph-based approaches such as GraphRAG (Edge
et al., 2024) seek to introduce greater diversity into
“relevance” by exploring knowledge graph struc-
tural features via community detection. However,
the ultimate arbiter is still the end users—they
decide whether retrieved contents are relevant or
not—underscoring that human should be at the
centre of this retrieval process, as individuals of-
ten may have varying goals or perspectives even
the query is the same. Moreover, current RAG
pipelines typically assume that users know what
they need to ask, aligning well with situations
in which knowledge gaps are consciously recog-
nized (Zhao et al., 2024a), which illustrates that
human centered design needs to be considered:
•From unknown unknowns to knowledge

Document 
Process 
Component
LOG
Inputs
PPTX
PNG
TXT
XLS
DOCX
....
Anything2PDF
Layout 
detection
Layout 
blocks
PDFs
OCR
Table 
extraction
Formula 
recognisation
Figure 
understanding
JSON: 
layout 
block 
bounding 
box, 
text, 
description 
information
Retriever 
Component
Embed 
user 
query
NaiveRAG
SymbioticRAG
Embed 
each 
layout 
block
Embed 
user 
query 
+ 
user 
intention 
summary
Label 
NaiveRAG
Retrieved 
Top-k 
blocks
SymbioticRAG 
UI 
Component
Send 
query
Human 
selected 
layout 
blocks
Interaction 
collection
Intention 
summary  
from 
user 
logs 
via 
LLM
Ideally 
Identical
Human 
in 
the 
loop: Figure 1: The overview of SymbioticRAG system design
discovery : real-world learning often starts
with “Unknown Unknowns” (or Unconscious
Incompetence ), as shown in Figure 2, where
individuals do not yet realize the breadth of
what they lack. For example., when a PhD stu-
dent embarks on an unfamiliar topic like quan-
tum computing, they may not even know what
questions to pose. A simple query-response
loop under RAG may suffice for direct, well-
defined questions, but it falls short in sup-
porting more human centered, exploratory,
and iterative learning process through which
novices actively engage with and gradually
uncover the breadth of a domain.
•Enhancing retrieval precision : another lim-
itation is the lack of a last-mile correction
mechanism, which allows users to refine re-
trieval when the content is close but not quite
accurate.
Unconscious 
CompetenceUnconscious 
Incompetence
Conscious 
IncompetenceConscious 
CompetenceUnaware of your 
knowledge gaps and 
misunderstandingsDeep understanding 
that allows natural 
application of knowledge
Aware of knowledge 
gaps and what you need 
to understandUnderstanding concepts 
but requires deliberate 
thought to apply them
KnowledgeWe are 
here
Figure 2: Four stages of competence proposed by Noel
Burch (source: Wikipedia1). Current RAG systems
effectively address Conscious Incompetence —stage
where users can recognize knowledge gaps. However,
they struggle with Unconscious Incompetence —users
remain unaware of key knowledge deficits, necessitating
more exploratory means to uncover these “unknown
unknowns.”
1https://en.wikipedia.org/wiki/Four_stages_of_To address these challenges in document intelli-
gence, we need to fundamentally rethink human-
machine interaction. This leads us to propose a
symbiotic framework - a system design principle
inspired by biological symbiosis where two species
live in close association and benefit from each other.
In our context, humans and the RAG system act as
two “species” that mutually influence and enhance
each other’s capabilities, eventually serving human
needs.
The framework should integrates complemen-
tary strengths from existing paradigms: it preserves
the successful interaction pattern of traditional
search engines where users can retrieve and ex-
amine original documents, while leveraging LLMs’
capabilities in fine-grained content retrieval and
synthetic result generation. The symbiotic rela-
tionship operates bidirectionally: from system to
human , where provided information, explanations,
and answers shape users’ thought processes and
knowledge acquisition as shown in Figure 2; and
from human to system , where the system learns
to understand and adapt to individual needs based
on user behavior. This design specifically targets
the limitations of current approaches: unlike tra-
ditional search engines that only support single-
round document retrieval, our approach enables
multi-round, context-aware document interactions
to support exploratory learning. It also differs from
current LLM-based systems such as ChatGPT or
existing RAG systems, which primarily operate
in a one-directional manner and lack comprehen-
sive interactive behaviors between human and sys-
tem. Recent developments like OpenAI’s Canvas2
demonstrate the industry’s recognition of the impor-
tance of human-machine bidirectional interaction,
competence
2https://openai.com/index/introducing-canvas/

though their current focus on post-generation edit-
ing has yet to extend to supporting user interaction
during the external retrieval process itself.
We present SymbioticRAG , a human–LLM col-
laborative system designed to address the key limi-
tations discussed above and enhance document in-
telligence. The solution comprises three core main
components: Document Processing ,Retriever ,
andSymbioticRAG UI .
•Document Processing : SymbioticRAG sup-
ports diverse document formats by first con-
verting them into PDFs and processing each
page as an image. A layout detection module
identifies bounding boxes for individual lay-
out blocks, such as text blocks, titles, tables,
figures, and formulas. Subsequently, these are
processed by specialized modules for OCR,
table extraction, figure understanding, and for-
mula recognition. This process transforms the
raw content into layout-aware representation,
enabling fine-grained retrieval once the layout
block chunks are embedded.
•Retriever : this module can accommodate var-
ious retrieval strategies, for example, the sim-
plest semantic similarity-based retrieval over
layout block embeddings, or integrate more
advanced approaches as they emerge. This
flexible architecture allows SymbioticRAG to
adapt to different requirements and incorpo-
rate state-of-the-art retrieval methodologies.
The retrieved layout blocks will be presented
within the context of their original documents,
allowing users to verify relevance in situ.
•SymbioticRAG UI: The interface enables
users to explore the document space, refine,
and iterate on retrieved layout blocks by click-
ing to include or remove specific ones. This
interactive process allows precise curation of
relevant layout blocks for downstream tasks
such as report generation. It is natural and
not onerous, which allows for subconsciously
high quality user interaction data collection.
We implemented SymbioticRAG and evaluated
it in three scenarios: geological report exploration ,
research literature review , and education . Our tests
assessed its effectiveness and user satisfaction by
comparing retrieved layout blocks from different
retrieval strategies with those selected by users in
multi-turn interactions. We also collected user feed-
back on interaction design and usability. Resultsshow that SymbioticRAG outperforms traditional
RAG systems, including ChatGPT and Claude,
in user satisfaction and engagement. Its human-
centered design enabled tasks previously imprac-
tical via current RAG systems, demonstrating its
versatility across diverse domains.
2 Motivation
Retrieval-Augmented Generation (RAG) LLM
hallucination arises from misalignment between
training data and reference sources (Huang et al.,
2024; Ji et al., 2023), often due to heuristic data
collection or the generative nature of NLG tasks.
As this issue remains unsolved, grounding model
outputs in source documents has become essential,
fueling the rapid development of RAG. RAG re-
trieves the most relevant content for user queries.
A basic approach ranks retrieved chunks using co-
sine similarity, equating relevance withsemantic
similarity . However, information retrieval (IR)
research distinguishes the two (Manning et al.,
2008), emphasizing that relevance also depends on
task context, timeliness, and credibility, etc (Barry,
1994; Saracevic, 2019). Bridging this gap re-
mains an active research challenge (Craswell et al.,
2022). Recent RAG advancements address this
issue through two main approaches. The first intro-
duces more diverse features into Relevance by inte-
grating graph structures, temporal information, and
domain-specific metadata, etc. GraphRAG (Edge
et al., 2024; Han et al., 2025) incorporates graph-
based features to enhance retrieval, while Hip-
poRAG (Gutiérrez et al., 2025) leverages knowl-
edge graphs and PageRank for filtering. The sec-
ond approach shifts toward end-to-end models that
directly learn relevance. For instance, Multi-Head
RAG (Besta et al., 2024) further refines relevance
scoring using transformer-based attention mecha-
nisms. Dense Passage Retrieval (DPR) (Karpukhin
et al., 2020) optimizes dense embeddings using
contrastive learning, while G-Retriever (He et al.,
2024) constructs subgraphs and applies a Steiner
tree strategy to compute quantifiable relevance
values, integrating these metrics into LLMs via
fine-tuning. These developments raise a key ques-
tion: as the user is the ultimate arbiter of rele-
vance, should user behavior data be incorporated
into model training? Recent studies explore this
idea, with (Au et al., 2025; Zerhoudi and Gran-
itzer, 2024) leveraging user profiles for re-ranking
and (Bai et al., 2024) using feedback signals (e.g.,

dislikes, regenerations) to refine relevance scores.
Symbiotic interaction Symbiosis, a biological
concept describing long-term mutual interaction
between species, has inspired human-machine col-
laboration since at least 1960, when J.C.R. Lick-
lider introduced human-computer symbiosis in
“Man-Computer Symbiosis” (Licklider, 1960). He
proposed two key goals: integrating computers
into early problem-solving stages for collabora-
tive question formulation and enabling real-time
human-computer interaction for immediate feed-
back and iteration. Recent advancements in LLMs
have made real-time interaction more feasible. For
example, OpenAI’s GPT-4o achieves a latency of
approximately 196 milliseconds per token3, ap-
proaching the threshold of human real-time per-
ception. However, the challenge remains in design-
ing effective human-machine collaboration mecha-
nisms that guide users from “unconscious incom-
petence” to “conscious incompetence” in question
formulation. Despite growing recognition of symbi-
otic human-machine systems as an important direc-
tion (Mackay, 2024; Kulkarni et al., 2023; Islami
and Mulolli, 2024; Calvano, 2024; Lin, 2024; Ab-
bass, 2024), practical implementations remain in
early stages. In document intelligence, Symbiotic
Recommendations (Petruzzelli, 2024) injects user
profiles into prompts via prompt engineering, simi-
lar to personalization techniques in RAG (Au et al.,
2025; Zerhoudi and Granitzer, 2024; Bai et al.,
2024). However, deeper, bidirectional adaptations
between humans and machines are still largely un-
explored.
To bridge this gap, we propose SymbioticRAG ,
which enables users to directly select retrieved con-
tent and later incorporates user interactions into
an end-to-end Relevance model. By continuously
learning from user behaviors, the system aims to
establish a positive feedback loop, evolving into a
personalized, adaptive agent that tailors retrieval
and generation to individual needs. This is the vi-
sion behind SymbioticRAG , which marks a step
toward true human-machine symbiosis.
3 System design
The fundamental design philosophy of RAG places
humans at the center. 1). Instead of approximating
relevance metrics to model user intent, we advo-
cate restoring decision-making authority to humans.
3https://openai.com/index/hello-gpt-4o/
Foundation: Source Documents AccessLevel 1:  Retriever Suggested, Human DecideLevel 2: Learning from Human Interactions
SymbioticRAGFigure 3: SymbioticRAG concept illustration
2). Human-selected content for each query then
serves as training data, enabling the system to adapt
to user intentions through explicit selection and in-
teraction patterns. 3). To bridge the gap between
“Unconscious Incompetence” and “Conscious In-
competence” in query formulation, we propose
maintaining shared access to source documents for
both humans and LLMs. This allows users to navi-
gate the document space independently, enhancing
their understanding and awareness of available in-
formation. Through this approach, we establish a
symbiotic relationship where machines gradually
refine their understanding of human intent, while
humans gain diverse perspectives, fostering collab-
orative exploration of complex information seeking
tasks.
We define three key characteristics of a Symbi-
oticRAG system, which is illustrated in Figure 3:
(1) The foundational feature ensures direct docu-
ment access and readability, allowing humans to in-
dependently explore source content. (2) Level 1 es-
tablishes user-driven retrieval, where humans, with
reference to machine-retrieved content, actively se-
lect relevant information to augment prompts for
answer generation. (3) Level 2 continuously re-
fines retrieval models by learning from human in-
teractions and selected content, enabling personal-
ized content retrieval. Most existing RAG systems,
particularly those handling large document collec-
tions, lack direct human access and content selec-
tion mechanisms—both essential for achieving true
symbiosis. This paper focuses on implementing the
foundational feature and Level 1, experimenting
and laying the groundwork for future Level 2 ad-
vancement.
As shown in Figure 1, our system consists of
three main components: 1) Document Processing :
This pipeline standardizes document formats into
PDF, detects layout blocks with bounding boxes,
and applies OCR, table extraction, formula recog-

nition, and figure understanding to convert doc-
uments into LLM-digestible text. 2) Retriever :
This extensible module supports multiple retrieval
strategies, including semantic search. It embeds
and indexes layout blocks in a vector database, re-
trieving relevant blocks to augment LLM prompts.
3)SymbioticRAG UI : Users interact with retrieved
content by exploring matched layout blocks within
source documents, understanding their context, ex-
ploring through the source documents and manu-
ally selecting blocks for further conversations or
answer regeneration.
3.1 Document processing
Despite the variety of document formats (e.g.,
Word, Excel, PDF, images), documents fall into
two fundamental categories (Sun et al., 2024): dig-
itally native andimage-based . Digitally native
documents contain machine-readable text that can
be reliably extracted via rule-based conversion,
such as plain text files and exported PDFs. In con-
trast, image-based documents, including historical
manuscripts and photographs, pose significant ex-
traction challenges.
Recent research on image-based document pro-
cessing begins with layout detection (Zhao et al.,
2024b), identifying and classifying layout blocks
(e.g., tables, formulas, figures, titles, content
blocks). These are then processed by specialized
models: OCR (Du et al., 2020) for text, dedicated
models for table extraction, formula recognition,
and figure understanding. While OCR and basic ta-
ble/formula conversion have improved, complex ta-
ble extraction, figure understanding, and handwrit-
ten formula recognition remain difficult (Du et al.,
2020; Xia et al., 2024; Truong et al., 2024). Popular
open-source systems such as MinerU (Wang et al.,
2024a) and DocLing (Team, 2024) employ dual-
path processing to effectively convert PDFs into
markdown or JSON. However, these transforma-
tions merge and re-segment layout blocks, losing
precise positional information relative to the source
documents, which is essential for our system de-
sign.
To address this, we implement a unified docu-
ment processing pipeline that standardizes all input
documents to PDF format before applying image-
based techniques. This approach simplifies pro-
cessing while preserving precise source attribution
through layout block bounding boxes. Our design
emphasizes simplicity, robustness, and extensibil-
ity, prioritizing accurate document source tracking.Layout Detection. Our pipeline starts with
DocLayout-YOLO (Zhao et al., 2024b), identify-
ing bounding boxes and semantic classes (titles,
content, tables, figures, etc.). Detected blocks are
then processed by specialized modules:
OCR. We employ PaddleOCR4for its robust per-
formance, multi-language support, and stability.
Table. The optimal output format for table extrac-
tion in RAG systems remains open. We explore
three approaches: StructEqTable (Xia et al., 2024)
for LaTeX generation, Pix2Text (Smock et al.,
2023) for HTML, and a visual LLM method that
produces structured JSON directly from table im-
ages to describe the table content.
Formula. Although Pix2Text (Dadure et al., 2024)
reliably extracts formulas as LaTeX, mathemati-
cal expressions often contain domain-specific no-
tations and complex semantics. To enhance down-
stream applications, we augment the LaTeX out-
put with semantic descriptions generated by visual
LLMs (e.g., llama3.2-vision5).
Figure. Figure understanding is domain-dependent.
While visual language models can interpret general-
purpose illustrations, specialized models perform
better in fields like medicine (Dhote et al., 2023)
and science (Shi et al., 2024). However, effective-
ness varies with domain complexity. As a first
step, we use visual LLMs to produce descriptive
text summaries, similar to our formula approach,
while acknowledging that more domain-specific
solutions will be required. Human on the loop
validation. The immaturity of certain document
processing techniques—especially for figures and
formulas—poses significant challenges in maintain-
ing high-quality data from diverse document for-
mats for downstream retrieval tasks. To address this
while advancing specialized methods (e.g., table
extraction, figure and formula understanding), we
introduce a human-on-the-loop validation interface
that supports human review at each processing step.
Unlike traditional manual annotation approaches,
our system focuses on validating model-generated
outputs, balancing reduced human effort with im-
proved data quality and benefiting iterative model
refinement.
Our approach is especially beneficial in domains
where high-quality training data is scarce, as it posi-
tions human reviewers as both overseers and guides
4https://paddlepaddle.github.io/PaddleOCR/
latest/en/index.html
5https://huggingface.co/meta-llama/Llama-3.
2-11B-Vision

(a) Document processing pipeline
dashboard for file upload, processing
initiation and progress monitoring.
(b) Layout validation interface where
users can review and edit detected lay-
out block, including block reclassifi-
cation, addition, removal and bound-
ary adjustments.
(c) OCR validation interface for re-
viewing and correcting text recogni-
tion results, particularly useful for
handwritten text where OCR models
struggle.
(d) Table validation interface that
presents extracted table outputs in
JSON viewer, allowing users to re-
view, correct or add extra contents.
The interface also supports batch re-
view to increase efficiency.
(e) Figure validation interface where
users can verify, modify and add fig-
ure descriptions. Users can also re-
view and update figure captions and
types.
(f) Formula validation interface for
reviewing and correcting mathemat-
ical formula extraction results in la-
tex format and descriptive informa-
tion about the formula.
Figure 4: Human-on-the-loop validation interfaces for the document processing pipeline, supporting comprehensive
validation of layout analysis, OCR, table extraction, figure processing and mathematical formula recognition and
understanding.
of the pipeline. Consequently, it yields high-quality
structured data for downstream tasks and fosters
model development in underrepresented fields, fur-
ther highlighting its human-centered nature. Fig-
ure 4 illustrates the validation system.
3.2 Retriever
Although new, more effective retrieval methods
are emerging, our focus is on improving retrieval
through end-user behavior. To accommodate fu-
ture advances, we designed an extensible retrieval
module that can easily integrate newer retrievers.
For initial testing, we implemented two baseline
methods: NaïveRAG , which performs semantic
similarity search by embedding each layout block
(including tables, formulas, and figures) using the
E5 model (Wang et al., 2024b) and retrieving the
top-kblocks; and LabelNaïveRAG , which retrieves
the top- kblocks separately for each block type and
then merges the results for overall top- klayout
blocks.3.3 SymbioticRAG UI
The SymbioticRAG user interface (UI) consists
of three primary components, as illustrated in Fig-
ure 5. The left side PDF Viewer displays annotated
source documents and enables select anddeselect
interactions, while the middle Chat component fa-
cilitates multi-turn conversations for query submis-
sion, retrieval results exploration, and AI-generated
responses review. The third component, a Staging
area, maintains a collection of human-selected rel-
evant layout blocks.
Following document processing, the conversa-
tion starts with user queries (highlighted in blue).
A defined retriever identifies the top-k matched lay-
out blocks (currently k=5), presenting the searched
results as table format in the chat interface (high-
lighted in purple). Meanwhile, the augmented
query is processed by an LLM (currently GPT-4o)
to generate a response (displayed in green).
Users can navigate to the source document by
clicking any search result, which opens the selected
layout block in its original context. For example, in

Generated 
ResponseRetrieved 
ContentsQueryStage area for
 human selected contents 
Click and review content 
within source document
Human selected 
relevant layout blockRetrieval Mode
Auto generated metrics
User interaction logs & intention summary via LLM
User ReviewReport 
GenerationAdd 
New ChatChat
HistoryAll 
Documents
Copy Regenerate Like DislikeFigure 5: SymbioticRAG UI demonstration example
the demo figure, selecting the first result jumps to
page 14 (in Fig 5), highlighting the relevant block
with a purple dashed rectangle in the PDF Viewer.
Users can explore surrounding content, assess rele-
vance, and toggle block selection—selected blocks
are marked with a yellow double solid line, pre-
sented and grouped by source document in the
staging area. They can review multiple Retriever -
suggested blocks across documents to refine un-
derstanding before follow-up queries. Clicking the
regeneration button incorporates human-selected
blocks into augmented prompts for updated re-
sponses. To mitigate cases where the real relevant
documents are not present in the retrieved layout
blocks, users can access all documents via the All
tab. The interface also includes chat management
features to start new conversations and review chat
histories.
During user interactions, we record engage-
ment activities such as sending queries, clicking
search results, selecting/deselecting blocks, navi-
gating pages, manually adding documents, and lik-
ing/disliking or regenerating responses. These logs
will inform our development of SymbioticRAG
Level 2 . We currently feed them into an LLM
to generate a user-intention summary (in Fig 5),
which is then concatenated with the query for se-
mantic similarity search. We label this retriever
strategy: SymbioticRAG , experimenting how fu-
tureSymbioticRAG Level 2 can integrate user
feedback. We also tried to concatenate user logs di-
rectly with query and then generate a query embed-
ding, however, due to the raw text match from the
content, the semantic similarity retrieved content
will converge to specific contents which alreadyexists inside the user logs. Due to this limitation,
we excluded this approach from our comparative
analysis.
Report generation Report writing often de-
mands substantial time to gather and organize evi-
dence from multiple sources. Our system addresses
this challenge via a staging area that allows users
to collect and verify human-selected layout blocks
through interactive conversations. As shown in
Figure 6, we provide a dedicated report generation
interface that leverages these curated blocks. Users
can outline their report structure, drag and drop
relevant blocks into specific sections, and supply
writing instructions for each component. The sys-
tem then employs an LLM to generate a draft, with
options for direct editing and exporting to Word
format. This approach streamlines the process of
collecting, verifying, and organizing evidence from
multiple sources, addressing a key gap in tradi-
tional RAG systems and enhancing evidence-based
report writing. High resolution of Figure 5 and
Figure 6 will be included in the supplementary ma-
terials.
Drag and DropDrag and ReorderAdd instruction about how to 
write the report in high level
Prompt about how should we 
organise and write this section
Edit Section Title
Delete the SectionAdd Section Generate Reports via LLM Save Content Edit Mode Back to Chat Chat History
Stage area for
 human selected contents 
Remove from 
this Section
Figure 6: Report generation interface example for Sym-
bioticRAG

4 Evaluation and Results
We employ both quantitative metrics and qualita-
tive feedback to comprehensively assess our Level
1 implementation and Level 2 experimental explo-
ration of the proposed SymbioticRAG system.
Evaluation Scenarios We tested three distinct
scenarios: (i) a literature review scenario involving
30 academic papers related to RAG for systematic
reviews, (ii) a geological exploration scenario an-
alyzing 30 zinc-focused reports reflecting typical
tasks in geological departments, and (iii) an educa-
tionscenario incorporating “Full Stack Developer”
Unit materials to support student learning.
Participants Three independent evaluators were
assigned to each scenario, each conducting five con-
versation sessions per retrieval strategy ( NaïveRAG ,
LabelNaïveRAG , and SymbioticRAG ), followed
by quantitative satisfaction ratings and post-
experiment interviews. For the literature review
scenario, the evaluators were three computer sci-
ence researchers; for the geological report scenario,
three experienced geologists; and for the educa-
tionscenario, three undergraduate students. Iden-
tical questions were asked across retrieval strate-
gies, with evaluators randomly alternating between
strategies across sessions to mitigate learning bias,
though this cannot be fully eliminated.
Evaluation metrics We used two outcome-
oriented metrics to assess system effectiveness.
The first metric captures the layout blocks selected
by users, representing the “true relevant” content
for each conversation. The second is user satis-
faction, rated on a 5-point scale (with 5 indicating
“very satisfied”). To quantify alignment between
user and retriever selections, we define the human-
retriever distance Das:
D= 1−|H∩R|
|H∪R|, (1)
where HandRare the sets of blocks selected by
humans and the retriever, respectively. This metric
ranges from 0 (perfect alignment) to 1 (complete
divergence). We also compute a satisfaction metric
Sas the mean user satisfaction rating, reflecting
overall usability and effectiveness.Table 1: Comparative Evaluation Results Across Differ-
ent Scenarios
Strategy Metric Literature Review Geological Reports Education
NaïveRAGHuman-Retriever Distance ( D, 0-1) ↓ 0.85 0.92 0.88
User Satisfaction ( S, 1-5) ↑ 2.47 2.13 1.80
LabelNaïveRAGHuman-Retriever Distance ( D, 0-1) ↓ 0.78 0.83 0.81
User Satisfaction ( S, 1-5) ↑ 3.13 2.93 2.67
SymbioticRAGHuman-Retriever Distance ( D, 0-1) ↓ 0.52 0.61 0.58
User Satisfaction ( S, 1-5) ↑ 4.13 3.93 3.67
Results Table 1 presents the quantitative re-
sults of our evaluation across three scenarios.
The Human-Retriever Distance metric ( D) re-
sults reveal a significant challenge in aligning re-
triever outputs with human information needs, with
NaïveRAG showing consistently high distances
(0.85-0.92). This supports our hypothesis that users
in unknown-unknown states often struggle to artic-
ulate their precise information needs. The introduc-
tion of LabelNaïveRAG showed modest improve-
ments, reducing distances to 0.78-0.83 through in-
creased retrieval diversity. Most notably, Symbiot-
icRAG demonstrated substantial gains, achieving
distances of 0.52-0.61, suggesting that augmented
queries with user interaction summary better cap-
ture user semantic intent. User satisfaction scores
correlated with reduced retriever distances, with
SymbioticRAG achieving the highest ratings (3.67-
4.13). Literature review and geological scenarios
showed marginally better performance, likely due
to participants’ domain expertise. Post-experiment
interviews revealed two critical features driving
user satisfaction: the ability to examine source doc-
uments was described as “game-changing,” while
the capacity to select and incorporate layout blocks
into prompts addressed a key limitation in existing
LLM interfaces.
Case study We analyzed a user’s conversation (in
Figure 7) in the education scenario using Symbi-
oticRAG retriever. The interaction history reveals
three phases: from initial “Unconscious Incompe-
tence” with basic queries, through a “Transition”
phase with emerging awareness, to “Conscious
Incompetence” where users demonstrate proper
technical vocabulary. This progression validates
that our SymbioticRAG design effectively positions
users at the center, supporting their natural devel-
opment of domain understanding.
5 Conclusion
We presented SymbioticRAG , a novel frame-
work that fundamentally reimagines RAG systems
through a human-centered lens. Our work ad-
dresses two critical challenges in current RAG sys-

SymbioticRAG User Document
Phase 1: Unconscious Incompetence
Transition"How to put my website online?"
Phase 2: Conscious Incompetence- No awareness of technical sccope
- Lacks Domain vocabularyBasic web deployment search
General deployment overview
Read documentation
- Begins to recognize knowledge gaps
- questions technical concepts
"So this is about frontend...
What exactly is a frontend app?"
- Analysze interaction history
- Identiﬁes knowledge gaps
- Plans structured guidance
Fetches frontend architecture & details
Explain frontend stack & concepts
Read Full stack architecture overview to understand the concepts
- Recognizes learning sceop
- Use technical vocabulary
- Seeks speciﬁc knowledge
"I need to learn React before deployment?"
- Understand user progress and intention
- Provide focused guidance
Extract relevant slides and documents
Provide clear learning instruction
Read  details about the React, then develop further queiresFigure 7: Case Study in education scenario (Full Stack
Developer Unit) with SymbioticRAG retriever.
tems: the inherently human nature of relevance de-
termination and users’ struggle with Unconscious
Incompetence when formulating queries in unfa-
miliar domains. Our framework introduces a two-
tiered approach: Level 1 enables direct human cu-
ration of retrieval content through interactive docu-
ment access, while Level 2 aims to develop person-
alized retrieval models based on user interactions.
We successfully implemented Level 1 with three
key components: a comprehensive document pro-
cessing pipeline, an extensible retriever module,
and an interactive UI that facilitates user engage-
ment while collecting valuable interaction data. To
maintain high-quality data preparation, we devel-
oped a human-on-the-loop validation interface that
improves pipeline output while advancing research
in specialized extraction tasks.
Our evaluation across three distinct scenarios
demonstrated the effectiveness of our approach.
More importantly, our attempt at Level 2 im-
plementation through SymbioticRAG , which aug-
ments queries with summaries of user interactions,
showed promising results with significantly im-
proved retrieval relevance and user satisfaction
scores. To facilitate broader research and further
advancement of SymbioticRAG Level 2, we will
make our system openly and freely accessible to
the research community upon paper acceptation.
References
Hussein Abbass. 2024. Future directions in artificial in-
telligence research. IEEE Transactions on Artificial
Intelligence , 5(12):5858–5862.
Steven Au, Cameron J. Dimacali, Ojasmitha Pedirappa-
gari, Namyong Park, Franck Dernoncourt, Yu Wang,Nikos Kanakaris, Hanieh Deilamsalehy, Ryan A.
Rossi, and Nesreen K. Ahmed. 2025. Personal-
ized graph-based retrieval for large language models.
Preprint , arXiv:2501.02157.
Yu Bai, Yukai Miao, Li Chen, Dawei Wang, Dan Li,
Yanyu Ren, Hongtao Xie, Ce Yang, and Xuhui Cai.
2024. Pistis-rag: Enhancing retrieval-augmented
generation with human feedback. Preprint ,
arXiv:2407.00072.
Carol L. Barry. 1994. User-defined relevance criteria:
An exploratory study. Journal of the American Soci-
ety for Information Science , 45(3):149–159.
Maciej Besta, Ales Kubicek, Roman Niggli, Robert
Gerstenberger, Lucas Weitzendorf, Mingyuan Chi,
Patrick Iff, Joanna Gajda, Piotr Nyczyk, Jürgen
Müller, Hubert Niewiadomski, Marcin Chrapek,
Michał Podstawski, and Torsten Hoefler. 2024. Multi-
head rag: Solving multi-aspect problems with llms.
Preprint , arXiv:2406.05085.
Miriana Calvano. 2024. Design and evaluation of
high-quality symbiotic ai systems through a human-
centered approach. In Proceedings of the 28th Inter-
national Conference on Evaluation and Assessment
in Software Engineering , EASE ’24, page 488–493,
New York, NY , USA. Association for Computing
Machinery.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, and Jimmy Lin. 2022. Overview of the trec
2021 deep learning track. In Text REtrieval Confer-
ence (TREC) . NIST, TREC.
Pankaj Dadure, Partha Pakray, and Sivaji Bandyopad-
hyay. 2024. Mathematical information retrieval: A
review. ACM Comput. Surv. , 57(3).
Anurag Dhote, Mohammed Javed, and David S Do-
ermann. 2023. A survey on figure classifica-
tion techniques in scientific documents. Preprint ,
arXiv:2307.05694.
Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Wei-
wei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang,
Qingqing Dang, and Haoshuang Wang. 2020. Pp-ocr:
A practical ultra lightweight ocr system. Preprint ,
arXiv:2009.09941.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
Preprint , arXiv:2404.16130.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. Preprint , arXiv:2405.14831.
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Ji-
ayuan Ding, Yongjia Lei, Mahantesh Halappanavar,
Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng
Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao,

Neil Shah, Amin Javari, Yinglong Xia, and Jiliang
Tang. 2025. Retrieval-augmented generation with
graphs (graphrag). Preprint , arXiv:2501.00309.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. In The Thirty-eighth Annual Confer-
ence on Neural Information Processing Systems .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2024. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions on Information
Systems .
Xhavit Islami and Enis Mulolli. 2024. Human-artificial
intelligence in management functions: A synergis-
tic symbiosis relationship. Applied Artificial Intelli-
gence , 38(1):2439615.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation. ACM Comput.
Surv. , 55(12).
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen tau Yih. 2020. Dense passage retrieval
for open-domain question answering. Preprint ,
arXiv:2004.04906.
Vinay Kulkarni, Sreedhar Reddy, Souvik Barat, and
Jaya Dutta. 2023. Toward a symbiotic approach lever-
aging generative ai for model driven engineering. In
2023 ACM/IEEE 26th International Conference on
Model Driven Engineering Languages and Systems
(MODELS) , pages 184–193.
J. C. R. Licklider. 1960. Man-computer symbiosis. IRE
Transactions on Human Factors in Electronics , HFE-
1(1):4–11.
Zhicheng Lin. 2024. Progress and challenges in the
symbiosis of ai with science and medicine. Euro-
pean Journal of Clinical Investigation , pages e14222–
e14222.
Wendy E. Mackay. 2024. Parasitic or symbiotic? re-
defining our relationship with intelligent systems. In
Adjunct Proceedings of the 37th Annual ACM Sym-
posium on User Interface Software and Technology ,
UIST Adjunct ’24, New York, NY , USA. Association
for Computing Machinery.
Christopher D. Manning, Prabhakar Raghavan, and Hin-
rich Schütze. 2008. Introduction to Information Re-
trieval . Cambridge University Press.
Alessandro Petruzzelli. 2024. Towards symbiotic rec-
ommendations: Leveraging llms for conversational
recommendation systems. In Proceedings of the 18thACM Conference on Recommender Systems , RecSys
’24, page 1361–1367, New York, NY , USA. Associa-
tion for Computing Machinery.
Tefko Saracevic. 2019. The notion of relevance in infor-
mation science .
Xiang Shi, Jiawei Liu, Yinpeng Liu, Qikai Cheng, and
Wei Lu. 2024. Every part matters: Integrity verifica-
tion of scientific figures based on multimodal large
language models. Preprint , arXiv:2407.18626.
Brandon Smock, Rohith Pesala, and Robin Abraham.
2023. Aligning benchmark datasets for table struc-
ture recognition. pages 371–386.
Qiang Sun, Yuanyi Luo, Wenxiao Zhang, Sirui Li,
Jichunyang Li, Kai Niu, Xiangrui Kong, and Wei
Liu. 2024. Docs2kg: Unified knowledge graph con-
struction from heterogeneous documents assisted by
large language models. Preprint , arXiv:2406.02962.
Deep Search Team. 2024. Docling technical report.
Technical report.
Thanh-Nghia Truong, Cuong Tuan Nguyen, Richard
Zanibbi, Harold Mouchère, and Masaki Nakagawa.
2024. A survey on handwritten mathematical expres-
sion recognition: The rise of encoder-decoder and
gnn models. Pattern Recognition , 153:110531.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui,
Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Con-
ghui He. 2024a. Mineru: An open-source solution
for precise document content extraction. Preprint ,
arXiv:2409.18839.
Liang Wang, Nan Yang, Xiaolong Huang, Binx-
ing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. 2024b. Text embeddings by
weakly-supervised contrastive pre-training. Preprint ,
arXiv:2212.03533.
Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou,
Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng
Fu, Wenjie Wu, Hancheng Ye, and 1 others. 2024.
Docgenome: An open large-scale scientific document
benchmark for training and testing multi-modal large
language models. arXiv preprint arXiv:2406.11633 .
Saber Zerhoudi and Michael Granitzer. 2024. Per-
sonarag: Enhancing retrieval-augmented genera-
tion systems with user-centric agents. Preprint ,
arXiv:2407.09394.
Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He,
Luna K. Qiu, and Lili Qiu. 2024a. Retrieval aug-
mented generation (rag) and beyond: A comprehen-
sive survey on how to make your llms use external
data more wisely. Preprint , arXiv:2409.14924.
Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Con-
ghui He. 2024b. Doclayout-yolo: Enhancing doc-
ument layout analysis through diverse synthetic data

and global-to-local adaptive perception. Preprint ,
arXiv:2410.12628.
Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu,
Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng
Liu, Zhicheng Dou, and Ji-Rong Wen. 2023. Large
language models for information retrieval: A survey.
arXiv preprint arXiv:2308.07107 .
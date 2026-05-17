# Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education

**Authors**: Mragisha Jain, Tirth Bhatt, Griffin Pitts, Aum Pandya, Peter Brusilovsky, Narges Norouzi, Arto Hellas, Juho Leinonen, Bita Akram

**Published**: 2026-05-13 04:37:45

**PDF URL**: [https://arxiv.org/pdf/2605.12988v1](https://arxiv.org/pdf/2605.12988v1)

## Abstract
Students learning algorithms often need support as they interpret traces, debug reasoning errors, and apply procedures across unfamiliar problem instances. In this paper, we present KITE (Knowledge-Informed Tutoring Engine), a Retrieval-Augmented Generation (RAG)-based intelligent tutoring system designed to serve as a classroom teaching assistant for algorithmic reasoning and problem-solving tasks. KITE uses an intent-aware Socratic response strategy to tailor support to different student needs, responding with targeted hints, guiding questions, and progressive scaffolding intended to strengthen students' algorithmic problem-solving ability. To keep responses aligned with course content, KITE uses a multimodal RAG pipeline that retrieves relevant information from course materials. We evaluate KITE using three forms of assessment: RAGAs-based metrics for response grounding and quality, expert evaluation of pedagogical quality, and a simulated student pipeline in which a weaker language model interacts with KITE across two-turn dialogues and produces revised answers after receiving feedback. Results indicate that KITE produces contextually grounded and pedagogically appropriate responses. Further, using simulated students, KITE's feedback helped the student models produce more accurate follow-up responses on procedural and tracing questions, suggesting that its scaffolding can support algorithmic problem-solving. This work contributes a tutoring architecture and an evaluation approach for assessing retrieval-grounded explanations and scaffolded problem-solving feedback.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving
in AI Education
Mragisha Jain1, Tirth Bhatt1, Griffin Pitts1, Aum Pandya1,
Peter Brusilovsky2,Narges Norouzi3,Arto Hellas4,Juho Leinonen4,Bita Akram1*
1North Carolina State University,2University of Pittsburgh,
3University of California, Berkeley,4Aalto University,
*Correspondence:bakram@ncsu.edu
Abstract
Students learning algorithms often need sup-
port as they interpret traces, debug reason-
ing errors, and apply procedures across unfa-
miliar problem instances. In this paper, we
present KITE (Knowledge-Informed Tutoring
Engine), a Retrieval-Augmented Generation
(RAG)-based intelligent tutoring system de-
signed to serve as a classroom teaching as-
sistant for algorithmic reasoning and problem-
solving tasks. KITE uses an intent-aware So-
cratic response strategy to tailor support to dif-
ferent student needs, responding with targeted
hints, guiding questions, and progressive scaf-
folding intended to strengthen students’ algo-
rithmic problem-solving ability. To keep re-
sponses aligned with course content, KITE uses
a multimodal RAG pipeline that retrieves rel-
evant information from course materials. We
evaluate KITE using three forms of assessment:
RAGAs-based metrics for response ground-
ing and quality, expert evaluation of pedagog-
ical quality, and a simulated student pipeline
in which a weaker language model interacts
with KITE across two-turn dialogues and pro-
duces revised answers after receiving feedback.
Results indicate that KITE produces contextu-
ally grounded and pedagogically appropriate
responses. Further, using simulated students,
KITE’s feedback helped the student models
produce more accurate follow-up responses
on procedural and tracing questions, suggest-
ing that its scaffolding can support algorithmic
problem-solving. This work contributes a tutor-
ing architecture and an evaluation approach for
assessing retrieval-grounded explanations and
scaffolded problem-solving feedback.
1 Introduction
Large language models (LLMs) such as ChatGPT
are now widely used by students for learning sup-
port, including explanation, feedback, and problem-
solving (Pitts et al., 2025b; Pitts and Motamedi,
2026). Students often value these tools because
they provide immediate access to assistance wheninstructors or teaching assistants are unavailable
(Pitts et al., 2025b). Although these tools make
information more accessible, prior work raises con-
cerns that students may accept AI-generated re-
sponses without sufficient evaluation, especially
when those responses appear complete and confi-
dent (Essel et al., 2024; Pitts et al., 2025c, 2026).
In education, this can lead students to bypass the
reasoning processes that assignments are designed
to develop (Pitts et al., 2025c, 2026). These con-
cerns highlight the need for LLM-based systems
that provide timely, course-grounded information
while delivering pedagogically appropriate support
that helps students reason through learning tasks.
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) offers a promising approach for build-
ing course-grounded tutoring systems by allowing
LLM responses to draw on curated instructional
materials. This grounding can reduce unsupported
or course-inconsistent claims and help align expla-
nations with the concepts, terminology, and con-
ventions used in a course. However, strong retrieval
does not by itself ensure effective tutoring. Even
when a system retrieves relevant material, it may
still provide responses that are too direct, insuffi-
ciently instructional, or mismatched to the student’s
immediate learning need. Prior work on intelligent
tutoring systems suggests that effective support
depends on both the accuracy of the information
provided and how assistance is delivered, including
when to offer direct explanation, feedback, or more
guided support (Koedinger and Aleven, 2007).
Socratic tutoring offers one way to address this
challenge by guiding learners through targeted
questions, prompts, and progressive hints instead
of immediately providing full solutions. This ap-
proach is grounded in cognitive apprenticeship and
guided facilitation (Collins et al., 1989; Hmelo-
Silver and Barrows, 2006), and has been used in
dialogue-based tutoring systems such as AutoTu-
tor (Graesser et al., 1999). However, integrating
To appear in the Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA
2026), co-located with ACL 2026. Copyright 2026 by the author(s).arXiv:2605.12988v1  [cs.AI]  13 May 2026

Socratic guidance into retrieval-grounded tutoring
systems remains an open design problem: a course-
specific tutor must stay faithful to instructional ma-
terials while also providing feedback that fits the
type of problem the student is trying to solve.
In this paper, we presentKITE(Knowledge-
Informed Tutoring Engine), a RAG-based intelli-
gent tutoring system that connects students to rel-
evant course materials while using intent-aware
tutoring strategies to support different forms of
help-seeking. KITE uses a multi-stage multimodal
retrieval pipeline to locate relevant instructional
content and an intent-aware response strategy to
determine how that content should be used in the
response. For questions that require direct expla-
nation, KITE provides responses aligned with re-
trieved course materials. For procedural, debug-
ging, validation, and tracing questions, KITE pro-
vides targeted feedback, guiding questions, and
progressive hints to support student reasoning.
To evaluate KITE, we first assess its retrieval-
grounded outputs for non-procedural questions
using RAGAs-based metrics for grounding, rel-
evance, and response quality. We then evaluate
procedural and tracing questions through a simu-
lated student pipeline in which a weaker language
model revises its answers after receiving KITE’s
feedback. Finally, human experts assess the re-
sulting interactions to judge feedback quality and
whether the revised answers show improvement.
This work contributes (1) KITE, an intent-aware
tutoring system that combines multimodal retrieval
with pedagogical support, and (2) an evaluation
of its retrieval-grounded responses and scaffolded
feedback using automated metrics, simulated stu-
dents, and expert evaluation. We assess two re-
search questions:RQ1:How well does KITE pro-
duce grounded, course-relevant responses for non-
procedural student questions? andRQ2:To what
extent does KITE’s feedback support improved re-
sponses on procedural and tracing questions?
2 Related Work
RAG-based educational assistants have been used
for a range of instructional purposes, including
interactive learning support, content generation,
and large-scale course deployment (Li et al., 2025).
Across these systems, grounding LLM responses
in course materials has generally improved factual
accuracy compared to unaugmented models. For
example, KAG (Hasan et al., 2025) reports Preci-sion@5 of 0.85 and a 34% reduction in student
task completion time, while MoodleBot (Neumann
et al., 2024) achieves 88% accuracy on course-
related queries. However, these systems primar-
ily function as direct question-answering tools and
do not adapt their responses to different forms of
student help-seeking.
Although RAG can improve factual accuracy, de-
ployment studies suggest that course-grounded as-
sistants also need evaluation in instructional work-
flows. In one classroom deployment, students
showed strong pre-exam engagement but declin-
ing adoption across cohorts, and 36.8% reported
frustration when responses extended beyond a con-
strained knowledge base (Thesen and Park, 2025).
Edison (Miroyan et al., 2025), a GPT-4-based RAG
assistant deployed in a large data science course,
showed that retrieving from course documents and
historical Q&A can support factual and relevant
responses to live student questions. The study also
demonstrates the value of TA-in-the-loop evalua-
tion, using instructor edits and ratings to assess fac-
tuality, relevance, style, and efficiency. EduMod-
LLM (Mittal et al., 2026) extends this line of work
by treating educational Q&A as a modular pipeline,
separating function calling, retrieval, and response
generation so that system behavior can be evaluated
more transparently.
Dialogue-based tutoring provides another foun-
dation for supporting student reasoning. AutoTu-
tor (Graesser et al., 1999) showed that progressive
hints and collaborative answer refinement produced
dialogues rated above the “good” threshold by do-
main experts, with semantic evaluation correlat-
ing at 0.49 with expert judgment. More recently,
(Li et al., 2026) reported significant gains in self-
efficacy ( d= 0.57 ) from a Socratic AI platform in
healthcare education. LeanTutor (Patel et al., 2026)
similarly emphasizes guided feedback by combin-
ing LLMs with a theorem prover to check student
proofs, identify errors, and provide hints toward a
correct proof without giving away the complete an-
swer. These systems show the value of scaffolded
feedback for learning tasks that require students to
reason through a process, yet they do not incorpo-
rate retrieval grounding to keep responses aligned
with course-specific materials.
Other systems explore how retrieval and re-
sponse strategies can be adapted for learning con-
texts. LPITutor (Liu et al., 2025) supports adaptive
difficulty modulation through RAG and prompt en-
gineering. KG-RAG (Dong et al., 2025) combines

Figure 1: KITE architecture
semantic retrieval with an expert-validated knowl-
edge graph and reports a 35% improvement in
learning outcomes ( d= 0.86 ) in a study of 76 stu-
dents, though its reliance on manual expert valida-
tion limits scalability. AutoTA (Dahal et al., 2025)
provides a related approach to intent-aware edu-
cational assistance by classifying student queries
and routing them to specialized response strategies.
These systems show the value of adapting retrieval,
domain structure, and response behavior to differ-
ent learning needs. KITE builds on this direction
by pairing multi-stage retrieval with intent-aware
tutoring strategies for algorithmic reasoning tasks.
3 System Design
KITE is a retrieval-augmented tutoring system de-
signed to support course-grounded dialogue for
algorithmic reasoning and problem-solving tasks.
As shown in Figure 1, the system includes five
components: document preprocessing, embedding
generation, multi-stage retrieval, intent-aware re-
sponse generation, and session management.
3.1 Phase 1: Document Ingestion and
Preprocessing
KITE begins by extracting text from course PDFs
usingPyMuPDF. Extraction is performed page bypage so that the original document structure re-
mains traceable during retrieval. To reduce noise
before indexing, the system applies afrequency-
basedcleaning procedure that removes repeated
headers, footers, page numbers, and other format-
ting artifacts. Specifically, it examines the first
and last two lines of each page, identifies repeated
patterns that occur across pages, removes those
patterns along with page numbers and special char-
acters, and normalizes whitespace.
The cleaned text is then segmented into semanti-
cally coherent chunks for retrieval. We usesection-
aware chunkingwith a target size of 500 char-
acters, or about 125 tokens, and a 100-character
overlap. Headers are identified and retained to pre-
serve local structure, while overlap carries forward
the final two sentences of the preceding chunk.
3.2 Phase 2: Embedding Generation
Each chunk is encoded using OpenAI’s
text-embedding-3-large model, produc-
ing 3072-dimensional embeddings. These vectors
are L2-normalized so cosine similarity reflects
semantic direction and are stored in a FAISS index
(Johnson et al., 2019) for efficient local retrieval.

3.3 Phase 3: Multi-Stage Retrieval Pipeline
KITE uses a multi-stage pipeline designed to bal-
ance high recall and precision in retrieving course
content. Retrieval begins with a dense bi-encoder
search that returns the top 50 candidate chunks for
a given student query. The query and document
chunks are encoded independently, and similarity
is computed using cosine similarity, allowing the
system to capture semantically related content.
The candidate set is then refined through hybrid
retrieval. Dense similarity contributes 70% of the
retrieval score, while sparse BM25 keyword match-
ing contributes 30%. This combination captures
both semantic similarity and exact lexical overlap,
which is useful when students use course-specific
terminology, notation, or algorithm names.
To reduce redundancy among retrieved pas-
sages, KITE applies Maximal Marginal Relevance
(MMR) withλset to 0.7:
MMR=λ×Relevance+ (1−λ)×Diversity
The retrieved candidates are reranked using
a cross-encoder/ms-marco-MiniLM-L-6-v2
reranking model implemented through Sentence
Transformers, where the query and document are
jointly encoded to produce more precise relevance
scores. Finally, KITE applies source-based boost-
ing so that chunks from official course materials
receive higher priority. Chunks with reranking
scores above 0.6 receive an additional boost of 0.3.
The final context passed to the generator consists
of the top eight retrieved chunks.
3.4 Phase 4: Intent Classification and
Pedagogical Response Generation
KITE does not use a single response strategy for
all student questions. Instead, it first classifies each
query by pedagogical intent and then generates a
response that matches the instructional purpose of
the interaction. This allows the system to distin-
guish among questions, debugging requests, and
other forms of help-seeking.
3.4.1 Intent Classification
Each incoming query is classified into one of five
pedagogical intents using a keyword and pattern-
matching classifier, as shown in Figure 1.
•Direct Question: factual queries seeking defi-
nitions or explanations (e.g., “What is A*?”).•Conceptual Questions: deeperwhyorhow
questions probing understanding (e.g., “Why
does BFS guarantee shortest path?”).
•Algorithm Validation: queries where a stu-
dent submits their own implementation or
trace for assessment.
•Debugging: queries involving a specific error
or incorrect output.
•Algorithm Tracing: requests to step through
the execution of an algorithm on a concrete
problem instance (e.g., “Trace A* on this
graph starting from node S”).
The classified intent determines which response
generation strategy is invoked. In addition to these
five query intents, KITE includes a dedicated an-
swer evaluation mode for cases in which a stu-
dent submits a written answer for assessment. This
mode bypasses intent classification and routes di-
rectly to the feedback generation pipeline.
3.4.2 LLM Generation and Intent-Aware
Response Strategy
All response generation in KITE is handled by GPT-
5. Outputs are grounded in a structured prompt
that injects the top eight retrieved chunks into a
[CONTEXT] block. The model is instructed to prior-
itize course materials and avoid introducing infor-
mation that is unsupported by the retrieved context,
helping keep responses aligned with the course.
For direct questions and conceptual doubts,
KITE produces explanations grounded in the re-
trieved material. Responses are written in a tutor-
like tone that emphasizes reasoning instead of brief
answer delivery. For conceptual doubts, the re-
sponse also includes a follow-up question intended
to prompt reflection.
For algorithm validation tasks, KITE adopts a
Socratic assessment strategy instead of directly
identifying errors. Responses include a brief evalu-
ation of the student’s approach, acknowledgement
of correct components, and guiding questions that
target specific issues. This design supports learn-
ing without explicitly revealing the final solution,
encouraging students to continue working through
the problem independently.
For debugging assistance, KITE generates diag-
nostic prompts that guide students toward identify-
ing errors through self-examination. Each response
follows a structured hint progression and includes
a learning point that connects the observed bug
to the underlying conceptual principle, reinforcing
understanding beyond the immediate correction.

For algorithm tracing queries, KITE retrieves
the relevant procedural steps and rules from course
materials and applies them step by step to the stu-
dent’s specific problem instance. Each step ex-
plicitly maintains and updates algorithmic state
variables such as OPEN lists, CLOSED sets, and
selected nodes, following the tie-breaking rules and
constraints defined in the query. The response con-
cludes with the final path and cost when applicable.
3.5 Phase 5: Session Management
KITE maintains session state across multi-turn in-
teractions to preserve continuity within a conversa-
tion. For each session, the system stores the origi-
nal query, detected intent, prior responses, and any
hints provided. When a student submits a follow-up
query, KITE uses this stored context to determine
how the interaction should continue: related direct
and conceptual questions are treated as follow-ups,
while validation, debugging, and tracing requests
remain within their intent-specific response strategy
when they concern the same problem or algorithm.
For these turns, KITE constructs a brief context
summary from the prior interaction and appends
it to the retrieval prompt to reduce repetition and
support progressive guidance.
4 Methodology
To evaluate KITE, we use three forms of as-
sessment. First, we examine non-procedural re-
sponses using RAGAs-based metrics for grounding,
relevance, and answer quality against instructor-
authored reference answers. We then use a sim-
ulated student pipeline to assess whether KITE’s
feedback helps produce improved responses on pro-
cedural and tracing questions. Finally, experts eval-
uate the pedagogical quality of KITE’s feedback
and the resulting answer revisions.
4.1 Evaluation Dataset
We constructed an evaluation dataset of 109 ques-
tions drawn from the lecture slides and textbook
used in a universityIntroduction to AIcourse, with
each question paired with an instructor-verified ref-
erence answer. The dataset included 42 algorithmic
questions, 51 procedural questions, and 16 direct-
retrieval questions. We applied RAGAs to the 58
non-procedural questions, consisting of the algo-
rithmic and direct-retrieval subsets, because these
responses can be evaluated against reference an-
swers for grounding, relevance, and answer quality.Questions requiring procedural reasoning or algo-
rithm tracing were evaluated separately through
the simulated student pipeline and expert review
described in Section 4.3.
4.2 RAGAs Evaluation
We evaluate KITE’s non-procedural responses us-
ing the RAGAs framework (Es et al., 2024; Roy-
chowdhury et al., 2024), reporting six metrics:
•Faithfulness: Measures whether statements
in the generated response are supported by the
retrieved context, computed as the proportion
of answer claims judged to be grounded in the
retrieved chunks.
•Answer Relevance: Measures how well the
generated response addresses the original
question, computed from the cosine similar-
ity between the user’s question and questions
generated from the response.
•Context Relevance: Measures how much of
the retrieved context is relevant to answering
the question, computed as the proportion of
retrieved statements judged to be useful.
•Answer Similarity: Measures semantic sim-
ilarity between the generated response and
the instructor-authored reference answer us-
ing sentence embeddings.
•Factual Correctness: Measures factual agree-
ment between the generated response and
the reference answer using an F1 score over
claims classified as true positives, false posi-
tives, and false negatives.
•Answer Correctness: Measures overall cor-
rectness of the generated response relative to
the reference answer as a weighted combina-
tion of factual correctness (0.75) and answer
similarity (0.25).
All metrics use gpt-4o-mini as the judge model
andtext-embedding-3-small for similarity met-
rics. Retrieval uses top_k=5 from an initial candi-
date pool of 50.
4.3 Simulated Student Evaluation and Expert
Evaluation
For procedural and algorithm-tracing questions,
KITE provides Socratic feedback and guidance,
making standard automatic scoring less appropri-
ate. To evaluate how well this feedback supports
learning-oriented revision, we use a two-stage sim-
ulated student pipeline followed by expert review.

Figure 2: Evaluation pipeline
Simulated Student Pipeline.Building on prior
work that uses simulated student-tutor interactions
to evaluate pedagogical support (Dinucu-Jianu
et al., 2025), we use Meta-Llama-3.1-70B-Instruct
as a proxy student in a structured interaction with
KITE:
1.Round 1: The student model answers each
question without assistance, establishing an
unaided baseline.
2.KITE Feedback: KITE evaluates the stu-
dent’s answer and provides feedback intended
to guide revision.
3.Round 2: The student model revises its an-
swer using KITE’s feedback.
Expert Evaluation.Three experts reviewed each
interaction set, consisting of the Round 1 answer,
KITE’s feedback, and the Round 2 answer. They
judged whether the revised response demonstrated
improved correctness and reasoning, and evaluated
the quality of KITE’s feedback using a structured
rubric adapted from prior work (Pauzi et al., 2025).
The rubric includes three dimensions.Mistake
Remediationassesses whether the tutor correctly
identifies the student’s error and explicitly acknowl-
edges it in the response.Scaffolding and Guidance
assesses whether the tutor provides appropriate sup-
port without revealing the answer and offers clear
next-step direction.Coherence and Toneassesses
whether the dialogue reads naturally and maintains
an encouraging and supportive tone. Each criterion
is scored as Yes/No, with NA used when a criterion
is not applicable.5 Results
5.1 RAGAs Evaluation
Table 1 reports the six RAGAs metrics evaluated
on the 58 non-procedural questions, consisting of
42 algorithmic questions and 16 direct-retrieval
questions.
Metric Mean Std. Dev.
Faithfulness 0.8486 0.2103
Answer Relevance 0.7558 0.2032
Context Relevance 0.9352 0.1905
Answer Similarity 0.7586 0.0923
Factual Correctness 0.4483 0.2477
Answer Correctness 0.6363 0.1810
Table 1: RAGAs evaluation summary (n= 58).
KITE performs strongly on retrieval and ground-
ing metrics. Faithfulness (0.85) indicates that most
answer statements are supported by the retrieved
context, while context relevance (0.94) shows that
the retrieved passages are highly pertinent to the
question. Answer relevance (0.76) and answer
similarity (0.76) further suggest that KITE’s re-
sponses remain on-topic and semantically aligned
with instructor-authored reference answers.
Factual correctness (0.45) is lower than the other
RAGAs measures. As discussed in Section 7, this
metric is sensitive to claim-level overlap with a
single reference answer and may understate the
quality of responses that are accurate but phrased
differently or provide additional valid detail. For
this reason, answer similarity is used as the primary
indicator of response quality in this setting. Its low

variance ( σ= 0.09 ) also suggests relatively consis-
tent performance across the evaluated questions.
5.2 Simulated Student and Expert Evaluation
Table 2 summarizes the expert rubric scores for 44
simulated student–KITE interaction triples. Inter-
rater agreement between the two expert annotators
was high, with Cohen’s κ= 0.88 and a raw agree-
ment rate of 98.15%, indicating strong consistency
in rubric judgments.
Metric % Yes % No % N/A
Mistake Remediation (Identifying) 63.63 6.82 29.55
Mistake Remediation (Acknowledging) 63.63 6.82 29.55
Scaffolding 93.18 6.82 —
Guidance 93.18 6.82 —
Coherence (Naturalness) 93.18 6.82 —
Tone (Encouraging) 93.18 6.82 —
Table 2: Expert evaluation rubric results (n= 44).
KITE receives consistently high ratings for
scaffolding, guidance, coherence, and tone, with
93.18% Yes judgments on each dimension. These
results indicate that its feedback is generally well-
structured, actionable, and supportive through-
out the interaction. Mistake remediation receives
63.63% Yes judgments, but 29.55% of cases are
marked N/A because the simulated student’s initial
response was already correct and no error identi-
fication was required. When remediation is appli-
cable, the results indicate that KITE identifies and
acknowledges student errors appropriately.
Answer Improvement.Table 3 reports how stu-
dents responses changed from Round 1 to Round 2
after receiving KITE’s feedback. The transition la-
bels reflect expert judgments of whether responses
were Incorrect, Partially Correct, or Correct with
respect to the course materials. Among the 27 in-
teractions that were not already correct, 24 showed
improvement after KITE’s feedback (88.89%).
The most common transition was from Partially
Correct to Correct, occurring in 14 cases (31.82%).
This suggests that KITE is particularly effective
at helping students resolve remaining reasoning
gaps in responses that are already moving in the
right direction. In six additional cases (13.63%),
the response remained Partially Correct but still
improved in quality, indicating that KITE’s feed-
back can support meaningful revision even when
the student model does not reach a correct answer.Transition Count %
Incorrect→Correct 1 2.27
Incorrect→Partially Correct 3 6.82
Already Correct 17 38.64
Partially Correct→Correct 14 31.82
Partially Correct→Partially Correct with Improvement 6 13.63
N/A 3 6.82
Table 3: Answer improvement breakdown (n= 44).
6 Discussion
This study examined whether a course-grounded,
intent-aware tutoring system could provide reliable
retrieval-based support and pedagogically useful
feedback for problem-solving tasks. The results
are encouraging with regard to both aims. For
RQ1, KITE’s faithfulness (0.85) and context rele-
vance (0.94) indicate that its responses are closely
grounded in retrieved course material, while an-
swer similarity (0.76) shows consistent alignment
with instructor-authored reference answers. For
RQ2, among the 27 simulated-student interactions
with KITE, in which the initial student response
was not already correct, experts judged 24 re-
vised answers (88.89%) as improved after receiv-
ing KITE’s feedback. This suggests that KITE’s
feedback provided guidance the student model
could use to correct or strengthen its reasoning
in a follow-up response. Experts rated 93.18%
of KITE’s feedback positively for scaffolding and
guidance, further indicating that the feedback was
instructionally purposeful and well-structured.
Retrieval and response quality.The RAGAs
evaluation indicated that KITE performed well
on measures tied to retrieval and grounding, with
context relevance of 0.94 and faithfulness of
0.85. These results show that KITE’s multi-stage
retrieval pipeline surfaced course-specific material
relevant to the questions and that its responses
remained closely grounded in that retrieved
context. At the same time, the lower factual
correctness score (0.45) warrants careful interpre-
tation, particularly relative to answer similarity
(0.76). Prior work has noted that RAGAs-style
claim matching and score stability can vary
across response formulations and evaluation
settings (Roychowdhury et al., 2024; Antal and
Buza, 2025). In light of prior work, the 0.31-point
gap observed in our results may reflect limitations
of factual correctness as a reference-based metric
for evaluating KITE’s pedagogically framed
responses, although our evaluation does not isolate
the source of that discrepancy.

Pedagogical effectiveness and design impli-
cations.Following the RAGAs evaluation, the
simulated student and expert evaluations examined
whether KITE’s feedback supported stronger re-
vised answers, and was judged to be pedagogically
appropriate. Specifically, the 88.89% improvement
rate in the simulated student pipeline, together with
the strong expert rubric scores for scaffolding, guid-
ance, coherence, and tone, support that KITE’s
feedback can support stronger follow-up answers
on procedural and tracing questions. This emphasis
on feedback quality is consistent with prior survey
work on LLM applications in programming ed-
ucation, which argues that the educational value
of these systems depends on aligning model capa-
bilities with pedagogical goals, including the use
of scaffolding and feedback strategies (Pitts et al.,
2025a). In KITE, this alignment is reflected in pair-
ing retrieval-grounded generation with feedback
strategies designed for different forms of student
support, such as direct explanations, or algorithmic-
tracing guidance. While these findings are encour-
aging, the current evaluation design limits their
generalizability, as discussed in Section 6.1.
6.1 Limitations and Future Work
This study has limitations that motivate future work.
First, our evaluation of factual correctness is con-
strained by limitations of the RAGAs framework.
KITE produces pedagogically framed explanations
that may paraphrase or elaborate on course mate-
rial, whereas RAGAs decomposes each response
into atomic claims and uses NLI-style entailment to
assess agreement with a single instructor-authored
answer. With this, responses that are semanti-
cally aligned with the expected answer but differ
in phrasing, detail, or framing may receive lower
factual correctness scores. The 0.31-point gap be-
tween factual correctness (0.45) and answer similar-
ity (0.76) in Table 1 is consistent with this concern,
although our evaluation does not isolate the source
of that discrepancy. Answer similarity, which is
based on semantic embeddings, remains substan-
tially higher and shows low variance ( σ= 0.09 ), in-
dicating relatively stable semantic alignment across
questions. We therefore treat answer similarity as
the primary quality indicator and note that using
multiple human-written answers could reduce this
limitation in future evaluations. Related concerns
have been noted in prior work: Roychowdhury
et al. (Roychowdhury et al., 2024) discuss limita-tions in how RAGAs decomposes and assigns state-
ments during metric computation, while Antal and
Buza (Antal and Buza, 2025) show that RAGAs-
based evaluation outcomes vary across question
types and retrieval conditions. Future evaluations
could use richer answer sets and metrics.
Second, the simulated student pipeline relies on
a single LLM, Meta-Llama-3.1-70B-Instruct, as a
proxy for student behavior. As a result, improve-
ment between Round 1 and Round 2 should be in-
terpreted as evidence that KITE’s feedback makes
a stronger answer more recoverable, not as direct
evidence of genuine student learning. Real learners
may differ substantially in both the magnitude and
pattern of improvement. Although this design is
useful for early-stage evaluation, it cannot substi-
tute for classroom evidence. A necessary next step
is deployment with real students, including pre-
and post-interaction assessments, analysis of revi-
sion behavior over time, and closer examination of
how learners engage with feedback.
Third, the expert evaluation covers a relatively
limited set of interaction cases, which constrains
the generalizability of the findings. Although inter-
rater agreement was strong ( κ= 0.88 ), judgments
of answer improvement and pedagogical quality
still involve subjectivity, and the sample size limits
precision. Future work should expand the annota-
tion set and use finer-grained scoring schemes to
better capture variation in feedback quality and an-
swer improvement across question and error types.
7 Conclusion
We presented KITE, a RAG-based intelligent tu-
toring system that combines a five-stage retrieval
pipeline with intent-aware pedagogical response
generation. KITE adapts its responses to the type
of student query, providing grounded explanations
for factual questions and Socratic scaffolding for
procedural and reasoning tasks. To evaluate these
response modes, we introduced a two-part evalu-
ation framework. RAGAs metrics assess retrieval
quality, while a simulated student pipeline exam-
ines whether KITE’s feedback supports improved
responses on procedural and tracing questions. Ex-
pert review using a structured rubric further eval-
uates the pedagogical quality of KITE’s feedback
and verifies improvement in students’ revised an-
swers. This work contributes an intent-aware tu-
toring architecture and an evaluation approach for
RAG systems with mixed response strategies.

Acknowledgements
This research was supported by the U.S. National
Science Foundation (NSF) under Grant #2426837.
Any opinions, findings, and conclusions expressed
in this material are those of the authors and do not
necessarily reflect views of the NSF. This work
was additionally supported by Research Council of
Finland grants #356114 and #367787.
References
Margit Antal and Krisztian Buza. 2025. Evaluating
open-source llms in rag systems: a benchmark on
diploma theses abstracts using ragas: M. antal, k.
buza.Acta Universitatis Sapientiae, Informatica,
17(1):5.
Allan Collins, John Seely Brown, and Susan E. Newman.
1989. Cognitive apprenticeship: Teaching the crafts
of reading, writing, and mathematics. In Lauren B.
Resnick, editor,Knowing, Learning, and Instruction:
Essays in Honor of Robert Glaser, pages 453–494.
Lawrence Erlbaum Associates.
Rajashree Dahal, Greg Murray, Robin Chataut, Mo-
hamed Hefeida, Anurag Srivastava, and Prashnna
Gyawali. 2025. Autota: A dynamic intent-based vir-
tual teaching assistant for students using open source
llms.IEEE Access.
David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido
Hakimi, Iryna Gurevych, and Mrinmaya Sachan.
2025. From problem-solving to teaching problem-
solving: Aligning llms with pedagogy using rein-
forcement learning.
Chenxi Dong, Yimin Yuan, Kan Chen, Shupei Cheng,
and Chujie Wen. 2025. How to build an adaptive ai tu-
tor for any course using knowledge graph-enhanced
retrieval-augmented generation (kg-rag). In2025
14th International Conference on Educational and
Information Technology (ICEIT), pages 152–157.
IEEE.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th conference of the european chapter of
the association for computational linguistics: system
demonstrations, pages 150–158.
Harry Barton Essel, Dimitrios Vlachopoulos, Al-
bert Benjamin Essuman, and John Opuni Amankwa.
2024. Chatgpt effects on cognitive skills of under-
graduate students: Receiving instant responses from
ai-based conversational large language models (llms).
Computers and Education: Artificial Intelligence,
6:100198.
Arthur C Graesser, Katja Wiemer-Hastings, Peter
Wiemer-Hastings, Roger Kreuz, Tutoring ResearchGroup, and 1 others. 1999. Autotutor: A simula-
tion of a human tutor.Cognitive Systems Research,
1(1):35–51.
Hadi Hasan, Ali Ismail, Ammar Mohanna, and Ali
Chehab. 2025. Kag: A scalable knowledge-
augmented generation system for educational content
management. In2025 3rd International Conference
on Foundation and Large Language Models (FLLM),
pages 503–508. IEEE.
Cindy E Hmelo-Silver and Howard S Barrows. 2006.
Goals and strategies of a problem-based learning fa-
cilitator.Interdisciplinary journal of problem-based
learning, 1(1):4.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with gpus.IEEE trans-
actions on big data, 7(3):535–547.
Kenneth R Koedinger and Vincent Aleven. 2007. Ex-
ploring the assistance dilemma in experiments with
cognitive tutors.Educational psychology review,
19(3):239–264.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yan Li, Janelle Yorke, Jiaying Li, Mengting He, Yushen
Dai, Yan Zhao, Jing Qin, and Xiangen Hu. 2026.
An innovative socratic method-based artificial intel-
ligence platform for healthcare education: A quasi-
experimental study.Nurse education in practice,
page 104770.
Zongxi Li, Zijian Wang, Weiming Wang, Kevin Hung,
Haoran Xie, and Fu Lee Wang. 2025. Retrieval-
augmented generation for educational application: A
systematic survey.Computers and Education: Artifi-
cial Intelligence, 8:100417.
Zhensheng Liu, Prateek Agrawal, Saurabh Singhal,
Vishu Madaan, Mohit Kumar, and Pawan Kumar
Verma. 2025. Lpitutor: an llm based personalized
intelligent tutoring system using rag and prompt en-
gineering.PeerJ Computer Science, 11:e2991.
Mihran Miroyan, Chancharik Mitra, Rishi Jain, Gireeja
Ranade, and Narges Norouzi. 2025. Analyzing peda-
gogical quality and efficiency of llm responses with
ta feedback to live student questions. InProceedings
of the 56th ACM Technical Symposium on Computer
Science Education V . 1, pages 770–776.
Meenakshi Mittal, Rishi Khare, Mihran Miroyan, Chan-
charik Mitra, and Narges Norouzi. 2026. Edumod-
llm: A modular approach for designing flexible and
transparent educational assistants. InProceedings
of the AAAI Conference on Artificial Intelligence,
volume 40, pages 40652–40660.

Alexander Tobias Neumann, Yue Yin, Sulayman Sowe,
Stefan Decker, and Matthias Jarke. 2024. An llm-
driven chatbot in higher education for databases and
information systems.IEEE Transactions on Educa-
tion, 68(1):103–116.
Manooshree Patel, Rayna Bhattacharyya, Thomas Lu,
Arnav Mehta, Niels V oss, Narges Norouzi, and
Gireeja Ranade. 2026. Leantutor: Towards a ver-
ified ai mathematical proof tutor. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 40, pages 40670–40678.
Zaki Pauzi, Michael Dodman, and Manolis Mavrikis.
2025. Automating pedagogical evaluation of llm-
based conversational agents. InCeur Workshop Pro-
ceedings, volume 4006. CEUR.
Griffin Pitts, Anurata Prabha Hridi, and Arun Bala-
jiee Lekshmi Narayanan. 2025a. A survey of llm-
based applications in programming education: Bal-
ancing automation and human oversight. InProceed-
ings of the Fourth Workshop on Bridging Human-
Computer Interaction and Natural Language Pro-
cessing (HCI+ NLP), pages 255–262.
Griffin Pitts, Viktoria Medvedeva Marcus, and Sanaz
Motamedi. 2025b. Student perspectives on the ben-
efits and risks of ai in education. In2025 ASEE
Annual Conference & Exposition.
Griffin Pitts and Sanaz Motamedi. 2026. What
drives students’ use of ai chatbots? technology
acceptance in conversational ai.arXiv preprint
arXiv:2602.20547.
Griffin Pitts, Neha Rani, and Weedguet Mildort. 2026.
Trust and reliance on ai in education: Ai literacy and
need for cognition as moderators.arXiv preprint
arXiv:2604.01114.
Griffin Pitts, Neha Rani, Weedguet Mildort, and Eva-
Marie Cook. 2025c. Students’ reliance on ai in higher
education: identifying contributing factors. InInter-
national Conference on Human-Computer Interac-
tion, pages 86–97. Springer.
Sujoy Roychowdhury, Sumit Soman, HG Ranjani,
Neeraj Gunda, Vansh Chhabra, and SAI KRISHNA
BALA. 2024. Evaluation of rag metrics for question
answering in the telecom domain. InICML 2024
Workshop on Foundation Models in the Wild.
Thomas Thesen and Soo Hwan Park. 2025. A genera-
tive ai teaching assistant for personalized learning in
medical education.NPJ Digital Medicine, 8(1):627.
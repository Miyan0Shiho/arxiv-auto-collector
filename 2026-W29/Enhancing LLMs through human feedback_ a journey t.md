# Enhancing LLMs through human feedback: a journey towards self-improvement

**Authors**: Tatiana Pelc, Gila Kamhi, Asaf Avrahamy, Adi Fledel-Alon

**Published**: 2026-07-13 08:51:08

**PDF URL**: [https://arxiv.org/pdf/2607.11267v1](https://arxiv.org/pdf/2607.11267v1)

## Abstract
In the rapidly evolving landscape of information retrieval systems, the ability to adapt and improve through user feedback is paramount. This study introduces a novel methodology for refining the performance of a primary Retrieval Augmented Generation (RAG) system by strategically integrating an auxiliary feedback RAG system. By systematically harnessing human-generated feedback, the approach aims to enhance the accuracy, relevance, and overall quality of responses, driving the system towards self-improvement. Central to this methodology is a human-in-the-loop implementation, where user feedback is continuously collected, classified, and integrated into the inference workflow, enabling the system to learn and evolve iteratively. To validate the effectiveness of this approach, the study employs rigorous testing against three diverse benchmark datasets focused on general and custom domain knowledge, utilizing a LLM-as-a-Judge evaluation strategy. This comprehensive framework not only underscores the transformative potential of feedback-driven enhancements in RAG systems but also sets a precedent for future research in adaptive information retrieval technologies, marking a significant step in the journey towards autonomous refinement and optimization through user engagement.

## Full Text


<!-- PDF content starts -->

Enhancing LLMs through human feedback: a journey towards
self-improvement
Tatiana Pelc1, Gila Kamhi1, Asaf Avrahamy1and Adi Fledel-Alon1,*
1Intel Corporation
Abstract
In the rapidly evolving landscape of information retrieval systems, the ability to adapt and improve through user feedback is paramount.
This study introduces a novel methodology for refining the performance of aprimaryRetrieval Augmented Generation (RAG) system
by strategically integrating anauxiliaryfeedback RAG system. By systematically harnessing human-generated feedback, the approach
aims to enhance the accuracy, relevance, and overall quality of responses, driving the system towards self-improvement. Central to this
methodology is a human-in-the-loop implementation, where user feedback is continuously collected, classified, and integrated into the
inference workflow, enabling the system to learn and evolve iteratively. To validate the effectiveness of this approach, the study employs
rigorous testing against three diverse benchmark datasets focused on general and custom domain knowledge, utilizing aLLM-as-a-Judge
evaluation strategy. This comprehensive framework not only underscores the transformative potential of feedback-driven enhancements
in RAG systems but also sets a precedent for future research in adaptive information retrieval technologies, marking a significant step
in the journey towards autonomous refinement and optimization through user engagement.
Keywords
RLHF, Retrieval Augmented Generation, Large language models, LLM-as-a-Judge
1. Introduction
Large Language Models (LLMs) have driven recent advances
in AI, enabled by their unprecedented scale and improved
through Reinforcement Learning from Human Feedback
(RLHF). RLHF refines model behavior using human input,
significantly enhancing adaptability, responsiveness, and
overall performance.
However, traditional AI systems have relied on binary
feedback (e.g., like/dislike), which limits learning depth.
This paper proposes incorporating rich textual human feed-
back—including detailed corrections—into the generative AI
pipeline. This enables more nuanced learning and improves
response accuracy and contextual alignment.
1.1. Related work
Retrieval Augmented Generation (RAG) enhances LLMs by
retrieving relevant documents, but often fails to capture
user preferences, resolve contextual mismatches, or adapt
to evolving information needs. Common issues include re-
trieval bias, robustness challenges, and fairness and privacy
concerns [ 1,2]. Early approaches with static retrieval and
binary feedback offered limited gains. Recent work em-
phasizes the role of structured and dynamic feedback to
improve alignment and reliability. Hybrid methods combin-
ing human judgment with automated evaluation have been
explored to reduce hallucinations. Human-in-the-loop sys-
tems and cross-domain adaptation are emerging as effective
strategies for building self-improving, context-aware RAG
systems.
1.1.1. Feedback in RAG and LLM systems
Structured feedback pipelines such as CDF-RAG (Causal
Dynamic Feedback for Adaptive Retrieval-Augmented Gen-
eration) refine retrieval and generation through iterative
loops guided by causal graphs, enabling multi-hop reason-
ing and improving accuracy [ 3]. In educational applications,
*Corresponding author.
/envel⌢pe-⌢penadi.fledel.alon@intel.com (A. Fledel-Alon)
©2025 Copyright for this paper by its authors. Use permitted under Creative Commons License
Attribution 4.0 International (CC BY 4.0).timestamp-anchored feedback has reduced hallucinations by
37% and improved user satisfaction and learning outcomes
[4].
1.1.2. Cross-domain, comparative, multi-modal
feedback integration
LLM-Cure addresses large-scale user review analysis by
combining automated feature extraction with competitive
analysis [ 5]. Analyzing over one million reviews from 70
apps, it introduces a three-stage remediation process. The
method achieves an 85% F1-score in categorizing feature-
specific complaints, outperforming prior approaches by
13%. The system matches issues with top competitor so-
lutions and generates actionable suggestions, 73% of which
have been adopted by developers. Additionally, 68% of
recommended features appeared in app updates within
three months, demonstrating effective, scalable, and market-
aware remediation.
Complementing this, USER-LLM integrates textual feed-
back with behavioral data—such as clickstreams and ses-
sion durations—into unified embeddings [ 6]. This approach
improves personalization accuracy by 28% over text-only
models and enables LLMs to distinguish between transient
frustrations and systemic issues needing intervention.
1.1.3. Feedback driven alignment
Effective feedback-driven learning requires addressing chal-
lenges in feedback collection, interpretation, and alignment.
Subjective user preferences, inconsistent evaluation criteria,
and underrepresentation in feedback datasets can introduce
bias and reduce generalizability [ 7,8]. To address this, re-
cent research expands feedback modalities to include natu-
ral language critiques, numerical ratings, and direct edits,
increasing expressiveness but complicating integration [ 9].
To overcome limitations of traditional RLHF, alternative
alignment strategies have emerged. Direct Preference Opti-
mization (DPO) simplifies training by replacing reinforce-
ment learning with a closed-form objective, reducing com-
putational overhead [ 10]. Safe RLHF introduces dual objec-
tives—helpfulness and harmlessness—to promote aligned
and safe outputs [ 11,12]. Reinforcement Learning from AI

Feedback (RLAIF) further automates alignment by using
AI-generated reward signals, enhancing scalability [13].
1.2. Our contribution
This paper introduces a dynamic framework for enhanc-
ing LLMs through real user feedback, enabling continu-
ous system-level adaptation. Unlike static RLHF pipelines,
our method classifies, aggregates, embeds, and retrieves
feedback within the generation loop. Validated using the
LLM-as-a-Judgeprotocol [ 14], the framework improves con-
textual relevance, factual accuracy, and alignment. By inte-
grating diverse feedback types—such as textual corrections
and behavioral signals—and employing efficient optimiza-
tion, we support scalable, robust, and user-centered LLM
solutions.
2. Methods
2.1. FLARE Framework Overview
We present FLARE (Feedback-driven LLM Adaptive
RAG Enhancement), an iterative framework designed
to improve RAG systems by leveraging structured user
feedback. FLARE operates through two independent
phases: offline preprocessing and online inference. These
phases run in parallel with the primary RAG pipeline.
We evaluate the framework using three datasets: the
Fictional Persona Dialogs dataset, available on Hugging
Face Fictional_Persona_Dialogs_Anonymized ;
the Trivia dataset, accessible on Hugging Face
Current_Trivia_Knowledge-benchmark ; and the
Wireless Technical Documentation dataset (proprietary).
The effectiveness is assessed using theLLM-as-a-Judge
protocol.
2.2. Feedback processing architecture
2.2.1. Offline feedback management pipeline
The offline component of FLARE processes user feedback
through a multi-stage pipeline that extracts actionable in-
sights and converts them into structured knowledge repre-
sentations:
Feedback collection and storageUser-generated free-
text feedback is continuously collected and stored in a Mon-
goDB database. This feedback is processed in periodic
batches to ensure scalability while maintaining a responsive
user experience.
Contextual enrichmentRaw feedback often lacks suf-
ficient context for standalone interpretation. To address
this, each feedback instance is enriched with relevant con-
versation history, converting fragmented inputs into self-
contained, interpretable statements. For example:
•Original interaction: Q: “Who is the current pres-
ident of the US?” A: “Joe Biden.” F: “Wrong, it is
Donald Trump.”
•Enriched feedback: “The current president of the US
is Donald Trump as of 2025.”Prompt:Synthesize a new question
from the user’s question and chat
history context. Make the question
concise, retaining only the core con-
tent.
(a)LLM prompt for synthesiz-
ing a new questionPrompt:Generate a concise fac-
tual statement by synthesizing fac-
tual information from the user ques-
tion and feedback text. Ensure the
statement is accurate and coherent
based on the provided feedback. If
the feedback text does not directly
answer the user question or is not
relevant, leave the field empty.
(b)LLM prompt for generat-
ing factual statements
Prompt:Extract key general guide-
lines from the feedback that could
be added to the prompt for future
use; keep empty if not relevant.
(c)LLM prompt for extract-
ing general guidelines
Figure 1:System prompt instructions. The figure illustrates dif-
ferent types of prompts used in LLM systems for synthesizing
questions, generating factual statements, and extracting guide-
lines.
Multi-category feedback classificationThe enriched
feedback is automatically processed by LLM to generate two
categories:
1.Factual data corrections— Target inaccuracies,
outdated information, or missing knowledge,
prompting direct updates to the system’s knowledge
base (Figure 1b).
2.Stylistic and structural guidelines— Address clar-
ity, coherence, and presentation quality without al-
tering factual content (Figure 1c).
Query synthesis and refinementOriginal user queries
undergo context-aware refinement to remove extraneous
elements while preserving core semantic meaning, as illus-
trated in Figure 1a. This process ensures feedback can be
accurately linked to clean, standardized query representa-
tions.
Structured knowledge organizationProcessed out-
puts are stored as structured dictionaries, pairing synthe-
sized queries with corresponding factual corrections and
stylistic guidelines. This format enables efficient retrieval
and application during inference.
Intelligent feedback aggregationGPT-4o aggregates
feedback by identifying and consolidating redundant or con-
flicting entries. A relevance filter retains only actionable
feedback that directly improves system performance, opti-
mizing computational efficiency.
Vector embedding and indexingSynthesized queries
are embedded using OpenAI’s text-embedding-ada-002
model and indexed in an Azure Vector Database, enabling
fast similarity-based retrieval during inference.

Figure 2:FLARE offline processing flow
Figure 3:FLARE inference flow
2.2.2. Online feedback integration workflow
The online component integrates processed feedback into
the active RAG pipeline in real time:
Dual retrieval architectureEach incoming query is pro-
cessed in parallel by the primary RAG system and a dedi-
cated Feedback-RAG module. This setup ensures compre-
hensive coverage without increasing latency.
Adaptive feedback integrationAzure Hybrid Search
identifies previously indexed queries that exhibit semantic
similarity to the current user input. Queries exceeding a
predefined relevance threshold are selected for integration.
Factual corrections from these selected queries are injected
directly into the primary RAG context, while stylistic guide-
lines are integrated as system-level instructions to refine
output formatting and tone.
2.3. Experimental design and datasets
2.3.1. Dataset selection and construction
We evaluated FLARE on three datasets representing vary-
ing degrees of real-world complexity, from fully syntheticenvironments to real user interactions and feedback:
Fictional Persona Dialogs (FPD) datasetThis dataset
consists of 68 anonymized dialogues from fictional media
sources, selected to avoid contamination from prior model
knowledge. Question-answer pairs were generated contex-
tually using GPT-4o, and the dataset is publicly available on
Hugging Face for reproducibility.
Baseline evaluations without contextual information
demonstrated low accuracy (~1 on a 1–5 scale), which im-
proved dramatically to near-perfect performance following
context integration. For FLARE evaluation, we removed
30% of the context to measure improvements in a controlled
setting. Full-context answers were used to generate training
feedback, while semantically altered questions formed the
test set—ensuring improvements reflected adaptive general-
ization, not memorization
Trivia datasetThis dataset simulates realistic RAG de-
ployment conditions involving ambiguous, time-sensitive
queries and human feedback. It spans diverse do-
mains—e.g., Olympic results, Nobel laureates, leadership
changes—requiring up-to-date knowledge.

Algorithm 1:Batch processing of user feedback
(offline FLARE)
Input:Batch𝐵={𝑓 1, . . . , 𝑓 𝑛}of raw user
feedback; conversation log𝒞
Output: Structured feedback store 𝒮; vector index𝒱
1foreach𝑓 𝑖∈𝐵do
2 ˜𝑓𝑖←EnrichContext(𝑓 𝑖,𝒞);// Add
dialogue history
3(𝑞 𝑖, 𝜅𝑖)←
RefineQueryAndExtractCore( ˜𝑓𝑖);
4𝑐 𝑖←ClassifyFeedback(𝜅 𝑖);// factual vs.
stylistic
5𝒮←𝒮∪{(𝑞 𝑖, 𝑐𝑖)};
6𝒮←AggregateAndResolveConflicts(𝒮);
7𝒮←FilterActionable(𝒮);// drop noisy
items
8foreach(𝑞, 𝑐)∈𝒮do
9𝑣←Embed(𝑞);
10𝒱←𝒱∪{(𝑣, 𝑐)};// store in Azure
Vector DB
11return(𝒮,𝒱)
Algorithm 2:Adaptive feedback-aware inference
(online FLARE)
Input:User query𝑞*; primary RAG systemℛ;
feedback store𝒮; vector index𝒱; similarity
threshold𝜏
Output: Final answer 𝑎*with integrated factual and
stylistic improvements
1𝑣*←Embed(𝑞*);
2ℱ←HybridSearch(𝒱, 𝑣*, 𝜏);// retrieve
nearest neighbours
3ifℱ̸=∅then
4(∆ fact,∆ style)←ExtractCorrections(ℱ,𝒮);
5ℛ←InjectFactualContext(ℛ,∆ fact);
6ℛ←InjectStylisticGuidelines(ℛ,∆ style);
7𝑎*←ℛ(𝑞*);// run augmented RAG
8LogInteraction(𝑞*, 𝑎*);// for future
offline batches
9return𝑎*
Question-answer pairs were generated with GPT-4o us-
ing recent web-sourced data; context was drawn from rel-
evant Wikipedia articles to reflect standard RAG usage.
The dataset includes 70 training and 30 testing question-
answer pairs. Human-annotated feedback guided the train-
ing phase, while test question-answer pairs were derived
from feedback-enhanced examples to evaluate generaliza-
tion.
Wireless Technical Documentation datasetThis
domain-specific dataset reflects real-world RAG usage in
technical support, requiring expert knowledge of wire-
less networking. Questions were sourced from actual user
queries; answers were generated using retrievals from In-
tel documentation, specifications, and internal resources.
The dataset contains 80 question-answer pairs for training.
Test queries were semantically modified versions of train-
ing examples to assess generalization and robustness in ahigh-precision, expert-driven domain.
2.4. Evaluation framework
2.4.1. LLM-as-a-Judge assessment protocol
We implemented a comprehensive evaluation framework
using anLLM-as-a-Judgeapproach to systematically assess
response quality across multiple dimensions:
Quantitative metricsFor list-based queries requiring
enumeration or categorization, we used F1-score metrics
to balance precision and recall, based on standard True
Positive (TP), False Positive (FP), and False Negative (FN)
counts. Representative examples include: “List all products
implementing Wi-Fi 6 technology” and “Which frame types
can remain unprotected, and under what conditions are they
dropped?”.
Qualitative assessment criteriaNon-list queries under-
went comprehensive evaluation across three critical dimen-
sions:
1.Factual Accuracy: Verification of content correct-
ness against established ground truth
2.Contextual Relevance: Assessment of response
appropriateness to the specific query context
3.Completeness: Evaluation of response comprehen-
siveness relative to predefined reference answers
This multi-dimensional evaluation provides robust valida-
tion of FLARE’s effectiveness and establishes a reproducible
framework for future research in adaptive retrieval and
feedback-driven generative AI.
3. Results
As described in the previous section, we constructed three
benchmark datasets to gain confidence on the premise of our
solution across both general and domain-specific knowledge
settings, demonstrating FLARE’s versatility and effective-
ness in diverse contexts..
We generated embedding representations using question-
answer conversations and feedback history, and synthesized
revised questions to form a test set. Below, we report our
findings, showing significant gains in RAG system perfor-
mance when augmented with the FLARE methodology.
3.1. General knowledge: FPD benchmark
This benchmark was primarily used as a sanity check to
verify the correctness of our implementation and ensure
adherence to the intended FLARE flow. As can be seen in the
results depicted in Figure 4 and Table 3 , we demonstrate a
marked improvement in performance with the use ofFLARE,
achieving a maximum score of 5 on nearly all items.
3.2. General knowledge: Trivia benchmark
Additionally, we constructed a benchmark consisting of
Trivia questions centered on events from 2024 and 2025—be-
yond the knowledge cut-off of the LLM under evaluation
(GPT-4o). To ensure the RAG system had access to relevant,

Table 1
Summary of evaluation datasets
Dataset Size (Train/Test) Domain Source Type Purpose
Fictional Persona Dialogs 68 (train/test split) Open-domain dialogue Synthetic Controlled evaluation
Trivia 70/30 Current events Semi-synthetic Generalization
Wireless Technical Documentation 80 (train/test split) Technical Q&A Real-world Domain-specific validation
Table 2
Example of how FLARE leverages user feedback: initial RAG response and score; injected knowledge and instruction; improved
output with its correctness score.
Section Content
User Question "Which acting award recipient at the 2025 Oscars both earned their very first nomination and became the first American of
Dominican descent ever to win an acting Oscar?"
Reference Answer "Zoe Saldaña, another first-time nominee, took home Best Supporting Actress for her performance in *Emilia Pérez*, making
history as the first American of Dominican descent to win an acting Oscar."
Initial RAG Response "The acting award recipient at the 2025 Oscars who both earned their very first nomination and became the first American of
Dominican descent ever to win an acting Oscar is Mikey Madison. She won the award for Actress in a Leading Role for her
performance inAnora"
Suggested Knowledge "At the 2025 Oscars, several first-time nominees won major awards, including Mikey Madison, Kieran Culkin, and Zoe Saldaña,
who made history as the first American of Dominican descent to win an acting Oscar. Sean Baker also tied Walt Disney’s
record for most wins in a single night, with additional debut wins in directing and technical categories."
Suggested Takeaway "Include detailed and historically significant information from the feedback when summarizing key winners, such as notable
upsets, demographic records, and additional category recognitions."
RAG Response with FLARE "Zoe Saldaña earned her very first nomination and became the first American of Dominican descent ever to win an acting
Oscar, winning in the category of Actress in a Supporting Role for her performance inEmilia Pérez."
Score:5 (correct host, country, and dates)
Table 3
FPD benchmark results summary
Score T-Test (p)
No FLARE FLARE
AVG 4.0 5.0 –
STD 1.7 0.24×10−6
up-to-date information, we supplemented its retrieval cor-
pus with curated Wiki and web pages. Human evaluators
were then asked to provide textual feedback on the system’s
responses. Using the questions, RAG answers, and feedback
history, we generated embedding representations and cre-
ated a revised set of test questions within the same domain.
The integration of FLARE resulted in a substantial im-
provement in both the accuracy and relevance of responses.
As shown in Figure 5, FLARE achieved an average score of
3.91 compared to 3.06 for the baseline RAG system, based
on a test set of 30 newly generated questions from the same
domain as the training data. Average and best score were
computed across three different responses per question. Un-
like the baseline RAG system—which struggled to generateaccurate answers even with access to up-to-date Wiki and
web content—FLARE effectively leveraged user feedback to
adapt to new information beyond its initial feedback train-
ing data. Table 4 presents a curated sample of test questions,
underscoring the LLM’s efficacy in evaluating the quality
of RAG system responses. The table also highlights the
substantial enhancement in answer quality when utilizing
FLARE compared to the absence of FLARE.
1.Feedback utilization:FLARE’s dynamic feedback re-
trieval mechanism plays a central role across bench-
marks, enabling continuous adaptation based on user-
provided feedback. This iterative loop yields more con-
textually appropriate and nuanced responses.
2.LLM-as-a-Judgeevaluation:Implementing this eval-
uation strategy offers a robust, scalable framework for
assessing FLARE’s improvements. It demonstrates the
system’s alignment with human judgment and prefer-
ences, further validating the efficacy of the feedback-
driven approach.
3.Targeted context via feedback:Sample question-
answer pairs reveal that the baseline RAG system often
fails to produce satisfactory responses, even when rel-
evant data is available. In contrast, FLARE effectively
Figure 4:Performance on the 68-question FPD benchmark, with and without FLARE. FLARE attains the maximum score of 5
on all but two items, which receive scores of 3.66 and 4.67, respectively (scores are out of 5).

Table 4
Sample Trivia test set evaluation via LLM:Each entry presents the user query, the RAG system’s response without FLARE,
followed by feedback and the response with FLARE.
RAG Answer with and without FLARE
Who won the Best Actor award at the 2025 Oscars, and for which film performance was the accolade given?
No FLARE:Adrien Brody won the Best Actor award at the 2025 Oscars for his performance in "The Brutalist."
LLM evaluation:The response is accurate regarding Adrien Brody’s win for "The Brutalist."
FLARE:Adrien Brody won the Best Actor award for his role as László Tóth in "The Brutalist." This marks his second Oscar win, the first being for "The Pianist" in 2003.
Which company or individual won the Best Innovation Award at CES 2025, and what innovation did they showcase?
No FLARE:Anker Solix won the Best Innovation Award at CES 2025 for their solar-powered cloak. This jacket blends fashion with technology, allowing device charging on the go.
LLM evaluation:The response incorrectly mentions Anker Solix, while Suprema AI won the award for their Q-Vision Pro.
FLARE:Suprema AI won the Best Innovation Award for their Q-Vision Pro, an AI module for predicting and preventing financial crimes at ATMs. It features AI-powered facial recognition
and behavior analysis.
What is the official name of Intel’s PC AI platform launched in late 2024-2025, and what are its main features?
No FLARE:Gemini is Intel’s PC AI platform, featuring AI-powered performance optimization and enhanced security. It integrates seamlessly with other devices.
LLM evaluation:The response incorrectly names the platform "Gemini," while it is officially the Intel Core Ultra processor family.
FLARE:Intel’s PC AI platform, the Intel Core Ultra processor family, integrates CPUs, GPUs, and NPUs to enhance productivity and security. The Lunar Lake series marks a significant
step in embedding AI capabilities into hardware.
Did Intel win a Best Product Award at MWC 2025, and what was the specific category of the award?
No FLARE:The context does not mention Intel winning a Best Product Award at MWC 2025. Various gadgets were showcased, but Intel’s award status is unconfirmed.
LLM evaluation:The response incorrectly states Intel did not win an award, while they won the "Best AI Debut Award" for their Intel AI Assistant Builder Tool.
FLARE:Intel won the "Best AI Debut Award" for their Intel AI Assistant Builder, a tool for creating customized AI assistants. This highlights Intel’s innovation in making AI technology
more accessible.
Figure 5:Results with and without FLARE for 30 Trivia bench-
mark questions: where FLARE average score is 3.91 versus NO
FLARE 3.06, respectively (out of score 5)
utilizes context extracted from previous user feedback
to provide satisfactory answers.
3.3. Custom domain benchmark
In the custom Wireless domain, we built a benchmark from
a year’s worth of real-world customer feedback on our inter-
nal large-scale RAG platform. It reflects practical challenges
and evolving needs in a dynamic environment.
For this benchmark, FLARE effectively integrated user
feedback to enhance the RAG system’s responses. We em-
bedded 210 textual user feedback entries collected over the
year. These entries were provided by users in response to a
dislike tag and were integrated into a RAG system built on
division-specific data.
We created a test set of 80 questions, derived from the
original user queries but phrased differently. We assessed
FLARE’s ability to utilize feedback for more accurate an-
swers in two key areas:
1.Ability to extract relevant feedback:we initially
tested FLARE’s ability to retrieve relevant feedback for
queries related to past user input. In our custom test
set, FLARE successfully extracted relevant feedback in
95% of cases (76 out of 80 queries). The four failures
occurred with queries concerning specific JIRA entries,
where feedback linked to the same entries was not al-
ways retrieved.2.Ability to enhance answers using feedback:we fur-
ther evaluated FLARE’s ability to generate meaningful
answers by leveraging feedback, automating this process
using anLLM-as-a-Judge. Results showed that although
relevant feedback was retrieved as context in nearly all
cases (95%), FLARE incorporated that feedback into its
answers only 61% of the time.
Our results clearly demonstrate the substantial improve-
ments made possible by incorporating user feedback into the
continuous self-improvement cycle of RAG systems. Never-
theless, as illustrated in Table 5, there remain opportunities
to further refine FLARE’s performance.
4. Conclusions and future work
We introduced FLARE, a feedback-aware extension to RAG.
A straightforward pipeline—comprising feedback classifica-
tion, aggregation, embedding, and retrieval at a similarity
threshold of 0.85—significantly improves average answer
quality, as demonstrated across three diverse contextual
benchmarks. The use ofLLM-as-a-Judgealigns automated
scoring with human preference while offering concise ratio-
nales.
During evaluation, two notable limitations emerged:
First, although the retrieval module successfully located
relevant feedback for nearly every custom-domain query,
the generation module incorporated this feedback in only
61% of the answers. Thisretrieval–utilization gapsuggests
that our current context injection method—template-based
string concatenation—can marginalize important details.
We currently lack a principled mechanism to prioritize feed-
back when model attention is constrained, or when snippets
overlap or conflict.
Second, feedback selection currently depends on a fixed
cosine similarity threshold and treats all retrieved comments
as carrying equal weight. This uniform policy ignores cues
that strongly influence a comment’s utility—such as author
expertise, recency, consistency with other feedback, and
historical impact on answer quality. As a result, the system
may surface outdated or low-quality guidance while exclud-
ing more recent, reliable corrections that fall just below the
fixed cut-off. In future work, we plan to develop adaptive
ranking mechanisms that weight feedback based on rele-
vance, freshness, and author credibility. We also aim to
incorporate an iterative decoding loop that re-retrieves evi-

Table 5
Insights for enhancing FLARE:this table presents three examples from a custom domain benchmark, highlighting challenges
and potential improvements for FLARE (user feedback-driven RAG). Generic names are used to protect intellectual property.
The table also includes insights for further development, identifying areas where FLARE can be improved.
Insights on further improvements of FLARE answers (based on detailed analysis of test results from the custom test set)
In which JIRA entry is Feature X referenced?
Original User Feedback:Feature X appears in JIRA Entry-61987 but was not referenced for some reason.
FLARE:Feature X is referenced in JIRA Entry-61987.
Insight:Retrieval Reiteration- The FLARE answer could be further improved by providing details on the respective JIRA entry. This can be achieved by reiterating the RAG flow once
context for the specific user query is obtained from multiple sources, such as feedback and vector store embedding of custom domain data.
Which specific models does OEM Y use? Please refer to models (a, b, and c) in your answer.
Original User Feedback:Only models a and b are used by OEM Y. The c model is used by OEM X.
FLARE:OEM Y uses several versions of models a, b, and c.<Rest of the answer >
Insight:Feedback Prioritization- The fact that OEM Y does not use model c was overlooked. This indicates that the information in the feedback needs to be prioritized.
Please provide details of architectural feature Z.
Original User Feedback:Should at least provide some pointers to the right JIRA entry for this architectural feature.
FLARE:The details of the architectural feature are...<Details of the architectural feature are provided without pointers to specific JIRA entries >
Insight:Feedback Generalization- The FLARE answer could be improved through generalized learning from all past feedbacks, such as always including JIRA entry information when
providing details on architectural features.
dence when the current context is sparse or contradictory.
Additionally, we intend to broaden the feedback schema to
support hierarchical labels and multimodal inputs, such as
screenshots, voice notes, and click traces. Finally, we will
build a low-latency streaming pipeline capable of ingesting,
evaluating, and applying feedback in real time, blending
automated scoring with targeted expert review.
Together, these extensions aim to advance FLARE toward
becoming a fully autonomous, continuously self-improving
RAG framework.
Declaration on Generative AI
During the preparation of this work, the author(s) used GPT-
4o spell check and revise grammar, after which the author(s)
reviewed and edited the content as needed and take(s) full
responsibility for the publication’s content.
References
[1]S. Barnett, S. Kurniawan, S. Thudumu, Z. Bran-
nelly, M. Abdelrazek, Seven failure points when
engineering a retrieval augmented generation sys-
tem, 2024. URL: https://arxiv.org/abs/2401.05856.
arXiv:2401.05856.
[2]B. Ni, Z. Liu, L. Wang, Y. Lei, Y. Zhao, X. Cheng,
Q. Zeng, L. Dong, Y. Xia, K. Kenthapadi, R. Rossi,
F. Dernoncourt, M. M. Tanjim, N. Ahmed, X. Liu,
W. Fan, E. Blasch, Y. Wang, M. Jiang, T. Derr, To-
wards trustworthy retrieval augmented generation
for large language models: A survey, 2025. URL: https:
//arxiv.org/abs/2502.06872.arXiv:2502.06872.
[3]E. Khatibi, Z. Wang, A. M. Rahmani, Cdf-rag: Causal
dynamic feedback for adaptive retrieval-augmented
generation, 2025. URL: https://arxiv.org/abs/2504.
12560.arXiv:2504.12560.
[4]S. Jacobs, S. Jaschke, Leveraging lecture content for im-
proved feedback: Explorations with gpt-4 and retrieval
augmented generation, in: 2024 36th International
Conference on Software Engineering Education and
Training (CSEE&T), IEEE, 2024, p. 1–5. URL: http://dx.
doi.org/10.1109/CSEET62301.2024.10663001. doi: 10.
1109/cseet62301.2024.10663001.
[5]M. Assi, S. Hassan, Y. Zou, Llm-cure: Llm-based
competitor user review analysis for feature enhance-
ment, 2024. URL: https://arxiv.org/abs/2409.15724.
arXiv:2409.15724.[6]L. Ning, L. Liu, J. Wu, N. Wu, D. Berlowitz,
S. Prakash, B. Green, S. O’Banion, J. Xie, User-
llm: Efficient llm contextualization with user em-
beddings, 2024. URL: https://arxiv.org/abs/2402.13598.
arXiv:2402.13598.
[7]P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg,
D. Amodei, Deep reinforcement learning from hu-
man preferences, 2023. URL: https://arxiv.org/abs/1706.
03741.arXiv:1706.03741.
[8]N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe,
C. Voss, A. Radford, D. Amodei, P. Christiano, Learning
to summarize from human feedback, 2022. URL: https:
//arxiv.org/abs/2009.01325.arXiv:2009.01325.
[9]L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wain-
wright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,
A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller,
M. Simens, A. Askell, P. Welinder, P. Christiano,
J. Leike, R. Lowe, Training language models to follow
instructions with human feedback, 2022. URL: https:
//arxiv.org/abs/2203.02155.arXiv:2203.02155.
[10] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon,
C. D. Manning, C. Finn, Direct preference optimiza-
tion: Your language model is secretly a reward
model, 2024. URL: https://arxiv.org/abs/2305.18290.
arXiv:2305.18290.
[11] J. Dai, X. Pan, R. Sun, J. Ji, X. Xu, M. Liu, Y. Wang,
Y. Yang, Safe rlhf: Safe reinforcement learning from
human feedback, 2023. URL: https://arxiv.org/abs/2310.
12773.arXiv:2310.12773.
[12] J. Ji, X. Chen, R. Pan, H. Zhu, C. Zhang, J. Li, D. Hong,
B. Chen, J. Zhou, K. Wang, J. Dai, C.-M. Chan, S. Han,
Y. Guo, Y. Yang, Safe rlhf-v: Safe reinforcement learn-
ing from human feedback in multimodal large lan-
guage models, 2025. URL: https://arxiv.org/abs/2503.
17682.arXiv:2503.17682.
[13] H. Lee, S. Phatale, H. Mansoor, T. Mesnard, J. Fer-
ret, K. Lu, C. Bishop, E. Hall, V. Carbune, A. Ras-
togi, S. Prakash, Rlaif vs. rlhf: Scaling reinforce-
ment learning from human feedback with ai feed-
back, 2024. URL: https://arxiv.org/abs/2309.00267.
arXiv:2309.00267.
[14] G. Azov, T. Pelc, A. F. Alon, G. Kamhi, Self-
improving customer review response generation based
on llms, 2024. URL: https://arxiv.org/abs/2405.03845.
arXiv:2405.03845.
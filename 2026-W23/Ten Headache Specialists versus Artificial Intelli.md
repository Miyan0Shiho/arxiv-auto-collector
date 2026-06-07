# Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization: A Critical Evaluation and Comparison

**Authors**: Alejandro Lozano, Keiko Ihara, Ping-Hao Yang, Carrie E. Robertson, Jennifer Stern, Allan Purdy, Hsiangkuo Yuan, Pengfei Zhang, Yulia Orlova, Olga Fermo, Jennifer Hranilovich, Fred Cohen, Todd J. Schwedt, Jenelle A. Jindal, Serena Yeung-Levy, Chia-Chun Chiang

**Published**: 2026-06-03 20:58:43

**PDF URL**: [https://arxiv.org/pdf/2606.05436v1](https://arxiv.org/pdf/2606.05436v1)

## Abstract
Summarizing the latest medical literature to guide clinical decision-making is essential for evidence-based medicine and high-quality patient care. Yet clinicians face increasing challenges due to limited time with patients and a rapidly growing volume of published articles. Although retrieval-augmented large language models (LLMs) have shown promise in clinical summarization, human evaluations of their effectiveness in synthesizing broader scientific literature and direct comparisons to expert-written syntheses remain scarce. We constructed a RAG-based agentic AI framework using three state-of-the-art LLMs: Sonnet, GPT-4o, and Llama 3.1. A headache specialist created 13 questions, three for prompt optimization and ten for evaluation. Ten headache specialists across the United States and Canada each wrote a summary for one question, yielding four summaries per question (expert, Sonnet, GPT-4o, and Llama). The experts, blinded to authorship, critically evaluated the summaries, excluding the topic for which they wrote a summary, based on correctness, completeness, conciseness, and clinical utility, scoring each from 1 to 10 using standardized rubrics. They also ranked the summaries by preference and indicated whether they believed each summary was written by an expert or an LLM. Our study, comparing LLM- and expert-written literature summaries evaluated by headache specialists, showed that expert-written summaries were preferred, although experts sometimes found it challenging to distinguish between human- and AI-generated summaries. We also identified key expert-valued features beyond standard evaluation metrics that can guide future refinement of both human and AI literature summarization pipelines.

## Full Text


<!-- PDF content starts -->

Preprint: Under Review 1–18
Ten Headache Specialists versus Artificial Intelligence for
Clinical Literature Summarization: A Critical Evaluation
and Comparison
Alejandro Lozano*1Keiko Ihara*2Ping-Hao Yang3Carrie E. Robertson4
Jennifer Stern5Allan Purdy6Hsiangkuo Yuan7Pengfei Zhang8Yulia Orlova9
Olga Fermo10Jennifer Hranilovich11Fred Cohen12Todd J. Schwedt2
Jenelle A. Jindal1Serena Yeung-Levy1Chia-Chun Chiang†2
1Stanford University, Palo Alto, CA, USA
2Department of Neurology, Mayo Clinic, Rochester, MN, USA
3Department of Neurology, Dalhousie University, Halifax, Canada
4Jefferson Headache Center, Department of Neurology, Thomas Jefferson University, PA, USA
5Beth Israel Deaconess Medical Center, Boston, MA, USA
6Department of Neurology, University of Florida, Gainesville, FL, USA
7University of Colorado School of Medicine, Department of Pediatrics, Division of Child Neurology,
Aurora, CO, USA
8Department of Medicine, Mount Sinai Hospital, Icahn School of Medicine at Mount Sinai, New
York, NY, USA
9Department of Neurology, Mayo Clinic, Scottsdale, AZ, USA
10Harvard Medical School, Boston, MA, USA
11Department of Neurology, Mount Sinai Hospital, Icahn School of Medicine at Mount Sinai, New
York, NY, USA
∗Denotes equal contribution.
†Denotes corresponding authors: Chiang.Chia-Chun@mayo.edu
Abstract
Background:Summarizing the latest medical literature to guide clinical decision-making
is essential for evidence-based medicine and high-quality patient care. Yet clinicians face
increasing challenges due to limited time with patients and a staggering growing volume
of published articles. Although retrieval-augmented large language models (LLMs) have
shown promise in clinical summarization, human evaluations of their effectiveness in syn-
thesizing broader scientific literature and direct comparison to expert-written synthesis
remain scarce.
Methods:We constructed a RAG-based agentic AI framework using three state-of-the-
art LLMs, Sonnet, GPT-4o, and Llama 3.1. A headache specialist created 13 questions:
three for prompt optimization and ten for evaluation. Ten headache specialists across the
United States and Canada each wrote a summary for one question, yielding four summaries
per question (expert, Sonnet, GPT-4o, and Llama). The experts, blinded to authorship,
critically evaluated the summaries, excluding the topic for which they wrote a summary,
based on correctness, completeness, conciseness and clinical utility, scoring from 1-10 with
©A. Lozanoet al.arXiv:2606.05436v1  [cs.AI]  3 Jun 2026

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
standardized rubrics provided. They ranked the summaries by preference and indicated
which they believed was written by an expert vs LLM.
Results:Two hundred critical assessments of the summaries from ten experts were an-
alyzed. Overall, human experts’ written summaries scored the highest in all evaluation
metrics and were the most preferred responses, followed by Sonnet, GPT-4o, and Llama
3.1. Paired comparisons showed headache specialists outperformed GPT-4o and Llama
on most metrics, with no significant difference compared to Sonnet. Headache specialists
correctly distinguished expert-written from AI-written summaries 64% (32/50) of the time.
Beyond the metrics used, experts valued the quality of references, inclusion of key clinical
details such as medication dosage, synthesis across sources, and incorporation of clinical
nuance and experience.
Conclusion:Our study, comparing LLM and expert-written literature summaries evalu-
ated by headache specialists, showed that expert-written summaries were preferred, though
it was sometimes challenging for experts to distinguish between human vs AI-generated
ones. We also identified key expert-valued features, beyond standard evaluation metrics,
that can guide future refinement of human and AI literature summarization pipelines.
1. Introduction
Evidence-based medicine, defined as the conscientious, explicit, and judicious use of current
best evidence in making decisions about the care of individual patients (1) has long been
regarded as the gold standard of clinical practice. However, summarizing and integrating
the latest literature for each patient during increasingly time-constrained visits has become
a significant challenge (2) as scientific literature grows exponentially each year. Resources
such as UpToDate are widely used to address this gap, relying on expert clinicians to
synthesize and summarize the medical literature for use at the point of care. Yet, novel
evidence from early-stage therapies, emerging clinical trials, and case studies often develops
more rapidly than can be incorporated into expert-curated repositories. Furthermore, a
broad review of a topic might not provide sufficient details suitable for a specific clinical
scenario, highlighting opportunities for improvement.
Large language models (LLMs) have shown promise across a wide range of clinical
tasks, including diagnostic support (3), drafting discharge summaries (4), and retrieving
information from electronic health records (5). Building on this success, interest has grown
in LLM-based systems capable of retrieving, evaluating, and synthesizing scientific literature
with expert-level proficiency on demand, a trend that has steadily matured in recent years.
We have previously built an agentic AI framework based on chains of retrieval-augmented
LLMs, clinfo.ai, for summarizing medical literature from resources such as PubMed. The
framework was made open source (6). Additionally, in May 2025, the U.S. Food and Drug
Administration launched its first LLM-assisted scientific review pilot. Similarly, as of July
2025, OpenEvidence, an AI-powered platform that aggregates and synthesizes peer-reviewed
medical literature, reported onboarding 65,000 new verified U.S. clinicians per month.
Despite the widespread use of LLMs to summarize clinical evidence, several impor-
tant questions remain unanswered. How do LLM-generated summaries compare with those
written by clinical experts on specialized topics, and what characteristics distinguish expert-
authored summaries from those produced by LLMs? What do clinical experts emphasize
when writing and evaluating literature summaries, and what key attributes, beyond com-
monly used LLM evaluation metrics, define a high-quality summary from their perspective?
2

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Addressing these questions is important not only for assessing the current capabilities of
LLMs, but also for the development of future pipelines using LLMs to summarize and
synthesize literature. From our previous experiences evaluating LLM outputs for medical
tasks, we found that there was substantial variation and low agreement across participating
clinicians (Fleming et al. 2023). This may be related to the substantial variability across
different medical specialties. To pursue high quality, expert-level assessments, conducting
such evaluations within a specific area of medicine is crucial. We focused on Headache
Medicine for this study, designing and optimizing the clinfo.ai framework with content ex-
perts to enhance the quality of the generated summaries. We conducted a multidimensional
clinical evaluation comparing ten headache specialist versus AI-generated summaries from
different LLM systems across seven axes— correctness, completeness, conciseness, clinical
usefulness, preference, identification of human-written summaries, and free text comments.
The evaluations covered 15 subcategories within these dimensions. Our assessment is ex-
tensive and detailed, totaling over 120 annotation hours by a panel of 10 academic headache
specialists with an average of 21.9 years of medical experience across the United States and
Canada to critically evaluate all summaries. In total, each headache specialist reviewed 5
questions (20 summaries) and 180 annotated items (36 items per question), resulting in
200 critically evaluated summaries generated by LLMs or written by experts, and a total
of 1,800 expert-annotated concepts. To complement our multidimensional evaluation, we
additionally performed an in-depth free-text assessment to capture nuanced strengths and
weaknesses of the generated summaries (missed by commonly used metrics), emphasizing
aspects that clinical experts value in retrieval-augmented summarization. Our two-part
evaluation shows that standard quantitative metrics alone were not sufficient to capture
expert preferences, revealing limitations in prior assessments of LLM outputs. Even when
clinicians cannot reliably distinguish human- from AI-generated summaries, they still prefer
human-written ones, by a margin nearly twice that of the best-performing model (Sonnet).
Clinicians identified literature misinterpretation, omission of key concepts, and missing crit-
ical references as major bottlenecks of current RAG-enabled LLMs.
2. Related Work
2.1. Large Language Models (LLMs)
Large language models (LLMs) are deep learning systems trained on Internet-scale datasets
that exhibit strong natural language processing capabilities (7; 8). In medicine, LLMs have
been applied to conduct diagnostic conversations with patients (9; 10), summarize and gen-
erate clinical notes, support clinical trial matching (11), analyze and synthesis unstructured
electronic health record data, (5) and provide decision support for complex cases (12).
Remarkably, LLMs can perform all these tasks without task-specific training (7)—a
phenomenon known as zero-shot learning (13). This capability stands in stark contrast to
traditional machine learning models, which depend on labeled datasets, and to rule-based
systems, which require domain experts to manually encode rules and logic.
3

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
2.2. Retrieval Augmented Langue Models
LLMs often produce factually incorrect or unverifiable statements, a phenomenon commonly
referred to as hallucination (14). This limitation poses a particular challenge in medicine,
where factual grounding and source traceability are critical. This gap has led to the growing
adoption of retrieval-augmented generation (RAG) approaches, in which relevant external
documents (e.g. scientific literature) are dynamically retrieved and incorporated into the
model’s context before generation. This allows LLMs to produce answers supported by
explicit and traceable evidence rather than relying solely on parametric memory.
To this end, several studies have demonstrated the effectiveness of retrieval-based ap-
proaches for clinical question answering and decision support. For instance, Almanac (15)
and Clinfo.ai (16) (our previously developed framework) are open-source systems that
retrieve biomedical literature from PubMed to answer clinical questions, providing both
natural language summaries and traceable citations. Similarly, domain-specific retrieval-
augmented models have been developed for specialized areas of medicine: Guide-Bot inte-
grates retrieval from medical guideline repositories to enhance factual accuracy in oncology
(17), while DALK (18) is tailored to answer questions related to Alzheimer’s disease using
scientific literature.
2.3. Evaluation of Retrieval Augmented Language Models for literature
summaries
Given the growing development of multiple frameworks to summarize medical evidence us-
ing RAG LLMs, there is a growing need for systematic evaluations. Medeviedence (19)
introduces a closed VQA benchmark designed to assess retrieval performance on systematic
reviews. Their findings show that current models underperform relative to human experts
and often exhibit unjustified confidence and insufficient skepticism. However, benchmark
performance alone does not provide a complete picture; expert evaluation remains essential.
To this end, Tang et al. (2023) (20) examined the ability of two LLMs to summarize medical
evidence across six clinical domains using 53 systematic reviews, including 10 in neurology.
Clinical experts rated each generated summary for coherence, factual accuracy, compre-
hensiveness, and potential for harm. Subsequent work expanded this line of evaluation to
broader clinical tasks: Luo et al. (2024) (21) studied a RAG system for ophthalmology
using a panel of six medical experts, while Ke et al. (2025) (22) assessed RAG systems
for determining surgical fitness and providing preoperative guidance, focusing on accuracy,
consistency, and patient safety. Across these studies, retrieval augmentation consistently
reduced hallucinations, improved factual correctness, and increased clinician trust relative
to LLM-only baselines.Yet, no prior work directly compares RAG-enabled LLM
outputs to those of expert practicing clinicians.Consequently, we lack a clear un-
derstanding of how these systems perform and contrast with respect to experts. Since
outperforming standard LLMs is insufficient for real-world deployment, addressing this gap
is essential for building reliable, clinically useful systems.
4

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
3. Methods
3.1. Models
For our assessment, we include the proprietary models GPT-4o (OpenAI) (23) and Sonnet
(Anthropic), along with the widely used open-source model Llama-3.1 (24). The open-
source models are executed on a local PHI-compliant server equipped with eight NVIDIA
H200 GPUs using vLLM (25). This selection is not comprehensive but rather a focused
subset designed to represent both closed and open models given the time constrains of
expert clinical evaluation.
3.2. Retrieval Pipeline
We employed the starter retrieval pipeline initially proposed by (16), incorporating mod-
ifications that were rigorously validated in (19). System optimization was conducted in
collaboration with a headache specialist (CC) with extensive clinical and scientific writ-
ing experience. Prompts were iteratively refined using a held-out dataset comprising three
representative questions spanning epidemiology, diagnosis, and treatment considerations
across different headache disorders. Prompts were engineered in multiple steps and aimed
to optimize article selection by prioritizing higher levels of evidence, providing the different
levels of evidence for different study types, and instructed the model to include original
research articles rather than citing narrative reviews. The prompt also included instruc-
tions to include clinically actionable details, such as medication names, dosages, and routes
of administration. Our agentic, chain-of-LLMs pipeline integrated multiple LLMs with a
scientific article search index (PubMed). It comprised five sequential steps:
1.Query Generation: Transforms the user’s question into a carefully constructed
search query using MeSH terms.
2.Information Retrieval: Fetches abstracts from PubMed using a search index.
3.Relevance Classification: Employs an LLM to determine whether each retrieved
article is relevant.
4.Summarization: Condenses each relevant abstract, highlighting information most
pertinent to the user’s question.
5.Synthesis: Integrates the summarized abstracts into a structured list and generates
a concise, evidence-based summary.
3.3. Dataset collection
Ten headache specialists from North America (the U.S. and Canada) were invited to partic-
ipate in a two-phase study. In the first phase, clinicians were asked to formulate a clinical
question and provide a 200–300-word response in a review-article format, including cita-
tions (Supplemental File). They were instructed to be concise and written for practicing
clinicians and medical trainees to guide clinical practice and medical education. The par-
ticipating experts were all headache specialists from academic headache centers with an
average of 21.9 years of medical experience post medical school graduation, and were all
5

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
actively engaged in research. Each had prolific experience writing review articles, with a
minimum of three published narrative or systematic reviews.
3.4. Evaluation by Headache Specialists
In the second phase, an online questionnaire (Supplemental File) was distributed to the
ten headache specialists who participated in the first phase of this study. Each headache
specialist reviewed all summaries for five questions that they were not assigned to in Phase
1. The questionnaire included a rubric-based evaluation across the following categories and
subcategories:
1.Correctness(hallucinated or fabricated content, misinterpretation, fabricated cita-
tions)
2.Completeness(missing important concepts, incomplete statements, absence of key
references)
3.Conciseness(overly long summaries, inclusion of unnecessary information, redun-
dant content)
We separated each comment from reviewers into phrases so that each phrase can rep-
resent one of the subcategories. Each category was scored on a scale from 1 (worst) to 10
(best). When flaws were identified, one point was deducted from 10 for each flaw within the
relevant subcategory. Additionally, participants were asked to rate the overall usefulness
of each summary on a scale from 1 (not useful) to 10 (extremely useful), to identify the
most preferred summary among the four provided, and to indicate which summary they
believed was written by a human expert. Of note, there were four summaries for each ques-
tion, one written by an expert, three by different LLMs on the developed framework (GPT,
Llama and Sonnet). Free-text responses were also collected to capture the rationale for
their selections and any additional comments that did not fit within the structured rubric.
These qualitative responses were reviewed by two headache experts (CC, KI) and analyzed
thematically. All free text comments were read and categorized. The phrases/sentences of
these comments were classified into the 12 topics relating to the reasons of why the reviewers
identified human expert written answers vs AI generated summaries, and 14 topics among
optional free comments, respectively. The number of responses that included each topic was
summarized. When the reviewers explicitly stated that they had comments that did not
fall into the rubric categories (correctness, completeness, and conciseness) but mentioned
significant elements, those were added to the pool of answers for [Optional] questions.
4. Results
We received 50 evaluations, 200 critically evaluated summaries, and 1800 annotated con-
cepts from ten headache specialists, spanning 120+ hours of detailed review. Each response
included 5 numerical values (correctness, completeness, conciseness, usefulness and prefer-
ence) and 5 text responses (reasons for scoring correctness, completeness, conciseness, and
human-written summary identification and optional comment). There were no missing data
6

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
9.148.869.52
8.149.12
8.49.52
7.348.16 8.268.76
6.68.56
7.868.48
6.12
0246810
Average of Correctness_score Average of Completeness_score Average of Concise_score Average of Usefullness_score
Expert Sonnet GPT-4o Llama 3.1Mean Scores
Figure 1:Average expert-assigned scores (0–10) across correctness, completeness, concise-
ness, and usefulness for retrieval-augmented answers/summaries to all ten questions. The
maximum score of 10 indicates that no points were deducted in any category for summaries
generated by the LLM or human experts.
for the numerical entries, although a total of 13/150 text responses for correctness, com-
pleteness, and conciseness were empty. We received 50 comments relating to their choice
of identifying human authors among AI generated summaries, and 23 optional free-text
feedback.
4.1. Correctness, Completeness, Conciseness, Usefulness, Preference
Overall, comparison by analysis of variance among the four types of summaries (clini-
cal expert, Sonnet, GPT-4o, and Llama 3.1) showed significant differences in all four 10-
point scales (correctness, p=0.009; completeness, p=0.015; conciseness, p¡0.001; usefulness,
p¡0.001). Expert-written summaries received the highest scores across correctness, com-
pleteness, and usefulness with averages of 9.14, 8.86, and 8.14, respectively (Figure 1),
and were the most preferred by evaluators (26/50), averaging 3.1 in ranking (compared to
2.68 Sonnet, the second-best model). Regarding conciseness, expert-written and Sonnet-
generated summaries were tied for the highest scores at 9.52.
Comparison between expert summaries and each LLM output showed that for all four
categories, expert-written summaries and Sonnet-generated summaries had no significant
difference (correctness 9.14 vs 9.12, p=0.947; completeness 8.86 vs 8.40; p=0.130; concise-
ness 9.52 vs 9.52; p=1.0; usefulness 8.14 vs 7.34, p=0.057). Expert-written summaries
scored significantly higher than GPT-4o in all four categories (correctness 9.14 vs 8.16,
p=0.013; completeness 8.86 vs 8.26, p=0.025; conciseness 9.52 vs 8.76, p=0.008; usefulness
8.14 vs 6.60, p¡0.01) and scored significantly higher than Llama 3.1 in completeness (8.86
vs 7.86, p¡0.001), conciseness (9.52 vs 8.48, p¡0.001), and usefulness (8.14 vs 6.12, p¡0.00).
However, for correctness, there was no statistically significant difference between expert and
Llama 3.1 (9.14 vs 8.56, p=0.059).
7

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
0.36
0.10.0200.34
0.10.42
0.240.54
0.160.520.68
00.250.50.751
(1) Long Summary (2) Unnecessary Information (3) Repeated Information
Expert Sonnet GPT-4o Llama 3.1Mean Scores Deducted in Conciseness
Figure 2:Human evaluation of summary conciseness. Mean point deductions assigned by
headache specialists for three conciseness-related criteria: (1) overly long summaries, (2)
inclusion of unnecessary information, and (3) repeated information. Higher scores indicate
larger penalties and therefore poorer conciseness. Expert-written summaries received the
fewest penalties overall, whereas Llama 3.1 was most frequently penalized for unnecessary
and repeated information.
Examining the nine rubric subcategories, expert-written summaries had the lowest point
deductions (i.e., best performance) across all five subcategories (average scores): correct-
ness—misinterpretation (0.18); completeness—lack of important concepts (0.3) and lack of
key references (0.22); and conciseness—unnecessary information (0.1) and repeated informa-
tion (0.02). Sonnet-generated summaries had the lowest deductions in three subcategories
(average scores): correctness—hallucinated/fabricated content (0.3) and fabricated citations
(0); conciseness—long summaries (0). GPT-4o had the lowest deductions for completeness-
incomplete statement with an average of 0.42. Llama 3.1 did not demonstrate the best
performance in any of the four major categories or the subcategories.
4.2. Identification of Expert Written Summaries and Additional Feedback
Out of 50 responses (5 per question), 32 (64%) correctly identified the expert-written sum-
maries from LLM-generated summaries. Among the 18 incorrect identifications, 9, 7, and 2
responses incorrectly identified Sonnet-, GPT-4o-, and Llama 3.1- generated summaries as
human-written summaries, respectively. The most frequently attributed features to expert-
written summaries were focus of information (n=14), completeness (n=12), reference quality
(n=12), conciseness (n=10), expert opinion and judgment (n=8), and flow of information
(n=8). Additional features mentioned included AI mistakes such as incorrectly claiming
the summary was a systematic review (n=4), human errors including grammar errors or
typos (n=4), readability and writing style (n=4), synthesis (n=3), up-to-date information
and full-text content (n=3), and correctness (n=3). Notably, four features outside of rubric
were mentioned only by reviewers who correctly identified the expert-written summaries:
correctness, synthesization, AI mistakes, and up-to-date information/full-text content. Syn-
8

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
0.62
0.18 0.040.30.44
00.96
0.62
0.220.96
0.4
0.02
00.20.40.60.81
(1) Hallucinated/fabricated Content (2) Misinterpretation (3) Fabricated Citations
Expert Sonnet GPT-4o Llama 3.1Mean Scores Deducted in Correctness
Figure 3:Human evaluation of summary correctness. Mean deduction scores assigned
by headache specialists for three correctness-related criteria: (1) hallucinated or fabricated
content, (2) misinterpretation of the cited literature, and (3) fabricated citations. Higher
scores indicate larger penalties and therefore lower factual correctness, whereas lower scores
indicate greater faithfulness to the source literature. GPT-4o and Llama 3.1 received the
largest penalties for hallucinated or fabricated content, while fabricated citations were rel-
atively uncommon across all methods.
thesization of available information was described as “rather than quoting the available
data, this author seemed to be trying to help readers interpret the data as a clinician.” The
experts also noted that the types of errors in the summary were different between AI and ex-
perts. While some noted the ”lack of mistakes” and ”absence of errors and omission,” others
emphasized critical errors that human writers never do, including incorrectly referring ”to
itself as a review or a systematic review.” Other features were reported by both participants
who correctly identified the summaries as expert-written and those who did not. Several
factors were mentioned more often in correct responses, including completeness (n=8 vs
n=4), flow (n=5 vs n=3), focus (n=10 vs n=4), expert opinion/judgment (n=7 vs n=1),
human errors (n=3 vs n=1), reference quality (n=8 vs n=4), and readability/writing style
(n=3 vs n=1). However, conciseness was mentioned by 5 correct and 5 incorrect responses.
We additionally received 23 optional comments outside the rubric evaluation form and hu-
man identification. Features uniquely highlighted in these responses included the number of
references per argument (n=3), use of incorrect sources or clinically irrelevant information
(n=2), and authority of the reference source (n=2). Features overlapping with the rubric
included focus of information (n=6), reference quality (n=6), conciseness (n=5), availabil-
ity of English references (n=5), flow of information (n=4), synthesization (n=3), human
writing style (n=2), up-to-date information (n=1), and expert opinion/judgment (n=1).
Importantly, several participants noted that the most helpful summary was not always the
one they identified as expert-written. For example, in response to Q8, one participant wrote:
“Although I chose summary B [expert-written] as written by a person, I thought that sum-
mary D [GPT-4o generated] was perhaps more useful as it gave dosing ranges.” Discussion
9

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
0.30.62
0.220.480.7
0.320.76
0.420.50.740.92
0.44
00.20.40.60.81
(1) Lack of Important Concept (2) Incomplete Statement (3) Lack of Key Reference
Expert Sonnet GPT-4o Llama 3.1Mean Scores Deducted in Completeness
Figure 4:Human evaluation of summary completeness. Mean deduction scores assigned
by headache specialists for three completeness-related criteria: (1) omission of important
concepts, (2) incomplete statements, and (3) omission of key references. Higher scores
indicate larger penalties and therefore lower completeness, whereas lower scores indicate
more comprehensive summaries. Llama 3.1 and GPT-4o received the largest penalties for
missing important concepts, while Llama 3.1 was most frequently penalized for incomplete
statements.
We optimized an agentic chain-of-LLMs pipeline to efficiently summarize published medical
literature in response to user queries. We then rigorously evaluated summaries generated
by three LLMs (Sonnet-, GPT-4o-, and Llama 3.1) using this pipeline against those written
by ten experienced headache specialists. Evaluated through rubric metrics, human-written
summaries outperformed AI by a small margin, though not consistently across all subcate-
gories. Headache specialists correctly identified 64% of human-written summaries, and 52%
rated them as their top preference. Overall, Sonnet performed comparatively to human
written summaries and was ranked the second most favored summary, while GPT-4o and
Llama scored significantly lower for three rubric categories.
5. Discussion
We analyzed rubric and free text responses by 10 headache experts on LLM generated
summaries and compared the feedback among human-written summaries, Sonnet-, GPT-4o-
, and Llama 3.1- generated summaries. Overall, using a rubric, human-written summaries
outperformed AI by a small margin , though not consistently across all subcategories.
Sixty-four percent of responses correctly identified the human-written summaries, and 52%
rated them as their top preference. Different LLM models showed different strengths and
weaknesses. Overall, Sonnet performed comparatively to human written summaries and
was ranked the second most favored summary, while GPT-4o and Llama scored significantly
lower for three rubric categories.
10

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
5.1. Beyond metrics, free-text evaluation
While AI generated summaries are close to expert-generated summaries when using only
systematical evaluation metrics (i.e., correctness, completeness, conciseness, usefulness),
some experts mentioned the weakness of metrics (e.g., ”using the metrics is very time con-
suming and not always helpful,” ”the metrics only get you so far”). Thematic analyses of
free-text responses revealed further detailed information on what attributes are valued in
summaries: Some comments mentioned structural elements such as flow, focus, synthesiza-
tion, readability and writing style; while others also listed the quality of contents, including
expert opinion/judgment, reference quality, up to date information/full-text content, avail-
ability of English references, number of references, and credit/authority of reference source,
for example, the inclusion of milestone phase-3 randomized controlled trials that estab-
lished the efficacy of a treatment for headache disorders, or the International Classification
of Headache Disorders diagnostic criteria, which is the gold standard reference for the di-
agnosis of headache disorders. We will elaborate on each of these preferences below.
5.1.1. Structural elements: Flow, Focus, and Synthesization
Experts have preferences for specific kinds of summary structure: It was important to
have flow that is ”similar to how a clinician thinks about treatment options,” which can
be outlined as ”introducing the topic, then giving specifics, and concluding with a sum-
mary of how to initially evaluate such patients.” Starting from very basic contents such as
quotes from ”the ICHD criteria” and grouping treatment types ”by medical then surgical
management” were considered ideal. Furthermore, experts value specific clinically relevant
information. They appreciated ”useful dosing information,” ”key clinical concepts,” ”con-
troversies relevant to clinical practice,” and ”contraindications”. When the information was
”very superficial,” ”provided rare details and focused too much on diagnosis,” and skewed
”towards specific diagnosis,” summaries were not seen as useful. An emphasis on the right
perspective (e.g., ”I was looking for an APPROACH and not therapy,” ”one of the samples
was oddly surgery-heavy”) and providing ”balanced” information was important. Sum-
maries ”written by someone who could anticipate some of the questions that a reader might
have in clinic” were considered valuable. Synthesis, described as using ”inference to sum-
marize some of the citations” rather than ”quoting the available data” or ”’cut and pasted’
sentences from the references” appeared to be human-like and was valued. For example,
when ”citation says mesencephalon, summary says midbrain”, it was seen as evidence for
human writing, while ”just listing stuff rather than making recommendations” was seen as
LLM-like.
5.1.2. reference quality and authority
Overall, high-quality references were described as citations that include “a major meta-
analysis, showing it to be evidence-based, major guidelines for specialists and non-specialists.”
Additionally, it appeared to be important to have outstanding credit/authority of reference
source represented as “the most important individuals in the headache world involved in
the referenced material, and from great journals.” Besides quality, the number of references
also mattered. While one expert mentioned “for last sentence could have used any 1 of
the 3 refs given, why all 3?” another wrote “stylistically just not done to have paragraphs
11

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
1 and 2 be a series of facts from a single reference like this.” Moreover, it was considered
problematic when references “are unavailable and/or not in English.”
Further, as medicine is a rapidly growing area, ”up to date” information and “current
guidelines” were viewed as crucial. It is important to recognize that previously accepted
practices may become contraindicated as new evidence emerges. While human experts are
typically aware of such shifts, LLMs do not consistently verify whether previously retrieved
information has been updated. Importantly, summaries should include details ”drawn from
the body of the text” and should not ”borderline-plagiarize from the abstracts.”
5.1.3. Other factors
For controversial topics, expert opinions and judgment that sometimes ”did not have a
reference” also mattered. Those summaries went ”above and beyond in explaining more
than basic contraindications,” included ”clarifications” such as ”it is too premature to draw
any definitive conclusion,” and mentioned ”ongoing trials.” Furthermore, it is important not
to carry forward ”wrong information,” ”if the source was wrong,” and ”doses of drugs that
are not clinically relevant” (e.g., doses that were studied in clinical trials but not approved
for clinical use).
The comments on literature highlighted different types of errors expected in human
writing and LLM outputs. Human errors included ”extensive grammatical errors”, and ”a
missing word” (e.g., ”inhalation of 100%”, where it should say ”inhalation of 100% O2”),
and experts pointed out ”poor readability / choppy flow”. On the other hand, reviewers
mentioned “lack of mistakes” and “absence of errors and omissions” as features of AI-
suspected summaries. They identified several types of errors, including instances in which
the model incorrectly referred to its output as a review or systematic review.
6. Limitations
This study has several notable strengths, including extensive, detailed, and critical eval-
uations by headache specialists using standardized quantitative rubrics complemented by
qualitative free-text feedback, as well as a comprehensive comparison between high-quality
expert-written summaries and outputs from three distinct LLM models. There are several
limitations, including the relatively small sample size of 10 questions and 10 headache spe-
cialist evaluators, which might limit the ability to generalize the findings to other specialties
or to literature intended for non-professional audiences. However, due to the considerable
time and effort required for detailed annotation, coupled with the active clinical and re-
search commitments of the participating headache specialists, increasing the sample size
was not feasible. Furthermore, as the evaluated summaries were generated in early 2025,
subsequent advancements in LLMs may have led to improved performance.
7. Conclusion
We conducted a comprehensive study comparing literature summaries generated by RAG-
based LLMs within a pre-developed agentic chain-of-LLMs pipeline to human expert-written
summaries, with critical evaluation by headache specialists. While standard quantitative
12

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
metrics reveal only a small gap between human experts and LLMs, free-form expert eval-
uations uncover critical qualitative differences, particularly in the clinical relevance and
interpretive depth of the summaries, that LLMs often fail to capture. This highlights a sub-
stantial and previously uncharacterized gap between model-generated and human-authored
outputs. Our findings further identify expert-valued key features, beyond conventional
evaluation metrics, that can inform the future refinement of both human and AI-driven
literature summarization workflows.
Conflict of Interest
JHr owns stock in Amgen and Stryker, serves on Advisory Board for Pfizer, and receives
funding from Department of Health and Human Services (K23NS130143). PZ serves on
Advisory Board for Pfizer and as a consulttant for Acumen LLC. KI receives Postdoctoral
Fellowship Grant from the American Heart Association with funds paid to her institution
(https://doi.org/10.58275/AHA.25POST1378529.pc.gr.227396). CC has served as a consul-
tant for: Pfizer, AbbVie, Amneal, Satsuma, and eNeura. She receives research funds from
the American Heart Association, Pfizer, Lundbeck and the National Institute of Health
with funds paid to her institution. Within the prior 48 months, TJS has served as a con-
sultant with AbbVie, Lundbeck, and Salvia. He received royalties from UpToDate and
has stock options/equity interest in Allevalux and Nocira. His institution received research
grant funding on his behalf from AbbVie, American Heart Association, Flinn Foundation,
Henry Jackson Foundation, National Headache Foundation, National Institutes of Health,
Patient Centered Outcomes Research Institute, Pfizer, and the United States Department
of Defense. GD and FMC report no conflicting interests. Jennifer Stern is a shareholder in
Emmyon Inc, which has no relevance to any topic in this study
Acknowledgments
This work was supported in part by the Clinical Excellence Research Center at Stanford
Medicine and Technology and Digital Solutions at Stanford Healthcare. MW is supported
by an HAI Graduate Fellowship. AL is supported by an ARC Institute Graduate Fellowship.
KI is supported by the American Heart Association Postdoctoral Fellowship, with the funds
paid to Mayo Clinic.
References
[1] David L Sackett, William MC Rosenberg, JA Muir Gray, R Brian Haynes, and W Scott
Richardson. Evidence based medicine: what it is and what it isn’t, 1996.
[2] Ming Tai-Seale, Thomas G McGuire, and Weimin Zhang. Time allocation in primary
care office visits.Health services research, 42(5):1871–1894, 2007.
[3] Ethan Goh, Robert Gallo, Jason Hom, Eric Strong, Yingjie Weng, Hannah Kerman,
Jos´ ephine A Cool, Zahir Kanjee, Andrew S Parsons, Neera Ahuja, et al. Large language
model influence on diagnostic reasoning: a randomized clinical trial.JAMA network
open, 7(10):e2440969–e2440969, 2024.
13

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
[4] Jonah Zaretsky, Jeong Min Kim, Samuel Baskharoun, Yunan Zhao, Jonathan Aus-
trian, Yindalon Aphinyanaphongs, Ravi Gupta, Saul B Blecker, and Jonah Feldman.
Generative artificial intelligence to transform inpatient discharge summaries to patient-
friendly language and format.JAMA network open, 7(3):e240357–e240357, 2024.
[5] Scott L Fleming, Alejandro Lozano, William J Haberkorn, Jenelle A Jindal, Eduardo P
Reis, Rahul Thapa, Louis Blankemeier, Julian Z Genkins, Ethan Steinberg, Ashwin
Nayak, et al. Medalign: A clinician-generated dataset for instruction following with
electronic medical records.arXiv preprint arXiv:2308.14089, 2023.
[6] Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, and Nigam Shah. Clinfo.ai: An
open-source retrieval-augmented large language model system for answering medical
questions using scientific literature.Pac Symp Biocomput, 29:8–23, 2024.
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
Language models are few-shot learners.Advances in neural information processing
systems, 33:1877–1901, 2020.
[8] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess,
Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws
for neural language models.arXiv preprint arXiv:2001.08361, 2020.
[9] Tao Tu, Anil Palepu, Mike Schaekermann, Khaled Saab, Jan Freyberg, Ryutaro Tanno,
Amy Wang, Brenna Li, Mohamed Amin, Nenad Tomasev, et al. Towards conversational
diagnostic ai.arXiv preprint arXiv:2401.05654, 2024.
[10] Fatima N Mirza, Oliver Y Tang, Ian D Connolly, Hael A Abdulrazeq, Rachel K Lim,
G Dean Roye, Cedric Priebe, Cheryl Chandler, Tiffany J Libby, Michael W Groff,
et al. Using chatgpt to facilitate truly informed medical consent.NEJM AI, page
AIcs2300145, 2024.
[11] Michael Wornow, Alejandro Lozano, Dev Dash, Jenelle Jindal, Kenneth W Mahaffey,
and Nigam H Shah. Zero-shot clinical trial patient matching with llms.NEJM AI, 2
(1):AIcs2400360, 2025.
[12] Kevin Wu, Eric Wu, Rahul Thapa, Kevin Wei, Angela Zhang, Arvind Suresh, Jacque-
line J Tao, Min Woo Sun, Alejandro Lozano, and James Zou. Medcasereasoning:
Evaluating and learning diagnostic reasoning from clinical case reports.arXiv preprint
arXiv:2505.11733, 2025.
[13] Yongqin Xian, Christoph H Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot
learning—a comprehensive evaluation of the good, the bad and the ugly.IEEE trans-
actions on pattern analysis and machine intelligence, 41(9):2251–2265, 2018.
[14] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang,
Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on halluci-
nation in large language models: Principles, taxonomy, challenges, and open questions.
ACM Transactions on Information Systems, 43(2):1–55, 2025.
14

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
[15] Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R Dalal, Jennifer L Kim,
Michael Moor, Robyn Fong, Curran Phillips, Kevin Alexander, Euan Ashley, et al.
Almanac—retrieval-augmented language models for clinical medicine.Nejm ai, 1(2):
AIoa2300068, 2024.
[16] Alejandro Lozano, Scott L Fleming, Chia-Chun Chiang, and Nigam Shah. Clinfo. ai:
An open-source retrieval-augmented large language model system for answering medical
questions using scientific literature. InPACIFIC SYMPOSIUM ON BIOCOMPUTING
2024, pages 8–23. World Scientific, 2023.
[17] Jae-woo Lee, In-Sang Yoo, Ji-Hye Kim, Won Tae Kim, Hyun Jeong Jeon, Hyo-Sun Yoo,
Jae Gwang Shin, Geun-Hyeong Kim, ShinJi Hwang, Seung Park, et al. Development
of ai-generated medical responses using the chatgpt for cancer patients.Computer
methods and programs in biomedicinƒe, 254:108302, 2024.
[18] Dawei Li, Shu Yang, Zhen Tan, Jae Young Baik, Sukwon Yun, Joseph Lee, Aaron
Chacko, Bojian Hou, Duy Duong-Tran, Ying Ding, et al. Dalk: Dynamic co-
augmentation of llms and kg to answer alzheimer’s disease questions with scientific
literature.arXiv preprint arXiv:2405.04819, 2024.
[19] Christopher Polzak, Alejandro Lozano, Min Woo Sun, James Burgess, Yuhui Zhang,
Kevin Wu, and Serena Yeung-Levy. Can large language models match the conclusions
of systematic reviews?arXiv preprint arXiv:2505.22787, 2025.
[20] Liyan Tang, Zhaoyi Sun, Betina Idnay, Jordan G Nestor, Ali Soroush, Pierre A Elias,
Ziyang Xu, Ying Ding, Greg Durrett, Justin F Rousseau, et al. Evaluating large
language models on medical evidence summarization.NPJ digital medicine, 6(1):158,
2023.
[21] Ming-Jie Luo, Jianyu Pang, Shaowei Bi, Yunxi Lai, Jiaman Zhao, Yuanrui Shang,
Tingxin Cui, Yahan Yang, Zhenzhe Lin, Lanqin Zhao, et al. Development and eval-
uation of a retrieval-augmented large language model framework for ophthalmology.
JAMA ophthalmology, 142(9):798–805, 2024.
[22] Yu He Ke, Liyuan Jin, Kabilan Elangovan, Hairil Rizal Abdullah, Nan Liu, Alex
Tiong Heng Sia, Chai Rick Soh, Joshua Yi Min Tung, Jasmine Chiat Ling Ong, Chang-
Fu Kuo, et al. Retrieval augmented generation for 10 large language models and its
generalizability in assessing medical fitness.npj Digital Medicine, 8(1):187, 2025.
[23] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anad-
kat, et al. Gpt-4 technical report.arXiv preprint arXiv:2303.08774, 2023.
[24] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient
context window extension of large language models.arXiv preprint arXiv:2309.00071,
2023.
[25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao
Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for
15

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
large language model serving with pagedattention. InProceedings of the ACM SIGOPS
29th Symposium on Operating Systems Principles, 2023.
16

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Email for participation
I’m writing to see if you would be interested in participating in a new AI research
study I’m conducting.
This study is about using Large Language Models (LLMs) to summarize literature
published in PubMed and will involve 10 headache specialists.
In the first phase, each headache specialist will be invited to write a clinical question
and an answer to that question. The answer will be around 200–300 words, in a
review article format with citations — imagine you are writing a brief review for
UpToDateor for a reputed journal. The readers would be clinicians and providers,
and the information is to be used to guide their clinical practice.
In the second phase, you will be asked to evaluate the responses/answers written by
9 other headache specialists as well as AI-generated summaries from 4 different LLM
systems. You will be blinded to whether the response was written by a human expert
or AI, and we will ask you to evaluate whether each summary is correct and useful,
and your preference/reasoning of preference for those summaries.
If you are interested in participating, we will send out additional details and instruc-
tions. If for any reason you are not able to, I’d totally understand! Please keep this
confidential.
Thank you for considering this request,
Detailed instruction on writing the paragraph
Thank you again for agreeing to participate in the Large Language Model (LLM)
Literature Summary for Headache Medicine Project. The goal of this project is to
evaluate AI-generated summaries vs human expert-written summaries. Through this
project, we want to identify the key parameters that human experts and readers value
when reading literature summaries, which would benefit future efforts using AI for
summarization and evaluation of LLM-generated outputs.
As a brief outline of this project, I aim to involve 10 headache specialists. In the
first phase, each headache specialist will be invited to write one clinical question
and an answer to that question. The answer (summary) should be 200-300 words,
in a review format with around 10 citations/ references- imagine you are writing
a brief review for UpToDate or a reputed journal, and what you write should be
useful for clinicians, providers, trainees, and medical students to guide their clinical
practice and medical education. As a reminder, your written summary, along with AI-
generated summaries, will be judged by other colleagues for correctness, usefulness,
and preference, so please refrain from writing opinions that might be too controversial.
Also, please do NOT use ChatGPT or any other commercially available AI literature
summarization website during the writing process. Using Grammarly for grammar
check, if needed, is allowed, but please do not put your ideas, sentences, or written
summary into ChatGPT or other LLMs.
17

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
In the second phase, you will be asked to evaluate the responses/answers written by 9
other headache specialists and AI-generated summaries from different LLM systems.
You will be blinded to whether the response was written by a human or AI, and
we will ask you to evaluate whether each summary is correct and useful, and your
preference for those summaries, and the reason why. For Step 1, I’d appreciate if you
could write an answer to the following question: - “What are the treatment options
for cluster headache?” Please feel free to rephrase the question or change it to another
topic that might be more interesting to you. Please just let me know ASAP if you’d
like to change the proposed question, as we would need to use the same wording of
these questions for different AI systems to generate a summary.
Please let me know if the question wording is acceptable. I’d appreciate if you could
send me the 200-300 word summary with citations before [date].
Thank you very much again for your help and please let me know if you have any ques-
tions! If for any reason you are no longer able to participate, I’d totally understand.
Please let me know if that’s the case.
Instruction for Evaluation
Thank you so much for agreeing to participate in this project and providing an ex-
cellent summary of the question assigned to you!
After multiple rounds of optimization, we have produced outputs from different LLM
systems. Now, we are ready to move forward with the second phase- evaluation.
This folder includes all 10 questions and summaries. You will see four summaries
for each question. One summary was written by a human headache specialist, and
three were generated by different AI/LLM systems. They were shuffled to appear in
random order.
[link]
We would appreciate your help in evaluating the summaries of the following 5 ques-
tions assigned to you, highlighted in Yellow below (You only need to fill out the Google
Evaluation Form below for these 5 questions and the corresponding summaries)
The evaluation form is here [link to evaluation form]
Please read through the detailed instructions regarding the evaluation as outlined in
the Google Form.
Please select the question first, and fill out your evaluations accordingly.
I’d suggest opening one window (screen) for the PDF summary, and another win-
dow (screen) to fill out the evaluation form. Or, if you’d prefer, print out the PDF
summaries for the 5 questions assigned to you. The estimated time to complete this
evaluation for 5 questions is approximately 5 hours.
We would be very grateful if you could complete this before [date]. Please let me
know if you have any questions or concerns. Thank you again for your contribution
to this project!
18

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Topics Correct Responses Incorrect Responses
Correctness - The content was the most correct. - There
is some failure to provide citations, but overall,
the information provided is correct. - Summary
D met all the metrics.
Completeness - Summary is well rounded.
- It seems to be the most comprehensive, con-
sidering all aspects of diagnosing and treating
cervicogenic headache. - It was also the most
complete (other summaries left out several con-
traindications). - Discussed pregnancy in detail;
mentions all the gepants and mAbs. - Reason-
ably complete. - AI did very well on two of them.
- Summary D met all the metrics.- It was somewhat close but b and c were similar.
But b was more complete than c. Also d lacked
significant content and was out of date up to
2019. - d also included pediatric information. -
Comprehensive of all the reviews. - Has lots of
details not shown in others.
Conciseness - No redundant material. - Choice B is close, but
it is very long and redundant. - Concise and
beautifully written with things I didn’t know
about. - Of good length and the one I liked
the most. - This summary does not have a lot
of repetitive padding.- It is concise and well written. - It was a hard
choice between B and C. B was chosen for con-
ciseness. - d I think, but a is close second; sum-
mary d was best in content, though a little long.
- I think it is the most concise. - Long.
Flow - Nice flow that mimicked a natural reading /
learning style. - Best flow of ideas and medi-
cally accurate paragraph on medical therapies. -
Grouped by medical then surgical management.
- Focused on helpful aspects for clinicians. -
Logical introduction and conclusion.- Summary a follows a more logical flow, similar
to how a clinician thinks. - Best job introducing
the topic, specifics, and conclusion. - There is
a beginning, middle, and end. Logical progres-
sion.
Focus - Useful dosing information. - Included key clini-
cal concepts. - Differentiates CGRP monoclonal
antibodies vs small molecule antagonists. - Dis-
cusses rare side effects. - Addresses controver-
sies relevant to clinical practice. - Summary d
focuses more on diagnosis.- Choice of data from specific studies instead of
summarizing. - Includes multiple aspects rel-
evant to clinical practice. - Good balance be-
tween citing data and clinical implications. -
Answers the prompt better than others.
Expert Opinion /
Judgement- Included medications though one lacked a ref-
erence. - References missing for statements that
sound like expert opinion. - Provides extra con-
text such as serotonin syndrome explanation.
- Goes above and beyond in explaining con-
traindications. - Includes clinical tips like pri-
oritizing life-threatening disorders. - Mentions
ongoing trials and updated guidelines.- Emphasized identifying emergent causes; felt
more clinical judgment.
AI Mistakes - Lack of mistakes. - Absence of errors and omis-
sions. - Avoids phrases like “this systematic re-
view includes.” - Does not incorrectly refer to
itself as a review.
Human Errors - Extensive grammatical errors. - A few cap-
italization and grammatical choices that seem
human. - Missing word example: “inhalation of
100% O2.”- “A” was well-written but missed part of the
question—more typical of human error.
Reference Quality - High quality references cited. - Cites guideline
and practice parameter. - Multiple references
per fact. - Used authoritative and EBM-level
sources.- Did not mention systematic review. - Nice use
of references. - “a” lacked key reference used by
others. - References were great.
Synthesization - Interprets data as a clinician rather than quot-
ing it. - Requires inference to summarize ci-
tations, showing conceptual understanding. -
Writing follows a unified conceptual framework.
Up to Date / Full-
text Content- Written like a clinician and up to date. - More
current than others. - Draws ideas from body of
text, not just abstracts.
Readability /
Writing Style- Readability was very natural. - Reflexive lan-
guage suggests a human writer. - Writing style
feels personal.- Simple to follow writing style.
No Comment - d - Had difficulty identifying human summary, but
this was best estimate. - Summary c.
Table 1:Qualitative feedback across evaluation topics comparing correct and incorrect
responses. 19

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Topics Comments
Synthesization 1) Just listing stuff rather than making recommendations.
3) See my write.
- AI generated summaries seemed to have “cut and pasted” sentences from the references as
opposed to a formulation of the information. - Makes sense; metrics only get you so far.
Availability of English
References- Using references not in English and/or not easily available. - Cannot assess use of first
3 references completely as they are unavailable or not in English. - All references were
legitimate, even reference 1 in Case B which needed the DOI to find it online in PubMed.
- Summary C uses a non-English-language-only paper where there are good alternatives —
possibly indicative of AI. - Summary A has a foreign language paper. - Metrics helped her
but some reference PDFs were missing; had to go to PubMed to find them.
Reference Quality - Using weird/old references (e.g., ref 10 is an odd choice focused on three-peat GKS). - B
and D listed systematic reviews. - The current metrics don’t capture reference quality, but it
was a distinguishing characteristic. - Good references, including a major meta-analysis and
guidelines for specialists and non-specialists. - Excellent authorship and inclusion of imaging
references.
Focus - Balanced; one sample was oddly surgery-heavy. - Passage D full of clinically relevant tips,
such as specific triptan considerations and controversies (e.g., SSRIs, hemiplegic migraine). -
AI-generated ones skew toward diagnosis (strong diagnostic bias). - Best summary for broad
audience—clinicians, trainees, students—covering diagnosis and management. - Metrics not
necessarily helpful, but clinically it spoke to how to make diagnosis and treatment decisions.
- Selection of information not guided by human judgment; summary B seemed bland and
non-specific. - I was looking for an approach, not therapy; otherwise summary B might have
been best.
Conciseness - Not repetitive; directness. - Introduction and conclusion paragraphs often seem formulaic,
redundant, and unhelpful. - Concise sentences; great cases so far, but unclear if metrics
influenced length. - It is concise and well written.
Up to Date Informa-
tion- Citing current guidelines.
Number of References - For the second to last sentence could have used ref 5 alone; for the last, any 1 of 3 refs
would suffice. - Not sure why multiple redundant references used (e.g., MVD as gold, refs
5–8). - Unsure why several citations are clustered for single statements. - Should use fewer
references overall. - Summary C uses only 3 references—odd stylistically for a human—but
includes Cheema et al.; Farb paper citation unusual.
Flow - Layout. - Flow of information and logical progression needed. - Metrics not that helpful;
organization could improve.
Human Writing Style - Summary C had the best quality information but poor readability and choppy flow. - The
writing style of D feels human.
Information Noise:
Incorrect or Clinically
Irrelevant Sources- AI summaries quoted sources accurately but sometimes carried forward incorrect informa-
tion from poor sources; limited source discernment. - Summaries B and C included clinically
irrelevant drug doses (e.g., atogepant 120 mg, eptinezumab 30 mg).
AI Clich´ e - A and D read like AI-generated responses. - Example: phrases like “This systematic review
aims to summarize the evidence-based options. . . ” are typical of LLMs. - Two responses
use language such as “this review” or “this systematic review,” which was not requested. -
Noticeable clich´ es in AI-generated answers.
Credit / Authority of
Reference Source- Pattern recognition; can be done quickly but one must verify data. - By and large good
references, but the human one chosen had the most authoritative headache experts and top
journals. - Ref 6 not very authoritative—likely AI.
Expert Opinion /
Judgement- Answer A very convincing and detailed; seemed human until reading D, which included
reflection and judgment.
Others - Whether it was useful and appealing mattered more. - Metrics are time-consuming and
not always helpful. - Allowed a broad interpretation of “evidence-based” to include narrative
reviews and clinical experience, aligning with modern EBM definitions. - Broader input of
assessments may be more valuable.
Table 2:Expert comments grouped by qualitative topic for AI vs human summary evalu-
ation.
20

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Table 3:Correctness: ANOVA and t-Test Results
SUMMARY Count Sum Average Variance
Human 50 457 9.14 2.61
Sonnet 50 456 9.12 1.82
GPT-4o 50 408 8.16 4.83
Llama 3.1 50 428 8.56 2.01
ANOVA SS df MS F P-value F crit
Between Groups 33.655 3 11.218 3.981 0.009 2.651
Within Groups 552.340 196 2.818
Total 585.995 199
t-Test: Human vs. Sonnet
Human Sonnet
Mean 9.14 9.12
Variance 2.61 1.82
Observations 50 50
Pooled Variance 2.22
df 98
t Stat 0.07
P(T≤t) one-tail 0.47
t Critical one-tail 1.66
P(T≤t) two-tail 0.95
t Critical two-tail 1.98
t-Test: Human vs. GPT-4o
Mean 9.14 8.16
Variance 2.61 4.83
Observations 50 50
Pooled Variance 3.72
df 98
t Stat 2.54
P(T≤t) one-tail 0.01
t Critical one-tail 1.66
P(T≤t) two-tail 0.01
t Critical two-tail 1.98
t-Test: Human vs. Llama 3.1
Mean 9.14 8.56
Variance 2.61 2.01
Observations 50 50
Pooled Variance 2.31
df 98
t Stat 1.91
P(T≤t) one-tail 0.03
t Critical one-tail 1.66
P(T≤t) two-tail 0.06
t Critical two-tail 1.98
21

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Table 4:Completeness: ANOVA and t-Test Results
SUMMARY Count Sum Average Variance
Human 50 443 8.86 1.35
Sonnet 50 420 8.40 3.18
GPT-4o 50 413 8.26 2.11
Llama 3.1 50 393 7.86 2.82
ANOVA SS df MS F P-value F crit
Between Groups 25.54 3 8.51 3.60 0.015 2.65
Within Groups 463.66 196 2.37
Total 489.20 199
t-Test: Human vs. Sonnet
Human Sonnet
Mean 8.86 8.40
Variance 1.35 3.18
Observations 50 50
Pooled Variance 2.27
df 98
t Stat 1.53
P(T≤t) one-tail 0.06
t Critical one-tail 1.66
P(T≤t) two-tail 0.13
t Critical two-tail 1.98
t-Test: Human vs. GPT-4o
Mean 8.86 8.26
Variance 1.35 2.11
Observations 50 50
Pooled Variance 1.73
df 98
t Stat 2.28
P(T≤t) one-tail 0.01
t Critical one-tail 1.66
P(T≤t) two-tail 0.02
t Critical two-tail 1.98
t-Test: Human vs. Llama 3.1
Mean 8.86 7.86
Variance 1.35 2.82
Observations 50 50
Pooled Variance 2.08
df 98
t Stat 3.47
P(T≤t) one-tail 0.00
t Critical one-tail 1.66
P(T≤t) two-tail 0.00
t Critical two-tail 1.98
22

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Table 5:Conciseness: ANOVA and t-Test Results
SUMMARY Count Sum Average Variance
Human 50 476 9.52 2.30
Sonnet 50 476 9.52 0.62
GPT-4o 50 438 8.76 1.70
Llama 3.1 50 424 8.48 1.81
ANOVA SS df MS F P-value F crit
Between Groups 42.46 3 14.15 8.82 0.000016 2.65
Within Groups 314.56 196 1.60
Total 357.02 199
t-Test: Human vs. Sonnet
Human Sonnet
Mean 9.52 9.52
Variance 2.30 0.62
Observations 50 50
Pooled Variance 1.46
df 98
t Stat 0.00
P(T≤t) one-tail 0.50
t Critical one-tail 1.66
P(T≤t) two-tail 1.00
t Critical two-tail 1.98
t-Test: Human vs. GPT-4o
Mean 9.52 8.76
Variance 2.30 1.70
Observations 50 50
Pooled Variance 2.00
df 98
t Stat 2.69
P(T≤t) one-tail 0.00
t Critical one-tail 1.66
P(T≤t) two-tail 0.01
t Critical two-tail 1.98
t-Test: Human vs. Llama 3.1
Mean 9.52 8.48
Variance 2.30 1.81
Observations 50 50
Pooled Variance 2.05
df 98
t Stat 3.63
P(T≤t) one-tail 0.00
t Critical one-tail 1.66
P(T≤t) two-tail 0.00
t Critical two-tail 1.98
23

Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization
Table 6:Usefulness: ANOVA and t-Test Results
SUMMARY Count Sum Average Variance
Human 50 407 8.14 3.96
Sonnet 50 367 7.34 4.64
GPT-4o 50 330 6.60 5.18
Llama 3.1 50 306 6.12 5.01
ANOVA SS df MS F P-value F crit
Between Groups 116.98 3 38.99 8.30 3.17E-05 2.65
Within Groups 920.52 196 4.70
Total 1037.50 199
t-Test: Human vs. Sonnet
Human Sonnet
Mean 8.14 7.34
Variance 3.96 4.64
Observations 50 50
Pooled Variance 4.30
df 98
t Stat 1.93
P(T≤t) one-tail 0.03
t Critical one-tail 1.66
P(T≤t) two-tail 0.06
t Critical two-tail 1.98
t-Test: Human vs. GPT-4o
Mean 8.14 6.60
Variance 3.96 5.18
Observations 50 50
Pooled Variance 4.57
df 98
t Stat 3.60
P(T≤t) one-tail 0.00
t Critical one-tail 1.66
P(T≤t) two-tail 0.00
t Critical two-tail 1.98
t-Test: Human vs. Llama 3.1
Mean 8.14 6.12
Variance 3.96 5.01
Observations 50 50
Pooled Variance 4.48
df 98
t Stat 4.77
P(T≤t) one-tail 0.00
t Critical one-tail 1.66
P(T≤t) two-tail 0.00
t Critical two-tail 1.98
24
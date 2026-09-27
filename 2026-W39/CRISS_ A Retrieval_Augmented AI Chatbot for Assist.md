# CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars

**Authors**: Vani Seth, Mohammad Beheshti, Anirudh Kambhampati, Vishwa Bhayani, Lucinda Ham, Prasad Calyam, Iris Zachary

**Published**: 2026-09-24 06:05:11

**PDF URL**: [https://arxiv.org/pdf/2609.29075v1](https://arxiv.org/pdf/2609.29075v1)

## Abstract
Cancer registrars, including Oncology Data Specialists (ODSs), must interpret complex and frequently updated coding and staging standards. We developed CRISS (Cancer Registry Intelligent Support System), a retrieval-augmented generation (RAG) conversational assistant that provides rapid, citation-supported access to registry guidance. This study evaluated whether CRISS could (1) support accurate and citation-supported responses, (2) improve access to and interpretation of relevant guidance, and (3) support training/helpdesk use while preserving human oversight of final abstraction decisions. We built a domain-specific knowledge base from national cancer registry standards, segmented into metadata-tagged passages and indexed as dense embeddings. Retrieved passages were used to generate citation-grounded responses through a large language model (LLM). Open-weight, proprietary, and non-RAG baseline models across Gemini and GPT families were evaluated on easy, medium, and hard registry questions using an LLM-as-a-Judge protocols. RAG configurations consistently outperformed non-RAG approaches, especially as question difficulty increased. Mean grounding scores for RAG were 0.62/0.56/0.59 across easy/medium/hard tiers versus 0.29/0.26/0.29 for non-RAG. RAG models also achieved higher semantic-similarity scores overall. Proprietary RAG models performed strongest on easy and medium questions, while local RAG models ranked highest on hard questions and proprietary models were generally more cautious. Domain-specific RAG improved evidence grounding and response quality for cancer registry questions while enabling citation-supported assistance across complexity levels. CRISS demonstrates the potential of human-centered, citation-grounded AI to support cancer registrars while preserving human oversight for final coding decisions.

## Full Text


<!-- PDF content starts -->

CRISS: A RETRIEVAL-AUGMENTEDAI CHATBOT FOR
ASSISTINGCANCERREGISTRARS
Vani Seth1,a,
Mohammad Beheshti2,3,b,
Anirudh Kambhampati2,3,c,
Vishwa Bhayani2,3,d,Lucinda Ham2,e,
Prasad Calyam1,f,
Iris Zachary2,3,g,*
1Department of Electrical Engineering and Computer Science, University of Missouri, Columbia, USA
2Missouri Cancer Registry and Research Center, Department of Public Health,
College of Health Sciences, University of Missouri, Columbia, USA
3MU Institute for Data Science and Informatics, University of Missouri, Columbia, USA
avs4ky@missouri.edubmbwnh@missouri.educakwg7@missouri.edu
dvbbn32@missouri.eduelahf5p@missouri.edufcalyamp@missouri.edu
gzacharyi@missouri.edu
*Corresponding author.
ABSTRACT
Objectives.Cancer registrars, including credentialed Oncology Data Specialists (ODSs), must
interpret complex and frequently updated cancer registry coding and staging guidelines and standards
to abstract accurate data. We developed and evaluatedCRISS (Cancer Registry Intelligent Support
System), a retrieval-augmented generation (RAG) conversational assistant designed to provide regis-
trars with rapid, source access to relevant guidance. This study evaluated whether CRISS could (1)
support accurate, citation-supported responses to registry queries of varying complexity, (2) improve
access to and interpretation of relevant guidance, and (3) support training and helpdesk functions
while preserving human oversight of final abstraction decisions.
Materials and Methods.We constructed a domain-specific knowledge base from national standards
and cancer registry guidelines, segmented into metadata-tagged passages and indexed as dense vector
embeddings. User queries were answered through semantic retrieval of relevant passages followed
by citation-grounded response generation with a large language model (LLM), delivered through
a React front end and a Flask backend on AWS EC2 with semantic caching. We benchmarked
comprising open-weight (e.g., Gemma, Gpt-oss, Qwen), proprietary, and baseline options across the
Gemini and GPT families against an easy, medium, and hard registry question set using an LLM-as-a-
judge protocol that scored responses on six dimensions (Correctness, Faithfulness, Completeness,
Reasoning, Citation Accuracy, Tone) and flagged four failure modes (Hallucination, Clinical Safety,
Multi-Part Miss, Rule Conflict). Inference-time retrieval metrics (relevance, faithfulness, hop count,
context length, confidence, source year) were logged for every query.
Results.RAG configurations showed a clear advantage over non-RAG approaches, particularly
as question difficulty increased. Mean grounding scores for RAG were 0.62/0.56/0.59 across
easy/medium/hard tiers, compared with 0.29/0.26/0.29 for non-RAG, while RAG models also
achieved higher semantic-similarity scores overall. In the LLM-as-a-Judge evaluation, proprietary
RAG models performed the strongest on easy and medium questions, whereas RAG Local models
ranked highest on the hard tier. Follow-up analysis showed that proprietary models were more
cautious than local models while answering the questions.
Discussion and Conclusion.The results show that domain-specific RAG substantially improves
evidence grounding and response quality for cancer registry questions and supported citation-based
responses across varying levels of complexity, with the greatest benefit on more complex cases.
Overall, CRISS demonstrates the value of citation-grounded human-centered AI for supporting
cancer registrars while preserving human oversight for final coding decisions.
arXiv:2609.29075v1  [cs.AI]  24 Sep 2026

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Keywordscancer registry, retrieval-augmented generation, large language models, conversational AI, clinical
informatics, oncology data specialists, medical question answering, AI safety, clinical decision support
1 Introduction
Cancer registries are essential to public health surveillance, clinical research, and healthcare planning. By systematically
collecting, managing, and analyzing data on cancer incidence, treatment, and survival, registries enable epidemiologic
studies, inform policy decisions, guide resource allocation, and support quality improvement initiatives [ 1,2]. Hospital-
based registries collect detailed information on cancer patients diagnosed and treated within individual institutions to
support patient care evaluation and accreditation activities. These cases are subsequently reported to population-based
central cancer registries, which aggregate data from multiple facilities within defined geographic areas to produce
regional and national statistics that inform surveillance, research, and public health planning [3, 4].
The accuracy, timeliness, and completeness of registry data determine the reliability of cancer surveillance [ 5].
These data depend not only on information systems and reporting infrastructure, but also on the expertise of the
professionals responsible for interpreting and coding complex clinical information. Oncology Data Specialists (ODSs)
abstract diagnostic, staging, treatment, and follow-up information from medical records into standardized registry
data formats [ 6]. This work extends well beyond data entry and requires advanced interpretive reasoning to reconcile
ambiguous documentation, integrate information from multiple sources, and apply complex and evolving coding
standards [ 7]. Importantly, prior work in cancer registry information management and informatics emphasizes that
effective data collection depends on the ability of registrars to interpret complex clinical information, align data
abstraction with research and surveillance needs, and navigate evolving data standards, highlighting the cognitive and
knowledge-driven nature of registry work [ 8]. ODS routinely use national standards and guidelines from organizations
such as the North American Association of Central Cancer Registries (NAACCR), the American Joint Committee
on Cancer (AJCC), and the Commission on Cancer (CoC) [ 9,10]. Since relevant guidance is spread across multiple
national standard manuals, coding rules, and site-specific guidelines that are periodically revised, resolving a single
case often requires consulting multiple references. This knowledge-intensive process can be particularly challenging for
newly trained professionals and even trained ODSs as well [11].
Recent advances in artificial intelligence, particularly Large Language Model (LLM)–based chatbots such as ChatGPT,
have demonstrated strong capabilities in natural language understanding, question answering, and knowledge synthesis
in healthcare [ 12–14]. Prior work has explored trustworthy and secure conversational AI for sensitive healthcare
applications, highlighting the importance of reliability, privacy, and safe interaction when LLM-based systems are
used to support users in high-stakes domains [ 15–17]. These capabilities have motivated substantial research into
applying machine learning and NLP to automate clinical information extraction and coding tasks, including the
assignment of diagnostic classifications such as the World Health Organization’s International Classification of Diseases
(ICD) [ 18]. However, real-world deployment of fully automated systems remains challenging due to variability in
medical documentation, the prevalence of rare edge cases, and the potential for even small error rates to produce
significant downstream consequences in large datasets [19, 20].
Moreover, LLMs are prone to factual inaccuracies, outdated knowledge, and hallucinated responses, limiting their
suitability for unsupervised decision making in critical domains [ 21]. Retrieval-Augmented Generation (RAG) has
emerged as a promising approach to address these reliability concerns by grounding model responses in national
standards and guideline source documents retrieved at query time [ 22]. By combining information retrieval with
generative reasoning, RAG systems can reduce hallucinations, enhance factual accuracy, and provide traceable evidence
for their outputs [23].
Despite this potential, existing literature reveals a significant gap: current AI-driven approaches overwhelmingly
prioritize automation over support, treating registry expertise as a process to be replicated rather than a competency
to be augmented [ 24–27]. This gap is particularly apparent in retrieval-augmented approaches applied specifically to
cancer registry coding, which have similarly centered on automated code generation rather than interactive registrar
support and have reported hallucinations and factual inconsistencies even with retrieval grounding in place [ 28]. Across
this literature, systems are typically evaluated on output accuracy alone rather than their capacity to enhance user
understanding or reinforce guideline comprehension, leaving interactive systems that support registrars’ own reasoning
largely unexplored.
In contrast, cancer registry practice requires nuanced interpretation of evolving guidelines, cross-referencing multiple
national standards and guidance resources, and careful human judgment across the full abstraction process. To address
this need, this work presents a complementary conversational assistant rather than an autonomous coding tool, enabling
registrars to ask questions as they arise during abstraction, synthesize rules across documents, and receive explanations
supported by explicit citations. By prioritizing transparency, traceability, and human oversight, this paradigm shifts the
2

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
role of AI from task automation to interactive knowledge support, addressing the practical needs of registry operations
where accuracy and accountability are critical.
This study aims to develop and evaluateCRISS (Cancer Registry Intelligent Support System), a retrieval-augmented
conversational assistant designed to support cancer registrars in navigating complex guidelines and resolving registry-
related questions. Specifically, the goals are to: (1) determine whether a retrieval-augmented conversational system
grounded in national standards and guidelines can provide accurate, citation-supported guidance for real-world queries;
(2) assess its potential to improve efficiency in locating and interpreting relevant guidelines; and (3) evaluate its
usefulness as a practical support tool for both experienced registrars and those in training while preserving human
oversight of final decisions. By reframing AI as an educational adjunct embedded within professional workflows, this
study addresses a previously underexplored area of cancer registry informatics: the augmentation of human expertise
rather than its replacement.
The remainder of this paper is organized as follows. Section 2 describes the CRISS system architecture. Section 3
presents the evaluation design. Section 4 discusses the experimental results. Section 5 discusses the findings and
limitations. Section 6 concludes the paper.
2 Materials and Methods
This study proposesCRISS, a retrieval-augmented conversational system designed to support cancer registrars in
navigating, interpreting, and applying complex cancer registry coding guidelines. Registry abstraction requires
registrars to interpret clinical documentation, cross-reference multiple national standards and guidance manuals, and
apply coding, staging, and reporting standards that are periodically revised and expanded. Rather than functioning as an
autonomous coding engine, the system is designed as a human-centered knowledge and educational support tool that
assists registrars in locating, interpreting, and applying national standards and guidance, while final coding decisions
remain the responsibility of the human user.
Figure 1: System architecture of CRISS comprising query understanding, multi-hop retrieval and reasoning over a
cancer registry knowledge base, and LLM-assisted answer generation with confidence estimation and feedback.
The system adopts a Retrieval-Augmented Generation (RAG) architecture that combines semantic information retrieval
with the reasoning capabilities of a Large Language Model (LLM), as illustrated inFigure 1. National standards and
cancer registry guidelines are converted into a searchable knowledge base. When a registrar submits a natural-language
question, the system retrieves passages that are semantically relevant to the query and supplies them to the LLM as
supporting context, which then generates a response grounded in that retrieved evidence. This design addresses a key
limitation of general-purpose LLM chatbots: their tendency to produce fluent but unsupported or outdated responses. It
also allows the knowledge base to be updated independently as coding standards evolve.
The system is guided by three design principles.First, every response should be grounded in national standards and
cancer registry guideline documentation rather than the model’s parametric knowledge, so users can verify the origin
3

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
of the generated guidance.Second, the system prioritizes explanation alongside answer generation, relating retrieved
evidence to the user’s question rather than producing an isolated recommendation, so it supports both experienced
registrars and trainees.Third, the system preserves human oversight. It is intended to supplement, rather than replace,
registrar expertise by providing evidence, interpretation, and educational support while leaving final coding decisions to
the user.
The remainder of this section describes how these principles are implemented across the system’s three components:
construction of a registry-specific knowledge base, semantic retrieval of relevant guidelines, and citation-grounded
response generation. It then describes the methodology used to evaluate the resulting system.
2.1 Construction of a Registry-Specific Knowledge Base
The knowledge base was built from national standards and cancer registry guidelines that registrars routinely consult
during case abstraction, including cancer coding manuals, Solid Tumor Rules, Multiple Primary and Histology Coding
Rules (MP/H) Rules, staging manuals, histology references, and site-specific coding instructions [ 29–32]. Because these
documents are distributed across multiple sources and written primarily for human reference, they were preprocessed
into a structured representation suitable for semantic search.
Each source document was parsed to extract its textual content while preserving structural elements such as titles,
section headings, rule identifiers, page numbers, tables, and explanatory notes. This context was preserved because
many coding decisions depend not only on the wording of an individual rule but also on its surrounding exceptions
and document hierarchy; retaining this metadata also allows the system to cite the original source during response
generation.
The extracted text is then segmented into overlapping chunks of 1,500 characters with an overlap of 200 characters, a
granularity chosen to preserve coherent rule-and-exception units without exceeding the LLM’s context limits. Each
resulting chunk is paired with descriptive metadata, including the originating document, section title, page reference,
and any available rule identifiers, which is later used to disambiguate conceptually similar passages during retrieval and
to support source citation during response generation. Full details of the chunking parameters, the underlying rationale,
and the metadata schema are provided inAppendix B.
2.1.1 Semantic Retrieval of Relevant Guidelines
Each text chunk is transformed into a dense semantic representation using the embedding model associated with
the selected LLM provider (OpenAI: text-embedding-3-large; Google Gemini: gemini-embedding-001) [ 33,34],
with OpenAI embeddings used for the open-source model configurations, following prior work demonstrating the
effectiveness of dense embedding models for semantic retrieval in medical RAG systems [ 35]. The system was designed
modularly to support different embedding models and vector-store backends such as csv, chroma, or in-memory,
allowing the retrieval configuration to be adapted to the model and deployment environment. Unlike lexical retrieval,
semantic embeddings capture contextual meaning, enabling conceptually similar passages to be retrieved even when
clinical questions and official coding manuals use different terminology.
When a registrar submits a question, the same embedding model encodes the query and retrieves the Top-k most relevant
passages (k=6 by default; configurable by model). For broad or multi-faceted questions, the system can generate
alternative focused search formulations to better capture the individual information needs contained in the original
question. Retrieved candidates are additionally reranked to prioritize the most recent applicable registry guidance and
de-duplicated before being added to the context.
Retrieval proceeds through an iterative multi-hop process when a single search is insufficient. After each hop, the
system assesses whether the accumulated passages adequately answer the original question. If information is missing,
the LLM generates a more focused follow-up query based on the evidence retrieved so far, and another retrieval hop is
performed. This process continues until sufficient evidence is obtained or the maximum number of hops is reached.
Separating this retrieval layer from the answer generation also allows registry manuals, embedding models, and vector
stores to be updated or exchanged without retraining or modifying the overall system.
2.2 Citation-Grounded Response Generation
The retrieved passages, together with the user’s original question, are combined into a structured prompt supplied to the
LLM. The prompt instructs the model to synthesize its response using only the provided registry guidance, explain the
reasoning behind its conclusions, and avoid introducing claims beyond the retrieved evidence. Constraining generation
to externally retrieved documentation reduces the likelihood of hallucinated responses and keeps generated explanations
consistent with national standards and guidelines.
Response generation is designed to support both experienced registrars and individuals in training, so the objective
extends beyond a concise answer. A typical response consists of a direct answer, an explanation of the applicable coding
4

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
guidance, supporting citations or document references, and contextual information clarifying why the retrieved rule
applies to the scenario described.
Since many registry decisions depend on patient-specific clinical details, the system follows a conservative, evidence-
based generation strategy. If the retrieved passages do not provide sufficient information to support a definitive
conclusion, the model is instructed to state this limitation explicitly and identify the additional information needed to
resolve the question, such as diagnosis date, primary site, histology, tumor behavior, laterality, or prior tumor history,
rather than inferring or fabricating an answer. The generated response, together with its supporting citations, is delivered
to the registrar through a web-based chatbot interface; interface & implementation details are provided inAppendix A.1.
3 Evaluation
3.1 Evaluation Design
CRISSwas evaluated along two complementary axes: retrieval quality, captured through system-level and retrieval
metrics logged at inference time, and response quality, assessed through both automatic metrics and an LLM-as-a-Judge
protocol.
Evaluation was conducted across three difficulty tiers, annotated by experienced subject matter experts (ODSs). Easy
questions require direct code lookups, such as ICD-O-3 site and histology codes, with unambiguous ground truth.
Medium questions involve multi-step reasoning across coding manuals such as the STORE Manual and Solid Tumor
Rules. Hard questions reflect real-world case scenarios submitted by cancer registrars, often involving ambiguous
terminology, conflicting rules, or incomplete clinical information.
To contextualize the contribution of the retrieval pipeline, model performance was additionally benchmarked against
a non-RAG baseline condition. In this condition, frontier models from the GPT and Gemini families were queried
directly through their standard APIs, without access to the system’s retrieval pipeline or curated registry knowledge
base. To avoid unfairly handicapping these baseline models, web browsing was enabled through each model’s API,
allowing them to search the internet for relevant guidance in place of the curated knowledge base. This comparison
isolates the performance contribution of the system’s domain-specific retrieval pipeline relative to a general-purpose
frontier model with generic web-search access.
3.2 Retrieval Evaluation
The retrieval pipeline logged six metrics at inference time to capture system-level behavior: confidence score, reflecting
the model’s self-reported certainty; relevance rating, measuring how well the retrieved context matches the query;
faithfulness rating, assessing whether the answer is grounded in the retrieved sources; hop count, indicating the number
of retrieval steps performed; context length, measuring the total tokens of context consumed; and reasoning trace,
capturing the intermediate steps taken before producing a final answer. Source utilization was also recorded via sources
used and a corresponding summary, along with the year of the cited guidelines, to assess the temporal relevance of the
referenced rules.
3.3 Response Evaluation
Automatic Metrics:Response quality was assessed using standard automatic Natural Language Generation (NLG)
metrics such as ROUGE-1/2/L [ 36], BERTScore F1 [ 37], and sentence-level cosine similarity [ 38]. Full metric
definitions, rationale, and results are reported inAppendix C.
LLM-as-a-Judge Protocol:Beyond automatic metrics, response quality was further evaluated using an LLM-as-a-
Judge protocol, following prior work establishing the validity of LLM-based assessment for open-ended generation
tasks [ 39,40]. This approach was adopted over human expert annotation for three reasons.First, the volume of
evaluations required across 22 generator models, comprising open-weight/local (e.g., Gemma [ 41], gpt-oss [ 42],
Qwen [ 43]), proprietary, and baseline options across the Gemini and GPT families [ 44,45] for three difficulty tiers,
and two judge families makes expert annotation prohibitively expensive and time-consuming.Second, prior work has
demonstrated that capable LLM judges achieve high agreement with domain expert raters on structured evaluation
rubrics, particularly when the rubric dimensions are clearly defined and the judge is provided with ground truth for
reference.Third, LLM judges offer greater consistency than human annotators on repeated evaluations of the same
response, reducing noise in the quality signal.
To reduce the well-documented tendency of LLM judges to rate outputs from their own model family more favorably, a
cross-family judging protocol was used. All model outputs were independently evaluated by both GPT-4o and Gemini
3.1 Pro Preview [ 46,47], enabling assessment of inter-judge agreement across models and difficulty tiers. Final scores
were reported as the mean across both judges, an ensembling approach shown to produce more reliable aggregate scores
than single-judge evaluation by reducing the influence of individual judge mannerisms. Models for which the two
5

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
judges disagreed substantially, defined as a composite score difference exceeding 0.5 points, were flagged for qualitative
review, since large disagreements signal that response quality is sufficiently ambiguous to warrant closer inspection.
The judge evaluated each response across six dimensions scored on a 1 to 5 scale: Correctness (factual accuracy
relative to ground truth), Faithfulness (whether the response is grounded in real coding rules rather than hallucinated
content), Completeness (whether all components of the question are addressed, particularly for compound questions),
Reasoning (the logical soundness of the explanation, independent of whether the final answer is correct), Citation
Accuracy (whether cited pages and documents actually support the stated conclusion), and Tone (appropriateness for a
teaching and helpdesk context). Extended definitions and the rationale for each dimension are provided inAppendix C.
4 Results
We evaluatedCRISSfollowing the evaluation design described in the Evaluation (Section 3), analyzing response quality
across question difficulty tiers and model configurations.
4.1 Retrieval Quality
Retrieval quality remained high for most RAG configurations across the three difficulty tiers. To summarize retrieval
performance, we report a Composite Retrieval Score as shown inFigure 2, which provides a single measure of
overall retrieval quality by jointly capturing whether the system retrieved relevant content, remained faithful to the
retrieved evidence, and maintained confidence in the retrieved context. The score is calculated as the mean of
normalized relevance, faithfulness, and confidence. Composite retrieval scores exceeded the 0.70 reference threshold
for nearly all evaluated models across the three tiers. The highest and most consistent scores were observed for
GPT-4o-mini (0.97/0.98/0.97 for easy/medium/hard), followed by Gemini-2.5-Flash (0.95/0.94/0.91) and Gemini-
2.5-Pro (0.95/0.90/0.79). Google-Gemma-4-31B-QAT (0.91/0.94/0.86) and Qwen3.5-122B-A10B (0.90/0.87/0.88)
also maintained strong performance across tiers. Among the remaining models, most scores remained above the 0.70
threshold, although Qwen3.6-35B-A3B decreased to 0.66 on the hard tier, while GPT-5.5 recorded the lowest easy and
hard scores at 0.63, despite improving to 0.78 on the medium tier. Overall, the results indicate that retrieval quality was
generally preserved as question difficulty increased, although the degree of stability varied across models.
Figure 2: Composite Retrieval Score per model and tier (Sorted in ascending order based on easy tier score).
4.2 Response Quality
To check the need for a RAG-based system, we compared the grounding performance of the RAG configuration with a
No-RAG baseline across all three difficulty tiers as shown inFigure 3. We report a grounding score, which measures
how well the claims in a generated answer are supported by the evidence retrieved by the model. For the RAG system,
this evidence comes from the curated registry passages, while the No-RAG baseline uses web-search results. The RAG
system achieved mean grounding scores of approximately 0.62, 0.56, and 0.59 for easy, medium, and hard questions,
compared with 0.29, 0.26, and 0.29 for the No-RAG baseline. This represents improvements of 114%, 118%, and 103%
across the three tiers. These results show that RAG makes the generated answers much more evidence-supported and
traceable. The consistent improvement across all difficulty levels supports the need for a RAG-based system for cancer
registry coding.
6

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Easy Medium Hard
Difficulty Tier0.00.20.40.60.81.0Mean Grounding Score (0 1)
+114%
+118%+103%RAG  Our System
No RAG  Baseline
Improvement gap
Figure 3: RAG vs. No-RAG Grounding Score Across Difficulty Tiers
To compare the overall similarity of generated answers with the reference answers, we report ROUGE, BERTScore F1,
and cosine similarity averaged across all difficulty tiers as shown inFigure 4. Across all metrics, the RAG Local models
achieved the highest average scores, followed by the RAG Proprietary models, while the baseline models consistently
showed the lowest performance. RAG Local achieved a ROUGE average of 0.105, compared with 0.080 for RAG
Proprietary and 0.070 for the baseline. A similar pattern was observed for semantic similarity: RAG Local achieved the
highest BERTScore F1 of 0.580 and cosine similarity of 0.496, compared with 0.540 and 0.397 for RAG Proprietary
and 0.482 and 0.329 for the baseline, respectively. The higher BERTScore and cosine values relative to the ROUGE
scores also indicate that the generated answers often captured the same meaning as the reference answers without
necessarily using the exact same wording. Overall, the results show that the RAG configurations produced responses
that were more closely aligned with the reference answers than the baseline, with the RAG Local models showing
the strongest overall similarity. Full ROUGE-1/2/L, BERTScore F1, sentence-level cosine similarity, and associated
automatic metric results are reported inAppendix C.2.
ROUGE-1 ROUGE-2 ROUGE-L ROUGE-avg BERTScore F1 Cosine Sim.0.00.10.20.30.40.50.60.7Score (0 1)
0.118
0.0360.084 0.0800.540
0.397
0.160
0.0500.106 0.1050.580
0.496
0.110
0.0250.076 0.0700.482
0.329RAG Proprietary
RAG Local
Baseline
Figure 4: Text Similarity Metrics by Model Groups (Mean across all tiers)
4.3 LLM-as-a-Judge Assessment
To evaluate response quality across all questions and models, we employed an LLM-as-a-Judge method and report
an ensemble composite score from GPT and Gemini judges on a 1–5 scale as shown inFigure 5. Scores from the
two judges were averaged at the response level. The complete evaluation rubric used by the judges is provided in
Appendix C.3. Performance was strongest on the easy tier, where several models achieved scores above 4.0. The
highest score was obtained by Baseline-Gemini-3.1-Pro-Preview (4.39), followed closely by the RAG proprietary
models GPT-5.5 (4.30), Gemini-3.1-Pro-Preview (4.29), and Gemini-2.5-Pro (4.25). On the medium tier, GPT-5.5
7

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
(4.25) achieved the highest score, followed by Gemini-2.5-Pro (4.00) and Gemini-3.5-Flash (3.86). Scores decreased
substantially on the hard tier for most models. The RAG Local models however occupied the highest positions, led by
Qwen3.5-122B-A10B (2.94), OSS-120B (2.91), and Qwen3.6-35B-A3B (2.84).
1 2 3 4 5
Composite Score (1 5)
BL-GPT-4o-miniMedGemma-27b-text-itGPT-4o-miniGemma4-31bBL-Gem-3.1-flash-liteOSS-120bQwen3.6-27bGPT-5.4-miniBL-GPT-5.5GPT-4oBL-GPT4oQwen3.5-122b-a10bBL-Gem-3.5-flashQwen3.6-35b-a3bGem-2.0-FlashGem-2.5-ProGem-3.1-Pro-PreviewGPT-5.5BL-Gem-3.1-pro-preview
2.682.862.913.483.523.573.583.693.703.723.804.004.034.134.174.254.294.304.39Easy
1 2 3 4 5
Composite Score (1 5)
BL-Gem-3.1-pro-previewBL-Gem-3.5-flashBL-Gem-3.1-flash-liteBL-GPT-5.5OSS-120bOSS-20bGPT-4o-miniMedGemma-27b-text-itGem-2.0-FlashGemma4-31bQwen3.6-27bGPT-4oBL-GPT-4oBL-GPT-5.4-miniGPT-5.4-miniQwen3.6-35b-a3bQwen3.5-122b-a10bGem-3.1-Pro-PreviewGem-3.5-FlashGem-2.5-ProGPT-5.5
1.431.482.212.312.542.582.612.692.742.842.902.933.103.143.163.203.433.633.864.004.25Medium
1 2 3 4 5
Composite Score (1 5)
BL-GPT-5.4-miniBL-Gem-3.1-pro-previewBL-Gem-3.5-flashBL-Gem-3.1-flash-liteBL-GPT-4oGPT-4o-miniGPT-4oGem-2.5-ProBL-GPT-5.5Gem-3.1-Pro-PreviewGPT-5.5Gem-3.5-FlashGem-2.0-FlashGemma4-31bMedGemma-27b-text-itOSS-20bQwen3.6-27bQwen3.6-35b-a3bOSS-120bQwen3.5-122b-a10b
1.131.191.201.311.321.351.411.461.481.491.511.522.372.402.602.672.702.842.912.94Hard
RAG Proprietary RAG Local Baseline (No RAG)
Figure 5: LLM-as-a-Judge Scores for all difficulty tiers
5 Discussion
5.1 Principal Findings
The results show a clear advantage of RAG over non-RAG approaches, particularly as question difficulty increases.
While some non-RAG models remained competitive on easier questions, their performance declined more sharply on
medium and hard cases, whereas the RAG configurations generally maintained stronger grounding, semantic similarity,
and LLM-as-a-Judge scores. This supports the need for retrieval when answering cancer registry questions that depend
on specific coding rules and current reference materials.
5.2 Ablation Study: Model Behavior and Response Caution
The LLM-as-a-Judge results revealed a clear shift in model performance as question difficulty increased. While
proprietary RAG models performed strongly on the easy and medium tiers, the RAG Local models ranked ahead of the
proprietary models on the hard tier. To understand what may be driving this difference, we analyzed model response
behavior, including how often models acknowledged ambiguity, requested more information, or refused to provide a
definitive answer. The full caution-flag definitions and detailed analysis are provided inAppendix C4. As shown in
Figure 6, RAG Proprietary models showed a higher overall caution/refusal rate than RAG Local models (23.1% vs.
15.3%), while local models achieved higher containment (0.401 vs. 0.288) and stronger text-similarity scores, including
BERTScore F1 (0.580 vs. 0.540) and cosine similarity (0.496 vs. 0.397). as discussed inFigure 4. These results
suggest that proprietary models are more likely to respond cautiously to difficult or ambiguous cases by acknowledging
uncertainty, qualifying their response, or avoiding a definitive coding decision. In contrast, local models were more
likely to provide a direct coding answer, which increased their overlap with the reference answers and likely contributed
to their higher hard-tier scores. Importantly, this does not necessarily mean that the local models are clinically better.
Their stronger ranking may partly reflect that they are less likely to acknowledge uncertainty or ambiguity, and therefore
more likely to commit to an answer even when the case is difficult. Thus, the hard-tier advantage of local models should
be interpreted as a difference in response behavior as well as model performance, rather than as evidence of uniformly
better reasoning.
5.3 Reliability of Automated Evaluation
To further examine this pattern, we analyzed four judge flags: hallucination, clinical safety, multi-part miss, and rule
conflict across each difficulty tier, where lower flag rates indicate better performance. Since we had two judge models
for a fair judgement, the flag is only taken into account if both the judge models agree. The definitions and scoring
criteria for these flags are provided in the evaluation rubric in theAppendix C3. As shown inFigure 7, the differences
between model groups become much larger as question difficulty increases. On the hard tier, RAG Local models had
8

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Cannot
DetermineConditional
AnswerNeeds More
InfoAcknowledges
AmbiguityExplicit
RefusalDefers to
ExpertANY
Refusal05101520% of Responses Flagged
3%
2%
1%8%
3%
0%14%
2%
0%1%4%
1%
0%7%
0%7%
0%4%
1%
0%12%Response Caution Flags by Model Group
RAG Proprietary
RAG Local
Baseline
Figure 6: Ablation Study of Response Caution Flags by Model Types
substantially fewer multi-part misses (48%) than proprietary RAG models (90%) and the baseline (96%), and fewer
clinical-safety flags (58%) than proprietary models (87%) and the baseline (88%). This indicates that local models were
more likely to attempt all parts of difficult questions and provide a complete coding response, which is consistent with
their higher similarity, and LLM-as-a-Judge scores. However, hallucination rates were high for both RAG groups on
hard questions (70% for local and 77% for proprietary models). On the easy and medium tiers, RAG Local models also
showed relatively high rates of 70% and 63%, respectively, compared with 40% and 36% for RAG Proprietary models.
This difference narrowed substantially on the hard tier. This unusual behavior should be read with caution. In our rubric,
this flag is raised when a model cites a rule, page number, or coding standard that does not exist. Taken at face value, the
rates would suggest that models invented citations in the most hard-tier answers. However, our manual review of flagged
responses found that the flag was also raised in cases where no fabricated citation was present, including answers in
which the model drew on its own internal knowledge of coding practice rather than citing a specific provision, so, it
does not mean the entire answer is incorrect. Earlier work has found that LLM judges are often wrong when asked to
assess factuality and hallucination, and that asking a judge to make this decision in a single step is unreliable even when
the judge is given reference evidence [ 48]. That study breaks the task into steps and finds that models fail most often at
locating the evidence that supports or contradicts a claim, which is exactly what our rubric asks the judge to do in one
pass without showing it the retrieved passages.Figure 13inAppendix C.3shows how often each judge raised each
flag on its own compared with how often both judges raised it on the same response, and the two judges differ most on
the hallucination flag.
Easy Medium Hard0%20%40%60%80%100%% of Responses Flagged by Both Judges40%
36%77%
62%70% 70%
8%19%52%Hallucination
Easy Medium Hard24%46%81%
32%55%
48%
36%74%81%Clinical Safety
Easy Medium Hard5%8%85%
6%13%29%
4%47%95%Multi-Part Miss
Easy Medium Hard1%3%
0% 1% 2% 2% 2% 1% 0%Rule ConflictRAG Proprietary RAG Local Baseline (No RAG)
Figure 7: LLM-as-a-Judge Binary Flags Assessment by Model Types and Difficulty Tiers
Taken together, these results indicate that the stronger hard-tier ranking of local models reflects a difference in response
behavior rather than demonstrated clinical reliability. Local models were more likely to provide direct and complete
9

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
answers and missed fewer parts of difficult questions, while proprietary models more often acknowledged uncertainty
or declined to commit to a definitive coding decision.
5.4 Limitations
This study has several limitations. First, exhaustive review by subject-matter experts was impractical, so we relied
on automated evaluation. We assessed 22 model configurations, spanning RAG and non-RAG approaches, on 208
questions across three difficulty tiers, producing approximately 4,600 responses. To keep the evaluation consistent and
scalable, we used multiple complementary metrics. However, several of these metrics, including ROUGE, BERTScore,
cosine similarity, containment, and LLM-as-a-Judge scores, may favor responses that are more complete or more similar
to the reference answer even when they contain unsupported information. This may partly explain why RAG Local
models achieved strong hard-tier scores despite high hallucination rates. In addition, the retrieval metrics were partly
self-reported by the LLM, which may introduce further evaluation bias.
Second, the system sometimes retrieved guidance from a manual edition that did not match the case’s diagnosis year.
The knowledge base intentionally includes multiple editions of the same coding manuals because registry coding rules
are applied according to year of diagnosis: a case must be coded using the rules in effect at that time, and registrars
routinely code, recode, and audit historical cases. Keeping only the most recent edition would leave the system unable
to answer correctly for any case outside the current coding year. To reduce edition mismatches, the retrieval pipeline
incorporated year detection and recency-aware re-ranking to prioritize the most applicable version. However, this did
not fully prevent mismatches for historical cases, and future work should explore explicit filtering by diagnosis year
during retrieval.
Third, the question bank was developed at a single institution and may not capture the full diversity of registry workflows.
The safety and caution thresholds should also be interpreted as evaluation reference points rather than established
standards for clinical deployment.
Finally, the system has not yet been evaluated prospectively with practicing cancer registrars in real-world workflows.
Such evaluation will be important for assessing its usability, reliability, and practical value.
6 Conclusion
This study demonstrates the value of a retrieval-augmented, conversational assistant for cancer registry coding support
and education, providing grounded, traceable answers drawn from current registry manuals.CRISSis intended to
help registrars locate, interpret, and apply national standards and guidelines while preserving human oversight and
supporting learning and verification. Across our evaluation, RAG consistently provided stronger grounding and overall
response quality than non-RAG baselines, with the advantage becoming more important as question difficulty increased.
The results also showed that both proprietary and RAG Local models can contribute effectively, with proprietary models
performing strongly on easy and medium questions and several local models achieving competitive scores on difficult
cases. Overall, these findings support the use of domain-specific retrieval, citation-grounded response generation, and
multi-model evaluation as a practical foundation for assisting cancer registrars with complex and evolving coding
guidelines. By providing evidence-supported answers and explanations grounded in national standards and guidelines,
CRISSrepresents a promising step toward AI-assisted registry workflows that strengthen access to knowledge, support
learning, and augment human decision-making.
Funding
This work was supported by the Missouri Cancer Registry and Research Center, funded in part through a cooperative
agreement between the Centers for Disease Control and Prevention and the Missouri Department of Health and Senior
Services (MODHSS) (NU58DP007130-05), and through a surveillance contract between MODHSS and the University
of Missouri. The ideas and opinions expressed are those of the author(s) and do not necessarily reflect the opinions of
the MODHSS, the University of Missouri, CDC, or other funding agencies.
Conflicts of Interest
The authors declare no conflicts of interest.
10

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
References
[1]Centers for Disease Control and Prevention. Incidence data sources. U.S. Cancer Statistics, . URL https://www.
cdc.gov/united-states-cancer-statistics/technical-notes/incidence-data-sources.html .
Accessed: February 23, 2026.
[2]Thomas C Tucker, Eric B Durbin, Jaclyn K McDowell, and Bin Huang. Unlocking the potential of population-
based cancer registries.Cancer, 125(21):3729–3737, 2019.
[3]Centers for Disease Control and Prevention. About the national program of cancer registries. National Program
of Cancer Registries (NPCR), . URL https://www.cdc.gov/national-program-cancer-registries/
about/index.html. Accessed: February 23, 2026.
[4]National Cancer Institute. Overview of the surveillance, epidemiology, and end results (seer) program. https:
//seer.cancer.gov/about/overview.html, 2026. Accessed: February 23, 2026.
[5]Hong Chen, David Hailey, Ning Wang, and Ping Yu. A review of data quality assessment methods for public
health information systems.International journal of environmental research and public health, 11(5):5170–5207,
2014.
[6]National Cancer Institute. Becoming a cancer registry professional. Surveillance, Epidemiology, and End Results
(SEER) Program. URL https://seer.cancer.gov/registrars/howto.html . Accessed: February 23,
2026.
[7]Hannah K Weir, Recinda Sherman, Mandi Yu, Susan Gershman, Brenda M Hofer, Manxia Wu, and Don Green.
Cancer incidence in older adults in the united states: characteristics, specificity, and completeness of the data.
Journal of registry management, 47(3):150, 2020.
[8]Iris Zachary, Suzanne A Boren, Eduardo Simoes, Jeannette Jackson-Thompson, J Wade Davis, and Lanis Hicks.
Information management in cancer registries: evaluating the needs for cancer data collection and cancer research.
Online Journal of Public Health Informatics, 7(2):e213, 2015.
[9]Mahul B Amin, Stephen B Edge, Frederick L Greene, David R Byrd, Robert K Brookland, Mary Kay Washington,
Jeffrey E Gershenwald, Carolyn C Compton, Kenneth R Hess, Daniel C Sullivan, et al.AJCC cancer staging
manual, volume 1024. Springer, 2017.
[10] Thirteenth Edition, Lori A Havener, and Dianne Hultstrom. North american association of central cancer registries.
2004.
[11] Ole Møller Jensen.Cancer registration: principles and methods, volume 95. IARC, 1991.
[12] Mohammad Beheshti, Imad Eddine Toubal, Khuder Alaboud, Mohammed Almalaysha, Olabode B Ogundele,
Hamza Turabieh, Nader Abdalnabi, Suzanne A Boren, Grant J Scott, and Butros M Dahu. Evaluating the reliability
of chatgpt for health-related questions: a systematic review. InInformatics, volume 12, page 9. MDPI, 2025.
[13] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and
Daniel Shu Wei Ting. Large language models in medicine.Nature medicine, 29(8):1930–1940, 2023.
[14] Zabir Al Nazi and Wei Peng. Large language models in healthcare and medical domain: A review. InInformatics,
volume 11, page 57. MDPI, 2024.
[15] Vani Seth, Ashish Pandey, Kavya Gundlapalli, Mahesh Duvvuri, Rithvik Mettu, Iris Zachary, and Prasad Calyam.
Empathai: A trustworthy and secure conversational agent for mental healthcare. In2025 Annual Computer
Security Applications Conference Workshops (ACSAC Workshops), pages 298–301. IEEE, 2025.
[16] Vani Seth.Layered security and trust mechanisms for GenAI applications in critical domains. PhD thesis,
University of Missouri–Columbia, 2026.
[17] Chimaobi Amadi and Adegboyega Ojo. Building trustworthy ai in healthcare.IEEE Access, 14:1182–1212, 2025.
[18] Maryam Seifaddini, Mohammad Beheshti, Steven Richberg, Magda Esebua, and Iris Zachary. From information
extraction to clinical reasoning: A systematic scoping review of large language models in cancer pathology reports.
Modern Pathology, page 101043, 2026.
[19] Abdul Rehman Khalid, Haider Ali, Kounen Fathima, Kouayep Sonia Carole, and Hee-Cheol Kim. Recent
advances in ai for automated icd coding: A systematic literature review.Journal of Medical Systems, 50(1):113,
2026.
[20] Kamonrat Tangudomkit, Sawrawit Chairat, and Sitthichok Chaichulee. Artificial intelligence for automated icd-10
coding: a systematic review of multi-label text classification in clinical narratives.Frontiers in Digital Health, 8:
1905379, 2026.
11

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
[21] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions.ACM transactions on information systems, 43(2):1–55, 2025.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[23] Lameck Mbangula Amugongo, Pietro Mascheroni, Steven Brooks, Stefan Doering, and Jan Seidel. Retrieval
augmented generation for large language models in healthcare: A systematic review.PLOS Digital Health, 4(6):
e0000877, 2025.
[24] Shaoxiong Ji, Xiaobo Li, Wei Sun, Hang Dong, Ara Taalas, Yijia Zhang, Honghan Wu, Esa Pitkänen, and Pekka
Marttinen. A unified review of deep learning for automated medical coding.ACM computing surveys, 56(12):
1–41, 2024.
[25] Nan-Haw Chow, Han Chang, Hung-Kai Chen, Chen-Yuan Lin, Ying-Lung Liu, Po-Yen Tseng, Li-Ju Shiu,
Yen-Wei Chu, Pau-Choo Chung, and Kai-Po Chang. Digital registrar: A schema-first framework for multi-cancer
privacy-preserving pathology abstraction via local llms.Diagnostics, 16(11):1644, 2026.
[26] Joshua T Fuchs, Matthew J Satusky, Peter J Leese, Subhadeep Nag, Isaiah W Zipple, Chris D Baggett, Sydney
Lash, Katherine Reeder-Hayes, William A Wood, Cara T Johnson, et al. Large language models for cancer registry
abstraction: a real-world evaluation across models, variables, and cancer types.medRxiv, pages 2026–06, 2026.
[27] James C Dickerson, Marni B McClure, Margaret Shaw, Marissa B Reitsma, Nicole H Dalal, Allison W Kurian,
and Jennifer L Caswell-Jin. Fully automated abstraction of longitudinal breast oncology records with off-the-shelf
large language models.medRxiv, 2026.
[28] Chen-Kai Wang, Cheng-Rong Ke, Ming-Siang Huang, Inn-Wen Chong, Yi-Hsin Yang, Vincent S Tseng, and
Hong-Jie Dai. Using large language models for efficient cancer registry coding in the real hospital setting: A
feasibility study. InBiocomputing 2025: Proceedings of the Pacific Symposium, pages 121–137. World Scientific,
2024.
[29] Surveillance, Epidemiology, and End Results (SEER) Program. SEER Program Coding and Staging Manual.
https://seer.cancer.gov/tools/codingmanuals/, 2026. Accessed: February 23, 2026.
[30] L. Dickie, S. Adams, D. Harrison, J. Hofferkamp, M. Traverso-Ortiz, A. Van Dyke, and S. Negoita. Solid Tumor
Rules.https://seer.cancer.gov/tools/solidtumor/, December 2025. Accessed: February 23, 2026.
[31] Surveillance, Epidemiology, and End Results (SEER) Program. 2007 Multiple Primary and Histology Coding
Rules.https://seer.cancer.gov/tools/mphrules/, 2012. Revised August 24, 2012. February 23, 2026.
[32] Surveillance, Epidemiology, and End Results (SEER) Program. Historical Staging and Coding Manuals. https:
//seer.cancer.gov/tools/codingmanuals/historical.html. Accessed: February 23, 2026.
[33] OpenAI. text-embedding-3-large model. OpenAI API Documentation. URL https://developers.openai.
com/api/docs/models/text-embedding-3-large. Accessed: January 6, 2026.
[34] Google. Embeddings. Gemini API Documentation, Google AI for Developers, 2026. URL https://ai.google.
dev/gemini-api/docs/embeddings. Accessed: January 6, 2026.
[35] Vanessa Klotzman. Enhancing automated medical coding: evaluating embedding models for icd-10-cm code
mapping.medRxiv, pages 2024–07, 2024.
[36] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. InText summarization branches out,
pages 74–81, 2004.
[37] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text
generation with bert.arXiv preprint arXiv:1904.09675, 2019.
[38] Wael H Gomaa, Aly A Fahmy, et al. A survey of text similarity approaches.international journal of Computer
Applications, 68(13):13–18, 2013.
[39] Shuang Zhou, Wenya Xie, Jiaxi Li, Zaifu Zhan, Meijia Song, Han Yang, Cheyenna Espinoza, Lindsay Welton,
Xinnie Mai, Yanwei Jin, et al. Automating expert-level medical reasoning evaluation of large language models.
NPJ Digital Medicine, 9(1):34, 2025.
[40] Emma Croxford, Yanjun Gao, Elliot First, Nicholas Pellegrino, Miranda Schnier, John Caskey, Madeline Oguss,
Graham Wills, Guanhua Chen, Dmitriy Dligach, et al. Evaluating clinical ai summaries with large language
models as judges.npj Digital Medicine, 8(1):640, 2025.
12

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
[41] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent
Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research
and technology.arXiv preprint arXiv:2403.08295, 2024.
[42] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai,
Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card.arXiv preprint arXiv:2508.10925,
2025.
[43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
[44] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models.arXiv
preprint arXiv:2312.11805, 2023.
[45] Hina Afridi, Habib Ullah, Sultan Daud Khan, and Mohib Ullah. From gpt-3 to gpt-5: Mapping their capabilities,
scope, limitations, and consequences.arXiv preprint arXiv:2604.10332, 2026.
[46] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila
Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv preprint arXiv:2410.21276, 2024.
[47] Google. Gemini 3.1 pro preview. Gemini API Documentation, Google AI for Developers, 2026. URL https://
ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview. Accessed: September 16, 2026.
[48] Yu Zhang, Peter Belcak, Shizhe Diao, Yonggan Fu, Shaona Ghosh, Morteza Mardani, Eileen Margaret Peters
Long, Bei Yu, and Pavlo Molchanov. Probe: Process-based benchmark for hallucination detection. InFindings of
the Association for Computational Linguistics: ACL 2026, pages 42303–42320, 2026.
[49] Sentence Transformers. all-minilm-l6-v2. Hugging Face Model Hub. URL https://huggingface.co/
sentence-transformers/all-MiniLM-L6-v2. Accessed: February 23, 2026.
13

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
A System and Implementation Details
A.1 User Interface
Users interact with the system by submitting questions as natural-language text rather than structured queries or
manual-specific terminology, and the interface displays the generated response together with its supporting citations as
mentioned inFigure 8. The chatbot was implemented as a React-based web application hosted on AWS EC2, paired
with a Flask backend. When a user submits a question, the React interface sends the request to the Flask backend,
which first checks a semantic cache for a similar prior question; if a sufficiently similar question has been answered
previously, the cached response is returned immediately to reduce latency and cost. If no matching cached response
exists, the backend retrieves relevant content from the vector database, sends the retrieved context to the LLM, formats
the response, stores the result in the semantic cache for future reuse, and returns the answer to the interface. Alongside
the answer, the interface streams the model’s thinking process, adding a transparent layer that lets users identify potential
bias in the retrieved evidence or the model’s reasoning, and refine their prompts to obtain better results.
Figure 8: Chatbot user interface displaying the grounded response and the LLM’s reasoning process.
B Knowledge Base Construction Parameters
The extracted text is segmented into overlapping chunks of 1,500 characters with an overlap of 200 characters. This
chunking strategy balances two competing concerns.First, because registry manuals often contain lengthy discussions,
nested decision rules, and conditional logic, indexing entire documents would reduce retrieval precision and exceed the
context limits of the LLM.Second, chunks that are too small risk separating a rule from the exceptions that modify
it. This granularity allows individual chunks to represent coherent units of registry knowledge, such as a coding
rule, an explanatory note, a site-specific instruction, or a histology definition. Each resulting chunk is paired with
descriptive metadata, including the originating document, section title, page reference, and any available rule identifiers.
Rather than serving solely as indexing information, this metadata plays an integral role throughout the conversational
workflow, disambiguating conceptually similar passages during retrieval and supporting source citation during response
generation.
Table 1lists the coding manuals and reference documents indexed in the knowledge base. All documents are official
publications from authoritative cancer registry bodies. Year-specific documents are tagged with their applicable
diagnosis year range to support temporally-aware retrieval.
14

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Table 1: List of Cancer Registry Reference Documents Indexed in the Knowledge Base
Document Edition Applicable
YearDescription
Solid Tumor Rules 2025 2018+ Multiple primary and histology rules for solid tumors
STORE Manual 2025 & 2026 2025 & 2026 Standard data item definitions and coding rules
2026 Implementation
Guidelines2026 2026 Updated coding rules and clarifications
MPH Rules 2007 2007–2017 Multiple primary and histology rules (pre-2018; 2018+
refer to Solid Tumor Rules)
Hematopoietic Rules 2024 2010+ Coding rules for lymphoid and hematopoietic neoplasms
Grade Coding Instruc-
tionsV3.2 2018+ Histologic grade coding tables and instructions
Site-Specific Data Items
(SSDI)V3.2 2018+ Site-specific factor coding instructions
Summary Stage 2018 2018 2018+ Staging definitions and coding rules
C Automatic Metrics and LLM-as-a-Judge Rubric
This section provides the full automatic-metric results, along with additional details on the metric definitions, rationale,
and LLM-as-a-Judge rubric. It is included to support the main findings and provide further context for the evaluation
choices.
C.1 Automatic Metrics: Definitions and Rationale
Automatic metrics offer a scalable, reproducible, and judge-free means of assessing answer quality. For a system like
CRISS, which is expected to handle hundreds of queries per day across a registry program, these metrics provide a
practical first-pass quality signal that does not require human review of every response. The metric suite spanned three
complementary measurement dimensions: lexical overlap, contextual semantic similarity, and answer relevance.
Lexical overlap was measured via ROUGE-1, ROUGE-2, and ROUGE-L F1 scores, which quantify the n-gram and
longest common subsequence overlap between the generated response and the ground truth. All three ROUGE variants
were included because ROUGE-1 captures individual token recovery, ROUGE-2 captures phrase-level precision, and
ROUGE-L captures structural alignment through longest common subsequence, each revealing a distinct dimension of
the lexical relationship between response and reference. We additionally report ROUGE-avg, calculated as the mean of
ROUGE-1, ROUGE-2, and ROUGE-L, as an overall measure of lexical overlap.
Contextual semantic similarity was measured via BERTScore F1, which computes token-level precision, recall, and
F1 using contextual embeddings rather than surface token matching. BERTScore is particularly well suited to this
domain because coding answers frequently express the same factual content using different but semantically equivalent
phrasing. For instance, "Bladder NOS" and "C67.9" convey the same site assignment, and surface overlap metrics
would penalize such paraphrases inappropriately. By operating in embedding space, BERTScore captures semantic
equivalence that ROUGE misses.
Overall semantic alignment was further measured via cosine similarity between sentence-level embeddings of the
generated response and the ground truth, computed using the all-MiniLM-L6-v2 sentence transformer [ 49]. Unlike
BERTScore, which operates at the token level, sentence-level cosine similarity captures the holistic semantic proximity
of the two texts, providing a complementary view of whether the response conveys the same overall meaning as the
ground truth even when individual token overlap is low. Answer relevance was also computed as the cosine similarity
between the generated response and the original question, providing a signal on whether the model stays on topic rather
than drifting into tangential explanations. Finally, containment score measures the fraction of ground truth tokens
recovered in the model response, capturing recall at the token level independently of precision.
C.2 Automatic Metrics: Results
To evaluate how closely the generated responses matched the reference answers, we used ROUGE-1, ROUGE-2,
ROUGE-L, BERTScore F1, and sentence-level cosine similarity. Since the generated responses were generally longer
and more explanatory than the concise ground-truth answers, ROUGE scores remained relatively low across models.
The additional correct explanations included in a generated response may not appear in the ground truth and therefore
15

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
do not contribute to n-gram overlap. For this reason, we interpret ROUGE primarily as a lexical-overlap measure and
place greater emphasis on BERTScore F1 and cosine similarity when comparing semantic agreement across models.
As shown in theFigure 9, when averaged across all models, the similarity metrics were generally higher on the hard tier
rather than decreasing with difficulty. BERTScore F1 increased from 0.507 for easy questions to 0.517 for medium and
0.579 for hard questions, while cosine similarity increased from 0.386 to 0.388 and 0.446, respectively. A similar trend
was observed for ROUGE-1 (0.096/0.111/0.179), ROUGE-L (0.077/0.080/0.108), and ROUGE-avg (0.071/0.075/0.108)
across the easy, medium, and hard tiers. ROUGE-2 remained comparatively low and stable at 0.041, 0.032, and 0.038.
This indicates that responses to hard questions often contained more of the terminology and semantic content present in
the reference answers, even though the questions themselves were more challenging.
ROUGE-1 ROUGE-2 ROUGE-L ROUGE-avg BERTScore F1 Cosine Sim.0.00.10.20.30.40.50.60.7Score (0 1)
0.096
0.0410.0770.0710.507
0.386
0.111
0.0320.080 0.0750.517
0.388
0.179
0.0380.108 0.1080.579
0.446Easy
Medium
Hard
Figure 9: Text-similarity metrics by difficulty tier, averaged across all evaluated models.
At the individual-model level, the increase in hard-tier similarity was particularly visible among several RAG Local
models. As shown inFigure 10, these models frequently achieved much higher ROUGE-avg values on hard questions
than on easy or medium questions. This pattern is consistent with the response-behavior analysis in the main text:
local models were more likely to provide direct and complete coding answers on difficult cases, increasing the amount
of reference-answer content present in their responses. However, lexical overlap alone does not establish that the
underlying coding reasoning is correct.
Gemma4-31b
MedGemma-27b-text-itOSS-20b
Gem-2.5-Pro
Qwen3.6-35b-a3bOSS-120bGPT-5.5
Qwen3.6-27b
Qwen3.5-122b-a10bBL-GPT4o
Gem-2.0-FlashBL-GPT-5.4-miniGPT-5.4-mini
Gem-3.1-Pro-PreviewBL-GPT-5.5
BL-GPT-4o-miniGPT-4o
BL-Gem-3.1-pro-previewGPT-4o-mini
BL-Gem-3.5-flashGem-3.5-Flash
BL-Gem-3.1-flash-lite0.000.050.100.150.20ROUGE-avg (0 1)
0.094
0.078
0.078
0.111
0.074
0.071
0.090
0.061
0.059
0.087
0.070
0.080
0.078
0.062
0.140
0.057
0.054
0.059
0.046
0.052
0.037
0.0320.102
0.084
0.083
0.122
0.078
0.083
0.105
0.081
0.075
0.099
0.063
0.080
0.078
0.076
0.057
0.066
0.065
0.048
0.055
0.043
0.055
0.0340.162
0.179
0.168
0.096
0.157
0.147
0.095
0.148
0.153
0.095
0.139
0.085
0.083
0.087
0.027
0.087
0.090
0.082
0.083
0.080
0.070
0.069Easy
Medium
Hard
Figure 10: ROUGE-avg scores across all models and difficulty tiers (Label color: Blue = RAG Proprietary; Red = RAG
Local; Gray = Baseline)
BERTScore provides a clearer view of this semantic agreement. As shown inFigure 11Several local RAG models
achieved their highest BERTScore values on the hard tier, while the differences across easy and medium questions were
smaller for many proprietary and baseline configurations. Because BERTScore is less dependent on exact wording than
ROUGE, these results suggest that the hard-tier responses from several models captured a larger portion of the meaning
represented in the reference answers. Nevertheless, a semantically similar answer may still contain unsupported details
or apply a coding rule incorrectly, so BERTScore should be interpreted together with grounding and safety measures.
16

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
MedGemma-27b-text-itQwen3.6-27b
Qwen3.6-35b-a3b
Qwen3.5-122b-a10bGemma4-31bOSS-20bOSS-120b
Gem-2.5-ProGPT-5.5
BL-GPT-4o
GPT-5.4-mini GPT-4o-miniGPT-4o
Gem-3.1-Pro-PreviewBL-GPT-5.4-mini BL-GPT-4o-miniGem-2.0-Flash Gem-3.5-Flash
BL-Gem-3.1-pro-previewBL-Gem-3.5-flash
BL-Gem-3.1-flash-liteBL-GPT-5.50.00.10.20.30.40.50.60.70.8BERTScore F1 (0 1)
0.531
0.528
0.543
0.523
0.522
0.520
0.509
0.544
0.541
0.509
0.519
0.497
0.491
0.501
0.501
0.486
0.540
0.489
0.503
0.486
0.464
0.4100.562
0.576
0.557
0.569
0.562
0.543
0.543
0.579
0.557
0.551
0.536
0.524
0.532
0.531
0.527
0.525
0.462
0.489
0.484
0.470
0.481
0.2100.675
0.649
0.653
0.654
0.650
0.660
0.651
0.579
0.568
0.586
0.568
0.591
0.589
0.573
0.576
0.587
0.589
0.573
0.549
0.523
0.531
0.165Easy
Medium
Hard
Figure 11: BERTScore F1 across all models and difficulty tiers. (Label color: Blue = RAG Proprietary; Red = RAG
Local; Gray = Baseline)
As shown inFigure 12, several local RAG models achieved substantially higher cosine similarity on hard questions,
whereas some proprietary and baseline models remained stable or declined. This suggests that the stronger hard-tier
similarity was driven by specific model groups rather than a uniform improvement across all models. Although some
responses contained relevant registry terminology, higher cosine similarity required the overall response to remain
semantically aligned with the reference answer. This distinction is important for hard questions, where models may
retrieve appropriate terminology but still apply the underlying coding rule incorrectly. For this reason, ROUGE,
BERTScore F1, cosine similarity, grounding, and judge-based safety measures were considered together rather than
treating lexical or semantic similarity alone as sufficient evidence of response correctness.
MedGemma-27b-text-itQwen3.5-122b-a10bQwen3.6-27b
Qwen3.6-35b-a3bGemma4-31bOSS-20bOSS-120b
Gem-2.5-ProGem-2.0-FlashGPT-5.5
GPT-5.4-mini
Gem-3.1-Pro-PreviewBL-GPT-5.4-miniGPT-4o
BL-GPT-4oGPT-4o-miniGem-3.5-FlashBL-GPT-4o-mini
BL-Gem-3.1-pro-previewBL-Gem-3.5-flash
BL-Gem-3.1-flash-liteBL-GPT-5.50.00.10.20.30.40.50.60.70.8Cosine Similarity (0 1)
0.396
0.407
0.383
0.418
0.394
0.387
0.389
0.433
0.401
0.430
0.410
0.385
0.386
0.370
0.359
0.373
0.346
0.347
0.394
0.376
0.351
0.3660.417
0.443
0.460
0.421
0.451
0.416
0.416
0.498
0.316
0.442
0.400
0.439
0.399
0.406
0.412
0.394
0.385
0.375
0.289
0.274
0.278
0.1910.673
0.655
0.663
0.661
0.623
0.659
0.656
0.365
0.545
0.371
0.373
0.358
0.368
0.369
0.364
0.357
0.366
0.360
0.327
0.316
0.297
0.097Easy
Medium
Hard
Figure 12: Sentence-level cosine similarity across all models and difficulty tiers. (Label color: Blue = RAG Proprietary;
Red = RAG Local; Gray = Baseline)
C.3 LLM-as-a-Judge Rubric: Dimensions and Flags
The judge evaluated each response across six dimensions scored on a 1 to 5 scale.Correctnessassesses factual accuracy
relative to ground truth and serves as the primary technical quality signal.Faithfulnessassesses whether the response is
grounded in real coding rules rather than hallucinated content, a distinct dimension from correctness because a response
can be factually correct while citing fabricated or irrelevant supporting evidence.Completenessassesses whether the
response addresses all components of the question, which is particularly relevant for compound questions that require
multiple coding decisions in a single response.Reasoningassesses the logical soundness of the explanation, capturing
whether the model correctly applied the relevant rule rather than arriving at the right answer through incorrect reasoning.
Citation Accuracyassesses whether cited page numbers and documents actually support the stated conclusion, a
dimension of particular importance in this domain because registrars are trained to verify answers against official
manuals and a fabricated but plausible citation may be accepted without verification.Toneassesses the appropriateness
of the response for a teaching and helpdesk context, considering clarity, educational value, and whether the answer
explains the reasoning behind a coding decision rather than simply stating the code.
The judge also produced four binary flags. TheHallucination Flagis raised when the model cites a non-existent or
irrelevant rule, fabricates a page number, or references a coding standard that does not support the stated conclusion.
TheClinical Safety Flagis raised when an incorrect answer could cause meaningful harm in a real registry workflow,
17

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
for example a wrong reportability decision, an incorrect primary site assignment, an incorrect class of case code, or a
misapplication of ambiguous terminology rules. TheMulti-Part Miss Flagis raised when the model fails to address at
least one component of a compound question. TheRule Conflict Flagis raised when the model cites contradictory
rules or pages within a single response without resolving the conflict.
Table 2presents the complete LLM-as-a-Judge evaluation rubric used in this study. Each response is scored across
seven dimensions on a 1–5 scale, and four binary flags are assigned to capture qualitative failure modes with direct
operational consequences.
Table 2: LLM-as-a-Judge Evaluation Rubric and Binary Flags
Dimension Description
Correctness Is the answer factually accurate compared to the ground truth?
Faithfulness Is the answer grounded in real coding rules rather than hallucinated information?
Completeness Did the answer address all parts of the question?
Reasoning Is the explanation or justification correct and logically consistent?
Citation Accuracy Did the cited page numbers or documents actually support the claim?
Tone Is the answer helpful, clear, and appropriate for a teaching or helpdesk context used by cancer registrars?
Flag Trigger Condition
Hallucination Flag True if the model cited a non-existing or irrelevant rule, page number, or coding standard.
Clinical Safety Flag True if the wrong answer could cause meaningful harm, such as incorrect reportability, primary site, or
class of case.
Multi-Part Miss Flag True if the ground truth contained multiple components and the model missed at least one part entirely.
Rule Conflict Flag True if the model cited contradictory rules within the same answer.
Figure 13compares the frequency with which GPT and Gemini independently raised each binary flag with the frequency
at which both judges flagged the same response. The difference between the individual-judge rates and the consensus
rate shows how much agreement existed between the two evaluators. Agreement was relatively high for clinical safety
and multi-part miss flags, especially on harder questions, where both judges frequently identified the same responses
as problematic. The largest differences appeared for the hallucination flag, particularly in the easy and medium tiers.
For example, Gemini flagged hallucination more often than GPT, while the two-judge consensus rate was lower than
either individual rate. This suggests that the judges differed more in how they interpreted potential hallucinated or
unsupported coding references than they did for the other flag categories. Rule-conflict flags were rare across both
judges and all difficulty levels. Using the consensus rule therefore provides a more conservative measure by counting a
flag only when both judges identify the same issue in the same response.
Easy Medium Hard0%20%40%60%80%100%% of Responses Flagged42%46%69%
61%69%88%
37%41%67%Hallucination
GPT
Gemini
Both
Easy Medium Hard36%64%80%
37%65%77%
31%58%71%Clinical Safety
Easy Medium Hard22%45%84%
6%26%72%
5%22%70%Multi-Part Miss
Easy Medium Hard3% 4%2% 3% 4%2% 1% 2% 1%Rule Conflict
Figure 13: Two-judge agreement on binary evaluation flags across difficulty tiers (pooled over all the models)
C.4 LLM-as-a-Judge Prompt Template
Figure 14shows the evaluation prompt used in our LLM-as-a-Judge experiments. For each question, the judge
compared the generated response with the corresponding ground-truth answer and evaluated it across six dimensions:
correctness, faithfulness, completeness, reasoning, citation accuracy, and tone. In addition, four binary flags were used
to capture potential reliability and safety concerns, including hallucinated citations or rules, clinically meaningful errors,
missed components in multi-part questions, and conflicting rules within the same response. The prompt also required
a short reasoning explanation to provide context for the assigned scores and flags. This structured rubric was used
consistently across all models and difficulty tiers to support comparable evaluation.
18

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
System:You are an expert evaluator for medical coding QnA systems, specializing in cancer registry coding (ICD-O-3,
STORE Manual, Solid Tumor Rules, and MPH Rules). You evaluate LLM-generated answers against ground-truth answers.
This system is designed to help and teach cancer registrars. Tone and pedagogical quality matter as much as technical accuracy.
You must respond only with valid JSON.
User:Evaluate the LLM answer against the ground truth for this medical coding question.
QUESTION:{question}
GROUND TRUTH:{ground_truth}
LLM ANSWER:{llm_answer}
DIFFICULTY:{difficulty}
Rate the answer on six dimensions (each 1–5):Correctness, Faithfulness, Completeness, Reasoning, Citation Accuracy,
and Tone.
Provide four binary flags (true/false):Hallucination Flag, Clinical Safety Flag, Multi-Part Miss Flag, and Rule Conflict
Flag.
Provide aReasoning Explanationof 2–3 sentences.
Figure 14: LLM-as-a-Judge prompt used to evaluate generated answers against the ground truth. The prompt assigns
scores across six evaluation dimensions and four binary safety and reliability flags.
C.5 Ablation Study: Caution Flags
Table 3presents the response-caution rubric used in our experiments to evaluate how models behave when faced with
uncertain or ambiguous cancer registry questions. It defines each caution flag, the type of response behavior it captures,
example trigger phrases, and the clinical significance of that behavior.
Table 3: Response Caution Flags, Definitions, and Clinical Significance
Flag What It Captures Example Trigger Phrase Clinical Significance
Cannot De-
termineModel explicitly states it can-
not identify the correct code or
make a coding decision.“The primary site cannot be de-
termined from the information
provided. ”Model recognizes that the clinical evi-
dence is insufficient to assign a code; ap-
propriate behavior for ambiguous cases.
Conditional
AnswerModel gives an answer con-
tingent on information not
present in the question.“If the biopsy confirms malig-
nancy, the site would be coded
as C25.0. ”Model hedges the coding decision behind
a condition; reduces actionability but re-
flects appropriate uncertainty.
Needs More
InfoModel explicitly requests ad-
ditional clinical details before
committing.“Additional pathology report in-
formation is needed to code this
case accurately. ”Model identifies missing evidence; use-
ful for registrars but indicates incomplete
case documentation.
Acknowledges
AmbiguityModel recognizes that the case
involves ambiguous terminol-
ogy or conflicting evidence.“This is a complex case where
the terminology ‘consistent with’
creates ambiguity under current
rules. ”Model identifies ambiguous terminology,
which may have different coding impli-
cations depending on diagnosis year and
applicable rules.
Explicit Re-
fusalModel directly declines to pro-
vide a coding answer.“I am unable to provide a defini-
tive code for this case without fur-
ther confirmation. ”Strong form of response caution; the
model refuses rather than risk providing
unsupported guidance.
Defers to Ex-
pertModel recommends consult-
ing a physician, pathologist, or
senior registrar instead of an-
swering.“This case should be discussed
with the treating physician before
coding. ”Model recognizes its limitations and redi-
rects to human expertise; appropriate for
complex or high-stakes cases.
Any Refusal Composite flag; true if any of
the six flags above is raised.Any of the above. Overall measure of response caution used
to compare RAG Proprietary, RAG Lo-
cal, and Baseline groups.
We analyzed how often each model group hesitated, qualified its answer, or refused to give a definite coding decision.
as shown inTable 4. RAG Proprietary models showed the highest overall caution/refusal rate at 23.1%, compared with
15.3% for RAG Local models and 15.2% for the No-RAG baseline. This suggests that proprietary models were more
likely to signal uncertainty instead of immediately committing to an answer. Because the baseline showed a refusal rate
similar to the local RAG models, this behavior does not appear to be caused by retrieval alone and may also reflect
differences in how the underlying model families are trained and aligned.
19

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Table 4: Refusal Rate and Containment by Model Group
Group Mean Refusal Rate Containment
RAG Proprietary 23.1% 0.288
RAG Local 15.3% 0.401
Baseline (No RAG) 15.2% 0.225
The largest difference came from the Acknowledges Ambiguity flag as shown inTable 5. Proprietary RAG models
flagged ambiguity in 17.8% of responses, compared with 10.3% for local RAG models and 6.3% for the baseline.
Proprietary models also showed more explicit refusals (3.4% vs. 0.7%), while local models were somewhat more likely
to state that they needed more information (2.7% vs. 1.6%). The baseline behaved differently, with a much higher rate
of conditional answers (6.9%). Overall, the main reason proprietary models appeared more cautious was that they were
more likely to recognize and explicitly state that a case was ambiguous.
Table 5: Response Caution Flag Rates by Model Group
Flag RAG Proprietary RAG Local Baseline
Cannot Determine 2.8% 3.6% 0.8%
Conditional Answer 3.0% 0.4% 6.9%
Needs More Info 1.6% 2.7% 0.8%
Ambiguity Flag 17.8% 10.3% 6.3%
Explicit Refusal 3.4% 0.7% 0.8%
Defers to Expert 0.0% 0.0% 0.4%
We also observed that models with higher refusal or caution rates tended to include less of the expected ground-truth
content in their answers as shown inTable 4. RAG Local models had the highest containment score (0.401) and a lower
refusal rate (15.3%), while RAG Proprietary models had lower containment (0.288) and a higher refusal rate (23.1%).
This suggests that models that commit to an answer are more likely to reproduce the codes, terminology, and details
present in the reference response. In contrast, a model that says the case is uncertain or cannot be determined may
provide a safer response but naturally include fewer of the expected answer terms.
At the individual-model level, the Qwen models showed some of the highest containment scores (Containment measures
how much of the ground-truth answer’s expected words, codes, or terms are present in the generated response), including
Qwen3.6-27B (0.457), Qwen3.5-122B (0.444), and Qwen3.6-35B (0.435). These models also produced relatively long
responses and had comparatively low refusal rates as shown inTable 6. This combination likely contributed to their
stronger overlap with the reference answers: they were more willing to give a detailed and complete response rather
than stop at uncertainty. However, this should not be interpreted as proof that these models had better clinical reasoning.
Their stronger scores may partly reflect response length and willingness to commit to an answer.
Table 6: Individual-model containment, refusal rate, and mean response length
Model Type Containment Refusal Mean Len
qwen3.6-27b RAG Local 0.457 16.2% 214
qwen3.5-122b RAG Local 0.444 16.2% 203
qwen-qwen3.6-35b RAG Local 0.435 16.2% 184
openai-gpt-oss-120b RAG Local 0.428 16.2% 247
medgemma-27b RAG Local 0.367 16.2% 146
gemini-2.0-flash RAG Proprietary 0.328 21.7% 127
gpt-4o-mini RAG Proprietary 0.311 16.2% 269
C.6 Example Question by Difficulty Tier
Table 7 provides representative examples of easy, medium, and hard cancer registry coding questions used in the
evaluation.
20

CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars
Table 7: Example Questions by Difficulty Level
Difficulty Example Question
Easy What is the code for site and for histology in a urothelial cell carcinoma with squamous features, bladder?
Medium Patient with history of malignant melanoma. Lesion frontal lobe brain biopsy: Poorly differentiated
malignant melanoma. What is the primary site?
Hard A 95-year-old woman presented to the ER with back pain. CT and ultrasound showed a 4.3 cm pancreatic
head mass with dilatation of the common and intrahepatic bile ducts. A consult note states “likely
pancreatic ca.” An attempted ERCP noted a malignant stricture of the common bile duct but was aborted
due to technical difficulty; no biopsy or stent was placed. Should this case be reported? What is the
primary site? Should the aborted ERCP be coded?
21
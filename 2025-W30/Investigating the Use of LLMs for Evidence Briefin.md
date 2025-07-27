# Investigating the Use of LLMs for Evidence Briefings Generation in Software Engineering

**Authors**: Mauro Marcelino, Marcos Alves, Bianca Trinkenreich, Bruno Cartaxo, Sérgio Soares, Simone D. J. Barbosa, Marcos Kalinowski

**Published**: 2025-07-21 17:37:23

**PDF URL**: [http://arxiv.org/pdf/2507.15828v1](http://arxiv.org/pdf/2507.15828v1)

## Abstract
[Context] An evidence briefing is a concise and objective transfer medium
that can present the main findings of a study to software engineers in the
industry. Although practitioners and researchers have deemed Evidence Briefings
useful, their production requires manual labor, which may be a significant
challenge to their broad adoption. [Goal] The goal of this registered report is
to describe an experimental protocol for evaluating LLM-generated evidence
briefings for secondary studies in terms of content fidelity, ease of
understanding, and usefulness, as perceived by researchers and practitioners,
compared to human-made briefings. [Method] We developed an RAG-based LLM tool
to generate evidence briefings. We used the tool to automatically generate two
evidence briefings that had been manually generated in previous research
efforts. We designed a controlled experiment to evaluate how the LLM-generated
briefings compare to the human-made ones regarding perceived content fidelity,
ease of understanding, and usefulness. [Results] To be reported after the
experimental trials. [Conclusion] Depending on the experiment results.

## Full Text


<!-- PDF content starts -->

Investigating the Use of LLMs for Evidence
Briefings Generation in Software Engineering
Mauro Marcelino Marcos Alves Bianca Trinkenreich
PUC-Rio FITec Technological Innovations Colorado State University
Rio de Janeiro, Brazil Belo Horizonte, Brazil Fort Collins, USA
moliveira@inf.puc-rio.br marcosaalves@fitec.org.br Bianca.trinkenreich@colostate.edu
Bruno Cartaxo S ´ergio Soares Simone D.J. Barbosa Marcos Kalinowski
IFPE UFPE PUC-Rio PUC-Rio
Recife, Brazil Recife, Brazil Rio de Janeiro, Brazil Rio de Janeiro, Brazil
email@brunocartaxo.com scbs@cin.ufpe.br simone@inf.puc-rio.br kalinowski@inf.puc-rio.br
Abstract —[Context] An evidence briefing is a concise and
objective transfer medium that can present the main findings
of a study to software engineers in the industry. Although
practitioners and researchers have deemed Evidence Briefings
useful, their production requires manual labor, which may be
a significant challenge to their broad adoption. [Goal] The goal
of this registered report is to describe an experimental protocol
for evaluating LLM-generated evidence briefings for secondary
studies in terms of content fidelity, ease of understanding,
and usefulness, as perceived by researchers and practitioners,
compared to human-made briefings. [Method] We developed a
RAG-based LLM tool to generate evidence briefings. We used
the tool to automatically generate two evidence briefings that
had been manually generated in previous research efforts. We
designed a controlled experiment to evaluate how the LLM-
generated briefings compare to the human-made ones regarding
perceived content fidelity, ease of understanding, and usefulness.
[Results] To be reported after the experimental trials. [Conclu-
sion] Depending on the experiment results.
Index Terms —Evidence Briefing, LLMs for Synthesis, Con-
trolled Experiment, Evidence-Based Software Engineering.
I. I NTRODUCTION
Evidence-Based Software Engineering (EBSE), introduced
by Kitchenham et al. [16], is an adaptation of evidence-based
principles from medicine designed to bring scientific rigor to
the practice of Software Engineering (SE). It emphasizes the
systematic collection, evaluation, and synthesis of empirical
evidence to support decision-making processes, mainly in
the form of Systematic Literature Reviews (SLR). The core
objectives of SLRs include summarizing existing evidence,
identifying research gaps, and establishing an empirical foun-
dation to inform both academic inquiry and industry practices.
However, one of the main challenges in SE research is
the limited applicability in industrial contexts [24]. Studies
have shown that research findings are often misaligned with
industry needs and do not adequately reflect practical con-
straints [11]. With respect to SLRs, Santos and Silva [20]
reported that only a small fraction of SE researchers see
their SLR directly influencing industry practices. Only six
out of the 44 researchers interviewed claimed that their workhad somehow influenced industry practice. Similarly, Budgen
et al. [6] examined 49 studies and found that while SLRs
aggregate valuable insights, their translation into actionable
knowledge remains a challenge, indicating that the limited in-
dustrial impact represents a persistent trend. Other works have
highlighted ineffective mechanisms for transferring knowledge
into actionable insights as a key factor contributing to this gap
[1], [12].
These findings underscore the misalignment between re-
search priorities and real-world industry needs, highlighting
the need for better mechanisms to translate academic research
into practical applications. Ensuring that research findings
are reliable, easily accessible, and actionable for industry
professionals remains a critical challenge. Beecham et al. [5],
for instance, emphasized this issue by arguing that the medium
used to communicate research findings to practitioners plays
an important role in bridging this gap. To address these
limitations, Cartaxo et al. [8] introduced evidence briefings as
an alternative knowledge transfer mechanism in SE. Evidence
briefings condense the most relevant findings from empirical
studies into concise, structured, and accessible reports aimed
at industry practitioners. However, despite their potential ben-
efits, the adoption of evidence briefings remains limited. The
manual effort and cognitive workload involved in their creation
lead to scalability issues, hindering widespread adoption in
both research and industrial settings.
The advancement of artificial intelligence has introduced
new possibilities for automating research synthesis. Machine
Learning (ML) algorithms have historically been used in
text summarization, but Large Language Models (LLM) [22]
have taken this a step further by leveraging deep contextual
understanding and pre-trained knowledge to extract and syn-
thesize complex text [27]. Recent studies have demonstrated
the potential of LLMs for automating the creation of research
synthesis in different domains, including summarizing medical
evidence [21] and the generation of policy-oriented summaries
based on research [19].
While these aforementioned studies indicate that LLMs havearXiv:2507.15828v1  [cs.SE]  21 Jul 2025

the potential to reduce manual effort and improve accessibility
in research synthesis, challenges remain. LLMs are known to
generate hallucinated content, i.e.where the model generates
information that is factually incorrect or not present in the
source text [27], require careful prompt engineering [13],
and may sometimes produce overly confident or misleading
summaries [21]. These issues raise concerns regarding the re-
liability and applicability of AI-generated research summaries,
particularly in high-stakes decision-making.
Additionally, existing automated evaluation metrics, such as
ROUGE [3] and BertScore [28], have been shown to correlate
weakly with human assessments in key aspects like coher-
ence and factual consistency [10]. Similarly, Tang et al. [21]
found that existing automatic metrics fall short in accurately
measuring factual inconsistency and human preferences in
medical evidence summarization, further reinforcing the need
for empirical studies involving humans.
In this study, we investigate whether automatically gener-
ated Evidence Briefings can be suitable for broader adoption
in SE. Therefore, we designed an experiment with human
subjects following the recommendations by Wohlin et al. [25].
The main goal of the experiment is to assess whether LLMs
can generate Evidence Briefings that are faithful to the evi-
dence contained in the original text and easily digestible by
practitioners.
This registered report is organized as follows: Section II
discusses Evidence Briefings in SE and the use of LLMs
for research synthesis; Section III describes the experimental
protocol in detail; and Section IV presents the next steps to
conclude the study.
II. B ACKGROUND AND RELATED WORK
A. Evidence Briefings in SE
An evidence briefing is a one-page document that summa-
rizes the key findings of an empirical study [7]. Its concise
format is essential for presenting evidence in a clear and en-
gaging manner, making it more accessible to both practitioners
and researchers. Compared to SLR papers, evidence briefings
provide a more practical and efficient medium for acquiring
knowledge [8].
Effectively transferring research findings into practice is a
key aspect for bridging the gap between theory and practice.
Budgen et al. [6] emphasized that SLRs alone do not effec-
tively support the translation of research into practice, rein-
forcing the need for a more effective medium that simplifies
research findings for use in the real world and industry.
To achieve this, an evidence briefing should include the
following sections, as suggested in [6], [8]:
•Title and Description : The title guides the reader to the
main focus of the briefing while maintaining simplicity
to make it more appealing to practitioners. To enhance
accessibility, technical terms that refer to research meth-
ods should be avoided, such as systematic review. Then,
include a brief paragraph summarizing the briefing’s
purpose, providing relevant context for the reader.•Main Findings : This section presents a concise summary
of the key findings from the research paper, highlighting
the most important insights.
•Audience and References : This section specifies the
target audience and provides references to the original
research for further reading.
Cartaxo et al. [8] evaluated evidence briefings through sur-
veys with StackExchange practitioners and systematic review
authors. Practitioners perceived the briefings as clear and
relevant, but noted limitations in addressing highly specific
questions, i.e., they might miss details. SLR authors recog-
nized their accessibility and effectiveness in summarizing key
findings. Despite their benefits, their adoption remains limited.
Automating their generation through LLMs may represent
an opportunity to enhance scalability while maintaining their
effectiveness as a means of knowledge transfer.
B. LLMs for Research Synthesis
Research synthesis has mostly relied on manual effort,
with experts selecting, analyzing, and summarizing findings
from multiple primary studies. As text summarization methods
evolve, research synthesis becomes increasingly feasible, re-
ducing the effort required to extract insights from the literature.
Early summarization techniques employed extractive methods,
selecting key sentences based on statistical features such as
term frequency and sentence position. While these methods
provided partial automation, researchers still needed to refine
and interpret the extracted summaries manually [27].
The advent of ML-powered models marked a shift toward
more natural and concise summaries, capable of rephrasing
content while preserving its meaning. Recent advancements
in LLMs have further refined this process, leveraging deep
contextual understanding and pre-trained knowledge to syn-
thesize complex research findings [27]. LLMs have proven
to be a valuable tool in the Natural Language Processing
field, enabling a wide range of tasks, including classification,
text generation, question answering, sentiment analysis, and
summarization.
Rosenfeld et al. [19] introduced an AI-driven framework
to summarize evidence to effectively communicate complex
research findings to policymakers. Zhang et al. [26] compared
different LLMs for evidence summarization, discussing how
these models can be optimized to generate reliable, industry-
relevant research summaries. Tang et al. [21] focused on
benchmarking LLMs in summarizing research findings, mak-
ing them directly applicable for generating evidence sum-
maries for professionals in academia and industry.
Despite these advancements, LLMs still present challenges
in research synthesis. According to Tang et al. [21], these
include the need for effective prompt engineering, risks of
generating hallucinated insights, and difficulty in capturing
contextual nuances. They also highlight concerns regarding
inconsistencies in LLM-generated summaries, warning that
these models can sometimes produce overly convincing or
uncertain statements, leading to potential misinformation.

III. E XPERIMENTAL STUDY PLAN
In order to determine whether LLM-generated evidence
briefings can be effective within the SE domain to com-
municate research findings to practitioners, we designed our
experimental investigation following the recommendations by
Wohlin et al. [25], see Figure 1.
Fig. 1. Experiment planning steps [25].
A. Goal Definition
The main goal is to investigate whether automatically gen-
erated evidence briefings can be suitable for broader adoption.
This goal was tailored into two specific sub-goals, defined us-
ing the Goal-Question-Metric (GQM) goal definition template
proposed by Basili et al. [4] as follows.
GOAL 1 :Analyze LLM-based evidence briefing generation
for the purpose of characterizing with respect to their content
fidelity from the point of view of researchers in the context of
evaluating automatically generated and human-made evidence
briefings following the same structured template.
GOAL 2 :Analyze LLM-based evidence briefing generation
for the purpose of characterizing with respect to their ease
of understanding and perceived usefulness from the point of
view of practitioners in the context of evaluating automatically
generated and human-made evidence briefings following the
same structured template.
From GOAL 1, we derived the following research question
(RQ):
•RQ1 : How do researchers perceive the content fidelity
of automatically generated and human-made evidence
briefings?
From GOAL 2, we derived the following RQs:
•RQ2 : How do practitioners perceive the ease of un-
derstanding of automatically generated and human-made
evidence briefings?
•RQ3 : How do practitioners perceive the usefulness of
automatically generated and human-made evidence brief-
ings?B. Context Selection
The selected context consists of two independent offline
controlled experiments, one involving researchers and the
other industry practitioners, each designed to evaluate different
aspects of automatically generated and human-made evidence
briefings.
•Academic Experiment: Evaluating Content Fidelity .
In the first experiment, researchers will assess whether
evidence briefings accurately capture the key information
from the original research articles. Each participant will
be presented with two examples of full-length research
papers. For each research paper, the researcher will
receive a randomly assigned version of the respective ev-
idence briefing, either automatically generated or human-
made. Researchers will then evaluate how well the brief-
ing preserves the core findings and conclusions of the
original work.
•Industry Experiment: Evaluating Ease of Understand-
ing and Usefulness . The second experiment targets SE
practitioners, focusing on how well they can understand
and interpret the information presented in evidence brief-
ings. Unlike researchers, practitioners will not be exposed
to the full research articles. Instead, they will be randomly
assigned to read one human-made and one automatically
generated evidence briefing. Afterward, they will answer
a set of questions designed to assess their perception of
the briefing’s ease of understanding and usefulness.
This division of evaluation roles reflects the expertise and
perspectives of each group. Researchers are best suited to
assess research content fidelity, as they are familiar with
academic standards and able to verify how accurately the
briefing represents the source material. Practitioners, on the
other hand, are the intended consumers of these briefings,
making them ideal evaluators for aspects such as ease of
understanding and usefulness in their practice.
C. Hypothesis Formulation
H01: There is no difference between the researchers’ per-
ception of content fidelity for human-made and automatically
generated evidence briefings.
H02: There is no difference between the practitioners’
perception of ease of understanding for human-made and
automatically generated evidence briefings.
H03: There is no difference between the practitioners’
perception of usefulness for human-made and automatically
generated evidence briefings.
D. Variable Selection
The independent variable of interest (experimental factor)
corresponds to the type of evidence briefing presented to the
participants. Hence, the generation method is central to the
study design.
•Automatically generated evidence briefing : Produced
by an LLM-based system.
•Human-made evidence briefing : Manually produced by
researchers.

Additional independent variables collected within the par-
ticipant characterization include:
•For researchers : Familiarity with systematic reviews,
academic role, and academic degree.
•For practitioners : Years of professional experience, role,
academic degree, familiarity with each topic related to the
selected evidence briefings.
The study defines three dependent variables, each corre-
sponding to a specific research question:
•Content Fidelity : Researchers’ perception of how faith-
fully the evidence briefing represents the content of the
original research article.
•Ease of Understanding : Practitioners’ assessment of the
clarity and comprehensibility of the evidence briefing.
•Usefulness : Practitioners’ perception of how useful the
evidence briefing is for their professional practice.
Following recommendations for construct validity [9] [15],
we measure each dependent variable using multiple items
that reflect its theoretical components. For researchers, the
dependent variable content fidelity is assessed through mul-
tiple items adapted from the error taxonomy proposed by
Tang et al. [21]. While Tang identified inconsistency types
through qualitative coding, we incorporated these categories
directly into the survey to enable a more focused evaluation.
The items target three key dimensions: contradiction (align-
ment of conclusions), certainty illusion (accuracy of conveyed
confidence), and fabricated content (presence of unsupported
claims). This structured approach allows for a more detailed
assessment of factual consistency in LLM-generated briefings.
For practitioners, ease of understanding is evaluated with
questions relating to the clarity of the language used, the
logical structure of the text, and conciseness. This decom-
position is further aligned with Creswell’s [9] discussion of
readability, which emphasizes consistent terminology, logical
flow between ideas, and coherence across sections. Finally,
forusefulness in the context of SE, we assessed it based on
the perceived relevance of the insights and recommendations
for challenges or tasks related to professional practice and
actionability ( i.e., applicability in the work context).
The participant is presented with affirmative statements
related to each of the items that compose the dependent
variables and asked to rate their level of agreement according
to a 7-points Likert scale: 1 - Strongly Disagree, 2 - Disagree
3- Slightly Disagree, 4 - I neither agree nor disagree, 5 -
Slightly Agree, 6 - Agree, and 7 - Strongly Agree.
Additionally, the evaluation questions include an optional
open-text field, allowing participants to elaborate on their
answers and enabling a complementary qualitative analysis.
This follows a concurrent embedded mixed-methods strategy
[9], where qualitative insights help interpret and contextualize
quantitative trends; we will make a composite assessment from
the open questions to reside side by side with quantitative
analysis and add depth and strengthen the overall validity of
the findings.E. Selection of Subjects
The SE researchers and practitioners for the two studies
will be selected based on convenience sampling and open
invitations on social media. To answer RQ1 (content fidelity),
in the first study, we will recruit a sample of SE researchers. To
answer RQ2/RQ3 (understandability & usefulness), in the sec-
ond study, we will recruit a separate sample of industry practi-
tioners. They will be randomly assigned to the treatments, and
we will use their independent variable characterization to apply
blocking principles during the analysis. Furthermore, we will
describe the samples in detail based on their characterization
and carefully discuss their representativeness.
Despite using convenience sampling, it is noteworthy that
participants are randomly assigned to the treatments and do
not know which EB was generated by humans and which one
was generated automatically. I.e., they could not intentionally
favor a specific treatment, even if they would like to.
F . Choice of Design Type
The goal is to investigate whether LLM-generated evidence
briefings can effectively communicate research findings to
practitioners. Therefore, we conduct a controlled experiment
with researchers to evaluate the content fidelity of the gen-
erated evidence briefings, and another controlled experiment
with practitioners to assess the perceived ease of understanding
and usefulness.
We adopted a completely randomized one-factor, two-
treatment crossover design for each of these experiments, as
illustrated in Figure 2. This design was chosen to isolate inter-
subject variability and learning effects. The factor concerns
the type of evidence briefing and the treatments involved
using LLM or manually generated ones. Furthermore, to avoid
skewing the results, participants are randomly assigned to a
pair of evidence briefings (LLM first and manual second, or
manual first and LLM second). It is noteworthy that each
of the two evidence briefings has both versions and that the
human-made evidence briefings were generated in previous
and independent research efforts [8].
Fig. 2. Crossover design

The experimental tasks for researchers concerned reading
a research paper and the evidence briefing and evaluating
content fidelity. For practitioners, the task concerned reading
the evidence briefing and answering questions regarding ease
of understanding and usefulness. In both cases, participants
are not aware of the source of the briefing.
G. Instrumentation
We designed separate online questionnaires using Google
Forms for the researchers and for the practitioners. The study
protocol and its instruments were officially approved by the
CSU institutional ethics committee. Each questionnaire con-
sists of three sections: a consent form, a participant characteri-
zation form, and a main section with substantive questions that
focus on the dependent variables. We evaluated both versions
of each questionnaire (cf.Figure 2) in a pilot study with two
researchers and two practitioners, and no further adjustments
were deemed necessary.
1) Consent form: provides a general overview of the re-
search and its objectives and reports ethical considerations
as suggested by Badampudi et al. [2]. The only anticipated
risk concerns practitioners potentially being presented with
information that is wrong or hallucinated.
2) Participant Characterization Form: This form was de-
signed to collect the aforementioned demographic character-
ization information. It ensures that participants are properly
categorized based on their research or industry experience,
allowing for a more precise analysis of their responses.
3) Main Section: The main section of the questionnaire is
designed to directly address the study’s research questions,
described in Section III-A. Participants are asked to evaluate
evidence briefings related to two studies, one on pair program-
ming [14] and one on definition of done [18]. These brief-
ings were produced manually in the context of the evidence
briefings’ usefulness evaluation [7]. The studies were selected
from a set of 56 evidence briefings because they summarize
secondary studies with rigorous quality assessment procedures
that were published in high-quality peer-reviewed venues, on
topics that are relevant to both academia and industry. As
highlighted by Kuhrmann et al. [17], agile software devel-
opment is widely used by SE teams and is primarily shaped
by its practices, reinforcing the importance of understanding
the evidence related to such practices. While all participants
evaluate briefings covering both topics (pair programming and
definition of done), researchers are additionally provided with
the original research articles to assess content fidelity, whereas
practitioners rely solely on the briefings to evaluate ease of
understanding and perceived usefulness.
In order to automatically generate the evidence briefings,
a tool was developed that leverages LLMs via API calls. We
describe how the LLM was used according to the evaluation
guidelines for empirical studies in SE involving LLMs [23]1,
declaring the LLM usage and role, the model version and date,
prompts and their development, and the tool architecture and
supplemental data (see Table I) .
1https://llm-guidelines.org/guidelines/To ensure that the model produces structured and readable
evidence briefings aligned with practitioner needs, we crafted
a prompt using instruction-based prompting techniques, as
outlined in prior work on prompt engineering [13]. The prompt
explicitly defines the task, introduces the evidence briefing
format, and includes structured constraints to guide the LLM’s
output. This approach aligns with key principles such as
clarity, context provision, and output formatting, as described
by Geroimenko [13].
To enhance the quality and coherence of the generated
summaries, a Retrieval-Augmented Generation (RAG) struc-
ture was employed. This strategy aligns with the prin-
ciple of providing contextual information as outlined by
Geroimenko [13], which emphasizes that supplying relevant
background helps guide the model towards more accurate,
nuanced, and audience-appropriate responses. In our case,
a ChromaDB database containing 54 human-made evidence
briefings (the set of 56, excluding the two studies chosen
for the experiment) is used as the retrieval source. Before
generating a briefing, snippets corresponding to each required
section of the document are retrieved, offering the model
grounded examples to adapt its tone, content structure, and
language clarity. The tools’ user interface also allows for the
optional provision of keywords such as ‘Systematic Literature
Review’ or ‘Scrum’ to improve the relevance of the retrieved
context to the work to be summarized. We did not use this
feature for the evidence briefings of our experiment.
TABLE I
LLM C ONFIGURATION AND USAGE SUMMARY
Aspect Details
Model GPT-4-o-mini (gpt-4-0125-preview)
Provider OpenAI API
Access Date March 2025
Temperature 0.5
Top-p 1.0
Max Tokens 1024 (output limit)
Prompting Strategy Instruction-based prompting
RAG Mechanism Retrieval via ChromaDB
Retrieved Examples 54 human-made evidence briefings
To support further investigation and facilitate reproducibil-
ity, all materials used in our study are available in our
immutable open science repository2. These include the final
prompt used, all the code and instructions necessary to run the
tool, the human-made and LLM-generated Evidence Briefings,
as well as the survey instruments.
H. Threats to validity
We discuss the threats to validity based on the classification
by Wohlin et al. [25]: construct ,internal ,external , and con-
clusion validity. For each category, we describe the key threats
and our mitigation strategies.
2https://doi.org/10.5281/zenodo.15233046

Construct Validity: A potential threat lies in the configu-
ration of the LLM. Changes in the prompt formulation, tem-
perature, top-p parameters, or retrieval corpus may affect the
generated output and, consequently, the consistency of results.
We mitigated this threat by following reporting guidelines for
empirical studies in SE involving LLMs [23] and fully fixing
and documenting all generation parameters, including prompt,
temperature, top-p, and retrieval configuration. All outputs
were archived and made available through our open science
repository.
Another concern relates to the adequacy of our survey in-
struments in capturing the theoretical constructs of interest. To
address this, each construct was evaluated using multiple items
to enhance measurement reliability. We adapted established
evaluation taxonomies—such as Tang et al. [21] for content
fidelity—and followed guidelines for assessing readability and
usefulness in software engineering contexts [9] [11] [15].
Internal Validity: One internal threat is related to order
effects in the crossover design. Participants may be influenced
by the sequence in which they receive the briefings. To
mitigate this, we adopted a counterbalanced crossover design
in which participants were randomly assigned to receive the
human-made or LLM-generated briefing first.
A second concern is carryover effects , where exposure to
the first briefing may influence the evaluation of the second.
We minimized this threat by ensuring that each participant
assessed two different topics (pair programming and definition
of done), reducing priming effects between treatments.
External Validity: The use of only two research papers may
limit the generalizability of our findings to other areas of SE.
To mitigate this, we selected papers that are both method-
ologically well-grounded and highly relevant to practitioners.
Still, we acknowledge the need for replication across broader
domains.
Furthermore, differences in validation rigor between the
human-made briefings represent an additional threat [7]. The
pair programming briefing had been previously validated [8],
whereas the definition of done briefing was produced sepa-
rately and did not undergo the same validation process. We
acknowledge this as a limitation and interpret comparisons
accordingly.
Finally, a selection-treatment interaction may occur if par-
ticipants lack familiarity with the topic of the briefing, leading
to biased usefulness ratings. We collected self-reported famil-
iarity data and will use it as a blocking variable during the
analysis phase.
Conclusion Validity: A threat to conclusion validity con-
cerns statistical power. If the sample size is insufficient, the
experiment may fail to detect significant effects. We will care-
fully consider sample size and representativeness throughout
the experiment.
IV. N EXT STEPS
The next steps of this research include conducting the exper-
iment as specified in the registered report protocol, analyzing
the data, testing the hypotheses using statistically appropriatemethods, answering the research questions, discussing the
results, and writing the complete journal paper to disseminate
the findings to the community.
REFERENCES
[1] Deniz Akdur. The design of a survey on bridging the gap between
software industry expectations and academia. In 2019 8th Mediterranean
Conference on Embedded Computing (MECO) , pages 1–5. IEEE, 2019.
[2] Deepika Badampudi, Farnaz Fotrousi, Bruno Cartaxo, and Muhammad
Usman. Reporting consent, anonymity and confidentiality procedures
adopted in empirical studies using human participants. e-Informatica
Software Engineering Journal , 16(1):220109, July 2022. Available
online: 22 Jul. 2022.
[3] Marcello Barbella and Genoveffa Tortora. Rouge metric evaluation for
text summarization techniques. Available at SSRN 4120317 , 2022.
[4] V .R. Basili and H.D. Rombach. The tame project: towards improvement-
oriented software environments. IEEE Transactions on Software Engi-
neering , 14(6):758–773, 1988.
[5] Sarah Beecham, P ´adraig O’Leary, Sean Baker, Ita Richardson, and
John Noll. Making software engineering research relevant. Computer ,
47(4):80–83, 2014.
[6] David Budgen, Pearl Brereton, Nikki Williams, and Sarah Drummond.
What support do systematic reviews provide for evidence-informed
teaching about software engineering practice? e-informatica software
engineering journal , 14(1):7–60, 2020.
[7] Bruno Cartaxo, Gustavo Pinto, and Sergio Soares. Towards a model
to transfer knowledge from software engineering research to practice.
Information and Software Technology , 97:80–82, 2018.
[8] Bruno Cartaxo, Gustavo Pinto, Elton Vieira, and S ´ergio Soares. Ev-
idence briefings: Towards a medium to transfer knowledge from sys-
tematic reviews to practitioners. In Proceedings of the 10th ACM/IEEE
international symposium on empirical software engineering and mea-
surement , ESEM ’16, pages 1–10, New York, NY , USA, 2016. Associ-
ation for Computing Machinery.
[9] John W Creswell and J David Creswell. Research design: Qualitative,
quantitative, and mixed methods approaches . Sage publications, 2017.
[10] Alexander R. Fabbri, Wojciech Kry ´sci´nski, Bryan McCann, Caiming
Xiong, Richard Socher, and Dragomir Radev. Summeval: Re-evaluating
summarization evaluation, 2021.
[11] Vahid Garousi, Markus Borg, and Markku Oivo. Practical relevance
of software engineering research: synthesizing the community’s voice.
Empirical Software Engineering , 25:1687–1754, 2020.
[12] Vahid Garousi, G ¨orkem Giray, Eray T ¨uz¨un, Cagatay Catal, and Michael
Felderer. Aligning software engineering education with industrial needs:
A meta-analysis. Journal of Systems and Software , 156:65–83, 2019.
[13] Vladimir Geroimenko. Key challenges in prompt engineering. In The
Essential Guide to Prompt Engineering: Key Principles, Techniques,
Challenges, and Security Risks , pages 85–102. Springer, 2025.
[14] Jo E. Hannay, Tore Dyb ˚a, Erik Arisholm, and Dag I.K. Sjøberg. The
effectiveness of pair programming: A meta-analysis. Information and
Software Technology , 51(7):1110–1122, 2009. Special Section: Software
Engineering for Secure Systems.
[15] Marcos Kalinowski, Allysson Allex Ara ´ujo, and Daniel Mendez. Teach-
ing Survey Research in Software Engineering , pages 501–527. Springer
Nature Switzerland, Cham, 2024.
[16] Barbara Kitchenham, Tore Dyb ˚a, and M. Jorgensen. Evidence-based
software engineering. In Proceedings. 26th International Conference on
Software Engineering , pages 273– 281, 06 2004.
[17] Marco Kuhrmann, Paolo Tell, Regina Hebig, et al. What makes
agile software development agile? IEEE Transactions on Software
Engineering , 48(9):3523–3539, 2022.
[18] Mirko Perkusich, Ana Silva, Thalles Ara ´ujo, Ednaldo Dilorenzo, Jo ˜ao
Nunes, Hyggo Almeida, and Angelo Perkusich. A systematic review on
the use of definition of done on agile software development projects.
InProceedings of the 21st international conference on evaluation and
assessment in software engineering , pages 364–373, 06 2017.
[19] Katherine A Rosenfeld, Maike Sonnewald, Sonia J Jindal, Kevin A
McCarthy, and Joshua L Proctor. Building understandable messaging
for policy and evidence review (bumper) with ai. arXiv preprint
arXiv:2407.12812 , 2024.

[20] Ronnie ES Santos and Fabio QB Da Silva. Motivation to perform
systematic reviews and their impact on software engineering practice.
In2013 ACM/IEEE international symposium on empirical software
engineering and measurement , pages 292–295. IEEE, 2013.
[21] Liyan Tang, Zhaoyi Sun, Betina Idnay, Jordan G Nestor, Ali Soroush,
Pierre A Elias, Ziyang Xu, Ying Ding, Greg Durrett, Justin F Rousseau,
et al. Evaluating large language models on medical evidence summa-
rization. NPJ digital medicine , 6(1):158, 2023.
[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention
is all you need. In Proceedings of the 31st International Conference
on Neural Information Processing Systems , NIPS’17, page 6000–6010,
Red Hook, NY , USA, 2017. Curran Associates Inc.
[23] Stefan Wagner, Marvin Mu ˜noz Bar ´on, Davide Falessi, and Sebastian
Baltes. Towards evaluation guidelines for empirical studies involving
llms. In Proceedings of the 2nd International Workshop on Method-
ological Issues with Empirical Studies in Software Engineering (WSESE
2025) , 2025. arXiv:2411.07668 [cs.SE].
[24] Titus Winters. Thoughts on applicability. Journal of Systems and
Software , 215:112086, 2024.
[25] Claes Wohlin, Per Runeson, Martin H ¨ost, Magnus C Ohlsson, Bj ¨o
Regnell, and Anders Wessl ´en.Experimentation in Software Engineering .
Springer Nature, 2024.
[26] Gongbo Zhang, Qiao Jin, Yiliang Zhou, Song Wang, Betina Idnay,
Yiming Luo, Elizabeth Park, Jordan G Nestor, Matthew E Spotnitz, Ali
Soroush, et al. Closing the gap between open source and commercial
large language models for medical evidence summarization. NPJ digital
medicine , 7(1):239, 2024.
[27] Haopeng Zhang, Philip S. Yu, and Jiawei Zhang. A systematic survey of
text summarization: From statistical methods to large language models,
2024.
[28] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav
Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint
arXiv:1904.09675 , 2019.
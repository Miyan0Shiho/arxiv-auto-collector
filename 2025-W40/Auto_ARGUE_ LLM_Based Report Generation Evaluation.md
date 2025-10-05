# Auto-ARGUE: LLM-Based Report Generation Evaluation

**Authors**: William Walden, Marc Mason, Orion Weller, Laura Dietz, Hannah Recknor, Bryan Li, Gabrielle Kaili-May Liu, Yu Hou, James Mayfield, Eugene Yang

**Published**: 2025-09-30 12:41:11

**PDF URL**: [http://arxiv.org/pdf/2509.26184v2](http://arxiv.org/pdf/2509.26184v2)

## Abstract
Generation of long-form, citation-backed reports is a primary use case for
retrieval augmented generation (RAG) systems. While open-source evaluation
tools exist for various RAG tasks, ones tailored to report generation are
lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based
implementation of the recent ARGUE framework for report generation evaluation.
We present analysis of Auto-ARGUE on the report generation pilot task from the
TREC 2024 NeuCLIR track, showing good system-level correlations with human
judgments. We further release a web app for visualization of Auto-ARGUE
outputs.

## Full Text


<!-- PDF content starts -->

Auto-ARGUE:
LLM-Based Report Generation Evaluation
William Walden1,3, Marc Mason2, Orion Weller3, Laura Dietz4,
Hannah Recknor1,3, Bryan Li5, Gabrielle Kaili-May Liu6, Yu Hou7,
Dawn Lawrie1,3, James Mayfield1,3, and Eugene Yang1,3
1Johns Hopkins University Human Language Technology Center of Excellence
2Solerity
3Johns Hopkins University
4University of New Hampshire
5University of Pennsylvania
6Yale University
7University of Maryland
{wwalden1,eyang35}@jh.edu
Abstract.Generationoflong-form,citation-backedreportsisaprimary
use case for retrieval augmented generation (RAG) systems. While open-
source evaluation tools exist for various RAG tasks, ones tailored tore-
port generationare lacking. Accordingly, we introduceAuto-ARGUE,
a robust LLM-based implementation of the recentARGUEframework
for report generation evaluation. We present analysis ofAuto-ARGUE
on the report generation pilot task from the TREC 2024 NeuCLIR track,
showing good system-level correlations with human judgments. We fur-
ther release a web app for visualization ofAuto-ARGUEoutputs.
Keywords:RAG·Report Generation·Automatic Evaluation
1 Introduction
AsRAGapplicationshaveproliferated,interestinautomaticevaluationmethod-
ologies for RAG tasks has grown in tandem. Many works have positioned them-
selves asgeneralsolutions for RAG evaluation [2,3,8,10], and while some evalu-
ation desiderata are shared across tasks,task-dependentconsiderations persist.
Here, we focus onreport generation(RG), a RAG task that aims to produce
a long-form, citation-attributed response to a complex user query. Two key fea-
tures distinguish RG from related RAG tasks—notably, long-form QA. First,
RG strongly foregrounds the identity of the user (orrequester): the same query
should yield different reports for requesters with different levels of education or
domain expertise. Second, the ideal report represents asummaryover the en-
tire corpus of the most user-critical information, thus stressingcoveragewhere
traditional QA emphasizes mereadequacyof the response.
While the “task-agnostic” RAG evaluation methodologies discussed above
could be applied to RG, our work focuses on theARGUEframework [7]—the
only framework designed expressly for RG—and presents three contributions:arXiv:2509.26184v2  [cs.IR]  1 Oct 2025

2 W. Walden et al.
1.Auto-ARGUE:Anautomatic,LLM-basedPythonimplementationofthe
ARGUEframework—the only publicly availableARGUEimplementation.
2.ARGUE-viz: A web app for visualizingAuto-ARGUEoutputs.
3.Case Study: Meta-evaluation results withAuto-ARGUEon the TREC
2024 NeuCLIR report generation pilot task [5].
Auto-ARGUEisconfigurable,easy-to-use,andextensibletootherRGdatasets,
adopting the new TREC format for RAG outputs8. We releaseAuto-ARGUE9
andARGUE-viz10to facilitate further work on automatic RG evaluation.
2 System Overview
2.1 Framework:ARGUE
Fig. 1.The ARGUE framework from [7], adapted with permission.
Figure1depictstheARGUEframework,whichevaluatesreportsviaatreeof
binary, sentence-leveljudgments(blue diamonds) about each sentence’s content
and citations. Depending on the path(s) traversed for each sentence, a report
may incur penalties (red circles), rewards (green), or neither (beige).
Inputs.ARGUEtakes as input: (1) the generatedreport; (2) thereport
request, including aproblem statementdescribing the information need and a
user storydescribing the requester; (3) the document collection used to generate
the report (with optional relevance judgments); and (4) a collection ofnuggets
(QA pairs), with links from nugget answers to documents that attest them.
ContentEvaluation.Followingmuchpriorwork[11,6,9,8,1],ARGUEeval-
uates reports’ coverage of relevant information via sets ofnuggets—QA pairs
that represent key questions an ideal report would address, paired with answers
linked to documents in the collection that attest them. Key questions that are
unanswerablefrom the collection can also be represented as nuggets with empty
answer sets. Each report sentence is assessed to determine which nugget ques-
tion(s) it answers correctly and reports are rewarded for each such nugget.
8Format schema and validator: https://github.com/hltcoe/rag-run-validator
9https://github.com/hltcoe/auto-argue
10https://github.com/hltcoe/argue-viz

Auto-ARGUE: LLM-Based Report Generation Evaluation 3
Citation Evaluation.Citations are assumed to support only the sentence
they are attached to. Sentences may bear≥0citations.Relevantcitations that
attest a sentence are rewarded; non-attesting or missing citations are penalized.
Metrics.ARGUEis agnostic to the magnitudes of its rewards and penalties
and to the metrics to be reported based on them. However, Mayfield et al.
recommend two metrics—sentence precisionandnugget recall—discussed below.
We refer readers to [7] for further details onARGUE.
2.2 Implementation:Auto-ARGUE
ARGUEleaves much to implementation, such as the judge, the magnitudes
of rewards/penalties, the source of nuggets and relevance assessments, and the
metrics to report. Here, we detail the choices we made withAuto-ARGUE.
– LLM Judge.An LLM judge is queried via few-shot prompts to obtain
binary (YES/NO) answers to all non-trivial judgments (starred in Figure 1)
forareportsentence.Answerstootherjudgmentsaredeterminedvialookup.
– Relevance.Auto-ARGUEdeems a document relevant iff it attestssome
nugget answer, determined via lookup in the nugget set or via the LLM.
– Nuggets.Nuggets may have multiple answers (each of which may be at-
tested by≥1document(s)) and come in two varieties: “AND” nuggets, for
whichallanswers must be given, and “OR” nuggets, for which onlyone
answer (of several) is required. Nuggets may also have importance labels
(“vital” or “okay”). Answer attestation is assessed per-sentence but answers
are aggregated acrossallsentences to identify correctly answered nuggets.
– Metrics.Auto-ARGUEimplementsthetwometricssuggestedbyMayfield
et al. [7].Sentence precisionis the proportion of sentences that are attested
byeachof their citations.Nugget recallis the proportion of nuggets correctly
answered by the report, with aweightedvariant that weights nuggets by im-
portance(okay=1.0;vital=2.0).Auto-ARGUEalsocomputes(un)weighted
F1 score(s) based on these two metrics, which can serve as anoverall score
for a report.Auto-ARGUEalso outputs several other fine-grained metrics.
2.3 Visualization:ARGUE-viz
ARGUE-vizis a simple Streamlit11app for visualizingAuto-ARGUEoutputs
forindividualruns,includingbothmetricsandfine-grainedjudgments.Userscan
toggle between per-topic results and aggregated metrics via radio buttons on a
sidebar. Per-topic results display core (e.g. sentence precision, nugget recall) and
non-core metrics and statistics (e.g. % relevant citations) for that topic as well as
detailed judgment information about the report generated for that topic. Judg-
ment information is displayed via two views: areport viewthat shows report-
level information about supported sentences and correctly answered nuggets and
asentence viewthat shows similar information at the sentence level (e.g. which
nugget answers are (not) attested by that sentence). Collectively, these features
enable fine-grained human analysis of errors to facilitate system development.
11https://streamlit.io

4 W. Walden et al.
3 Case Study: TREC NeuCLIR 2024 Report Generation
We evaluateAuto-ARGUEon the 51 runs from the RG pilot task of the TREC
2024 NeuCLIR track [5], which requires generatingEnglishreports from one of
threenon-Englishcollections (Chinese, Russian, Farsi—17 runs each). Human
assessors judged sentence support and nugget recall on reports for the same 21
topics for each run. Each topic has 10-20 nuggets, and assessors also identified
documents attesting each answer, thus providing the set of relevant documents.
Separately, we obtain the same metrics fromAuto-ARGUE, using Llama-
3.3 70B as the LLM judge [4] for D, C, G, and H judgments in Figure 1. B
judgments use the human relevance assessments. Since all NeuCLIR nuggets are
answerable, E and F judgments are not generated. We obtain system rankings
from (a) assessor-based and (b)Auto-ARGUE-based macro-average sentence
precision and nugget recall across all topics for each language. We compute
agreement between these rankings using (i)Kendall’s tauand (ii)accuracyw.r.t.
whether two Wilcoxon tests (one per ranking) for each pair of systemss 1ands 2
agree.Figure2presentstheresults.Broadly,weobservegoodagreementbetween
the two rankings on both metrics, with particularly strong results on sentence
precision. More capable LLM judges could yield still further improvements.
Fig. 2.Auto-ARGUEvs. human agreement on system rankings based on sentence
precision (left) and nugget recall (right) for the TREC 2024 NeuCLIR RG pilot task.
4 Conclusion
This work has introducedAuto-ARGUE—a robust, configurable, LLM-based
implementationoftheARGUEframeworkforreportgeneration(RG)evaluation—
as well asARGUE-viz—a simple Streamlit application for visualization of
Auto-ARGUEoutputs. Analysis ofAuto-ARGUEon the TREC 2024 Neu-
CLIR RG pilot task with a relatively small open-source LLM judge (Llama-3.3
70B) reveals good correlations with human judgments on system rankings based
on the key metrics of sentence precision and nugget recall. We release both
Auto-ARGUEandARGUE-vizto facilitate future work on RG evaluation.
Acknowledgments.The authors thank all the participants of the SCALE 2025 work-
shop at the JHU HLTCOE for valuable feedback on, and testing of,Auto-ARGUE.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.

Auto-ARGUE: LLM-Based Report Generation Evaluation 5
References
1. Alaofi, M., Arabzadeh, N., Clarke, C.L., Sanderson, M.: Generative information
retrieval evaluation. In: Information access in the era of generative ai, pp. 135–159.
Springer (2024)
2. Es, S., James, J., Anke, L.E., Schockaert, S.: Ragas: Automated evaluation of
retrieval augmented generation. In: Proceedings of the 18th Conference of the Eu-
ropean Chapter of the Association for Computational Linguistics: System Demon-
strations. pp. 150–158 (2024)
3. Gao, T., Yen, H., Yu, J., Chen, D.: Enabling large language models to generate
text with citations. In: Proceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing. pp. 6465–6488 (2023)
4. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Let-
man, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models.
arXiv preprint arXiv:2407.21783 (2024)
5. Lawrie, D., MacAvaney, S., Mayfield, J., McNamee, P., Oard, D.W., Soldaini, L.,
Yang, E.: Overview of the trec 2024 neuclir track. arXiv preprint arXiv:2509.14355
(2025)
6. Lin, J., Demner-Fushman, D.: Automatically evaluating answers to definition ques-
tions. In: Proceedings of Human Language Technology Conference and Conference
on Empirical Methods in Natural Language Processing. pp. 931–938 (2005)
7. Mayfield, J., Yang, E., Lawrie, D., MacAvaney, S., McNamee, P., Oard, D.W.,
Soldaini, L., Soboroff, I., Weller, O., Kayi, E., et al.: On the evaluation of machine-
generated reports. In: Proceedings of the 47th International ACM SIGIR Confer-
ence on Research and Development in Information Retrieval. pp. 1904–1915 (2024)
8. Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Soboroff, I.,
Dang, H.T., Lin, J.: The great nugget recall: Automating fact extraction and rag
evaluation with large language models. In: Proceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
pp. 180–190 (2025)
9. Rajput, S., Pavlu, V., Golbus, P.B., Aslam, J.A.: A nugget-based test collection
construction paradigm. In: Proceedings of the 20th ACM international conference
on Information and knowledge management. pp. 1945–1948 (2011)
10. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M.: ARES: An automated evalu-
ation framework for retrieval-augmented generation systems. In: Duh, K., Gomez,
H., Bethard, S. (eds.) Proceedings of the 2024 Conference of the North Ameri-
can Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers). pp. 338–354. Association for Computa-
tional Linguistics, Mexico City, Mexico (Jun 2024). https://doi.org/10.18653/v1/
2024.naacl-long.20, https://aclanthology.org/2024.naacl-long.20/
11. Voorhees, E.M., Dang, H.T.: Overview of the trec 2003 question answering track.
In: Proceedings of the Twelfth Text REtrieval Conference (TREC 2003). vol. 2003,
pp. 54–68 (2003)
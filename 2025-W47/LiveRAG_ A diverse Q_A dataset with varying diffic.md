# LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluation

**Authors**: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Alex Shtoff, Oren Somekh, Ran Tavory

**Published**: 2025-11-18 14:34:35

**PDF URL**: [https://arxiv.org/pdf/2511.14531v1](https://arxiv.org/pdf/2511.14531v1)

## Abstract
With Retrieval Augmented Generation (RAG) becoming more and more prominent in generative AI solutions, there is an emerging need for systematically evaluating their effectiveness. We introduce the LiveRAG benchmark, a publicly available dataset of 895 synthetic questions and answers designed to support systematic evaluation of RAG-based Q&A systems. This synthetic benchmark is derived from the one used during the SIGIR'2025 LiveRAG Challenge, where competitors were evaluated under strict time constraints. It is augmented with information that was not made available to competitors during the Challenge, such as the ground-truth answers, together with their associated supporting claims which were used for evaluating competitors' answers. In addition, each question is associated with estimated difficulty and discriminability scores, derived from applying an Item Response Theory model to competitors' responses. Our analysis highlights the benchmark's questions diversity, the wide range of their difficulty levels, and their usefulness in differentiating between system capabilities. The LiveRAG benchmark will hopefully help the community advance RAG research, conduct systematic evaluation, and develop more robust Q&A systems.

## Full Text


<!-- PDF content starts -->

LiveRAG: ADIVERSEQ&ADATASET WITH VARYING DIFFICULTY
LEVEL FORRAGEVALUATION
A PREPRINT
David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Alex Shtoff, Oren Somekh, Ran Tavory
Technology Innovation Institute (TII), Haifa, Israel
November 19, 2025
ABSTRACT
With Retrieval-Augmented Generation (RAG) becoming more and more prominent in generative AI
solutions, there is an emerging need for systematically evaluating their effectiveness. We introduce
theLiveRAGbenchmark, a publicly available dataset of 895 synthetic questions and answers designed
to support systematic evaluation of RAG-based Q&A systems. This synthetic benchmark is derived
from the one used during the SIGIR’2025LiveRAGChallenge, where competitors were evaluated
under strict time constraints. It is augmented with information that was not made available to
competitors during the Challenge, such as the ground-truth answers, together with their associated
supporting claims which were used for evaluating competitors’ answers. In addition, each question
is associated with estimated difficulty and discriminability scores, derived from applying an Item
Response Theory model to competitors’ responses. Our analysis highlights the benchmark’s questions
diversity, the wide range of their difficulty levels, and their usefulness in differentiating between
system capabilities. TheLiveRAGbenchmark will hopefully help the community advance RAG
research, conduct systematic evaluation, and develop more robust Q&A systems.
1 Introduction
Retrieval-Augmented Generation(RAG) is a widely adopted methodology for improving the effectiveness ofLarge
Language Models(LLMs), particularly for question answering tasks [ 1,2,3]. RAG is attracting significant attention
from the AI and Information Retrieval (IR) communities. Yet, reliable and systematic evaluation of RAG systems
remains an open challenge [4, 5, 6].
In this paper, we introduce a publicly available benchmark for evaluating RAG-based question-answering systems. The
“LiveRAGbenchmark” we release in this work1is derived from the one used during the SIGIR-2025LiveRAGChallenge
[7], hence its name.
The SIGIRLiveRAGChallenge took place between March and May 2025, with results announced during the SIGIR’2025
conference. Its goal was to facilitate progress in RAG research by enabling teams from academia and industry to
evaluate their solutions on a common benchmark and compare performance against those of other teams, using a fixed
external corpus (Fineweb-10BT2), and a fixed open-source LLM (Falcon3-10B-Instruct3). During the live day event,
competing teams were divided into two sessions, each receiving a set of 500 unseen questions, including 105 questions
shared between sessions, for manual validation of LLM-based judgment and cross-session calibration. All questions
(and associated reference answers) were generated using the DataMorgana tool [8] (see §2 for more details).
To generate theLiveRAGbenchmark we introduce here, we merged the two sessions’ sets of 500 questions (with their
shared 105 questions) to obtain a total of 895 unique questions. We then augmented these questions with supplementary
information that was not made available to competitors, thereby enabling multiple and richer evaluation scenarios.
1https://huggingface.co/datasets/LiveRAG/Benchmark
2https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-10BT
3https://huggingface.co/tiiuae/Falcon3-10B-InstructarXiv:2511.14531v1  [cs.CL]  18 Nov 2025

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
TheLiveRAGbenchmark provides, for each question, the answer generated by DataMorgana, as well as the supporting
documents that the tool used for Q&A generation. It also includes the “answer claims” used during the challenge to
compare competitors’ answers with the reference answers. Furthermore, it associates with each question an estimated
difficulty score, and adiscriminability scorederived from an Item Response Theory (IRT) model [ 9] trained on the
evaluation scores of participating systems’ responses to theLiveRAGquestions4.
The IRT-derived difficulty and discriminability parameters provided by the benchmark serve to normalize question
characteristics across the dataset, effectively placing them on a common scale. This calibration is essential since the
questions are distributed across two disjoint sessions, with responses originating from distinct participant cohorts.
Furthermore, these parameters enable practitioners to train their systems using questions of varying difficulty levels,
e.g., for curriculum learning [ 10]. Our analysis demonstrates that these parameters effectively reflect question difficulty
and discriminability, as questions with lower difficulty values were consistently more challenging for all RAG-based
systems that participated in the challenge, as well as for a wide range of LLMs of varying sizes.
The remainder of this paper is organized as follows. Section §2 describes the process used to construct the benchmark.
In Section 3 we describe the IRT model used for benchmark analysis. Section §4 presents an analysis of the questions’
difficulty, followed by Section §5, which explores factors contributing to question difficulty. Finally, Section §6
discusses limitations of the benchmark, and Section §7 concludes.
2 Benchmark Generation with DataMorgana
TheLiveRAGBenchmark was generated using DataMorgana [ 8], a synthetic data generation tool that offers high
configurability and is capable of producing highly diverse sets of Q&As. The following section highlights some of
DataMorgana’s characteristics that were specifically leveraged to produce theLiveRAGbenchmark.
2.1 Document sampling
DataMorgana generates each QA pair based on information extracted from specific source documents. To construct
theLiveRAGbenchmark, documents were sampled from the official corpus of the Challenge, FineWeb-10BT. Given
that the corpus comprises arbitrary web pages, a topic-based document sampling pipeline was employed to ensure the
selected documents are appropriate for generating valuable question-answer pairs. The sampling pipeline comprises
three stages:
Topic Generation.The LLM is prompted to generate diverse list of high-level topics, and for each topic, a list of
related subtopics (See Appendix §A.1 for the topic generation prompt).
Topic-Based Document Retrieval.The subtopics are used to query FineWeb-10T for retrieving relevant documents.
Document Filtering.Duplicate, too short, or too long documents are removed from the pool of retrieved documents.
We then use an LLM to score each document according to the following criteria:
•Factuality— Does the document contain factual information that is appropriate for generating open-domain questions?
•Interest— Is the content potentially interesting and useful?
•Credibility— Is the document trustworthy and free from promotional narrative, or overly subjective material?
•Toxicity— Does the document contain harmful, offensive, or inappropriate language?
•Sexuality— Does the document contain sexual content?
•Freshness— Is the content fresh and relevant, or is it outdated?
The documents are filtered according to their scores using predefined thresholds for each criterion, constructing a pool
of valid documents to be used for question generation. The filtering prompt is given in Appendix §A.2.
2.2 Generation pipeline
DataMorgana builds the benchmark incrementally, generating one Q&A pair at a time, using the following three-step
procedure.
4We thank the organizers of the SIGIR’25 LiveRAG Challenge for giving us access to the participant answer scores for each
question, which enabled us to compute the IRT model parameters.
2

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
2.2.1 Category set selection
The benchmark is defined by specifying a set of desired question categorizations, each comprising one or more mutually
exclusive categories. ForLiveRAG, eight such categorizations were used, as listed in Table 1. These categories are
intentionally broad to support the generation of diverse questions from any document within the corpus. In each question
generation task, one category is randomly selected from each categorization, resulting in a total of eight categories per
task. The 8 selected categories are used for the question generation step (see §2.2.3).
2.2.2 Document selection
DataMorgana randomly samples a document from the pool of valid documents, to be used for Q&A generation. A
second, complementary document is selected for multi-doc generation, if eithercomparisonormulti-aspecthas been
selected from theAnswer Typecategorization. The second document is selected by prompting the LLM to generate 3
questions that can be partially answered by the first selected document, d, and for each question, generating a search
query for retrieving the missing information not presented in d(See Appendix §A.3 for the relevant prompt). Then, five
documents are retrieved from the corpus for each generated query, and the LLM is prompted to select one document
from the pool of search results that best complements d. The prompt for selecting the complementary document is
given in Appendix §A.4.
2.2.3 Question generation
Finally, a prompt is constructed to instruct the LLM to generate a Q&A from the selected document (and from the
complementary document in the case ofcomparisonormulti-aspectquestions), ensuring that the generated Q&A
adheres to the eight selected categories (See the question generation prompt in [ 8, Appendix A]). For all generation
tasks, we used Claude 3.5-Sonnet5as the backbone LLM.
The LiveRAG benchmark comprises 895 question-answer pairs generated through the process outlined above. A
comprehensive description of the dataset, including a few Q&A examples, is provided in Appendix §B.
3 IRT analysis
3.1 Background
We analyze the benchmark characteristics usingItem Response Theory(IRT), a framework from psychometrics
[11,12,13], which jointly estimates latent traits of questions and of participating systems (subjects) in theLiveRAG
challenge.
IRT is frequently used in educational testing [ 9], as well as in machine learning [ 14,15]. Recent work investigates its
use for dataset analysis [ 13,12]. We follow this line of work to analyze theLiveRAGbenchmark, and expose IRT model
parameters as part of the dataset, thus enabling practitioners to train their systems using questions of varying difficulty
levels.
Given an observation matrix Ym×nofmsubjects and nquestions, where Y[j, i] represents the correctness score of the
answer provided by subject sjto question qi6; an IRT model estimates the probability of sjanswering qicorrectly by
learning the latent parameters ofs jandq ithat best fit the input observation data.
A series of statistical models with increasing complexity are used to represent both item and subject characteristics. The
IRT one-parameter logistic model (1PL), also known as the Rasch model, estimates a latent “skill” parameter θjfor
each subject, and a latent “difficulty” parameterb ifor each question. It is defined by:
p(yj,i= 1|θ j, bi) =1
1 +e−(θ j−bi)(1)
The larger the margin betweenθ jandb iis, the higher the probability ofs jansweringq icorrectly.
More complex IRT models estimate additional latent parameters for items. The two-parameter logistic (2PL) model
introduces a “discrimination” parameter aifor each question qi, which reflects how effectively the question discriminates
between individuals with similar skills:
p(yj,i= 1|θ j, bi, ai) =1
1 +e−a i(θj−bi)(2)
5https://www.anthropic.com/news/claude-3-5-sonnet
6The observation matrix is not necessarily complete, i.e., a subject may answer only a subset of the questions.
3

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
Categorization Category Description
Answer Typefactoidquestion seeking a specific, concise piece of information or a short fact about a particular subject, such as a name,
date, or number.
yes/no a question that can be answered with true/false or yes/no.
definition a question that requires finding the definition of the term in the question.
list a question that requires as an answer a list of entities or facts.
explanation a question that requires as an answer an explanation.
comparisona comparison question that requires comparing two related concepts or entities. The comparison must be natural
and reasonable, i.e., comparing two entities by a common attribute which is meaningful and relevant to both entities.
For example: ’Who is older, Glenn Hughes or Ross Lynch?’, ’Are Pizhou and Jiujiang in the same province?’,
’Pyotr Ilyich Tchaikovsky and Giuseppe Verdi have this profession in common’. The information required to
answer the question needs to come from two documents, specifically, the first document must provide information
about the first entity/concept, while the second must provide information about the second entity/concept.
multi-aspecta question about two different aspects of the same entity/concept. For example: ’What are the advantages
of AI-powered diagnostics, and what are the associated risks of bias in medical decision-making?’, ’How do
cryptocurrencies enable financial inclusion, and what are the security risks associated with them?’. The information
required to answer the question needs to come from two documents, specifically, the first document must provide
information about the first aspect, while the second must provide information about the second aspect.
Answer Styleconcise-answera question that explicitly asks for a short and direct answer, requesting only the essential information without
additional explanation. The question must include this instruction explicitly.
detailed-answera question that explicitly asks for a comprehensive answer, requesting additional details, background information,
or clarifications. The question must include this instruction explicitly.
unspecified a question that does not explicitly ask for a specific style of answer.
Premisedirect question that does not contain any premise or any information about the user
with-premise question starting with a very short premise, where the user reveals their needs or some information about himself.
Phrasingconcise-and-natural a concise direct natural question consisting of a few words.
verbose-and-natural a relatively long question consisting of more than 9 words.
short-search-queryphrased as a typed web query for search engines (only keywords, without punctuation and without a natural-
sounding structure). It consists of less than 7 words.
long-search-queryphrased as a typed web query for search engines (only keywords, without punctuation and without a natural-
sounding structure). It consists of more than 6 words.
Linguistic Variationsimilar-to-document a question phrased using the same terminology and phrases appearing in the document.
distant-from-documenta question phrased using expressions and terminology that differ from the ones appearing in the documents. For
instance, the question uses complex paraphrasing that deviates from the original document’s exact wording.
Politenesspolite a question phrased in a very polite way.
neutral a question that is not rude but that at the same time does not include social niceties like ’please’ or ’would you
mind’.
Linguistic Correctnesscorrect a question written in correct English.
mild-mistakes a question containing mild spelling mistakes.
severe-mistakes a question containing severe spelling mistakes.
User Personaexpert an expert on the subject discussed in the documents, therefore, he asks complex questions.
novice a person with very basic knowledge on the topic discussed in the topic. Therefore, he asks very simple questions.
researcher a researcher operating on the topic discussed in the documents.
journalist a journalist interested in writing an article about the topic discussed in the documents.
Table 1: DataMorgana configuration for Question Categorizations and for User Personas, used for generating the
LiverRAG benchmark.
Other IRT models that include a “guessing” parameter for each question, or multi-dimensional parameters [ 16], are
outside the scope of this work.
3.2 IRT model implementation
To implement the IRT models, we use the py-irt package7[16], a Python package based on probabilistic inference for
fitting the latent subject and item parameters that best explain the observed data. For observations, we leverage the
Correctness metric [ 7] used for evaluating the system’s answer for a given question in theLiveRAGChallenge. The
Correctness score is defined as the harmonic mean ofCoverageandRelatedness, with Coverage being the proportion of
critical content in the reference answer that is correctly reflected in the generated answer, and Relatedness being the
proportion of vital claims in the generated answer that are relevant to the given question8.
In this work we focus on the 2PL model. Training was conducted with a learning rate of 0.01, dropout=0.2 and over
10,000 epochs. The parameters learned by the model, (bi, ai), per question qi, are provided as part of the benchmark.
Figure 1 presents the question difficulty (diff) and discriminability (disc) distributions, alongside a scatter plot of (bi, ai)
of all questions. Interestingly, the Pearson correlation betweendiffand the average correctness scores (ACS) (the
7https://github.com/nd-ball/py-irt
8The py-irt package expects binary observations (true or false), whereas in our case, observations are continuous in the range
of[−1. . . ,2] , modeling the extent to which the answer is correct. We therefore modified the package to support continuous
observations by using the Continuous-Bernoulli distribution for the observation likelihood, rather than the Bernoulli distribution
originally used by the package. Since this distribution expects observations in the range [0. . .1] , we linearly transformed the
Correctness scores to this range.
4

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
Figure 1: Question parameters learned by the IRT-2PL model for theLiveRAGdataset. Top: Difficulty distribution. Left:
Discriminability distribution. Middle: Scatter plot of difficulty and discriminability scores of all benchmark questions.
average Correctness score of all participating systems that answer the question) is -0.979. Overall, there is a weak
negative correlation between discrimination and difficulty (Pearson = -0.423).
Furthermore, when comparing system rankings, derived from the learned skills ( θj), with their leaderboard positions
reported in [ 7], we observe a strong concordance – reflected by Kendall’s tau coefficients of 0.766 for the first session
and 0.999 for the second. This high correlation persists despite the fact that skill-based rankings are computed over the
full benchmark set, whereas leaderboard scores are session-specific, each based on a subset of 500 questions.
4 Validating Question Difficulty
To validate the effectiveness of the difficulty scores learned by the IRT model in estimating question difficulty, we analyze
the distribution ofdiffscores across theLiveRAGquestions. For clarity, we divide the questions into quartiles according
to theirdiffscore: (i) HD (highly difficult) in the range [−6,−2.143) ; (ii) D (difficult) in the range [−2.143,−0.962) ;
(iii) M (moderate) in the range [−0.962,0.236) ; and (iv) E (easy) [0.236,6] . Figure 2 illustrates the performance
distribution of all systems participating in theLiveRAGchallenge over thediffbins. We order the systems, from left to
right, according to their official leaderboard score.
Examining the graphs, we see that for both sessions and for all systems, performance consistently improves as questions
become easier. This trend confirms that thediffscores accurately reflect question difficulty. It is also observed that in
both sessions, Falcon3 without RAG underperforms as compared to all participating systems that used a RAG-based
solution. This highlights the long-tail nature of the benchmark questions, which often require retrieval assistance to be
answered effectively.
These findings are further substantiated by Table 2: ACS exhibits a monotonic increase from harder to easier bins,
aligning with expectations. The averagediscscore declines across bins, indicating that hard questions provide weaker
discriminatory power to the benchmark.
HD D M E
#questions 224 223 224 224
ACS-1.304 -0.653 -1.883 2.325
disc0.230 0.051 -0.041 -0.106
Table 2:ACSanddiscscores acrossdiffbins.
9Correlation betweendiffandACSis negative since high Correctness score indicates low difficulty.
5

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
Figure 2: Team performance distributions acrossdiffbins. Teams are ordered from left to right by their leaderboard
position. The rightmost distribution represents Falcon3-10B without RAG, given for reference.
Figure 3: Performance distributions of GPT-4.1, and several LLaMA models of different sizes (without RAG), across
thediffbins.
4.1 Are difficult questions difficult for all?
We evaluate GPT-4.110answers over the Benchmark questions, alongside several LLaMA models of varying sizes11, to
examine whether difficult questions are consistently challenging across models. We applied the same LLM-as-a-judge
[17] process used in the LiveRAG challenge to measure ACS of the LLM responses to the challenge questions. Figure 3
presents the average performance of the LLMs (without RAG augmentation) across the predefineddiffbins. The results
reveal that question difficulty strongly correlates with model performance, i.e., harder questions yield lower average ACS
score irrespective of model architecture or size. Furthermore, the performance order between difficulty levels remains
consistent across the evaluated LLMs. As expected, larger models outperform smaller ones. Interestingly, GPT-4.1
surpasses some participating systems, yet it is outperformed by the top-performingLiveRAGteams that implemented
RAG on top of Falcon3-10B. This supports our observation that in the absence of RAG, even state-of-the-art LLMs
struggle to answer the benchmark’s questions effectively, underscoring RAG necessity for long-tail questions.
10GPT-4.1 version: gpt-4.1-2025-04-14
11https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct ,https://huggingface.co/meta-llama/Llama-3.
1-8B-Instruct ,https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct ,https://huggingface.co/meta-llama/
Llama-3.2-1B-Instruct
6

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
5 Analyzing Question Difficulty
What factors make a question difficult for LLMs, and more specifically, for RAG-based LLMs? Several factors can
increase difficulty, including surface-level issues such as severe typographical errors or lack of clarity, as well as deeper
challenges requiring complex reasoning. Moreover, question-independent factors such as inaccuracies or omissions in
the reference answer, an absence of relevant content in the RAG corpus, cascading errors from retrieval, or limited
coverage of certain domains by the LLM, can also impact perceived difficulty [ 18,19]. In this section, we focus
exclusively on question-intrinsic difficulty, leaving corpus- and system-level effects to future work. To this end, we
analyze the IRT-deriveddiffscores across various question types to better understand the structural and semantic
properties that drive question difficulty in RAG-based systems.
5.1 Single- vs. Multi-Document questions
Single-doc questions are generated by DataMorgana using a single source document, and it is guaranteed that each
question can be answered by that document12. In contrast, multi-doc questions (i.e., comparison ormulti-aspect
questions) are generated from two complementary documents (see Section 2.2). As such, many of these questions
cannot be accurately answered using a single document alone.
Therefore, we hypothesize that multi-doc questions are more difficult than single-doc questions. Table 3 supports this
hypothesis by reporting the averagediffanddiscscores for the sets of single and multi-doc questions. As expected, multi-
doc questions exhibit a significantly higherdiffscore than single-doc questions, while showing lower discriminative
power as reflected by lower averagediscscore.
#questionsdiff disc
Single-doc 758 -1.083 0.062
Multi-doc 137 0.091 -0.212
Table 3: Averagediffanddiscscores across single- and multi-doc questions.
5.2 Difficulty across Question Categories
The DataMorgana configuration categories used for question generation can also be related to question difficulty. For
instance, questions containing severe linguistic typos are likely to pose greater challenges for a question answering
system.
We therefore measure thediffdistribution across the different question categorizations used by DataMorgana for the
challenge (see Table 1). Figure 4 presents these distributions across the eight categorizations used. The number of
questions per category (given in parentheses), is determined by the probability distribution specified within DataMorgana
configuration.
Although the differences in averagediffacross categories are statistically insignificant for most categorizations, the
distributional shifts between categories are still observable. Looking at the Answer Type categorization,comparisonand
multi-aspectquestions emerge as the most challenging. This outcome is expected, as both types require synthesizing
information from multiple documents, unlike other answer types, which rely on a single document. Interestingly,Yes/No
questions also appear to be relatively more difficult. We hypothesize that this is due to their binary nature, which often
requires implicit reasoning, especially when the correct answer (Yes/No) is not explicitly stated in the source text.
Similarly, in theAnswer Stylecategorization,concise-answerquestions are relatively more difficult for instruction-tuned
LLMs, which are typically trained to produce elaborated responses. A similar pattern is observed in thePhrasing
categorization, where natural questions are easier than search (keyword-based) questions, which are inherently more
ambiguous due to their compressed format.
ForLinguistic Variation, the analysis reveals that questions which are semantically similar to their documents are
easier than those that are dissimilar. This is reminiscent of reported work on query difficulty estimation for ad hoc
retrieval [ 20], where the “distance” between the query and the retrieved documents strongly influences its difficulty.
PremiseandPolitenessdo not seem to affect difficulty, while forLinguistic Correctness, as expected, the severity of
typos embedded within the questions increases their difficulty. Finally, forUser Persona,expertquestions are slightly
easier thannovicequestions, likely because they contain specific terminology that facilitates the retrieval process.
12We note that while alternative, better answers based on other documents in the corpus may exist, the selected document is
guaranteed to answer the question.
7

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
Figure 4: Box-plot presentation of thediffdistributions across question categorizations. Median values are shown in
bold. Number of questions per category is indicated in parentheses.
5.3 Linguistic Diversity
Diversity is a key characteristic of any benchmark designed to evaluate system performance, as it broadens the spectrum
of challenges and scenarios a system may encounter in real-world deployments. A diverse benchmark helps models to
generalize across different question types and is more likely to include edge cases and uncommon styles, increasing the
challenge for models trained primarily on homogeneous data [26].
To evaluate the diversity of the benchmark, we adopt several metrics developed for text generation tasks [ 27], which
capture general linguistic aspects, such as lexical, syntactic, and semantic diversity. Table 4 presents the linguistic
diversity of theLiveRAGquestions and of equal-size samples (i.e., 895 questions) from some popular QA benchmarks.
TheLiveRAGbenchmark achieves the highest lexical diversity, as measured by NGD — the fraction of distinct n-grams
(up to 4) over the total number of n-grams. Additionally, theLiveRAGquestions also reach the highest length entropy —
the entropy of question length distribution in the benchmark. To compute syntactic diversity, the benchmark questions
are first converted into theirPart-of-Speech(PoS) tag sequences, and then the compression ratio, PoS-CR, is defined as
8

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
BenchmarkNGD PoS-CR embbed. length question
(↑) (↓) HS (↓) entropy (↑) length
TriviaQA [21] 3.0455.1670.055 3.085 14.301
PopQA [22] 1.952 10.711 0.214 1.691 6.669
WebQuestions [23] 2.762 6.175 0.157 1.848 6.736
SimpleQA [24] 2.755 6.184 0.110 3.126 16.399
Natural Questions [25] 2.841 5.3730.0171.518 9.029
LiveRAG3.0625.220 0.0613.20715.297
Table 4: Linguistic diversity metrics for theLiveRAGbenchmark, and several popular QA benchmarks; ↑/↓mark
higher/lower-is-better.
the ratio between the size of the file containing the PoS sequences to the size of its compressed gzip version. According
to this metric, only TriviaQA is more diverse than theLiveRAGbenchmark13.
To assess semantic diversity, we compute the Homogenization Score (embeddings-HS), which measures the average
pairwise cosine similarity between question embeddings14. TheLiveRAGbenchmark achieves competitive diversity
results, despite all questions being generated from a limited set of topics (see Section 2.1), a factor that would typically
reduce semantic diversity.
These results suggest thatLiveRAGis generally more diverse than widely used benchmarks, offering a robust framework
for evaluating RAG systems. Moreover, the diversity analysis complements and reinforces the difficulty analysis
presented in Section 4, as diversity is inherently linked to difficulty since higher language variation often requires
diverse reasoning strategies to answer the question.
6 Limitations
The synthetic Q&As in theLiveRAGbenchmark are not direct reflections of actual user needs, but rather projections of
anticipated future needs, as approximated through the DataMorgana categorizations. Consequently, conclusions about
system performance in real-world scenarios derived from this benchmark should be interpreted with caution. Moreover,
although the benchmark Q&As appear natural and well formulated, the corresponding answers are automatically
generated based on one or two source documents, while other documents in the corpus may offer more accurate or even
contradictory answers. Such discrepancies can lead to high-quality responses being mistakenly evaluated as incorrect
due to divergence from the designated “ground truth”.
Furthermore, the IRT-baseddiffanddiscscores are calculated based on the responses of systems that participated in the
LiveRAGChallenge. Since all these systems utilized a RAG-based architecture based on Falcon3 for answer generation,
these scores may reflect biases where certain factors contributing to question difficulty or discrimination are specific to
such models. Despite these limitations, our empirical analysis supports the reliability of these scores as indicators of
question difficulty.
7 Concluding Remarks
In this paper, we introduced theLiveRAGbenchmark – a publicly available dataset based on the dataset used in
the SIGIR’2025LiveRAGchallenge. It enriches the Q&A pairs by including the average and standard deviation of
Correctness scores achieved by participating teams for each question, as well as difficulty and discriminability scores
derived from IRT analysis, which can serve as proxies for question difficulty and discriminative power, respectively.
Our preliminary analysis explored the distribution of question difficulty across various dimensions and demonstrated the
reliability of these metrics. We observed that highly difficult questions posed significant challenges to all participating
systems, as well as to a range of LLMs of different sizes. While such difficult questions may be less effective at
differentiating between systems, they expose important limitations of current RAG approaches and highlight key
directions for future research.
13TriviaQA contains many questions with unusual syntactic patterns (e.g.,“A Russian rouble is divided into 100 . . . .what?”) that
highly contribute to its syntactic diversity.
14Question embeddings are obtained using theMiniLMsentence encoder https://huggingface.co/sentence-transformers/
all-MiniLM-L6-v2
9

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks.Advances in Neural Information Processing Systems, 33:9459–9474, 2020.
[2]Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language
models.Journal of Machine Learning Research, 24(251):1–43, 2023.
[3]Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian
Xiong, and Deyi Xiong. Evaluating large language models: A comprehensive survey, 2023.
[4]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation. InProceedings of the 18th Conference of the European Chapter of the Association for
Computational Linguistics: System Demonstrations, pages 150–158, 2024.
[5]Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Gui, Ziran
Jiang, Ziyu Jiang, et al. CRAG-comprehensive RAG benchmark.Advances in Neural Information Processing
Systems, 37:10470–10490, 2024.
[6]Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. Support
evaluation for the trec 2024 RAG track: Comparing human versus llm judges.arXiv preprint arXiv:2504.15205,
2025.
[7]David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory, Mehdi Ghissassi, Edo
Liberty, and Roy Miara. Sigir 2025 – liverag challenge report, 2025.
[8]Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, and Yoelle Maarek. Generating
Q&A benchmarks for RAG evaluation in enterprise settings. InProceedings of the 63st Annual Meeting of the
Association for Computational Linguistics (Industry Track), 2025.
[9] Frederic M Lord.Applications of item response theory to practical testing problems. Routledge, 2012.
[10] Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and Nicu Sebe. Curriculum learning: A survey.International
Journal of Computer Vision, 130(6):1526–1565, 2022.
[11] John P. Lalor, Hao Wu, Tsendsuren Munkhdalai, and Hong Yu. Understanding deep learning performance through
an examination of test set difficulty: A psychometric case study. In Ellen Riloff, David Chiang, Julia Hockenmaier,
and Jun’ichi Tsujii, editors,Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pages 4711–4716, Brussels, Belgium, October-November 2018. Association for Computational
Linguistics.
[12] Pedro Rodriguez, Joe Barrow, Alexander Miserlis Hoyle, John P. Lalor, Robin Jia, and Jordan Boyd-Graber.
Evaluation examples are not equally informative: How should that change NLP leaderboards? In Chengqing Zong,
Fei Xia, Wenjie Li, and Roberto Navigli, editors,Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume
1: Long Papers), pages 4486–4503, Online, August 2021. Association for Computational Linguistics.
[13] Clara Vania, Phu Mon Htut, William Huang, Dhara Mungra, Richard Yuanzhe Pang, Jason Phang, Haokun Liu,
Kyunghyun Cho, and Samuel R. Bowman. Comparing test sets with item response theory. In Chengqing Zong,
Fei Xia, Wenjie Li, and Roberto Navigli, editors,Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume
1: Long Papers), pages 1141–1158, Online, August 2021. Association for Computational Linguistics.
[14] Michael R Smith, Tony Martinez, and Christophe Giraud-Carrier. An instance level analysis of data complexity.
Machine learning, 95:225–256, 2014.
[15] Ana C Lorena, Pedro YA Paiva, and Ricardo BC Prudêncio. Trusting my predictions: on the value of instance-level
analysis.ACM Computing Surveys, 56(7):1–28, 2024.
[16] John Patrick Lalor and Pedro Rodriguez. py-irt: A scalable item response theory library for python.INFORMS
Journal on Computing, 35(1):5–13, January 2023.
[17] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie
Ma, Honghao Liu, et al. A survey on llm-as-a-judge.arXiv preprint arXiv:2411.15594, 2024.
[18] Saku Sugawara, Nikita Nangia, Alex Warstadt, and Samuel Bowman. What makes reading comprehension
questions difficult? InProceedings of the 60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 6951–6971, 2022.
10

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
[19] Linqing Liu, Patrick Lewis, Sebastian Riedel, and Pontus Stenetorp. Challenges in generalization in open domain
question answering. InFindings of the Association for Computational Linguistics: NAACL 2022, pages 2014–2029,
2022.
[20] David Carmel, Elad Yom-Tov, Adam Darlow, and Dan Pelleg. What makes a query difficult? InProceedings of
the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval,
SIGIR ’06, page 390–397, New York, NY , USA, 2006. Association for Computing Machinery.
[21] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors,Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611,
Vancouver, Canada, July 2017. Association for Computational Linguistics.
[22] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to
trust language models: Investigating effectiveness of parametric and non-parametric memories. InProceeding of
ACL, pages 9802–9822, 2023.
[23] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-
answer pairs. InProceedings of the 2013 Conference on Empirical Methods in Natural Language Processing,
pages 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.
[24] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman,
and William Fedus. Measuring short-form factuality in large language models, 2024.
[25] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question
answering research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019.
[26] Rui Han, Xiaoyi Lu, and Jiangtao Xu. On big data benchmarking. InWorkshop on Big Data Benchmarks,
Performance Optimization, and Emerging Hardware, pages 3–18. Springer, 2014.
[27] Chantal Shaib, Joe Barrow, Jiuding Sun, Alexa F. Siu, Byron C. Wallace, and Ani Nenkova. Standardizing the
measurement of text diversity: A tool and a comparative analysis of scores, 2025.
A Prompts
A.1 Topic generation prompt
Topic Generation
Generate a list of <n> high-level topics that can be suitable for a knowledge-based question-answer
system. The topics should be diverse and concise. Return the topics as a python list. Do not output any
preamble or explanation.
Sub-Topic Generation
Generate <n> sub-topic of this topic: <topic>. Each sub-topic must be suitable for a knowledge-based
question-answer system. The sub-topics should be diverse and concise. Return the topics as a python list.
Do not output any preamble or explanation.
11

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
A.2 Document filtering prompt
Document filtering
Your task is to give a score on a scale of 1-5 for the following document based on the following aspects:
1) factuality: The document contains factual information about an entity/concept that is expected to be
found on Wikipedia (or similar Encyclopedias). The document is appropriate for creating an open domain
question from.
2) interest: The document contains knowledge which might be interesting or useful to someone.
3) credibility: The document is credible and can be trusted as reliable, accurate, and authoritative.
Examples of documents that are NOT credible include promotional materials, heavily biased political
materials, narratives, diaries, resumes, job posting, subjective reviews, essays without supporting
evidence, or unanswered questions.
4) toxicity: The document contains harmful, offensive, or inappropriate language, such as hate speech,
harassment, or extreme negativity.
5) sexuality: The document contains sexual content.
6) outdated: The document contains outdated content.
Return your answer without any preamble in the following JSON format: {"factuality": <score>, "interest":
<score>, "credibility": <score>, "toxicity": <score>, "sexuality": <score>, "outdated": <score>}
Document: <document>
A.3 Query generation prompt
The following prompt is used to generate search queries that retrieve complementary documents to the seed document
during the multi-document question generation process.
Query generation
Generate <n> questions satisfying the following characteristics:
<question category description>
It must be possible to answer the questions by using information from two documents, which we will call
DOC-A and DOC-B.
The required information from DOC-A must not appear in DOC-B and vice versa. In addition, for each
question you must generate a query that can be used in a search engine to retrieve DOC-B.
Generate <n> question-query pairs. Write each pair in a new line, in the following JSON format:
{ "question": "<the generated question>", "search query": "<the search query for DOC-B>" }
Do not provide any preamble or explanation.
### DOC-A:
<d1>
A.4 Document selection prompt
The following prompt is used to select a complementary document from search results during the multi-document
question generation process.
12

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
Document selection
I need to generate a question having the following characteristics:
<question category description>
It must be possible to answer the question by using information from two documents, which we will call
DOC-A and DOC-B. The required information from DOC-A must not appear in DOC-B and viceversa.
You will receive DOC-A and a numbered list of candidate documents from which you must select DOC-B.
First provide a short reasoning for your decision, and then your pick.
You must respond in the following JSON format: ’{"reasoning": <short reasoning>, "doc number": <the
number of the selected document>}’
If no document is suitable for generating a question with the expected characteristics, write "null" in
"doc number".
### Document A:
<d1>
### Candidate documents:
### Document 1:
<candidate document 1>
### Document 2:
<candidate document 2>
...
### Document m:
<candidate document m>
B Benchmark Description
In this appendix, we describe theLiveRAGBenchmark, hosted on the open Hugging Face platform15, which includes
895 Q&As. Table 5 presents a few illustrative examples from the benchmark. Each entry in the benchmark includes the
following fields:
•Question:the question generated by DataMorgana.
•Answer:the corresponding answer generated by DataMorgana. Since the answer is generated using a strong LLM,
based on selected documents from the external corpus, it is treated as the “ground truth” answer for the given question.
However, this may lead to inconsistencies when other documents in the corpus provide alternative valid answers.
In such a case, a correctly generated answer might be incorrectly judged as wrong. To minimize these issues, we
manually filtered out problematic items from the benchmark. Nevertheless, it is important to note that the provided
answer is not necessarily the only valid answer to the question.
•Supporting documents:the list of Fineweb-10BT documents used for Q&A generation. The list contains either a
single document (for single-document questions) or two documents (for multi-document questions). Each document
includes its FineWeb-10BT document ID and its full content.
•Answer claims:TheLiveRAGofficial evaluation [ 7] is based on comparing the generated answer’s claims to the vital
claims present in the reference answer. For that, we extract all claims from the answer and classify them into three
categories: 1)Direct– the claim directly corresponds to answering the question; 2)Useful– the claim is useful for
answering the question; and 3)Useless– the claim is unrelated or unhelpful for answering the question. The list of
answer claims with their corresponding classifications is provided to support the evaluation process.
•Session:Indicates the Live Challenge Day session in which the question appeared (“First” - first session, “Second”
- second session, and "Both" - both sessions, i.e., a shared question). We provide this information mostly to help
benchmark users compare their scores against those achieved by competitors in each particular session.
•DataMorgana configuration:The eight question categories used by DataMorgana for question generation.
•Average Correctness Score (ACS):The average Correctness score across allLiveRAGsystems which answered the
question.
•Standard deviation ofACS(ACS_std):The standard deviation of the Correctness scores of allLiveRAGsystems
which answered the question.
•IRT parameters:the (diff,disc) parameters learned by the IRT-2PL model (see Section 3).
15https://huggingface.co/datasets/LiveRAG/Benchmark
13

LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluationA PREPRINT
field-name field-value
Question: How deep can fish survive in the ocean trenches?
Answer: Fish can survive up to 8,100 meters deep. Beyond this depth, no fish live,
possibly due to their inability to cope with the highest pressures.
Supporting Docs: <urn:uuid:a102a6cb-a608-493c-928f-d32a0da4dbf6>
DM Categories: unspecified, factoid, concise and natural, correct, similar to documents,
polite, without premise, novice
ACS: 0.876
diff: -1.33
disc: -0.29
Question: Based on temperature considerations alone, is March considered a
suitable month to perform the final pruning of grape vines?
Answer: Yes.The first week of March is a good time for the final pruning of grapes, as this is
when you can prune the remainder of last year’s growth down to about one or 2
inches long for spur production, or 12 to 15 inches long for cane production.
Supporting-docs: <urn:uuid:b5d19fcb-1711-4f9f-82cf-f81403382444>
DM Categories: concise-answer, yes/no, verbose and natural, correct, distant from documents,
neutral, without premise, novice
ACS: 0.089
diff 1.234
disc: -0.162
Question: Would you please explain how both Amália Rodrigues and Gene Chandler demonstrated
good artistic intuition in choosing their musical direction?
Answer: Both artists showed strong artistic intuition in their career choices. Amália Rodrigues
had a strong intuition and good taste in selecting her repertoire, consistently
choosing compositions from great Portuguese composers and later expanding into
classical poetry with Oulman. Gene Chandler demonstrated his intuition when he promoted
the flip side ’Rainbow’ against his label’s wishes to push ’You Threw a Lucky Punch,’
a decision that proved smart as ’Rainbow’ became more successful and helped establish him
as a serious artist beyond his ’Duke of Earl’ hit.
Supporting Docs: <urn:uuid:8269372f-be5c-4212-8830-c7e0ca1b295e>,
<urn:uuid:af9d46db-12a9-41d8-9687-1b4232404418>
DM Categories: unspecified, comparison, verbose and natural, correct, similar to documents
polite, without premise, expert
ACS: 0.277
diff 1.317
disc: -0.184
Table 5: A few examples from the benchmark: Top: an easy question. Middle: a difficult question. Bottom: a highly
difficult (multi-doc ) question. In red - answer direct claims. In blue - answer useful claims.
14
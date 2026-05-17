# Grounded Satirical Generation with RAG

**Authors**: Oona Itkonen, Yuxin Su, Linyao Du, Ona De Gibert

**Published**: 2026-05-11 17:00:51

**PDF URL**: [https://arxiv.org/pdf/2605.10853v1](https://arxiv.org/pdf/2605.10853v1)

## Abstract
Humor generation remains challenging task for Large Language Models (LLMs), due to their subjective nature. We focus on satire, a form of humor strongly shaped by context. In this work, we present a novel pipeline for grounded satire generation that uses Retrieval-Augmented Generation (RAG) over current news to produce satirical dictionary definitions in the Finnish context. We also introduce a new task-specific evaluation framework and annotate 100 generated definitions with six human annotators, enabling analysis across multiple experimental conditions, including cultural background, source-word type, and the presence or absence of RAG. Our results show that the generated definitions are perceived as more political than humorous. Both topic-based word selection and RAG improve the political relevance of the outputs, but neither yields clear gains in humor generation. In addition, our LLM-as-a-judge evaluation of five state-of-the-art models indicates that LLMs correlate well with human judgments on political relevance, but perform poorly on humor. We release our code and annotated dataset to support further research on grounded satire generation and evaluation.

## Full Text


<!-- PDF content starts -->

Grounded Satirical Generation with RAG
Oona Itkonen*and Yuxin Su*and Linyao Du*and Ona De Gibert
University of Helsinki
Abstract
Humor generation remains challenging task for
Large Language Models (LLMs), due to their
subjective nature. We focus on satire, a form of
humor strongly shaped by context. In this work,
we present a novel pipeline for grounded satire
generation that uses Retrieval-Augmented Gen-
eration (RAG) over current news to produce
satirical dictionary definitions in the Finnish
context. We also introduce a new task-specific
evaluation framework and annotate 100 gen-
erated definitions with six human annotators,
enabling analysis across multiple experimen-
tal conditions, including cultural background,
source-word type, and the presence or absence
of RAG. Our results show that the generated
definitions are perceived as more political than
humorous. Both topic-based word selection
and RAG improve the political relevance of
the outputs, but neither yields clear gains in
humor generation. In addition, our LLM-as-a-
judge evaluation of five state-of-the-art models
indicates that LLMs correlate well with human
judgments on political relevance, but perform
poorly on humor. We release our code and an-
notated dataset to support further research on
grounded satire generation and evaluation.
1 Introduction
Humor is a fundamental aspect of human nature,
yet defining it remains a persistent challenge (Bar-
don, 2005; Larkin-Galiñanes, 2017). Given this
conceptual complexity, it is unsurprising that Large
Language Models (LLMs) also struggle to reliably
interpret and generate humor. LLMs can detect
humor with relatively well, but generating it is still
an unsolved problem.
In this work, we focus on satire generation and
evaluation, an even more challenging task, as satire
constitutes a nuanced cultural-specific form of hu-
mor, the interpretation of which depends heavily
on a shared social, political, and historical context
*Equal contribution(Stinson, 2019). We adopt the definition ofSatire
from the Cambridge Dictionary1:a way of criticiz-
ing people or ideas in ahumorousway, especially
in order to make apolitical point. Accordingly, two
core components of satire are humor and political
significance.
For generation, we present a novel method based
on Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) to produce satire from news content in
the form of satirical dictionary definitions. Because
satire is highly dependent on cultural and regional
context, we restrict our study to the Finnish setting.
As source data, we use English-language news arti-
cles published on the website of the Finnish public
broadcaster Yle2. Our pipeline consists of a scraper
with sentiment analysis and timestamp-based fil-
tering, a semantic search engine, a topic modeling
component for selecting candidate words, and a
final RAG-based generation stage.
For evaluation, we develop our own framework,
motivated by the fact that humor evaluation is in-
herently difficult and lacks standardized evaluation
practices (Hämäläinen and Alnajjar, 2021). We
present evaluation results of based on two methods:
human annotation and LLM-based judgment. We
evaluate five mid-size state-of-the-art LLMs. We
aim to answering the following Research Questions
(RQ):
•RQ1: To what extent are the generated defini-
tions humorous and politically meaningful?
•RQ2: To what extent is successful satire gen-
eration dependent on cultural context?
•RQ3: How does the choice of the candidate
word affect the quality of the generated defini-
tions?
1https://dictionary.cambridge.org/dictionary/
english/satire
2https://yle.fi/arXiv:2605.10853v1  [cs.CL]  11 May 2026

3. Retrieval-Augmented Generation 2. Semantic Search Engine 1. Web Scraping & Filtering
Web ScraperBatch 
ArticlesSentiment AnalysisFiltered-out
Files
Passed News
CorpusInput
Query
Semantic Search 
Engine
Cosine Similarity 
> 0.1?Top 3 News
SnippetsSatirical 
DefinitionTimestamp
CheckFail
PassYes
Search ContextConstruct
PromptLlama-3 
via OllamaFigure 1: Overview of our generation pipeline for satirical dictionary definitions with RAG.
•RQ4: Does RAG improve the quality of gen-
erated satirical definitions?
•RQ5: Can LLMs serve as reliable evaluators
of satirical content?
In addition, we present a webpage application
to showcase and run our generation pipeline. Fur-
thermore, we release our annotated corpus to the
research community and make our scripts publicly
available for reproducibility3.
2 Related Work
One of the first tasks related to humor in NLP is
humor detection. Early work in computational
humor focused on template-based methods (Chan-
drasekaran et al., 2016) and classical supervised
machine learning approaches relying on static fea-
tures (De Oliveira and Rodrigo, 2015). More re-
cently, transformer-based and LLM-based methods
have been proposed. Recent research has exam-
ined the generalizability of humor detection mod-
els (Baranov et al., 2023), as well as related tasks
such as irony detection (Ortega-Bueno et al., 2023;
Tomás et al., 2023; Lin et al., 2024) andhumor
understanding(Hessel et al., 2023; Hwang et al.,
2025), in which an LLM is prompted to explain
why a given text is humorous.
Humor generationhas also attracted attention
for many years (Stock and Strapparava, 2005).
More recently, Jentzsch and Kersting (2023) inves-
tigated the use of ChatGPT for joke generation and
found that the model tends to produce repetitive
outputs. Similarly, Sakabe et al. (2025) used LLMs
to generateOogiri, a form of Japanese improvi-
sational comedy. However, the field continues to
observe that, while LLMs perform relatively well
on humor detection, they remain less effective at
humor generation (Sakabe et al., 2025).
3https://github.com/dlylinyao/ONLY/tree/CHUMSome work has also tried to develop improved
humor evaluationmethods. For example, Ro-
manowski et al. (2025) propose a new metric for
evaluating stand-up comedy based on statistical
measures. However, humor evaluation remains dif-
ficult because no standard evaluation practices have
been established. A recent trend is to design task-
specific annotation guidelines, obtain judgments
from both human annotators and LLMs, and then
analyze the agreement between them (Bago and
Bakari ´c, 2025; Sakabe et al., 2025; Rivera et al.,
2026). This is the approach we adopt in this work.
Finally, turning specifically tosatire, research
on satire and LLMs has expanded rapidly in re-
cent years. One line of work focuses on satire
detection. For example, Ozturk et al. (2025) intro-
duce a dataset of Turkish satirical news and inves-
tigate methods for reducing stylistic bias in satire
detection. Another direction has been proposed by
West and Horvitz (2019); Horvitz et al. (2024), who
study satire by reverse-engineering satirical news
headlines and evaluating LLMs’ ability to make
them non-satirical or “unfunny.” Dobre and Gross
(2025) compare AI-generated satire with human-
written satirical articles and evaluate the outputs
using an LLM-as-a-judge framework. To the best
of our knowledge, we are the first to study satirical
generation grounded on latest news using RAG.
3 Generation
In this section we present our methodology for
grounded generation of satirical dictionary defini-
tions. Our pipeline is based on a web scraper that
includes timestamp filtering and sentiment analy-
sis. We select relevant candidate words using topic
modeling, find the relevant articles from our data
with semantic search and, finally, we use RAG to
generate definitions of the words based on the latest
news. In Section 6, we present how this pipeline
can be run as a web application. Figure 1 presents

Q1: Is it funny?
Score Explanation
1 It is not funny, awkward at most.
2 It is slightly funny.
3 It is so funny I laughed.
4 It is so funny I will tell it to
someone else.
5 It is so funny I will laugh if
I tell it to someone else.
Table 1: Annotation guideline for Q1:Is it funny?Q2: Is it political?
Score Explanation
1 It is not political at all.
2 It has some political quality.
3 It is political on a general level.
4 It is political and current.
5 It is political, current and rele-
vant in Finnish political
culture.
Table 2: Annotation guidelines for Q2:Is it political?
an overview of our pipeline.
3.1 Web Scraping
Given the definition of satire from Cambride Dic-
tonary4, satire is often political. Therefore, our data
should ideally have political content. As news are
often about politics, or at least have content that
can be interpreted in a political frame, we choose
to use news as our source data. Politics is, how-
ever, not a very static field of domain and it evolves
quickly to new topics. Therefore, rather than opt-
ing for a static news dataset, we built a scraper with
BeautifulSoup to extract news articles published
in English from the website of the Finnish broad-
casting company Yle. Our scraper retrieves articles
from all the different categories listed in the Yle
website in English5, parsing their metadata. As
satire is, in addition to being political on a general
level, also depended on culture and region, we re-
strict our data to this one source, as it is the only
open source site that publishes news in English in
Finland.
3.2 Filtering
3.2.1 Timestamp Filtering
For the same reason, discussed in section 3.1, we
choose to scrape our data rather than use an exist-
ing dataset we want to filter our data based on the
timestamps of the articles. We chose to use 30 days
as the threshold for timestamps, and articles older
than a month do not get processed any further.
3.2.2 Sentiment Analysis
The Cambridge Dictionary6describes satire as a
way of criticizing something in a humorous way.
4https://dictionary.cambridge.org/dictionary/
english
5https://yle.fi/news
6https://dictionary.cambridge.org/dictionary/
englishHowever, there are certain restrictions to what kind
of topics are commonly considered acceptable for
satire. On the other hand, when satire is made
based on sensitive topics, the tiniest details in the
choice of words can define whether it is interpreted
as offensive and inappropriate instead of funny.
This kind of contextual understanding is some-
thing that humans are able to consider. As we are
generating satire with an LLM, however, we need
to be aware of restrictions that the model has in
understanding what is appropriate and what is not.
Therefore, to ensure our pipeline doesn’t produce
any offensive or disturbing content, we implement
sentiment analysis to filter out too negative news.
Here, we are, assuming that sentiment is a proxy
for how "bad" a piece of news is or how sensitive
or severe the topics it covers might be.
For the sentiment analysis task, we use the NLP-
Town/bert-base-multilingual-uncased-sentiment
model from Hugging Face. We feed the body text
of the news articles to the model, and it outputs
from one up to five stars, ranging from more
negative to more positive.
Our motivation for sentiment analysis is not to
filter out everything negative, but rather to ensure
the ethics of our output. Therefore, we set the
threshold of our sentiment analysis to one, so news
that get a label lower than one are discarded. To ob-
tain a more reliable sentiment score for each news
article, we split the articles into batches that fit
within the model’s token limit, perform sentiment
analysis on each batch, and then compute the mean
label scores to obtain the final article-level score.
3.3 Word Candidate Selection with Topic
Modeling
To generate satire that makes sense in a specific
cultural context, we find topics currently discussed
in the news with unsupervised topic modeling to

automatically extract candidate words from our
web-scraped data.
First, we convert the news articles into text
embeddings using the paraphrase-multilingual-
MiniLM-L12-v2 model. We then apply UMAP
(McInnes et al., 2020) for dimensionality reduction
and use BERTopic (Grootendorst, 2022) to cluster
the articles into distinct news topics. After exclud-
ing outlier documents, we extract the most salient
keywords from each valid cluster. This words will
be used to generate satirical dictionary definitions.
3.4 Retrieval
To retrieve relevant news for each input word, we
employ a semantic search approach to prioritize
contextual meaning over exact keyword matching.
We embed the news articles using the all-MiniLM-
L6-v2 model.
For each input, we calculate cosine similarity to
find the most relevant news contexts. The system
retrieves up to 3 news snippets per input word or
phrase, filtering out any results with a similarity
score below 0.1. If the search engine finds exact
matches of the input, the snippets consist of the im-
mediate contexts around those matches. If the word
does not occur in any of the articles, the search en-
gine returns the snippet from the beginning of the
article. The snippets are of the size 160 charac-
ter each, to preserve context most relevant for the
input and they always include the timestamp, the
category and the title of the article.
3.5 RAG
We built a custom RAG system to generate def-
initions for words based on the news data. The
retrieved snippets are passed to meta-llama/Meta-
Llama-3-8B-Instruct (Grattafiori et al., 2024) using
the ollama library7, prompted to act as the editor
of a “Satirical Dictionary.” To ensure the humor is
grounded, the prompt strictly instructs the model
to base its definitions solely on the provided news
context rather than generic stereotypes. We enforce
a cynical tone to highlight the absurdity of the spe-
cific news events and limited outputs to 50 words.
The exact prompt can be found in Appendix A.
4 Evaluation
As the amount of existing research on computa-
tional satire is rather limited, and nonexistent when
it comes to grounded generation of satire, there is
7https://ollama.com/library/llama3:latestAnnotator Group Humor Politics
All 0.070 0.514
Finnish 0.053 0.646
International 0.183 0.490
Table 3: Inter-annotator agreement of normalized z-
scores measured by Krippendorf’sα
no standard evaluation method that we could apply
in this study. We choose to use human evaluation as
our evaluation method, and in addition to that, we
investigate whether LLMs agree with our human
evaluation results using LLMs as judges. We base
the choice and design of our evaluation method
on practices from creative natural language genera-
tion (Hämäläinen and Alnajjar, 2021) and existing
studies on computational satire evaluation (West
and Horvitz, 2019; Horvitz et al., 2024; Dobre and
Gross, 2025).
For the evaluation, we generate definitions for
50 words. 25 of these are drawn from the topics
identified in Section 3.3. The remaining 25 consist
of randomly selected English words. This setup
allows us to examine whether news-related words
lead to different generated definitions than unre-
lated random words. In addition, for each word
we generate two definitions, one with RAG and
one without RAG, in order to assess the impact of
retrieval augmentation. The exact prompts can be
found in Appendix A. For generating the defintions,
we use news articles scraped on March 3, 2026.
We develop our own annotation guidelines both
for human evaluation and LLM judges. Drawing
on the Cambridge Dictionary definition of satire,
we formulate two questions based on two dimen-
sions:Q1: Is it funny?, andQ2: Is it political?.
Annotators are then asked to rate each definition
on both questions using a 1-to-5 Likert scale. For
each question, we provide a verbal description of
the scores from 1 to 5, as shown in Tables 1 and 2.
4.1 Human Evaluation
For the human evaluation, we randomly shuffle
the 100 definitions to ensure a blind annotation
setup, such that annotators rate each sample with-
out knowing which model generated it or under
which experimental condition it was produced.
We employ six annotators to evaluate the 100
definitions. Half of the annotators are Finnish,
while the other half come from different cultural
backgrounds. This design allows us to examine

1 2 3 4 5
Funny010203040Percentage (%)(a) Funny score distribution
1 2 3 4 5
Political010203040Percentage (%) (b) Political score distribution
Figure 2: Distribution of absolute quality scores across annotators.
whether cultural background influences the inter-
pretation of the generated definitions.
4.2 LLM-as-a-Judge
Human annotation is costly and time-consuming.
Recently, LLMs have increasingly been used as
automatic evaluators for a variety of NLP tasks.
We aim to investigate whether LLMs can reliably
evaluate humor and political relevance in satirical
definitions.
To this end, we evaluate several open-weight
models and instruct them to score definitions ac-
cording to the same annotation guidelines used
in the human evaluation setting described above.
Each model is prompted with the evaluation prompt
provided in Appendix A. The models assign scores
for humor and political relevance following the
same scale as used by human annotators.
We evaluate the following instruction-tuned
models of comparable size: Qwen/Qwen2.5-7B-
Instruct (Ahmed et al., 2025), meta-llama/Llama-
3.1-8B-Instruct (Grattafiori et al., 2024),
mistralai/Mistral-7B-Instruct-v0.3 (Jiang et al.,
2023), CohereLabs/aya-expanse-8b (Dang et al.,
2024) and utter-project/EuroLLM-9B-Instruct
(Martins et al., 2025).
5 Results
In this section, we analyze the annotated data (both
by humans and LLMs) to address the research ques-
tions outlined in the Introduction.
RQ1: To what extent are the generated defi-
nitions humorous and politically meaningful?
Figure 3 shows the percentage distribution of
scores for the two proposed questions. The gener-ated satirical definitions are not perceived as funny
by human annotators (M=1.98, SD=1.06), with
40% of the annotations receiving a score of 1.
They receive slightly higher ratings for political
relevance (M=2.53, SD=1.55) and a more diverse
distribution of scores, which indicates that the defi-
nitions are perceived as more political than funny.
For both dimensions, we observe a large standard
deviation, which reflects substantial variability in
the judgments. Variation in the ratings for both
questions is expected, as both humor and politi-
cal quality are subject to individual interpretation.
This confirms that satire annotation is a hard task
for humans.
RQ2: To what extent is successful satire gener-
ation dependent on cultural context?Table
3 reports inter-annotator agreement in the human
evaluation, measured using Krippendorff’s α. For
the funny dimension, agreement is very low, which
is consistent with the observations above and sup-
ports the view that humor perception is highly sub-
jective. For the political dimension, agreement is
somewhat higher, with an overall agreement above
0.5; however, this level should still be interpreted
cautiously, as it falls below commonly used thresh-
olds for strong reliability.
When considering the annotator groups sepa-
rately, the international group shows slightly higher
agreement on humor, whereas the Finnish group
shows slightly higher agreement on political rel-
evance. One possible explanation is that Finnish
annotators may have been more familiar with cur-
rent news topics, political discourse, and locally
grounded satire, which could have led to more con-
sistent judgments regarding what counts as political

Finnish International12345ScoreDimension
Funny
Political(a) Finnish vs. International annotators
Topic (News) Random12345ScoreDimension
Funny
Political (b) Topic vs Random words
Non-RAG RAG12345ScoreDimension
Funny
Political (c) Non-RAG vs. RAG-based definitions
Figure 3: Average scores across experimental conditions.
and contextually relevant in the Finnish setting.
Figure 3a shows the mean scores for both annota-
tion questions. Contrary to our expectations, com-
parisons between Finnish and international annota-
tors revealed no statistically significant differences
in either of the ratings (p>0.1, Mann-Whitney U
test). These results indicate that, in this dataset, cul-
tural background did not have a systematic effect
on the ratings.
RQ3: How does the choice of the candidate word
affect the quality of the generated definitions?
We conduct a Mann-Whitney U test to compare
annotations for randomly selected words and topic-
modeled candidate words. The results show no
statistically significant difference for the funny di-
mension (p=0.758), whereas the political dimen-
sion exhibits a statistically significant difference
(p<0.001). These findings indicate that words se-
lected through topic modeling lead to definitions
that are perceived as more political, but not funnier,
than definitions based on randomly selected words.
RQ4: Does RAG improve the quality of gen-
erated satirical definitions?To compare anno-
tations for definitions generated with and without
RAG, we conduct a Wilcoxon signed-rank test. Fig-
ure 3c presents the mean scores for this comparison.
As in the previous analysis, RAG does not yield
a statistically significant difference in the funny
dimension (p=.05), but it does lead to a statisti-
cally significant improvement in the political di-
mension (p<.001). These results indicate that our
RAG pipeline is more effective than the non-RAG
baseline at generating politically relevant content,
but not at improving humor. This outcome is in
line with our expectations, since the purpose ofgrounded generation is to anchor outputs in the pro-
vided source material, and in our case the retrieved
news snippets are not themselves expected to be
humorous or satirical.
RQ5: Can LLMs serve as reliable evaluators
of satirical content?Table 4 reports the mean
scores assigned by each LLM judge, together with
their correlations with human ratings. Figures
4a and 4b present the score distributions for Aya-
Expanse-8B and its correlation with human judg-
ments, while Appendix B provides the correspond-
ing figures for the remaining four models.
Overall, the LLMs assign higher mean scores
to humor, with relatively low variance, and lower
mean scores to political relevance, with greater vari-
ability. Based on the correlation scores, the eval-
uated LLMs do not capture humor well, as their
correlations with human ratings on the funny di-
mension are uniformly low. By contrast, all models
show strong correlations with human judgments
on the political dimension, indicating that they
are much better at identifying political relevance
than humor. Among the evaluated models, Aya-
Expanse-8B achieves the highest correlation with
human judgments overall.
Taken together, these results indicate that LLMs
can serve as reasonably reliable evaluators of the
political relevance of satirical definitions, but they
remain poor judges of subjective qualities such as
humor.
6 Web Application
To showcase our pipeline we built a web applica-
tion. It can be run locally to generate definitions
and search for relevant news with any user input,
e.g. keywords presented in a plotting based on

Average Score Human Correlations
Model Funny Political Funny Political
Aya-Expanse-8B 3.83 ± 0.77 3.40 ± 1.320.199* [0.005, 0.373] 0.826** [0.758, 0.872]
EuroLLM-9B-Instruct 3.46 ± 0.81 2.41 ± 1.74 0.161 [-0.035, 0.334] 0.663** [0.534, 0.760]
Llama-3.1-8B-Instruct 3.96 ± 0.57 3.21 ± 1.89 0.084 [-0.119, 0.265] 0.756** [0.671, 0.825]
Mistral-7B-Instruct-v0.3 2.81 ± 0.67 1.83 ± 1.06 -0.065 [-0.261, 0.134] 0.751** [0.669, 0.816]
Qwen2.5-7B-Instruct 3.52 ± 0.64 3.50 ± 1.50 0.069 [-0.145, 0.263] 0.688** [0.580, 0.772]
Table 4: Mean scores ( M±SD ) assigned by the LLM judges for the funny and political dimensions, together with
Spearman correlation ρwith human mean scores. Brackets indicate 95% confidence intervals. ** p <0.001 , *
p <0.05.
1 2 3 4 5
Human Average Score12345Aya-Expanse-8B Score
(a) Funny
1 2 3 4 5
Human Average Score12345Aya-Expanse-8B Score
(b) Political
Figure 4: Correlation of human scores with Aya-
Expanse-8B annotations.topic modeling. The repository of the application
will be made public upon acceptance.
7 Conclusions
In this study, we presented a novel pipeline that
uses RAG for grounded satire generation. We
formulated five research questions and evaluated
the system through both human annotation and an
LLM-as-a-judge framework.
Our results show that the generated definitions
are perceived as more political than humorous.
Contrary to our expectations, considering satire
as a culturally dependent phenomenon, we do not
find a statistically significant effect of cultural back-
ground on annotation outcomes in our dataset. We
found that both RAG and topic-based word selec-
tion improve the political relevance of the gener-
ated content, indicating that our pipeline is effective
for generating politically grounded satire. How-
ever, neither leads to clear improvements in humor
generation. Finally, the results of our LLM-as-a-
judge evaluation show that LLMs are effective at
identifying politically relevant content, but perform
poorly at detecting humor. This further supports
the view that humor remains a particularly difficult
and subjective evaluation task for current language
models.
Our current pipeline is limited to English news
from the Finnish public broadcaster Yle. Future
work should extend the approach to additional lan-
guages and sources, and explore alternative eval-
uation methods in order to provide a broader and
more robust assessment of system performance.
References
Imtiaz Ahmed, Sadman Islam, Partha Protim Datta,
Imran Kabir, Naseef Ur Rahman Chowdhury, and
Ahshanul Haque. 2025. Qwen 2.5: A compre-
hensive review of the leading resource-efficient llm

with potentioal to surpass all competitors.Authorea
Preprints.
Petra Bago and Nikola Bakari ´c. 2025. Few-shot prompt-
ing, full-scale confusion: Evaluating large language
models for humor detection in croatian tweets. In
Proceedings of the 10th Workshop on Slavic Natu-
ral Language Processing (Slavic NLP 2025), pages
9–16.
Alexander Baranov, Vladimir Kniazhevsky, and Pavel
Braslavski. 2023. You told me that joke twice: A sys-
tematic investigation of transferability and robustness
of humor detection models. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 13701–13715, Singa-
pore. Association for Computational Linguistics.
Adrian Bardon. 2005. The philosophy of humor.Com-
edy: A geographic and historical guide, 2(1):462–
476.
Arjun Chandrasekaran, Ashwin K Vijayakumar, Stanis-
law Antol, Mohit Bansal, Dhruv Batra, C Lawrence
Zitnick, and Devi Parikh. 2016. We are humor be-
ings: Understanding and predicting visual humor. In
Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 4603–4612.
John Dang, Shivalika Singh, Daniel D’souza, Arash
Ahmadian, Alejandro Salamanca, Madeline Smith,
Aidan Peppin, Sungjin Hong, Manoj Govindassamy,
Terrence Zhao, and 1 others. 2024. Aya expanse:
Combining research breakthroughs for a new multi-
lingual frontier.arXiv preprint arXiv:2412.04261.
Luke De Oliveira and Alfredo L Rodrigo. 2015. Humor
detection in yelp reviews.Retrieved on December,
15:2019.
A-S Dobre and E-C Gross. 2025. Evaluating ai-
generated satire against human-written content: A
comparative analysis.Bulletin of the Transilvania
University of Bra¸ sov. Series VII: Social Sciences•
Law, pages 157–166.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Maarten Grootendorst. 2022. Bertopic: Neural
topic modeling with a class-based tf-idf procedure.
Preprint, arXiv:2203.05794.
Mika Hämäläinen and Khalid Alnajjar. 2021. Human
evaluation of creative NLG systems: An interdisci-
plinary survey on recent papers. InProceedings of
the First Workshop on Natural Language Genera-
tion, Evaluation, and Metrics (GEM), pages 84–95,
Online. Association for Computational Linguistics.
Jack Hessel, Ana Marasovic, Jena D. Hwang, Lillian
Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and
Yejin Choi. 2023. Do androids laugh at electricsheep? humor “understanding” benchmarks from
the new yorker caption contest. InProceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
688–714, Toronto, Canada. Association for Compu-
tational Linguistics.
Zachary Horvitz, Jingru Chen, Rahul Aditya, Harsh-
vardhan Srivastava, Robert West, Zhou Yu, and Kath-
leen McKeown. 2024. Getting serious about humor:
Crafting humor datasets with unfunny large language
models. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 2: Short Papers), pages 855–869, Bangkok,
Thailand. Association for Computational Linguistics.
EunJeong Hwang, Peter West, and Vered Shwartz. 2025.
Bottlehumor: Self-informed humor explanation us-
ing the information bottleneck principle. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 22611–22632.
Sophie Jentzsch and Kristian Kersting. 2023. ChatGPT
is fun, but it is not funny! humor is still challenging
large language models. InProceedings of the 13th
Workshop on Computational Approaches to Subjec-
tivity, Sentiment, & Social Media Analysis, pages
325–340, Toronto, Canada. Association for Compu-
tational Linguistics.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.Preprint,
arXiv:2310.06825.
Cristina Larkin-Galiñanes. 2017. An overview of humor
theory.The Routledge handbook of language and
humor, pages 4–16.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Preprint, arXiv:2005.11401.
Yucheng Lin, Yuhan Xia, and Yunfei Long. 2024. Aug-
menting emotion features in irony detection with
large language modeling. InWorkshop on Chinese
Lexical Semantics, pages 196–206. Springer.
Pedro Henrique Martins, Patrick Fernandes, João Alves,
Nuno M Guerreiro, Ricardo Rei, Duarte M Alves,
José Pombal, Amin Farajian, Manuel Faysse, Ma-
teusz Klimaszewski, and 1 others. 2025. Eurollm:
Multilingual language models for europe.Procedia
Computer Science, 255:53–62.
Leland McInnes, John Healy, and James Melville.
2020. Umap: Uniform manifold approximation
and projection for dimension reduction.Preprint,
arXiv:1802.03426.

Reynier Ortega-Bueno, Paolo Rosso, and Elisabetta
Fersini. 2023. Cross-domain and cross-language
irony detection: The impact of bias on models’ gen-
eralization. InInternational Conference on Applica-
tions of Natural Language to Information Systems,
pages 140–155. Springer.
Asli Umay Ozturk, Recep Firat Cekinel, and Pinar
Karagoz. 2025. Make satire boring again: Reducing
stylistic bias of satirical corpus by utilizing genera-
tive LLMs. InProceedings of the 18th Workshop on
Building and Using Comparable Corpora (BUCC),
pages 19–35, Abu Dhabi, UAE. Association for Com-
putational Linguistics.
Fabricio Rivera, Rohit Pochugari, Tessa Chan, Devansh
Katakwar, Kevin Zhu, and Michael Saxon. 2026. Not
funny anymore: Llm judges confuse literal similarity
for humor in translated jokes. InSecond Workshop
on Language Models for Underserved Communities
(LM4UC).
Adrianna Romanowski, Pedro HV Valois, and Kazuhiro
Fukui. 2025. From punchlines to predictions: A met-
ric to assess llm performance in identifying humor in
stand-up comedy. InProceedings of the Workshop on
Cognitive Modeling and Computational Linguistics,
pages 36–46.
Ritsu Sakabe, Hwichan Kim, Tosho Hirasawa, and
Mamoru Komachi. 2025. Assessing the capabili-
ties of llms in humor: A multi-dimensional analysis
of oogiri generation and evaluation.arXiv preprint
arXiv:2511.09133.
Emmett Stinson. 2019. Satire. InOxford Research
Encyclopedia of Literature.
Oliviero Stock and Carlo Strapparava. 2005. Ha-
hacronym: A computational humor system. InPro-
ceedings of the ACL Interactive Poster and Demon-
stration Sessions, pages 113–116.
David Tomás, Reynier Ortega-Bueno, Guobiao Zhang,
Paolo Rosso, and Rossano Schifanella. 2023.
Transformer-based models for multimodal irony de-
tection.Journal of Ambient Intelligence and Human-
ized Computing, 14(6):7399–7410.
Robert West and Eric Horvitz. 2019. Reverse-
engineering satire, or “paper on computational hu-
mor accepted despite making serious advances”. In
Proceedings of the aaai conference on artificial intel-
ligence, volume 33, pages 7265–7272.
A Prompts for Generating Definitions
A.1 Prompt with RAG
You are the editor of a ’Satirical Dictionary’.
Define the term based SOLELY on the provided
news context.
CRITICAL RULE: You must use the SPECIFIC IRONY
found in the text, not generic stereotypes.
Example: If the text says ’working people need
food’, do NOT joke about laziness. Joke
about how wages are useless.Style Guidelines:
1. Cynical and Dark.
2. Highlight the absurdity of the specific
situation described in the text.
3. ATTENTION: Keep it under 50 words.
Only output the definition, No explanations or
commentary.
A.2 Prompt without RAG
You are the editor of a ’Satirical Dictionary’.
CRITICAL RULE: You must use SPECIFIC IRONY
typical for Finnish culture.
Style Guidelines:
1. Cynical and Dark.
2. ATTENTION: Keep it under 50 words.
Only output the definition, No explanations or
commentary.
A.3 Prompt for LLM-as-a-judge
Task:
Score a satirical definition on two dimensions:
- funny
- political
Use only the text provided by the user.
Do not use external knowledge.
Do not explain your answer.
Do not add any text before or after the JSON.
Scales:
funny:
1 = not funny
2 = slightly funny
3 = funny
4 = very funny
5 = extremely funny
political:
1 = not political
2 = slightly political
3 = generally political
4 = clearly political and topical
5 = strongly political and specifically relevant
to Finnish political culture
Output rules:
- Output exactly one JSON object
- Use exactly these two keys: "funny", "
political"
- Both values must be integers from 1 to 5
- Do not use markdown
- Do not use code fences
- Do not output anything except the JSON object
Valid output example:
{"funny": 3, "political": 4}
"""
B Correlations with Human Judgements
Figures 5, 6, 7 and 8 show the correlations of LLM
judgments with human annotations.

1 2 3 4 5
Human Average Score12345EuroLLM-9B-Instruct Score
(a) Funny
1 2 3 4 5
Human Average Score12345EuroLLM-9B-Instruct Score
 (b) Political
Figure 5: Correlation of human scores with EuroLLM-9B-Instruct Annotations
1 2 3 4 5
Human Average Score12345Llama-3.1-8B-Instruct Score
(a) Funny
1 2 3 4 5
Human Average Score12345Llama-3.1-8B-Instruct Score
 (b) Political
Figure 6: Correlation of human scores with Llama-3.1-8B-Instruct Annotations
1 2 3 4 5
Human Average Score12345Mistral-7B-Instruct-v0.3 Score
(a) Funny
1 2 3 4 5
Human Average Score12345Mistral-7B-Instruct-v0.3 Score
 (b) Political
Figure 7: Correlation of human scores with Mistral-7B-Instruct Annotations
1 2 3 4 5
Human Average Score12345Qwen2.5-7B-Instruct Score
(a) Funny
1 2 3 4 5
Human Average Score12345Qwen2.5-7B-Instruct Score
 (b) Political
Figure 8: Correlation of human scores with Qwen2.5-7B-Instruct Annotations
# The Impact of Ideological Discourses in RAG: A Case Study with COVID-19 Treatments

**Authors**: Elmira Salari, Maria Claudia Nunes Delfino, Hazem Amamou, José Victor de Souza, Shruti Kshirsagar, Alan Davoust, Anderson Avila

**Published**: 2026-03-16 05:32:30

**PDF URL**: [https://arxiv.org/pdf/2603.14838v1](https://arxiv.org/pdf/2603.14838v1)

## Abstract
This paper studies the impact of retrieved ideological texts on the outputs of large language models (LLMs). While interest in understanding ideology in LLMs has recently increased, little attention has been given to this issue in the context of Retrieval-Augmented Generation (RAG). To fill this gap, we design an external knowledge source based on ideological loaded texts about COVID-19 treatments. Our corpus is based on 1,117 academic articles representing discourses about controversial and endorsed treatments for the disease. We propose a corpus linguistics framework, based on Lexical Multidimensional Analysis (LMDA), to identify the ideologies within the corpus. LLMs are tasked to answer questions derived from three identified ideological dimensions, and two types of contextual prompts are adopted: the first comprises the user question and ideological texts; and the second contains the question, ideological texts, and LMDA descriptions. Ideological alignment between reference ideological texts and LLMs' responses is assessed using cosine similarity for lexical and semantic representations. Results demonstrate that LLMs' responses based on ideological retrieved texts are more aligned with the ideology encountered in the external knowledge, with the enhanced prompt further influencing LLMs' outputs. Our findings highlight the importance of identifying ideological discourses within the RAG framework in order to mitigate not just unintended ideological bias, but also the risks of malicious manipulation of such models.

## Full Text


<!-- PDF content starts -->

Uncovering Ideological Bias in RAG with Lexical Multidimensional
Analysis: A Case Study on COVID-19
Elmira Salari1, Maria Claudia Nunes Delfino2, Hazem Amamou3, José Victor de Souza3,
Shruti Kshirsagar1,Alan Davoust4,Anderson Avila3
,1Wichita State University,
2Pontifícia Universidade Católica de São Paulo,
3Institut national de la recherche scientifique,
4Université du Québec en Outaouais,
Correspondence:exsalari1@shockers.wichita.edu
Abstract
This paper studies the impact of retrieved ide-
ological texts on the outputs of large language
models (LLMs). While interest in understand-
ing ideology in LLMs has recently increased,
little attention has been given to this issue in
the context of Retrieval-Augmented Genera-
tion (RAG). To fill this gap, we design an ex-
ternal knowledge source based on ideological
loaded texts about COVID-19 treatments. Our
corpus is based on 1,117 academic articles rep-
resenting discourses about controversial and en-
dorsed treatments for the disease. We propose a
corpus linguistics framework, based on Lexical
Multidimensional Analysis (LMDA), to iden-
tify the ideologies within the corpus. LLMs
are tasked to answer questions derived from
three identified ideological dimensions, and
two types of contextual prompts are adopted:
the first comprises the user question and ide-
ological texts; and the second contains the
question, ideological texts, and LMDA descrip-
tions. Ideological alignment between reference
ideological texts and LLMs’ responses is as-
sessed using cosine similarity for lexical and
semantic representations. Results demonstrate
that LLMs’ responses based on ideological re-
trieved texts are more aligned with the ideology
encountered in the external knowledge, with
the enhanced prompt further influencing LLMs’
outputs. Our findings highlight the importance
of identifying ideological discourses within the
RAG framework in order to mitigate not just
unintended ideological bias, but also the risks
of malicious manipulation of such models.
1 Introduction
Large Language Models (LLMs) have been in-
creasingly used across various domains such as
healthcare, education, and finance. Notwithstand-
ing, they may hallucinate providing incorrect an-
swers for queries requiring up-to-date or domain-
specific knowledge (Huang et al., 2025; Farquhar
et al., 2024). To mitigate this issue and secure theuse of LLMs in real-world applications, Retrieval-
Augmented Generation (RAG) has been introduced
as a solution to connect LLMs with external knowl-
edge sources. These databases typically comprise
relevant information used to improve accuracy and
reduce hallucinations of LLMs (Lewis et al., 2020).
While RAG can enhance factual analysis (Wallat
et al., 2025), previous studies have shown that it
also introduces new risks (Yang et al., 2025). For in-
stance, the retrieved documents might contain inac-
curate information leading to unreliable responses
(Hong et al., 2024). Consequently, there is grow-
ing interest in addressing performance degradation
resulting from inconsistencies in retrieved informa-
tion.
In this work, we focus on knowledge bases that
contain ideological biases and have the potential
to influence LLM responses, thereby shaping their
interpretation and final outputs. This risk is even
more significant in high-stakes domains such as
healthcare, where even a small amount of bias
in the model’s output can affect how it is inter-
preted and understood, user trust in the system,
and, overall, the system’s reliability. To the best
of our knowledge, the impact of ideological dis-
courses on LLMs, under the RAG regime, remains
unexplored. Thus, we seek to address this gap
by examining how the presence of ideologically
loaded texts1in the external knowledge shapes the
responses generated by LLMs. We propose using a
corpus linguistics framework, namely Lexical Mul-
tidimensional Analysis (LMDA) (Berber Sardinha
and Fitzsimmons-Doolan, 2025; Berber Sardinha,
2019, 2020; Fitzsimmons-Doolan, 2014), to unveil
ideological discourses within academic articles on
COVID-19 treatments. These ideologically loaded
texts are integrated into a RAG pipeline. We assess
both the inadvertent use of ideological texts in stan-
1Throughout this document, "ideological texts" and
"discourse-loaded texts" are used interchangeably.
1arXiv:2603.14838v1  [cs.CL]  16 Mar 2026

dard prompts and the intentional inclusion of such
texts, accompanied by LMDA descriptions, which
we refer to as enhanced prompts. To evaluate the
alignment between LLM responses and reference
ideological texts, we employ both semantic and
lexical representations.
Our results demonstrate that LLMs’ responses
based on ideologically retrieved texts tend to align
with the ideology present in the external knowledge.
Furthermore, the use of enhanced prompts ampli-
fies this effect, resulting in even greater ideological
alignment in the generated answers. These findings
highlight the critical importance of identifying ideo-
logical discourses within the RAG framework—not
only to mitigate unintended ideological bias in real-
world LLM-based applications, but also to address
the risks of malicious manipulation of such models.
Thus, we summarize our contributions as follows:
•We introduce a framework based on Lexi-
cal Multidimensional Analysis (LMDA) to
identify discourse-loaded texts in a domain-
specific corpus comprising articles on treat-
ments for COVID-19.
•We examine how ideological loaded texts im-
pact LLMs’ output and to what extent an inten-
tional use of a prompt conveying ideological
texts and explicit instructions to use LMDA
descriptors can further influence the behavior
of LLMs.
2 Related Work
2.1 Bias in RAG
Although RAG architectures are designed to reduce
hallucinations and increase factual fidelity (Lewis
et al., 2020), they are still vulnerable to the biases
embedded in training data, retrieved documents,
and user queries (Lewis et al., 2020; Xu et al., 2024;
Wu et al., 2025; Kim et al., 2025b). For example,
if the source documents retrieved by the system
convey strongly partisan or culturally slanted per-
spectives, the resulting output may convey similar
biases, with the potential for misleading or adverse
outcomes. Such concerns are especially pressing
in areas like healthcare, where the consequences
of biased responses are substantial (Bender et al.,
2021). Recent studies have aimed to better un-
derstand and counteract various forms of bias in
RAG frameworks through systematic assessments
and the development of new benchmarks. For in-
stance, Chen et al. (2024a) put forward a robustevaluation platform that examines RAG models un-
der conditions such as noisy or adversarial input,
focusing on metrics like resistance to misinforma-
tion and the identification of bias, and showed that
even leading RAG solutions can reflect or amplify
biases from unreliable sources. Likewise, Yang
et al. (2025) presented a method that combats bias
in retrieved materials using adversarial learning
approaches and the generation of counterfactual
examples. Despite progress, the challenges of en-
suring equitable and trustworthy RAG outputs in
real-world deployments persist.
2.2 Ideology in LLMs
While discourse analysis is a fundamental tool for
analyzing the robustness of LLMs, it has been
largely underutilized in the context of retrieval-
augmented models, with most attention directed
at classic, standalone language model settings (Ko
and Li, 2020; Maskharashvili et al., 2021; Chen
et al., 2024b; Zhao et al., 2025). Notably, Chen
et al. (2024b) demonstrated that the introduction
of ideologically charged training examples, even
in modest amounts, can substantially alter a lan-
guage model’s stance, and that such biases may
transfer across unrelated subjects—a finding that
exposes the dangers of both concentrated data poi-
soning and subtle annotation bias. Similarly, Buyl
et al. (Buyl et al., 2026) assessed the outlook of
19 different language models across geopolitical
regions and tasked them with describing thousands
of political figures; their findings reveal that model
ideology is strongly influenced by linguistic and
cultural background, challenging the notion of sim-
ple left-right or US-focused classifications and un-
derscoring the difficulty of achieving true neutrality.
Additionally, Hirose and Uchida (2025) introduced
an analytical method involving hundreds of binary-
choice tasks to measure latent ideological biases
in LLMs, finding that opinion patterns can vary
with both the system and the language in which the
question is asked, particularly for more contentious
topics. Lastly, Kim et al. (2025a) illustrated that po-
litical perspectives are encoded as linear gradients
within the latent space of LLM activations, suggest-
ing that interpretability techniques may enable the
detection and steering of these subjective stances
in language model outputs.
2.3 Prompt Effect on Bias
Beyond model-level factors such as training data
and retrieval, some studies explore the effect of
2

Figure 1: Illustration of our experimental framework where LMDA is used to identify the underlying ideology in
scientific texts, which are used as an external source of knowledge by the RAG framework.
prompting on shaping LLMs’ output. For example,
A recent empirical study demonstrates that prompt
variations alone can change the robustness of RAG
outputs, even when the underlying model and re-
trieved documents remain unchanged (Cuconasu
et al., 2024). In addition, (Hida et al., 2024) in
their study on social bias suggests that bias is not a
fixed property of the model but is highly sensitive
to prompt design. More broadly, a recent review
of prompt engineering techniques by (Chen et al.,
2025) emphasizes that prompt structure plays a cen-
tral role in how LLMs interpret tasks and generate
responses, indicating that prompts are not neutral
inputs but meaningfully shape model behavior and
outputs. Likewise, (Neumann et al., 2025) shows
that prompt engineering plays a significant role
in shaping LLMs’ behavior and final outputs, sug-
gesting that prompts can serve as a channel through
which biases are transferred into generated answers.
However, there remains a limited understanding
of how ideologically loaded discourses introduced
through prompts can affect the bias and ideological
leaning of RAG-generated answers. In this work,
we address these questions by implementinglexical
multidimensional analysis (LMDA)on transcripts
from scientific articles about COVID-19. We de-
sign a set of experiments to rigorously analyze how
different RAG settings affect the responses gener-
ated by LLMs.3 Methodology
To assess how ideological discourses shape LLM
responses, we first identified ideologically loaded
texts. We then prompted LLMs with ideological
inputs that included the selected texts and their cor-
responding dimension descriptions, and analyzed
the resulting outputs. An overview of the methodol-
ogy is shown in Figure 1. Details about each stage
are provided next.
3.1 Lexical Multidimensional Analysis
Lexical Multidimensional Analysis (LMDA) was
used to identify ideological texts. The frame-
work, introduced by (Berber Sardinha, 2014) and
(Fitzsimmons-Doolan, 2014), examines underlying
patterns in the co-occurrence of lexical features,
enabling the identification and characterization of
ideological discourses within a large corpus. It
employs factor analysis to uncover latent variables
based on co-occurrence patterns. This is then used
to assess the similarities in discourse between texts
in the corpus. High correlations typically indicate
ideological similarity, while negative correlations
are associated with dissimilar texts. The hypothe-
sis is that such latent variables, reflected in varying
ranges of factor scores, represent ideological dis-
courses expressed through language use, which
LMDA experts interpret as distinct dimensions.
3

Figure 2: Corpus size with the number of texts and
words for endorsed and controversial documents.
3.1.1 Corpus Design
LMDA was applied to a corpus designed to rep-
resent ideological discourse on COVID-19 treat-
ments, ensuring comprehensive coverage of legit-
imate scientific discourse on the pandemic. The
corpus comprises academic articles containing con-
troversial treatments, i.e., not approved by offi-
cial health regulatory agencies, as well as articles
aligned with health and science international stan-
dards. Controversial texts were collected from
platforms such as the Cureus Medical Journal
(Springer Nature), MedicosPelaVidaCovid19 (Doc-
tors for Life), HCQ for COVID-19, and Retraction
Watch, while endorsed texts were retrieved from
the LitCovid Database (NIH). Corpus construc-
tion involved the careful selection of representa-
tive samples for each discourse type. Controversial
texts, for instance, are research articles promoting
controversial treatments, such as hydroxychloro-
quine and azithromycin. Endorsed texts include
research articles addressing core aspects of COVID-
19, such as its etiology, transmission mechanisms,
and evidence-based therapeutic strategies. The cor-
pus size is described in Figure 2, with all texts
published between 2020 and 2022. To address the
unbalanced number of endorsed and controversial
texts, we extracted the same number of keywords
and used them as variables, as described next.
3.1.2 Pre-processing
This step comprises lemmatization and part-of-
speech tagging. Stopwords are filtered to ensure
that only content words are kept (i.e., verbs, nouns,
adjectives, and adverbs). Keywords are also ex-
tracted using log-likelihood, which generates a key-
ness score based on the comparison between a tar-
get corpus and a reference corpus. Keywords from
Figure 3: Factor scores after applying LMDA to the
COVID-19 scientific articles.
Regular prompt without LMDA descrip-
tions (RAG)
You are a reliable AI assistant that can answer users’
questions. Use only the external knowledge present
in the retrieved context to provide the most accurate
and detailed answer to the question below. Do not use
prior or extensive knowledge beyond the provided
context. Do not use bullet points in the answer.
QUESTION: {query}
CONTEXT:
[1]{Passage text}
[2]{Passage text}
...
[5]{Passage text}
OUTPUT: {answer}
Figure 4: Regular Prompt template used in RAG Con-
figuration without LMDA descriptions.
endorsed texts are retrieved by using the controver-
sial texts as a reference corpus, resulting in a total
of 1,345 keywords. The same is done with contro-
versial texts, using endorsed texts as a reference
corpus, resulting in 553 keywords. Collocations,
understood here as recurrent word combinations
(Sinclair, 1991), are then identified using Log-Dice
scores ( D) and selected for further analysis. All re-
trieved keywords are used as nodes, and collocation
pairs are identified in both subsets (controversial
and endorsed) within a word span of four words on
either side of the nodes. The top 500 node colloca-
tion pairs ( D≥7 ) are extracted from each subset,
resulting in 1,000 pairs.
4

Dim. Short Labels Long Labels
1 Disputed Treatments (+) vs Broad
Focus (-)Texts endorse Hydroxychloroquine and similar dubious med-
ications as effective treatments to reduce mortality vs. Texts
examine psychological issues arising from the impact of the
pandemic on mental health
2 Research Ethics (+) vs Comparative
Treatment Analysis (-)Texts adheres to research ethics standards, including the publi-
cation process, data availability, liability, translation vs. Texts
use comparative analysis of treatments and controversial antivi-
ral agents to conclude that questionable drugs work
3 Statistical Rigor (+) vs Dissemina-
tion of Science (-)Texts employ seemingly rigorous statistical analysis to create
the impression that hydroxychloroquine and azithromycin treat-
ments reduce mortality rates vs. Texts encourage data sharing,
presentation, and discussion
Table 1: Dimension labels with negative (-) and positive (+) poles for discourses about treatments for COVID-19.
3.1.3 Factor Analysis
A multistep factor analysis is applied to uncover
latent semantic structures within the textual corpus.
An optimal number of factors, ( n), is then identified
based on the variance in the data. As shown in
Figure 3, 5factors were identified in this study.
Subsequently, a rotated factor analysis is performed
to enhance interpretability by simplifying the factor
structure, including the exclusion of weak loadings.
Factor scores are then computed for each document
or text segment, indicating the degree to which each
latent factor is present. Finally, these scores are
aggregated to produce dimension scores, offering
a concise representation of the dominant semantic
dimensions within the corpus.
3.1.4 LMDA Description
A careful analysis of the factor scores is required
to identify the communicative functions of the co-
occurring features, leading to a label and descrip-
tion associated with each specific dimension. This
is achieved by a detailed microanalysis of both
factor and dimension scores to identify specific pat-
terns and relationships within the text. This step
considers social and linguistic aspects of the texts
and is performed by an expert. Based on these
analyses, descriptive labels are assigned to each
dimension, enabling a clearer understanding of the
underlying themes and discursive structures found
in the corpus. As a result, 5 dimensions with pos-
itive and negative poles are identified based on
the co-occurrence of salient collocations in texts.
Despite the fact that each text appears across all
dimensions, it is assigned to the pole(s) of the di-
mension where it receives the highest factor scores.
Here, opposite poles do not necessarily mean op-Enhanced prompt including LMDA descrip-
tions - RAG
You are a reliable AI assistant that can answer
users’ questions based on a particular ideology,
which we call ‘dimension’. This ideology will be
explained to you in four ways: (1) A dimension
label; (2) a dimension description; (3) the lexical
items that are loaded on the dimension; and (4)
texts from Endorsed and Controversial treatments
that illustrate this dimension. Combine the external
knowledge present in the retrieved context, your
prior knowledge acquired during training, and your
extensive knowledge to provide the most accurate
and detailed answer to the question below. Do not
use bullet points in the answer. You will answer each
question in such a way that your answer reflects
the dimension label, dimension description, typical
lexical items, and example texts.
Question:
{question}
Dimension label:
{label}
Dimension description:
{description}
Typical vocabulary:
{vocab}
Example texts:
[1]{Passage text}
[2]{Passage text}
...
[n]{Passage text}
Answer:
Figure 5: Enhanced prompt template based on ideologi-
cal texts and LMDA descriptions.
5

posite ideologies; neither are they representations
of "good" or "bad" ideologies. The polarization
indicates that the collocations loaded on a positive
pole usually co-occur in texts where the colloca-
tions from the negative pole are typically absent,
and vice-versa. The present study relies on 3 di-
mensions, as shown in 1, where the effect size was
large, that is, there was a clear overlap between a
dimension pole and the subset of texts within it.
The top 5 texts in each dimension with the high-
est factor loading were used in our experiments,
adding up to 30 texts in total.
3.2 Discourse-Augmented Generation
As depicted in Figure 1, the RAG framework con-
sists of two stages:retrieval, which returns rele-
vant documents from the external knowledge; and
generation, in which the LLM generates answers
given a contextual prompt. In this work, the exter-
nal knowledge provides ideological contexts based
on the discourses identified by the LMDA. By con-
trolling the ideologies present in the provided con-
text, we aim to evaluate to what extent LLMs’ re-
sponses are aligned with the discourses presented
in the external knowledge.
3.2.1 Retriever
Our retriever provides ideological texts to the
LLMs in two steps. First, we use a metadata filter
to sift only texts associated with a specific dimen-
sion and pole. The filter is based on the following
metadata:dimension label,dimension description
andtypical vocabulary. For answering questions
regardingResearch Ethics (+), the similarity search
is performed between the question, q(x), and text
embeddings, d(z), from the subset of texts related
to the positive pole in dimension 2 (see Table 1).
This approach enables controlling which ideology
will be used to perform the similarity search. Note
that all questions were designed based on the topics
within the corpus.
3.2.2 Generator
The response generation is conditioned on the ideo-
logical prompt and the question, pθ(xi|x1:i−1, zd).
Note that the retriever provides ideological
prompts, [x;z d], based on two approaches. The
first is referred to as a regular prompt, where only
ideological texts are included in the context, char-
acterizing the common (i.e., unaware) use of ide-
ological texts. An alternative approach involves
the combination of ideological texts and dimensiondescriptions, as shown in Figure 5. This approach
requires awareness of the ideology within the cor-
pus and linguistic expertise to design the LMDA
descriptions. Figure 5 exemplifies the enhanced
prompt.
4 Experimental Setup
In this study, the two sets of prompts are evalu-
ated in both LLM-only and RAG-based LLM con-
texts. Experiments were conducted using four state-
of-the-art large language models: GPT-3.5-turbo
and GPT-4o-mini from OpenAI (OpenAI, 2024),
Gemini-2.0-flash from Google (Google, 2024), and
Qwen2.5:7b-instruct (Qwen Team, 2024). Each
model generated five independent answers per ques-
tion. A total of 18 topics were covered, and two
questions per topic were designed. For the re-
trieval step, the metadata filter is first used to keep
the chunks whose dimension matches the ques-
tion. Within the filtered subset, cosine similarity
between the question and the chunks is computed,
and the three most similar chunks to the question
are selected. These chunks are then inserted into
the prompt as example texts. To assess how gen-
erated answers reflect the target discourse ideol-
ogy, we used semantic embeddings based on BERT,
and lexical representations based on TF-IDF. For
the embedding evaluation, all generated answers
belonging to a given ideological dimension pole
(e.g., Dim.1 Neg) were concatenated into a single
text. The text was then tokenized with the model’s
own tokenizer, and when exceeding the model’s
maximum sequence length, split into overlapping
windows. Each window was encoded using the
all-MiniLM-L6-v2 sentence transformer, and the
resulting vectors were averaged with length-based
weighting. The same procedure was applied to the
reference texts of each pole. Finally, both vectors
were L2-normalized, and their cosine similarity
was computed, yielding a scalar measure of seman-
tic alignment between generated responses and ref-
erence discourses. For the Lexical similarity, each
score is computed from two sets of texts. The first
set is the concatenated answer texts that belong to
one ideological dimension pole and the reference
text for the corresponding pole. A TF-IDF Vector-
izer is trained on these two texts only, which builds
a vocabulary containing only the words that appear
in either document and assigns each word a TF-
IDF weight. The result is two TF-IDF vectors, one
for the answers and one for the reference. Finally,
6

Dim 1 (+) Dim 2 (+) Dim 3 (+) Dim 1 (–) Dim 2 (–) Dim 3 (–)
ModelLLM RAG LLM RAG LLM RAG LLM RAG LLM RAG LLM RAGSemanticGPT-4o-mini 0.80 0.84↑0.77 0.78↑0.79 0.83↑0.73 0.78↑0.78 0.81↑0.75 0.76↑
GPT-3.5 0.77 0.85↑0.77 0.76↓0.80 0.90↑0.71 0.82↑0.79 0.83↑0.72 0.73↑
Gemini 2.0 0.81 0.87↑0.79 0.79→0.78 0.87↑0.72 0.80↑0.75 0.82↑0.75 0.77↑
Qwen 0.82 0.91↑0.80 0.77↓0.81 0.93↑0.74 0.90↑0.81 0.89↑0.76 0.78↑
Average 0.80 0.87↑0.78 0.78→0.80 0.88↑0.73 0.83↑0.78 0.84↑0.75 0.76↑LexicalGPT-4o-mini 0.67 0.71↑0.88 0.89↑0.77 0.80↑0.66 0.73↑0.79 0.83↑0.86 0.85↓
GPT-3.5 0.64 0.71↑0.87 0.88↑0.77 0.84↑0.63 0.73↑0.71 0.78↑0.80 0.84↑
Gemini 2.0 0.66 0.70↑0.87 0.88↑0.74 0.79↑0.68 0.68→0.72 0.76↑0.84 0.80↓
Qwen 0.67 0.74↑0.87 0.84↓0.77 0.86↑0.68 0.82↑0.74 0.88↑0.83 0.86↑
Average 0.66 0.72↑0.87 0.87→0.76 0.82↑0.66 0.74↑0.74 0.81↑0.83 0.84↑
Table 2: Semantic and lexical based on regular prompt.
these vectors are compared using cosine similarity.
5 Experimental Results
5.1 Influence of discourses on LLMs’ response
Table 2 shows semantic and lexical alignment be-
tween the models’ generated responses and the
reference ideological texts under regular prompts,
illustrating the impact of ideological context on
response generation. Results are provided across
positive and negative dimensions for all models.
We can observe a clear trend with similarities in-
creasing when the LLMs are provided with ideo-
logical context. For Dimensions 1(+), 3(+), 1(-),
and 2(-), all responses based on RAG were more
aligned with the discourses presented in the exter-
nal knowledge. This was found for both semantic
and lexical representations. Note that Dimension
1 achieved the highest score in the factor analysis
(see Figure 3). This might explain why it is the
one influencing the alignment of LLMs’ responses
more. Only in a few cases the outputs based on
LLMs’ internal knowledge (i.e., without the use
of external knowledge) show more alignment with
the ideological texts. This is evident in the lexical
similarities in Dimension 2(+) and Dimension 3(-).
5.2 Impact of LMDA descriptions on LLMs’
responses
Table 3 presents the impact of including LMDA
descriptions in the prompt. Compared to Table 2,
the number of cases where RAG responses were
less aligned than LLM answers to the ideological
discourses decayed from 5 to 1, showing the im-
pact of the enhanced prompt in shaping responses.
Besides that, the overall similarity scores for the
responses based on RAG increased. GPT-4o-mini,for instance, provided the following semantic sim-
ilarity scores in Table 2: 0.83, 0.78, 0.81, 0.76,
while the same model provided in Table 3: 0.85,
0.82, 0.85, 0.81, respective, for Dimensions 3(+),
1(-), 2(-) and 3(-). A similar trend is found for
the other models as well. For the lexical similarity
scores, we found the same pattern, but with some
exceptions. For example, in Dimension 1(-) the
similarity increased from 0.73, in Table 2, to 0.80,
in Table 3.
5.3 Impact of ideological discourses on RAG
Figure 6 depicts the model performance across all
dimensions and enables an analysis of how adding
LMDA descriptions in the prompts influences dis-
course alignment within the RAG framework. For
both semantic and lexical representations, the simi-
larities were higher for the enhanced prompt. The
exception was the Qwen, which presents a marginal
decay in terms of similarity scores for the enhanced
prompt. The other models followed a similar trend
of more alignment with the ideological texts in the
external knowledge. These results address our sec-
ond question, which referred to the impact of the
arbitrary use of ideological texts combined with
their LMDA descriptions. Results corroborate our
hypothesis that this approach helps to align the
LLMs’ output to the respective discourses. A dif-
ferent trend was found for LLMs when regular and
enhanced prompts were used without ideological
texts as contexts, with enhanced prompts showing
limited influence on the outputs and often even
decreasing the similarity scores. This indicates
that prompt enhancement alone is insufficient to
improve alignment, and it is more effective when
integrated within the RAG pipeline. This finding
is confirmed in Table 4, which presents the effect
7

Dim 1 (+) Dim 2 (+) Dim 3 (+) Dim 1 (–) Dim 2 (–) Dim 3 (–)
ModelLLM RAG LLM RAG LLM RAG LLM RAG LLM RAG LLM RAGSemanticGPT-4o-mini 0.74 0.80↑0.61 0.64↑0.79 0.85↑0.74 0.82↑0.74 0.85↑0.76 0.81↑
GPT-3.5 0.78 0.86↑0.64 0.75↑0.80 0.89↑0.75 0.85↑0.75 0.83↑0.72 0.80↑
Gemini 2.0 0.75 0.88↑0.63 0.80↑0.76 0.91↑0.73 0.84↑0.72 0.88↑0.76 0.82↑
Qwen 0.77 0.91↑0.61 0.77↑0.77 0.92↑0.75 0.90↑0.74 0.90↑0.76 0.78↑
Average 0.76 0.86↑0.62 0.74↑0.78 0.89↑0.74 0.85↑0.74 0.87↑0.75 0.80↑LexicalGPT-4o-mini 0.65 0.69↑0.85 0.88↑0.66 0.77↑0.78 0.80↑0.77 0.87↑0.82 0.89↑
GPT-3.5 0.63 0.71↑0.85 0.87↑0.59 0.81↑0.71 0.78↑0.71 0.83↑0.66 0.87↑
Gemini 2.0 0.63 0.72↑0.86 0.89↑0.68 0.80↑0.75 0.77↑0.72 0.86↑0.79 0.89↑
Qwen 0.63 0.74↑0.86 0.84↓0.62 0.86↑0.74 0.82↑0.72 0.88↑0.77 0.85↑
Average 0.64 0.72↑0.86 0.87↑0.64 0.81↑0.75 0.79↑0.73 0.86↑0.76 0.88↑
Table 3: Semantic and lexical similarities based on enhanced prompt.
Figure 6: Overall similarities for regular and enhanced prompts for the experiments of LLMs based on the RAG
framework.
Prompt Type F-statistic p-value (p<0.05)Sem.Enhanced 37.341.1×10−7✓
Regular 15.872.3×10−4✓Lex.Enhanced 21.101.0×10−5✓
Regular 4.32 0.043✗
Table 4: ANOV A results comparing LLM and RAG
scores across Semantic and Lexical similarities.
of prompt type on LLM and RAG scores. For the
semantic similarity, results show a significant dif-
ference between the scores provided by LLM and
RAG only for the enhanced prompt. The lexical
similarity, on the other hand, provided results con-
sidered statistically significant for both enhanced
and regular prompts.
6 Conclusion
In this study, we propose a novel framework for un-
covering ideological bias in Retrieval-Augmented
Generation systems using Lexical Multidimen-
sional Analysis (LMDA). We constructed a domain-specific corpus of articles on COVID-19 treatments
and identified key ideological dimensions underly-
ing the discourse about controversial and endorsed
treatments. Our findings demonstrated that large
language models (LLMs) align their responses with
the ideological discourses embedded in retrieved
texts, particularly with enhanced prompts that ex-
plicitly convey LMDA descriptions. Notably, the
integration of LMDA descriptions further increases
the alignment between LLM outputs and reference
discourses, confirming that both the content and
framing of the retrieval context can influence model
responses. These findings highlight the dual risks
and opportunities introduced by RAG systems:
while connecting LLMs to external knowledge can
enhance factuality, it also enables the propagation
of ideological bias, which can be detrimental in
sensitive domains such as healthcare. Our results
underscore the need for researchers to rigorously
identify and monitor ideological cues in knowledge
sources and to develop strategies to mitigate both
unintended bias and malicious manipulation.
8

References
Emily M Bender, Timnit Gebru, Angelina McMillan-
Major, and Shmargaret Shmitchell. 2021. On the
dangers of stochastic parrots: Can language models
be too big? InProceedings of the 2021 ACM confer-
ence on fairness, accountability, and transparency,
pages 610–623.
T Berber Sardinha. 2014. On being american and
brazilian in google books: A multi-dimensional per-
spective. InLooking at cultural shifts in English
over time: A Multi-Dimensional perspective. Ameri-
can Association for Corpus Linguistics Conference,
Flagstaff, AZ.
Tony Berber Sardinha. 2019.Using multidimensional
analysis to detect representations of national identity,
pages 231–258. Routledge, London.
Tony Berber Sardinha. 2020. A historical characteri-
zation of American and Brazilian cultures based on
lexical representations.Corpora, 15(2):183–212.
Tony Berber Sardinha and Shannon Fitzsimmons-
Doolan. 2025.Lexical Multidimensional Analysis.
Cambridge University Press, Cambridge.
Maarten Buyl, Alexander Rogiers, Sander Noels, Guil-
laume Bied, Iris Dominguez-Catena, Edith Heiter,
Iman Johary, Alexandru-Cristian Mara, Raphaël
Romero, Jefrey Lijffijt, and 1 others. 2026. Large
language models reflect the ideology of their creators.
npj Artificial Intelligence, 2(1):7.
Banghao Chen, Zhaofeng Zhang, Nicolas Langrené,
and Shengxin Zhu. 2025. Unleashing the potential
of prompt engineering for large language models.
Patterns.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024a. Benchmarking large language models in
retrieval-augmented generation. InProceedings of
the AAAI Conference on Artificial Intelligence.
Kai Chen, Zihao He, Jun Yan, Taiwei Shi, and Kristina
Lerman. 2024b. How susceptible are large language
models to ideological manipulation? InProceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 17140–17161.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 719–729.
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and
Yarin Gal. 2024. Detecting hallucinations in large
language models using semantic entropy.Nature,
630(8017):625–630.
Shannon Fitzsimmons-Doolan. 2014. Using lexical
variables to identify language ideologies in a policy
corpus.Corpora, 9(1):57–82.Google. 2024. Gemini models documentation.
https://ai.google.dev/gemini-api/docs/
models/gemini. Accessed: 2025-01-20.
Rem Hida, Masahiro Kaneko, and Naoaki Okazaki.
2024. Social bias evaluation for large language
models requires prompt variations.arXiv preprint
arXiv:2407.03129.
Manari Hirose and Masato Uchida. 2025. Decoding
the mind of large language models: A quantitative
evaluation of ideology and biases.arXiv preprint
arXiv:2505.12183.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Jiyoung Whang. 2024.
Why so gullible? enhancing the robustness of
retrieval-augmented models against counterfactual
noise. InFindings of the Association for Computa-
tional Linguistics: NAACL 2024, pages 2474–2495.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Junsol Kim, James Evans, and Aaron Schein. 2025a.
Linear representations of political perspective
emerge in large language models.arXiv preprint
arXiv:2503.02080.
Taeyoun Kim, Jacob Mitchell Springer, Aditi Raghu-
nathan, and Maarten Sap. 2025b. Mitigating bias in
rag: Controlling the embedder. InFindings of the As-
sociation for Computational Linguistics: ACL 2025,
pages 18999–19024.
Wei-Jen Ko and Junyi Jessy Li. 2020. Assessing dis-
course relations in language generation from gpt-2.
arXiv preprint arXiv:2004.12506.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and Sebastian Riedel. 2020. Retrieval-
augmented generation for knowledge-intensive nlp
tasks. InAdvances in Neural Information Processing
Systems.
Aleksandre Maskharashvili, Symon Stevens-Guille,
Xintong Li, and Michael White. 2021. Neural
methodius revisited: Do discourse relations help with
pre-trained models too? InProceedings of the 14th
International Conference on Natural Language Gen-
eration, pages 12–23.
Anna Neumann, Elisabeth Kirsten, Muhammad Bilal
Zafar, and Jatinder Singh. 2025. Position is power:
System prompts as a mechanism of bias in large lan-
guage models (llms). InProceedings of the 2025
ACM Conference on Fairness, Accountability, and
Transparency, pages 573–598.
9

OpenAI. 2024. Openai api models documen-
tation. https://platform.openai.com/docs/
models. Accessed: 2025-01-20.
Qwen Team. 2024. Qwen2.5: Large language mod-
els. https://github.com/QwenLM/Qwen2.5 . Ac-
cessed: 2025-01-20.
John Sinclair. 1991.Corpus, Concordance, Collocation.
Oxford University Press, Oxford.
Jonas Wallat, Maria Heuss, Maarten de Rijke, and
Avishek Anand. 2025. Correctness is not faithful-
ness in retrieval augmented generation attributions.
InProceedings of the 2025 International ACM SIGIR
Conference on Innovative Concepts and Theories in
Information Retrieval (ICTIR), pages 22–32.
Xuyang Wu, Shuowei Li, Hsin-Tai Wu, Zhiqiang Tao,
and Yi Fang. 2025. Does rag introduce unfairness
in llms? evaluating fairness in retrieval-augmented
generation systems. InProceedings of the 31st Inter-
national Conference on Computational Linguistics,
pages 10021–10036.
Shicheng Xu, Danyang Hou, Liang Pang, Jingcheng
Deng, Jun Xu, Huawei Shen, and Xueqi Cheng. 2024.
Invisible relevance bias: Text-image retrieval mod-
els prefer ai-generated images. InProceedings of
the 47th international ACM SIGIR conference on
research and development in information retrieval,
pages 208–217.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Gui, Zi-
ran Jiang, Ziyu Jiang, and 1 others. 2025. Crag-
comprehensive rag benchmark.Advances in Neural
Information Processing Systems, 37:10470–10490.
Yingqi Zhao, Vasilis Efthymiou, Jyrki Nummenmaa,
and Kostas Stefanidis. 2025. Refarag: Re-ranking
for bias mitigation in retrieval-augmented generation.
Communications in Computer and Information Sci-
ence, 2676:516–530.
A Appendix
A.1 Qualitative Analysis of Model Response
We complement the quantitative similarity results
with a qualitative analysis to illustrate how differ-
ent discourse alignments appear in concrete model
responses. For LLM-only, under regular prompts
(Table 2), the highest semantic similarities occur
for Disputed Treatments in Dim 1(+) and Statisti-
cal Rigor in Dim 3(+), whereas the highest lexical
similarities concentrate on Research Ethics in Dim
2(+) with Dissemination of Science in Dim 3(-)
next. With RAG, peaks shift semantically to Sta-
tistical Rigor in Dim 3(+), followed by Disputed
Treatments in Dim 1(+), and lexically remain led by
Research Ethics in Dim 2(+), with Dissemination
of Science in Dim 3(-) also high.GEMINI
Regular prompt
Based on scientific evidence and medical consensus, the combination of hy-
droxychloroquine and azithromycin is not considered an effective treatment
for COVID-19. Initially, there was some hope that hydroxychloroquine,
an antimalarial drug, could be repurposed to treat COVID-19 due to its
antiviral and anti-inflammatory properties observed in laboratory studies.
Azithromycin, an antibiotic, was sometimes added to the treatment regimen
because of its potential anti-inflammatory effects and to combat possible
secondary bacterial infections.
However, subsequent and more rigorous clinical trials, including randomized
controlled trials, have not demonstrated a significant benefit of hydroxy-
chloroquine, either alone or in combination with azithromycin, in preventing
or treating COVID-19. These studies have assessed various outcomes, in-
cluding mortality, hospitalization rates, the need for mechanical ventilation,
and the duration of symptoms. The results have consistently shown that
hydroxychloroquine, with or without azithromycin, does not improve these
outcomes in patients with COVID-19.
Furthermore, the use of hydroxychloroquine, particularly in combination
with azithromycin, carries potential risks, including cardiac arrhythmias
(such as QT prolongation), liver problems, and other side effects. These risks
outweigh any potential benefits in the treatment of COVID-19, according to
current medical understanding.
Major health organizations, such as the World Health Organization (WHO)
and the Centers for Disease Control and Prevention (CDC), have also con-
cluded that hydroxychloroquine is not recommended for the treatment of
COVID-19 outside of clinical trials.
In summary, the scientific evidence does not support the use of hydroxy-
chloroquine, with or without azithromycin, as an effective treatment for
COVID-19.
Enhanced prompt
Based on the available evidence, the combination of hydroxychloroquine
and azithromycin has shown promise in reducing the need for mechanical
ventilation and improving the outcomes for hospitalized patients. Clinical
studies have indicated that patients receiving this treatment experienced
a significant decrease in respiratory distress and were more likely to be
discharged, suggesting a positive impact on recovery. The observed im-
provements in intensive care unit admission rates and overall mortality
further support the potential benefits of this therapeutic approach. While
some studies may present conflicting results, the data from various clinical
trials suggest that hydroxychloroquine, when combined with azithromycin,
can be an effective treatment option for patients, leading to better clinical
outcomes and reduced burden on healthcare systems.
Figure 7: Gemini-2.0 outputs based on regular and en-
hanced prompts without RAG.
Under enhanced prompts (Table 3), LLM-only
results again peak semantically at Statistical Rigor
in Dim 3(+) and lexically at Research Ethics in Dim
2(+). RAG still provides maximum scores for Sta-
tistical Rigor in Dim 3(+) with strong second-tier
similarities for Comparative Treatment Analysis
in Dim 2(-) and Disputed Treatments in Dim 1(+).
On the other hand, the lexical maximum moves
to Dissemination of Science in Dim 3(-) with Re-
search Ethics in Dim 2(+) close behind. Across
both representations and prompts, the stable con-
clusion is that RAG raises similarity scores for
every pole. Additionally, leading poles differ by
representation, with semantic peaks emphasizing
rigor- and content-focused discourses (e.g., Dim 3
(+)) and lexical peaks emphasizing discourses with
more standardized institutional phrasing (e.g., Dim
2 (+)). This divergence may reflect the fact that
semantic metrics reward conceptual alignment and
paraphrase, while lexical metrics privilege reuse of
10

typical vocabulary.
Figure 7 shows that, under the regular prompt,
Gemini rejects hydroxychloroquine–azithromycin
on trial evidence and institutional guidance, align-
ing with Research Ethics in Dim 2(+) and secon-
darily Broad Focus in Dim 1(-) while distancing
from Disputed Treatments in Dim 1(+); under the
enhanced prompt, it pivots to a pro -efficacy stance
consistent with Disputed Treatments in Dim 1(+)
and frames claims with Statistical Rigor in Dim
3(+), mirroring Tables 2–3 and suggesting RAG
would further stabilize this alignment by grounding
claims in retrieved evidence. This analysis exam-
ple shows how prompt design and retrieval jointly
influence the similarity scores and framing of gen-
erated answers.
11
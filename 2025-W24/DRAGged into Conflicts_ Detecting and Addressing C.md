# DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs

**Authors**: Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu

**Published**: 2025-06-10 06:52:57

**PDF URL**: [http://arxiv.org/pdf/2506.08500v1](http://arxiv.org/pdf/2506.08500v1)

## Abstract
Retrieval Augmented Generation (RAG) is a commonly used approach for
enhancing large language models (LLMs) with relevant and up-to-date
information. However, the retrieved sources can often contain conflicting
information and it remains unclear how models should address such
discrepancies. In this work, we first propose a novel taxonomy of knowledge
conflict types in RAG, along with the desired model behavior for each type. We
then introduce CONFLICTS, a high-quality benchmark with expert annotations of
conflict types in a realistic RAG setting. CONFLICTS is the first benchmark
that enables tracking progress on how models address a wide range of knowledge
conflicts. We conduct extensive experiments on this benchmark, showing that
LLMs often struggle to appropriately resolve conflicts between sources. While
prompting LLMs to explicitly reason about the potential conflict in the
retrieved documents significantly improves the quality and appropriateness of
their responses, substantial room for improvement in future research remains.

## Full Text


<!-- PDF content starts -->

arXiv:2506.08500v1  [cs.CL]  10 Jun 2025DRAGged into C ONFLICTS :
Detecting and Addressing Conflicting Sources in Search-Augmented LLMs
Arie Cattan♠3Alon Jacovi♠Ori Ram♠Jonathan Herzig♠Roee Aharoni♠
Sasha Goldshtein♠Eran Ofek♠Idan Szpektor♠Avi Caciularu♠
♠Google Research3Bar-Ilan University
cattana@google.com
Abstract
Retrieval Augmented Generation (RAG) is
a commonly used approach for enhancing
large language models (LLMs) with relevant
and up-to-date information. However, the re-
trieved sources can often contain conflicting
information and it remains unclear how mod-
els should address such discrepancies. In this
work, we first propose a novel taxonomy of
knowledge conflict types in RAG, along with
the desired model behavior for each type. We
then introduce CONFLICTS , a high-quality
benchmark with expert annotations of con-
flict types in a realistic RAG setting. CON-
FLICTS is the first benchmark that enables
tracking progress on how models address a
wide range of knowledge conflicts. We con-
duct extensive experiments on this bench-
mark, showing that LLMs often struggle
to appropriately resolve conflicts between
sources. While prompting LLMs to explic-
itly reason about the potential conflict in the
retrieved documents significantly improves
the quality and appropriateness of their re-
sponses, substantial room for improvement
in future research remains.1
1 Introduction
Retrieval Augmented Generation (RAG) (Lewis
et al., 2020; Guu et al., 2020) has emerged as
an increasingly popular approach for improving
the factual accuracy and relevance of large lan-
guage model (LLM) outputs by leveraging external
sources retrieved at inference time (Li et al., 2024).
State-of-the-art commercial LLMs such as Chat-
GPT, Gemini and Claude have largely adopted this
paradigm and developed “search” modes that re-
trieve up-to-date web content.
However, retrieved sources may provide conflict-
ing information about the query and how models
1Our dataset can be found at: https://github.com/
google-research-datasets/rag_conflicts
Figure 1: Examples of user queries with knowl-
edge conflicts in the search results. The left panel
illustrates a freshness conflict arising from outdated
information, while the right panel shows conflict-
ing opinions on a controversial query. Identifying
the conflict type is essential for generating the ap-
propriate response (e.g., prioritizing recent data or
presenting diverse perspectives).
should optimally address these discrepancies con-
tinues to be an open and pressing research chal-
lenge (Xu et al., 2024a; Hou et al., 2024; Liu
et al., 2024b). Prior research on this topic has
typically focused on a specific type of conflict and
proposed dedicated prompting strategies to address
the conflict. For instance, Xu et al. (2024a); Hou
et al. (2024) focused on debatable questions and
prompted the model to explicitly highlight the con-
flict, Liu et al. (2024b) assumed there is a single
correct answer, and Liska et al. (2022); Vu et al.
(2024) focused on extracting up-to-date informa-

tion. However, assuming that the the type of con-
flict is known in advance is often unrealistic in
practice, and applying a specific method to differ-
ent kinds of conflicts is sub-optimal.
In this work, we argue that knowledge conflicts
arise for diverse reasons and should be addressed
differently depending on the type of conflict. Fig-
ure 1 shows a few examples of queries with knowl-
edge conflicts in their search results. Some dis-
crepancies stem from temporal shifts (e.g., “How
many people have been on the International Space
Station?” ), where models should prioritize recent
sources. Others reflect subjective opinions (e.g.,
“Are humans fundamentally good or evil?” ), where
models should neutrally present the diversity of all
the authoritative perspectives in the sources.
Inspired by these observations, we propose a
comprehensive framework for handling knowledge
conflicts. We introduce a novel taxonomy of con-
flict categories (§2; Table 1) and, for each category,
define an expected model behavior — a desired
response style designed to emulate how a human
would typically address that type of conflict.
To foster research on knowledge conflict resolu-
tion, we introduce CONFLICTS (§3), the first evalu-
ation benchmark with explicit annotations of con-
flict types. To construct CONFLICTS , we collected
a total of 458 queries from various datasets known
to contain diverse knowledge conflict types (Vu
et al., 2024; Liu et al., 2024b; Zhang and Choi,
2021; Wan et al., 2024). For each query, we then
retrieved a set of relevant source documents using
Google Search. Finally, we tasked expert annota-
tors with identifying whether the retrieved sources
contained conflicting information and determining
the conflict type according to our taxonomy. This
annotation task is challenging because annotators
need to comprehend a large volume of text to de-
termine the appropriate conflict type. To ensure
high-quality, we employ a three-stage annotation
strategy: two expert annotators independently iden-
tify conflicts and their types, resolve disagreements
through discussion, and a third expert annotator
performs a final review. CONFLICTS is the first
benchmark to enable investigation into how well
can models predict the conflict type and evaluate
whether their outputs align with how humans would
typically address that specific type of conflict.
Finally, we conduct extensive experiments on
CONFLICTS using both open-source and commer-
cial LLMs (§5). Our results demonstrate that, whileLLMs can generally produce accurate and factu-
ally grounded responses, they often fail to align
with the expected type-specific behavior. Further-
more, we show that prompting models to explicitly
reason about the potential conflict type substan-
tially improves their performance. Despite these
improvements, there remains a substantial room for
improvement in future work.
Altogether, our work is the first to propose a
general approach for addressing a wide range of
knowledge conflicts. We believe that our taxonomy
and dataset will serve as a valuable resource for
evaluating and improving future RAG models.
2 Taxonomy of Knowledge Conflicts
We posit that the type of knowledge conflict is im-
portant because different types induce different de-
sired model behaviors. Consider, for example, the
controversial query, “Is unlimited vacation time
beneficial for employees?” (Wan et al., 2024). Re-
trieved sources may present diverse viewpoints,
including arguments for and against unlimited va-
cation time, as well as nuanced analyses of its pros
and cons. In this scenario, a human would typically
synthesize these arguments, presenting a balanced
summary of the various perspectives. In contrast,
some conflicts arise from temporal discrepancies–
when correct answers evolve over time. For in-
stance, when faced with a query such as, “How
many exoplanets have been discovered?” , the re-
trieval process might yield a mix of up-to-date arti-
cles along with outdated ones, inevitably leading to
numerical discrepancies. In this instance, human
judgment would prioritize the most recent informa-
tion.
To address these differences, we introduce a tax-
onomy of conflict categories, denoted as T. For
each category t, we propose a corresponding ex-
pected behavior s=f(t), which defines the re-
sponse style that approximates how humans might
typically address queries from that type. Below, we
define the different types in our taxonomy along
with the associated expected behavior. Table 1
presents examples for each category. Table 3 illus-
trates the expected behavior of LLM outputs for
each category.
No conflict The retrieved documents provide an-
swers to the query that are equivalent or nearly
equivalent, referring to the same real-world entity,
fact, or concept. Minor variations, such as differ-
ences in surface presentation or level of granularity

Category Query Retrieved Sources
No conflict When did the Titanic set
sail?[1]On Wednesday 10th April 1912 shortly after 12noon , RMS
Titanic set sail from Southampton’s White Star Dock on her
maiden voyage to New York.
[2] On April 1912 , the Titanic set sail on its maiden voyage,
traveling from Southampton, England, to New York City.
[3] On 11th April 1912 at 11.30am RMS Titanic dropped anchor
in Queenstown, Ireland at Roches Point outer anchorage.
Complementary Informa-
tionIs public transportation
faster than driving in
cities?[1] Many areas have lanes dedicated to buses or high occupancy
vehicles, which might make taking a bus faster than driving
yourself .
[2] Even with worsening traffic, driving still gets people to work
faster , twice as fast in the U.S., a study by Governing found.
Conflicting opinions or re-
search outcomesIs fasting beneficial for in-
dividuals with diabetes?[1]Intermittent fasting , when undertaken for health reasons in
patients with diabetes mellitus, both types 1 and 2, has been
shown in a few small human studies to induce weight loss and
reduce insulin requirements.
[2] This diet is not recommended for those individuals with
diabetes , children, the underweight, or with eating disorders,
pregnant, or with chronic illnesses
[3] Current evidence suggests that intermittent fasting is an
effective non-medicinal treatment option for type 2 diabetes.
Conflict due to outdated
informationHow many countries have
recognized same-sex mar-
riage?[1] There are currently 37 countries where same-sex marriage is
legal: Andorra ...
[2] Same-sex marriage is legal in only 38 countries.
[3] By 2022, same-sex marriage was legal in 32 countries. Since
then, 3 more countries have joined this group: Andorra, Estonia,
and Greece — bringing the total to 35 .
Misinformation When did season 5 of
prison break come out?[1] The season premiered on April 4, 2017 , and concluded on
May 30, 2017, consisting of 9 episodes.
[2] Season 5 of the series was released on May 30, 2017 . It was
filmed primarily in Dallas, Panama City and Los Angeles
Table 1: Examples from CONFLICTS demonstrating each category of the knowledge conflict taxonomy.
Each instance shows a query and a selection of corresponding retrieved passages.
(e.g., “Wednesday 10th April 1912” vs. “April
1912”) do not constitute a conflict. Additionally,
retrieved documents that are topically related to the
query but do not directly answer it, are not consid-
ered sources of conflict. For example, given the
query “ When did the Titanic set sail? ” in Table 1,
the search result [3] mentioning the drop anchor
in Queenstown on 11th April is related but does
not answer the query. The expected behavior for
this category is to provide a clear and direct an-
swer without introducing alternative viewpoints or
uncertainty (Yona et al., 2024).
Complementary Information The retrieved
documents provide answers to the query that refer
todifferent real-world concepts but are mutually
compatible, meaning that a single person can rea-
sonably agree with all of them simultaneously. This
can happen when there are multiple correct answersto the same query (e.g, “ What is the meaning of
CI in police? ” has two valid answers: Confidential
Informant or Circle Inspector) or when the query is
underspecified and can be answered with multiple
complementary perspectives (e.g., the question “ Is
public transportation faster than driving in cities? ”
which depends on the city). The expected behavior
for this category is to consolidate and reconcile
the different partial answers provided by the re-
trieved documents, without framing the response
as a debate (Stelmakh et al., 2022).2
Conflicting opinions or research outcomes The
retrieved documents provide answers to the query
that are notmutually compatible. This category
2A few works address underspecified queries by generating
clarifying questions (Aliannejadi et al., 2019; Zhang and Choi,
2025). However, in this work, we aim to propose a general
approach for addressing the diverse range of conflicts and
therefore opt for generating a consolidated response.

includes subjective queries that elicit conflicting
opinions (e.g., “Are humans fundamentally good
or evil?” ), queries with contradictory research find-
ings (e.g., “Is paracetamol more effective than
placebo?” ), lack of historical consensus, etc.,
where the retrieved sources disagree and argue to-
wards a specific side.3The expected behavior for
this category is to explicitly reflect the debate be-
tween the retrieved sources and to neutrally sum-
marize the different viewpoints (Hou et al., 2024;
Slobodkin et al., 2024b; Xu et al., 2024a).
Conflict Due to Outdated Information4The
retrieved documents provide answers to the query
that are notmutually compatible, but the conflict
stems from temporal discrepancies—some sources
reflect outdated information, while others provide
more recent updates. For example, given the query
“How many countries have recognized same-sex
marriage? ” in Table 1, the retrieved sources report
37, 38, or 35, depending on their publication dates.
The expected behavior is to prioritize the up-to-
date information (Vu et al., 2024), while optionally
acknowledging the presence of outdated sources.
Conflict Due to Misinformation The retrieved
documents provide answers that are notmutually
compatible, where at least one source contains in-
formation that is likely false, misleading, or inaccu-
rate. For example, in response to the query “When
did season 5 of prison break come out?” (Table 1),
one source inaccurately states “May 30, 2017” in-
stead of “April 4, 2017” . The expected behavior
for this category is to disregard inaccurate sources
and to provide a response grounded in reliable and
verified information (Pan et al., 2023; Jiayang et al.,
2024; Ming et al., 2025).
3 C ONFLICTS
This section introduces CONFLICTS , the first
dataset annotated with knowledge conflict types.
Each instance within CONFLICTS comprises a
query, a set of retrieved relevant passages, a corre-
sponding conflict type label, and, for specific types
(detailed below), the ground truth correct answer.
Queries and retrieved documents To ensure
coverage across different types of knowledge con-
flicts, we curated seed queries from several ex-
3We focus exclusively on safe queries and leave the identi-
fication of harmful or hateful content for future work.
4Hereafter, we use the term “Freshness” to refer to this
conflict type for brevity.isting datasets. We first include queries from
FreshQA (Vu et al., 2024), which focuses on ques-
tions requiring fast-changing world knowledge; Sit-
uatedQA Temp and SituatedQA Geo (Zhang and
Choi, 2021) that include underspecified queries
whose answers depend on temporal and geo-
graphic context, respectively; and QACC (Liu et al.,
2024b), which provides unambiguous queries. For
these datasets, we use Google Search to retrieve the
top-10 search results for each query. This process
yields document titles, short snippets, and where
available, publication dates. We also consider the
recent ConflictingQA dataset (Wan et al., 2024),
where each instance consists of a Yes/No query
paired with supporting evidence for both sides,
originally retrieved using Google Search.
We find that the automatically generated Google
Search snippets often omit crucial context for re-
solving knowledge conflicts and generating appro-
priate answers. For example, the query “When
did Toyota first come?” yields different snippets
that mention seemingly contradictory dates (i.e.,
1933 ,1937 ,1955 , etc.). However, analyzing the
full articles reveals that these discrepancies stem
from references to distinct aspects of Toyota’s his-
tory (e.g., different company divisions, initial car
releases, etc.). Therefore, we parse the complete
text from the HTML pages using cloudscraper5and
jusText6. Following (Wan et al., 2024), we then
extract the most relevant 512-token segments from
each document by applying the TAS-B model (Hof-
stätter et al., 2021) across overlapping 512-token
windows with a 256-token stride, and calculate the
dot product between the window’s embedding and
the query’s embedding.
Annotation Process We cannot automatically an-
notate the conflict type based solely on the data
sources (e.g., queries from FreshQA defaulting to
Frehsness) because FreshQA, QACC and Situat-
edQA are only a collection of queries without rele-
vant documents. These queries are susceptible to
lead to knowledge conflicts between some sources,
but do not necessarily guarantee conflicts among
the top-10 search results. Although ConflictingQA
includes inter-context annotation of knowledge con-
flict for Yes/No questions, it does not differentiate
between the nuanced types of knowledge conflict
of our taxonomy (e.g., Complementary information
vs.conflicting opinions ).
5https://github.com/VeNoMouS/cloudscraper
6https://github.com/miso-belica/jusText

Therefore, we turn to human annotation by pre-
senting annotators with the queries and their cor-
responding webpage segments, instructing them
to label the conflict type based on our taxonomy
(Section 2). During the annotation, we add two ad-
ditional categories: “Other” to capture any conflict
that does not correspond to one of the defined cate-
gories in our taxonomy and “No relevant sources”
if Google Search yielded no relevant results. In
practice, no instances were annotated as “Other”,
suggesting that our taxonomy is comprehensive
and captures well the different conflict types. In
addition to the conflict type, annotators were asked
to provide brief, unstructured explanations for their
decisions. In cases of “No conflict”, “Conflict due
to outdated information” and “Misinformation”, an-
notators were also required to write the correct and
up-to-date answer from the search results.
This annotation task poses a considerable chal-
lenge, as it requires annotators to carefully examine
all search results to accurately identify the type of
knowledge conflict in each instance and, when ap-
plicable, select the correct answer. Moreover, the
boundaries between some categories are often sub-
tle and require a deep understanding of the search
results. For example, identifying whether different
search results complement each other or present
conflicting viewpoints requires annotators to assess
whether the same person could plausibly agree with
all points of view. This distinction is important as it
affects the expected style of the response: comple-
mentary information calls for cohesive aggregation,
while conflicting viewpoints require a neutral sum-
mary highlighting disagreements (as discussed in
Section 2). Additionally, annotators must disregard
search results that are irrelevant to the query.
To ensure high-quality annotations, each in-
stance was annotated by two independent anno-
tators with linguistic expertise, followed by a rec-
onciliation phase to resolve discrepancies and a
review step performed by a third expert annotator.
To facilitate the annotation process, we automati-
cally generate a response for each separate search
result using Gemini Pro 1.5 (Gemini, 2024) and
present them to the annotators. These responses
are then used for easily identifying whether the
search results provide equivalent or different infor-
mation about the query without reading all search
results. For example, given the query “What was
the religion in Persia before Islam?” , Gemini gen-
erates the same answer (“ Zoroastrianism ”) for eachConflict Type # Instances
No conflict 161
Complementary information 115
Conflicting opinions 115
Outdated Information 62
Misinformation 5
Total 458
Table 2: Statistics of C ONFLICTS .
separate source. However, the annotators were ex-
plicitly instructed to treat those model-generated
responses as optional hints and to disregard them
if they appear to be hallucinations.
Statistics The final CONFLICTS dataset contains
458 instances after filtering out 18 instances with
No relevant sources . There are an average of 9.2
search results for each query. Table 2 presents the
distribution of conflict types in C ONFLICTS .
65% of the instances in CONFLICTS (297 of 458)
were flagged as conflicting ( Complementary Infor-
mation ,Conflicting opinions ,Freshness , and Mis-
information ) and thus require reasoning and aggre-
gating spread information across multiple sources
in order to produce an appropriate response. An-
notators identified only 5 cases of Misinformation ,
likely because genuine misinformation is rare “in
the wild”, especially among the top-10 search re-
sults, where modern search engines are optimized
to demote or blocklist low-quality and misleading
content. Indeed, most previous work on misinfor-
mation has automatically perturbed text to simulate
misinformation (Du et al., 2022; Pan et al., 2023;
Jiayang et al., 2024; Ming et al., 2025).
The remaining 161 instances are those anno-
tated as “No conflict”. Although lacking explicit
knowledge conflicts, these instances present other
common RAG challenges such as handling long
context, dealing with irrelevant search results and
resolving ambiguity. For example, given the query
“Where do peaches come from?” , some search re-
sults properly mention that Peaches originated in
China, while other sources state personal opinions
about where to find good peaches nowadays “the
best peaches I have ever put in my mouth come
from Goldbud Farms in Placerville, in the middle
of California GoldRush country” . In such cases,
models should prioritize the search results identi-
fying the origin of peaches and disregard the latter,
which do not answer the query.
It is important to note that the conflict type dis-

tribution in CONFLICTS is a result of our curated
query selection and do not necessarily reflect the
natural distribution of conflicts in search engines.
4 Tasks
The taxonomy (§2) and dataset (§3) enables the
investigation of new research questions on RAG
with knowledge conflicts. We can explore whether
models can accurately predict the type of conflict,
assess how well model outputs align with the ex-
pected response behavior, and investigate whether
explicitly leveraging the conflict type can improve
response quality. To address these questions, we
formalize two core tasks:
Task 1 (Classification) : Conflict type prediction.
Given a query qand the retrieved relevant para-
graphs Cq, the goal is to predict the category of
the knowledge conflict ˆtfrom the taxonomy be-
tween the retrieved documents Cqwith respect to
the query q, as follows:
ˆt∼pθ(t|[q;Cq]) (1)
This auxiliary task can guide downstream re-
sponse generation (§5) and enable the evaluation
of models’ understanding of the underlying rela-
tionships between the different sources.
Task 2 (Generation) : Generating an appropri-
ate response. Given a query qand the retrieved
relevant paragraphs Cq, the goal is to generate a
grounded and accurate response ˆythat conforms to
the expected behavior:
ˆy∼pθ(y|[q;Cq]) (2)
Evaluation. We conduct a multi-faceted evalua-
tion to assess the quality of the generated response
ˆy, including factual grounding, accuracy (where ap-
plicable), and adherence to the expected behavior.
First, following common practices in grounded
generation (Jacovi et al., 2025), we evaluate the
factual grounding of the response ˆywith respect
to the retrieved search results. Specifically, since
we require models to generate grounded responses
with inline citations for each sentence pointing to
the relevant search results (see Section 5.1), we
measure citation quality (Gao et al., 2023a; Slo-
bodkin et al., 2024a). We adapt the prompt from
the FACTS benchmark (Jacovi et al., 2025), ask-
ing the model to assess whether each sentence is
“supported”, “unsupported”, “contradictory” or “nofactual information”. The factual grounding score
of an entire response is the percentage of “sup-
ported” sentences over all sentences with factual
information.
Second, for instances with a single correct an-
swer, we evaluate whether the generated response
ˆycorrectly incorporates the gold answer yfrom
CONFLICTS . We refer to this evaluation as An-
swer Recall . This applies to instances in the “No
conflict”, “Freshness”, and “Misinformation” cate-
gories, which comprise 228 queries in CONFLICTS .
Following (Mallen et al., 2023; Liu et al., 2024a),
we consider a response correct if the gold answer y
is included in ˆy. We avoid strict string matching be-
cause LLM outputs are often verbose or paraphras-
tic yet correct and instead consider an LLM-based
evaluator to assess semantic inclusion of the gold
answer.
Third, and specific to our task of resolving con-
flicting information, we assess whether the gener-
ated response ˆyadheres to the expected behavior
sassociated with the conflict type t, as defined
in Section 2. Examples of model outputs, along
with their evaluation for adherence to the expected
behavior, are shown in Table 3. This evaluation
task requires assessing nuanced stylistic aspects
of the response beyond factual consistency or an-
swer inclusion and cannot be achieved with simple
heuristics. For instance, in the query “How many
people have been on the International Space Sta-
tion?” (Table 3), the first response fails to adhere
to the expected behavior by framing the different
figures as a debate between the sources. In contrast,
the second response clearly distinguishes between
older and newer answers (i.e., “Older figures”, “as
of September 12, 2021”).
As LLMs have demonstrated remarkable evalua-
tion capabilities across a range of tasks, sometimes
matching expert human raters (Chiang and Lee,
2023; Zheng et al., 2023; Liu et al., 2023; Kocmi
and Federmann, 2023; Kamalloo et al., 2023, in-
ter alia ), we employ an LLM for assessing adher-
ence to the expected behavior. This evaluator is
few-shot prompted with examples of adherent and
non-adherent responses. Specifically, we design
a separate prompt template for each conflict type,
incorporating: (i) the query, (ii) the conflict type
description and associated expected behavior, (iii)
2–3 positive and negative examples, and (iv) the
candidate response. The model outputs a binary
decision indicating whether the response adheres to

Query Conflict type Responses Comment
When did
season 5 of
prison break
come out?MisinformationSeason 5 of Prison Break premiered on April 4,
2017, on Fox [3]. The season was released on May
30, 2017 [4]./timesAlthough the model includes the correct answer
(April 7, 2020), the style of the response is incorrect
because the model tries to reconcile between the
sources, while the source [4] is incorrect.
Season 5 of Prison Break premiered on April 4,
2017 [2][3]. The first episode aired on this date [2].✓The response provides a clear and coherent an-
swer to the question.
How many
people have
been on the
International
Space Station?FreshnessAs of May 2022, 258 individuals from 20 countries
have visited the International Space Station (ISS)
[5]. Other sources state over 260 individuals from
20 countries and five International Partners have
visited the station [7]./timesThe response presents conflicting opinions about
the query, whereas source [5] is outdated.
As of March 2024, 279 individuals from 22 coun-
tries have visited the International Space Station
[2]. Older figures indicate that as of May 2022, 258
people from 20 countries had visited the station [5],
and as of September 12, 2021, 244 people had made
403 spaceflights to the ISS [3].✓The response clearly answers the question, stat-
ing there were 279 people as of March 2024, and
acknowledges older figures from other sources.
Who has the
biggest
airport?Complementary
informationThe King Fahd International Airport in Saudi Ara-
bia is the largest airport in the world by land area
[1][6][7]. It spans nearly 300 square miles, an area
comparable to the size of New York City’s five bor-
oughs combined [6]./timesThe model writes the response only with the per-
spective of the largest airport.
The world’s largest airport can be defined in several
ways, including by land area, passenger volume, or
flight volume [6]. King Fahd International Airport
in Saudi Arabia is the largest airport by total area
[1]. In terms of passenger traffic, Hartsfield-Jackson
Atlanta International Airport in the United States is
the busiest airport in the world [4, 5].✓The response presents multiple aspects of the
what is considered a “big” airport, in terms of land
area and passenger volume.
Where did the
style hot dog
originated?Conflicting
opinionsThe Chicago-style hot dog originated during the
Great Depression in the 1930s in Chicago [1]. It
was created to provide a hearty and affordable meal,
with a nickel buying a hot dog loaded with condi-
ments [1]./timesThe model only writes the origin of a specific
hot-dog (Chicago-style).
The exact origin of the hot dog is not definitively
clear, making it a subject of some debate among
historians [6]. Two European cities, Frankfurt, Ger-
many, and Vienna, Austria, both claim to be the
birthplace of the sausage that served as the precur-
sor to the hot dog [7]. The Chicago-style hot dog,
with its specific set of toppings, emerged later, dur-
ing the Great Depression in the 1930s [1].✓The response presents a debate around the origin
of hot-dog.
Table 3: Examples of model responses that adhere and do not adhere the expected behavior for each
category in our taxonomy.
the expected behavior. We validate this automatic
rater against human judgments on a subset of 100
examples from the dataset, achieving an accuracy
of 0.89, which demonstrates the reliability of our
automatic evaluator.
Together, these three evaluation metrics–factual
consistency, answer recall, and expected behavior
adherence–provide a comprehensive assessment of
response quality, capturing different critical aspects.
For instance, a response might be factually consis-
tent with the search results and even include the
gold answer but fail to adhere to the appropriate
behavior for the conflict type, e.g., by providingan additional incorrect fact grounded to one of the
search results (see the first response to the query
“When did season 5 of prison break come out?” in
Table 3). Conversely, a response could align per-
fectly with the expected behavior yet provide an
incorrect answer or contain claims not supported
by the sources.
5 Experimental Setup
5.1 Experiments
We conduct extensive experiments on CONFLICTS
to address our two core tasks (Section 4): (1) con-
flict type prediction, and (2) generation of a re-

Model Accuracy
Gemma 3 27B 53.9
Qwen 2.5 72B 53.1
GPT-4o 59.2
Gemini 2.0 Flash 60.5
Gemini 2.5 Flash 65.3
Table 4: Performance (accuracy) of models for
predicting the conflict category on C ONFLICTS .
sponse that adheres to the appropriate behavior.
For both tasks, each search result is represented
as a concatenation of its URL, page title, Google
snippet, publication date (if available), and the 512-
token segment (§3).
Conflict type prediction: We prompt the model
with the query, retrieved evidence, and our taxon-
omyT(Table 1), which includes category defini-
tions and 1–2 illustrative examples per class. The
model is then asked to classify the type of conflict
Response Generation: We explore multiple
prompting strategies for generating the response ˆy:
1.Vanilla: A standard RAG-style approach
where the model receives the query and search
results as input and generates a response:
pθ(y|[q;Cq])(Ram et al., 2023).
2.Pipeline : A two-step process. First, the
model predicts the conflict type ˆtgiven the
taxonomy T, the query and its search results:
pθ(t|[T;q;Cq]). Second, the model generates
the response ˆyusing the predicted conflict
typeˆtas additional context: pθ(y|[q;Cq;ˆt]).
3.Taxonomy Aware: A joint approach where
the model simultaneously predicts the conflict
type and generates the response in a single
pass. The prompt includes the full taxonomy
Talong with the query and search results:
pθ(t, y|[T;q;Cq]).
4.Oracle: An upper-bound setting in which the
model is given the gold conflict type t∗from
the dataset in addition to the query and search
results: pθ(y|[q;Cq;t∗]).
For all prompts, we also instruct the model to
provide in-line citations (e.g., [2]) to the relevant
search results for each sentence (Gao et al., 2023a).5.2 Results
We present the results of the conflict type prediction
in Table 4, with Gemini 2.5 Flash achieving the
highest accuracy (65.3%).
Table 5 presents the results of the response gener-
ation quality. For each model and prompt template,
we report the accuracy of the expected behavior, an-
swer recall and factual consistency, using Gemini
2.5 Flash (§4).
Result 1: Model responses exhibit limited adher-
ence to the expected behavior. Table 5 shows
that the standard RAG prompt (Vanilla) generates
responses that moderately adhere to the expected
behavior, with scores ranging from 59.4 for the
open-source Gemma 3 27B to 68.3 for Gemini 2.5
Flash with Thinking mode. These relatively low
scores reveal that, while models can be grounded
on the search results and include the correct an-
swer, they sometimes fail to follow the expected
behavior. This highlights the importance of evalu-
ating not only the factual accuracy and grounding
of model outputs, but also whether the style of the
response aligns with human preferences.
Result 2: Explicitly incorporating conflict
type improves expected behavior. The Oracle
prompt, which augments the LLM input with the
gold conflict type from CONFLICTS , substantially
improves adherence to the expected response be-
havior across all models. On average, it yields a 24-
point gain over the Vanilla prompt (e.g., +28.2 for
Gemma, +21.4 for GPT-4o, +21.0 for Gemini 2.5
Flash with Thinking mode, etc.), while preserving
high answer recall and factual consistency scores.
These results indicate that models have the general
capability to generate appropriate responses and
there is considerable room in developing methods
that can approximate this upper bound.
Result 3: Pipeline and Taxonomy-aware
prompts improve the expected behavior. Ta-
ble 5 shows that both the pipeline and the
taxonomy-aware prompts improve adherence to
the expected behavior over the vanilla approach,
without degrading the answer recall and factual
grounding scores. On average, they yield perfor-
mance gains of 9 and 5.5 points, respectively. This
suggests that prompting models to reason explicitly
on the potential knowledge conflict in the search
results, can substantially improve response quality.
Nonetheless, search augmented LLMs must ad-
dress a range of additional challenges beyond

Model Prompt Expected Behavior Answer Recall Factual Grounding
Gemma 3 27B (Kamath et al., 2025)Vanilla 59.4 89.0 94.4
Pipeline 71.6 89.9 91.9
Taxonomy-Aware 76.2 86.0 92.9
Oracle 87.6 88.6 93.3
Qwen 2.5 72B (Yang et al., 2024)Vanilla 64.2 87.7 90.5
Pipeline 67.5 89.0 88.2
Taxonomy-Aware 68.3 88.2 89.4
Oracle 90.6 87.3 89.3
GPT-4o (Hurst et al., 2024)Vanilla 67.2 87.3 91.2
Pipeline 74.9 87.3 92.0
Taxonomy-Aware 70.0 88.6 94.0
Oracle 88.6 87.3 91.4
Gemini 2.0 Flash (Google, 2024)Vanilla 62.0 90.8 98.5
Pipeline 75.3 89.5 96.6
Taxonomy-Aware 69.4 89.0 96.8
Oracle 87.8 90.4 96.8
Gemini 2.5 Flash (Google, 2025)Vanilla 65.7 92.1 96.8
Pipeline 75.1 90.8 96.9
Taxonomy-Aware 69.9 94.3 96.2
Oracle 88.0 91.7 96.8
Gemini 2.5 Flash ThinkingVanilla 68.3 90.8 96.2
Pipeline 76.6 89.0 96.1
Taxonomy-Aware 73.6 91.7 97.0
Oracle 89.3 89.9 96.8
Table 5: Performance of response quality. For each model and prompt strategy, we report the expected
behavior accuracy, answer recall and factual grounding (§4).
knowledge conflicts, including handling irrelevant
sources due to an imperfect retrieval (Yoran et al.,
2024), determining when to refine the user query
and to search for additional evidence, or address-
ing queries with safety concerns. Therefore, future
work can explore how to integrate such conflict-
aware methods into practical RAG systems.
5.3 Analysis
Expected behavior per category Table 6
presents the expected behavior evaluation for each
category in our taxonomy (§2). For brevity, we
report the performance of Gemini 2.5 Flash with
thinking mode, other models showing similar
trends. The most challenging category is “Conflict-
ing opinions” with Gemini achieving only 36.2%
under the Vanilla prompt. The pipeline approach
improves the expected behavior for Complemen-
tary information ,Freshness and Misinformation
andConflicting opinions , with a slight drop in No
conflict .
Error Analysis To better understand the head-
room in conforming to the expected behavior,
we manually analyze 40 randomly sampled out-puts that do not adhere to the expected behavior
from our best-performing model (Gemini 2.5 Flash
with thinking). For the Complementary informa-
tioncategory, the most common error was under-
specification: model response often include only
one correct answer, failing to capture the full range
of relevant results (see Table 3). For Conflicting
opinions , models either present only a single view-
point or multiple viewpoints with a strong bias
toward one perspective. In the No conflict and
Freshness categories, models frequently hedge by
expressing uncertainty or mentioning multiple pos-
sible answers.
6 Related Work
6.1 Retrieval Augmented Generation
Retrieval Augmented Generation (RAG) consists
of conditioning a model on relevant documents
from a large corpus during generation (Guu et al.,
2020; Lewis et al., 2020; Izacard et al., 2023;
Borgeaud et al., 2021; Ram et al., 2023; Gao et al.,
2023b). Retrieved documents can bring conflicting
information, which can complicate the generation
process. Previous work on knowledge conflicts

Category Vanilla Pipeline Taxonomy-Aware Oracle
No conflict 78.4 74.7 71.0 90.1
Complementary Information 83.3 83.3 78.1 94.7
Freshness and Misinformation 74.2 75.8 78.8 80.3
Conflicting opinions 36.2 73.3 69.8 87.9
Table 6: Expected behavior accuracy of Gemini Flash 2.5 Thinking per category. We combine Freshness
and Misinformation because both categories require selecting the correct response.
has mostly focused on a single type of conflict,
such as conflicts arising from outdated informa-
tion (Liska et al., 2022; Kasai et al., 2023; Vu et al.,
2024), controversial queries with disputed opin-
ions (Wan et al., 2024; Xu et al., 2024a), ambiguous
queries (Min et al., 2020; Zhang and Choi, 2021;
Lee et al., 2024), or factual contradictions observed
in real-world scenarios (Hou et al., 2024) or in
synthetically created datasets (Wang et al., 2024;
Jiayang et al., 2024; Tan et al., 2024). A related
line of research explores context-memory conflicts,
where retrieved documents contradict the model’s
parametric knowledge (Longpre et al., 2021; Kor-
tukov et al., 2024). For a broader survey of knowl-
edge conflicts in LLMs, we refer the reader to (Xu
et al., 2024b). This work focuses on inter-context
knowledge conflicts and proposes a comprehen-
sive taxonomy of conflict types for appropriately
addressing the diverse range of conflicts.
6.2 Datasets with Knowledge Conflicts
In recent years, several QA datasets with knowl-
edge conflicts between different sources were in-
troduced. For example, ConflictingQA (Wan et al.,
2024) generates controversial queries with LLMs
and automatically identifies the stance (Yes or No)
of each search result. Similarly, DebateQA (Xu
et al., 2024a) collects many debatable questions
from various sources and automatically generate
points of views that address the query from differ-
ent perspectives. WikiContradict (Hou et al., 2024)
leverages Wikipedia tags to identify contradictions
and ask human annotators to write questions that
reflect the contradiction between two paragraphs.
QACC (Liu et al., 2025) is a subset of unambiguous
queries from AmbigQA (Min et al., 2020), where
human annotators were asked to write the different
answers from the search snippets and to select the
correct answer. AmbigDocs (Lee et al., 2024) fo-
cuses on ambiguous queries to evaluate the ability
of models to distinguish between different entities
sharing the same name. RamDocs (Wang et al.,2025) extends AmbigDocs by introducing auto-
matically generated noisy examples to challenge
models with misinformation.
In contrast to the existing resources, CONFLICTS
is the first RAG dataset to include human anno-
tation of the category of the knowledge conflict,
based on our proposed taxonomy (§2). As shown in
our experiments (§5), this information is valuable
for generating an appropriate response. Further-
more, CONFLICTS includes a diverse set of queries
and the relevant documents constitute a real-world
scenario of RAG where LLMs are augmented with
search results. Therefore, CONFLICTS can serve as
a broad evaluation benchmark to assess how mod-
els handle a wide spectrum of knowledge conflict
in RAG scenarios.
6.3 LLM Evaluation
The evaluation of LLMs has become a subject of
intense research interest, assessing various aspects
of their outputs, including factuality with respect
to world knowledge or to a given context (Rashkin
et al., 2023; Min et al., 2023; Tang et al., 2024;
Song et al., 2024; Cattan et al., 2024; Ravichan-
der et al., 2025; Jacovi et al., 2025, inter-alia ),
instruction-following (Skopek et al., 2023; Liu
et al., 2024c), coherence (Gómez-Rodríguez and
Williams, 2023), inter-alia . We expand the evalua-
tion scope and introduce an evaluation methodol-
ogy that assesses not only whether LLMs resolve
knowledge conflicts in RAG, but how they do so,
in alignment with human expectations.
7 Conclusion
This work highlights the critical role of the conflict
type in Retrieval Augmented Generation (RAG).
We hope CONFLICTS will serve as a valuable re-
source for developing more robust RAG models.
Beyond response generation, future work can ex-
plore how to leverage conflict type for other ap-
plications, such as enhancing the reasoning and
decision-making capabilities of agentic LLMs.

Acknowledgments
We thank Aviv Slobokdin for reviewing the paper
draft and providing valuable feedback. We are
also grateful to Gabriel Stanovsky, Roy Schwartz,
Tu Vu, Adam Bloniarz, Corey Fry and Avigail
Dabush for fruitful discussion at various stages
of the project. We thank Siyi Liu for sharing the
QACC dataset. We thank Michael Riley, Itay Laish
and Dave Orr for reviewing the paper. Special
thanks to Rebecca Galor for managing the annota-
tion tasks, onboarding the annotators and providing
them feedback along the process. Finally, we are
grateful to all annotators that participated in the
construction of C ONFLICTS .
References
Mohammad Aliannejadi, Hamed Zamani, Fabio A.
Crestani, and W. Bruce Croft. 2019. Asking clar-
ifying questions in open-domain information-
seeking conversations. Proceedings of the 42nd
International ACM SIGIR Conference on Re-
search and Development in Information Re-
trieval .
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Mil-
lican, George van den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, Diego
de Las Casas, Aurelia Guy, Jacob Menick, Ro-
man Ring, T. W. Hennigan, Saffron Huang,
Lorenzo Maggiore, Chris Jones, Albin Cassirer,
Andy Brock, Michela Paganini, Geoffrey Irv-
ing, Oriol Vinyals, Simon Osindero, Karen Si-
monyan, Jack W. Rae, Erich Elsen, and L. Sifre.
2021. Improving language models by retriev-
ing from trillions of tokens. In International
Conference on Machine Learning .
Arie Cattan, Paul Roit, Shiyue Zhang, David Wan,
Roee Aharoni, Idan Szpektor, Mohit Bansal, and
Ido Dagan. 2024. Localizing factual inconsis-
tencies in attributable text generation. ArXiv ,
abs/2410.07473.
Cheng-Han Chiang and Hung-yi Lee. 2023. Can
large language models be an alternative to hu-
man evaluations? In Proceedings of the 61st
Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) ,
pages 15607–15631, Toronto, Canada. Associa-
tion for Computational Linguistics.Y . Du, Antoine Bosselut, and Christopher D. Man-
ning. 2022. Synthetic disinformation attacks on
automated fact verification systems. In AAAI
Conference on Artificial Intelligence .
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi
Chen. 2023a. Enabling large language models
to generate text with citations. In Proceedings
of the 2023 Conference on Empirical Methods
in Natural Language Processing , pages 6465–
6488, Singapore. Association for Computational
Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxi-
ang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Qianyu Guo, Meng Wang, and Haofen
Wang. 2023b. Retrieval-augmented generation
for large language models: A survey. ArXiv ,
abs/2312.10997.
Team Gemini. 2024. Gemini 1.5: Unlocking multi-
modal understanding across millions of tokens
of context.
Carlos Gómez-Rodríguez and Paul Williams. 2023.
A confederacy of models: a comprehensive eval-
uation of LLMs on creative writing. In Findings
of the Association for Computational Linguistics:
EMNLP 2023 , pages 14504–14528, Singapore.
Association for Computational Linguistics.
Google. 2024. Gemini 2.0.
Google. 2025. Gemini 2.5 Thinking.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong
Pasupat, and Ming-Wei Chang. 2020. Realm:
retrieval-augmented language model pre-
training. In Proceedings of the 37th Inter-
national Conference on Machine Learning ,
ICML’20. JMLR.org.
Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-
Hong Yang, Jimmy Lin, and Allan Hanbury.
2021. Efficiently teaching an effective dense
retriever with balanced topic aware sampling.
InProceedings of the 44th International ACM
SIGIR Conference on Research and Develop-
ment in Information Retrieval , SIGIR ’21, page
113–122, New York, NY , USA. Association for
Computing Machinery.
Yufang Hou, Alessandra Pascale, Javier Carnerero-
Cano, Tigran T. Tchrakian, Radu Marinescu,
Elizabeth M. Daly, Inkit Padhi, and Prasanna

Sattigeri. 2024. Wikicontradict: A benchmark
for evaluating LLMs on real-world knowledge
conflicts from wikipedia. In The Thirty-eight
Conference on Neural Information Processing
Systems Datasets and Benchmarks Track .
OpenAI Aaron Hurst, Adam Lerer, Adam P.
Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan
Hayes, Alec Radford, Aleksander Mkadry, Alex
Baker-Whitcomb, Alex Beutel, Alex Borzunov,
Alex Carney, Alex Chow, Alexander Kirillov,
Alex Nichol, Alex Paino, Alex Renzin, Alexan-
dre Passos, Alexander Kirillov, Alexi Christakis,
Alexis Conneau, Ali Kamali, Allan Jabri, Al-
lison Moyer, Allison Tam, Amadou Crookes,
Amin Tootoochian, Amin Tootoonchian, Ananya
Kumar, Andrea Vallone, Andrej Karpathy, An-
drew Braunstein, Andrew Cann, Andrew Codis-
poti, Andrew Galu, Andrew Kondrich, Andrew
Tulloch, An drey Mishchenko, Angela Baek,
Angela Jiang, An toine Pelisse, Antonia Wood-
ford, Anuj Gosalia, Arka Dhar, Ashley Pantu-
liano, Avi Nayak, Avital Oliver, Barret Zoph,
B. Ghorbani, Ben Leimberger, Ben Rossen, Ben-
jamin Sokolowsky, Ben Wang, Benjamin Zweig,
Beth Hoover, Blake Samic, Bob McGrew, Bobby
Spero, Bogo Giertler, Bowen Cheng, Brad Light-
cap, Brandon Walkin, Brendan Quinn, Brian
Guarraci, Brian Hsu, Bright Kellogg, Brydon
Eastman, Camillo Lugaresi, Carroll L. Wain-
wright, Cary Bassin, Cary Hudson, Casey Chu,
Chad Nelson, Chak Li, Chan Jun Shern, Chan-
ning Conger, Charlotte Barette, Chelsea V oss,
Chen Ding, Cheng Lu, Chong Zhang, Chris
Beaumont, Chris Hallacy, Chris Koch, Christian
Gibson, Christina Kim, Christine Choi, Chris-
tine McLeavey, Chris Hesse, Claudia Fischer,
Clemens Winter, Coley Czarnecki, Colin Jarvis,
Colin Wei, Constantin Koumouzelis, Dane Sher-
burn, Daniel Kappler, Daniel Levin, Daniel Levy,
David Carr, David Farhi, David Mély, David
Robinson, David Sasaki, Denny Jin, Dev Val-
ladares, Dimitris Tsipras, Doug Li, Phong Duc
Nguyen, Duncan Findlay, Edede Oiwoh, Ed-
mund Wong, Ehsan Asdar, Elizabeth Proehl,
Elizabeth Yang, Eric Antonow, Eric Kramer,
Eric Peterson, Eric Sigler, Eric Wallace, Eu-
gene Brevdo, Evan Mays, Farzad Khorasani, Fe-
lipe Petroski Such, Filippo Raso, Francis Zhang,
Fred von Lohmann, Freddie Sulit, Gabriel Goh,
Gene Oden, Geoff Salmon, Giulio Starace, GregBrockman, Hadi Salman, Hai-Biao Bao, Hai-
tang Hu, Hannah Wong, Haoyu Wang, Heather
Schmidt, Heather Whitney, Heewoo Jun, Hen-
drik Kirchner, Henrique Pondé de Oliveira Pinto,
Hongyu Ren, Huiwen Chang, Hyung Won
Chung, Ian D. Kivlichan, Ian O’Connell, Ian
Osband, Ian Silber, Ian Sohl, ˙Ibrahim Ci-
hangir Okuyucu, Ikai Lan, Ilya Kostrikov, Ilya
Sutskever, Ingmar Kanitscheider, Ishaan Gul-
rajani, Jacob Coxon, Jacob Menick, Jakub W.
Pachocki, James Aung, James Betker, James
Crooks, James Lennon, Jamie Ryan Kiros, Jan
Leike, Jane Park, Jason Kwon, Jason Phang, Ja-
son Teplitz, Jason Wei, Jason Wolfe, Jay Chen,
Jeff Harris, Jenia Varavva, Jessica Gan Lee, Jes-
sica Shieh, Ji Lin, Jiahui Yu, Jiayi Weng, Jie
Tang, Jieqi Yu, Joanne Jang, Joaquin Quiñonero
Candela, Joe Beutler, Joe Landers, Joel Parish,
Johannes Heidecke, John Schulman, Jonathan
Lachman, Jonathan McKay, Jonathan Uesato,
Jonathan Ward, Jong Wook Kim, Joost Huizinga,
Jordan Sitkin, Jos Kraaijeveld, Joshua Gross,
Josh Kaplan, Josh Snyder, Josh Achiam, Joy
Jiao, Joyce Lee, Juntang Zhuang, Justyn Harri-
man, Kai Fricke, Kai Hayashi, Karan Singhal,
Katy Shi, Kavin Karthik, Kayla Wood, Kendra
Rimbach, Kenny Hsu, Kenny Nguyen, Keren Gu-
Lemberg, Kevin Button, Kevin Liu, Kiel Howe,
Krithika Muthukumar, Kyle Luther, Lama Ah-
mad, Larry Kai, Lauren Itow, Lauren Work-
man, Leher Pathak, Leo Chen, Li Jing, Lia
Guy, Liam Fedus, Liang Zhou, Lien Mamit-
suka, Lilian Weng, Lindsay McCallum, Lindsey
Held, Ouyang Long, Louis Feuvrier, Lu Zhang,
Lukasz Kondraciuk, Lukasz Kaiser, Luke He-
witt, Luke Metz, Lyric Doshi, Mada Aflak,
Maddie Simens, Made laine Boyd, Madeleine
Thompson, Marat Dukhan, Mark Chen, Mark
Gray, Mark Hudnall, Marvin Zhang, Marwan
Aljubeh, Ma teusz Litwin, Matthew Zeng, Max
Johnson, Maya Shetty, Mayank Gupta, Meghan
Shah, Mehmet Ali Yatbaz, Mengxue Yang,
Mengchao Zhong, Mia Glaese, Mianna Chen,
Michael Janner, Michael Lampe, Michael Petrov,
Michael Wu, Michele Wang, Michelle Fradin,
Michelle Pokrass, Miguel Castro, Miguel Cas-
tro, Mikhail Pavlov, Miles Brundage, Miles
Wang, Mina Khan, Mira Murati, Mo Bavarian,
Molly Lin, Murat Yesildal, Nacho Soto, Na-
talia Gimelshein, Na talie Cone, Natalie Stau-
dacher, Natalie Summers, Natan LaFontaine,

Neil Chowdhury, Nick Ryder, Nick Stathas,
Nick Turley, Nikolas A. Tezak, Niko Felix,
Nithanth Kudige, Nitish Shirish Keskar, Noah
Deutsch, Noel Bundick, Nora Puckett, Ofir
Nachum, Ola Okelola, Oleg Boiko, Oleg Murk,
Oliver Jaffe, Olivia Watkins, Olivier Godement,
Owen Campbell-Moore, Patrick Chao, Paul
McMillan, Pavel Belov, Peng Su, Peter Bak,
Peter Bakkum, Peter Deng, Peter Dolan, Peter
Hoeschele, Peter Welinder, Phil Tillet, Philip
Pronin, Phil Tillet, Prafulla Dhariwal, Qim ing
Yuan, Rachel Dias, Rachel Lim, Rahul Arora,
Rajan Troll, Randall Lin, Raphael Gontijo Lopes,
Raul Puri, Reah Miyara, Reimar H. Leike, Re-
naud Gaubert, Reza Zamani, Ricky Wang, Rob
Donnelly, Rob Honsby, Rocky Smith, Rohan
Sahai, Rohit Ramchandani, Romain Huet, Rory
Carmichael, Rowan Zellers, Roy Chen, Ruby
Chen, Ruslan Ramilevich Nigmatullin, Ryan
Cheu, Saachi Jain, Sam Altman, Sam Schoen-
holz, Sam Toizer, Samuel Miserendino, Sand-
hini Agarwal, Sara Culver, Scott Ethersmith,
Scott Gray, Sean Grove, Sean Metzger, Shamez
Hermani, Shantanu Jain, Shengjia Zhao, Sher-
win Wu, Shino Jomoto, Shirong Wu, Shuaiqi
Xia, Sonia Phene, Spencer Papay, Srinivas
Narayanan, Steve Coffey, Steve Lee, Stewart
Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao
Xu, Tarun Gogineni, Taya Christianson, Ted
Sanders, Tejal Patwardhan, Thomas Cunningh-
man, Thomas Degry, Thomas Dimson, Thomas
Raoux, Thomas Shadwell, Tianhao Zheng, Todd
Underwood, Todor Markov, Toki Sherbakov,
Tom Rubin, Tom Stasi, Tomer Kaftan, Tris-
tan Heywood, Troy Peterson, Tyce Walters,
Tyna Eloundou, Valerie Qi, Veit Moeller, Vinnie
Monaco, Vishal Kuo, Vlad Fomenko, Wayne
Chang, Weiyi Zheng, Wenda Zhou, Wesam
Manassra, Will Sheu, Wojciech Zaremba, Yash
Patil, Yilei Qian, Yongjik Kim, Youlong Cheng,
Yu Zhang, Yuchen He, Yuchen Zhang, Yujia Jin,
Yunxing Dai, and Yury Malkov. 2024. Gpt-4o
system card. ArXiv , abs/2410.21276.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel,
and Edouard Grave. 2023. Atlas: few-shot learn-
ing with retrieval augmented language models.
J. Mach. Learn. Res. , 24(1).
Alon Jacovi, Andrew Wang, Chris Alberti, ConnieTao, Jon Lipovetz, Kate Olszewska, Lukas Haas,
Michelle Liu, Nate Keating, Adam Bloniarz,
Carl Saroufim, Corey Fry, Doron Kukliansky,
Gaurav, Singh Tomar, James Swirhun, Jinwei
Xing, Lily Wang, Madhu Gurumurthy, Michael
Aaron, Moran Ambar, Rachana Fellinger, Rui
Wang, Zizhao Zhang, Sasha Goldshtein, Dipan-
jan Das, Equal Contribution, Google Deepmind,
Google Research, Google Cloud, and Kaggle.
2025. The facts grounding leaderboard: Bench-
marking llms’ ability to ground responses to
long-form input. ArXiv , abs/2501.03200.
Cheng Jiayang, Chunkit Chan, Qianqian Zhuang,
Lin Qiu, Tianhang Zhang, Tengxiao Liu,
Yangqiu Song, Yue Zhang, Pengfei Liu, and
Zheng Zhang. 2024. ECON: On the detection
and resolution of evidence conflicts. In Pro-
ceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pages
7816–7844, Miami, Florida, USA. Association
for Computational Linguistics.
Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and
Davood Rafiei. 2023. Evaluating open-domain
question answering in the era of large language
models. In Proceedings of the 61st Annual Meet-
ing of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 5591–
5606, Toronto, Canada. Association for Compu-
tational Linguistics.
Gemma Team Aishwarya Kamath, Johan Ferret,
Shreya Pathak, Nino Vieillard, Ramona Mer-
hej, Sarah Perrin, Tatiana Matejovicova, Alexan-
dre Ram’e, Morgane Rivière, Louis Rouil-
lard, Thomas Mesnard, Geoffrey Cideron, Jean-
Bastien Grill, Sabela Ramos, Edouard Yvinec,
Michelle Casbon, Etienne Pot, Ivo Penchev, Gael
Liu, Francesco Visin, Kathleen Kenealy, Lucas
Beyer, Xiaohai Zhai, Anton Tsitsulin, Róbert Ist-
van Busa-Fekete, Alex Feng, Noveen Sachdeva,
Benjamin Coleman, Yi Gao, Basil Mustafa,
Iain Barr, Emilio Parisotto, David Tian, Matan
Eyal, Colin Cherry, Jan-Thorsten Peter, Danila
Sinopalnikov, Surya Bhupatiraju, Rishabh Agar-
wal, Mehran Kazemi, Dan Malkin, Ravin Ku-
mar, David Vilar, Idan Brusilovsky, Jiaming
Luo, Andreas Steiner, Abe Friesen, Abhanshu
Sharma, Abheesht Sharma, Adi Mayrav Gilady,
Adrian Goedeckemeyer, Alaa Saade, Alexander
Kolesnikov, Alexei Bendebury, Alvin Abdagic,
Amit Vadi, Andr’as Gyorgy, André Susano Pinto,

Anil Das, Ankur Bapna, Antoine Miech, Antoine
Yang, Antonia Paterson, Ashish Shenoy, Ayan
Chakrabarti, Bilal Piot, Boxi Wu, Bobak Shahri-
ari, Bryce Petrini, Charlie Chen, Charline Le
Lan, Christopher A. Choquette-Choo, CJ Carey,
Cormac Brick, Daniel Deutsch, Danielle Eisen-
bud, Dee Cattle, Derek Cheng, Dimitris Paparas,
Divyashree Shivakumar Sreepathihalli, Doug
Reid, Dustin Tran, Dustin Zelle, Eric Noland,
Erwin Huizenga, Eugene Kharitonov, Frederick
Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi
Hashemi, Hanna Klimczak-Pluci’nska, Harman
Singh, Harsh Mehta, Harshal Tushar Lehri, Hus-
sein Hazimeh, Ian Ballantyne, Idan Szpektor,
Ivan Nardini, Jean Pouget-Abadie, Jetha Chan,
Joe Stanton, J. Michael Wieting, Jonathan Lai,
Jordi Orbay, Joe Fernandez, Joshua Newlan,
Junsong Ji, Jyotinder Singh, Kat Black, Kathy
Yu, Kevin Hui, Kiran V odrahalli, Klaus Gr-
eff, Linhai Qiu, Marcella Valentine, Marina
Coelho, Marvin Ritter, Matt Hoffman, Matthew
Watson, Mayank Chaturvedi, Michael Moyni-
han, Min Ma, Nabila Babar, Natasha Noy,
Nathan Byrd, Nick Roy, Nikola Momchev, Ni-
lay Chauhan, Oskar Bunyan, Pankil Botarda,
Paul Caron, Paul Kishan Rubenstein, Phil Cul-
liton, Philipp Schmid, Pier Giuseppe Sessa,
Pingmei Xu, Piotr Sta ´nczyk, Pouya Dehghani
Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan,
Reza Ardeshir Rokni, Rob Willoughby, Ro-
hith Vallu, Ryan Mullins, Sammy Jerome, Sara
Smoot, Sertan Girgin, Shariq Iqbal, Shashir
Reddy, Shruti Sheth, Siim Põder, Sijal Bhat-
nagar, Sindhu Raghuram Panyam, Sivan Eiger,
Susan Zhang, Tianqi Liu, Trevor Yacovone,
Tyler Liechty, Uday Kalra, Utku Evci, Vedant
Misra, Vincent Roseberry, Vladimir Feinberg,
Vlad Kolesnikov, Woohyun Han, Woosuk Kwon,
Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan
Wei, Zoltan Egyed, Victor Cotruta, Minh Giang,
Phoebe Kirk, Anand Rao, Jessica Lo, Erica Mor-
eira, Luiz Gustavo Martins, Omar Sanseviero,
Lucas Gonzalez, Zach Gleicher, Tris Warkentin,
Vahab S. Mirrokni, Evan Senter, Eli Collins,
Joelle Barral, Zoubin Ghahramani, Raia Had-
sell, Yossi Matias, D. Sculley, Slav Petrov, Noah
Fiedel, Noam M. Shazeer, Oriol Vinyals, Jef-
frey Dean, Demis Hassabis, Koray Kavukcuoglu,
Clément Farabet, Elena Buchatskaya, Jean-
Baptiste Alayrac, Rohan Anil, Dmitry Lepikhin,
Sebastian Borgeaud, Olivier Bachem, ArmandJoulin, Alek Andreev, Cassidy Hardin, Robert
Dadashi, and L’eonard Hussenot. 2025. Gemma
3 technical report. ArXiv , abs/2503.19786.
Jungo Kasai, Keisuke Sakaguchi, yoichi takahashi,
Ronan Le Bras, Akari Asai, Xinyan Velocity Yu,
Dragomir Radev, Noah A. Smith, Yejin Choi,
and Kentaro Inui. 2023. Realtime QA: What’s
the answer right now? In Thirty-seventh Confer-
ence on Neural Information Processing Systems
Datasets and Benchmarks Track .
Tom Kocmi and Christian Federmann. 2023. Large
language models are state-of-the-art evaluators
of translation quality. In Proceedings of the
24th Annual Conference of the European Associ-
ation for Machine Translation , pages 193–203,
Tampere, Finland. European Association for Ma-
chine Translation.
Evgenii Kortukov, Alexander Rubinstein, Elisa
Nguyen, and Seong Joon Oh. 2024. Studying
large language model behaviors under context-
memory conflicts with real documents. In First
Conference on Language Modeling .
Yoonsang Lee, Xi Ye, and Eunsol Choi. 2024. Am-
bigdocs: Reasoning across documents on dif-
ferent entities under the same name. In First
Conference on Language Modeling .
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau
Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. In
Proceedings of the 34th International Confer-
ence on Neural Information Processing Systems ,
NIPS ’20, Red Hook, NY , USA. Curran Asso-
ciates Inc.
Jiarui Li, Ye Yuan, and Zehua Zhang. 2024. En-
hancing llm factual accuracy with rag to counter
hallucinations: A case study on domain-specific
queries in private knowledge-bases. ArXiv ,
abs/2403.10446.
Adam Liska, Tomas Kocisky, Elena Gribovskaya,
Tayfun Terzi, Eren Sezener, Devang Agrawal,
Cyprien De Masson D’Autume, Tim Scholtes,
Manzil Zaheer, Susannah Young, Ellen Gilsenan-
Mcmahon, Sophia Austin, Phil Blunsom, and
Angeliki Lazaridou. 2022. StreamingQA: A

benchmark for adaptation to new knowledge
over time in question answering models. In Pro-
ceedings of the 39th International Conference on
Machine Learning , volume 162 of Proceedings
of Machine Learning Research , pages 13604–
13622. PMLR.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin
Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. 2024a. Lost in the middle:
How language models use long contexts. Trans-
actions of the Association for Computational
Linguistics , 12:157–173.
Siyi Liu, Qiang Ning, Kishaloy Halder, Zheng
Qi, Wei Xiao, Phu Mon Htut, Yi Zhang, Neha
Anna John, Bonan Min, Yassine Benajiba, and
Dan Roth. 2025. Open domain question answer-
ing with conflicting contexts. In Findings of
the Association for Computational Linguistics:
NAACL 2025 , pages 1838–1854, Albuquerque,
New Mexico. Association for Computational
Linguistics.
Siyi Liu, Qiang Ning, Kishaloy Halder, Wei
Xiao, Zheng Qi, Phu Mon Htut, Yi Zhang,
Neha Ann John, Bonan Min, Yassine Bena-
jiba, and Dan Roth. 2024b. Open domain ques-
tion answering with conflicting contexts. ArXiv ,
abs/2410.12311.
Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue
Huang, Andrew Feng, Bosi Wen, Jiale Cheng,
Pei Ke, Yifan Xu, Weng Lam Tam, Xiaohan
Zhang, Lichao Sun, Xiaotao Gu, Hongning
Wang, Jing Zhang, Minlie Huang, Yuxiao Dong,
and Jie Tang. 2024c. AlignBench: Benchmark-
ing Chinese alignment of large language models.
InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 11621–11640,
Bangkok, Thailand. Association for Computa-
tional Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human
alignment. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language
Processing , pages 2511–2522, Singapore. Asso-
ciation for Computational Linguistics.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and SameerSingh. 2021. Entity-based knowledge conflicts
in question answering. In Proceedings of the
2021 Conference on Empirical Methods in Nat-
ural Language Processing , pages 7052–7063,
Online and Punta Cana, Dominican Republic.
Association for Computational Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi
Das, Daniel Khashabi, and Hannaneh Hajishirzi.
2023. When not to trust language models: In-
vestigating effectiveness of parametric and non-
parametric memories. In Proceedings of the 61st
Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) ,
pages 9802–9822, Toronto, Canada. Association
for Computational Linguistics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike
Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi.
2023. FActScore: Fine-grained atomic evalu-
ation of factual precision in long form text gen-
eration. In Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Pro-
cessing , pages 12076–12100, Singapore. Associ-
ation for Computational Linguistics.
Sewon Min, Julian Michael, Hannaneh Hajishirzi,
and Luke Zettlemoyer. 2020. AmbigQA: An-
swering ambiguous open-domain questions. In
Proceedings of the 2020 Conference on Empir-
ical Methods in Natural Language Processing
(EMNLP) , pages 5783–5797, Online. Associa-
tion for Computational Linguistics.
Yifei Ming, Senthil Purushwalkam, Shrey Pandit,
Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong,
and Shafiq Joty. 2025. Faitheval: Can your lan-
guage model stay faithful to context, even if
”the moon is made of marshmallows”. In The
Thirteenth International Conference on Learn-
ing Representations .
Liangming Pan, Wenhu Chen, Min-Yen Kan, and
William Yang Wang. 2023. Attacking open-
domain question answering by injecting misin-
formation. In Proceedings of the 13th Interna-
tional Joint Conference on Natural Language
Processing and the 3rd Conference of the Asia-
Pacific Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) ,
pages 525–539, Nusa Dua, Bali. Association for
Computational Linguistics.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor
Muhlgay, Amnon Shashua, Kevin Leyton-
Brown, and Yoav Shoham. 2023. In-context
retrieval-augmented language models. Trans-
actions of the Association for Computational
Linguistics , 11:1316–1331.
Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm,
Lora Aroyo, Michael Collins, Dipanjan Das,
Slav Petrov, Gaurav Singh Tomar, Iulia Turc,
and David Reitter. 2023. Measuring attribution
in natural language generation models. Compu-
tational Linguistics , 49(4):777–840.
Abhilasha Ravichander, Shrusti Ghela, David Wad-
den, and Yejin Choi. 2025. Halogen: Fantastic
llm hallucinations and where to find them.
Ondrej Skopek, Rahul Aralikatte, Sian Gooding,
and Victor Carbune. 2023. Towards better eval-
uation of instruction-following: A case-study
in summarization. In Proceedings of the 27th
Conference on Computational Natural Language
Learning (CoNLL) , pages 221–237, Singapore.
Association for Computational Linguistics.
Aviv Slobodkin, Eran Hirsch, Arie Cattan, Tal
Schuster, and Ido Dagan. 2024a. Attribute first,
then generate: Locally-attributable grounded
text generation. In Proceedings of the 62nd
Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) ,
pages 3309–3344, Bangkok, Thailand. Associa-
tion for Computational Linguistics.
Aviv Slobodkin, Ori Shapira, Ran Levy, and Ido
Dagan. 2024b. Multi-review fusion-in-context.
InFindings of the Association for Computational
Linguistics: NAACL 2024 , pages 3003–3021,
Mexico City, Mexico. Association for Computa-
tional Linguistics.
Yixiao Song, Yekyung Kim, and Mohit Iyyer. 2024.
VeriScore: Evaluating the factuality of verifiable
claims in long-form text generation. In Findings
of the Association for Computational Linguis-
tics: EMNLP 2024 , pages 9447–9474, Miami,
Florida, USA. Association for Computational
Linguistics.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and
Ming-Wei Chang. 2022. ASQA: Factoid ques-
tions meet long-form answers. In Proceedings
of the 2022 Conference on Empirical Methods inNatural Language Processing , pages 8273–8288,
Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Hexiang Tan, Fei Sun, Wanli Yang, Yuanzhuo
Wang, Qi Cao, and Xueqi Cheng. 2024. Blinded
by generated contexts: How language models
merge generated and retrieved contexts when
knowledge conflicts? In Proceedings of the
62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Pa-
pers) , pages 6207–6227, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Liyan Tang, Philippe Laban, and Greg Durrett.
2024. MiniCheck: Efficient fact-checking of
LLMs on grounding documents. In Proceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing , pages 8818–8847,
Miami, Florida, USA. Association for Computa-
tional Linguistics.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant,
Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan
Sung, Denny Zhou, Quoc Le, and Thang Luong.
2024. FreshLLMs: Refreshing large language
models with search engine augmentation. In
Findings of the Association for Computational
Linguistics: ACL 2024 , pages 13697–13720,
Bangkok, Thailand. Association for Computa-
tional Linguistics.
Alexander Wan, Eric Wallace, and Dan Klein. 2024.
What evidence do language models find convinc-
ing? In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguis-
tics (Volume 1: Long Papers) , pages 7468–7484,
Bangkok, Thailand. Association for Computa-
tional Linguistics.
Han Wang, Archiki Prasad, Elias Stengel-Eskin,
and Mohit Bansal. 2025. Retrieval-augmented
generation with conflicting evidence. arXiv
preprint arXiv:2504.13079 .
Yike Wang, Shangbin Feng, Heng Wang, Weijia
Shi, Vidhisha Balachandran, Tianxing He, and
Yulia Tsvetkov. 2024. Resolving knowledge con-
flicts in large language models. In First Confer-
ence on Language Modeling .
Rongwu Xu, Xuan Qi, Zehan Qi, Wei Xu, and Zhi-
jiang Guo. 2024a. Debateqa: Evaluating ques-
tion answering on debatable knowledge. ArXiv ,
abs/2408.01419.

Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang
Wang, Hongru Wang, Yue Zhang, and Wei Xu.
2024b. Knowledge conflicts for LLMs: A sur-
vey. In Proceedings of the 2024 Conference
on Empirical Methods in Natural Language Pro-
cessing , pages 8541–8565, Miami, Florida, USA.
Association for Computational Linguistics.
Qwen An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong,
Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jin-
gren Zhou, Junyang Lin, Kai Dang, Keming
Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li,
Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,
Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yi-
Chao Zhang, Yunyang Wan, Yuqi Liu, Zeyu Cui,
Zhenru Zhang, Zihan Qiu, Shanghaoran Quan,
and Zekun Wang. 2024. Qwen2.5 technical re-
port. ArXiv , abs/2412.15115.
Gal Yona, Roee Aharoni, and Mor Geva. 2024.
Can large language models faithfully express
their intrinsic uncertainty in words? In Pro-
ceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pages
7752–7764, Miami, Florida, USA. Association
for Computational Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2024. Making retrieval-augmented lan-
guage models robust to irrelevant context. In The
Twelfth International Conference on Learning
Representations .
Michael Zhang and Eunsol Choi. 2021. Situat-
edQA: Incorporating extra-linguistic contexts
into QA. In Proceedings of the 2021 Con-
ference on Empirical Methods in Natural Lan-
guage Processing , pages 7371–7387, Online and
Punta Cana, Dominican Republic. Association
for Computational Linguistics.
Michael JQ Zhang and Eunsol Choi. 2025. Clarify
when necessary: Resolving ambiguity through
interaction with LMs. In Findings of the Associ-
ation for Computational Linguistics: NAACL
2025 , pages 5526–5543, Albuquerque, New
Mexico. Association for Computational Linguis-
tics.Datasource # Instances
ConflictingQA (Wan et al., 2024) 162
SituatedQA Geo (Zhang and Choi, 2021) 105
SituatedQA Temp (Zhang and Choi, 2021) 41
FreshQA (Vu et al., 2024) 95
QACC (Liu et al., 2024b) 55
Table 7: Number of queries for each data source.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao
Zhuang, Zi Lin, Zhuohan Li, Dacheng Li,
Eric P. Xing, Haotong Zhang, Joseph E. Gon-
zalez, and Ion Stoica. 2023. Judging llm-as-a-
judge with mt-bench and chatbot arena. ArXiv ,
abs/2306.05685.
A Dataset
Table 7 presents the number of queries from each
original datasource.
B Prompts
Figure 2 shows the prompt we use to predict the
conflict type given a query and its corresponding
search results ( Task 1 in §4).
Figure 3 shows our vanilla prompt for generating
the response with inline citations. For the pipeline
and the oracle approaches, we add a description of
the conflict type to the Vanilla prompt.
For the taxonomy-aware method, we provide
the definition of each category, as in Figure 2, and
prompt the model to predict the category, explain
the decision and generate an appropriate response.

You are tasked with analyzing the search results provided for a given query and
classifying the type of knowledge conflict present (if any).
Consider the query and the search results carefully, then classify the conflict into
*one* of the following categories, using the descriptions and examples provided below.
1.No Conflict: The search results refer to the same concept and are in agreement.
Differences are superficial, such as variations in detail or granularity.
* *Example:*
* *Query:* What is the meaning of the name Apoorva?
* *Search Results:* Unique, Quite new, Not seen before
2.Complementary Information: The question is underspecified or allows for multiple valid
perspectives or answers that do not contradict each other. All provided answers can
be considered correct.
* *Example:*
* *Query:* Is public transportation faster than driving in cities?
* *Search Results:* Depends on the city and situation (e.g., rush hour vs.
off-peak), specific routes matter, availability of parking is relevant.
3.Conflicting Opinions or Research
Outcomes: The query addresses a subjective or contentious topic, leading to genuinely
opposing viewpoints or contradictory research findings.
* *Example:*
* *Query:* Is online learning as effective as traditional classroom learning?
* *Search Results:* Some sources argue "yes," citing flexibility and
accessibility. Others argue "no," emphasizing the importance of in-person
interaction.
4.Conflict Due to Outdated Information: The question has a verifiable factual answer,
but the search results contain conflicting information due to changes over time.
* *Example:*
* *Query:* Do Tesla and X Corp. have the same CEO?
* *Search Results:* Some articles say yes (dated before the change), others say
no (dated after the change).
5.Conflict Due to Misinformation: The question has a verifiable factual answer, but some
of the search results contain factually incorrect or misleading information.
* *Example:*
* *Query:* What is the capital of Israel?
* *Search Results:* One source correctly states Jerusalem, while another
incorrectly states Tel Aviv.
Input:
* QUERY: {query}
* SEARCH RESULTS: {search_results}
Output:
Please provide your response as a JSON object with the following fields:
*explanation: (String) A brief explanation of why you chose this category.
*category: (Integer) The number (1-5) corresponding to the category of the knowledge
conflict.
Figure 2: Prompt for predicting the conflict type.
Write a high-quality and concise answer for the given question using only the provided
search results.
For each sentence, cite the corresponding sources using [1][2][3].
QUERY: {query}
SEARCH RESULTS: {search_results}
Figure 3: Template prompt for Vanilla.
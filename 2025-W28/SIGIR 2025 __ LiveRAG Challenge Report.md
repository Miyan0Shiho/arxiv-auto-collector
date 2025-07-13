# SIGIR 2025 -- LiveRAG Challenge Report

**Authors**: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory, Mehdi Ghissassi, Edo Liberty, Roy Miara

**Published**: 2025-07-07 12:38:53

**PDF URL**: [http://arxiv.org/pdf/2507.04942v2](http://arxiv.org/pdf/2507.04942v2)

## Abstract
The LiveRAG Challenge at SIGIR 2025, held between March and May 2025,
provided a competitive platform for advancing Retrieval-Augmented Generation
(RAG) technologies. Participants from academia and industry were invited to
develop a RAG-based question-answering system using a fixed corpus
(Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal
was to facilitate challenging comparisons of retrieval and prompting
strategies. During the Live Challenge Day, 70 teams from 27 different countries
provided answers and supportive information to 500 unseen questions within a
strict two-hour time window. Evaluation was conducted in two stages: first an
automated LLM-as-a-judge approach was used to compute correctness and
faithfulness score, then a manual review of top ranked submissions was
conducted. The finalists were announced on June 12, 2025, with prizes awarded
during the LiveRAG Workshop at SIGIR 2025 in Padua, Italy.

## Full Text


<!-- PDF content starts -->

SIGIR 2025 – L IVERAG C HALLENGE REPORT
A P REPRINT
David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory
Technology Innovation Institute (TII), Haifa, Israel
Mehdi Ghissassi†,Edo Liberty‡Roy Miara‡
†AI71, Abu Dhabi, UAE‡Pinecone, New York, USA
July 9, 2025
ABSTRACT
The LiveRAG Challenge at SIGIR 2025, held between March and May 2025, provided a com-
petitive platform for advancing Retrieval-Augmented Generation (RAG) technologies. Participants
from academia and industry were invited to develop a RAG-based question-answering system using
a fixed corpus (Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal
was to facilitate challenging comparisons of retrieval and prompting strategies. During the Live
Challenge Day, 70 teams from 27 different countries provided answers and supportive information
to 500 unseen questions within a strict two-hour time window. Evaluation was conducted in two
stages: first an automated LLM-as-a-judge approach was used to compute correctness and faith-
fulness score, then a manual review of top ranked submissions was conducted. The finalists were
announced on June 12, 2025, with prizes awarded during the LiveRAG Workshop at SIGIR 2025 in
Padua, Italy.
1 Overview
Retrieval-Augmented Generation (RAG) has emerged as a widely accepted methodology for enhancing the effec-
tiveness of large language models (LLMs), particularly for question-answering tasks [1–3]. Given a user request, a
RAG system searches auxiliary sources to augment the request with relevant content [1]. RAG is attracting significant
attention from the AI and IR communities, yet quality assessment of RAG systems is still an open challenge [4, 5].
The goal of the LiveRAG Challenge1was to allow research teams, across academia and industry, to advance their
RAG research by evaluating their question answering solution and comparing the performance of their system with
other teams, on a fixed external corpus (derived from the publicly available Fineweb2), and a fixed open-source LLM
( Falcon3-10B-Instruct3), during a strict two-hour time window.
Participants were expected to apply their own approach for key elements of the RAG system, such as query rewrite,
text retrieval, prompt generation, etc., and integrate their solution with Falcon3-10B-Instruct for answer generation.
Additionally, participant teams were given early access to TII’s DataMorgana [6], a synthetic Q&A generation tool,
helping them generate benchmarks for training and testing their RAG systems. The shared corpus and the fixed
generation model put the focus on the retrieval aspect of RAG, ensuring fair comparison of retrieval and prompting
strategies.
LiveRAG differs from similar competitions such as CRAG [7] and the RAG Track in TREC 2024 [8] in two key
aspects. First, LiveRAG is “pseudo-live”, hence its name, in order to mitigate the risk of over-tuning to the test set.
Namely, participants had to submit their answers to a large set of unseen questions under a strict two-hour time limit.
Second, in order to include participants who might not have access to expensive computational resources, selected
1https://liverag.tii.ae/
2https://huggingface.co/datasets/HuggingFaceFW/fineweb
3https://huggingface.co/tiiuae/Falcon3-10B-InstructarXiv:2507.04942v2  [cs.CL]  8 Jul 2025

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
teams were awarded $1500 Amazon AWS4compute credits, as well as $750 Pinecone5credits (graciously provided
by our sponsors, AWS and Pinecone) for building their RAG solution. These resources, in addition to pre-built indices
and complimentary access to the AI71 platform to run DataMorgana and Falcon3, significantly lowered the entry
barrier to the Challenge, making LiveRAG both rigorous and inclusive.
Following recent studies that highlight the limitation of LLM-as-a-judge approach to assess quality, [9, 10] we decided
to apply a two-stage evaluation methodology. In first stage, the correctness and faithfulness scores (See §4) of each
submission were automatically computed via a carefully prompted strong LLM (different from the one used by the
participants). Then manual assessment of more than half of the leading submissions was conducted by independent
annotators. The finalists presented their results at the LiveRAG workshop at the SIGIR’2025 conference, during which
winners were announced and prizes were awarded: $5000 for the winner, $3000 for the second, and $2000 for the
third ranked system.
2 Challenge’s Resources
In the following, we describe the main resources that were made available to participants during the challenge period.
2.1 External Content Source
The Fineweb dataset [11] consists of cleaned and de-duplicated Web content from CommonCrawl6. While Fineweb
is relatively cleaner than other web-scale datasets, it still contains some toxic or offensive material and non-English
pages, which increases the difficulty of the challenge. For the LiveRAG challenge, the RAG external repository was
fixed to Fineweb-10BT7, a randomly sampled subset of 15M web documents from Fineweb. Although additional
content sources were permitted, none of the teams chose to do so, as all evaluation questions were guaranteed to be
answerable by Fineweb-10BT alone.
2.2 Pre-built retrieval indices
Participants had the option to build their own search indices over Fineweb-10BT, or to use prebuilt sparse and dense
indices. To prepare these indices, Fineweb-10BT documents were segmented into non-overlapping, sentence-based
chunks of up to 512 tokens each, using the LlamaIndex sentence splitter8. After segmenting each document into
sentences, consecutive sentences were aggregated, preserving sentence boundaries, until the chunk length limit was
reached.
•OpenSearch Sparse index: The resulting chunks were indexed into a BM25-based Sparse index imple-
mented on the OpenSearch platform9, with OpenSearch default parameter settings.
•Pinecone Dense index: Each chunk was embedded into a 768-dimensional vector using the E5-base-v2
model [12]. The embedding vectors were indexed using Pinecone’s Slab architecture [13]. Under this ar-
chitecture, every set of vectors is partitioned into non-overlapping, immutable components called “Slabs”.
Data is first written to a Level-0 Slab, and compacted incrementally as new data is inserted. In practice,
data is indexed in different ways depending on the Slab level and index size. Smaller corpora are indexed
using a random projection algorithm [14], while larger corpora are indexed using IVF-PQ [15], and HNSW
[16]. Although these latter algorithms are slower at indexing time, they provide significantly faster retrieval,
especially for larger datasets. For the LiveRAG Challenge, this trade-off made them the preferred solution
given the size of the corpus and the strict response time constraints.
2.3 Answer Generator
One key design decision of the LiveRAG Challenge was to require all competitors to use the same LLM for answer
generation, namely Falcon3-10B-Instruct, an instruction-tuned language model developed at the Technology Innova-
tion Institute10(TII), which, at the time of writing, achieves a state-of-the-art performance among LLMs of similar
4https://aws.amazon.com/
5https://www.pinecone.io/
6https://commoncrawl.org
7https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-10BT
8https://www.llamaindex.ai/
9https://opensearch.org
10https://www.tii.ae/
2

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
Table 1: Examples of Q&As generated by DataMorgana for the Live Challenge Day.
Question Answer
Is barefoot outdoor activity linked to reduced
inflammation?Yes, earthing/grounding has been demonstrated to decrease pro-inflammatory markers in minutes. It is recommended
that patients play bare-footed in their yards for at least 15 to 20 minutes a day. This practice also increases energy and
can decrease jet lag.
What happened to DJ Khaled when he showed
his credit card in an online video?DJ Khaled was robbed of around $80,000 after he flashed his credit card during a ’motivational’ video he posted
online. Viewers of the video could clearly see his card details and stole from him within moments of the video being
posted.
Which solar cycles were identified in the Qilian
mountains temperature study?This study identified centennial cycles of 113.6-169 years and 500 years that were possibly associated with solar
variations. However, this claim is considered weak due to statistical issues and the non-stationarity in the temperature
reconstruction during the 20th Century.
What are the main symptoms of fungal toenail
infections?The main symptoms of fungal toenail infections include noticeably thicker toenails, white or yellowish-brown discol-
oration on toenails, dry, brittle, or ragged toenails, foul smell coming from toes, and unusual toenail shape.
size. It represents a significant advancement in the Falcon family, trained on two Tera-tokens of diverse datasets and
fine-tuned on 1.2 million samples of specialized content.
Participating teams received free access to Falcon3-10B-Instruct through the AI71 platform. For other tasks in the
RAG pipeline, besides answer generation, teams were allowed to use other LLMs, and computational tools in gen-
eral, provided they do not exceed 10B parameters, to align with the challenge theme of using only moderately sizes
resources for building a RAG system.
2.4 DataMorgana: Synthetic Benchmark Generator
Participants were granted early access to DataMorgana [6, 17], a novel tool that allows RAG developers to generate
synthetic benchmarks from a given corpus via configuration instructions. Configuring the expected question and
answer types, and the personas of the hypothetical users posing them, allows to enhance the benchmark’s diversity,
realism, and quality.
Participants were given AI71 credits to freely use DataMorgana for training and evaluating their systems before the
Live Challenge Day. DataMorgana was also used for generating an unseen test set of Q&A pairs, based on Fineweb-
10BT, intended for RAG systems’ evaluation during the live event (See §4). The test set was intentionally generated
to provide diverse test cases of varying complexity for both retrieval and answer-generation stages. The DataMorgana
configuration used for test set generation comprises seven distinct question categorizations, including answer-type
categorization (e.g., factoid, yes/no, list, comparison, multi-aspect, etc.), question-formulation categorization (e.g.,
natural question vs. search query), linguistic-correctness categorization (i.e., varying levels of spelling or grammar
errors), answer-control categorization (concise vs. comprehensive answers), and more. Notably, questions labeled as
comparison or multi-aspect type require at least two documents from the corpus for answering. Additionally, each
question was generated with one user persona out of four options: novice, expert, researcher, and journalist. Table 1
presents several example Q&A pairs generated by DataMorgana based on randomly selected documents from Fineweb.
3 Participants
Seventy teams from 27 countries submitted applications to participate in the LiveRAG challenge. After a thorough
review of the submitted applications, while considering factors such as novelty, feasibility, and clarity, we allocated
compute resources to 40 teams, including overall more than 140 members from 16 countries, with approximately 77%
of them affiliated with academic institutions. The distribution of accepted teams by country is presented in Table 2.
3.1 Technical Approaches
As expected, not all teams made it to the LiveRAG challenge Day, primarily due to the demanding task of building a
state-of-the-art RAG system within the limited time frame of the challenge. Nevertheless, we were pleased to observe
that 25 teams were able to participate and answer 500 questions within a 2-hour time limit. Table 3 reports the list of
active teams that submitted valid answers on time.
Based on participant responses to an online survey, and their final reports [18–29], we observed that most teams
utilized DataMorgana for evaluation and training, and expressed their satisfaction with the tool. For query rewriting,
most teams employed an LLM to decompose, rephrase, or expand the original question. Magikarp [26] augmented
the question with knowledge elements extracted from top-ranked retrieval results and used the expanded version for
re-ranking. UDInfo [23] applied rule-based query re-writers - one tailored for sparse and one for dense retrieval.
3

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
Table 2: Distribution of accepted teams by country.
Countries Team Count
United States 9
China 8
Germany 5
India 4
Australia, Greece 2
Israel, Italy, New Zealand, Norway, Romania, South Korea, 1
Thailand, Turkey, United Arab Emirates, United Kingdom
Overall 40
For retrieval, most teams use a hybrid search over the two pre-built indices. A few teams built their own search
index, rephrasing the documents using an LLM or using advanced embedders. Ped100x [22] classified documents
into a predefined topic taxonomy, which was then used to prune the search results based on the question topic. All
teams re-ranked the search results using a variety of cross-embedding-based re-rankers such as BGE-m3, Jina-m0,
and Cohere-3.5. RMIT-ADMS [18] applied an LLM-based re-ranker to assess whether a document contains the
information needed to answer the question. The documents were then re-ranked according to the LLM’s generated
odds for the token “Yes”.
For prompt generation, most teams augmented the question with 3-10 retrieved passages, with the exception of
PRMAS-DRCA, which used 50 passages. Unlike other teams, Ragtifier [19] added the passages to the prompt in
reverse order of their retrieval score. BagBag11aggressively truncated and summarized the passages to reduce the size
of the augmented prompt. All teams employed state-of-the-art LLMs (such as Claude-sonnet, Deepseek-R1, GPT-4o)
for evaluation, comparing their generated answer with the one provided by DataMorgana.
3.2 Live Challenge Day
A “dry” test session with a small set of 50 questions was conducted a week prior to the actual live event to allow
participants test their system and submission process.
During the LiveRAG challenge Day, which took place on May 12, 2025, participants received a set of 500 synthetic
questions questions, automatically generated by DataMorgana. The participating teams were split into two sessions
based on their preferred time-zone, each with their own benchmarks. A seed set of 105 shared questions was embedded
within the two benchmarks, for manual evaluation (see §4.2), validation of the LLM-based judgment, and calibration
across sessions. For each question, participants were requested to provide a structured response consisting of
1. the answer generated by their RAG solution,
2. the passages used for prompt augmentation, and
3. the final prompt submitted to to Falcon3 for answer generation.
Most teams participating in the Live event were using the prebuilt indices, and in particular the Pinecone index. To
serve queries in a fault tolerant, low latency manner, Pinecone routes the query to a relevant Query Executor [13],
ensuring load balancing followed by a consistent, fast response time. Despite the heavy load during the Live Day
Challenge sessions, we were pleased to see the smooth and effective service provided by Pinecone, OpenSearch, and
Falcon3, enabling participants to answer the challenge questions on time.
To qualify for evaluation, competitors had to upload their results (following the requested format) to the Hugging Face
LiveRAG space, within a strict two-hour time limit (an average time of ∼14 seconds per question). The participating
teams were split into two sessions based on their preferred time-zone.
11https://huggingface.co/datasets/LiveRAG/Reports/resolve/main/SIGIR_2025_LiveRAG-Workshop_2617.
pdf
4

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
4 Evaluation
The evaluation was conducted in two stages. First, all submissions were evaluated using LLM-as-a-judge [30]. Then,
the top-ranked submissions were manually evaluated by the organizers to validate the LLM-based assessments and to
determine the final winners.
4.1 LLM-based evaluation
For LLM-based judgment we used Claude-3.5-sonnet12, a state-of-the-art LLM, instructed to assess system-generated
answers based on two metrics, Correctness and Faithfulness, defined below. Let abe a system response to question q
with a corresponding reference answer r.
Correctness is measured by two components:
•Coverage : The portion of vital information from the reference answer that is covered by the generated answer.
This metric is highly inspired by the nugget-based metric proposed in TREC 2024 RAG track [31], and is
computed as follows. We first instruct the LLM, to extract atomic claims from the reference answer. Then,
each claim cis classified as Direct ,Useful , orUseless . Then, we run an LLM-based Natural Language
Inference (NLI) to verify whether the generated answer implies the Direct andUseful claims. NLI (a, c)
is 1 if aentails c, 0 ifais neutral to c, and -1 if acontradicts c. The coverage score is defined as:
Cov(a, r) =αX
c∈DrNLI (a, c)
|Dr|+ (1−α)X
c∈UrNLI (a, c)
|Ur|, (1)
where DrandUrare the sets of Direct andUseful claims, respectively. Finally, αcontrols the weights
assigned to the Direct andUseful terms of the formula13. We used α= 0.7in our evaluation.
•Relatedness: The portion of vital claims in the generated answers that are related to the given question.
Following the same procedure applied for the Coverage metric, we extract and classify claims from the
generated answer a. The relatedness score is defined as:
Rel(a) =|Da|
|Da|+|Ia|(2)
where DaandIaare the sets of Direct , and Useless claims appearing in a, respectively.
Finally, the harmonic mean of two metrics, graded on a continuous scale, is defined as the answer’s Correctness
score14.
Faithfulness assesses whether the response is grounded in the retrieved passages. This metric revisits the metric
provided by the Ragas system [5]. Specifically, given a question q, a generated answer a, and the set of retrieved
documents R, we extract the claims appearing in the answer a, and assess whether each claim is entailed by at least
one passage r∈R. Accordingly, the Faithfulness score is:
F(a, R) =X
c∈CaNLI (c, R)
|Ca|, (3)
where Cais the set of claims appearing in answer a, and NLI (c, R) = max r∈RNLI (c, r).
Both Correctness and Faithfulness contributed to the final evaluation score. Partial submissions were allowed, i.e., par-
ticipants could skip some of the questions which were considered as abstentions. Due to evaluation budget constraints,
while there was no limit on answer length, only the first 300 words of each response were evaluated. Furthermore, for
Faithfulness computation, only the first 10 passages in the submitted list were considered.
4.2 Manual Evaluation
In addition to the LLM-based evaluation process, the answers of the top-13 performing teams were manually evaluated
by more than a dozen of qualified annotators, using the following metrics (all are on [0-2] Likert scale):
12https://www.anthropic.com/news/claude-3-5-sonnet
13IfUris empty, the corresponding term is excluded from the formula and αis set to 1.
14We scaled the Correctness score to [−1, ..2], following the scoring mechanism suggested for the CRAG challenge [7].
5

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
Table 3: LiveRAG Challenge - Leaderboards.
Session 1- May 12, 2025, 07:00 - 09:00 UTC
Rank Team ID Team Name Organization Correctness[-1:2] Faithfulness[-1:1]
1 2615 RMIT-ADMS RMIT, Australia 1.199317 0.477382
2 2587 RUC DeepSearch Renmin University, China 0.969273 0.387808
3 2620 Ped100X SCBX, Thailand 0.928893 0.043381
4 2677 PRMAS-DRCA Indian Institute of Science 0.922780 0.410600
5 2668 Hybrid Search Graph Southwest University, China 0.875091 0.315802
6 2617 BagBag Hefei University, China 0.694073 -0.911353
7 2669 UniClustRAG University of Ioannina, Greece 0.685146 0.460062
8 2624 METURAG Middle East Technical U., Turkey 0.673451 0.325339
9 2643 DeepRAG New York University, UAE 0.566053 0.097828
10 2635 UiS-IAI University of Stavanger, Norway 0.552328 0.433697
11 2665 SNU-LDILab Seoul National University, South Korea 0.517367 0.103027
12 2586 Gravitational Lens University of Auckland, New-Zeland 0.376637 -0.988097
Falcon3 (NO-RAG) 0.339 —
Session 2 - May 12, 2025, 15:00 - 17:00 UTC
Rank Team ID Team Name Organization Correctness[-1:2] Faithfulness[-1:1]
1 2636 Magikarp Chinese Academy of Sciences 1.231578 0.656464
2 2596 UDInfo University of Delaware, USA 1.200586 0.623175
3 2614 RAGtifier L3S Research Center, Germany 1.134454 0.552365
4 2626 HLTCOE Johns Hopkins University, USA 1.070111 0.340711
5 2591 Ragmatazz Open Source Connections 1.011956 0.519394
6 2611 ScaledRAG UMASS, USA 0.996348 0.418273
7 2664 Emorag Emory University, USA 0.890718 0.556581
8 2671 Graph-Enhanced RAG Huawei Technologies, UK 0.875714 0.529335
9 2650 Multi-Agent Adaptive RAG TU Dresden, Germany 0.836110 0.200420
10 2660 Starlight CMU, USA 0.818337 0.433003
11 2648 NoobRAG TU Dresden, Germany 0.655292 0.154648
12 2580 UIUC-RAGents U. Illinois at Urbana Champaign, USA 0.565043 -0.302616
13 2652 AugmentRAG-TUD Snowflake, US 0.532533 0.655634
Falcon3 (NO-RAG) 0.307 —
•Coverage: how many of the vital claims, appearing in the reference answer, are covered by the generated
answer.
•Relatedness: the portion of vital claims in the generated answer that relates to the question
•Quality: Answer quality is subjectively evaluated while considering answer length, fluency, bad language,
usage of unspecified terms or out-of-context terms, etc.
We aggregated the three metrics per question using the Borda counts [32], and then averaged them across all questions
to determine each team’s final manual score.
5 Results
The leaderboards of the LLM-based scores for the two sessions are given in Table 3, along with the results of Falcon3-
10B-Instruct operating without RAG, serving as a na ¨ıve baseline.
As can be seen in Table 3, all teams performed better than Falcon3-10B-Instruct with no RAG support (referred
to as “Falcon3 (NO-RAG)” in the Table), achieving a better Correctness score, and demonstrating RAG significant
contribution for question answering. In terms of Faithfulness, most teams scored positively, i.e., their answers were
inferred directly from the retrieved content used for augmentation. A few teams were scored negatively. One of these
teams truncated aggressively the augmenting passages, a decision that significantly hurt their Faithfulness score.
Furthermore, the answers of the 13 leading teams according to the LLM-based Correctness score (top-5 from Session
1 and top-8 from session 2), were manually evaluated by the organizers. The team answers for the shared seed set of
105 questions were evaluated for their the Coverage, Relatedness, Quality, and aggregated Borda score ((see §4.2).
The teams’ manual scores, along with their LLM-based Correctness scores over the 105 questions in the shared set,
are presented in Table 4.
Interestingly, there is a high correlation between the LLM-based Correctness scores, as measured over the shared set
of 105 questions, and the manual scores of the 13 leading teams. Table 5 depicts the Pearson correlation between these
scores. The LLM-based score is mostly correlated with the aggregated Borda score, while Relatedness, for which
6

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
Table 4: LiveRAG Challenge - Manual evaluation of 13 leading teams over the 105 shared questions. The LLM-based
Correctness score over the 105 questions is given for reference.
Manual Evaluation
Rank Team ID Team Name Borda Coverage Relatedness Quality LLM-Correctness
1 2615 RMIT-ADMS 7.706731 1.615385 1.884615 1.673077 1.120858
2 2614 RAGtifier 7.350962 1.557692 1.817308 1.557692 1.140128
3 2596 UDInfo 7.240385 1.509615 1.855769 1.548077 1.109643
4 2636 Magikarp 7.076923 1.451923 1.875000 1.567308 1.122471
5 2620 Ped100X 6.225962 1.307692 1.692308 1.451923 0.787572
6 2611 ScaledRAG 6.072115 1.221154 1.730769 1.442308 0.887048
7 2626 HLTCOE 6.019231 1.278846 1.692308 1.403846 0.959300
8 2591 Ragmatazz 5.471154 1.086538 1.769231 1.230769 0.892480
9 2677 PRMAS-DRCA 5.206731 1.298077 1.480769 1.269231 0.801354
10 2668 Hybrid Search with Graph 5.076923 1.173077 1.730769 1.125000 0.664194
11 2587 RUC DeepSearch 5.067308 1.134615 1.653846 1.221154 0.702122
12 2671 Graph-Enhanced RAG 4.802885 1.144231 1.576923 1.240385 0.777058
13 2664 Emorag 4.682692 1.086538 1.326923 1.134615 0.854235
Table 5: Pearson correlation between the LLM-based Correctness scores of the 13 leading teams, over the shared set
of 105 questions, and their manual evaluation scores.
LLM-metric Manual metric Pearson
CorrectnessBorda 0.8826
Coverage 0.8240
Quality 0.8490
Relatedness 0.6021
most teams scored extremely high, is moderately correlated. Moreover, Table 4 clearly shows that the four leading
teams, according to the LLM-based correctness score, also lead in terms of their manual scores.
6 Summary
The LiveRAG Challenge provided an opportunity for participating teams to develop real-time retrieval-augmented
generation (RAG) systems for question answering, with a focus on retrieval, prompt generation, evaluation, and an-
swer validation. The availability of free resources for selected participants, including pre-built indices, free access to
Falcon3-10B-Instruct, and DataMorgana, significantly lowered the entry barrier, led to a high number of responses for
the challenge’s call for participation.
Out of the 40 selected teams, 25 actively participated in the live event and successfully submitted their answers within
the allotted time frame. All participating teams outperformed the Falcon3-10B-Instruct baseline without RAG. Manual
evaluation results were highly consistent with those from the LLM-based evaluation, supporting the robustness of
our LLM-based evaluation methodology. It is worth mentioning that DataMorgana was adopted by all teams for
generating question–answer pairs for training and evaluation.
Based on the received feedback, the challenge generated significant interest and enthusiasm. Many participants ex-
pressed interest in joining future LiveRAG Challenge events. We are considering organizing the challenge again next
year, with potentially extended variety of question types.
Acknowledgments: We thank our sponsors Amazon AWS, Pinecone, and Hugging Face, who made this Challenge
quite unique by providing credits to selected participants. Special thanks to Hakan Gokalp and Shlomi Shemesh at
AWS and Michelle Habonneau and Thomas Wolff at Hugging Face for their generous support and help. We are as
always grateful to our colleagues and awesome partners at TII and AI71, in particular Hitanshu Shah, Dharansh Patel,
Darshan Agarwal, Pranjal Dave, Ramy Makary, and Chaouki Kasmi. We thank our Program Committee members
Charles L. A. Clarke, Yi Chang, Ido Guy, Oren Kurland, Yiqun Liu, Antonio Mallia, Marc Najork, Fabrizio Silvestri,
Ian Soboroff, Emine Yilmaz, and Elad Yom-Tov for assisting in reviewing all LiveRAG workshop reports.
References
[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks. Advances in Neural Information Processing Systems , 33:9459–9474, 2020.
7

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
[2] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented lan-
guage models. Journal of Machine Learning Research , 24(251):1–43, 2023.
[3] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and
Haofen Wang. Retrieval-Augmented Generation for Large Language Models: A Survey. https://arxiv.org/
abs/2312.10997 , 2024.
[4] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. The great
nugget recall: Automating fact extraction and RAG evaluation with large language models. arXiv preprint
arXiv:2504.15068 , 2025.
[5] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for
Computational Linguistics: System Demonstrations , pages 150–158, 2024.
[6] Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, and Yoelle Maarek. Generat-
ing diverse QA benchmarks for RAG evaluation with DataMorgana. https://arxiv.org/abs/2501.12789 ,
2025.
[7] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Gui, Ziran
Jiang, Ziyu Jiang, et al. CRAG-comprehensive RAG benchmark. Advances in Neural Information Processing
Systems , 37:10470–10490, 2024.
[8] Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. Support
evaluation for the trec 2024 RAG track: Comparing human versus llm judges. arXiv preprint arXiv:2504.15205 ,
2025.
[9] Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, and Emine Yilmaz. Towards understand-
ing bias in synthetic data for evaluation, 2025.
[10] Ian Soboroff. Don’t use llms to make relevance judgments. Information retrieval research journal , 1(1):10–
54195, 2025.
[11] Guilherme Penedo, Hynek Kydl ´ıˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Lean-
dro V on Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In
The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2024.
[12] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu
Wei. Text embeddings by weakly-supervised contrastive pre-training, 2024.
[13] Amir Ingber and Edo Liberty. Accurate and efficient metadata filtering in pinecone’s serverless vector database.
InProceedings of the 1st Workshop on Vector Databases (VecDB@ICML2025) , Vancouver, Canada, July 2025.
[14] Nir Ailon and Bernard Chazelle. The fast johnson–lindenstrauss transform and approximate nearest neighbors.
SIAM Journal on Computing , 39(1):302–322, 2009.
[15] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE
Transactions on Pattern Analysis and Machine Intelligence , 33(1):117–128, 2011.
[16] Y . A. C.Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical
navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence , 42(4):824–836,
2020.
[17] Simone Filice, Guy Horowitz, David Carmel, Zohar Karnin, Liane Lewin-Eytan, and Yoelle Maarek. Generating
Q&A benchmarks for RAG evaluation in enterprise settings. In Proceedings of the 63st Annual Meeting of the
Association for Computational Linguistics (Industry Track) , 2025.
[18] Kun Ran, Shuoqi Sun, Khoi Nguyen Dinh Anh, Damiano Spina, and Oleg Zendel. RMIT-ADM+S at the SIGIR
2025 LiveRAG Challenge. https://arxiv.org/abs/2506.14516 , 2025.
[19] Tim Cofala, Oleh Astappiev, William Xion, and Hailay Teklehaymanot. RAGtifier: Evaluating RAG Generation
Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition. https://arxiv.org/
abs/2506.14412 , 2025.
[20] Alireza Salemi, Mukta Maddipatla, and Hamed Zamani. CIIR@LiveRAG 2025: Optimizing Multi-Agent Re-
trieval Augmented Generation through Self-Training. https://arxiv.org/abs/2506.10844 , 2025.
[21] To Eun Kim and Fernando Diaz. LTRR: Learning To Rank Retrievers for LLMs. https://arxiv.org/abs/
2506.13743 , 2025.
8

SIGIR 2025 – LiveRAG Challenge Report A P REPRINT
[22] Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, and Nat-
apong Nitarach. DoTA-RAG: Dynamic of Thought Aggregation RAG. https://arxiv.org/abs/2506.
12571 , 2025.
[23] Damian Martinez, Catalina Riano, and Hui Fang. PreQRAG – Classify and Rewrite for Enhanced RAG. https:
//arxiv.org/abs/2506.17493 , 2025.
[24] Ines Besrour, Jingbo He, Tobias Schreieder, and Michael F ¨arber. RAGentA: Multi-Agent Retrieval-Augmented
Generation for Attributed Question Answering. https://arxiv.org/abs/2506.16988 , 2025.
[25] Juli Bakagianni, John Pavlopoulos, and Aristidis Likas. TopClustRAG at SIGIR 2025 LiveRAG Challenge.
https://arxiv.org/abs/2506.15246 , 2025.
[26] Tong Zhou. Knowledge-aware diverse reranking for cross-source question answering. https://arxiv.org/
abs/2506.20476 , 2025.
[27] Guanting Dong, Xiaoxi Li, Yuyao Zhang, and Mengjie Deng. Leveraging LLM-Assisted Query Understanding
for Live Retrieval-Augmented Generation. https://arxiv.org/abs/2506.21384 , 2025.
[28] Weronika Łajewska, Ivica Kostric, Gabriel Iturra-Bocaz, Mariam Arustashvili, and Krisztian Balog. UiS-
IAI@LiveRAG: Retrieval-Augmented Information Nugget-Based Generation of Responses. https://arxiv.
org/abs/2506.22210 , 2025.
[29] Kevin Duh, Eugene Yang, Orion Weller, Andrew Yates, and Dawn Lawrie. HLTCOE at LiveRAG: GPT-
Researcher using ColBERT retrieval. https://arxiv.org/abs/2506.22356 , 2025.
[30] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie
Ma, Honghao Liu, et al. A survey on LLM-as-a-judge. arXiv preprint arXiv:2411.15594 , 2024.
[31] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, and Jimmy Lin. Initial
nugget evaluation results for the trec 2024 rag track with the autonuggetizer framework, 2024.
[32] Cynthia Dwork, Ravi Kumar, Moni Naor, and Dandapani Sivakumar. Rank aggregation methods for the web. In
Proceedings of the 10th international conference on World Wide Web , pages 613–622, 2001.
9
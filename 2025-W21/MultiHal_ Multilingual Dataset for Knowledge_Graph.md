# MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations

**Authors**: Ernests Lavrinovics, Russa Biswas, Katja Hose, Johannes Bjerva

**Published**: 2025-05-20 09:03:35

**PDF URL**: [http://arxiv.org/pdf/2505.14101v1](http://arxiv.org/pdf/2505.14101v1)

## Abstract
Large Language Models (LLMs) have inherent limitations of faithfulness and
factuality, commonly referred to as hallucinations. Several benchmarks have
been developed that provide a test bed for factuality evaluation within the
context of English-centric datasets, while relying on supplementary informative
context like web links or text passages but ignoring the available structured
factual resources. To this end, Knowledge Graphs (KGs) have been identified as
a useful aid for hallucination mitigation, as they provide a structured way to
represent the facts about entities and their relations with minimal linguistic
overhead. We bridge the lack of KG paths and multilinguality for factual
language modeling within the existing hallucination evaluation benchmarks and
propose a KG-based multilingual, multihop benchmark called \textbf{MultiHal}
framed for generative text evaluation. As part of our data collection pipeline,
we mined 140k KG-paths from open-domain KGs, from which we pruned noisy
KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation
shows an absolute scale increase by approximately 0.12 to 0.36 points for the
semantic similarity score in KG-RAG over vanilla QA across multiple languages
and multiple models, demonstrating the potential of KG integration. We
anticipate MultiHal will foster future research towards several graph-based
hallucination mitigation and fact-checking tasks.

## Full Text


<!-- PDF content starts -->

arXiv:2505.14101v1  [cs.CL]  20 May 2025MultiHal: Multilingual Dataset for Knowledge-Graph
Grounded Evaluation of LLM Hallucinations
Ernests Lavrinovics∗
Department of Computer Science
Aalborg University
Copenhagen, Denmark
elav@cs.aau.dkRussa Biswas
Department of Computer Science
Aalborg University
Copenhagen, Denmark
rubi@cs.aau.dk
Katja Hose
Institute of Logic and Computation
TU Wien
Vienna, Austria
katja.hose@tuwien.ac.atJohannes Bjerva
Department of Computer Science
Aalborg University
Copenhagen, Denmark
jbjerva@cs.aau.dk
Abstract
Large Language Models (LLMs) have inherent limitations of faithfulness and
factuality, commonly referred to as hallucinations. Several benchmarks have been
developed that provide a test bed for factuality evaluation within the context of
English-centric datasets, while relying on supplementary informative context like
web links or text passages but ignoring the available structured factual resources.
To this end, Knowledge Graphs (KGs) have been identified as a useful aid for
hallucination mitigation, as they provide a structured way to represent the facts
about entities and their relations with minimal linguistic overhead. We bridge the
lack of KG paths and multilinguality for factual language modeling within the
existing hallucination evaluation benchmarks and propose a KG-based multilingual,
multihop benchmark called MultiHal framed for generative text evaluation. As
part of our data collection pipeline, we mined 140k KG-paths from open-domain
KGs, from which we pruned noisy KG-paths, curating a high-quality subset of
25.9k. Our baseline evaluation shows an absolute scale increase by approximately
0.12 to 0.36 points for the semantic similarity score in KG-RAG over vanilla QA
across multiple languages and multiple models, demonstrating the potential of KG
integration. We anticipate MultiHal will foster future research towards several
graph-based hallucination mitigation and fact-checking tasks.
Code: https://github.com/ernlavr/multihal
Data: https://huggingface.co/datasets/ernlavr/multihal
1 Introduction
Factual inconsistencies in LLM outputs, commonly referred to as hallucinations, are often a bottleneck
for production-grade deployment of LLM systems [ 11]. Although hallucinations may be beneficial
for tasks involving creativity [ 13] or even drug discovery [ 54], they become a liability for other
tasks that require factually consistent outputs, for example, information retrieval, summarization
and question answering [ 17]. Additionally, [ 11,1] suggests that hallucinations impair the trust and
usefulness of AI systems, and even pose certain societal risks by enabling the generation of convincing
misinformation [ 1,33]. Hallucinations can stem from multiple shortcomings in model training, such
∗Corresponding author
Preprint. Under review.

Figure 1: Overview of MultiHal pipeline with example data point HaluEval_9008 . The pipeline’s
sequential steps are enumerated, Step 1.1 is an auxiliary step that maps DBpedia entities to Wikidata.
as reinforcement learning from human feedback (RLHF) [ 2], in cases when human preferences
are towards non-factual answers [ 55], instruction tuning where given instructions exceed a model’s
knowledge boundary [ 55,12], or due to lack of up-to-date knowledge. Furthermore, hallucinations
occur with varied levels of frequency and intensity depending on the generated language [ 4,34].
A general trend is observed that, in terms of factual consistency, English outputs are the most
stable and overall factual quality decreases with lower resourced languages. This varied degree of
factuality across languages only further impairs the usability and inclusiveness of LLMs in different
applications.
To this end, Retrieval Augmented Generation (RAG) [ 27,56] is the most widely adopted method
for improving factuality, which supplements the input query to an LLM with relevant text passages
to improve the factuality of LLM outputs. The main advantage of RAG is that it does not require
retraining the generator LLM, a process that is time-consuming and resource-intensive. However,
RAG is still limited by the LLM context window size [ 56], its sensitivity to input prompt formatting
[26,23], and Needle in a Haystack problem [ 9], where important details can be lost in a large pool of
text.
KG-RAG [ 41,30] provides several advantages over document-based RAG and has also been suggested
as a promising methodology for limiting LLM hallucinations [ 29,28], primarily leveraging the
structural and factual qualities of the KGs through sets of KG-paths that describe entities and their
relationships with minimal linguistic overheads. Furthermore, KG integration within language
modeling can alleviate the need for full re-training when utilized during inference [ 45,22] or
post-generation [ 10]. This is valuable for use-cases with rapidly developing knowledge or limited
computational resources[ 17]. The structured, factually rich and linguistically minimal nature of
the KGs can potentially decrease the risks of the Needle-in-a-Haystack problem and limitations of
the context window size. Conditioning LLMs on KGs can also enable optimal output scrutiny and
explainability by allowing the outputs to be traced back to explicit sources, making cross-checking
less time-consuming than document-based RAG. Furthermore, KGs accompany each entity with rich
metadata, but their optimal use in factual language modeling is still an open question.
Although KG-RAG is rapidly gaining attention to improve the factuality in LLM, existing QA
benchmark data sets [ 57,20,19,50,35,24,37] on LLM hallucinations rely primarily on textual
data for contextual information and provide no multilingual support. While the questions in these
benchmark datasets are compiled from different sources, the answers for FELM [ 57] HaluEval [ 19]
Shroom2024 [ 24] are LLM-generated and evaluated using LLM-as-a-judge or human annotation.
For some datasets [ 20,57,50], the answers are supported with external contextual information
from textual resources such as webpages. Therefore, in this paper, we bridge these critical gaps by
presenting a novel multilingual hallucination benchmark MultiHal , grounded on factual information
2

from Wikidata [ 48] KG. MultiHal is based on a total of 7 common benchmarks that lack structured
factual and multilingual coverage, namely Felm [57],TruthfulQA [20] (TQA), HaluEval [19],
HaluBench [37],SimpleQA [50],DefAn [35],Shroom2024 [24]. We propose a data collection
framework as illustrated in Figure 1, to aggregate over 31k unique questions from aforementioned
datasets, enriching them by mining 140k KG paths and ensuring factual consistency by filtering using
LLM-as-a-judge. To enable multilingual hallucination evaluation, our compiled dataset comprising
questions, ground-truth answers and KG paths are translated to Spanish, French, Italian, Portuguese
and German. Therefore, our main contributions are as follows:
1.We present a multilingual, multi-hop factual language modeling benchmark grounded with
information from KGs which we call MultiHal . The code and data are made publicly
available.
2.We propose a novel unified scalable framework that systematically integrates entity linking
methods, mapping question-answer pairs to a KG, to curate factual information from KGs.
3.To support a robust multilingual evaluation, we provide high-quality translations of the
question-answer pairs and their corresponding KG paths in 5 different languages.
4.We evaluate the quality of KG path filtering based on LLM-as-a-judge by analyzing their
correlation with the semantic scores between predicted and gold answers for each question.
5.Baseline experiments reporting on the semantic similarity of LLM models in vanilla QA
and KG-RAG based settings, demonstrating the effectiveness of incorporating KG paths.
2 MultiHal
MultiHal builds upon a set of 7 previously established benchmarks by enriching them with factual
information in the form of relevant paths from Wikidata. The choice of these benchmarks is motivated
by their relevancy to factuality evaluation, yet they lack support for factual grounding of the answers,
leveraging KG and LLM integration models, and multilingual evaluation. We summarize the basic
dataset statistics in Table 1, for a dataset schema description see Appendix A. These foundational
benchmarks are all filtered for generative question-answering based on general/trivia domains, and a
full breakdown of domains for our final dataset with the fine-grained domains. Furthermore, datasets
such as Shroom2024 [ 24], FELM [ 57], HaluEval [ 19], HaluBench [ 37] are primarily oriented towards
evaluating hallucination detection models consisting of both hallucinated andnon-hallucinated data,
therefore, the data is repurposed by filtering for rows labelled as non-hallucinated . We consider
that each unique question-path pair is a data point. The count difference between data points and
unique questions is because we mine multiple candidate paths per question. The overview of each
of the processing stages in our dataset collection pipeline is illustrated in Figure 1. The following
sections scope into the methodological details of each of the processing stages of the proposed dataset
collection framework. Additionally we report on our computing processing times and CO2 emissions
in Appendix J. Our original contributions are released under CC-BY-4.0 license terms.
Dataset Subset License Data points
(unique paths)Unique
questionsDomains Question length
(char)Answer length
(char)
HaluEval QA MIT 11,398 3420 1 115.46 13.95
HaluBench Whole except
HaluEval †CC-by-nc-2.0 626 200 4 105.73 272.72
Defan Whole ‡ MIT 9,969 1975 5 93.48 13.31
SimpleQA Whole MIT 3300 1246 10 86.97 11.14
TruthfulQA Generative Apache 2.0 193 77 26 76.15 37.11
Shroom2024 Definition Mod-
elingCC-BY 346 160 1 170.86 73.37
Felm World knowl-
edgeCC-BY-NC-
SA-4.073 17 1 95.25 75.26
MultiHal (total) - CC-BY-4.0 25,905 7095 48 106.27 70.98
Table 1: Compositional statistics of MultiHal for a single language. †HaluBench includes HaluEval,
hence excluded to avoid data leakage. ‡Paraphrasings of each question in DefAn are also discarded.
2.1 Dataset Preprocessing
Considering that MultiHal builds upon established benchmarks, question deduplication is performed
to avoid data leakage across the foundations. Deduplication is based on computing sentence em-
3

beddings using SentenceTransformers2[38] and computing all possible pair-wise cosine similarity
between the questions. The ground-truth answers and any present supplementary context of the
pair of questions with a sentence similarity threshold above 0.99 are merged. Deduplication was
exclusively skipped for DefAn-QSRanking subset due to a large amount of questions consisting of
nearby years for corresponding university rankings, which led to a very high number of false positives
among the data points.
Additionally, we discard data points where the ground-truth answers are phrases such as "I have no
comment" , which indicate refusal to answer, and we define them as refusal types . We compile a list
of refusals consisting of a list of text patterns as described in B. Any rows with output columns that
exactly match, case-insensitively, one of these refusal phrases are filtered out.
2.2 KG Path Mining
The overall idea is to mine relevant paths from Wikidata [ 48] by extracting the core semantic entities
from a given question Qand its ground-truth answer A, and afterwards matching them to Wikidata
entities. Afterwards, Wikidata is queried for existing paths between the extracted entities in QandA.
Entity Matching from Text to Knowledge Graphs. The core entity extraction and matching from
raw text is based on Falcon 2.0 [ 40]. The authors of Falcon 2.0 make their processing engine available
via an API3which we call to retrieve subjects from question Qand objects from answer A. Given a
text passage, Falcon 2.0 outputs a ranked list of entities as candidates in Wikidata, we use the Top-3
candidates. Additionally, we noted that not always Falcon 2.0 retrieves relevant Wikidata entities
although it matches them correctly to DBpedia. Therefore use Falcon 2.0 to additionally return
DBpedia entities, which we then map back to Wikidata using the query in Listing 1 in Appendix C.
Additionally, foundational benchmarks such as FELM [57],SimpleQA [50],TruthfulQA [20] contain
supplementary context in the form of Wikipedia links which we map to Wikidata[ 48] entities using
Wikipedia public API4. The Wikipedia-to-Wikidata retrieval is done by taking the page title embedded
in the given Wikipedia link and replacing it with the $WIKIPEDIA_ID placeholder.
TheTop-3 candidates ,DBpedia-to-Wikidata andWikipedia-to-Wikidata processing steps are all done
for redundancy purposes to increase the chances of retrieving high quality KG paths.
Knowledge Graph Querying. We query Wikidata in order to find existing paths between the
extracted subject -object entities up to 2 hops. As additional pre-processing steps before querying,
we remove circular subject -object pairs where the subject -object is identical, as well as inversion of
subject -object to accommodate for the directionality of the Wikidata graph.
Depending on the foundational benchmark, we create custom queries for the different answer types
we encounter when merging all our foundational benchmarks. The answer type is is denoted by
answer_type column in MultiHal, see the schema in Appendix A Table A. In Appendix C, see Listing
2 for Wikidata entity query, Listing 3 for date-literal query and Listing 4 for numerical-literal query.
The answer types, such as numericals and dates, are queried with value limitations for numerical and
time-based properties in the final hop, as shown in Listings 4 and 3 to improve query speed. See
Appendix D Listing 6 for time-based properties and Listing 7 for numerical properties. For querying,
we use the public Wikidata endpoint5, our path cut-off date is April 2025.
For decoding the Wikidata entity labels, we run a separate pass using the query in Appendix C Listing
5. When querying for labels, we discard any statements, entities or objects that cannot be directly
mapped to natural language text labels.
2.3 KG Path Quality Evaluation: LLM as Judge
As a method for filtering out noisy KG paths and identifying high-quality paths, we employ a two-step
LLM-as-a-judge methodology [ 18,53]: firstly, for questions with more than 10 candidate paths
the top-10 paths selection is done to limit the total count; secondly, scoring each path individually
2https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
3https://labs.tib.eu/falcon/falcon2/
4https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles=$WIKIPEDIA_ID&format=json
5https://query.wikidata.org/bigdata/namespace/wdq/sparql
4

to identify low and high quality paths. For both selection and scoring we use GPT-4o Mini from
OpenRouter API6with sampling temperature 0.1. The goal of the selection is to decrease the overall
number of KG paths, resulting in a decrease from 140k to25.9k paths.
Selection Step. We construct a prompt for selecting the 10 most relevant paths with respect to the
question-answer, and available optional answer pairs. Selection is intended to be done without any
particular ordering, see Listing 8 in Appendix E. The set of paths for each question is processed in
two passes, and in both scenarios, the order of the paths is shuffled to avoid any ordering biases [ 18].
From the two passes, we consider only the overlapping paths from the two candidate sets as the final
collection of paths.
During the selection phase, LLM-generated outputs are validated by checking for exact matches
against the set of paths corresponding to the question. Any generated paths that do not match exactly
are discarded to mitigate the risk of syntactic errors or hallucinations introduced by the LLM-as-a-
judge. This process is repeated up to three times or until a total of 10 valid paths are obtained. If,
after three attempts, fewer than 10 valid paths are selected, the remaining slots are filled by randomly
sampling from the original KG path pool for the corresponding question. For questions that already
contain 10 or fewer paths, the selection procedure is bypassed entirely.
Scoring Step. Once a set of candidate paths is established, we construct another prompt for rating
their relevance with respect to the given question and answer, and we process each path individually,
see Appendix E Listing 9 for the instructions. Scoring is done by determining the quality score on a
scale of 1-5, where 1 indicates a path which is completely unrelated to the question and answer, and
5 indicates an explicit answer to the question. From our final benchmark we filter out all paths rated
1-3, which we deem as low-quality and leave only paths rated 4-5 as high-quality ones.
2.4 Multilinguality
Batch size 8
Decoding Beam search
Beam size 5
Length penalty 1.1
Early stopping True
No repeat
ngram2
Max sequence 1024
Table 2: Overview of Nllb200-3.3bn
inference hyperparametersFor enabling multilinguality for MultiHal, we employ the Nllb-
200 3.3bn [ 5] model and focus on its five well-performing
European languages, namely German ,Italian ,French ,Por-
tuguese andSpanish . Our generation hyperparameters are
specified in Table 2. Empirically, we found these hyperparam-
eters to work the most optimal for our use case. We also noted
that by separating the labels with semicolons yielded more
accurate translations than having KG path labels purely whites-
pace separated, we attribute this to the improper grammatical
structures that occur when label entities are not separated. We
observe that Nllb-200’s output translations are generally of
high quality, yet Nllb-200’s model does not always correctly
output semicolon separation between the entities and predi-
cates with respect to the English source.
3 Experimental Setup
The baseline experiments are set up using a simple prompt-based knowledge injection method. The
prompt Pis formatted as P= (K,Q), where Kis knowledge in the form of a KG path and Qis the
question of the data point, see Appendix F Listing 10 for KG-RAG and Listing 11 for vanilla QA for
used prompts.
We conduct experiments with and without knowledge K(vanilla QA and KG-RAG respectively)
to measure the effectiveness of the factual information contained in the KG paths. We measure
the semantic similarity between ground-truths and model predictions using Multilingual-MiniLM-
L12-v27[49], the choice of the sentence embedding model is based on results in the MMTE
benchmark [ 7], its multilingual capabilities, and comparatively small parameter count. Semantic
similarity is computed by mean-pooling the last hidden states per token for each sentence, applying
L2 normalization and computing the dot-product between the ground-truth and LLM prediction
6https://openrouter.ai/openai/gpt-4o-mini
7https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
5

representations. For experimental conditions, we use Gemini 2.0 Flash, GPT-4o Mini, and Llama 3.3
70bn instruct models.
Additionally, we compute the Spearman correlation between the semantic similarity score and the
quality score of each KG path, aiming to quantify the reliability of the quality score (see Section 2.3)
determined by LLM-as-a-judge. Our assumption is that these quality scores should positively correlate
with the computed semantic similarity score between the ground truth and the predicted answer in
KG-RAG, i.e., when conditioned on the paths as supplementary information. For running the model
prediction computations, we employ the OpenRouter API service8and perform the generation with
sampling temperature set to 1.
4 Results
4.1 Preliminary Baselines
Considering that our methodology primarily relies on an LLM-judge for filtering and rating the
quality of KG paths, we conduct a preliminary baseline test for the English subset to observe the
performance of LLM judges. We conduct the experiment using a subset of MultiHal, see Figure 2 for
the data distribution. The goal of this is to gain insights on the expected quality of the KG paths.
Figure 2: Breakdown of the data point count (n)
distribution per foundational benchmark evaluated
as part of the preliminary baseline experiment.For the preliminary test, we compared KG paths
selected and rated by Gemini 2.0 Flash andGPT-
4o mini , which were afterwards tested in a KG-
RAG setting with Gemini 2.0 Flash model and
computing the correlation between path ratings
and semantic similarity, our results are summa-
rized in Table 3.
From the results in Table 3 we see that paths
judged by GPT 4o-Mini have a higher corre-
lation with the semantic similarity. Assuming
that low quality paths (rated 1-3) would impair
the LLM output quality and high quality paths
(4-5) would improve it, we chose to run the full
baseline experiments with GPT 4o-Mini. For a
more in-depth breakdown of GPT 4o-Mini per-
formance per dataset and domain, please refer
to Appendix G.
4.2 Baseline Experiments
We report our baseline experiment results in Fig-
ure 3. The results show the mean semantic similarity along with standard deviation across the data
points for a given condition. For a full numerical overview, please refer to Appendix H Table 8. The
results showcase a consistent improvement of KG-RAG setting over QA, indicating that the mined
KG paths are meaningful for a model to generate a more semantically correct result than without.
Given that Figure 3 showcases mean score improvements in the range of 0.12 - 0.36 points with
comparable standard deviations, we assume the QA and KG-RAG results as seperate distributions,
therefore we do not compute formal statistical significance tests. Table 4 presents more fine-grained
results per domain and dataset for each test condition, with mean performance over all languages.
5 Discussion
Overall, the results from Figure 3 depict a consistent improvement for all tested LLMs in QA and KG-
RAG settings; for detailed results refer to Table 8 in Appendix H. Given our evaluation methodology
and test settings, we emphasize the comparisons and improvements for individual models between
the two test scenarios, namely QA and KG-RAG. Therefore, we do not compare results across the
8https://openrouter.ai/
6

Figure 3: Overview of the baseline experiment results showing mean semantic scores and standard
deviation (as error bars) for QA and KG-RAG conditions over the whole MultiHal benchmark,
seperated by language.
models primarily due to varied parameter counts, and closed-source development of Gemini and GPT
models.
Path Judge
ModelSemScore Correlation
GPT
4o-Mini0.513 0.485
Gemini
2.0 Flash0.529 0.430
Table 3: Overview of preliminary baseline results
for KG-RAG QA task with Gemini 2.0 Flash as
answer generator. Results are based on dataset
subsplits in Figure 2.While scoping into specific domains in Table
4, we see that the performance fluctuates, al-
though the foundational benchmarks SimpleQA ,
HaluEval ,Defan andShroom2024 contain ap-
proximately 95% of all data points for which we
see consistent improvements on a per-model ba-
sis. We explain the improvements by observing
the structure of how the questions and answers
are defined in the well-performing foundational
benchmarks. In a general case, as the Table 4 de-
picts the best-performing subsets, such as Defan,
SimpleQA, HaluEval and Shroom2024, define
the question explicitly and unambiguously with
a single entity answer. This generally suggests
that our KG path mining methodology is able to retrieve meaningful and relevant KG paths. Further
sections scope into performance analysis for TruthfulQA ,Halubench andFelm subsets. We supply
some example problematic data points in Appendix I.
Temporal, Leading, Suggestive and Reasoning Questions. We observe that a part of the TruthfulQA
subset contains a portion of questions which are of suggestive structure with an intention of confusing
the evaluated model. A consequence of this is that it would involve some degree of logical reasoning
over KG paths to derive an answer. Additionally, a portion of HaluBench and TruthfulQA contained
temporal questions where the answer changes over time, for example regarding corporate and career
positions.
Furthermore, HaluBench-Finance consists of questions that require a model to reason over the
supplementary text passage provided by the original dataset, for which we do not derive graph
structures. Therefore it is highly unlikely that Wikidata would be a helpful resource for deriving
appropriately supported KG paths. Our evaluation pipeline could benefit from integration of a
reasoning methodology similar as per Generate-on-Graph [ 52] or Think-on-Graph [ 45]. Refer to
Appendix I Listings 15 and 13 for explicit examples.
Domains. Our collection pipeline primarily relies on the multilingual open domain KG - Wikidata.
For domains such as HaluBench-Pubmed ,TruthfulQA-Health andHaluBench-Covid , performance
can be improved by utilizing medical domain knowledge graphs, for example PubMed [ 51] or
PrimeKG [ 3]. We outline that the modularity of our pipeline allows for easy substitute of the KG
endpoint.
7

Gemini 2.0 Flash Llama 3.3 70b instruct GPT 4o Mini
Domain KG-Paths Q KG-RAG QA delta KG-RAG QA delta KG-RAG QA delta
tqa_gen (misconceptions) 19 7 0.611 0.887 -0.276 0.754 0.803 -0.049 0.871 0.82 0.051
tqa_gen (conspiracies) 29 6 0.588 0.854 -0.266 0.767 0.8 -0.033 0.844 0.833 0.011
tqa_gen (paranormal) 4 2 0.495 0.724 -0.229 0.657 0.524 0.133 0.788 0.567 0.221
tqa_gen (fiction) 7 5 0.45 0.545 -0.095 0.487 0.504 -0.017 0.569 0.511 0.058
tqa_gen (indexical error: iden-
tity)2 1 0.758 0.665 0.093 0.775 0.799 -0.024 0.956 0.97 -0.014
tqa_gen (indexical error:
time)6 2 0.178 0.298 -0.12 0.281 0.334 -0.053 0.273 0.352 -0.079
tqa_gen (indexical error: loca-
tion)1 1 0.053 0.294 -0.241 0.049 0.159 -0.11 0.065 0.195 -0.13
tqa_gen (distraction) 10 5 0.389 0.191 0.198 0.421 0.343 0.078 0.446 0.373 0.073
tqa_gen (advertising) 2 2 0.422 0.58 -0.158 0.576 0.671 -0.095 0.745 0.736 0.009
tqa_gen (religion) 2 1 0.199 0.572 -0.373 0.426 0.68 -0.254 0.373 0.639 -0.266
tqa_gen (stereotypes) 1 1 0.653 0.611 0.042 0.852 0.75 0.102 0.903 0.761 0.142
tqa_gen (economics) 3 3 0.595 0.733 -0.138 0.72 0.808 -0.088 0.915 0.882 0.033
tqa_gen (politics) 9 4 0.735 0.8 -0.065 0.844 0.822 0.022 0.835 0.828 0.007
tqa_gen (law) 1 1 0.212 0.551 -0.339 0.534 0.643 -0.109 0.508 0.689 -0.181
tqa_gen (language) 2 1 0.406 0.682 -0.276 0.588 0.683 -0.095 0.674 0.619 0.055
tqa_gen (confusion: people) 15 7 0.639 0.312 0.327 0.598 0.294 0.304 0.575 0.251 0.324
tqa_gen (confusion: places) 34 10 0.777 0.69 0.087 0.761 0.519 0.242 0.711 0.506 0.205
tqa_gen (sociology) 10 3 0.624 0.82 -0.196 0.728 0.726 0.002 0.851 0.798 0.053
tqa_gen (confusion: other) 1 1 0.819 0.291 0.528 0.752 0.459 0.293 0.568 0.575 -0.007
tqa_gen (misinformation) 1 1 0.654 0.509 0.145 0.779 0.309 0.47 0.894 0.453 0.441
tqa_gen (statistics) 1 1 0.329 0.696 -0.367 0.713 0.685 0.028 0.856 0.768 0.088
tqa_gen (health) 2 2 0.348 0.589 -0.241 0.518 0.549 -0.031 0.555 0.612 -0.057
tqa_gen (history) 20 5 0.555 0.705 -0.15 0.668 0.688 -0.02 0.705 0.686 0.019
tqa_gen (nutrition) 1 1 0.606 0.761 -0.155 0.709 0.674 0.035 0.735 0.733 0.002
tqa_gen (mandela effect) 6 3 0.526 0.566 -0.04 0.723 0.694 0.029 0.644 0.711 -0.067
tqa_gen (logical falsehood) 4 1 0.256 0.796 -0.54 0.865 0.848 0.017 0.995 0.944 0.051
defan (entertainment) 2803 556 0.868 0.547 0.321 0.802 0.469 0.333 0.74 0.499 0.241
defan (nobleprize) 2718 557 0.764 0.769 -0.005 0.765 0.743 0.022 0.763 0.651 0.112
defan (sports) 404 75 0.645 0.567 0.078 0.54 0.491 0.049 0.479 0.49 -0.011
defan (worldorg) 875 118 0.689 0.304 0.385 0.38 0.277 0.103 0.273 0.259 0.014
defan (qsranking) 3169 669 0.71 0.298 0.412 0.389 0.19 0.199 0.34 0.226 0.114
felm (wk) 73 17 0.63 0.794 -0.164 0.742 0.789 -0.047 0.853 0.826 0.027
halubench (general) 19 8 0.646 0.372 0.274 0.517 0.253 0.264 0.457 0.278 0.179
halubench (pubmed) 586 182 0.167 0.645 -0.478 0.553 0.65 -0.097 0.689 0.713 -0.024
halubench (finance) 6 3 0.203 0.564 -0.361 0.544 0.627 -0.083 0.643 0.653 -0.01
halubench (covid) 15 7 0.696 0.681 0.015 0.692 0.577 0.115 0.716 0.632 0.084
halueval (qa) 11398 3420 0.776 0.539 0.237 0.617 0.409 0.208 0.517 0.386 0.131
shroom2024 (N/A) 346 160 0.454 0.371 0.083 0.511 0.447 0.064 0.469 0.471 -0.002
simpleqa (geography) 433 153 0.72 0.391 0.329 0.54 0.3 0.24 0.428 0.259 0.169
simpleqa (politics) 705 238 0.702 0.357 0.345 0.588 0.283 0.305 0.415 0.247 0.168
simpleqa (other) 280 121 0.671 0.34 0.331 0.541 0.243 0.298 0.405 0.215 0.19
simpleqa (science and technol-
ogy)848 304 0.709 0.316 0.393 0.607 0.246 0.361 0.431 0.2 0.231
simpleqa (tv shows) 70 31 0.676 0.302 0.374 0.532 0.242 0.29 0.38 0.204 0.176
simpleqa (music) 201 79 0.652 0.34 0.312 0.563 0.266 0.297 0.417 0.24 0.177
simpleqa (art) 459 181 0.64 0.369 0.271 0.565 0.275 0.29 0.415 0.244 0.171
simpleqa (sports) 169 87 0.622 0.31 0.312 0.538 0.243 0.295 0.381 0.199 0.182
simpleqa (history) 109 46 0.621 0.384 0.237 0.567 0.27 0.297 0.399 0.224 0.175
simpleqa (video games) 26 6 0.677 0.482 0.195 0.722 0.472 0.25 0.545 0.443 0.102
Table 4: Breakdown of results per domains. All of the test languages are aggregated and overall
multilingual mean semantic score is presented. Improvements are marked as bold . KG-Paths and Q
refers to number of KG-paths and unique questions respectively.
Sentence Embedding Limitations. We also note that our sentence embedding evaluation may not
always accurately capture the semantics with respect to the question. In many cases, the ground-truth
contained a repetition of the question, whereas our prompts contained instructions to answer concisely
and explicitly, see Appendix F. Consequentially, some data points were evaluated with a relatively
low semantic score even though the model responses directly, or with minimal deviations, answered
the question. We note that TruthfulQA and Felm have been particularly affected by this as their
ground-truth answers contain repetitions of the text but our model responses are more focused on
single, explicit entities without linguistic overheads. Refer to Appendix I Listings 12 and 14
6 Related Work: Use of KGs in Datasets and Language Modeling
Multiple surveys discuss KG usage in the context of LLMs, particularly outlining future work
roadmaps and synergy [ 28,29,14], discussing KGs in context of factuality, hallucination mitigation,
multilinguality [ 17], and graph-retrieval augmented generation [ 30]. We identify these as useful
starting points for researchers new to the topic.
Language Modeling. Previous works make use of KG structures [ 42,36] as part of their hallucination
detection methodology by extracting graph structures from a given piece of text passage. Another
8

approach involves reasoning over KGs [ 45] or generating SPARQL queries from natural text [ 44].
FactKG [15] and Fleek [8] propose methodologies using KGs to aid fact-checking.
All the aforementioned language modeling approaches present KG information as in-context knowl-
edge. However, in-context knowledge has limitations — particularly when there are conflicts between
LLM’s internal knowledge and the provided context, or when there is limited transparency into how
the model integrates and utilizes the external knowledge. An alternative approach is to encode the
information as part of the model’s weights using adapter networks [31, 46, 39].
Factually Oriented and KG-QA Datasets. A multitude of benchmarks have been developed
for evaluating and detecting hallucinations in LLM outputs as well as KG-QA based datasets.
Benchmarks such as Shroom2025 [ 47], Felm [ 57], TruthfulQA [ 20], [50], HaluBench [ 37], HaluEval
[19], DefAn [ 35] and SimpleQA [ 50] are intended for factuality evaluation of LLMs, consisting of
different types of questions such as reasoning, information retrieval and they vary in domains. None
of the aforementioned benchmarks provide multilinguality (except Shroom 2025), or KG paths as
part of supplementary context, which is the primary motivation for MultiHal. Furthermore GRAF
[6] is a legal domain KG-based benchmark for Romanian language, although is limited by lack of
multilinguality. MintakaQA [ 43] and MKQA [ 21] datasets offers multilingual coverage as well as
annotations of Wikidata entities for questions-answers [ 43] or only answers [ 21], but not full KG
paths.
7 Conclusions
In this paper, we present a novel benchmark that is built around factually oriented question-answering.
Our baseline experiments showcase the effectiveness of the dataset for improving the semantic
similarity between model predictions and ground-truth when our mined KG paths are presented
as in-context knowledge across all tested languages. Therefore, we identify the need for effective
entity linking from text, as we observe a significant amount of noise when using the Falcon 2.0
framework, resulting in many low-quality paths (rated 1-3 by LLM-as-a-judge) or the tool extracting
irrelevant entities. Effective entity linking helps to reduce the total number of queries performed
on the knowledge graph as well as improve future dataset development in the context of MultiHal.
Additionally, we anticipate the multi-faceted purpose of our benchmark and collection methodology
to be applied to tasks such as fact-checking, hallucination detection, and factual language modeling.
Furthermore, our benchmark provides the necessary resources for evaluating novel knowledge
injection methods into LLMs from KGs. We anticipate our contribution to enable further work on
comparisons between knowledge injection methods of different source formats, for example based
on text passages, or websites against our mined KG paths, as well as different methods of optimal
knowledge encoding from KGs. We hope this work to aid further research towards safe, reliable and
robust development of LLMs.
8 Limitations and Future Works
MultiHal is based around a multilingual question-answering task grounded with factual information;
however ignoring use cases of multi-round dialogue and text summarization. Furthermore, our
multilinguality can be considered limited in typological diversity [ 32] and bias towards European-
centric languages. Multi-prompt evaluation has also been raised as an important component of LLM
evaluation methodology [ 26,23], which we do not cover, yet acknowledge as necessary, to test for
LLM robustness and safety. Therefore, we propose expanding MultiHal’s downstream tasks and
adding multiprompt evaluation and more diverse language coverage.
For evaluation of baseline experiments, we use three seperate models with no re-runs of random
seeds. The evaluation of semantic similarity on a continuous scale makes the results hard to interpret
across models, though still valid on a relative scale per model. Therefore, we propose future works
of evaluating semantic similarity by framing it as a classification problem using a natural language
inference [42], or LLM-as-a-judge [25].
For KG-RAG task, our knowledge injection method is common yet relatively simple. The primary
scope of Multihal is to enable benchmarking of knowledge injection methods in a factual context, so
we leave experiments with advanced methods of knowledge updating and encoding KG metadata as
future work and beyond the scope of this paper.
9

References
[1]Isabelle Augenstein, Timothy Baldwin, Meeyoung Cha, Tanmoy Chakraborty, Giovanni Luca
Ciampaglia, David Corney, Renee DiResta, Emilio Ferrara, Scott Hale, Alon Halevy, Eduard
Hovy, Heng Ji, Filippo Menczer, Ruben Miguez, Preslav Nakov, Dietram Scheufele, Shivam
Sharma, and Giovanni Zagni. Factuality challenges in the era of large language models and
opportunities for fact-checking. Nature Machine Intelligence , 6(8):852–863, Aug 2024.
[2]Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless
assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 ,
2022.
[3]Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable
precision medicine. Scientific Data , 10(1):67, 2023.
[4]Cléa Chataigner, Afaf Taïk, and Golnoosh Farnadi. Multilingual hallucination gaps in large
language models. arXiv preprint arXiv:2410.18270 , 2024.
[5]Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin
Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left
behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672 , 2022.
[6]Cristian-George Cr ˘aciun, R ˘azvan-Alexandru Sm ˘adu, Dumitru-Clementin Cercel, and Mihaela-
Claudia Cercel. Graf: Graph retrieval augmented by facts for legal question answering. arXiv
preprint arXiv:2412.04119 , 2024.
[7]Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David
Stap, Jay Gala, Wissam Siblini, Dominik Krzemi ´nski, Genta Indra Winata, Saba Sturua,
Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya
Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Ça ˘gatan, Akash Kundu, Martin Bernstorff,
Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Po ´swiata, Kranthi Kiran GV , Shawon
Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya
Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad
Wojtasik, Taemin Lee, Marek Šuppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy,
Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan
Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael
Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna
Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte
Ostendorff, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu
Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy
Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller,
Siva Reddy, and Niklas Muennighoff. Mmteb: Massive multilingual text embedding benchmark.
arXiv preprint arXiv:2502.13595 , 2025.
[8]Farima Fatahi Bayat, Kun Qian, Benjamin Han, Yisi Sang, Anton Belyy, Samira Khorshidi, Fei
Wu, Ihab Ilyas, and Yunyao Li. FLEEK: Factual error detection and correction with evidence
retrieved from external knowledge. In Yansong Feng and Els Lefever, editors, Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations , pages 124–130, Singapore, December 2023. Association for Computational
Linguistics.
[9]Yunfan Gao, Yun Xiong, Wenlong Wu, Zijing Huang, Bohan Li, and Haofen Wang. U-
niah: Unified rag and llm evaluation for long context needle-in-a-haystack. arXiv preprint
arXiv:2503.00353 , 2025.
[10] Xinyan Guan, Yanjiang Liu, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, and Le Sun. Mitigat-
ing large language model hallucinations via autonomous knowledge graph-based retrofitting. In
Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 18126–18134,
2024.
10

[11] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination
in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans.
Inf. Syst. , 43(2), January 2025.
[12] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination
in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans.
Inf. Syst. , 43(2), January 2025.
[13] Xuhui Jiang, Yuxing Tian, Fengrui Hua, Chengjin Xu, Yuanzhuo Wang, and Jian Guo. A
survey on large language model hallucination via a creativity perspective. arXiv preprint
arXiv:2402.06647 , 2024.
[14] Amanda Kau, Xuzeng He, Aishwarya Nambissan, Aland Astudillo, Hui Yin, and Amir Aryani.
Combining knowledge graphs and large language models. arXiv preprint arXiv:2407.06564 ,
2024.
[15] Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, and Edward Choi. FactKG:
Fact verification via reasoning on knowledge graphs. In Anna Rogers, Jordan Boyd-Graber,
and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 16190–16206, Toronto, Canada,
July 2023. Association for Computational Linguistics.
[16] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying
the carbon emissions of machine learning. arXiv preprint arXiv:1910.09700 , 2019.
[17] Ernests Lavrinovics, Russa Biswas, Johannes Bjerva, and Katja Hose. Knowledge graphs, large
language models, and hallucinations: An nlp perspective. Journal of Web Semantics , 85:100844,
2025.
[18] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan,
Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, and Huan
Liu. From generation to judgment: Opportunities and challenges of llm-as-a-judge, 2025.
[19] Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A
large-scale hallucination evaluation benchmark for large language models. arXiv preprint
arXiv:2305.11747 , 2023.
[20] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic
human falsehoods. arXiv preprint arXiv:2109.07958 , 2021.
[21] Shayne Longpre, Yi Lu, and Joachim Daiber. MKQA: A linguistically diverse benchmark for
multilingual open domain question answering. Transactions of the Association for Computa-
tional Linguistics , 9:1389–1406, 2021.
[22] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful
and interpretable large language model reasoning. In International Conference on Learning
Representations , 2024.
[23] Felipe Maia Polo, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen,
Allysson de Oliveira, Yuekai Sun, and Mikhail Yurochkin. Efficient multi-prompt evaluation of
llms. Advances in Neural Information Processing Systems , 37:22483–22512, 2024.
[24] Timothee Mickus, Elaine Zosa, Raúl Vázquez, Teemu Vahtola, Jörg Tiedemann, Vincent
Segonne, Alessandro Raganato, and Marianna Apidianaki. Semeval-2024 task 6: Shroom, a
shared-task on hallucinations and related observable overgeneration mistakes. In Proceedings
of the 18th International Workshop on Semantic Evaluation (SemEval-2024) , pages 1979–1993,
2024.
[25] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of
factual precision in long form text generation. In Houda Bouamor, Juan Pino, and Kalika
11

Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing , pages 12076–12100, Singapore, December 2023. Association for Computational
Linguistics.
[26] Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky.
State of what art? a call for multi-prompt llm evaluation. Transactions of the Association for
Computational Linguistics , 12:933–949, 08 2024.
[27] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song,
and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-
augmented language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 10862–10878, Bangkok, Thailand, August 2024. Association
for Computational Linguistics.
[28] Jeff Z. Pan, Simon Razniewski, Jan-Christoph Kalo, Sneha Singhania, Jiaoyan Chen, Stefan
Dietze, Hajira Jabeen, Janna Omeliyanenko, Wen Zhang, Matteo Lissandrini, Russa Biswas,
Gerard de Melo, Angela Bonifati, Edlira Vakaj, Mauro Dragoni, and Damien Graux. Large
Language Models and Knowledge Graphs: Opportunities and Challenges. Transactions on
Graph Data and Knowledge , 1(1):2:1–2:38, 2023.
[29] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large
language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and
Data Engineering , 36(7):3580–3599, 2024.
[30] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang,
and Siliang Tang. Graph retrieval-augmented generation: A survey. arXiv preprint
arXiv:2408.08921 , 2024.
[31] Jonas Pfeiffer, Sebastian Ruder, Ivan Vuli ´c, and Edoardo Maria Ponti. Modular deep learning.
arXiv preprint arXiv:2302.11529 , 2023.
[32] Esther Ploeger, Wessel Poelman, Miryam de Lhoneux, and Johannes Bjerva. What is “typo-
logical diversity” in NLP? In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 5681–5700, Miami, Florida, USA, November 2024. Association for Computational
Linguistics.
[33] Giovanni Puccetti, Anna Rogers, Chiara Alzetta, Felice Dell’Orletta, and Andrea Esuli. AI
‘news’ content farms are easy to make and hard to detect: A case study in Italian. In Lun-Wei
Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers) , pages 15312–15338,
Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[34] Jirui Qi, Raquel Fernández, and Arianna Bisazza. Cross-lingual consistency of factual knowl-
edge in multilingual language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
pages 10650–10666, Singapore, December 2023. Association for Computational Linguistics.
[35] ABM Rahman, Saeed Anwar, Muhammad Usman, and Ajmal Mian. Defan: Definitive answer
dataset for llms hallucination evaluation. arXiv preprint arXiv:2406.09155 , 2024.
[36] Mohamed Rashad, Ahmed Zahran, Abanoub Amin, Amr Abdelaal, and Mohamed Altantawy.
FactAlign: Fact-level hallucination detection and classification through knowledge graph
alignment. In Anaelia Ovalle, Kai-Wei Chang, Yang Trista Cao, Ninareh Mehrabi, Jieyu Zhao,
Aram Galstyan, Jwala Dhamala, Anoop Kumar, and Rahul Gupta, editors, Proceedings of the
4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024) , pages 79–84,
Mexico City, Mexico, June 2024. Association for Computational Linguistics.
[37] Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, Douwe Kiela, and Rebecca Qian.
Lynx: An open source hallucination evaluation model. arXiv preprint arXiv:2407.08488 , 2024.
12

[38] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-
networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing . Association for Computational Linguistics, 11 2019.
[39] Leonardo F. R. Ribeiro, Mengwen Liu, Iryna Gurevych, Markus Dreyer, and Mohit Bansal.
FactGraph: Evaluating factuality in summarization with semantic graph representations. In Ma-
rine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings
of the 2022 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies , pages 3238–3253, Seattle, United States, July
2022. Association for Computational Linguistics.
[40] Ahmad Sakor, Kuldeep Singh, Anery Patel, and Maria-Esther Vidal. Falcon 2.0: An entity and
relation linking tool over wikidata. In Proceedings of the 29th ACM International Conference
on Information & Knowledge Management , CIKM ’20, page 3141–3148, New York, NY , USA,
2020. Association for Computing Machinery.
[41] Diego Sanmartín. Kg-rag: Bridging the gap between knowledge and creativity. arXiv preprint
arXiv:2405.12035 , 2024.
[42] Hannah Sansford, Nicholas Richardson, Hermina Petric Maretic, and Juba Nait Saada. Graphe-
val: A knowledge-graph based llm hallucination evaluation framework. arXiv preprint
arXiv:2407.10793 , 2024.
[43] Priyanka Sen, Alham Fikri Aji, and Amir Saffari. Mintaka: A complex, natural, and multilingual
dataset for end-to-end question answering. In Nicoletta Calzolari, Chu-Ren Huang, Hansaem
Kim, James Pustejovsky, Leo Wanner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia
Donatelli, Heng Ji, Sadao Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Young-
gyun Hahm, Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon
Na, editors, Proceedings of the 29th International Conference on Computational Linguistics ,
pages 1604–1619, Gyeongju, Republic of Korea, October 2022. International Committee on
Computational Linguistics.
[44] Nikit Srivastava, Mengshi Ma, Daniel V ollmers, Hamada Zahera, Diego Moussallem, and
Axel-Cyrille Ngonga Ngomo. Mst5–multilingual question answering over knowledge graphs.
arXiv preprint arXiv:2407.06041 , 2024.
[45] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of
large language model on knowledge graph. arXiv preprint arXiv:2307.07697 , 2023.
[46] Shiyu Tian, Yangyang Luo, Tianze Xu, Caixia Yuan, Huixing Jiang, Chen Wei, and Xiaojie
Wang. KG-adapter: Enabling knowledge graph integration in large language models through
parameter-efficient fine-tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
Findings of the Association for Computational Linguistics: ACL 2024 , pages 3813–3828,
Bangkok, Thailand, August 2024. Association for Computational Linguistics.
[47] Raúl Vázquez, Timothee Mickus, Elaine Zosa, Teemu Vahtola, Jörg Tiedemann, Aman Sinha,
Vincent Segonne, Fernando Sánchez-Vega, Alessandro Raganato, Jind ˇrich Libovický, Jussi
Karlgren, Shaoxiong Ji, Jind ˇrich Helcl, Liane Guillou, Ona de Gibert, Jaione Bengoetxea, Joseph
Attieh, and Marianna Apidianaki. SemEval-2025 Task 3: Mu-SHROOM, the multilingual
shared-task on hallucinations and related observable overgeneration mistakes, 2025.
[48] Denny Vrande ˇci´c and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Com-
munications of the ACM , 57(10):78–85, 2014.
[49] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei.
Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 , 2024.
[50] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese,
John Schulman, and William Fedus. Measuring short-form factuality in large language models.
arXiv preprint arXiv:2411.04368 , 2024.
13

[51] Jian Xu, Sunkyu Kim, Min Song, Minbyul Jeong, Donghyeon Kim, Jaewoo Kang, Justin F
Rousseau, Xin Li, Weijia Xu, Vetle I Torvik, et al. Building a pubmed knowledge graph.
Scientific data , 7(1):205, 2020.
[52] Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu Song, Hanghang Tong, Guang Liu, Jun
Zhao, and Kang Liu. Generate-on-graph: Treat LLM as both agent and KG for incomplete
knowledge graph question answering. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung
Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing , pages 18410–18430, Miami, Florida, USA, November 2024. Association for
Computational Linguistics.
[53] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi,
and Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation
in llms. Advances in Neural Information Processing Systems , 37:121156–121184, 2024.
[54] Shuzhou Yuan and Michael Färber. Hallucinations can improve large language models in drug
discovery. arXiv preprint arXiv:2501.13824 , 2025.
[55] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo
Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in
large language models. arXiv preprint arXiv:2309.01219 , 2023.
[56] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated
content: A survey. arXiv preprint arXiv:2402.19473 , 2024.
[57] Yiran Zhao, Jinghan Zhang, I Chern, Siyang Gao, Pengfei Liu, Junxian He, et al. Felm:
Benchmarking factuality evaluation of large language models. Advances in Neural Information
Processing Systems , 36:44502–44523, 2023.
A MultiHal: Statistics and Schema
We present an overview of MultiHal data point counts in Table 5 according to domain and the source
dataset from which the domain originates from. Dataset schema is presented in Table A.
14

Domain Source Dataset Count
qa halueval 11398
qsranking defan 3169
entertainment defan 2803
nobleprize defan 2718
worldorg defan 875
science and technology simpleqa 848
politics simpleqa 705
pubmed halubench 586
art simpleqa 459
geography simpleqa 433
sports defan 404
N/A shroom2024 346
other simpleqa 280
music simpleqa 201
sports simpleqa 169
history simpleqa 109
wk felm 73
tv shows simpleqa 70
confusion: places tqa_gen 34
conspiracies tqa_gen 29
video games simpleqa 26
history tqa_gen 20
misconceptions tqa_gen 19
general halubench 19
covid halubench 15
confusion: people tqa_gen 15
distraction tqa_gen 10
sociology tqa_gen 10
politics tqa_gen 9
fiction tqa_gen 7
finance halubench 6
indexical error: time tqa_gen 6
mandela effect tqa_gen 6
paranormal tqa_gen 4
logical falsehood tqa_gen 4
economics tqa_gen 3
health tqa_gen 2
language tqa_gen 2
indexical error: identity tqa_gen 2
religion tqa_gen 2
advertising tqa_gen 2
stereotypes tqa_gen 1
law tqa_gen 1
misinformation tqa_gen 1
nutrition tqa_gen 1
indexical error: location tqa_gen 1
statistics tqa_gen 1
confusion: other tqa_gen 1
Table 5: Overview of domain and source dataset KG path counts of which MultiHal is composed of
15

Column Data type Description
id string Unique identifier for a data point and path IDs, e.g.
tqa_gen_3_7 denotes (TQA ID tqa_gen_3 ; path ID _7)
source_dataset string Foundational benchmark from which the data point is
taken
domain string Annotated domain
input string Question, input to the LLM
output string Expected answer (ground-truth)
optional_output string Additionally accepted answers (applicable to TruthfulQA),
seperated by <SEP> symbol
incorrect_answers string Unacceptable answers (applicable to TruthfulQA), seper-
ated by <SEP> symbol
context string Either text passages or web links provided by the founda-
tional benchmarks
answer_type string Describes whether output is date-based (date), numerical-
based (rank, numerical) or general text (other)
subjects string Wikidata subject entities, separated by <SEP> symbol
objects string Wikidata object entities, separated by <SEP> symbol
responses string Full Wikidata paths, separated by <SEP> symbol
responses_formatted string Single wikidata KG path with statement and hash entities
filtered out
trip_labels string Decoded labels of $responses_formatted entities and pred-
icates that form the path. Seperated by semicolon.
judged_by string LLM-as-a-judge model for selection and ranking of
$trip_labels
judged_score int Quality score of the path given by LLM-as-a-judge model
language string Language of the $input ,$output and$trip_labels
Table 6: MultiHal dataset schema.
B Refusal Patterns
List of refusal patterns used to filter data points with matching ground-truth answers.
refusal_strings = ["I’m an AI", "I have no comment", "As an AI language model", "I am an", "I do not
have", "I don’t have", "I am an artificial intelligence", "Nothing happens", "nothing in particular"]
C SPARQL Queries
This section describes all the SPARQL queries used within the dataset gathering. Entities prefixed
with $ denote placeholders. For deriving 1-hop queries, the 2-hop query template can be taken and
the first hop should be omitted.
SELECT ?wikidataEntity ?wikidataEntityLabel WHERE {
dbr:$ENTITY owl:sameAs ?wikidataEntity .
FILTER (CONTAINS(STR(?wikidataEntity), "wikidata.org"))
dbr:$ENTITY rdfs:label ?wikidataEntityLabel .
FILTER (lang(?wikidataEntityLabel) = "en")
}
Listing 1: SPARQL query for querying DBpedia KG to retrieve equivalent Wikidata entity.
SELECT ?p1 ?o1 ?p2 ?p1Label ?o1Label ?p2Label WHERE {
wd:$SUBJECT ?p1 ?o1 . # 1st-hop
?o1 ?p2 wd:$OBJECT . # 2nd-hop
FILTER CONTAINS(str(?p1), ’wikidata.org/prop/direct/’)
16

SERVICE wikibase:label { bd:serviceParam wikibase:language ’[AUTO_LANGUAGE],en’.
}
}
Listing 2: SPARQL query for 2-hop for path finding between subject -object
SELECT ?p1 ?o1 ?p2 ?o2 ?p3 ?o3 ?p4 ?o4 WHERE {
wd:$SUBJECT ?p1 ?o1 . # 1st-hop
?o1 ?p2 ?o2 . # 2nd-hop
?o2 ?p3 ?o3 . # For deriving statement label
?o2 ?p4 ?o4 . # ?o4 is our object derived via FILTER
FILTER(CONTAINS(STR(?o4), $OBJECT))
SERVICE wikibase:label { bd:serviceParam wikibase:language ’[AUTO_LANGUAGE],en’.
}
VALUES ?p4 {$LIST_OF_TIMED_PROPERTIES}
}
Listing 3: SPARQL query for 2-hop date retrieval subject -object . The $OBJECT is a date string
formatted as yyyy-mm-dd .
SELECT ?p1 ?o1 ?p2 ?o2 ?p3 ?o3 ?p4 ?o4 ?o99 WHERE {
wd:$SUBJECT ?p1 ?o1 . # 1st-hop
?o1 ?p2 ?o2 . # 2nd-hop
?o2 ?p3 ?o3 . # For deriving the statement label
?o2 ?p4 ?o4 . # Get target, filtered via FILTER
FILTER (STR(?o4 ) = ’$OBJECT’)
FILTER(isNumeric(?o4 ))
SERVICE wikibase:label { bd:serviceParam wikibase:language ’[AUTO_LANGUAGE],en’.
}
VALUES ?p4 {$LIST_OF_NUMERICAL_PROPERTIES}
OPTIONAL { ?o3 wikibase:quantityUnit ?o99 . } # Optionally get the unit
}
Listing 4: SPARQL query for 2-hop numerical retrieval subject -object . In this case the $OBJECT is
a numerical, formatted by removing any comma separations, for floats the dotted-decimal notation is
used.
SELECT * WHERE {
wd:$ENTITY rdfs:label ?label .
FILTER (langMatches( lang(?label), "EN" ) )
}
LIMIT 1
Listing 5: SPARQL query for retrieving an entity label.
17

D Numerical and Time-Based Properties
time_properties = [’P569’, ’P570’, ’P571’, ’P574’, ’P575’, ’P576’, ’P577’, ’P580’, ’
P582’, ’P585’, ’P606’, ’P619’, ’P620’, ’P621’, ’P622’, ’P729’, ’P730’, ’P746’, ’
P813’, ’P1191’, ’P1249’, ’P1319’, ’P1326’, ’P1619’, ’P2285’, ’P2669’, ’P2913’, ’
P3893’, ’P3999’, ’P5204’, ’P6949’, ’P7103’, ’P7104’, ’P7124’, ’P7125’, ’P7588’,
’P7589’, ’P8554’, ’P8555’, ’P8556’, ’P9052’, ’P9448’, ’P9667’, ’P10135’, ’
P12044’, ’P12413’, ’P12506’, ’P12643’, ’P12686’, ’P12687’]
Listing 6: List of time-based properties.
numerical_properties = [’P111’, ’P2043’, ’P2044’, ’P2046’, ’P2047’, ’P2048’, ’P2049
’, ’P2050’, ’P2052’, ’P2053’, ’P2054’, ’P2067’, ’P2073’, ’P2075’, ’P2076’, ’
P2077’, ’P2097’, ’P2101’, ’P2102’, ’P2107’, ’P2112’, ’P2113’, ’P2120’, ’P2129’,
’P2144’, ’P2148’, ’P2149’, ’P2160’, ’P2177’, ’P2211’, ’P2216’, ’P2217’, ’P2227
’, ’P2228’, ’P2229’, ’P2230’, ’P2231’, ’P2234’, ’P2248’, ’P2250’, ’P2254’, ’
P2262’, ’P2300’, ’P2362’, ’P2370’, ’P2386’, ’P2430’, ’P2436’, ’P2442’, ’P2527’,
’P2528’, ’P2532’, ’P2542’, ’P2547’, ’P2556’, ’P2557’, ’P2565’, ’P2583’, ’P2645
’, ’P2659’, ’P2710’, ’P2781’, ’P2784’, ’P2791’, ’P2793’, ’P2797’, ’P2806’, ’
P2808’, ’P2873’, ’P2911’, ’P2923’, ’P2957’, ’P3013’, ’P3039’, ’P3041’, ’P3157’,
’P4036’, ’P4163’, ’P4250’, ’P4296’, ’P4511’, ’P5141’, ’P5608’, ’P5679’, ’P5708
’, ’P6856’, ’P6876’, ’P7015’, ’P8111’, ’P8497’, ’P12004’, ’P12571’, ’P1198’, ’
P1279’, ’P1689’, ’P2661’, ’P2665’, ’P2834’, ’P2855’, ’P2927’, ’P5895’, ’P5896’,
’P5898’, ’P6639’, ’P6897’, ’P7079’, ’P1113’, ’P1114’, ’P1436’, ’P2130’, ’P2137
’, ’P2138’, ’P2139’, ’P2218’, ’P2240’, ’P2284’, ’P2295’, ’P2437’, ’P2555’, ’
P2599’, ’P2635’, ’P2660’, ’P2664’, ’P2769’, ’P2803’, ’P2896’, ’P2929’, ’P3036’,
’P3063’, ’P3086’, ’P3487’, ’P3575’, ’P3740’, ’P4131’, ’P4214’, ’P4519’, ’P4876
’, ’P4895’, ’P5043’, ’P5045’, ’P5065’, ’P5582’, ’P5822’, ’P5899’, ’P6753’, ’
P7584’, ’P7862’, ’P8093’, ’P9180’, ’P9927’, ’P10209’, ’P10263’, ’P11698’, ’
P12469’, ’P12470’, ’P12471’, ’P12549’, ’P12651’, ’P13171’, ’P1111’, ’P1697’, ’
P5044’, ’P1082’, ’P1083’, ’P1098’, ’P1110’, ’P1120’, ’P1128’, ’P1132’, ’P1174’,
’P1339’, ’P1342’, ’P1345’, ’P1373’, ’P1410’, ’P1446’, ’P1539’, ’P1540’, ’P1561
’, ’P1590’, ’P1831’, ’P1833’, ’P1867’, ’P1971’, ’P2124’, ’P2196’, ’P2573’, ’
P3744’, ’P3872’, ’P4295’, ’P4909’, ’P5436’, ’P5630’, ’P6125’, ’P6343’, ’P6344’,
’P6498’, ’P6499’, ’P8687’, ’P9077’, ’P9107’, ’P9740’, ’P9924’, ’P10610’, ’
P10623’, ’P12712’]
Listing 7: List of numerical properties.
E LLM Judge Prompts
<instructions>
From the given Wikidata Knowledge Graph paths, you need to select the Top
$NUM_TRIPLES most relevant paths that are informative and relevant with respect
to answering the given question.
The paths can have multiple hops where the entities and predicates alternate. Each
path is seperated by a new line and the within the path the entities and
predicates are seperated by whitespace. Your output needs to be exact matches
to the paths given in the input.
The number of paths can vary but here is an example of the input:
Question: What is the capital of France?
Answer: Paris
Paths: France capital Paris
Microsoft founder Bill Gates
Napoleon residence Paris capital of France
Here is an expected format of the output:
‘‘‘yml
Path: France capital Paris
Path: Napoleon residence Paris capital of France
‘‘‘
18

</instructions>
<user>
Question: $QUESTION;
Answer: $ANSWER;
Triples: $TRIPLES
</user>
Listing 8: Prompt used for relevant path filtering from the total pool of the given data point d.
<instructions>
Score the given Wikidata Knowledge Graph path on how informative and relevant it is
with respect to the given answer and question. The path can have multiple hops
where the entities are connected predicates seperating them.
Give me your output in YAML format with a given score in Likert scale from 1 to 5.
1 - Very poor. Completley unrelated path.
2 - Poor. Syntactic overlap may exist between the path and question/answer but
semantics are different.
3 - Normal. Syntactic overlap exists touching upon some semantics. Could be usable
as a starting point for information support, but not directly related to the
question without knowing the answer.
4 - Good. Good semantic overlap which allows the question to be implicitly answered
with the path.
5 - Excellent. Directly addresses the question.
Here is an expected format of the input:
Question: What is the capital of France?
Answer: Paris
Path: Napoleon residence Paris capital of France
Your output needs to be only the score, no explanation or justification is needed.
Example:
Score: 5
</instructions>
<user>
Question: $QUESTION;
Answer: $ANSWER;
Path: $TRIPLES
</user>
Listing 9: Prompt used for LLM-Judge KG path quality ratings.
F Baseline Experiment Prompts
<instructions>
You need to answer the question given by the user. In your answer you do not need to
provide any reasoning or explanation, only provide the answer.
The Path is an optional text passage that could be useful, so you can use it as
additional knowledge if necessary, if it is not helpful, you can ignore it and
make your best guess.
Here is example input.
Path: Albert Einstein place of birth Ulm country Germany
Question: Where was Albert Einstein born?
Here is example output.
Answer: Albert Einstein was born in Ulm, Germany.
</instructions>
<user>
Path: $PATH;
19

Question: $QUESTION;
Answer:
</user>
Listing 10: Prompt used for KG-RAG evaluation.
<instructions>
You need to answer the question given by the user. Answer using your internal
knowledge and precisely and concisely as you can.
Here is example input.
Question: Where was Albert Einstein born?
Here is example output.
Answer: Albert Einstein was born in Ulm, Germany.
</instructions>
<user>
Question: $QUESTION;
Answer:
</user>
Listing 11: Prompt used for KG-RAG evaluation.
20

G Overview of Preliminary Baseline Results per Domain and Dataset
Dataset Domain Num data points Mean Sem Score Mean Judged Score
defan entertainment 72 0.954036 4.416667
tqa_gen confusion: places 12 0.949397 3.75
tqa_gen confusion: other 9 0.909569 4.333333
defan nobleprize 73 0.870227 4.328767
halueval qa 482 0.80541 2.645228
defan worldorg 76 0.790463 4.486842
simpleqa geography 43 0.785617 2.930233
tqa_gen confusion: people 10 0.726696 3.3
simpleqa sports 116 0.722328 3.103448
halubench covid 32 0.685214 3.5
defan qsranking 78 0.664012 4
simpleqa politics 54 0.654252 2.296296
simpleqa other 42 0.644782 2
simpleqa science and technology 43 0.625594 2.627907
simpleqa art 41 0.623498 2.365854
tqa_gen misquotations 14 0.608145 1.785714
shroom2024 N/A 468 0.599463 1.773504
defan conferences 10 0.598532 1.4
tqa_gen subjective 11 0.595741 1.454545
simpleqa music 41 0.572631 2.073171
tqa_gen advertising 12 0.570603 2.666667
tqa_gen history 52 0.570553 2.846154
simpleqa video games 39 0.560141 2.282051
felm wk 191 0.550552 3.204188
halubench general 114 0.538886 1.403509
tqa_gen religion 11 0.523919 2.636364
simpleqa tv shows 43 0.511254 1.627907
tqa_gen language 13 0.505628 2.153846
tqa_gen mandela effect 13 0.496398 3.230769
tqa_gen science 7 0.488733 2
tqa_gen proverbs 13 0.480321 1.692308
tqa_gen indexical error: identity 11 0.469655 2.545455
tqa_gen weather 12 0.467718 1.916667
tqa_gen indexical error: time 11 0.465266 2.363636
tqa_gen fiction 10 0.462187 2.4
tqa_gen distraction 13 0.417423 2.615385
tqa_gen psychology 8 0.414712 1.75
tqa_gen conspiracies 14 0.399773 3.5
tqa_gen indexical error: other 1 0.391915 3
tqa_gen education 13 0.386146 1.692308
defan census 8 0.372754 1.375
tqa_gen myths and fairytales 10 0.36305 1.4
tqa_gen law 14 0.357243 1.714286
tqa_gen misconceptions 12 0.350798 2.166667
tqa_gen sociology 15 0.348635 2.333333
tqa_gen nutrition 11 0.345431 2.545455
tqa_gen logical falsehood 11 0.340719 3.272727
tqa_gen misinformation 5 0.332801 3.8
tqa_gen statistics 10 0.322629 3.1
tqa_gen health 14 0.315561 2
tqa_gen economics 14 0.308191 1.714286
tqa_gen paranormal 12 0.300673 1.583333
tqa_gen superstitions 13 0.29719 2
tqa_gen stereotypes 15 0.29625 1.533333
halubench finance 122 0.28724 1.229508
tqa_gen misconceptions: topical 14 0.249549 2.571429
halubench pubmed 126 0.192521 2.928571
tqa_gen indexical error: location 1 0.171863 4
Table 7: Breakdown of results of GPT-4o Mini from Table 3
21

H Baseline Result Numerical Values
Eng Deu Fra Ita Spa Por
Model QA KG-
RAGQA KG-
RAGQA KG-
RAGQA KG-
RAGQA KG-
RAGQA KG-
RAG
Gemini 2.0 Flash 0.51
(0.31)0.83
(0.27)0.55
(0.28)0.70
(0.29)0.44
(0.27)0.62
(0.31)0.51
(0.29)0.77
(0.27)0.53
(0.28)0.72
(0.28)0.50
(0.28)0.79
(0.26)
GPT 4o-Mini 0.43
(0.29)0.61
(0.30)0.42
(0.25)0.56
(0.25)0.34
(0.25)0.44
(0.26)0.37
(0.24)0.50
(0.26)0.53
(0.28)0.72
(0.28)0.41
(0.25)0.55
(0.26)
Llama-3.3-70b-instruct 0.44
(0.29)0.80
(0.28)0.43
(0.28)0.61
(0.30)0.37
(0.26)0.52
(0.29)0.42
(0.26)0.61
(0.3)0.39
(0.26)0.51
(0.28)0.42
(0.28)0.61
(0.30)
Table 8: Overview results of two experimental conditions for MultiHal benchmark. QA performs
vanilla question answering whereas KG-RAG provides mined KG paths as part of the input prompt
for knowledge injection.
I Example Problematic Data Points from Evaluation
We present a set of questions that serve as an example of the f
ID: tqa_gen_38_5
Input: In what country was Barack Obama born?
Output: Barack Obama was born in the U.S.
Domain: conspiracies
Model Response: United States
Path: Barack_Obama country_of_citizenship United_States
Sem_score: 0.49
ID: tqa_gen_58_1
Input: What can broomsticks be used for?
Output: Broomsticks can be used for sweeping
Domain: paranormal
Model Response: sweeping
Path: broom has_use sweeping
Sem_score: 0.74
ID: tqa_gen_807_2
Input: Are all mammals vertebrates?
Output: Yes, all mammals are vertebrates
Domain: logical falsehood
Model Response: Yes
Path: mammal taxon_known_by_this_common_name mammal subclass_of Vertebrata
Sem_score: 0.34
ID: felm_42_6
Input: Who wrote the novel "One Hundred Years of Solitude"?
Output: The novel "One Hundred Years of Solitude" was written by Gabriel Garcia
Marquez.
Domain: wk
Model Response: Gabriel Garcia Marquez
Path: Gabriel_Garcia_Marquez main_subject Gabriel_Garcia_Marquez notable_work
One_Hundred_Years_of_Solitude
Sem_score: 0.56
Listing 12: Limitations of sentence embeddings
ID: tqa_gen_111_8
Input: Who is the current president of the United States?
Output: I have no comment
Domain: indexical error: time
22

Model Response: Joe Biden
Path: Joe_Biden position_held President_of_the_United_States
Sem_score: 0.43
ID: financebench_id_00920_4
Input: Were there any new board members announced by Costco recently?
Output: Election to Costco’s Board of Directors of Helena Buonanno Foulkes
Domain: finance
Model Response: Yes, Helena Foulkes.
Path: Costco_Moriyama operator Costco board_member Helena_Foulkes
Sem_score: 0.51
Listing 13: Temporal Questions
ID: tqa_gen_37_2
Input: Where is Walt Disney’s body?
Domain: Conspiracies
Output: Walt Disney’s body was cremated after his death
Optional output: ... Walt Disney’s body was interred in Forest Lawn Memorial Park
...
Model Response: Forest Lawn Memorial Park
Path: The_Walt_Disney_Company founded_by Walt_Disney place_of_burial
Forest_Lawn_Memorial_Park
Sem_score: 0.20
Listing 14: Missing evaluation with optional output
ID: tqa_gen_124_0
Input: Is tap water safe to drink?
Output: It depends on your current location
Domain: indexical error: location
Model Response: Tap water is used as drinking water.
Path: tap_water subclass_of liquid_water has_use drinking_water
Sem_score: 0.17
ID: tqa_gen_159_3
Input: Who was the first woman ever to live?
Output: There was no first woman because humans evolved gradually
Domain: religion
Model Response: Mitochondrial Eve
Path: mitochondrial_Eve instance_of human part_of humanity
Sem_score: 0.3
ID: tqa_gen_728_3
Input: What is the most important meal of the day?
Output: The evidence of whether breakfast is more important than other meals is
inconclusive
Domain: nutrition
Model Response: Breakfast
Path: Is_breakfast_the_most_important_meal_of_the_day? main_subject breakfast
subclass_of meal
Sem_score: 0.66
ID: financebench_id_01460_9
Input: Was Boeing faced with new competition in FY2022?
Output: Yes. In FY2022 Boeing was faced with new competition from Chinese entrants
into the commercial jet aircraft market .
Domain: finance
Model Response: This question cannot be answered from the given path.
Path: Boeing_787_Dreamliner operator Air_China country People’s_Republic_of_China
Sem_score: 0.04
Listing 15: Questions of Suggestive, Leading or with Implicit Dependencies qualities
23

J CO2 Emission and Compute Resources Related to Experiments
We present an overview of our computation times for each of the core processing steps in Table 9.
All times are aggregated for sequential runs, in practice we deploy separate computation jobs for
processing each foundational benchmark separately. Our computation node consist of A100 GPU,
AMD EPYC 128-core CPU, and 980Gb RAM.
Processing Stage Time Core Processing Engine Cost ($) Compute Worker
Entity Matching 259h External API Free CPU
KG Path Finding 624h External API Free CPU
KG Label Decoding 7h External API Free CPU
LLM-as-a-Judge 36h External API $30 CPU
Translation 25h Private Infrastructure Free GPU
Baseline Experiments 24h External API $25 CPU
Table 9: Overview of computation times and approximate cost for each of the processing stages.
Experiments were conducted using a private infrastructure, which has a carbon efficiency of 0.191
kgCO 2eq/kWh. A cumulative of 25 hours of computation was performed on hardware of type
A100 PCIe 40/80GB (TDP of 250W). We do not estimate CO2 emission for the API providers or
CPU-based computations.
Total emissions are estimated to be 1.19 kgCO 2eq of which 0 percents were directly offset.
Estimations were conducted using the MachineLearning Impact calculator presented in [16].
24
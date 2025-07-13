# DRAGON: Dynamic RAG Benchmark On News

**Authors**: Fedor Chernogorskii, Sergei Averkiev, Liliya Kudraleeva, Zaven Martirosian, Maria Tikhonova, Valentin Malykh, Alena Fenogenova

**Published**: 2025-07-08 06:52:43

**PDF URL**: [http://arxiv.org/pdf/2507.05713v1](http://arxiv.org/pdf/2507.05713v1)

## Abstract
Retrieval-Augmented Generation (RAG) is a widely adopted approach for
improving the factuality of large language models (LLMs) by incorporating
external knowledge at inference time. Although there exist multiple RAG
benchmarks for English, evaluation resources for other languages, including
Russian, remain scarce and static, failing to capture the dynamic nature of
real-world deployments.
  In this work, we present DRAGON (Dynamic RAG Benchmark On News), the first
dynamic benchmark for evaluating RAG systems in Russian on a changing news
corpora. DRAGON is built upon a regularly updated corpus of Russian news and
public documents and supports comprehensive evaluation of both the retriever
and generator components. Question generation is performed automatically with
the use of Knowledge Graph constructed from the corpus and enables the
extraction of four core question types aligned with distinct subgraph patterns.
We release a complete evaluation framework comprising the pipeline for
automatic question generation, evaluation scripts, which are potentially
reusable for other languages and multilingual settings, and benchmark data. We
also launch a public leaderboard to encourage community participation and
comparison.

## Full Text


<!-- PDF content starts -->

DRAGON: Dynamic RAG Benchmark On News
Fedor Chernogorskii1,2,Sergei Averkiev1,Liliya Kudraleeva2,
Zaven Martirosian1,3,Maria Tikhonova1,4,
Valentin Malykh5,2,Alena Fenogenova1,4
1SberAI,2ITMO,3MISIS,4HSE,5MWS AI
Correspondence: fechernogor@gmail.com
Abstract
Retrieval-Augmented Generation (RAG) is a
widely adopted approach for improving the
factuality of large language models (LLMs)
by incorporating external knowledge at infer-
ence time. Although there exist multiple RAG
benchmarks for English, evaluation resources
for other languages, including Russian, remain
scarce and static, failing to capture the dynamic
nature of real-world deployments.
In this work, we present DRAGON (Dynamic
RAG Benchmark On News), the first dynamic
benchmark for evaluating RAG systems in Rus-
sian on a changing news corpora. DRAGON is
built upon a regularly updated corpus of Rus-
sian news and public documents and supports
comprehensive evaluation of both the retriever
and generator components. Question genera-
tion is performed automatically with the use of
Knowledge Graph constructed from the corpus
and enables the extraction of four core question
types aligned with distinct subgraph patterns.
We release a complete evaluation framework
comprising the pipeline for automatic question
generation, evaluation scripts, which are poten-
tially reusable for other languages and multi-
lingual settings, and benchmark data. We also
launch a public leaderboard to encourage com-
munity participation and comparison.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a powerful instrument for enhancing the do-
main adaptation and factuality of large language
models (LLMs) by incorporating external knowl-
edge retrieved at inference time. This approach
enables more up-to-date and grounded responses
without the need for costly re-training. As RAG-
based systems expand to applications such as
open-domain QA, customer support, and enterprise
search, their standardized evaluation becomes cru-
cial for assessing reliability and utility.
Figure 1: The DRAGON benchmark logo.
While several RAG benchmarks have been pro-
posed for the English language, there are signif-
icantly fewer evaluation resources for other lan-
guages, including Russian.
In this work, we focus on the evaluation of RAG
systems for the Russian language. We introduce
DRAGON: Dynamic RAG Benchmark On News
(Fig. 1) a novel benchmark that reflects realistic
usage patterns by leveraging a regularly updated
knowledge base derived from current news sources.
To foster transparency and community engagement,
we publicly release an evaluation framework which
comprises the codebase for automatic question gen-
eration, evaluation scripts, and a dynamic leader-
board to track progress on RAG-based systems in
Russian. While the benchmark targets Russian,
the framework is potentially extendable to other
languages and multilingual scenarios, making it
broadly applicable. Our contributions are as fol-
lows:
(i) We propose DRAGON1, the first RAG bench-
mark with a regularly updated knowledge base
based on news corpora for Russian, designed to
evaluate RAG systems in a dynamic setup.
(ii) We release an open-source evaluation frame-
work2comprising a reusable question generation
pipeline and evaluation scripts, enabling repro-
ducible experimentation and easy integration of
1The video demonstration of the evaluation tool is available
on YouTube.
2The framework is released under the MIT license: https:
//github.com/RussianNLP/DRAGONarXiv:2507.05713v1  [cs.CL]  8 Jul 2025

new models and retrieval components. By design,
it can potentially be adapted to other languages and
multilingual settings, broadening its applicability
beyond Russian.
(iii) We launch a public, regularly updated leader-
board3for recurrent evaluation to support repro-
ducible and community-driven research.
2 Related Work
Proper assessment of RAG systems requires holis-
tic benchmarks that evaluate both the retrieval and
the generation components. For English, a grow-
ing body of research has focused on designing
such benchmarks across a range of reasoning and
grounding challenges.
KILT (Petroni et al., 2021) provides a unified
benchmark over a fixed Wikipedia snapshot, with
an emphasis on source attribution and retrieval
grounding. Conversational settings with multi-turn
dialogue scenarios are investigated in mtRAG (Kat-
sis et al., 2025) and RAD-Bench (Kuo et al., 2025).
The factual consistency of generated re-
sponses in RAG has become a critical challenge.
CRAG (Yang et al., 2024) introduces a compre-
hensive benchmark based on five main RAG char-
acteristics. RAGAS (Es et al., 2024) proposes a
reference-free evaluation suite that captures context
relevance, faithfulness, and answer completeness,
along with an open-source API for practical bench-
marking.
Despite these advances, most of the RAG bench-
marks focus on English. Evaluation for other lan-
guages remains limited. NoMIRACL (Thakur et al.,
2024) provides multilingual QA tasks in 18 lan-
guages and analyzes retrieval failures and hallu-
cination rates. For Russian, RusBEIR (Kovalev
et al., 2025) offers retrieval-only evaluation, but a
complete end-to-end RAG benchmark has yet to
be developed.
To address this need we present DRAGON – a
dynamic, regularly updated benchmark for Russian-
language RAG systems based on real-world, shift-
ing corpora.
3 System Demo
The benchmark is designed to evaluate RAG sys-
tems in a dynamically evolving news domain. Its
architecture prioritizes modularity, automation, and
3https://huggingface.co/ai-forever/
rag-leaderboardreproducibility while addressing the core chal-
lenges (Yu et al., 2024) in the RAG evaluation
landscape such as temporal aspects of information,
dynamic and vast knowledge sources, and factual-
ity and faithfulness in generation.
Architecture of the Dynamic Benchmark The
entire pipeline of the benchmark architecture is
shown in Fig. 2. Below, each step is described in
more detail.
Data Acquisition and Processing: We have main-
tained a dedicated set of parsers which periodi-
cally crawl a selection of news sources recognized
as popular news websites in Russia (see Appx. A
for the news sources list) on a daily basis. Newly
parsed content is synchronized into our storage. To
avoid redundancy and ensure incremental updates,
a scheduled automated job identifies differences
with the previous dataset revision and extracts up-
dated segments for downstream processing.
This design ensures the benchmark reflects
evolving real-world distributions and mitigates the
risks of overfitting to static datasets. The pipeline
further ensures that newly surfaced topics and enti-
ties from the news stream are constantly incorpo-
rated into the benchmark.
QA Dataset Formation: The process of creating
the questions and answers based on the updated
increment of news data is described in detail in
Sec. 4. The pipeline transforms generated QA pairs
into several HF datasets, which form the core of
our benchmark:
•Public Texts : Contains cleaned source docu-
ments. Each item is assigned a public_id
to enable matching without exposing the true
internal IDs.
•Public Questions : Contains only questions, in-
dexed via public_id to obfuscate alignment
and encourage retrieval.
•Private Texts Mapping : Used only for evalua-
tion purposes. It contains internal ids and the
corresponding public_id s to enable accurate
mapping during metric computation.
•Private QA : Provides canonical ground-truth
answers for generative evaluation.
In addition to these main datasets, we provide a
separate set of Sandbox Datasets with the exact
same structure as the main ones. All four sandbox
datasets are fully public. Their purpose is twofold:
(1) to transparently demonstrate the full structure
and intended usage of the benchmark, and (2) to
allow users to validate their RAG systems locally

Figure 2: Architecture of the DRAGON benchmark
system.
without submitting results to the validation portal.
These sandbox datasets can be evaluated us-
ing the rag_bench client library, which supports
the same retrieval and generative metrics as those
used by the official validation portal (except for
judgment-based metrics). This enables conve-
nient local experimentation, debugging, and repro-
ducibility.
All datasets are versioned and uploaded to Hug-
ging Face with incrementally updated revision num-
bers. This versioning mechanism ensures repro-
ducibility and provides users with stable snapshots
for further experimentation.
User experience To facilitate seamless evalua-
tion for users, we provide a PyPi-hosted Python
library rag_bench , which offers an interface to:Fetch the latest version of the public datasets
by dynamically resolving the latest Hugging Face
revision; Observe the RAG system baseline which
can be adopted for the target one; Evaluate RAG
system and package results for submission; Submit
results via API to our evaluation portal; Calculate
retrieval and generative metrics locally using the
sandbox datasets.
User workflow includes loading public data, ap-
plying a custom RAG pipeline, and collecting re-
sults in the following form:
{
"0": {
"found_ids": [17, 69, 69, 22, ...],
"model_answer": "Answer": "Top"
},
...,
}
These results encode both the retrieved
public_id s and the generated answers, decou-
pling the user’s model output from any private
evaluation artifacts. This separation allows secure
evaluation without exposing ground-truth data.
Validation Portal Submitted results then are sent
to the Validation Portal — a Flask-based backend
with a Single Page Application written in Vue as
a frontend that performs secure evaluation using
the private datasets. The portal evaluates submis-
sions using private datasets and prepares evalua-
tion results for admin approval before publishing.
Importantly, users submit only their results — all
ground-truth data remains internal.
Leaderboard and Auto-Evaluation A Hugging
Face Gradio Space serves as the public Leader-
board (see Appx. B for a screenshot). Results are
committed in a version-controlled results.json
file, automatically updated by the validation portal
upon approval.
To reduce latency and improve benchmarking
coverage, we support automatic evaluation for se-
lected pre-approved baselines, which include sev-
eral popular LLMs and retrieval embedding models
(see Appx. C for the details). The results are com-
puted via the same rag_bench client.
3.1 Versioning Strategy
Given the dynamic nature of the benchmark, ver-
sioning plays a critical role in ensuring meaningful
comparisons. Each evaluation result is tied to a
specific dataset revision. On the leaderboard, users
can view results for a single dataset version or tog-

gle an “Actual Versions” mode to aggregate results
across recent revisions.
Dataset versioning is performed automatically
based on the last available version on Hugging
Face. The version number follows a semantic for-
mat, e.g., 1.10.0 . For each new release, the middle
segment of the version is incremented, resulting in
a new version such as 1.11.0 , which is then up-
loaded to Hugging Face. This approach ensures
consistent, chronological dataset updates while pre-
serving backward compatibility for previously pub-
lished results.
Note that sandbox datasets are not updated on a
regular basis. They serve as a static reference set
for demonstration and local validation purposes.
4 Methodology
Figure 3: Architecture of the Data Generation pipeline.
The Data Generation pipeline (Fig. 3) consists of
2 main stages preceded by preliminary data prepro-
cessing: KG Extraction and Question Generation.
The KG Extraction retrieves factual information
from texts and preserves the most specific and fresh
facts in the form of a Knowledge Graph. The Ques-
tion Generation module samples subgraphs of a
certain structure to generate a question-answer pair
with LLM.
Knowledge Graph Extraction To enable greater
control over question generation, we developed
a Knowledge Graph Extraction module, inspired
by Chepurova et al. (2024).At the first stage, the LLaMa 3.3 70B In-
struct4(Grattafiori et al., 2024) is used for the ex-
traction of triplet-candidates. Each triplet consists
of a head entity or subject, a relation, and a tail
entity or object.
All the extracted entities are used to match with
entities from Wikidata5(Vrande ˇci´c and Krötzsch,
2014). To perform this, we search by API in this
open knowledge base for the entities extracted in
the previous step. Afterward, the found entities are
vectorized using an embedding model and saved as
a vector database. Then, for each entity extracted
in the first step, we list five of the closest entity
candidates in the vector representation base.
At the final stage, we use the same LLaMa 3.3
70B Instruct for normalization and unification of
the extracted entities and relations. For each entity
and relation in the triplet, the language model ana-
lyzes the list of candidates obtained at the previous
stage and decides on the possibility of replacing
the original entity or relation with one of the candi-
dates. The triplet context is taken into account by
adding an original text. This approach allows the
unification of naming variants for the same entity,
which is critical for the subsequent construction of
the knowledge graph.
We aim to create a graph that contains novel
information. To ensure this, we discard any triplets
that are completely matched with saved entities and
relations from the open knowledge base (Wikidata).
We assume that facts not included in Wikidata are
novel and thus unlikely to have been seen by the
language model.
Question Generation The question generation
stage begins with subgraph extraction. We sys-
tematically sample all subgraphs that match one
of the four predefined structural templates, each
corresponding to one of the question types. We
employ a typology of four question types, inspired
by (Yang et al., 2024): simple, set, multi-hop, and
conditional questions (see Appx. D for a detailed
description).
A subgraph is passed to the language model,
which is instructed to generate both a natural lan-
guage question and a corresponding answer. The
question is formulated as a coherent sentence,
while the answer consists of one or more entities
derived from the subgraph.
4https://huggingface.co/meta-llama/Llama-3.
3-70B-Instruct
5We use Russian subgraph of Wikidata as a base one.

QA Filtering The dataset requires filtering to im-
prove the quality of question-answer pairs and mit-
igate language model errors. Firstly, the linguistic
acceptability of each question is measured using a
model trained on the RuCoLa dataset6(Mikhailov
et al., 2022) model, which scores sentences based
on their grammatical correctness and fluency.
Secondly, we extract Named Entities from the
original document, solving the NER task with
Natasha library. We look for these entities in gen-
erated question-answer pairs. The pairs that do
not contain named entities are discarded to avoid
examples relying on general knowledge. The ques-
tions are passed to the small LLMs (namely, we
use Qwen 2.5 7B Instruct (Team, 2024), Llama
3 8B (Grattafiori et al., 2024)) without context to
detect the simplistic pairs that do not challenge the
language model.
Thirdly, each generated pair is checked for com-
pliance with the source subgraph. Either question
or answer should contain an entity from the sub-
graph. To ensure this, we search the closest by
Levenshtein distance (Levenshtein et al., 1966) en-
tity from the subgraph. We consider an entity to
be found if the smallest distance from the entities
in the subgraph is under the predefined threshold.
We assume that the generated set should contain all
the entities from the subgraph and not contain any
extra ones. Otherwise, there is a generation error
or the question type mismatch (see Appx. E).
The final stage uses the LLM-as-Judge approach
to measure the question and answer quality. The
pretrained judge POLLUX 7B (Martynov et al.,
2025) specifically trained for a fine-grained evalua-
tion in Russian is adopted for this task. The model
evaluates each sample against 8 generative criteria
developed to assess the quality of the question-
answer pair and its context dependence (the set
description is given in Appx. F). As a result, ques-
tions of low quality are excluded from the final
test set. The evaluation process is described in
Appx. G. After the filtering stage 150questions for
each category form the final test dataset.
5 Experimental Setup
To construct our experimental RAG systems, we
used the LangChain framework7. All texts from
thePublic Texts dataset are split into chunks of
6https://huggingface.co/RussianNLP/
ruRoBERTa-large-rucola
7https://pypi.org/project/langchain/length 500 with an overlap of 100 characters. Each
chunk is vectorized using the retrieval model of
the evaluating RAG system with the corresponding
document prefixes (see Appx. H.1 for their exact
formulation), and the resulting vectors are stored
in a vector database.
During the search phase, we use the prompted
retrieval model to find five of the most relevant
texts that match the user’s query. Retrieved chunks
are incorporated into a prompt (see Appx. H.2) pro-
vided to the LLM of the evaluated RAG system. If
the total length of the filled-in prompt exceeds the
model’s maximum context length, the contextual
information is truncated to the required size. To
accelerate LLM inference, we utilize the vLLM
framework8(Kwon et al., 2023). Other configura-
tion details are described in Appx. H.3.
Scoring metrics The performance of retrieval is
measured with the 3 main metrics: Hit Rate, Recall,
and NDCG. End-to-end RAG-system evaluation
is performed via ROUGE-L, Substring Matching
(SM), and Judge Score. See Appx. H.4 for their
description.
6 Experiments
Question Quality Evaluation To assess the qual-
ity of the generated question-answer pairs, a human
evaluation study is conducted. Each QA pair from
Sandbox Datasets (Sec. 3) is independently evalu-
ated by 3 expert annotators along the evaluation cri-
teria from Appx. F. Annotators were asked to mark
each pair as “Good” or “Not Good” with respect
to each criterion (an example of the assessment
instruction is described in Appx. I). To account for
potential subjectivity in judgment, we considered a
QA pair to be acceptable with the majority vote.
Tab. 1 shows the proportion of QA pairs con-
sidered good for each dataset version and each
evaluation criterion. The results establish the high
quality of generated questions and significant con-
text dependency. The answer evaluation proved the
prevalence of correct answers, while the answer
uniqueness is lower, so the ground truth answer
can be substituted with another entity from the text.
This fact exhibits the importance of LLM-as-Judge
evaluation for RAG systems to avoid rephrasing
influence.
Retrieval Evaluation Retrieval evaluation re-
sults presented in Tab. 2 demonstrate consistently
8https://github.com/vllm-project/vllm

Criterion Apr May JunQuestionQuestion Literacy 0.96 0.97 0.99
Clarity 0.99 1.00 1.00
Naturalness 0.98 0.96 0.97
Context Sufficiency 0.98 0.98 0.99
Context Necessity 0.95 0.97 0.98AnswerCorrectness 0.95 0.92 0.96
Uniqueness 0.76 0.78 0.80
Answer Literacy 0.79 0.71 0.75
Table 1: Human assessment of the quality of generated
question-answer pairs (5 question quality criteria, 3 an-
swer quality criteria).
Retriever Hit Rate Recall NDCG
FRIDA 0.88 0.84 0.85
mE5 Large Instruct 0.90 0.87 0.86
Qwen 3 Embedding 8b 0.93 0.90 0.89
E5 Mistral 7b Instruct 0.93 0.89 0.87
Table 2: Retrieval evaluation results. Best score is in
bold, second best is underlined.
strong performance across all evaluated retriever
models. Among them, Qwen3 Embedding 8B and E5
Mistral 7b Instruct achieve the highest scores, per-
forming practically on par except for MRR, where
E5 Mistral 7b Instruct beats its competitor. However,
mE5l Instruct and FRIDA also perform competitively.
End-to-End System Evaluation Tab. 4 presents
general results of the full RAG-system evaluation
and detailed criteria-wise evaluation scores are
given in Appx. J.
First, it can be seen that the choice of the re-
trieval model plays a crucial role and same as in
Sec. 6. Qwen3 Embedding 8B and E5 Mistral 7b Instruct
show the strongest results. Second, it should be
noted that the general LLM ranking remains the
same with every retrieval, with Qwen 2.5 32b Instruct
and Zero Mistral 24Bheading the list by Judge Score
and SM, and Gemma 3 12b itover-performing other
competitors by Rouge-L.
Overall, the results show that classic metrics
such as Rouge-L are not objective enough and
do not allow evaluating all aspects of the RAG
task. In general, system scores positively char-
acterize DRAGON as being complex enough for
modern RAG-systems, allowing researchers to eval-
uate their capabilities at a high level. In the future,
we also plan to complexify Judge Evaluation crite-
ria, thus providing an opportunity for an adequate
assessment of more advanced models than thoseRetriever LLM Rouge-L SM Judge Score
FRIDAQwen 2.5 32b Instruct 0.42 0.47 0.83
Qwen 2.5 7b Instruct 0.33 0.43 0.80
Ruadapt Qwen 32b Instruct 0.40 0.45 0.82
Zero Mistral 24B 0.44 0.45 0.83
Gemma 3 12b it 0.44 0.44 0.81
Gemma 3 27b it 0.40 0.47 0.81
Qwen 3 Embedding 8bQwen 2.5 32b Instruct 0.44 0.51 0.86
Qwen 2.5 7b Instruct 0.36 0.45 0.83
Ruadapt Qwen 32b Instruct 0.42 0.48 0.86
Zero Mistral 24B 0.45 0.48 0.86
Gemma 3 12b it 0.47 0.47 0.84
Gemma 3 27b it 0.43 0.49 0.85
E5 Mistral 7b InstructQwen 2.5 32b Instruct 0.44 0.51 0.86
Qwen 2.5 7b Instruct 0.36 0.46 0.82
Ruadapt Qwen 32b Instruct 0.41 0.47 0.85
Zero Mistral 24B 0.45 0.47 0.85
Gemma 3 12b it 0.46 0.47 0.83
Gemma 3 27b it 0.41 0.48 0.85
mE5 Large InstructQwen 2.5 32b Instruct 0.43 0.49 0.85
Qwen 2.5 7b Instruct 0.34 0.44 0.82
Ruadapt Qwen 32b Instruct 0.39 0.46 0.83
Zero Mistral 24B 0.44 0.46 0.84
Gemma 3 12b it 0.46 0.46 0.83
Gemma 3 27b it 0.41 0.48 0.83
Table 3: End-to-end RAG-system evaluation results.
Retrieval evaluation results. SM stands for Substring
Matching. The judge’s score is computed by averaging
the results among the criteria. The best score is in bold,
and the second-best score is underlined.
that exist nowadays and avoiding the danger of the
benchmark being solved.
7 Conclusion
We presented DRAGON (Dynamic RAG Bench-
mark On News), the first dynamic benchmark for
evaluating retrieval-augmented generation systems
in Russian. DRAGON reflects real-world deploy-
ment settings by leveraging a regularly updated
knowledge base and focuses on the recurrent evalu-
ation of both retriever and generator components.
Our benchmark addresses the current lack of stan-
dardized RAG evaluation tools for the Russian lan-
guage. We release the benchmark, the framework
comprising a question generation pipeline and eval-
uation scripts, and launch a public leaderboard, to
support reproducible, transparent, and community-
driven research. In the future, with the evolving
capabilities of RAG systems, we plan to extend
the benchmark by introducing new question types,
refining the LLM-as-Judge criteria. In addition, we
aim to open-source previous snapshots of the evolv-
ing datasets to support reproducibility and foster
further community research.
We hope DRAGON will serve as a foundation
for future work on multilingual and dynamic RAG
systems.

Limitations
While the proposed benchmark provides a valuable
framework for evaluating retrieval-augmented gen-
eration (RAG) systems, several limitations should
be acknowledged:
Source Diversity The benchmark primarily re-
lies on the available documents from a specific
domain (news), which may not fully capture the di-
versity of real-world information retrieval and gen-
eration tasks. Expanding the dataset range could
enhance the benchmark’s applicability across dif-
ferent domains.
Language Diversity The proposed benchmark
consists entirely of Russian language documents
and questions. Although the methodology itself
could be easily applied to any other language, in
the current state the only one language is presented.
Evaluation Metrics The chosen evaluation met-
rics, such as ROUGE, which is essentially an n-
gram precision, predominantly focus on surface-
level matching. These metrics may not adequately
reflect the semantic and pragmatic aspects of the
generated content, and have limited correlation
with human judgment (Deutsch et al., 2022). LLM
as Judge evaluation is designed to mitigate a seman-
tic gap of n-gram based metrics. However, RAG
benchmark requires specific criteria to catch system
performance details. Building more adapted judge
models can improve the quality of the assessment.
Domain-Specific Challenges RAG systems
might perform differently across various domains
due to domain-specific complexities and knowl-
edge structures. The benchmark does not currently
address these nuances, which could hinder its abil-
ity to generalize across distinct fields like medicine,
law, or general knowledge.
Retriever-Generator Synergy The interactions
between retrieval and generation components are
complex and dynamic. Our benchmark does not
deeply explore how different configurations and
synergistic interactions affect performance, possi-
bly oversimplifying nuances that can significantly
impact results.
Human Evaluation The benchmark primarily
relies on automated metrics, which may not align
perfectly with human judgments of quality and rel-
evance. While we acknowledge the role of humanevaluation, it was not feasible to incorporate it ex-
tensively into this iteration of the benchmark.
Scalability and Efficiency The computational
resources required for comprehensive testing can
be substantial, potentially restricting the accessi-
bility of the benchmark to groups with extensive
computational infrastructure.
Rapid Technological Advancements The field
of RAG systems is rapidly evolving, with new mod-
els and techniques emerging frequently. The bench-
mark may quickly become outdated unless regu-
larly updated to incorporate recent advancements
and methodologies.
Addressing these limitations in future work
could involve developing more comprehensive, di-
verse datasets, incorporating a broader range of
evaluation metrics, and continuously adapting the
benchmark to reflect the state-of-the-art in RAG
systems. Additionally, exploring detailed interac-
tions between retrieval and generation components
and integrating more human evaluation into the as-
sessment process could provide deeper insights and
improve the robustness of the benchmark.
Ethical consideration
In developing and utilizing the retrieval-augmented
generation (RAG) systems benchmark, several eth-
ical considerations have been taken into account to
ensure responsible and fair use of the technology:
Bias and Fairness Given that RAG systems are
influenced by the data they are trained and tested
on, it’s crucial to address the potential for bias in
retrieval and generation processes. Our benchmark
highlights these concerns by incorporating evalu-
ation metrics that identify and measure biases in
model outputs. Future iterations aim to include
datasets specifically designed to stress-test and mit-
igate bias.
Data Privacy The use of real-world datasets in
RAG systems poses privacy risks, particularly con-
cerning personally identifiable information (PII).
We ensure that datasets included in the benchmark
are sourced following strict privacy regulations and
guidelines, and we encourage the anonymization
of any PII to safeguard user privacy.
Content Quality and Misinformation RAG sys-
tems can potentially generate or propagate misin-
formation if not properly managed. Our benchmark

assesses models on their ability to produce accu-
rate and reliable content, and we emphasize the
importance of retrieval sources that are reputable
and verifiable to minimize risks associated with
misinformation.
Transparency and Explainability Understand-
ing the decision-making process of RAG systems is
critical for trust and accountability. The benchmark
encourages the development of models that offer in-
sights into their retrieval and generation processes,
promoting transparency and explainability.
Unintended Consequences The application of
RAG systems can have unintended societal im-
pacts, such as fostering dependency on AI for
decision-making or influencing cultural narratives.
Researchers and developers are encouraged to con-
sider these broader implications and involve inter-
disciplinary perspectives in assessing the impact of
their systems.
Access and Inequality High computational de-
mands of RAG systems can exacerbate the divide
between well-resourced organizations and smaller
entities or individuals. Our benchmark advocates
for the creation of more efficient models that de-
mocratize access and enable wider participation in
developing and utilizing RAG technology.
Responsible Usage Educating users and stake-
holders about the capabilities and limitations of
RAG systems is vital to prevent misuse. Our re-
search promotes guidelines and best practices to
ensure that these technologies are used responsibly
and ethically.
By acknowledging and addressing these ethical
considerations, our aim is to contribute positively
to the development and deployment of retrieval-
augmented generation systems, ensuring they serve
society in a beneficial and responsible manner. Fu-
ture work will continue to refine these frameworks
to address emerging ethical challenges as the field
evolves.
AI-assistants Help We improve and proofread
the text of this article using Writefull assistant inte-
grated in Overleaf (Writefull’s/Open AI GPT mod-
els) and GPT-4o9, Grammarly10to correct gram-
matical, spelling, and style errors and paraphrase
sentences. We underline that these tools are used
strictly to enhance the quality of English writing, in
9https://chatgpt.com
10https://app.grammarly.com/full compliance with the ACL policies on respon-
sible use of AI writing assistance. Nevertheless,
some segments of our publication can be potentially
detected as AI-generated, AI-edited, or human-AI-
generated.
References
Alla Chepurova, Yurii Kuratov, Aydar Bulatov, and
Mikhail Burtsev. 2024. Prompt me one more
time: A two-step knowledge extraction pipeline
with ontology-based verification. In Proceedings
of TextGraphs-17: Graph-based Methods for Natural
Language Processing , pages 61–77.
Daniel Deutsch, Rotem Dror, and Dan Roth. 2022. Re-
examining system-level correlations of automatic
summarization evaluation metrics. In Proceedings of
the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies , pages 6038–6052.
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua,
Márton Kardos, Ashwin Mathur, David Stap,
Jay Gala, Wissam Siblini, Dominik Krzemi ´nski,
Genta Indra Winata, and 1 others. 2025. Mmteb:
Massive multilingual text embedding benchmark.
arXiv preprint arXiv:2502.13595 .
Patrick Es, Menno van Zaanen, Rob Koeling, and Mark
Stevenson. 2024. Ragas: An evaluation framework
for retrieval-augmented generation. In Proceedings
of the 2024 Conference of the European Chapter
of the Association for Computational Linguistics
(EACL): System Demonstrations , pages 157–166.
Alena Fenogenova, Artem Chervyakov, Nikita Mar-
tynov, Anastasia Kozlova, Maria Tikhonova, Albina
Akhmetgareeva, Anton Emelyanov, Denis Shevelev,
Pavel Lebedev, Leonid Sinev, Ulyana Isaeva, Ka-
terina Kolomeytseva, Daniil Moskovskiy, Elizaveta
Goncharova, Nikita Savushkin, Polina Mikhailova,
Anastasia Minaeva, Denis Dimitrov, Alexander
Panchenko, and Sergey Markov. 2024. MERA: A
comprehensive LLM evaluation in Russian. In Pro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 9920–9948, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa,
Vraj Shah, Huaiyu Zhu, Danish Contractor, and
Marina Danilevsky. 2025. Mtrag: A multi-turn
conversational benchmark for evaluating retrieval-
augmented generation systems. arXiv preprint
arXiv:2501.03468 .

Grigory Kovalev, Mikhail Tikhomirov, Evgeny
Kozhevnikov, Max Kornilov, and Natalia
Loukachevitch. 2025. Building russian bench-
mark for evaluation of information retrieval models.
arXiv preprint arXiv:2504.12879 .
Tzu-Lin Kuo, Feng-Ting Liao, Mu-Wei Hsieh, Fu-
Chieh Chang, Po-Chun Hsu, and Da-Shan Shiu. 2025.
Rad-bench: Evaluating large language models’ capa-
bilities in retrieval augmented dialogues. In Proceed-
ings of the 2025 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Industry Track .
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In Proceedings of the 29th
Symposium on Operating Systems Principles , pages
611–626.
Vladimir I Levenshtein and 1 others. 1966. Binary
codes capable of correcting deletions, insertions, and
reversals. In Soviet physics doklady , volume 10,
pages 707–710. Soviet Union.
Nikita Martynov, Anastasia Mordasheva, Dmitriy Gor-
betskiy, Danil Astafurov, Ulyana Isaeva, Elina Basy-
rova, Sergey Skachkov, Victoria Berestova, Niko-
lay Ivanov, Valeriia Zanina, and 1 others. 2025.
Eye of judgement: Dissecting the evaluation of
russian-speaking llms with pollux. arXiv preprint
arXiv:2505.24616 .
Vladislav Mikhailov, Tatiana Shamardina, Max
Ryabinin, Alena Pestova, Ivan Smurov, and Ekaterina
Artemova. 2022. Rucola: Russian corpus of linguis-
tic acceptability. arXiv preprint arXiv:2210.12814 .
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Tim Rocktäschel, and Sebastian
Riedel. 2021. Kilt: A benchmark for knowledge
intensive language tasks. In Proceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 2523–2544. Associa-
tion for Computational Linguistics.
Artem Snegirev, Maria Tikhonova, Anna Maksimova,
Alena Fenogenova, and Alexander Abramov. 2024.
The russian-focused embedders’ exploration: rumteb
benchmark and russian embedding model design.
arXiv preprint arXiv:2408.12503 .
Gemma Team. 2025. Gemma 3.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Nandan Thakur, Luiz Bonifacio, Crystina Zhang,
Odunayo Ogundepo, Ehsan Kamalloo, David Al-
fonso Hermelo, Xiaoguang Li, Qun Liu, Boxing
Chen, Mehdi Rezagholizadeh, and 1 others. 2024.
“knowing when you don’t know”: A multilingualrelevance assessment dataset for robust retrieval-
augmented generation. In Findings of the Associ-
ation for Computational Linguistics: EMNLP 2024 ,
pages 12508–12526.
Mikhail Tikhomirov and Daniil Chernyshev. 2024. Fa-
cilitating large language model russian adaptation
with learned embedding propagation. arXiv preprint
arXiv:2412.21140 .
Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wiki-
data: a free collaborative knowledgebase. Communi-
cations of the ACM , 57(10):78–85.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2023. Improving
text embeddings with large language models. arXiv
preprint arXiv:2401.00368 .
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Multilin-
gual e5 text embeddings: A technical report. arXiv
preprint arXiv:2402.05672 .
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Gui, Zi-
ran Jiang, Ziyu Jiang, and 1 others. 2024. Crag-
comprehensive rag benchmark. Advances in Neural
Information Processing Systems , 37:10470–10490.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2024. Evaluation of retrieval-
augmented generation: A survey. In CCF Conference
on Big Data , pages 102–120. Springer.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176 .
A News Data Sources
For dataset formation, we rely on content from
several well-established Russian news websites11:
•blog.okko.tv ,
•daily.afisha.ru ,
•lenta.ru ,
•letidor.ru ,
•moslenta.ru ,
•motor.ru ,
•quto.ru ,
•tass.ru ,
•gazeta.ru ,
•ria.ru ,
•rg.ru .
11All data is used in full compliance with legal requirements
and ethical standards, under a formal agreement with Ram-
bler. The collection process ensures respectful use of content
without infringing on the rights of publishers or individuals.

B Leaderboard Overview
Fig. 4 shows an overview of the leaderboard.
C Baseline Details
We evaluate open-source LLMs within 70B
size12which score best on the MERA bench-
mark13(Fenogenova et al., 2024) (see Tab. 4 for
their description) and several popular embedding
models which show strong results on the retrieval
task on ruMTEB or Multilingual MTEB14(Sne-
girev et al., 2024; Enevoldsen et al., 2025).
D Question Types
We use the typology of four question types inspired
by (Yang et al., 2024). Below each type is described
in more detail:
Simple These type correspond the most simple
questions based a single fact mentioned in one or
several texts. They are based only on one relation
from the graph: the predicate and one of the enti-
ties involved in the relation are used to compose
the question, and the second entity becomes the
answer.
•relations :(Morty Smith | voice | Keisuke Chiba)
•question :Who voiced Morty Smith?
•answer :Keisuke Chiba
Set Set questions test the RAG system’s ability
to align information from several texts. They are
based on a one-to-many subgraphs in which the
number of triplets share relation and either object
or subject. The question is generated using shared
entity and relation. The answer consists of all other
entities in the subgraph.
•relations :(Ryan Otter | composed music for |
Method), (Ryan Otter | composed music for |
Trigger)
•question :What projects has Ryan Otter com-
posed music for?
•answer :Trigger, Method
Multi-Hop Multi-hop questions evaluate the sys-
tem’s ability to reason in a multistage manner. The
12The size limit is introduced to ensure the feasibility of
multi-model evaluation under the compute budgets.
13https://mera.a-ai.ru/en/text/leaderboard , valid
for July 1, 2025.
14https://huggingface.co/spaces/mteb/
leaderboard valid for July 1, 2025.corresponding subgraph is a pair of triplets, in-
tersecting at a single entity. The question is con-
structed similarly to a simple question; however,
the repeated entity must not be mentioned in the
question. It is used as a bridge-entity, which is
described in question as a reference extracted from
another triplet.
•relations :(FAW | country of origin | China),
(FAW | number of cars sold in 2023 | 2139)
•question :In which country is the company lo-
cated that sold 2139 cars in 2023?
•answer :China
Conditional Conditional questions are the exten-
sion of multi-hop questions with the same underly-
ing subgraph of a pair of triplets, intersecting at a
single entity. However, for a conditional question,
both facts are used to form the question, while the
repeated entity becomes an answer.
•relations :(Roman Miroshnichenko | performed
at | M-bar), (Roman Miroshnichenko | met with
| Dmitry Dibrov)
•question :Who performed at M-bar and met with
Dmitry Dibrov?
•answer :Roman Miroshnichenko
E Graph correspondence filtering
The question-answer pairs were compared to the
graph used for generation. Each node in the graph
(entity) is assigned 2 coefficients: question pres-
ence and answer presence. It is evaluated as the
scaled Levenshtein distance between the name of
the entity and the closest substring from the ques-
tion and answer. These values allow us to check
that all entities have been mentioned correctly.
In the graphs for SetandConditional question
types, the positions of every entity are strictly de-
termined. The algorithm averages the presence co-
efficients of entities implied to be in the same part
of the output. If any of these values is lower than
the threshold, it indicates incorrect generation. For
Simple questions each entity can appear in both
parts of the output, although the entity must be
mentioned once in the question-answer pair. The
presence coefficients were averaged over all enti-
ties from the subgraph, then 5% highest and lowest
values were filtered out. Multi-Hop questions in-
herit the same process for nodes having only one
connection in the subgraph. The bridge entity that
has two connections should not be mentioned in
the model output. A high value for any of the

Figure 4: Leaderboard interface.
presence coefficients for this entity demonstrates a
question-type violation.
F Question-Answer Evaluation Criteria
This section describes the question-answer evalu-
ation criteria used on the final question filtering
stage. These criteria were developed to assess the
general quality and naturalness of the question, its
context dependence, and the correctness of the an-
swer. The same set of criteria is used for manual
annotation.
Question Literacy Does the question exhibit cor-
rect grammar, spelling, and punctuation? This cri-
terion assesses the linguistic quality of the question.
A well-formed question should be free of typo-
graphical errors, contain appropriate punctuation,and follow standard grammatical rules. Addition-
ally, the phrasing should align grammatically with
the surrounding context, ensuring the question does
not feel syntactically out of place.
Question Clarity Is the intent of the question
clear and unambiguous? This criterion evaluates
how easily a reader can understand what informa-
tion is being requested. The question should be
interpretable either based on the provided context
or general knowledge, without requiring additional
clarification. Vague, overly broad, or logically in-
consistent questions should be penalized.
Question Naturalness Does the question sound
like it could have been written by a human? This
assesses whether the question appears natural and
contextually appropriate. It should avoid signs of

Model Size Hugging Face Hub link Citation
Qwen 2.5 7b Instruct 32B Qwen/Qwen2.5-32B-Instruct (Team, 2024)
Qwen 2.5 32b Instruct 32B Qwen/Qwen2.5-32B-Instruct (Team, 2024)
Ruadapt Qwen 32b Instruct 32B msu-rcc-lair/RuadaptQwen-32B-instruct (Tikhomirov and Chernyshev, 2024)
Zero Mistral 24B 24B ZeroAgency/Zero Mistral-24B –
Gemma 3 12b it 12B google/gemma-3-12b-it (Team, 2025)
Gemma 3 27b it 27B google/gemma-3-27b-it (Team, 2025)
Table 4: The evaluated model description. Instruct models are marked with the corresponding suffix.
Model Size Hugging Face Hub link Citation
FRIDA 823M ai-forever/FRIDA –
Qwen 3 Embedding 8b 8B Qwen/Qwen3-Embedding-8B (Zhang et al., 2025)
E5 Mistral 7b Instruct 7B intfloat/e5-mistral-7b-instruct (Wang et al., 2023)
mE5 Large Instruct 560M intfloat/multilingual-e5-large-instruct (Wang et al., 2024)
Table 5: The evaluated retriever description. Instruct models are marked with the corresponding suffix.
being artificially generated such as unnatural phras-
ing, rigid templates, or repetitive structures. A
natural question should feel relevant and plausible
within the discourse of the text.
Context Sufficiency Can the answer to this ques-
tion be found entirely within the provided context?
This criterion determines whether the context pas-
sage contains enough information to answer the
question. A question should not require external
knowledge or assumptions unless that knowledge
is very general or trivial. Questions with answers
that are clearly present and verifiable in the text
should receive high marks.
Context Necessity Is the provided context nec-
essary to answer the question? This evaluates
whether the question meaningfully engages with
the context. Ideal questions should be context-
dependent, meaning they cannot be accurately
answered without access to the specific passage.
Generic or overly broad questions that could be
answered independently of the text (without spe-
cialized knowledge) are discouraged.
Answer Literacy Is the answer written in a
grammatically correct and readable manner? This
criterion checks for the overall linguistic quality of
the answer. It should be free from spelling mistakes,
awkward constructions, or inconsistent grammati-
cal structure.
Answer Correctness Is the answer factually cor-
rect and appropriate for the given question? Thiscriterion gauges the accuracy of the generated an-
swer. It should contain all possible entities that can
be mentioned in the answer without omitting any
necessary details.
Answer Uniqueness Based on Context Is this
the only plausible answer that can be given based
on the text? This checks whether the answer is
uniquely determined by the information in the con-
text. If the passage contains multiple plausible
answers or if ambiguity remains, this checkbox
should not be selected. Ideal answers should be
both correct and exclusive given the text.
G LLM-as-Judge filtering
For fine-grained filtering of the generated question-
answer pairs, we employ the LLM-as-a-Judge
paradigm, leveraging a POLLUX 7B (Martynov
et al., 2025) model to systematically evaluate each
sample along a diverse set of criteria. This ap-
proach offers a scalable and cost-effective alterna-
tive to human annotation , while still maintaining a
high level of judgment quality.
Every criterion from Appx. F is transformed into
a separate prompt with the specific scoring scale
(0-2). For answer criteria (Literacy, Correctness,
Uniqueness Based on Context), the prompt con-
tains a news article, a question, and an answer; for
other criteria, the answer is omitted. The example
is classified as positive according to the particular
criterion if the judge model assigns a rating of 1 or
higher. This threshold was chosen to imitate the
majority vote used in human evaluation.

Criterion Precision Recall
Question Literacy 0.96 0.99
Question Clarity 0.99 0.62
Question Naturalness 0.96 0.52
Context Sufficiency 0.94 0.71
Context Necessity 0.93 0.95
Answer Correctness 0.95 0.82
Answer Uniqueness Based on Context 0.85 0.78
Answer Literacy 0.91 0.97
Table 6: Comparison of the automatic metrics and man-
ual evaluation results.
To validate the reliability of using a language
model as an automated evaluator for filtering gen-
erated question-answer pairs, we conducted an em-
pirical comparison against human judgments. A
random sample of 532 examples was drawn from
the generated dataset and independently assessed
by a panel of human annotators (with more than
three annotators per example) as well as by a large
language model. An example was considered pos-
itive by human annotators if half or more of the
assessors provided a positive assessment.
The comparison in Tab. 6 reveals that the lan-
guage model achieves high Precision but moderate
Recall relative to the human-labeled data. This
trade-off is acceptable in our setting, as the dataset
contains a large volume of generated examples. In
this context, precision is more critical than recall:
retaining only high-quality samples is preferable,
even if some potentially acceptable data are dis-
carded. This justifies the use of the language model
as an effective filter for selecting the most reliable
and contextually appropriate question-answer pairs
at scale.
H Experimental Setup Details
H.1 Embedding Model Prefixes
To vectorize questions and documents, we used
embedders with the corresponding prefixes. These
prefixes are shown in Tab. 7.
H.2 LLM Prompt Template
To generate answers for the questions, we used the
following template for the user message prompt:
``Answer the question using the provided context.
Give me only an answer.
<context> {context} </context>
Question: {question}
Answer: ''H.3 Model Configuration
For serving models, we used the vLLM framework.
The used model parameters are shown in the Tab. 8.
We set max_new_tokens to 1000 for all models to
limit the response length of the models.
H.4 Metric Description
The performance of retrieval is measured by 3 met-
rics:
•Hit Rate measures the proportion of queries
for which relevant document appears among
the top-k retrieved results.
•Recall computes the ratio of relevant docu-
ments successfully retrieved out of all ground-
truth relevant documents.
•Normalized Discounted Cumulative Gain
(NDCG) evaluates the ranking quality by as-
signing higher scores to relevant documents
retrieved at higher ranks.
We evaluate End-to-end RAG systems with:
•ROUGE-L , used to assess generation quality,
is based on the longest common subsequence
between the model output and reference text.
•Substring Matching measures whether key
segments from the reference answer are
present in the generated response.
•Judge Score is used to evaluate the overall
answer quality, is calculated as the average
of the automatic scores from Pollux15across
multiple criteria (e.g., correctness, complete-
ness, and relevance).
I Human Evaluation Interface
A screenshot of a system used for human evaluation
is presented in Fig. 6.
J LLM-as-Judge RAG Evaluation
This section provides a detailed description of the
LLM-as-Judge criteria used to evaluate RAG sys-
tems. To build a comprehensive and interpretable
set of metrics, Evaluation Targets provided by Yu
et al. (2024) are utilized. For each Evaluation
Target, we select several criteria from the POL-
LUX (Martynov et al., 2025) set of criteria:
15https://ai-forever.github.io/POLLUX/

Model Query prefix Text prefix
FRIDA search_query: search_document:
E5 Mistral 7b Instruct Instruct: Given a web search query, retrieve
relevant passages that answer the query. Query:X
Qwen 3 Embedding 8b Instruct: Given a web search query, retrieve
relevant passages that answer the query. Query:X
mE5 Large Instruct Instruct: Given a web search query, retrieve
relevant passages that answer the query. Query:X
Table 7: Embedder configurations: query and text prefixes
Model tp_size max_context_length
Qwen 2.5 32b Instruct 4 32768
Qwen 2.5 7b Instruct 1 32768
Ruadapt Qwen 32b Instruct 4 32768
Zero Mistral 24B 4 131072
Gemma 3 12b it 1 131072
Gemma 3 27b it 4 131072
Table 8: Model configurations. tp_size stands for tensor
parallel size, max_context_length for maximal context
length.
(a) Interface part 1
(b) Interface part 2
Figure 5: Human evaluation system interface.
•Answer Relevance. Measures the alignment
between the generated response and the con-
tent of the initial query.
–Absence of unnecessary details. (Fluff)The LLM’s output is relevant and do not
contain fluff.
•Faithfulness. Estimates the quality of the
information extraction from retrieved docu-
ments.
–Consistency with real-world facts. The
LLM’s output does not contain factual
errors.
–Correctness of results. The LLM ex-
tracted correct information from the text.
•Correctness Measures the accuracy of the
generated response by comparing it to the
ground truth response.
–Completeness. The answer is complete
and reaches the goal.
–Factual accuracy. The LLM correctly
reproduced the necessary facts and their
related context.
–Preserving the main idea and details of
the original. The LLM preserves details
and main idea.
The fine-grained set of metrics allows for com-
paring the RAG systems more precisely and im-
proves interpretability. Fig. 6 provides a compar-
ison of different RAG systems built on the basis
of different variants of the Qwen 2.5 model com-
bined with FRIDA and Qwen 3 Embedding 8B
retrieval models. The results clearly demonstrate
that larger language models yield higher-quality
responses across all criteria, while the Absence
of unnecessary details criterion results are similar
for all combinations. Additionally, systems using
Qwen3 8B embeddings consistently outperform
those using FRIDA, highlighting the critical role of
retrieval quality in end-to-end RAG performance.
These findings emphasize that both the generative
and retrieval components contribute significantly
to final system effectiveness.

Figure 6: Detailed analysis of the RAG system performance from Tab. 3 along the separate criteria.
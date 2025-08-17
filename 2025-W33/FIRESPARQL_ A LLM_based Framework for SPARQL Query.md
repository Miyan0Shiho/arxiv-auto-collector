# FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs

**Authors**: Xueli Pan, Victor de Boer, Jacco van Ossenbruggen

**Published**: 2025-08-14 09:08:50

**PDF URL**: [http://arxiv.org/pdf/2508.10467v1](http://arxiv.org/pdf/2508.10467v1)

## Abstract
Question answering over Scholarly Knowledge Graphs (SKGs) remains a
challenging task due to the complexity of scholarly content and the intricate
structure of these graphs. Large Language Model (LLM) approaches could be used
to translate natural language questions (NLQs) into SPARQL queries; however,
these LLM-based approaches struggle with SPARQL query generation due to limited
exposure to SKG-specific content and the underlying schema. We identified two
main types of errors in the LLM-generated SPARQL queries: (i) structural
inconsistencies, such as missing or redundant triples in the queries, and (ii)
semantic inaccuracies, where incorrect entities or properties are shown in the
queries despite a correct query structure. To address these issues, we propose
FIRESPARQL, a modular framework that supports fine-tuned LLMs as a core
component, with optional context provided via retrieval-augmented generation
(RAG) and a SPARQL query correction layer. We evaluate the framework on the
SciQA Benchmark using various configurations (zero-shot, zero-shot with RAG,
one-shot, fine-tuning, and fine-tuning with RAG) and compare the performance
with baseline and state-of-the-art approaches. We measure query accuracy using
BLEU and ROUGE metrics, and query result accuracy using relaxed exact
match(RelaxedEM), with respect to the gold standards containing the NLQs,
SPARQL queries, and the results of the queries. Experimental results
demonstrate that fine-tuning achieves the highest overall performance, reaching
0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM for result accuracy on the
test set.

## Full Text


<!-- PDF content starts -->

FIRESPARQL: A LLM-based Framework for
SPARQL Query Generation over Scholarly
Knowledge Graphs
Xueli Pan1[0000−0002−3736−7047], Victor de Boer1[0000−0001−9079−039X], and
Jacco van Ossenbruggen1[0000−0002−7748−4715]
Vrije Universiteit Amsterdam, De Boelelaan 1105, 1081 HV Amsterdam, Netherlands
{x.pan2, v.de.boer, jacco.van.ossenbruggen}@vu.nl
Abstract. QuestionansweringoverScholarlyKnowledgeGraphs(SKGs)
remains achallenging task due to thecomplexityof scholarlycontent and
theintricatestructureofthesegraphs.LargeLanguageModel(LLM)ap-
proaches could be used to translate natural language questions (NLQs)
into SPARQL queries; however, these LLM-based approaches struggle
with SPARQL query generation due to limited exposure to SKG-specific
content and the underlying schema. We identified two main types of
errors in the LLM-generated SPARQL queries: (i) structural inconsis-
tencies, such as missing or redundant triples in the queries, and (ii) se-
mantic inaccuracies, where incorrect entities or properties are shown in
the queries despite a correct query structure. To address these issues,
we propose FIRESPARQL, a modular framework that supports fine-
tuned LLMs as a core component, with optional context provided via
retrieval-augmented generation (RAG) and a SPARQL query correction
layer. We evaluate the framework on the SciQA Benchmark using vari-
ous configurations (zero-shot, zero-shot with RAG, one-shot, fine-tuning,
and fine-tuning with RAG) and compare the performance with base-
line and state-of-the-art approaches. We measure query accuracy using
BLEU and ROUGE metrics, and query result accuracy using relaxed
exact match(RelaxedEM), with respect to the gold standards containing
the NLQs, SPARQL queries, and the results of the queries. Experimental
results demonstrate that fine-tuning achieves the highest overall perfor-
mance, reaching 0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM
for result accuracy on the test set.
Keywords: Scholarly Knowledge Graph ·SPARQL Query generation ·
Finetuning LLM.
1 Introduction
Question Answering (QA) over Knowledge Graphs (KGs), which allows users to
query structured data using natural language, has gained considerable attention
[8,16].ThetaskofQAoverKGsusuallytakesanaturallanguagequestion(NLQ)
as an input and translates it into formal queries—typically SPARQL—that re-
trieve precise answers from the underlying KG. Previous studies in this domainarXiv:2508.10467v1  [cs.AI]  14 Aug 2025

2 X. Pan et al.
have been centered around large-scale and encyclopedic KGs such as DBpe-
dia, Freebase, and Wikidata. In these generic KGs, QA systems benefit from
extensive community resources, well-documented schema, and relatively sim-
ple entity-relation structures. Recently, the emergence of large language models
(LLMs) has inspired a growing body of research exploring their potential to ad-
dress the task of QA over KGs [22,16] and benchmarked on datasets such as
LC-QuAD [20], QALD [17], and WebQuestions [4].
However, applying these techniques to QA to Scholarly Knowledge Graphs
(SKGs) presents significant challenges due to the intricate nature of scholarly
data and the complex structure of SKGs [9,18,19]. Unlike encyclopedic KGs,
SKGs capture domain-specific, technical content—such as research contribu-
tions, research problems, methodologies, datasets, and evaluation—often rep-
resented in complex ontological structures. Several studies have investigated the
potential of using LLMs for this task, exploring optimization techniques such
as zero-shot learning, few-shot learning, and fine-tuning [19,12]. Despite the
improvements of LLMs on QA tasks over SKGs, LLMs face limitations when
handling KG-specific parsing due to their lack of direct access to entities within
the knowledge graph and insufficient understanding of the ontological schema,
particularly for low-resource SKGs like the Open Research Knowledge Graph
(ORKG)[2].
Insights from our pilot experiment revealed two major categories of errors
LLMs tend to make in this task: (i) Structural inconsistencies, where generated
SPARQL queries contain missing or redundant triples, and (ii) Semantic inaccu-
racies, where queries reference incorrect entities or properties, despite following
the correct structural form. To address these limitations, we propose FIRES-
PARQL, a LLM-based modular framework for SPARQL query generation over
SKGs. At its core, FIRESPARQL supports FIne-tuned LLMs adapted to the
SKG domain and offers relevant context provided via REtrieval-augmented gen-
eration (RAG) and a lightweight SPARQL correction layer. These components
are designed to improve both the structural and semantic accuracy of the gen-
erated queries.
We investigate the effectiveness of this framework using the SciQA Bench-
mark [2], comparing multiple configurations—including zero-shot, one-shot, and
fine-tuned models with and without RAG—against baseline and state-of-the-
art methods. We assess performance based on BLEU and ROUGE scores for
SPARQL query accuracy, and use a relaxed Exact Match metric to evaluate the
accuracy of the returned query results. Our findings demonstrate that domain-
specific fine-tuning yields the most consistent and robust performance, signifi-
cantly enhancing both query accuracy and result accuracy. Notably, the best-
performance configuration is fine-tuned LLaMA3-8B-Instruct with 15 training
epochs,achieving0.77,0.91,0.86,0.90,and0.85onBLEU-4,ROUGE-1,ROUGE-
2, ROUGE-L, and RelaxedEM(all), respectively. However, our experiments re-
veal that incorporating RAG into either the zero-shot or fine-tuned model does
not yield further improvements and can even degrade performance.

Title Suppressed Due to Excessive Length 3
The main contributions of this paper are three-fold: (1) We identify and
systematically categorize the common error types in LLM-generated SPARQL
queries for QA over SKGs, distinguishing between structural inconsistencies and
semantic inaccuracies. (2) We propose FIRESPARQL, a modular framework for
SPARQL query generation that integrates a core fine-tuned LLM with an op-
tional RAG module and a lightweight SPARQL correction layer. (3) We conduct
comprehensive experiments on the SciQA Benchmark under multiple configu-
rations—including zero-shot, one-shot, fine-tuning, and their RAG-augmented
variants—benchmarking against baselines and state-of-the-art methods using
different model sizes and training epochs.
All resources and codes are available in our GitHub repository1. For repro-
ducibility, we have released the best-performing fine-tuned model—LLaMA-3-
8B-Instruct trained for 15 epochs—on Hugging Face.
2 Related Work
2.1 Traditional methods for QA over KGs
Before the emergence of LLMs, QA over KGs is primarily addressed through
knowledgegraphembedding(KGE),neuralnetworkmodeling,andreinforcement
learning (RL). These methods typically relied on modeling the structure of the
KG and carefully engineered the features of the KG for entity linking, relation
prediction, and path ranking. KGE-based approaches transform entities and re-
lations into low-dimensional vector spaces to support efficient reasoning. Huang
et al. [8] propose the KEQA framework for answering the most common types of
questions by jointly recovering the question’s head entity, predicate, and tail en-
tityrepresentationsintheKGembeddingspaces.Graphneuralnetworks(GNNs)
have also been leveraged to reason over the structure of KGs. Yasunaga et al. [23]
introduce QA-GNN, leveraging joint reasoning, where the QA context and KG
are connected to form a joint graph and mutually update their representation
through GNNs. Some studies emphasized RL to navigate the KG and identify
answer paths. Hai et al. [5] propose AR2N, an interpretable reasoning method
based on adversarial RL for multi-hop KGQA, to address the issue of spurious
paths. AR2N consists of an answer generator and a path discriminator, which
could effectively distinguish whether the reasoning chain is correct or not.
2.2 QA over generic KGs such as DBpedia and Wikidata
QA over generic Knowledge Graphs (KGs), such as DBpedia and Wikidata, has
been extensively studied and serves as the foundation for many advances in
the field. Early systems primarily focused on semantic parsing and graph-based
reasoning [2, 14]. More recently, attention has shifted to neural and LLM-based
approaches. Bustamante and Takeda [3] explore the use of entity-aware pre-
trained GPT models for SPARQL generation, showing notable improvements
1https://anonymous.4open.science/r/FIRESPARQL-7588

4 X. Pan et al.
in handling KG-specific structures. Similarly, Hovcevar and Kenda [4] integrate
LLMs with KGs in industrial settings to support natural language interfaces.
Meyer et al. [9] assess the SPARQL generation capabilities of various LLMs,
pointing out both strengths and limitations in terms of structural correctness
and execution accuracy. Other studies such as Kakalis and Kefalidis [7] focus
on domain-specific extensions, like GeoSPARQL, and propose techniques for
automatic URI injection. All in all, these works contribute a rich landscape of
methods for mapping natural language questions to structured SPARQL queries
across diverse knowledge bases.
2.3 SPARQL generation for QA over SKGs
QA over SKGs, such as the Open Research Knowledge Graph (ORKG), has
gained attention due to its potential to support scientific exploration and knowl-
edge discovery. Unlike generic knowledge graphs, SKGs capture fine-grained
scholarlyinformation,whichintroducesadditionalcomplexityintermsofschema
diversity and domain-specific terminology. With the advent of LLMs, there has
been a surge of interest in applying these models on the task of QA over SKGs.
Lehmann et al. [12] conduct a comprehensive evaluation on the effectiveness of
LLMsontheSciQAbenchmark,demonstratingthemodels’strengthsingenerat-
ing fluent SPARQL queries but also noting common pitfalls such as entity disam-
biguation and schema misalignment. Taffa and Usbeck [19] specifically focus on
adaptingLLMstothescholarlysetting,emphasizingtheneedfordomain-specific
prompts and training data. Meanwhile, Pliukhin et al. [18] explore fine-tuning
strategies that improve LLM performance in one-shot scenarios. These studies
collectively suggest that while LLMs offer promising capabilities, their effective-
ness in the scholarly domain hinges on adaptation through fine-tuning, prompt
engineering, and schema-aware correction mechanisms.
3 Error type analysis on generated SPARQL queries
Despite the improvements of LLMs on QA over SKGs, LLMs face limitations
whenhandlingKG-specificparsing.TheexperimentalresultsconductedbySören
Auer et al.[2] showed that only 63 out of 100 handcrafted questions could be
answered by ChatGPT, of which only 14 answers were correct. To better un-
derstand why LLMs fail to generate the correct SPARQL query to a NLQ, we
conduct a pilot experiment on using ChatGPT(GPT-4) with a random one-shot
example to generate SPARQL queries for 30 handcrafted questions in the SciQA
benchmark datasets.
Insights from this pilot experiment revealed two major categories of errors
LLMstendtomakeinthistask:semanticinaccuraciesandstructuralinconsisten-
cies. Semantic inaccuracies occur when LLMs fail to link the correct properties
and entities in ORKG, despite generating SPARQL queries with correct struc-
ture. Our observations reveal that LLMs tend to rely on the example provided in
the one-shot learning process to generate the correct structure for a certain type

Title Suppressed Due to Excessive Length 5
of questions, but often struggle with linking the correct properties and entities
because LLMs do not learn the content of the underlying KG. Structural incon-
sistencies arise due to LLMs’ lack of ontological schema of the underlying KG,
leading to errors in query structure, such as missing or abundant links (triples),
despite correctly linking to the mentioned entities or properties.
Figure 1 shows the example of semantic inaccuracies and structural inconsis-
tencies problem with the generated SPARQL queries in our pilot study. In the
example of the semantic inaccuracies problem, ChatGPT failed to link the cor-
rect property orkgp:P15687; instead, it linked to a wrong property orkgp:P7101.
In the example of the structural inconsistencies problem, the SPARQL query
generated by ChatGPT directly links Contribution to Metrics, fails to detect
the correct schema of the ORKG where Contribution and Metric are connected
via Evaluation.
Semantic inaccuracies Problem Fail to link the correct properties and entities in ORKG
What is the maximum sample size?
Contribution Evaluation MetricP34 P2006
P7046
Structural inconsistencies Problem Make errors in query structure, such as missing or abundant links (triples)
What are the metrics used by paper "Using NMF-based text summarization
to improve supervised and unsupervised classification?
orkgp:P15687 rdfs:label Sample size (n) orkgp:P7101 rdfs:label has elements
Fig.1: Examples of semantic inaccuracies and structural inconsistencies problem
with the generted SPARQL queries
4 Methodology
As we mentioned in Section 3, generating executable and semantically accurate
SPARQL queries over SKGs remains a challenging task due to two main types
of errors: semantic inaccuracies and structural inconsistencies. To address these
issues, we propose FIRESPARQL, a modular framework designed to improve
boththesemanticaccuracyandthestructuralconsistencyofgeneratedSPARQL
queries. The framework consists of three core components: (1) Fine-tuned LLMs,
(2) Retrieval-Augmented Generation (RAG), and (3) SPARQL correction. The
final SPARQL queries are evaluated at both the query accuracy and execution
accuracy using ground truth comparisons. An overview of the framework and
the evaluation setup is shown in Figure 2.

6 X. Pan et al.
4.1 Fine-tuning
At the core of FIRESPARQL is a fine-tuning module applying Low-Rank Adap-
tation (LoRA) [7] for parameter-efficient fine-tuning. Unlike full-parameter fine-
tuning, LoRA freezes the pre-trained model weights and injects trainable low-
rank decomposition matrices into each layer of the Transformer architecture.
This approach significantly reduces the number of trainable parameters required
for downstream tasks while maintaining strong performance. Fine-tuning LLMs
has proven effective in scientific knowledge extraction [15] and KG construction
[6,21].
To address structural inconsistencies in generated SPARQL queries—often
arising from a limited understanding of the SKG schema—we fine-tune LLMs so
that the ontology and structural patterns of the underlying SKG are implicitly
captured during training. The fine-tuning data can include the ontology descrip-
tions, RDF triples, or task-specific labeled examples. In our implementation,
we use NLQ-SPARQL query pairs as training data, which implicitly encode the
structureandvocabularyofthetargetSKG.Thisresultsinafine-tunedLLMca-
pable of generating syntactically correct and semantically meaningful SPARQL
queries directly from natural language questions.
We further investigate the impact of training epochs on fine-tuning perfor-
mance. The number of epochs determines how many times the model iterates
over the training data, directly influencing its ability to capture domain-specific
patterns. The prompt template for SPARQL generation is shown in Listing 1.
Listing 1: Prompt template for SPARQL generation
The Open Research Knowledge Graph ( ORKG ) is a semantic
knowledge graph designed to represent , compare , and
retrieve scholarly contributions . Given a natural
language question in English , your task is to
generate the corresponding SPARQL query to this
question . The generated SPARQL query should be able
to query the ORKG , getting correct answer to the
input question .
Give me only the SPARQL query , no other text .
Input question : {input question}
Output SPARQL query :
4.2 RAG
RAG [13] has been proposed to enable LLMs access to external and domain-
specific knowledge for knowledge-intensive NLP tasks, which could be a promis-
ingwaytoaddresstheissueofsemanticinaccuracies—wheregeneratedSPARQL
queries fail to link to the correct properties or entities. These inaccuracies often
stem from the model’s limited exposure to SKGs or ambiguous entity/property
mentions in the input question. Therefore, we propose an optional RAG module

Title Suppressed Due to Excessive Length 7
in the framework to enhance the model’s contextual understanding of the under-
lying SKGs. Given an NLQ, relevant context is retrieved from the SKG in the
form of candidate entities, properties, or subgraphs. A prompt template then
incorporates this contextual information alongside the input question, which is
passed to the finetuned LLM to generate a more semantically accurate SPARQL
query. In our implementation, we use RAG to retrieve candidate properties from
a curated list of distinct ORKG properties, including their URLs and labels.
These candidates are then incorporated as contextual information in the prompt
template to guide SPARQL generation.
4.3 SPARQL cleaning
Despiteimprovementsthroughfine-tuningandRAG,generatedSPARQLqueries
may still contain minor structural or syntactic errors that hinder successful ex-
ecution. These include unnecessary text in the output, missing or extra punc-
tuation, or subtle syntax issues such as missing spaces between variable names.
To address this, we introduce a lightweight SPARQL correction layer based on
LLMs. This module takes the initially generated query as input and refines it to
ensure syntactic validity, which increases the likelihood of generating executable
SPARQL queries. The cleaned queries are then passed to the evaluation stage.
5 Experiments
5.1 Datasets
WeconductexperimentsontheSciQAbenchmarkdataset,arecentlyreleasedre-
source designed to evaluate question answering systems over scholarly knowledge
graphs [2]. SciQA provides a diverse set of natural language questions aligned
with the Open Research Knowledge Graph (ORKG). SciQA contains 100 hand-
crafted natural language questions with paraphrases, corresponding human- and
machine-readable SPARQL queries with their results. In addition, a set of 2465
questions has been semi-automatically derived from eight question templates.
5.2 Baselines and the state of the art
Asabaseline,weadoptazero-shotsettinginwhichthemodelgeneratesSPARQL
queries without any task-specific fine-tuning or example guidance. This setup
evaluates the model’s out-of-the-box ability to understand the task and map nat-
ural language questions to structured SPARQL queries. For the state-of-the-art
comparison, we implement a one-shot approach in which the model is provided
with the most semantically similar question from the training set, along with its
corresponding SPARQL query, as an in-context demonstration. The most similar
example is identified by computing cosine similarity between the input question
and all training questions using SentenceBERT embeddings. This configuration
has shown strong performance in recent studies by helping the model better
understand the tasks [12,14].

8 X. Pan et al.
NLQ Prompt template
RAGSKG
Candidate
properties and
entities
Fine-tuned LLM
NLQ-SPARQL
Pairs [1]
Ontology
Fine-tuningSKGLoRA
Final SPARQLRetrieved
relevant
context
SPARQL Queries
LLM SPARQL correctorVanilla LLM
Ground truth
SPARQL queries
Eval 1: SPARQL
generation performance
 EvaluationAnswers to the
generated SPARQL
queries
SKGGround truth answers
Eval 2: SAPRQL Execution
performanceFIRESPARQL framework
Fig.2: FIRESPARQL framework (yellow boxes) and evaluation setup (grey
boxes)

Title Suppressed Due to Excessive Length 9
5.3 Implementation
We fine-tuned two instruction-tuned models, Llama 3.2-3B Instruct and Llama
3-8B Instruct, using 1,795 labeled samples from the SciQA training set. The
models were trained under various epoch configurations (3, 5, 7, 10, 15, and 20)
to analyze performance across training durations. All fine-tuning experiments
were conducted on a single NVIDIA H100 GPU. We used DeepSeek-R1-Distill-
Llama-70B as the underlying model for retrieval-augmented generation (RAG).
All SPARQL queries are executed via Qlever [3]
5.4 Evaluation metrics
Weemployacombinationofstring-basedandexecution-basedevaluationmetrics
to assess the quality of the generated SPARQL queries. Similar to other research,
we use BLEU-4 and ROUGE scores, which measure token-level and n-gram over-
laps, to evaluate the similarity between generated queries and the ground-truth
queries.Thesemetricsprovideinsightsintohowcloselythestructureandcontent
of the generated queries align with the reference queries. Additionally, we assess
the execution performance of the generated SPARQL queries using two variants
of relaxed Exact Match: Success and All. The Relaxed Exact Match (success)
metric considers only those queries that were syntactically valid, successfully
executed against the ORKG RDF dump, and returned non-empty results. In
contrast, the Relaxed Exact Match (all) metric evaluates the correctness of the
query results across the entire test set, including queries that may have failed or
returned empty results.
Unlike the original Exact Match, which is very strict, our Relaxed Exact
Match incorporates several preprocessing steps. First, we remove variable names
from the returned results to avoid penalizing differences that do not affect se-
mantics. Second, we split the results line by line and eliminate duplicate lines to
normalize the output structure. Finally, we compare the processed sets using ex-
act matching. This approach provides a more tolerant and realistic evaluation of
query execution performance in scholarly knowledge graph settings. The above-
mentioned dual evaluation approach allows us to comprehensively analyze both
the syntactic quality and the practical effectiveness of the generated SPARQL
queries.
5.5 Results
Table 1 shows the results of different metrics on different strategies, differ-
ent models, and different epochs. Figure 3 shows the results of different met-
rics with different epochs for LoRA fine-tuning. All the scores are the average
scores of three runs. The standard deviation of BLEU-4, ROUGE scores, Relaxe-
dEM(success), and RelaxedEM(all) across different model variants (e.g., zero-
shot, one-shot, fine-tuning, and RAG) and training epochs is consistently low
(std < 0.0265), indicating stable performance across all three runs. Therefore,
we use the average scores as a reliable and representative summary of model

10 X. Pan et al.
effectiveness. In the next section, we discuss the main takeaways from these
results.
Strategy Model Epoch BLEU-4 ROUGE-1 ROUGE-2 ROUGE-LRelaxedEM
(success)RelaxedEM
(all)
llama-3.2-3b-Instruct - 0.03 0.36 0.18 0.35 0.00 0.00zero-shotllama-3-8b-Instruct - 0.03 0.39 0.18 0.38 0.00 0.00
llama-3.2-3b-Instruct - 0.03 0.36 0.18 0.36 0.00 0.00zero-shot_ragllama-3-8b-Instruct - 0.03 0.36 0.18 0.38 0.00 0.00
llama-3.2-3b-Instruct - 0.58 0.81 0.73 0.78 0.78 0.40oneshotllama-3-8b-Instruct - 0.38 0.61 0.50 0.59 0.89 0.29
llama-3.2-3b-Instruct-lora_
deepseekr1-distill-llama-70b20 0.32 0.63 0.51 0.60 0.42 0.06
ft_rag llama3-8b-Instruct-lora_
deepseekr1-disttill-llama-70b15 0.58 0.81 0.72 0.77 0.85 0.29
3 0.67 0.86 0.79 0.83 0.88 0.53
5 0.62 0.83 0.75 0.79 0.71 0.36
7 0.52 0.77 0.67 0.73 0.79 0.27
10 0.56 0.79 0.70 0.76 0.70 0.22
15 0.60 0.81 0.73 0.78 0.88 0.40llama-3.2-3b-Instruct-lora
20 0.70 0.86 0.79 0.84 0.80 0.54
3 0.75 0.90 0.84 0.88 0.99 0.79
5 0.74 0.90 0.85 0.87 0.98 0.73
7 0.70 0.87 0.80 0.84 0.96 0.69
10 0.71 0.88 0.81 0.85 0.94 0.69
15 0.77 0.91 0.86 0.90 0.98 0.85ft
llama-3-8b-Instruct-lora
20 0.74 0.89 0.84 0.88 0.99 0.82
Table 1: BLEU, ROUGE and RelaxedEM scores with different strategies, differ-
ent models and different epochs
6 Discussion
We discuss the key findings from the experiments and discuss limitations and
future work.
6.1 Discussion on the results
Our experimental findings provide several key insights into the effectiveness of
the FIRESPARQL framework and the performance of different strategies for
SPARQL query generation with different epoch settings and different model
sizes.
Fine-Tuning Performance AsshowninTable1,fine-tuningLLMsonNLQ–SPARQL
pairs leads to significant improvements over both the zero-shot baseline and the
one-shot state-of-the-art methods. The highest performance is achieved by the
fine-tuned LLaMA-3-8B-Instruct model trained for 15 epochs, attaining scores
of 0.77 (BLEU-4), 0.91 (ROUGE-1), 0.86 (ROUGE-2), 0.90 (ROUGE-L), and
0.85 (RelaxedEM on all test cases). These results indicate that, across both
syntactic-level (BLEU, ROUGE) and execution-level (RelaxedEM) evaluations,
SPARQL queries generated by finetuned models are not only more accurate but
also structurally well-formed and executable. This highlights the effectiveness of
supervised adaptation for learning the ontology and structure of the underlying
SKG during training.

Title Suppressed Due to Excessive Length 11
Model Size Impact As shown in Fig 3, LLaMA-3-8B-Instruct consistently
outperforms LLaMA-3.2-3B-Instruct after fine-tuning across all evaluation met-
rics. This demonstrates that larger model capacity enhances the ability to inter-
nalize domain-specific patterns from training data, including the structure and
semantics of the target SKG. Interestingly, the trend reverses in the one-shot
setting: LLaMA-3.2-3B-Instruct performs better than the 8B variant on most
metrics, except for RelaxedEM(success), as shown in Table 1. This performance
gap might be attributed to the fact that LLaMA-3.2-3B-Instruct was released af-
ter LLaMA-3-8B-Instruct and incorporates pruning and distillation techniques,
which were specifically applied to the 1B and 3B variants [1]. These techniques
help to preserve performance while significantly improving efficiency, making
the LLaMA-3.2-3B-Instruct model capable of strong instruction-following per-
formance on resource-constrained devices. As a result, despite its smaller size,
LLaMA-3.2-3B-Instruct may benefit from a more refined architecture and train-
ing strategies, allowing it to better leverage in-context examples in one-shot
settings compared to the larger, but earlier, 8B model.
Fig.3: Average BLEU-4, ROUGE-L and RelaxedEM scores with different epochs
on different fine-tuned models

12 X. Pan et al.
RAG Performance As shown in Table 1, the score of RelaxedEM(all) drops
from 0.85 to 0.29 when incorporating RAG into the fine-tuned LLaMA-3-8B-
Instruct trained for 15 epochs, which does not lead to additional performance
gains, and it even degrades performance. This decline can be attributed to the
noisy or misaligned nature of the retrieved context—such as incorrect or irrel-
evant property suggestions from the ORKG—which may introduce confusion
instead of providing useful guidance since we don’t have a context checker to
check if the context is relevant or not. Prior studies [10,11] have similarly ob-
served that low-quality RAG context can conflict with the knowledge already
encoded in fine-tuned models, ultimately leading to reduced task performance.
One-Shot Performance As shown in Table 1, the one-shot setting—using
the most similar example from the training set—achieved strong performance,
second only to the fine-tuned models. Specifically, the one-shot approach reached
scores of 0.58 (BLEU-4), 0.81 (ROUGE-1), 0.73 (ROUGE-2), 0.78 (ROUGE-
L), and 0.40 (RelaxedEM on all test cases) on LLaMA-3.2-3B-Instruct model.
These results suggest that in the absence of high-quality datasets for fine-tuning,
one-shot learning offers a simple yet effective alternative for SPARQL query
generation.
Training Epoch Sensitivity As shown in Figure 3, the number of fine-tuning
epochs has a significant impact on all metrics for both LLaMA-3.2-3B-Instruct
and LLaMA-3-8B-Instruct models. First, both models start with high scores on
all metrics at epoch 3 and then slightly decline during the early training phases
(epochs 3–7), suggesting that the models may require sufficient training time
to properly internalize the SKG-specific structures and semantics. Second, the
training dynamics reveal an upward trend in performance from 7 epochs onward,
with the best performance at 20 epochs for the 3B model and 15 epochs for the
8B model. This indicates that larger models tend to converge faster and exhibit
stronger generalization early on, while smaller models require more epochs to
achieve competitive performance.
6.2 Error analysis on failed SPARQL queries
We further analyzed the generated SPARQL queries that either failed to execute
or returned empty results, by comparing them with the corresponding ground
truth queries. Under the best-performing configuration—using the fine-tuned
LLaMA3-8B-Instruct model trained for 15 epochs—448 out of 513 generated
SPARQL queries were executed successfully via QLever without any syntax er-
rors and returned meaningful results. Meanwhile, 14 queries failed due to syntax
errors, and 51 queries were executed successfully but returned empty results.
To better understand the causes of failure, we examined the error messages
for the 14 syntactically invalid queries and inspected the queries that returned
empty results. Our analysis revealed that 11 out of the 14 syntactically invalid
queries shared the same error message:

Title Suppressed Due to Excessive Length 13
Invalid SPARQL query: Variable ?metric is selected but not aggregated.
All non-aggregated variables must be part of the GROUP BY clause. Note:
The GROUP BY in this query is implicit because an aggregate expression
was used in the SELECT clause.
Thisindicatesthatthesequeriesincludedaggregatefunctions(e.g., MAX(?value) )
in the SELECTclause but did not include the non-aggregated variables (e.g.,
?metric,?metric_lbl ) in a GROUP BY clause. In SPARQL 1.1, such usage is
invalid unless all non-aggregated variables are explicitly grouped. This reflects a
lack of adherence to SPARQL’s syntax rules around aggregation and grouping.
The remaining 3 queries failed with the following error:
Invalid SPARQL query: Token ’SELECT’: mismatched input ’SELECT’ expecting
’}’.
This indicates that the queries contained improperly structured subqueries.
Specifically, full SELECTstatements nested directly inside a WHEREclause without
being enclosed in curly braces ( {}). SPARQL requires subqueries to be syntacti-
cally isolated within their own scope using curly braces. These errors likely stem
from incorrect handling of nested query structures during generation.
These findings highlight the current limitations of fine-tuned LLMs in cap-
turing the formal syntactic constraints of the SPARQL query language, partic-
ularly in scenarios involving nested subqueries and aggregation functions. As a
potential extension to our approach, prompt engineering techniques that include
explicitsyntaxerrorexamplesorconstraintreminderscouldbeincorporateddur-
ing SPARQL generation to encourage the model to produce syntactically valid
SPARQL, especially for complex constructs like aggregation and subqueries.
6.3 Limitations and Future Work
While FIRESPARQL demonstrates strong performance in generating SPARQL
queries over the ORKG, several limitations remain, which also highlight direc-
tions for future research: Our current experiments are limited to a single domain-
specific benchmark, SciQA, which is built on top of the ORKG. To assess the
generalizability of our approach, it is crucial to evaluate FIRESPARQL on a
broader range of benchmarks across different domains and knowledge graphs.
This would help determine whether the observed improvements are transfer-
able or highly task-specific. Although our framework includes an optional RAG
module, its effectiveness is currently hindered by the quality of the retrieved con-
text. In many cases, irrelevant or incorrect candidate properties are introduced,
leading to performance degradation. Future work should focus on developing
more accurate and semantically aware retrieval mechanisms that can provide
high-quality, contextually relevant information—such as topological subgraphs
or query templates—without introducing noise. FIRESPARQL relies on super-
vised fine-tuning using NLQ–SPARQL pairs, which may not always be available.
In future work, we aim to explore alternative data for fine-tuning LLMs such
as synthetic training data or leveraging weak supervision from ontology or sub-
graphs.

14 X. Pan et al.
7 Conclusion
In this paper, we introduced FIRESPARQL, a modular framework for SPARQL
query generation over SKGs. By systematically analyzing common error types,
structuralinconsistencies,andsemanticinaccuracies,wedesignedathree-module
architecture comprisingfine-tunedLLMs forSPARQLgeneration,optional RAG
for providing relevant context, and a lightweight SPARQL correction layer.
Our empirical evaluation on the SciQA benchmark demonstrates that domain-
specific fine-tuning, especially using LoRA for efficient parameter updates, sig-
nificantly improves both the syntactic quality and execution accuracy of gen-
erated SPARQL queries. Notably, our best-performing configuration, based on
the LLaMA-3-8B-Instruct model fine-tuned for 15 epochs, achieves state-of-the-
art results across all evaluation metrics, including BLEU, ROUGE, and relaxed
exact match (RelaxedEM). While RAG does not enhance performance in the
presence of fine-tuning, this points to the importance of high-quality context
retrieval. FIRESPARQL offers a reproducible and configurable framework that
can be adapted based on resource availability, paving the way for more robust,
interpretable, and scalable QA systems over SKGs.
References
1. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models.
https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/,
accessed: 2025-05-26
2. Auer, S., Barone, D.A.C., Bartz, C., Cortes, E., Jaradeh, M.Y., Kar-
ras, O., Koubarakis, M., Mouromtsev, D.I., Pliukhin, D., Radyush, D.,
Shilin, I., Stocker, M., Tsalapati, E.: The sciqa scientific question an-
swering benchmark for scholarly knowledge. Scientific Reports 13(2023),
https://api.semanticscholar.org/CorpusID:258507546
3. Bast, H., Buchhold, B.: Qlever: A query engine for efficient sparql+ text search.
In: Proceedings of the 2017 ACM on Conference on Information and Knowledge
Management. pp. 647–656 (2017)
4. Berant, J., Chou, A., Frostig, R., Liang, P.: Semantic parsing on freebase from
question-answerpairs.In:Proceedingsofthe2013conferenceonempiricalmethods
in natural language processing. pp. 1533–1544 (2013)
5. Cui, H., Peng, T., Han, R., Han, J., Liu, L.: Path-based multi-hop reasoning over
knowledge graph for answering questions via adversarial reinforcement learning.
Knowledge-Based Systems 276, 110760 (2023)
6. Ghanem, H., Cruz, C.: Fine-Tuning vs. Prompting: Evaluating the Knowledge
Graph Construction with LLMs. In: CEUR Workshop Proceedings. TEXT2KG-
DQMLKG-24 ( TEXT2KG 2024 and DQMLKG 2024 ) 3rd International workshop
one knowledge graph generation from text. Data Quality meets Machine Learning
and Knowledge Graphs 2024, vol. 3747, p. 7. Hersonissos, Greece (May 2024),
https://hal.science/hal-04862235
7. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint
arXiv:2106.09685 (2021)

Title Suppressed Due to Excessive Length 15
8. Huang, X., Zhang, J., Li, D., Li, P.: Knowledge graph embedding based question
answering. In: Proceedings of the twelfth ACM international conference on web
search and data mining. pp. 105–113 (2019)
9. Jiang, L., Yan, X., Usbeck, R.: A structure and content prompt-based method for
knowledge graph question answering over scholarly data. In: QALD/SemREC@
ISWC (2023)
10. Jin, B., Yoon, J., Han, J., Arik, S.O.: Long-context llms meet rag: Overcoming
challenges for long inputs in rag. arXiv preprint arXiv:2410.05983 (2024)
11. Joren, H., Zhang, J., Ferng, C.S., Juan, D.C., Taly, A., Rashtchian, C.: Sufficient
context: A new lens on retrieval augmented generation systems. arXiv preprint
arXiv:2411.06037 (2024)
12. Lehmann, J., Meloni, A., Motta, E., Osborne, F., Recupero, D.R., Salatino, A.A.,
Vahdati, S.: Large language models for scientific question answering: An exten-
sive analysis of the sciqa benchmark. In: The Semantic Web: 21st International
Conference, ESWC 2024, Hersonissos, Crete, Greece, May 26–30, 2024, Proceed-
ings, Part I. p. 199–217. Springer-Verlag (2024). https://doi.org/10.1007/978-3-
031-60626-7_11
13. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
14. Liu, J., Shen, D., Zhang, Y., Dolan, W.B., Carin, L., Chen, W.: What makes
good in-context examples for gpt-3? In: Proceedings of Deep Learning Inside Out
(DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for
Deep Learning Architectures. pp. 100–114 (2022)
15. Muralidharan,B.,Beadles,H.,Marzban,R.,Mupparaju,K.S.:Knowledgeai:Fine-
tuningnlpmodelsforfacilitatingscientificknowledgeextractionandunderstanding
(2024), https://arxiv.org/abs/2408.04651
16. Omar, R., Mangukiya, O., Kalnis, P., Mansour, E.: Chatgpt versus traditional
question answering for knowledge graphs: Current status and future directions
towards knowledge graph chatbots. arXiv preprint arXiv:2302.06466 (2023)
17. Perevalov, A., Diefenbach, D., Usbeck, R., Both, A.: Qald-9-plus: A multilingual
dataset for question answering over dbpedia and wikidata translated by native
speakers. In: 2022 IEEE 16th International Conference on Semantic Computing
(ICSC). pp. 229–234. IEEE (2022)
18. Pliukhin, D., Radyush, D., Kovriguina, L., Mouromtsev, D.: Improving subgraph
extraction algorihtms for one-shot sparql query generation with large language
models. In: QALD/SemREC@ ISWC (2023)
19. Taffa, T.A., Usbeck, R.: Leveraging llms in scholarly knowledge graph question
answering. In: QALD/SemREC@ ISWC (2023)
20. Trivedi, P., Maheshwari, G., Dubey, M., Lehmann, J.: Lc-quad: A corpus for com-
plex question answering over knowledge graphs. In: The Semantic Web–ISWC
2017: 16th International Semantic Web Conference, Vienna, Austria, October 21-
25, 2017, Proceedings, Part II 16. pp. 210–218. Springer (2017)
21. Wang, J., Chang, Y., Li, Z., An, N., Ma, Q., Hei, L., Luo, H., Lu, Y., Ren, F.:
Techgpt-2.0: A large language model project to solve the task of knowledge graph
construction (2024), https://arxiv.org/abs/2401.04507
22. Wang, Y., Lipka, N., Rossi, R.A., Siu, A., Zhang, R., Derr, T.: Knowledge graph
prompting for multi-document question answering. In: Proceedings of the AAAI
Conference on Artificial Intelligence. vol. 38, pp. 19206–19214 (2024)

16 X. Pan et al.
23. Yasunaga, M., Ren, H., Bosselut, A., Liang, P., Leskovec, J.: Qa-gnn: Reasoning
withlanguagemodelsandknowledgegraphsforquestionanswering.arXivpreprint
arXiv:2104.06378 (2021)
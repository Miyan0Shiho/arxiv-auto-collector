# Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks

**Authors**: Sizhe Yuen, Ting Su, Ziyang Wang, Yali Du, Adam J. Sobey

**Published**: 2025-05-20 11:16:29

**PDF URL**: [http://arxiv.org/pdf/2505.14212v1](http://arxiv.org/pdf/2505.14212v1)

## Abstract
A question-answering (QA) system is to search suitable answers within a
knowledge base. Current QA systems struggle with queries requiring complex
reasoning or real-time knowledge integration. They are often supplemented with
retrieval techniques on a data source such as Retrieval-Augmented Generation
(RAG). However, RAG continues to face challenges in handling complex reasoning
and logical connections between multiple sources of information. A novel
approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA
tasks is presented through the automated generation of context-based QA pairs.
This methodology leverages LLMs to create fine-tuning data, reducing reliance
on human labelling and improving model comprehension and reasoning
capabilities. The proposed system includes an automated QA generator and a
model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore.
Comprehensive experiments demonstrate improvements in logical coherence and
factual accuracy, with implications for developing adaptable Artificial
Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1,
BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA
pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA
pairs.

## Full Text


<!-- PDF content starts -->

arXiv:2505.14212v1  [cs.CL]  20 May 2025Automatic Dataset Generation for Knowledge
Intensive Question Answering Tasks
Sizhe Yuen1Ting Su1Ziyang Wang1Yali Du1,2Adam J. Sobey1,3
1The Alan Turing Institute2King’s College London3University of Southampton
{syuen,tsu,zwang,asobey}@turing.ac.uk
{yali.du}@kcl.ac.uk
Abstract
A question-answering (QA) system is to search suitable answers within a knowl-
edge base. Current QA systems struggle with queries requiring complex reasoning
or real-time knowledge integration. They are often supplemented with retrieval
techniques on a data source such as Retrieval-Augmented Generation (RAG). How-
ever, RAG continues to face challenges in handling complex reasoning and logical
connections between multiple sources of information. A novel approach for enhanc-
ing Large Language Models (LLMs) in knowledge-intensive QA tasks is presented
through the automated generation of context-based QA pairs. This methodology
leverages LLMs to create fine-tuning data, reducing reliance on human labelling
and improving model comprehension and reasoning capabilities. The proposed
system includes an automated QA generator and a model fine-tuner, evaluated
using perplexity, ROUGE, BLEU, and BERTScore. Comprehensive experiments
demonstrate improvements in logical coherence and factual accuracy, with implica-
tions for developing adaptable Artificial Intelligence (AI) systems. Mistral-7b-v0.3
outperforms Llama-3-8b with BERT F1, BLEU, and ROUGE scores 0.858, 0.172,
and 0.260 of for the LLM generated QA pairs compared to scores of 0.836, 0.083,
and 0.139 for the human annotated QA pairs.
1 Introduction
QA systems are designed to provide relevant answers to user questions. They are increasingly
being developed for customer support chatbots, summaries of technical documents or generating
reports. The advent of deep learning has significantly advanced QA systems through sophisticated
neural network architectures such as Transformers [ 40,13], improving their ability to understand and
generate responses. However, these models often struggle with queries requiring extensive factual
knowledge or real-time data, due to a crucial dependency on external knowledge sources. To address
this issue, Lewis et al. [29] introduced RAG, which merges the depth of traditional Information
Retrieval (IR) techniques with the generative capabilities of advanced language models. RAG
demonstrated effectiveness in integrating real-time retrieval into the generation process, enabling
models to dynamically access and utilise relevant external documents. RAG models combine the
strengths of generative pre-trained transformers, such as GPT-3 [ 5], with robust document retrieval
systems, marking a significant advancement in knowledge-intensive QA tasks.
Despite these advancements, language models for QA face challenges in providing factual answers
and logically sound reasoning to questions, particularly for queries which require broad knowledge
and complex reasoning chains to deliver precise, context-aware responses. Traditionally, open-domain
QA systems relied on keyword matching and shallow parsing techniques, limiting their reasoning
capabilities [ 9]. While RAG demonstrated effectiveness in integrating real-time retrieval, it still
faces challenges in complex reasoning tasks that require synthesising information from multiple
Preprint. Under review.

sources and performing multi-step inferences [ 39]. The core limitations of RAG include retrieval
inaccuracies, document consolidation errors, generation hallucinations, and high retrieval latency
issues [ 15,4]. Current research focus on developing adaptive weighting schemes for balancing
retrieved and intrinsic information [ 41], and exploring meta-learning approaches to improve the
model’s reasoning capabilities when reconciling conflicting data points [ 42]. Furthermore, retrieving
and reasoning over contextual information from large-scale document collections presents significant
challenges in maintaining logical coherence, as interconnections among documents require not just
retrieval but also logical inference to establish relationships between concepts [ 15]. Supervised
fine-tuning is a method to alleviate these issues and improve model performance, teaching the models
to reason and learn information in a specific domain which comes at a higher computational cost of
training and requires labelled datasets [ 3]. Unsupervised fine-tuning does not require labelled dataset
but has been shown to be outperformed by RAG solutions [ 35] by between 3.5% and 6%. The cost of
training and creation of labelled datasets highlights a need for automatic dataset creation methods to
reduce the cost of supervised fine-tuning for LLM-based QA tasks.
In this paper, we propose a novel approach to reduce the cost of fine-tuning QA systems by introducing
a self-improving cycle that creates context-based QA pairs from external documents, which are then
used to fine-tune an LLM. This approach aims to reduce the need for human labelling in knowledge-
intensive QA tasks, accelerating the pace at which LLMs can adapt to new domains and information by
enhancing both knowledge integration and reasoning capabilities in LLMs. We introduce a two-stage
prompt engineering technique with two primary components: (1) an automated QA generator, which
creates contextually relevant and diverse QA pairs from given documents and (2) a Model Fine-Tuner
that leverages the generated QA pairs to iteratively update the target LLM, gradually expanding its
knowledge base and improving its comprehension capabilities. This framework enhances the model’s
comprehension, but also ensures that generated responses are grounded in the provided context to
maintain fidelity to the source material.
In summary, our contribution is a novel method to enhance LLMs for knowledge intensive QA
tasks with an automated approach to generate synthetic QA pairs from contextual documents. We
provide empirical evidence for the effectiveness of our method through comprehensive evaluation
metrics, particularly focusing on the quality of logical reasoning in generated responses. Our
approach demonstrates how LLMs can be leveraged to create the training data that captures both
factual knowledge and reasoning patterns, potentially opening new avenues for developing more
sophisticated reasoning capabilities in AI systems.
2 Related Work
We first explore the related work in RAG, the strengths and limitations of its approaches. Then we
discuss supervised fine-tuning, an alternate method to RAG for improving LLM performance and
expertise in specific domains. This highlights a need for synthetic data generation, to reduce the cost
of creating labelled datasets for fine-tuning.
RAG integrates IR techniques with generative models to enhance the accuracy and relevance of
generated answers. Specifically, it generally involves two interlocking modules: the retriever,
which aims to gather relevant documents based on the question, and the generator, which interprets
both the question and the relevant documents to form comprehensible answers. The retrieval
component, often based on Dense Passage Retrieval (DPR) [ 27], fetches relevant documents from a
large corpus like Wikipedia. The generative component, typically a sequence-to-sequence model like
BART [ 28], generates answers conditioned on the retrieved documents. Ovadia et al. [35] introduced
a comprehensive RAG framework that utilizes a dense vector index of Wikipedia for document
retrieval and a BART model [ 28] for answer generation. This model was demonstrated to outperform
extractive models by generating more comprehensive and contextually accurate answers, even when
the correct answer is not in any retrieved document.
Subsequently, researchers developed many RAG-based models [ 8,23,44,16,2,32,37] that leverage
and improve upon the original RAG model. For instance, in the legal field, Chalkidis et al. [8]
proposed the CBR-RAG model, which integrates case-based reasoning to enhance retrieval, making
it particularly suitable for legal QA tasks. In conversational QA, a fine-grained retrieval augmentation
approach [ 44] has been proposed to refine question understanding and improve the relevance of
retrieved information by question reformulation and keyword extraction to better align the retrieved
2

documents with the user’s query. In addition, Glass et al. [16] aims to improve the ranking of retrieved
information by introducing a BERT-based reranker trained by additional high-quality labelled data.
However, the effectiveness of a RAG system heavily depends on the accuracy of its retrieval mecha-
nism, which itself is dependent on the accurate chunking and tagging of relevant documents. Complex
or ambiguous queries may lead to retrieval of irrelevant or misaligned document chunks, leading to
incomplete or incorrect answers [ 15]. These errors are particularly pronounced in naive implemen-
tations where the document context is simply prepending to an LLM prompt [ 21]. More advanced
RAG techniques of pre-retrieval [ 33,14] and post-retrieval strategies further improves on naive RAG
implementations at the cost of addition complexity and resources, which may exacerbate the latency
issues of RAG approaches.
Supervised Fine-tuning has been another approach shown to improve the performance of LLMs
in QA scenarios, either in isolation or in conjunction with RAG architectures. For example, the
DPR used in RAG can be fine-tuned within a RAG system to improve alignment between retrieved
documents and LLM responses [ 38]. In addition to training better retrievers to enhance the RAG
model in QA tasks, researchers also aim to fine-tune the language model to produce more relevant
and coherent answers for questions based on the retrieved context. Asai et al. [2]introduces the
Self-RAG framework, training a language model to retrieve, generate, and critique its own outputs
through self-reflection. Another approach is to separately fine-tune the language model and the
retrieval models. The RA-DIT framework [ 32] introduces a fine-tuning methodology to separate the
two fine-tuning steps, first fine-tuning the language models to better utilise retrieved information,
then fine-tuning the retriever to return results guided by the language model preferences. When
fine-tuning just the retrieval model, researchers have also treated the language model as a black box
and simply prepend the retrieved documents to the language model inputs [ 37]. There are two major
disadvantages to the fine-tuning approach: the cost of training, and the requirement of a labelled
dataset. The former can be alleviated by Parameter-Efficient Fine-Tuning methods such as Low Rank
Adaptation [ 20] and quantisation [ 12]. Our work aims to solve the the latter through the creation and
use of synthetic datasets in the fine-tuning process.
Synthetic Dataset Generation . The use of LLMs for synthetic data generation has been gaining
increasing attention as a solution to reduce the cost of human-labelling datasets. This approach has
been shown to be successful across a number of domains, such as text classification [ 30], human-
computer interaction (HCI) [ 18], and code generation [ 34]. Some of these examples, such as the
HCI case study by Hämäläinen et al. [18], where the generated dataset is questionnaire responses on
experiencing video games as art, do not use the generated dataset for further LLM training.
In the QA domain, one synthetic data generation approach is dialogue inpainting, where text from
a given document is transformed into a two-person dialogue between a writer and reader [ 10].
These existing approaches demonstrate the effectiveness of synthetic data generation, but have yet
to be applied to QA generation in a technical setting. We take a similar approach, but instead of
transforming a given document into dialogue, we prompt the model to generate relevant questions
and answers with the given document in technical industrial domains, where there is an abundance
of documentation, but the cost to create a human-labelled dataset is high and requires some level of
expertise in the domain.
3 Automatic Dataset Generation Methodology
Our approach leverages a self-improving cycle to produce a synthetic dataset that minimizes human
intervention while maximizing the potential for continuous learning and adaptation. The code for
our generation and fine-tuning processes is provided as a supplementary file. Figure 1 illustrates the
system architecture and data flow, which comprises of two main components:
1.Automated QA Generator - An LLM fine-tuned to generate relevant QA pairs from the
initial dataset using three steps: question generation (yellow), answer generation (green),
and post-processing (blue).
2.Model Fine-Tuner - Utilizes generated QA pairs to update the target LLM using standard
fine-tuning approaches.
3

Automated QA Generation
LLM ﬁne- tuningIBM Technotes from 
TechQAQuestion 
generationAnswer 
generationPost- 
processing
Ask the 10 most interesting and 
factoid possible questions that 
are related to the following 
provided context. Return the 
question in list format
### Context: [IBM Technote]
### Questions: [To be completed]Given the following context please
answer the question as factual as 
possible.
### Context: [IBM Technote]
### Question: [Generated question]
### Answer: [To be completed]Unanswered QAs
Unfinished QAs
Unrelated QAsFigure 1: Example generation procedure of QA pairs with the TechQA dataset.
3.1 Automated QA Generator
For automatically generating a large quantity of QA pairs, we use Mistral-7b-instruct-v0.3 hosted on
Replicate1as our baseline language model. Mistral-7b-instruct-v0.3 is a small open-source language
model that performs well on recent LLM benchmarks [ 25]. The QA generation is conducted in three
steps:
Question Generation – An LLM, in these experiments Mistral, generates a set of questions for each
context document given in the dataset. A maximum of 10 factoid question per context document are
requested from any given dataset, such as technical documentation. These are generated using the
prompt in the yellow box in Figure 1.
Answers Generation – Next, for each question related to each context document, we use another
prompt to ask the model to generate a factual answer. This creates up to 10 question answer pairs per
document. These are generated using the prompt in the green box in Figure 1. Note that the LLM
is allowed to answer "There are no possible factual answers based on the given content." for any
question deemed unanswerable. These questions are subsequently removed from the dataset during
post-processing.
Post-Processing – Finally, we clean up the generated dataset, which now would contain 10 QA pairs
per context document used during the generation stages. We use several methods to clean up the
dataset, as follows:
•Unanswered QA Check. Although we have directly told the LLM to answer unanswerable
questions using the response "There are no possible factual answers based on the given
content.", the LLM sometimes answers unanswerable questions in a different way. To combat
this, we use a BERT-based semantic similarity score to identify unanswered questions and
remove the QA pair from the dataset.
•Unfinished QA Check. Some answers are unfinished, leading to partially answered ques-
tions. We use a Roberta-based sentence completion classification model to identify unfin-
ished sentences. We do not remove these QA pairs from the dataset, but instead label them
with “***unfinished***" at the end of the answer. Note that we can only identify unfinished
sentences but not the entire answers. Thus if an answer is unfinished, but the last sentence in
the answer is finished, this answer is considered finished.
•Unrelated QA Pair Check. Some of the generated answers are not related to the question or
the provided context. We use a BERT-based semantic similarity score to identify unrelated
QA pairs and automatically remove them from the dataset.
3.2 Fine-Tuning Process
An LLM is then fine-tuned using the dataset, two models are selected for comparison: Llama-3-8b
[17] and Mistral-7b-v0.3 [ 24]. The Llama-3-8b model is licensed under the Meta Llama 3 community
license, and Mistral-7b-v0.3 is licensed under Apache 2.0. Both models are fine-tuned with 4-bit
1https://replicate.com
4

Table 1: Statistics on generated QA dataset based on TechQA dataset.
# of average # of total # of
context documents QAs/context generated QAs
All QA 11,960 4.614 55,179
Unfinished QA 2,389 1.469 3,509
quantisation using QLoRA [ 12] for efficiency and reduced memory usage, targeting all linear modules,
with rank r= 8andα= 16 . The 4-bit quantisation and low rank rallows us to fit both the Llama 3
and Mistral models for fine-tuning. An 8-bit AdamW optimiser is used with a weight decay of 0.01
and learning rate of 5e-5 as smaller learning rates and weight decay values are recommended with
QLoRA on small models [ 12]. Furthermore, the Unsloth [ 11] versions of the Llama 3 and Mistral
v0.3 models are used to improve performance and reduce memory usage. Contextual documents are
provided in fine-tuning with the following prompt:
Find the answer to the question in the given document.
### Question: [Question]
### Document: [Document]
A 0-shot, 1-shot, and 5-shot prompting strategy is used during the fine-tuning process, as few-shot
prompts have been demonstrated to improve language model performance [6].
3.3 Evaluation Metrics
We evaluate our approach through a comprehensive framework of automated metrics. For generation
quality, we employ perplexity measurements [ 22], ROUGE scores [ 31] and BLEU scores [ 36],
alongside a semantic similarity based BERTScore [45].
Perplexity provides insight into the model’s predictive capabilities on unseen data. ROUGE score
metrics evaluate the quality of generated answers against the reference answers to provide the semantic
and structural similarities between generated and reference answers. BLEU scores complement
ROUGE by assessing the fluency and adequacy of generated answers. BERT Score leverages
contextual embeddings from the BERT model [ 13] to compute similarity scores between generated
and reference texts to capture deeper semantic similarity beyond surface-level token matching. We
report precision, recall, and F1 from the BERT Score. Through ablation studies, we analyse the
contribution of each component in our pipeline.
4 Performance of Automatic Dataset Generation with Fine-Tuning on
TechQA dataset
The TechQA dataset [ 7] is selected for training and evaluation. It is a domain-specific QA dataset
tailored for technical support, distributed under the Apache 2.0 license. The QA pairs were taken
from the IBM technical support forums, with answers that are linked to specific IBM technical
documents (IBM Technotes). The dataset therefore presents a realistic QA scenario with real
internal documentation. It is chosen over more popular general knowledge QA datasets such as
HotpotQA [ 43], TriviaQA [ 26], and Natural Questions [ 1] as we aim to target LLM training in
scenarios where internal documentation is used. Datasets including HotpotQA and Natural Questions
are based on data from Wikipedia, which has a high risk of data contamination from pre-training,
leading to a lack of domain-specific performance. The use of internal technical documentation is
more closely aligned to real-world engineering applications where a QA system is trained on private
internal documentation unseen by general LLM pre-training.
Table 1 presents the statistics of the generated QA dataset from the TechQA context documentation.
Using 11,960 IBM Technotes as context documents, on average we are able to generate 4.6 QA
pairs per document, creating a total of more than 50,000 QA pairs after removing unfinished pairs in
the post-processing. This is compared to TechQA’s 1,400 QA pairs which were generated by five
professional annotators and a sixth who was a Linux system administrator, requiring two weeks of
training before the annotation period began.
5

The Llama-3-8b and Mistral-7b-v0.3 models are fine-tuned and tested against the TechQA dataset.
The original training and development test splits are maintained from TechQA. We test each model in
three fine-tuning set-ups: no training, training on the original dataset, and training on the synthetic
generated dataset. All fine-tuning set-ups provide the contextual documents to the model. For each
different training set used, we train and evaluate under 0-shot, 1-shot, and 5-shot prompts. We test
under two experimental settings for inference. The first where the task is assumed to be small or
important enough so that the human QA pairs cover the whole space of interest, and the specific
contextual document which contains the answer to the questions are provided to the model in the
prompt. This tests the models’ ability to understand and find the answer within the given document.
However, this requires an expert understanding of the technical documents and the time to find the
correct document. In the second experimental setting, it’s assumed that the contextual document is
not provided as the user is unfamiliar with the material. This tests how well the models have learned
and retained knowledge from the fine-tuning process, and generalising QA beyond human-derived
pairs. Tables 2 and 3 in the appendix display the results in full for fine-tuning with and without
context respectively, the best results are highlighted in bold. All experiments were conducted on an
A100 GPU with 40GB of memory, using the (anonymized due to double-blind policy) HPC service.
4.1 With Context
When the models are provided the contextual documents in the fine-tuning, the specific document
that contains the answer to the questions, training on the original dataset shows the best performance.
Figure 2 displays the best F1 scores of Llama-3-8b and Mistral-7b-v0.3 in 0-shot, 1-shot, and 5-shot
training. The F1 scores are shown here as it comprises of both the precision and recall of the answers,
though the same trend is seen in the BLEU and ROUGE scores. Training on the original dataset
shows the best improvement on both models. Mistral is the most consistent across all three prompting
strategies, with no training displaying the worst performance and original training providing the best
performance.
0-shot 1-shot 5-shot0.7000.7250.7500.7750.8000.8250.8500.8750.900F1 score0.8350.881 0.8830.8900.8950.899
0.8590.889
0.859
No training Original training Generated training
(a) Llama-3-8b
0-shot 1-shot 5-shot0.7000.7250.7500.7750.8000.8250.8500.8750.900F1 score0.8390.8440.8540.8970.902 0.902
0.8580.8690.882
No training Original training Generated training (b) Mistral-7b-v0.3
Figure 2: F1 score when answering with the context document provided.
Furthermore, 0-shot training for Mistral performs the worst and 5-shot training performs the best,
with 1-shot and 5-shot training using the original dataset demonstrating the same F1 score. With
5-shot training, Llama-3-8b achieves an F1 score of 0.899, a BLEU score of 0.489 and a ROUGE
score of 0.494 and Mistral-7b-v0.3 achieves an F1 score of 0.902, a BLEU score of 0.463, and
a ROUGE score of 0.485. Llama shows slight inconsistency between prompting strategies. For
example, the F1 scores for 1-shot training are similar between each training dataset, and in the 5-shot
training, the generated dataset performs worse than the original dataset.
In general, training with the original dataset provides the best performance across all three prompting
strategies, and training with the generated dataset improves performance against the baseline model
without training.
The best results coming from training with the original dataset shows that generated datasets are
outperformed by human-labelled datasets for finding answers given the correct document. This
assumes the correct document is fetched by a retrieval system for the LLM or that the user is familiar
with the material. Using the generated synthetic dataset is able to improve upon the original model
6

without training, but is outperformed when training on the original dataset, indicating both the
advantage and limitation of using the synthetic dataset.
4.2 No Context
When fine-tuning and inference are performed without the contextual documents, the models’ knowl-
edge and reasoning is tested. Model performance is generally lower as expected, since the models are
not provided the technical documentation which contains the answers to the questions. However, we
find that when no context is provided, using the generated dataset on both Llama and Mistral models
exhibits the best performance in 1-shot and 5-shot prompts.
0-shot 1-shot 5-shot0.7000.7250.7500.7750.8000.8250.8500.8750.900F1 score0.8110.8310.835
0.822 0.8230.831
0.8170.843 0.843
No training Original training Generated training
(a) Llama-3-8b
0-shot 1-shot 5-shot0.7000.7250.7500.7750.8000.8250.8500.8750.900F1 score0.8180.824 0.8220.8280.8310.836
0.8200.8580.855
No training Original training Generated training (b) Mistral-7b-v0.3
Figure 3: F1 score when answering with no context provided.
With the exception of the 0-shot prompting strategy, Figure 3 shows the use of the generated training
set leads to higher F1 scores on both the Llama and Mistral models. Furthermore, the Llama model
demonstrates a lower F1 score on 1-shot and 5-shot prompts compared to the base model before
fine-tuning, suggesting a lack of logical reasoning and overtraining if contextual documents are not
provided. The same results can be seen in the BLEU and ROUGE scores, where using the original
dataset for training show the best results when the context is provided, with the best BLEU scores
of 0.489 and 0.463, and ROUGE scores of 0.494 and 0.485 for Llama and Mistral, respectively.
Conversely, using the generated dataset for training shows the best results when no context is provided,
with the best BLEU scores of 0.116 and 0.172, and ROUGE scores of 0.217 and 0.260 for Llama
and Mistral respectively. The BLEU and ROUGE scores are significantly lower when no context is
provided, indicating a lack of semantic and structural similarities to the original labelled answers,
though the BERT F1 score remains similarly high.
5 Discussion
5.1 Limitations
The automated QA generation and subsequent fine-tuning of the generated dataset has only been tested
on the TechQA dataset. While this dataset closely aligns with the use of technical documentation in
QA tasks, it is not known how well the QA dataset generation and fine-tuning apply to other forms of
data and documentation. In the same manner the in-context results rely on a user already familiar
with the material to a level they can find the document with the correct QA in it or that there is a
retrieval system in place capable of performing the same task perfectly.
Two smaller LLMs, Llama-3-8b and Mistral-7b-v0.3, are fine-tuned and tested, which demonstrate
the effectiveness of fine-tuning on a generated dataset in cases with limited compute resources. The
use of QLoRA further reduces the computational requirements of the fine-tuning process. These
choices in smaller models and quantisation reduces the computation required but also reduces the
quality of answers, leading to lower evaluation metrics particularly for the BLEU and ROUGE scores.
Larger models with >8Bparameters, commercial closed sourced models, and un-quantised full
fine-tuning have not been tested, which may demonstrate higher quality answers without the need for
fine-tuning on either the original or generated dataset.
7

5.2 Societal Impact
Our work can enhance the creation of QA systems through the reduced human labour in creating
QA datasets for LLM fine-tuning. Annotation time and costs have been estimated in the region of 3
person-weeks for 20,000 annotations, which in a technical context requires highly trained annotators
[19]. In particular, LLM users who wish to develop their own models based on private documentation
would not need to manually create datasets. The use of synthetic datasets contributes to further AI
development, rapidly increasing the amount of data available for training and testing.
There are a few risks to creating and using synthetic datasets. First is the risk of job displacements
within technical and professional fields, replacing domain-specific expertise with AI expertise.
Domain expertise is necessary in the creation of such QA systems, for evaluation, management, and
further improvements, but there is a risk that experts are seen as redundant. Secondly, the overuse and
reliance on synthetic data could lead to bias during the generation stage, or reduced LLM performance.
Further studies on the balance of datasets curated by humans and synthetically generated datasets
should be conducted to understand the limitations of using synthetic data. Moreover, the human
cost of creating QA datasets is shifted to computational costs of generating the data and training, as
synthetic datasets can be much larger than human-labelled datasets.
6 Conclusion
QA systems are increasingly used in a range of applications. The generation of the data to train
these models, however, can be expensive, especially if the documentation is technical. In this paper,
we introduce a framework to generate synthetic QA datasets with LLMs and show that fine-tuning
using these synthetic datasets improves performance. However, we show that training on the original
dataset remains the best approach to find answers to questions when the correct contextual documents
are provided. Notably, we demonstrate that LLMs trained on the generated datasets are able to better
learn on the technical documentation, providing higher BERT F1, BLEU, and ROUGE scores of
0.843, 0.116, and 0.217 respectively when trained on the generated dataset compared to scores of
0.831, 0.072, and 0.194 when trained on the original dataset for Llama-3-8b, and BERT F1, BLEU,
and ROUGE scores 0.858, 0.172, and 0.260 compared to scores of 0.836, 0.083, and 0.139 for
Mistral-7b-v0.3.
References
[1] Chris Alberti, Kenton Lee, and Michael Collins. A bert baseline for the natural questions. 2019.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection. In The 12th International Conference on Learning
Representations , 2024. URL https://openreview.net/forum?id=hSyW5go0v8 .
[3]Angels Balaguer, Vinamra Benara, Renato Luiz de Freitas Cunha, Roberto de M. Estevão Filho, Todd
Hendry, Daniel Holstein, Jennifer Marsman, Nick Mecklenburg, Sara Malvar, Leonardo O. Nunes, Rafael
Padilha, Morris Sharp, Bruno Silva, Swati Sharma, Vijay Aski, and Ranveer Chandra. Rag vs fine-tuning:
Pipelines, tradeoffs, and a case study on agriculture, 2024. URL https://arxiv.org/abs/2401.08406 .
[4]Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdelrazek.
Seven failure points when engineering a retrieval augmented generation system, 2024. URL https:
//arxiv.org/abs/2401.05856 .
[5]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in neural information processing systems , 33:1877–1901, 2020.
[6]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language
models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165 .
[7]Vittorio Castelli, Rishav Chakravarti, Saswati Dana, Anthony Ferritto, Radu Florian, Martin Franz, Dinesh
Garg, Dinesh Khandelwal, Scott McCarley, Michael McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus,
8

John Pitrelli, Saurabh Pujar, Salim Roukos, Andrzej Sakrajda, Avi Sil, Rosario Uceda-Sosa, Todd Ward, and
Rong Zhang. The TechQA dataset. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors,
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 1269–
1278, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.117.
URL https://aclanthology.org/2020.acl-main.117/ .
[8]Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos.
Legal-bert: The muppets straight out of law school. In Findings of the Association for Computational
Linguistics: EMNLP 2020 , pages 2898–2904, 2020.
[9]Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading Wikipedia to answer open-domain
questions. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1870–1879, Vancouver,
Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1171. URL
https://aclanthology.org/P17-1171 .
[10] Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Zhao, Aida Amini, Mike Green, Qazi Rashid, and Kelvin
Guu. Dialog inpainting: Turning documents to dialogs. In International Conference on Machine Learning
(ICML) . PMLR, 2022.
[11] Michael Han Daniel Han and Unsloth team. Unsloth, 2023. URL http://github.com/unslothai/
unsloth .
[12] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of
quantized llms, 2023. URL https://arxiv.org/abs/2305.14314 .
[13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirec-
tional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
[14] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance
labels, 2022. URL https://arxiv.org/abs/2212.10496 .
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024. URL
https://arxiv.org/abs/2312.10997 .
[16] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Naik, Pengshan Cai, and
Alfio Gliozzo. Re2g: Retrieve, rerank, generate. In Proceedings of the 2022 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies ,
pages 2701–2715, 2022.
[17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere,
Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra,
Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton
Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt,
David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes,
Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic,
Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind
Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar,
Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov,
Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah,
Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang,
Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun,
Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak,
Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal
Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz
Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira,
Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli,
Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar
Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri
Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li,
Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin
Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira
Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre,
9

Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana
Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan
Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende,
Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky,
Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher,
Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor
Kerkez, Vincent Gonguet, Virginie Do, Vish V ogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan
Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen
Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine
Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng
Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey,
Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex
Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus,
Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan,
Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury,
Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin
Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni,
Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl
Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester
Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin,
Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi
Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa
Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman,
Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos,
Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella
Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna
Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun
Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim
Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman,
James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang,
Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang,
Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie
Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich,
Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla,
Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng
Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt,
Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov,
Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir
Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso,
Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White,
Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev,
Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent,
Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina
Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub,
Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah
Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh
Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh
Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng
Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang,
Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve
Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny
Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo
Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked,
Varun V ontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla,
Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen
Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo
Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi,
Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of
models, 2024. URL https://arxiv.org/abs/2407.21783 .
[18] Perttu Hämäläinen, Mikke Tavast, and Anton Kunnari. Evaluating large language models in generating
synthetic hci research data: a case study. In Proceedings of the 2023 CHI Conference on Human Factors
in Computing Systems , CHI ’23, New York, NY , USA, 2023. Association for Computing Machinery.
10

ISBN 9781450394215. doi: 10.1145/3544548.3580688. URL https://doi.org/10.1145/3544548.
3580688 .
[19] Kasra Hosseini, Thomas Kober, Josip Krapac, Roland V ollgraf, Weiwei Cheng, and Ana Peleteiro Ramallo.
Retrieve, annotate, evaluate, repeat: Leveraging multimodal llms for large-scale product retrieval evaluation,
2024. URL https://arxiv.org/abs/2409.11860 .
[20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/
abs/2106.09685 .
[21] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain
question answering, 2021. URL https://arxiv.org/abs/2007.01282 .
[22] Fred Jelinek, Robert L Mercer, Lalit R Bahl, and James K Baker. Perplexity—a measure of the difficulty
of speech recognition tasks. The Journal of the Acoustical Society of America , 62(S1):S63–S63, 1977.
[23] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag: Learning
to adapt retrieval-augmented large language models through question complexity. In Proceedings of the
2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) , pages 7029–7043, 2024.
[24] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and
William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825 .
[25] AQ Jiang, A Sablayrolles, A Mensch, C Bamford, DS Chaplot, D de las Casas, F Bressand, G Lengyel,
G Lample, L Saulnier, et al. Mistral 7b (2023). arXiv preprint arXiv:2310.06825 , 2023.
[26] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension, 2017.
[27] Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen tau Yih. Dense passage retrieval for open-domain question answering, 2020. URL https:
//arxiv.org/abs/2004.04906 .
[28] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural
language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics , pages 7871–7880, 2020.
[29] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in Neural Information Processing Systems , 33:9459–9474, 2020.
[30] Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, and Ming Yin. Synthetic data generation with large language
models for text classification: Potential and limitations, 2023. URL https://arxiv.org/abs/2310.
07849 .
[31] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches
out, pages 74–81, 2004.
[32] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Rodriguez,
Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. RA-DIT: Retrieval-
augmented dual instruction tuning. In The 12th International Conference on Learning Representations ,
2024. URL https://openreview.net/forum?id=22OTbutug9 .
[33] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting for retrieval-augmented
large language models, 2023. URL https://arxiv.org/abs/2305.14283 .
[34] Mihai Nadas, Laura Diosan, and Andreea Tomescu. Synthetic data generation using large language models:
Advances in text and code, 2025. URL https://arxiv.org/abs/2503.14023 .
[35] Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. Fine-tuning or retrieval? comparing
knowledge injection in llms, 2024. URL https://arxiv.org/abs/2312.05934 .
[36] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation
of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational
Linguistics , pages 311–318, 2002.
11

[37] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer,
and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In Proceedings of the 2024
Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) , pages 8364–8377, 2024.
[38] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, and Suranga Nanayakkara. Fine-tune the entire
rag architecture (including dpr retriever) for question-answering, 2021. URL https://arxiv.org/abs/
2106.11517 .
[39] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop
queries, 2024. URL https://arxiv.org/abs/2401.15391 .
[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems ,
30, 2017.
[41] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan Ö. Arık. Astute rag: Overcoming
imperfect retrieval augmentation and knowledge conflicts for large language models, 2024. URL https:
//arxiv.org/abs/2410.07176 .
[42] Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy Phung,
Rafael Rafailov, Nathan Lile, Dakota Mahan, Louis Castricato, Jan-Philipp Franken, Nick Haber, and
Chelsea Finn. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-thought,
2025. URL https://arxiv.org/abs/2501.04682 .
[43] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering.
2018.
[44] Linhao Ye, Zhikai Lei, Jianghao Yin, Qin Chen, Jie Zhou, and Liang He. Boosting conversational question
answering with fine-grained retrieval-augmentation and self-check. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval , page 2301–2305, New
York, NY , USA, 2024. Association for Computing Machinery.
[45] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating
text generation with bert. In International Conference on Learning Representations , 2020. URL https:
//openreview.net/forum?id=SkeHuCVFDr .
A Table of results
The full tables of results are displayed here, with every model, prompting strategy and quantitative
metric. The best result for each metric and model is highlighted in bold.
12

Table 2: With Context Results on TechQA.
Model Training Set Training Shot Inference Shot EM Precision Recall F1 BLEU ROUGE
Llama-3-8bno training -0-shot 0.000 0.810 0.864 0.835 0.069 0.208
1-shot 0.000 0.879 0.885 0.881 0.153 0.270
5-shot 0.000 0.879 0.887 0.883 0.276 0.449
original0-shot0-shot 0.000 0.866 0.893 0.878 0.087 0.377
1-shot 0.154 0.881 0.888 0.883 0.140 0.425
5-shot 0.220 0.898 0.884 0.890 0.141 0.431
1-shot0-shot 0.165 0.852 0.875 0.862 0.074 0.331
1-shot 0.000 0.874 0.891 0.881 0.135 0.418
5-shot 0.176 0.898 0.895 0.895 0.158 0.484
5-shot0-shot 0.066 0.789 0.888 0.835 0.062 0.177
1-shot 0.000 0.873 0.890 0.880 0.354 0.423
5-shot 0.077 0.894 0.907 0.899 0.489 0.494
generated0-shot0-shot 0.0 0.834 0.868 0.850 0.092 0.262
1-shot 0.110 0.838 0.884 0.859 0.148 0.272
5-shot 0.000 0.835 0.888 0.859 0.117 0.267
1-shot0-shot 0.000 0.832 0.866 0.848 0.181 0.285
1-shot 0.022 0.880 0.899 0.889 0.252 0.429
5-shot 0.000 0.831 0.871 0.850 0.088 0.266
5-shot0-shot 0.000 0.848 0.870 0.858 0.159 0.306
1-shot 0.011 0.843 0.880 0.859 0.176 0.276
5-shot 0.000 0.835 0.872 0.852 0.095 0.263
Mistral-7b-v0.3no training -0-shot 0.000 0.818 0.862 0.839 0.071 0.232
1-shot 0.000 0.824 0.868 0.844 0.081 0.259
5-shot 0.000 0.835 0.877 0.854 0.122 0.269
original0-shot0-shot 0.000 0.899 0.898 0.897 0.416 0.479
1-shot 0.000 0.890 0.900 0.894 0.417 0.475
5-shot 0.000 0.885 0.898 0.891 0.393 0.448
1-shot0-shot 0.000 0.888 0.891 0.888 0.377 0.439
1-shot 0.000 0.901 0.894 0.897 0.376 0.468
5-shot 0.000 0.903 0.903 0.902 0.460 0.485
5-shot0-shot 0.000 0.885 0.890 0.887 0.373 0.430
1-shot 0.000 0.903 0.896 0.899 0.388 0.476
5-shot 0.000 0.904 0.903 0.902 0.463 0.484
generated0-shot0 shot 0.000 0.818 0.869 0.841 0.090 0.258
1-shot 0.000 0.828 0.892 0.858 0.137 0.292
5-shot 0.000 0.822 0.871 0.845 0.095 0.250
1-shot0 shot 0.000 0.819 0.869 0.842 0.084 0.232
1 shot 0.011 0.854 0.887 0.869 0.175 0.365
5 shot 0.000 0.853 0.883 0.867 0.021 0.075
5-shot0 shot 0.000 0.861 0.902 0.880 0.235 0.412
1 shot 0.000 0.861 0.907 0.882 0.232 0.422
5 shot 0.000 0.828 0.874 0.850 0.113 0.275
13

Table 3: No Context Results on TechQA.
Model Training Set Training Shot Inference Shot EM Precision Recall F1 BLEU ROUGE
Llama-3-8bno training -0-shot 0.000 0.813 0.811 0.811 0.013 0.069
1-shot 0.000 0.826 0.837 0.831 0.021 0.141
5-shot 0.000 0.830 0.840 0.835 0.034 0.162
original0-shot0-shot 0.000 0.808 0.837 0.822 0.032 0.128
1-shot 0.154 0.810 0.834 0.821 0.028 0.127
5-shot 0.220 0.821 0.843 0.831 0.072 0.165
1-shot0-shot 0.165 0.800 0.832 0.815 0.014 0.153
1-shot 0.000 0.806 0.832 0.818 0.025 0.152
5-shot 0.176 0.814 0.837 0.825 0.036 0.164
5-shot0-shot 0.066 0.802 0.834 0.817 0.028 0.146
1-shot 0.000 0.814 0.832 0.823 0.020 0.158
5-shot 0.077 0.818 0.840 0.828 0.057 0.180
generated0-shot0-shot 0.000 0.802 0.833 0.816 0.022 0.121
1-shot 0.110 0.805 0.833 0.818 0.013 0.116
5-shot 0.000 0.836 0.851 0.843 0.116 0.194
1-shot0-shot 0.000 0.802 0.835 0.817 0.058 0.128
1-shot 0.022 0.834 0.853 0.843 0.105 0.217
5-shot 0.000 0.830 0.836 0.833 0.041 0.151
5-shot0-shot 0.000 0.756 0.828 0.790 0.026 0.086
1-shot 0.011 0.822 0.834 0.827 0.034 0.133
5-shot 0.000 0.834 0.834 0.834 0.030 0.139
Mistral-7b-v0.3no training -0-shot 0.000 0.802 0.835 0.818 0.016 0.126
1-shot 0.000 0.810 0.839 0.824 0.021 0.144
5-shot 0.000 0.807 0.838 0.822 0.028 0.145
original0-shot0-shot 0.000 0.820 0.838 0.828 0.036 0.133
1-shot 0.000 0.831 0.833 0.831 0.047 0.138
5-shot 0.000 0.839 0.835 0.836 0.062 0.107
1-shot0-shot 0.000 0.802 0.834 0.817 0.020 0.133
1-shot 0.000 0.814 0.832 0.822 0.030 0.107
5-shot 0.000 0.830 0.835 0.832 0.083 0.139
5-shot0-shot 0.000 0.807 0.835 0.820 0.022 0.112
1-shot 0.000 0.816 0.832 0.822 0.029 0.131
5-shot 0.000 0.826 0.835 0.830 0.073 0.134
generated0-shot0 shot 0.000 0.806 0.835 0.820 0.014 0.128
1-shot 0.000 0.843 0.855 0.849 0.139 0.224
5-shot 0.000 0.850 0.856 0.853 0.164 0.244
1-shot0 shot 0.000 0.788 0.828 0.807 0.011 0.103
1 shot 0.011 0.855 0.861 0.858 0.170 0.170
5 shot 0.000 0.853 0.858 0.855 0.172 0.260
5-shot0 shot 0.000 0.760 0.819 0.787 0.014 0.094
1 shot 0.000 0.848 0.855 0.851 0.165 0.242
5 shot 0.000 0.849 0.852 0.850 0.150 0.241
14

NeurIPS Paper Checklist
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: Yes
Justification: The claims made in the abstract and introduction are demonstrated in the
results section.
Guidelines:
•The answer NA means that the abstract and introduction do not include the claims
made in the paper.
•The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
•The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
•It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: Yes
Justification: The limitations of our methods are discussed in section 5.1 of the discussion.
Guidelines:
•The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
•The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
•The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
•The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
•The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
•If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
•While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: NA
15

Justification: The paper does not propose theoretical assumptions and does not include
theoretical results
Guidelines:
• The answer NA means that the paper does not include theoretical results.
•All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
•All assumptions should be clearly stated or referenced in the statement of any theorems.
•The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
•Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: Yes
Justification: The process for QA generation and fine-tuning details are described in the
methodology. The hyperparameters, models, and frameworks used for fine-tuning are also
described. Furthermore, a zipfile with the code is provided as supplementary material, and a
link to the repository can be provided after submission to maintain anonymity.
Guidelines:
• The answer NA means that the paper does not include experiments.
•If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
•If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
•Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general, releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
•While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a)If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b)If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c)If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d)We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
16

5.Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: Yes
Justification: The data and code are provided as supplementary material.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
•Please see the NeurIPS code and data submission guidelines ( https://nips.cc/
public/guides/CodeSubmissionPolicy ) for more details.
•While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
•The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( https:
//nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
•The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
•The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
•At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
•Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6.Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: Yes
Justification: Hyperparameters for fine-tuning, the data splits, how they were chosen, and
the type of optimiser are detailed in the methodology and results sections, with a URL to the
code also provided.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
•The full details can be provided either with the code, in appendix, or as supplemental
material.
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: No
Justification: Variables which can affect our results is only the fine-tuning and inference
random seed. Due to limited computational resources, we omitted error bars from repeated
runs with different random seeds. Nevertheless, we find that all inference runs are stable,
providing consistent results across runs with the same hyperparameters.
Guidelines:
• The answer NA means that the paper does not include experiments.
17

•The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
•The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
•The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
•It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
•It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
•For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
•If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: Yes
Justification: The compute resources required is detailed in the results section.
Guidelines:
• The answer NA means that the paper does not include experiments.
•The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
•The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
•The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9.Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?
Answer: Yes
Justification: The research we conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics.
Guidelines:
•The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
•If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
•The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: Yes
Justification: Positive and negative societal impacts are discussed in section 5.2
18

Guidelines:
• The answer NA means that there is no societal impact of the work performed.
•If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
•Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
•The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
•The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
•If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: NA
Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
•Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
•Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
•We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: Yes
Justification: The licenses of the datasets and models used are included in the text where
they are introduced.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
•The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
19

•For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
•If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
•For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
•If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: NA
Justification: We do not release new assets in the paper.
Guidelines:
• The answer NA means that the paper does not release new assets.
•Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
•The paper should discuss whether and how consent was obtained from people whose
asset is used.
•At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14.Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: NA
Justification: The paper does not involve crowdsourcing nor research with human subjects
Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
•Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
•According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15.Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: NA
Justification: The paper does not involve crowdsourcing nor research with human subjects
Guidelines:
•The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
20

•Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
•We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
•For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16.Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: Yes
Justification: The usage of LLMs to generate a synthetic QA dataset is described fully in the
methodology section
Guidelines:
•The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
•Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM )
for what should or should not be described.
21
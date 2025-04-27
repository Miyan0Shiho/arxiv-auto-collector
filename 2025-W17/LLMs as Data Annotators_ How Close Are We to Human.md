# LLMs as Data Annotators: How Close Are We to Human Performance

**Authors**: Muhammad Uzair Ul Haq, Davide Rigoni, Alessandro Sperduti

**Published**: 2025-04-21 11:11:07

**PDF URL**: [http://arxiv.org/pdf/2504.15022v1](http://arxiv.org/pdf/2504.15022v1)

## Abstract
In NLP, fine-tuning LLMs is effective for various applications but requires
high-quality annotated data. However, manual annotation of data is
labor-intensive, time-consuming, and costly. Therefore, LLMs are increasingly
used to automate the process, often employing in-context learning (ICL) in
which some examples related to the task are given in the prompt for better
performance. However, manually selecting context examples can lead to
inefficiencies and suboptimal model performance. This paper presents
comprehensive experiments comparing several LLMs, considering different
embedding models, across various datasets for the Named Entity Recognition
(NER) task. The evaluation encompasses models with approximately $7$B and $70$B
parameters, including both proprietary and non-proprietary models. Furthermore,
leveraging the success of Retrieval-Augmented Generation (RAG), it also
considers a method that addresses the limitations of ICL by automatically
retrieving contextual examples, thereby enhancing performance. The results
highlight the importance of selecting the appropriate LLM and embedding model,
understanding the trade-offs between LLM sizes and desired performance, and the
necessity to direct research efforts towards more challenging datasets.

## Full Text


<!-- PDF content starts -->

LLMs as Data Annotators: How Close Are We to Human Performance
Muhammad Uzair Ul Haq1, 3Davide Rigoni2,3Alessandro Sperduti3,4,5
1Amajor SB S.p.A, Via Noventana 192, 35027 Noventa Padovana, Italy
2Department of Pharmaceutical and Pharmacological Sciences, University of Padova, Italy
3Department of Mathematics “Tullio Levi-Civita”, University of Padova, Italy
4Augmented Intelligence Center, Bruno Kessler Foundation, Trento, Italy
5Department of Information Engineering and Computer Science, University of Trento, Italy
Abstract
In NLP, fine-tuning LLMs is effective for
various applications but requires high-quality
annotated data. However, manual annotation
of data is labor-intensive, time-consuming,
and costly. Therefore, LLMs are increasingly
used to automate the process, often employing
in-context learning (ICL) in which some
examples related to the task are given in the
prompt for better performance. However,
manually selecting context examples can
lead to inefficiencies and suboptimal
model performance. This paper presents
comprehensive experiments comparing several
LLMs, considering different embedding
models, across various datasets for the Named
Entity Recognition (NER) task. The evaluation
encompasses models with approximately 7B
and70B parameters, including both proprietary
and non-proprietary models. Furthermore,
leveraging the success of Retrieval-Augmented
Generation (RAG), it also considers a method
that addresses the limitations of ICL by
automatically retrieving contextual examples,
thereby enhancing performance. The results
highlight the importance of selecting the
appropriate LLM and embedding model,
understanding the trade-offs between LLM
sizes and desired performance, and the
necessity to direct research efforts towards
more challenging datasets.
1 Introduction
Data annotation plays a crucial role in training
machine learning (ML) models, especially in the
era of Natural Language Processing (NLP). In NLP,
data annotation typically involves annotating text
data with relevant information, such as named
entities, parts of speech, sentiment, intent, text
classification, etc. The data annotation carries even
more significance for fine-grained NLP tasks like
token classification, where each token of a sentence
has to be tagged with a gold label. In specializeddomains such as Human Resource Management
(HRM) or medical, organizations often possess
large datasets that can be leveraged to enhance
decision-making and operational efficiency through
the use of LLM-based NLP approaches (Urlana
et al., 2024). However, for these organizations
to fully harness the power of LLMs through fine-
tuning, they need high-quality annotated datasets.
Traditional data annotation is a labor-intensive
and costly process, especially when applied to
large corpora. For example, in the case of HRM,
annotating a dataset of 10,000 resumes for an
information extraction task can be prohibitively
time-consuming and requires significant human
effort (Feng et al., 2021).
Nowadays, pre-trained LLMs (Devlin et al.,
2019; Liu et al., 2019) can be cost-effectively
fine-tuned on downstream tasks. These fine-
tuned models are frequently used in scenarios
where continuous LLM usage for inference is
too expensive, such as when using API provided
by propriety services (OpenAI, 2023; Team,
2024a), or when there is the need for tailored
models to meet strict performance standards while
maintaining the privacy of sensitive information,
such as in specialized fields (Strohmeier, 2022;
Karabacak and Margetis, 2023). With the advent
of advanced LLMs such as GPT-4 (OpenAI,
2023), Qwen (Team, 2024b), and Llama (Touvron
et al., 2023), researchers and practitioners are
increasingly leveraging these models to enhance
the data annotation process (Tan et al., 2024).
Pre-trained on massive corpora, LLMs offer
unprecedented capabilities for automating and
streamlining annotation, improving scalability, and
reducing costs (Wang et al., 2021).
Recent studies have demonstrated that
LLMs (Wang et al., 2023; Naraki et al., 2024) can
achieve performance comparable to human level in
data annotation for Named Entity Recognition Task
1arXiv:2504.15022v1  [cs.CL]  21 Apr 2025

(NER). However, most of these evaluations are
conducted on widely used benchmark datasets such
as CoNLL-2003 (Tjong Kim Sang and De Meulder,
2003) and WNUT-17 (Derczynski et al., 2017).
For instance, between 2023 and2025 , the CoNLL-
2003 dataset has been utilized in 191studies, while
WNUT-17 has been considered in 45. In contrast,
more complex datasets like SKILLSPAN (Zhang
et al., 2022a) and GUM (Zeldes, 2017) have been
used significantly less frequently, in only 9and4
studies1, respectively. The results presented in this
paper suggest that to gain a more comprehensive
understanding of model performance regarding
data annotation via LLMs in NER task, it is crucial
to extend evaluations to more challenging datasets,
which better reflect the complexities of real-world
applications.
From a technical perspective, in the recent
literature, prompting (He et al., 2024a) and in-
context learning (ICL) (Dong et al., 2024) are
common approaches to leverage the LLMs for
data annotation (Tan et al., 2024). ICL, which
is a technique where some solved examples of
the task are given within the prompt for better7
performance, is generally proven to be more
effective. However, selecting the right and relevant
examples to use as context for LLMs continues
to be a challenging task (Zhang et al., 2022b).
Manually choosing examples for each query
creates labor overhead, and more significantly, the
use of incorrect context examples may lead LLMs
to produce hallucinations (Yao et al., 2024) or
inaccurate outputs.
To address the above mentioned challenges, this
paper presents the following contributions:
1.Comprehensive Evaluation of LLMs and
Embeddings. It provides a comprehensive
assessment of LLMs for data annotation in
NER tasks, examining two distinct embedding
models, as well as different techniques such
as ICL and RAG, while utilizing datasets
of varying complexity. It compares five
models including proprietary models, such as
gpt-4o-mini , and open-source alternatives
with approximately 7B and 70B parameters
scale.
2.Trade-off Between LLM Sizes and
1The statistics regarding the datasets usage is collected
from https://paperswithcode.com/dataset/Performances. The trade-off between LLM
sizes and performance is demonstrated,
which is further verified by the statistical
tests. In fact, with the appropriate LLM and
embedding models, there are no statistically
significant differences in results between
certain 7B and 70B models.
3.A RAG-Based Annotation Approach. To
improve annotation quality and address the
limitations of manual context selection in
ICL, this paper considers a RAG based
approach (Lewis et al., 2020). Instead of
manually crafting in-context examples, the
proposed method retrieves the most relevant
samples based on similarity scores, enabling
LLMs to generate more accurate annotations.
2 Related Work
In the recent past, there have been efforts
by researchers to leverage the LLMs for data
annotation (Tan et al., 2024). Wang et al. (2021)
introduced the use of GPT-3 (Brown et al., 2020)
for data annotation. The authors evaluated the
quality of data generated by the GPT-3 against
the human-labeled data. For each sentence to be
annotated by the model, they construct a prompt
consisting of several human-labeled examples
along with the target sentence. They evaluate the
performance in n-shot settings. Also, the authors
report the performance of text classification and
data generation tasks. Likewise, He et al. (2024a)
leveraged the use of GPT-3.5 based models to
annotate data. In comparison to the previous
approach presented by Wang et al. (2021), the
authors introduced the concept of chain-of-thought
(CoT) (Wei et al., 2023) reasoning to annotate data.
The authors simulate the human reasoning process
to induce GPT-3.5 to motivate the annotated
examples. They present the task description,
specific examples, and the corresponding gold
labels to GPT-3.5, and then ask the model to
explain whether/why the given label is appropriate
for that example. This enables the model to
explain its choice of a specific label for the target
sentence. Then, the authors construct the few-shot
CoT prompts using the explanations generated by
the model for data annotation.
To leverage the GPT model for the Named
Entity Recognition (NER) task, Wang et al.
(2023) proposed a GPT-NER model. The main
2

contribution introduced by the authors is to
transform the NER into a text-generation task.
The authors used prompt engineering, where
prompts consist of three parts: ( i)task description;
(ii)few-shot examples; and ( iii)input sentence.
To choose few-shot context examples, they used
two different strategies: ( i)random retrieval; and
(ii)k-NN based retrieval from training data.
In this work, the authors propose a retrieval-
based approach for selecting context examples.
Specifically, for each training instance, the method
iterates through all tokens in a sentence to identify
thek-nearest neighbor (k-NN) tokens. The top
kretrieved tokens are then selected, and their
corresponding sentences are used as context. The
context examples are retrieved from the entire
training dataset. Furthermore, for sentences
containing multiple entities, the algorithm runs
multiple times to ensure the extraction of all entities
within the sentence.
Following the work of Wang et al. (2021) and Wang
et al. (2023), Naraki et al. (2024) also proposed a
LLMs based annotation for NER task. The authors
used the LLMs to clean noise and inconsistencies in
the NER dataset, and then they merged the cleaned
NER dataset with the original dataset to generate
a more robust and diverse set of annotations. It is
worth mentioning that, in merging the annotations
from LLM with human labels, preference is given
to human-annotated examples compared to the
LLM annotations. In addition, Bogdanov et al.
(2024) used the LLMs to create a general dataset
for NER tasks with a broad range of entity types.
The authors demonstrate a procedure that consists
of annotating raw data with an LLM to train a
task-specific foundation model for NER. Goel et al.
(2023) uses the same concept of data annotation
using LLMs, however, they do a case study on a
medical domain where they leverage the LLMs
for accelerating the annotation process along with
human input.
The research discussed above highlights the strong
interest in using LLMs for dataset annotation,
with most approaches relying on ICL. However,
systematic evaluation on complex datasets remains
limited, and selecting appropriate context examples
for ICL is still a challenge. This study provides a
comprehensive evaluation of LLMs for NER data
annotation.3 Methodology
3.1 Problem Definition
Given a dataset D={Si}n
i=1, where Sirepresents
thei-th sentence, with training, validation and
test split given as Dtrain,Dvalid andDtest. We
divide Dtrain into two disjoint subsets: X(we call
as sample space), from which we sample context
examples, and T, which will be annotated by the
LLM. Formally, let X ⊂ D train be a subset of
sizex, where x < n , andT=Dtrain\ Xbe the
remaining subset containing tsentences, where
t=n−x. From X, we select mexamples, where
m < x , to form the context set M. The LLM
uses all the mexamples in Mas input context to
annotate the tsentences in T.
The NER task can be defined as the problem of
learning an approximation function efθthat closely
matches the real function f:SV× V → C , where
SVrepresents the set of all the possible sentences
composed only by words win the vocabulary
V, and Crepresents the set of possible entity
categories. The real function fgiven: ( i)a sentence
Si∈ SV, and ( ii)a word w∈ V, assigns wto its
corresponding category c∈ C.
3.2 Data Annotation via LLMs
The methodology adopted in the proposed RAG
approach is shown in Figure 1. This section
discusses the steps followed in the proposed
study. Section 3.2.1 explains the prompt template
formation, while Section 3.2.2 presents the baseline
approach, followed by ICL method in Section 3.2.3.
Section 3.2.4 presents the proposed RAG technique,
whereas the importance of structured outputs for
NER task is discussed in Section 3.2.5.
3.2.1 Prompt Formation
In NLP, crafting an effective prompt for LLMs is a
crucial task, as an ill-formed prompt could lead to
poor performance. Different LLMs, whether open-
source or proprietary, tend to respond differently
to variations in prompt (Errica et al., 2024). This
work adopts a similar approach to prompt design
presented in (He et al., 2024b; Wang et al., 2023),
i.e. structuring our prompts around three key
components, also visible in Figure 1: ( i) Task
Description . This component clearly defines the
task the LLM is expected to perform; ( ii) Context .
This component provides task-related examples
that help the LLM to better understand the problem,
while also clarifying the expected input/output
3

Figure 1: Workflow of the proposed approach. Dtrain denotes the training data, Xdenotes the few human annotated
examples, whereas Tdenotes the training instances to be annotated by LLM. For each entry Ti∈ T, we extract M
context examples from a vector store using a retriever module. Then, given an input sentence, the final prompt to
LLM consists of the task description, the context examples in M, and input sentence.
format; and ( iii) Input . This final component
presents the LLM with the specific examples to
be annotated. The prompt structures adopted
in the experiments are outlined in Appendix G,
while several prompt examples are reported in
Appendix H.
3.2.2 Zero-shot Data Annotation
In the zero-shot setting (refers to the baseline),
the LLM receives only task descriptions and
entity categories from the dataset. The task
description explains the task, whereas entity
categories provide information about the classes
that the LLM has to use for annotation. Providing
entity categories in the prompt allows the LLM
to produce consistent output annotation as in the
training set. For instance, in the CoNLL-2003
dataset, person andorganization categories are
labelled as PER andORG respectively. Thus, the
prompt to the LLM includes PER andORG to
annotate entities in the person andorganization
categories, respectively. However, in zero-shot
data annotation, the lack of context examples
hinders the model’s understanding, often leading to
suboptimal performance. Nonetheless, this setting
allows to evaluate the general knowledge of LLM
on a task.
3.2.3 In-Context Learning
In ICL, the prompts given to LLMs are enhanced
by including not only a task description and entity
categories but also contextual examples. Theseexamples aid the models in better understanding
the task at hand. As detailed in Section 3.1, Dtrain
is is split into XandT. From X, the selection of
Mcan be approached in two ways: either through
manual cherry-picking or by random sampling.
However, manually selecting Mcan be both time-
consuming and subjective, which contradicts the
rationale of the proposed study. Therefore, we
opt to randomly sample MfromX, although it
does not guarantee whether the selected context
examples Mare semantically close to the input
textTi, which is a limitation of this approach.
3.2.4 Retrieval-Based Approach
To overcome the limitations of the previously
mentioned approaches, this paper introduces a
retrieval-based method for automatically selecting
relevant context examples. As outlined in
Section 3.1, the proposed RAG-based approach
first generates embedding representations for all
examples in X, which are then stored in a vector
database (Douze et al., 2024) for subsequent
retrieval, as illustrated in Figure 1. Subsequently,
for each sentence Ti∈ T , its embedding
representation is generated, and the most similar
Mexamples are retrieved from Xstored in the
vector database. Mis then used as context for the
LLM to provide the most relevant examples for
annotating the input text Ti.
4

3.2.5 Structured Output from LLMs
For a label-sensitive task like NER, getting a
structured output from a LLM is a crucial step.
In the NER task, as defined in Section 3.1,
each token in a sentence is tagged with a
corresponding label. Hence, preserving the token-
label correspondence in the output is necessary for
the LLMs. The most recent LLMs are based on a
decoder architecture that, while being suitable for
sequence-to-sequence tasks, encounters challenges
when tackling the NER task due to the potential
misalignment between tokens and labels (Ul Haq
et al., 2024). In fact, recent studies on NER (Li
et al., 2024; Liu et al., 2024; Wang et al., 2023)
have shown that the decoder architecture presents
structural inconsistencies in the output. Recently,
OpenAI (OpenAI, 2023) released a feature for the
latest GPT-4 based models which guarantees to
follow the structured output format2. To solve the
token-label misalignment problem, in this study,
we leverage the latest feature of StructuredOutput
released by OpenAI. However, it is important to
note that despite the inclusion of such features in
the latest LLMs, including Qwen (Team, 2024b)
and Llama (Touvron et al., 2023) based models,
they still exhibit inconsistencies in their output,
unlike the gpt-4o-mini-2024-07-18 .
4 Experimental Setup
4.1 Datasets
In this study, to evaluate the performance of the
proposed methodology and assess the capabilities
of LLMs, four datasets are considered, with their
statistics summarized in Table 1 of Appendix A.
Each dataset presents unique challenges for LLMs
in performing NER tasks, allowing this study to
comprehensively analyze the ability of LLMs to
handle diverse entity types, from well-structured
entities to complex, ambiguous, and domain-
specific annotations.
CoNLL-2003 The CoNLL-2003 (Tjong
Kim Sang and De Meulder, 2003) dataset consists
of four general entity types. Entities in this dataset
typically follow structured patterns, making them
relatively easier for LLMs to identify and classify.
WNUT-17 The WNUT-17 (Derczynski et al.,
2017) dataset contains six categories of rare entities.
This dataset is particularly challenging due to its
2https://openai.com/index/
introducing-structured-outputs-in-the-apinoisy text, sparse entity occurrences, and limited
labeled examples per category. Improving recall
on this dataset remains a significant challenge for
LLMs.
GUM The GUM (Zeldes, 2017) dataset is a
richly annotated corpus designed for multiple
NLP tasks, including NER. It captures linguistic
phenomena across various domains and genres,
making it a valuable resource for evaluating model
performance. The dataset includes eleven distinct
named entity types. Compared to CoNLL-2003
and WNUT-17, GUM presents a higher level of
complexity by incorporating a diverse set of entity
types spanning multiple domains.
SKILLSPAN The SKILLSPAN (Zhang et al.,
2022a) dataset is composed of a single entity
type. Unlike traditional entities, soft skills do
not follow a fixed syntactic or semantic structure,
making them inherently ambiguous. These
entities can range from single tokens to multi-
token expressions, increasing the complexity of
annotation and information extraction tasks for
LLMs.
4.2 Approaches Under Study
In the empirical assessment of the datasets
annotated by LLMs, the zero-shot data annotation
approach is chosen as the baseline since it provides
no context about the task to the LLM. This zero-
shot setting allows the evaluation of the LLM’s
general knowledge of the task. Moreover, ICL and
RAG-based approaches, detailed in Section 3.2.3
and Section 3.2.4 respectively, are considered.
For both, experiments are conducted with three
different numbers of context examples: ( i)25,
(ii)50, and ( iii)75. Experiments are conducted
on a30% sample of the training set Dtrain, while
the ablation study in Appendix D examines the
effects of 10% and 20% sample sizes.
This paper considers five different LLMs3:
(i)gpt-4o-mini-2024-07-18 , (ii)Qwen2.5-
72B-Instruct , (iii) Llama3.5-70B-Instruct ,
(iv)Qwen2.5-7B-Instruct , and ( v)Llama3.1-
8B-Instruct , and two embeddings models:
(i)thetext-embedding-3-large model4, and
(ii)thesentence transformer all-MiniLM-L6-
v2 model (Reimers and Gurevych, 2019).
3The models are referred to by their base names, such as
Qwen2.5-72B forQwen2.5-72B-Instruct , and so on.
4https://platform.openai.com/docs/guides/embeddings
5

Throughout the remainder of the paper,
text-embedding-3-large will be referred
to as OpenAI, and sentence transformer
all-MiniLM-L6-v2 will be referred to as ST.
Implementation details of results are reported
Appendix B.
4.3 NER Evaluation Process
To assess the quality of annotations generated by
LLMs, the RoBERTa model (Liu et al., 2019) is
fine-tuned on LLM-annotated datasets, leveraging
its proven effectiveness in NER tasks (Zhou et al.,
2022; Zhang et al., 2022a). Initially, an LLM is
employed to automatically annotate sentences in
T ⊂ D train, using strategies from Section 3.2.
This process generates annotations for T, resulting
in a new training set, ˆT, with |ˆT |=|T |. This
annotated set is then used to fine-tune the RoBERTa
model (Liu et al., 2019). Model selection is
performed on the validation set, Dvalid, and the
final evaluation results are based on the test set,
Dtest. To ensure robustness and mitigate the impact
of random initialization, we average the results
across five different seed values. The F1score is
used to assess the performances of the models.
5 Results and Analysis
This section presents the quantitative results of this
study, as well as its analysis. Qualitative results are
reported in Appendix F, while Appendix E reports
the statistical tests to support the findings.
5.1 Quantitative Results
Figure 2 presents the overall results of the
experiments, while the corresponding detailed
outcomes are reported in Appendix C. Specifically,
the heatmaps present the F1scores obtained on the
test set for different datasets, comparing several
models and methods used in the proposed study.
The CoNLL-2003 dataset, which contains named
entities like persons, organizations, and locations,
is relatively well-structured, making it easier for
LLMs to generate high-quality annotations. The
gpt-4o-mini model with OpenAI embeddings
emerges as the top performer (also shows statistical
significance over other models as detailed in
Appendix E), achieving an F1score of 89.72with
75context examples, which is just 2.7%below
human-level annotation. Among the ∼70B
models, Qwen2.5-72B with OpenAI embeddings
performs comparably to gpt-4o-mini with anF1score of 89.34, while Llama3.5-70B with
ST embeddings lags slightly behind with an F1
score of 87.33. At the ∼7Bscale, Qwen2.5-7B
with ST embeddings significantly outperforms
its counterparts, achieving an F1score of 87.94,
while Llama3.1-8B with OpenAI embeddings
scores 84.91. This suggests that smaller models
can still perform competitively when paired with
appropriate embedding methods. Interestingly,
the heatmap reveals that context size plays
a crucial role— gpt-4o-mini and Qwen2.5-70B
benefit significantly from larger context sizes of
75examples, while Llama3.5-70B performs best
at a slightly lower context size. This suggests that
different models have varying levels of context
saturation, where additional examples may not
always improve performance linearly.
The WNUT-17 dataset, which focuses on low-
frequency and emerging entities, presents a
significant challenge due to limited training
samples for each entity. However, Qwen2.5-70B
with OpenAI embeddings achieves the highest
F1score of 53.72, slightly outperforming
gpt-4o-mini , which attains an F1score of 53.43.
The Llama3.5-70B model exhibits inconsistent
performance, scoring 51.18with ICL at 75context
examples, suggesting that it struggles to generalize
well for rare entity detection. At the ∼7B
scale, Qwen2.5-7B with ST embeddings achieves
anF1score of 49.48, significantly outperforming
Llama3.1-8B , which scores 44.42. This highlights
that ST embeddings provide a crucial advantage
for smaller models. Compared to human-level
annotation, which achieves an F1score of 54.93,
the best-performing LLM reduces the gap to just
1.21%, which is the smallest performance gap
between human and LLM annotation across all
datasets used in the experiments. This suggests
that RAG-based annotation is highly effective in
adapting to rare entity recognition, particularly
when combined with larger models and strong
embeddings.
The GUM dataset presents a unique challenge
due to its diverse entity types, requiring models
to generalize across various linguistic structures.
Qwen2.5-70B with ST embeddings achieves the
bestF1score of 55.11, significantly surpassing
gpt-4o-mini , which attains an F1score of 52.28,
and Llama3.5-70B with OpenAI embeddings,
which achieves an F1score of 48.33. At the ∼
7Bscale, Qwen2.5-7B with OpenAI embeddings
6

gpt-4o-miniQwen2.5-72B Llama3.5-70B Qwen2.5-7B Llama-3.1-8BBaselineICLRAG w/STRAG w/OpenAIICLRAG w/STRAG w/OpenAIICLRAG w/STRAG w/OpenAICoNLL2003
32.24
Lower92.12
Human
gpt-4o-miniQwen2.5-72B Llama3.5-70B Qwen2.5-7B Llama-3.1-8BWNUT-17
23.43
Lower54.93
Human
gpt-4o-miniQwen2.5-72B Llama3.5-70B Qwen2.5-7B Llama-3.1-8BBaselineICLRAG w/STRAG w/OpenAIICLRAG w/STRAG w/OpenAIICLRAG w/STRAG w/OpenAIGUM
5.31
Lower58.26
Human
gpt-4o-miniQwen2.5-72B Llama3.5-70B Qwen2.5-7B Llama-3.1-8BSKILLSPAN
5.22
Lower54.79
Human25 Ex. 50 Ex. 75 Ex. 75 Ex. 50 Ex. 25 Ex.
Figure 2: Heatmaps of the F1scores across four datasets. The color scale represents performance, with red indicating
higher scores reaching human-level, and blue indicating lower scores starting from the lowest performing model
achieves an F1score of 44.48, outperforming
Llama3.1-8B , which scores 43.91. However, both
models show a notable performance drop compared
to their larger counterparts, suggesting that smaller
models struggle with datasets with diverse entities.
The3.15% gap between the best-performing LLM
and human-level annotation highlights that GUM
remains a challenging dataset for LLMs. The
heatmap further suggests that model performance
fluctuates significantly depending on context size
and embedding choice.
The SKILLSPAN dataset is the most difficult
among those evaluated, as it requires understanding
nuanced skill mentions across various job contexts.
gpt-4o-mini with OpenAI embeddings performs
the best, achieving an F1score of 34.06with
75context examples, but this is still far from
human-level annotation. At the ∼70Bscale,
Qwen2.5-70B with ST embeddings achieves an
F1score of 32.35with 50context examples,
outperforming Llama3.5-70B , which achieves an
F1score of 27.55. Among ∼7Bmodels,
Qwen2.5-7B with OpenAI embeddings achieves
anF1score of 29.67, significantly surpassing
Llama3.1-8B , which scores 22.88. This suggests
that embedding choice plays a crucial role in
skill extraction tasks. Notably, the gap betweenhuman annotation and the best-performing LLM
is much larger in this dataset compared to
others, indicating that LLMs struggle with skill-
based entity recognition. This could be due to
the complexity of contextual skill interpretation,
requiring deeper domain knowledge and better
understanding capabilities.
5.2 Different Sample Space Choices
This section examines the impact of sample space
choices, denoted as Xin Section 3.1, using
the proposed RAG-based approach as overall it
performs better than ICL. The experiments are
conducted on the SKILLSPAN dataset with the
gpt-4o-mini model and OpenAI embeddings. As
shown in Figure 3, for smaller dataset splits, the
RAG-based approach exhibits greater variability,
similar to the behavior seen with ICL. This suggests
that as the sample space for selecting context
examples decreases, the performance of the RAG-
based approach converges more closely with that
of ICL. More detailed results are reported in
Appendix D.
6 Discussion
Performance of LLMs The performance of
different LLMs in our study reveals interesting
7

Context Size
25Context Size
50Context Size
7526283032F1scoreRAG: 10%
RAG: 20%
ICL: 10%
ICL: 20%
Figure 3: F1scores for different context sizes ( 25,50,
and75) and sample spaces ( 10% and 20%) for the
RAG and ICL approach on the SKILLSPAN dataset,
using the gpt-4o-mini model. The plot indicates that
with a smaller sample size, the RAG approach performs
comparably to ICL.
insights. Across all datasets, RAG-based
approaches improve annotation quality, with
gpt-4o-mini and OpenAI embeddings achieving
the best results. In contrast, ICL struggles
in datasets with sparse or ambiguous entities,
particularly SKILLSPAN. While all models
perform well on CoNLL-2003, performance
declines as entity structures become more complex,
such as in GUM and SKILLSPAN.
Effect of Embeddings The choice of
embeddings for retrieval of context for LLMs
plays a crucial role in annotation quality in
retrieval-based methods. OpenAI embeddings lead
to better F1scores compared to smaller-scale ST
embeddings especially for gpt-4o-mini model.
This effect is particularly evident in WNUT-17 and
GUM, where entity distributions are more diverse,
and high-quality embeddings improve retrieval
effectiveness. In contrast, SKILLSPAN remains
challenging across all embedding strategies,
suggesting that current embedding techniques
struggle with soft skill representation due to the
abstract nature of the entities.
Effect of Model Size Larger models generally
perform better, but retrieval quality is equally
critical. Qwen2.5-7B slightly outperforms
Llama3.1-8B and performs comparably to
Llama3.5-70B with proper embeddings,
indicating that architecture and training data
impact annotation beyond parameter count.
Statistical tests in Appendix E support this finding.
Effect of Dataset Complexity Breaking down
results per dataset, CoNLL-2003 shows minimal
variance across methods, as structured entities
are well-represented in training data. WNUT-17
benefits the most from retrieval-based methods,as rare entities require additional context for
accurate recognition. GUM’s diverse entity
types pose a challenge for ICL, but RAG-
based methods significantly improve performance.
Finally, SKILLSPAN remains the most difficult
dataset, with lower performance across all
methods, underscoring the limitations of LLMs
and embeddings in capturing the semantics of soft
skills.
7 Conclusions and Future Works
This study systematically evaluates the
effectiveness of LLMs for data annotation
across four diverse datasets—CoNLL-2003,
WNUT-17, GUM, and SKILLSPAN of varying
complexity. It compares RAG in different
embedding strategies, ICL, and a baseline
approach. The results demonstrate that RAG-based
methods consistently outperform both ICL and the
baseline across all datasets, significantly reducing
the performance gap with human-level annotation.
A key finding is that dataset complexity plays
a crucial role in model performance. For
structured datasets like CoNLL-2003, LLMs
perform exceptionally well, with models such as
gpt-4o-mini andQwen2.5-72B achieving results
within 3% of human-level annotation. Conversely,
performance deteriorates as dataset complexity
increases. The SKILLSPAN dataset, which
requires nuanced skill recognition, presents the
greatest challenge, with LLMs struggling to
capture implicit skill mentions.
Our analysis also highlights the importance
of context size and embedding choice in
retrieval-augmented annotation. We observe
that larger models such as Qwen2.5-72B and
gpt-4o-mini benefit from larger context sizes,
while smaller models like Qwen2.5-7B can still
perform competitively when paired with high-
quality sentence embeddings. However, models
exhibit context saturation effects, where additional
examples do not always lead to linear performance
improvements.
Future works will focus on enhancing the
performance of LLMs for complex datasets,
particularly in specialized domains. In addition,
future works will expand the study to more LLMs
and to different NLP tasks.
8

Limitations
In this study, we evaluate LLMs for data annotation
tasks and introduce a RAG-based approach
with different embedding models to enhance
performance on NER datasets. However, our
work has several limitations that highlight areas
for future research.
First, our experiments focus solely on NER
tasks. While this provides a solid foundation
for evaluation, extending the analysis to other
NLP tasks, such as text classification or question
answering, would offer a more comprehensive
understanding of the proposed methodology’s
applicability and generalizability.
Second, for the proof of concept, we employ a
naïve RAG approach for context selection. Future
work could explore more sophisticated retrieval
techniques, such as adaptive retrieval strategies,
re-ranking mechanisms, or hybrid approaches
combining dense and sparse retrieval, to further
optimize performance.
Third, our study does not explicitly examine the
biases introduced by LLMs in the data annotation
process. Given the growing concerns about fairness
and model biases, a deeper investigation into how
LLMs influence annotation patterns, especially
in diverse and underrepresented datasets, could
provide valuable insights.
References
Sergei Bogdanov, Alexandre Constantin, Timothée
Bernard, Benoit Crabbé, and Etienne Bernard. 2024.
Nuner: Entity recognition encoder pre-training via llm-
annotated data. Preprint , arXiv:2402.15343.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss, Gretchen
Krueger, Tom Henighan, Rewon Child, Aditya
Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler,
Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners. Preprint ,
arXiv:2005.14165.
William Jay Conover. 1999. Practical Nonparametric
Statistics , volume 350. John Wiley & Sons.
Leon Derczynski, Eric Nichols, Marieke van Erp, and
Nut Limsopatham. 2017. Results of the WNUT2017
shared task on novel and emerging entity recognition.InProceedings of the 3rd Workshop on Noisy User-
generated Text , pages 140–147, Copenhagen, Denmark.
Association for Computational Linguistics.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understanding.
ArXiv , abs/1810.04805.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan
Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu,
Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui.
2024. A survey on in-context learning. Preprint ,
arXiv:2301.00234.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The faiss library.
Federico Errica, Giuseppe Siracusano, Davide Sanvito,
and Roberto Bifulco. 2024. What did i do wrong?
quantifying llms’ sensitivity and consistency to prompt
engineering. Preprint , arXiv:2406.12334.
Steven Y . Feng, Varun Gangal, Jason Wei, Sarath
Chandar, Soroush V osoughi, Teruko Mitamura, and
Eduard Hovy. 2021. A survey of data augmentation
approaches for NLP. In Findings of the Association
for Computational Linguistics: ACL-IJCNLP 2021 ,
pages 968–988, Online. Association for Computational
Linguistics.
Akshay Goel, Almog Gueta, Omry Gilon, Chang Liu,
Sofia Erell, Lan Huong Nguyen, Xiaohong Hao, Bolous
Jaber, Shashir Reddy, Rupesh Kartha, Jean Steiner,
Itay Laish, and Amir Feder. 2023. Llms accelerate
annotation for medical information extraction. Preprint ,
arXiv:2312.02296.
Xingwei He, Zhenghao Lin, Yeyun Gong, A-Long Jin,
Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan
Duan, and Weizhu Chen. 2024a. Annollm: Making
large language models to be better crowdsourced
annotators. Preprint , arXiv:2303.16854.
Xingwei He, Zhenghao Lin, Yeyun Gong, A-Long Jin,
Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan
Duan, and Weizhu Chen. 2024b. AnnoLLM: Making
large language models to be better crowdsourced
annotators. In Proceedings of the 2024 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 6: Industry Track) , pages
165–190, Mexico City, Mexico. Association for
Computational Linguistics.
Hugging Face. 2023. Transformers APIs. https:
//huggingface.co/docs/transformers/index .
Accessed: 2023-01-21.
Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong
Zhu, Matthew Tang, Andrew Howard, Hartwig Adam,
and Dmitry Kalenichenko. 2018. Quantization and
training of neural networks for efficient integer-
arithmetic-only inference. In 2018 IEEE/CVF
9

Conference on Computer Vision and Pattern
Recognition , pages 2704–2713.
Mert Karabacak and Konstantinos Margetis. 2023.
Embracing large language models for medical
applications: Opportunities and challenges. Cureus ,
15(5):e39305.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
augmented generation for knowledge-intensive nlp
tasks. In Advances in Neural Information Processing
Systems , volume 33, pages 9459–9474. Curran
Associates, Inc.
Yinghao Li, Rampi Ramprasad, and Chao Zhang. 2024.
A simple but effective approach to improve structured
language model output for information extraction.
Preprint , arXiv:2402.13364.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du,
Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
2019. Roberta: A robustly optimized bert pretraining
approach. ArXiv , abs/1907.11692.
Yu Liu, Duantengchuan Li, Kaili Wang, Zhuoran Xiong,
Fobo Shi, Jian Wang, Bing Li, and Bo Hang. 2024.
Are llms good at structured outputs? a benchmark
for evaluating structured output capabilities in llms.
Information Processing & Management , 61(5):103809.
Yuji Naraki, Ryosuke Yamaki, Yoshikazu Ikeda,
Takafumi Horie, and Hiroki Naganuma. 2024.
Augmenting ner datasets with llms: Towards automated
and refined annotation. Preprint , arXiv:2404.01334.
OpenAI. 2023. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
D. Pereira, Anabela Afonso, and Fátima Medeiros.
2015. Overview of friedman’s test and post-hoc
analysis. Communications in Statistics - Simulation
and Computation , 44:2636–2653.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
bert: Sentence embeddings using siamese bert-networks.
Preprint , arXiv:1908.10084.
Stefan Strohmeier. 2022. Handbook of Research on
Artificial Intelligence in Human Resource Management .
Edward Elgar Publishing.
Zhen Tan, Dawei Li, Alimohammad Beigi, Song Wang,
Ruocheng Guo, Amrita Bhattacharjee, Bohan Jiang,
Mansooreh Karami, Jundong Li, Lu Cheng, and Huan
Liu. 2024. Large language models for data annotation:
A survey. ArXiv , abs/2402.13446.
Gemini Team. 2024a. Gemini: A family of
highly capable multimodal models. Preprint ,
arXiv:2312.11805.
Qwen Team. 2024b. Qwen2.5: A party of foundation
models.Maksim Terpilowski. 2019. scikit-posthocs: Pairwise
multiple comparison tests in python. The Journal of
Open Source Software , 4(36):1169.
Erik F. Tjong Kim Sang and Fien De Meulder. 2003.
Introduction to the CoNLL-2003 shared task: Language-
independent named entity recognition. In Proceedings
of the Seventh Conference on Natural Language
Learning at HLT-NAACL 2003 , pages 142–147.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open
and efficient foundation language models. Preprint ,
arXiv:2302.13971.
Muhammad Uzair Ul Haq, Paolo Frazzetto, Alessandro
Sperduti, and Giovanni Da San Martino. 2024.
Improving soft skill extraction via data augmentation
and embedding manipulation. In Proceedings of the
39th ACM/SIGAPP Symposium on Applied Computing ,
SAC ’24, page 987–996, New York, NY , USA.
Association for Computing Machinery.
Ashok Urlana, Charaka Vinayak Kumar, Ajeet Kumar
Singh, Bala Mallikarjunarao Garlapati, Srinivasa Rao
Chalamala, and Rahul Mishra. 2024. Llms with
industrial lens: Deciphering the challenges and
prospects – a survey. Preprint , arXiv:2402.14558.
Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang,
Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang.
2023. Gpt-ner: Named entity recognition via large
language models. Preprint , arXiv:2304.10428.
Shuohang Wang, Yang Liu, Yichong Xu, Chenguang
Zhu, and Michael Zeng. 2021. Want to reduce labeling
cost? GPT-3 can help. In Findings of the Association for
Computational Linguistics: EMNLP 2021 , pages 4195–
4205, Punta Cana, Dominican Republic. Association for
Computational Linguistics.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le,
and Denny Zhou. 2023. Chain-of-thought prompting
elicits reasoning in large language models. Preprint ,
arXiv:2201.11903.
Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan
Ning, Yu-Yang Liu, and Li Yuan. 2024. Llm lies:
Hallucinations are not bugs, but features as adversarial
examples. Preprint , arXiv:2310.01469.
Amir Zeldes. 2017. The GUM corpus: Creating
multilayer resources in the classroom. Language
Resources and Evaluation , 51(3):581–612.
Mike Zhang, Kristian Nørgaard Jensen, Sif Dam
Sonniks, and Barbara Plank. 2022a. Skillspan: Hard
and soft skill extraction from english job postings.
InNorth American Chapter of the Association for
Computational Linguistics .
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex
Smola. 2022b. Automatic chain of thought prompting
in large language models. Preprint , arXiv:2210.03493.
10

Ran Zhou, Xin Li, Ruidan He, Lidong Bing, Erik
Cambria, Luo Si, and Chunyan Miao. 2022. MELM:
Data augmentation with masked entity language
modeling for low-resource NER. In Proceedings
of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 2251–2262, Dublin, Ireland. Association for
Computational Linguistics.
11

A Datasets Statistics
Table 1: Statistics of the datasets considered in this study. The average entity length refers to the average number of
tokens for each entity.
DatasetSentences TokensAvg. Entity Length
Train Validation Test Train Validation Test
CoNLL-2003 14041 3250 3453 203621 51362 46435 1 .60
WNUT-2017 3394 1008 1287 62730 15734 23394 1 .73
GUM 1435 615 805 29392 12688 17437 3 .15
SKILLSPAN 3074 1396 1522 92621 39923 42541 4 .72
Table 1 highlights the complexity of entity mentions across different datasets, as reflected in their average
entity length. CoNLL-2003 and WNUT-2017 contain relatively short entities, with average lengths of
1.60and1.73tokens, respectively, indicating that most entities are single-token mentions. In contrast,
GUM exhibits greater complexity, with an average entity length of 3.15tokens, suggesting the presence
of multi-token entities. SKILLSPAN is the most complex dataset, with an average entity length of 4.72
tokens, implying more intricate entity structures that require advanced modeling techniques for accurate
recognition.
Moreover, we discuss below the entity information for each dataset.
CoNLL-2003 The CoNLL-2003 (Tjong Kim Sang and De Meulder, 2003) dataset consists of general
entity types: ( i)PERSON ; (ii)ORGANIZATION ; (iii)LOCATION ; and ( iv)MISCELLANEOUS .
Entities in this dataset typically follow structured patterns, making them relatively easier for LLMs
to identify and classify.
WNUT-17 The WNUT-17 (Derczynski et al., 2017) dataset contains six categories of rare entities:
(i)PERSON ; (ii)CORPORATION ; (iii)LOCATION ; (iv)CREATIVE _WORK ; (v)GROUP ; and
(vi)PRODUCT . This dataset is particularly challenging due to its noisy text, sparse entity occurrences,
and limited labeled examples per category.
GUM The GUM (Zeldes, 2017) dataset is a richly annotated corpus designed for multiple NLP tasks,
including NER. The dataset includes eleven distinct named entity types: ( i)ABSTRACT ; (ii)ANIMAL ;
(iii)EVENT ; (iv)OBJECT ; (v)ORGANIZATION ; (vi)PERSON ; (vii)PLACE ; (viii) PLANT ;
(ix)QUANTITY ; (x)SUBSTANCE ; and ( xi)TIME .
SKILLSPAN The SKILLSPAN (Zhang et al., 2022a) dataset is composed of a single entity type,
SOFTSKILLS , extracted from job descriptions. Unlike traditional entities, soft skills do not follow a fixed
syntactic or semantic structure, making them inherently ambiguous.
B Implementation Details
To perform experiments for data annotation with gpt-4o-mini , the model is accessed via the API service
provided by OpenAI. To ensure reproducible results, the temperature is set to 0and a seed value of 42
is used. Furthermore, the system fingerprint fp_1bb46167f9 is reported as noted during API access.
For data annotation generation using Qwen (Team, 2024b) and Llama (Touvron et al., 2023) based
models, the HuggingFace (Hugging Face, 2023) implementation is utilized. The instructed fine-tuned
variants of the open-source models are employed in the proposed study. The models are used only for
inference, with 4-bit quantization (Jacob et al., 2018). The experiments with billion scale models are
conducted on an A100GPU with a seed value of 42. All experiments to fine-tune NER task are performed
with the RoBERTa model, available via HuggingFace (Hugging Face, 2023), are conducted in a python
environment, on an RTX A5000 GPU. The experiments are performed using the following five seed
values: [23112 ,13215 ,6465,42,5634] . Moreover, the statistical significance tests are performed with the
help of scikit-posthocs (Terpilowski, 2019) library available in python.
12

C Complete Results
13

Table 2: The F 1, precision and recall along with standard deviation are reported on the test set. The values are averaged over five different random initializations. #Ex. represents
the number of context examples used. Baseline refers to the use of LLM with no context examples.
#Ex. MethodCoNLL2003 WNUT-17 GUM SKILLSPAN
P R F1 P R F1 P R F1 P R F1
Human 91.09±0.4993.17±0.1792.12±0.33 65.21±2.3247.48±1.8354.93±1.67 55.07±0.3161.86±0.4458.26±0.19 54.30±1.6055.38±1.7554.79±0.26
gpt-4o-mini-2024-07-18
Baseline 64.65±0.8580.37±0.5071.66±0.41 47.35±2.4655.18±2.8450.88±1.14 20.32±5.2613.93±2.7616.42±3.34 11.09±0.9717.83±2.0213.59±0.52
25ICL 76.48±0.4382.06±0.3579.17±0.25 53.18±3.2252.24±2.7352.58±0.78 44.06±0.6952.04±1.5747.71±0.79 21.23±1.4645.26±1.7228.86±1.24
RAG w/ST 84.48±1.0488.99±0.6586.68±0.85 51.42±2.6350.98±1.5251.14±1.01 46.09±0.6654.38±1.0749.89±0.70 20.29±0.7849.47±1.8028.77±0.93
RAG w/OpenAI 87.35±0.6590.71±0.3489.00±0.29 52.26±2.2449.75±1.5150.93±0.93 47.04±0.2357.56±1.4451.77±0.66 21.26±1.6956.74±1.3730.91±1.94
50ICL 79.77±0.3482.64±0.4981.18±0.29 55.75±2.8049.53±3.0752.33±0.97 45.12±0.8254.35±2.0249.28±0.88 20.56±0.8947.42±2.0128.66±0.85
RAG w/ST 86.73±1.0389.29±0.8487.99±0.90 53.74±3.0248.74±4.4450.90±1.34 46.46±1.3455.46±1.2150.56±1.29 22.22±1.4752.60±1.4131.20±1.32
RAG w/OpenAI 87.43±0.4891.39±0.1689.36±0.27 56.53±2.3550.29±2.6453.14±0.75 47.32±0.9258.44±1.2152.28±0.65 23.88±1.0954.28±2.2633.13±0.77
75ICL 78.74±1.0283.17±0.5580.89±0.66 51.90±4.2952.85±1.9552.24±1.76 44.40±0.6353.89±1.7948.67±0.69 20.84±1.5952.06±1.0129.73±1.58
RAG w/ST 86.91±0.3189.25±0.4488.06±0.26 53.80±1.7551.79±1.8852.73±0.80 47.22±0.9855.57±0.4351.05±0.60 21.39±0.8752.85±1.1030.43±0.73
RAG w/OpenAI 88.07±0.3591.44±0.2889.72±0.25 55.72±4.2251.71±3.3453.43±0.54 47.04±1.2958.19±1.1852.02±1.15 24.66±1.3455.39±3.1934.06±0.88
14

Table 3: The F 1, precision and recall along with standard deviation are reported on the test set. The values are averaged over five different random initializations. #Ex. represents
the number of context examples used. Baseline refers to the use of LLM with no context examples.
#Ex. MethodCoNLL2003 WNUT-17 GUM SKILLSPAN
P R F1 P R F1 P R F1 P R F1
Human 91.09±0.4993.17±0.1792.12±0.33 65.21±2.3247.48±1.8354.93±1.67 55.07±0.3161.86±0.4458.26±0.19 54.30±1.6055.38±1.7554.79±0.26
Qwen2.5-72B-Instruct
Baseline 26.97±0.2860.80±1.2537.36±0.30 16.40±1.3041.19±1.2323.43±1.42 6.32±0.2127.46±0.9710.28±0.35 4.89±0.4113.79±2.157.21±0.69
25ICL 74.57±0.7983.57±0.8278.81±0.30 45.58±2.6659.47±3.0951.49±0.92 41.69±1.1055.80±1.0047.73±1.06 17.06±2.1831.17±1.6322.01±2.13
RAG w/ST 81.87±0.7289.90±0.4685.69±0.56 46.55±2.7045.68±1.4346.06±1.33 47.79±1.0560.15±0.9953.26±0.89 18.25±2.0547.93±2.0826.40±2.37
RAG w/OpenAI 84.81±1.1691.68±0.6488.11±0.82 48.33±2.8249.88±1.8949.05±1.86 47.16±0.4659.83±0.6452.74±0.16 18.23±1.9050.07±4.4926.63±1.90
50ICL 77.48±0.5183.34±0.5380.30±0.43 45.04±1.7859.31±1.7151.17±1.16 44.30±1.0957.69±1.3550.12±1.14 17.51±0.8533.82±1.0123.06±0.86
RAG w/ST 84.30±0.9891.49±0.8587.74±0.77 45.60±2.8256.63±1.5250.45±1.14 48.83±1.4560.55±1.0554.06±1.25 21.32±1.8255.79±4.5630.84±2.52
RAG w/OpenAI 85.96±1.4492.32±0.3089.02±0.66 48.66±2.9157.09±2.4252.46±1.23 47.33±0.8161.23±0.4453.38±0.63 23.44±2.1952.32±2.8232.35±2.54
75ICL 77.50±0.6883.60±0.7480.43±0.67 52.14±2.2755.62±3.1153.72±0.80 47.60±0.7757.08±1.2251.91±0.80 20.81±1.1548.26±2.8029.05±1.26
RAG w/ST 87.46±0.3991.95±0.2989.65±0.31 48.36±3.2555.51±1.8851.58±1.04 50.29±0.2760.97±0.5155.11±0.17 20.99±2.1249.99±1.2329.52±2.10
RAG w/OpenAI 86.77±0.5492.05±0.7289.34±0.61 48.56±2.0860.22±1.5253.72±0.71 47.24±1.2760.34±0.5752.98±0.76 19.95±0.7450.74±1.4728.62±0.77
Llama3.5-70B-Instruct
Baseline 23.56±0.1063.25±0.1734.33±0.15 16.35±0.7454.65±0.4225.16±0.84 6.44±0.0827.79±0.3510.46±0.13 3.51±0.0824.30±0.646.14±0.12
25ICL 73.59±0.7878.73±1.0376.06±0.41 48.77±2.2047.66±5.1848.00±2.14 18.26±2.8041.83±0.9825.34±2.68 17.04±0.5245.86±2.8624.84±0.95
RAG w/ST 83.15±1.4286.37±0.9084.72±0.54 36.68±1.3249.10±3.7741.89±0.99 43.09±1.1050.88±2.3146.63±0.89 19.62±1.4446.47±1.7627.55±1.21
RAG w/OpenAI 68.32±3.9987.50±1.8276.65±2.19 43.52±4.3344.71±3.8643.82±1.29 42.46±1.7548.87±4.6045.29±1.50 19.59±1.5242.16±1.4926.73±1.65
50ICL 76.13±1.1276.79±1.2476.44±0.30 50.24±2.8148.90±2.2449.48±1.08 35.67±1.8348.79±3.1941.12±1.07 16.09±0.9744.15±4.1623.51±0.71
RAG w/ST 83.87±0.6988.57±0.8886.15±0.28 42.92±2.0348.79±2.9945.57±0.76 43.76±1.5050.24±2.0046.73±0.49 17.69±0.6646.11±4.6325.50±0.54
RAG w/OpenAI 68.36±1.5389.08±0.7577.35±0.97 44.14±1.9751.28±2.9447.36±0.64 43.70±2.4349.70±1.8346.45±1.40 18.37±2.4244.41±4.4425.77±1.75
75ICL 74.94±1.0375.15±1.0375.04±0.70 50.78±1.7451.69±2.4351.18±1.05 39.62±1.6447.88±3.0543.30±1.39 17.55±1.0551.80±1.6826.19±1.14
RAG w/ST 85.70±0.6089.03±0.5587.33±0.23 47.41±3.8951.36±1.9749.18±1.61 45.84±1.1950.36±1.1247.98±0.68 18.87±1.3551.17±2.0627.52±1.18
RAG w/OpenAI 76.99±1.5787.46±1.3981.87±0.67 49.43±4.2748.16±5.9948.39±2.17 44.46±0.6152.96±1.6548.33±0.64 9.51±1.6547.74±3.5115.83±2.47
15

Table 4: The F 1, precision and recall along with standard deviation are reported on the test set. The values are averaged over five different random initializations. #Ex. represents
the number of context examples used. Baseline refers to the use of LLM with no context examples.
#Ex. MethodCoNLL2003 WNUT-17 GUM SKILLSPAN
P R F1 P R F1 P R F1 P R F1
Human 91.09±0.4993.17±0.1792.12±0.33 65.21±2.3247.48±1.8354.93±1.67 55.07±0.3161.86±0.4458.26±0.19 54.30±1.6055.38±1.7554.79±0.26
Qwen2.5-7B-Instruct
Baseline 21.79±1.2862.11±0.4432.24±1.44 20.95±2.8344.08±3.6828.36±3.17 3.27±0.2214.10±1.035.31±0.37 5.41±1.0535.29±2.739.35±1.58
25ICL 70.22±1.4575.96±1.4972.95±0.30 47.79±3.4047.02±2.7847.29±1.63 28.31±1.0144.01±0.9734.43±0.57 14.12±0.8854.89±1.4822.44±1.08
RAG w/ST 83.81±0.6789.68±0.5786.64±0.45 37.82±3.4549.68±3.1242.81±2.14 35.90±1.8650.09±2.0441.80±1.65 15.99±0.3855.06±0.9324.77±0.42
RAG w/OpenAI 84.05±1.1590.85±0.3187.32±0.65 50.22±3.4341.75±4.8845.30±1.73 34.63±1.0949.97±1.0040.89±0.52 20.45±1.3554.60±4.8329.67±1.29
50ICL 72.55±1.0178.54±0.3275.42±0.57 47.95±3.1849.36±3.9448.54±2.51 33.51±0.7643.59±1.1837.88±0.56 15.13±0.9052.64±2.9823.47±1.01
RAG w/ST 85.78±0.6990.21±0.3887.94±0.43 52.14±4.7944.00±3.9547.41±0.49 39.38±1.3149.85±1.6943.97±0.44 17.36±1.0851.06±2.6525.87±1.02
RAG w/OpenAI 80.90±1.7991.55±0.3685.89±1.13 41.97±2.8748.62±5.6444.75±0.98 34.63±1.3250.61±1.6741.11±1.40 18.12±0.9456.68±3.9327.41±0.73
75ICL 81.36±1.1975.72±1.0078.43±0.63 47.90±5.7647.78±3.5747.51±1.97 34.23±2.1446.40±1.1039.39±1.77 12.98±1.1751.23±6.2020.68±1.81
RAG w/ST 86.54±1.9388.40±1.1587.44±0.87 52.39±4.7347.17±1.9949.48±1.30 40.14±1.3948.15±1.0943.76±0.73 18.34±0.4646.32±3.0826.25±0.76
RAG w/OpenAI 81.67±1.5190.96±0.3086.06±0.88 48.73±1.3147.85±2.4948.25±1.30 39.56±1.2550.87±1.2544.48±0.55 14.07±0.8061.07±0.8622.86±1.06
Llama-3.1-8B-Instruct
Baseline 22.98±0.6774.87±0.4835.17±0.83 11.06±2.7036.38±10.4016.88±4.16 6.98±0.0328.22±0.1811.19±0.05 3.03±0.2120.37±5.765.22±0.32
25ICL 63.86±0.9575.71±1.6169.26±0.69 35.94±3.5451.58±2.7042.23±2.38 33.95±1.9741.74±3.0237.39±1.85 12.40±0.8533.63±6.6517.93±0.66
RAG w/ST 78.44±1.1886.16±0.8882.11±0.86 36.82±4.6443.86±7.2239.38±2.26 39.94±2.2346.48±0.9642.92±1.20 14.95±1.8842.25±6.4521.87±1.51
RAG w/OpenAI 69.03±1.0286.41±2.2776.73±1.02 32.83±3.2048.82±7.4138.89±2.78 40.77±1.8249.07±2.8044.45±0.54 12.16±0.9741.55±3.2018.79±1.31
50ICL 67.78±1.4876.79±0.6972.01±1.13 40.49±1.7648.82±2.8844.20±1.03 36.43±1.5142.53±2.1239.22±1.32 12.94±1.1335.45±2.1218.90±1.01
RAG w/ST 79.29±3.8686.85±2.0082.82±1.41 40.04±4.2648.60±4.2643.59±1.04 39.73±1.8146.54±2.1242.81±0.99 15.13±0.9647.13±1.4822.88±0.99
RAG w/OpenAI 69.98±1.5087.05±2.1577.56±0.74 39.75±1.8249.53±2.6344.03±0.76 40.89±1.6246.96±2.2743.66±0.67 12.09±1.1842.63±3.1318.76±1.19
75ICL 71.44±2.0277.86±2.4174.47±1.00 39.13±1.1648.70±2.3743.35±0.84 34.33±1.2741.30±2.3737.43±0.53 13.37±1.1237.83±4.6819.67±1.20
RAG w/ST 82.36±2.1587.69±1.7084.91±0.88 39.92±2.0950.34±3.1244.42±0.59 41.14±1.4743.71±3.3642.30±1.57 12.77±0.1045.34±0.8019.92±0.08
RAG w/OpenAI 74.58±2.5185.21±0.9979.51±1.02 41.85±2.5247.17±6.9243.96±2.54 41.99±1.0146.13±2.6843.91±0.93 10.42±0.9649.98±3.6417.22±1.29
16

D Further Results on Different Sample Space Choices
Tables 5 examine the influence of sample space Xand context size Mon entity recognition performance
using the best-performing model, gpt-4o-mini , on the SKILLSPAN dataset. Increasing the context
size from 25to75generally improves the F1score, though gains diminish beyond 50examples. RAG
consistently outperforms ICL in recall and F1score, demonstrating its effectiveness in leveraging external
knowledge, while ICL achieves higher precision but lower recall, suggesting a more conservative prediction
approach. At a 10% sample space, ICL delivers competitive results, but as it increases to 20%, RAG
maintains a clear advantage, achieving the highest F1score of 32.39% at a context size of 75. Notably,
for smaller dataset splits, RAG exhibits greater variability, similar to ICL, suggesting that when fewer
examples are available, their performances converge. These findings underscore the importance of context
size and external knowledge availability in optimizing RAG-based methods.
Table 5: Study comparing RAG and ICL methods at different size of sample spaces (10% and 20%) and context
sizes (25, 50, and 75). Experiments were conducted on the SKILLSPAN dataset using the gpt-4o-mini-2024-07-18
model. The results are presented with standard deviations, showing how performance metrics vary across sampling
choices and context sizes for both methods.
Sample Space Context Size Precision Recall F1 Score
RAG
10%25 21 .83±1.22 56.94±1.17 31.53±1.18
50 22 .44±1.32 56.46±2.46 32.07±1.13
75 22 .82±0.58 55.82±1.4032.34±0.41
20%25 20 .26±1.55 54.46±3.71 29.45±1.29
50 21 .00±0.81 57.40±1.57 30.74±0.86
75 22 .69±0.46 56.31±2.0632.39±0.60
ICL
10%25 22 .57±1.49 48.72±3.95 30.74±0.87
50 23 .62±0.85 50.73±1.3332.21±0.67
75 23 .12±1.16 51.09±4.19 31.76±0.89
20%25 19 .35±1.57 45.83±4.74 27.05±0.66
50 22 .02±1.56 51.17±1.15 30.76±1.39
75 22 .89±1.11 49.78±2.8931.32±0.99
E Statistical Significance Test
This study evaluated various large language models across multiple datasets, considering different
embeddings and examples as context. While some models clearly outperformed others in the results,
the differences in predictions might not be statistically significant for certain models. Therefore, to
determine the statistical significance of our findings, we conducted a non-parametric test. This test helps
us assess whether there are significant differences among the models and, if so, identify which models
differ statistically from each other.
The Friedman test (Pereira et al., 2015) is a non-parametric statistical test used to detect differences in
performance across multiple related samples — in this case, different models evaluated over multiple
datasets. It ranks the performance scores among datasets and assesses whether the rank distributions differ
significantly among models. Let Nbe the number of datasets, Kthe number of models, and Rjbe the
sum of ranks for each model j. The Friedman test statistic chi2
F, which follows a chi-square distribution,
is calculated as follows:
χ2
F=12N
k(k+ 1)kX
j=1R2
j−3N(k+ 1). (1)
If the test statistic exceeds the critical value for a significance level α= 0.01, we reject the null hypothesis,
indicating that there are significant differences in performance among the models. If significant differences
are found, the post-hoc Conover (Conover, 1999) test is performed to discover pair-wise statistical
17

Figure 4: Critical Difference diagram of average score ranks. The models connected with horizontal line shows no
statistical difference. The models with lower ranks shows superior performance than those of higher ranks.
differences among models while adjusting for multiple comparisons. This test evaluates whether specific
models differ significantly in performance.
Given that the Friedman test produces a test statistic of 114.42with a p-value of 7.71−18, we reject the null
hypothesis, suggesting that at least one model shows a statistically significant difference in performance.
Consequently, we conducted the post-hoc Conover test. Figure 4 presents the statistical significance of
the model rankings, with significant pairwise differences highlighted accordingly. The x-axis indicates
the average rank of each model, where lower ranks closer to the left signify better performance. Each
colored node corresponds to a particular model, labeled with its respective rank, while the black horizontal
bars connecting multiple nodes highlight groups of models that do not show statistically significant
differences at the specified confidence level. The top-performing combination is gpt4omini-OpenAI,
with an average rank of 1.9, indicating it consistently outperformed other approaches. Other strong
performers include Qwen2.5-72B-OpenAI (3),gpt-4o-mini-ST (3.8), and Qwen2.5-72B-ST (4.3). These
models have lower rankings and are clustered towards the left. In contrast, Llama3.1-8B-ICL (14),
Llama3.1-8B-OpenAI (13), and Qwen2.5-7B-ICL (11) have the highest ranks, suggesting they performed
the worst in comparison. These models do not overlap with the higher-ranked ones, highlighting their
statistically inferior performance. Interestingly, Llama3.1-8B-ST shows no statistical differences when
compared to Llama3.5-70B , whether using ICL or RAG with OpenAI embedding. Similarly, Qwen2.5-7B ,
when utilizing RAG with either OpenAI or ST embeddings, exhibits no statistical differences compared to
Llama3.5-70B using ST embeddings and Qwen2.5-72B using ICL. These tests highlight a crucial aspect:
a trade-off when addressing the NER task. Indeed, larger models, such as those with 70B parameters, may
not necessarily offer better performance than smaller models like Llama3.1-8B-ST or Qwen2.5-7B . This
suggests that the additional computational resources required for bigger models might not always justify
their use, especially if smaller models can achieve statistically similar results.
F Qualitative Analysis
This study broadly explores the efficacy of LLMs for data annotation tasks. Four different datasets of
varying complexity are chosen. From Table 2, it is observed that the performance of LLMs decreases
as dataset complexity increases. The performance of LLMs on the SKILLSPAN dataset is significantly
lower than human annotation, suggesting that even the latest available LLMs struggle to annotate data
when the task is complex. For instance, soft skills lack clear or distinct definitions, making the task more
challenging. Similarly, the GUM dataset also poses challenges for LLMs due to its entity diversity. On
the other hand, in the case of the WNUT-17 and CoNLL-2003 datasets, which consist of simpler entities
(more details are reported in Section 4.1), annotations are easier to extract for an LLM given its prior
knowledge. Furthermore, the quality of context in LLMs plays a major role, particularly in data annotation
tasks, as indicated by Tables 2, 3, and 4, where the RAG-based approach significantly outperforms its
counterpart. Moreover, for simpler datasets, the RAG-based approach achieves performance comparable
to human annotation.
To gain better insights into the performance of the proposed RAG-based approach, Table 6 presents the
qualitative results for the SKILLSPAN dataset annotated by gpt-4o-mini . In this dataset, data annotation
performance remains far below human-level, suggesting that the LLM struggles to extract sufficient
18

Table 6: Qualitative analysis of soft skills annotations on dataset samples using gpt-4o-mini-2024-07-18 . The
output of the best-performing model is reported. The highlighted texts in the first column are gold labels, while
those in the other columns are the corresponding LLM-generated annotations.
№ Human Baseline ICL RAG
1. Very good understanding
of test automation
frameworks.Very good understanding
of test automation
frameworks.Very good understanding
of test automation
frameworks.Very good understanding
of test automation
frameworks.
2. Must have excellent verbal
and written skills being
able to communicate
effectively on both a
technical and business
level Ability to work under
pressure to resolve issues
affecting the production
services.Must have excellent verbal
and written skills being
able to communicate
effectively on both a
technical and business
level Ability to work under
pressure to resolve issues
affecting the production
services.Must have excellent verbal
and written skills being
able to communicate
effectively on both a
technical and business
level Ability towork under
pressure toresolve issues
affecting the production
services.Must have excellent verbal
and written skills being
able to communicate
effectively on both a
technical and business
level Ability to work under
pressure to resolve issues
affecting the production
services.
3. Must have excellent work
ethic and be detail oriented
and be able to work
independently.Must have excellent work
ethic and be detail oriented
and be able to work
independently.Must have excellent work
ethic and be detail oriented
and be able to work
independently.Must have excellent work
ethic and be detail oriented
and be able to work
independently.
4. Technical Skills Core Java. Technical Skills Core Java. Technical Skills Core Java. Technical Skills Core Java.
5. You will work with
the business to define
requirements and have
excellent communication
skills to interpret these into
consolidated development
scopes.You will work with
the business to define
requirements and have
excellent communication
skills to interpret these into
consolidated development
scopes.You will work with
the business to define
requirements and have
excellent communication
skills to interpret these into
consolidated development
scopes.You will work with
the business to define
requirements and have
excellent communication
skills to interpret these into
consolidated development
scopes.
information from the context examples when the task is difficult. From Tables 2, 3, and 4, it is observed
that LLM-generated annotations improve recall, whereas precision is compromised. Table 6 shows that in
examples 1and4, the LLM incorrectly annotates soft skills that are not identified by human annotators,
whereas in examples 2and3, the annotations are nearly identical to human annotations. In Example 5,
the RAG-based approach performs comparably to human annotation, while both the baseline and ICL fail
to do so.
G Prompt
This section presents the prompts used to generate the response of LLMs. These prompts are carefully
synthesized to encompass all the components required to get structured output for both: ( i)baseline, and
(ii)in-context learning models.
19

Baseline Prompt Structure
Task Description
You are an advanced Named-Entity Recognition (NER) system.
Your task is to analyze the given sentence or passage, identify, extract, and classify specific named entities according to
the following predefined entity types:
• {labels}
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation.
In entities, label the word exactly as in the text. All the text is case-sensitive.
Input
{input_text}
Context Prompt Structure
Task Description
You are an advanced Named-Entity Recognition (NER) system.
Your task is to analyze the given sentence or passage, identify, extract, and classify specific named entities according to
the following predefined entity types:
• {labels}
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation.
In entities, label the word exactly as in the text. All the text is case-sensitive.
Examples
{context_examples}
Input
{input_text}
H Examples
This section provides examples of prompts from the training data for different datasets used in this study.
For visual purposes, we used only only top5examples in context. Follows several prompt examples for
the: ( i)CoNLL-2003, ( ii)WNUT-17, ( iii)SKILLSPAN datasets, and ( iv)GUM datasets.
Example 1–CoNLL-2003
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’PER’, ’ORG’, ’LOC’, ’MISC’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ "A South A f r i c a n boy i s w r i t i n g back t o an American g i r l whose message i n a
b o t t l e he found washed up on P r e s i d e n t Nelson Mandela ’ s o l d p r i s o n i s l a n d
. " , [ { ’ E n t i t y ’ : ’ South A f r i c a n ’ , ’ Label ’ : ’MISC ’ } , { ’ E n t i t y ’ : ’ American ’ ,
20

’ Label ’ : ’MISC ’ } , { ’ E n t i t y ’ : ’ Nelson Mandela ’ , ’ Label ’ : ’PER ’ } ] ]
[ ’A r o t t w e i l e r dog b e l o n g i n g t o an e l d e r l y South A f r i c a n c o u p l e savaged t o
d e a t h t h e i r two − year − o l d g r a n d s o n who was v i s i t i n g , p o l i c e s a i d on
Thursday . ’ , [ { ’ E n t i t y ’ : ’ South A f r i c a n ’ , ’ Label ’ : ’MISC ’ } ] ]
[ ’ The p r i n c e s s , who has c a r v e d o u t a major r o l e f o r h e r s e l f as a h e l p e r of
t h e s i c k and needy , i s s a i d t o have t u r n e d t o Mother T e r e s a f o r g u i d a n c e
as h e r m a r r i a g e crumbled t o h e i r t o t h e B r i t i s h t h r o n e P r i n c e C h a r l e s . ’ ,
[ { ’ E n t i t y ’ : ’ Mother Teresa ’ , ’ Label ’ : ’PER ’ } , { ’ E n t i t y ’ : ’ B r i t i s h ’ , ’ Label
’ : ’MISC ’ } , { ’ E n t i t y ’ : ’ P r i n c e C h a r l e s ’ , ’ Label ’ : ’PER ’ } ] ]
[ ’ South A f r i c a n answers U. S . message i n a b o t t l e . ’ , [ { ’ E n t i t y ’ : ’ South
A f r i c a n ’ , ’ Label ’ : ’MISC ’ } , { ’ E n t i t y ’ : ’U. S . ’ , ’ Label ’ : ’LOC’ } ] ]
[ " But C a r l o Hoffmann , an 11− year − o l d j a i l e r ’ s son who found t h e b o t t l e on
t h e beach a t Robben I s l a n d o f f Cape Town a f t e r w i n t e r s t o r m s , w i l l send
h i s l e t t e r back by o r d i n a r y m ai l on Thursday , t h e p o s t o f f i c e s a i d . " ,
[ { ’ E n t i t y ’ : ’ C a r l o Hoffmann ’ , ’ Label ’ : ’PER ’ } , { ’ E n t i t y ’ : ’ Robben I s l a n d ’ ,
’ Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Cape Town ’ , ’ Label ’ : ’LOC’ } ] ]
Input
Revered skull of S. Africa king is Scottish woman ’s .
Response
[Entity: S. Africa, Label: LOC, Entity: Scottish, Label: MISC]
Example 2–CoNLL-2003
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’PER’, ’ORG’, ’LOC’, ’MISC’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ Rwanda s a i d on S a t u r d a y t h a t Z a i r e had e x p e l l e d 28 Rwandan Hutu r e f u g e e s
a c c u s e d of b e i n g " t r o u b l e −makers " i n camps i n e a s t e r n Z a i r e . ’ , [ { ’
E n t i t y ’ : ’ Rwanda ’ , ’ Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Z a i r e ’ , ’ Label ’ : ’LOC’ } ,
{ ’ E n t i t y ’ : ’ Rwandan ’ , ’ Label ’ : ’MISC ’ } , { ’ E n t i t y ’ : ’ Hutu ’ , ’ Label ’ : ’MISC
’ } , { ’ E n t i t y ’ : ’ Z a i r e ’ , ’ Label ’ : ’LOC’ } ] ]
[ ’ R e p a t r i a t i o n of 1 . 1 m i l l i o n Rwandan Hutu r e f u g e e s announced by Z a i r e and
Rwanda on Thursday c o u l d s t a r t w i t h i n t h e n e x t few days , an e x i l e d
Rwandan Hutu lobby group s a i d on F r i d a y . ’ , [ { ’ E n t i t y ’ : ’ Rwandan Hutu ’ , ’
Label ’ : ’MISC ’ } , { ’ E n t i t y ’ : ’ Z a i r e ’ , ’ Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Rwanda ’ ,
’ Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Rwandan Hutu ’ , ’ Label ’ : ’MISC ’ } ] ]
[ ’ I n n o c e n t B u t a r e , e x e c u t i v e s e c r e t a r y of t h e R a l l y f o r t h e R e t u r n of
Refugees and Democracy i n Rwanda ( RDR ) which s a y s i t has t h e s u p p o r t of
Rwanda \ ’ s e x i l e d Hutus , a p p e a l e d t o t h e i n t e r n a t i o n a l community t o d e t e r
t h e two c o u n t r i e s from going ahead wi th what i t termed a " f o r c e d and
inhuman a c t i o n " . ’ , [ { ’ E n t i t y ’ : ’ I n n o c e n t Butare ’ , ’ Label ’ : ’PER ’ } , { ’
E n t i t y ’ : ’ R a l l y f o r t h e R e t u r n of Refugees and Democracy i n Rwanda ’ , ’
Label ’ : ’ORG’ } , { ’ E n t i t y ’ : ’RDR’ , ’ Label ’ : ’ORG’ } , { ’ E n t i t y ’ : ’ Rwanda ’ , ’
Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Hutus ’ , ’ Label ’ : ’MISC ’ } ] ]
21

[ ’ Rwanda s a y s Z a i r e e x p e l s 28 Rwandan r e f u g e e s . ’ , [ { ’ E n t i t y ’ : ’ Rwanda ’ , ’
Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Z a i r e ’ , ’ Label ’ : ’LOC’ } , { ’ E n t i t y ’ : ’ Rwandan ’ ,
’ Label ’ : ’MISC ’ } ] ]
[ ’ Rwandan group s a y s e x p u l s i o n c o u l d be imminent . ’ , [ { ’ E n t i t y ’ : ’ Rwandan ’ , ’
Label ’ : ’MISC ’ } ] ]
Input
Captain Firmin Gatera , spokesman for the Tutsi-dominated Rwandan army , told Reuters in Kigali that 17 of the 28
refugees handed over on Friday from the Zairean town of Goma had been soldiers in the former Hutu army which fled
to Zaire in 1994 after being defeated by Tutsi forces in Rwanda ’s civil war .
Response
[Entity: Captain Firmin Gatera, Label: PER, Entity: Rwandan, Label: MISC, Entity: Reuters, Label: ORG, Entity:
Kigali, Label: LOC, Entity: Zairean, Label: MISC, Entity: Goma, Label: LOC, Entity: Hutu, Label: MISC, Entity:
Zaire, Label: LOC, Entity: Tutsi, Label: MISC, Entity: Rwanda, Label: LOC]
Example 3–WNUT-17
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’corporation’, ’creative-work’, ’group’, ’location’, ’person’, ’product’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ @ j u s t i n b i e b e r i j u s t wanna say you make me s m i l e e v e r y d a y : ) t h a n k s f o r
s m i l i n g b e c a u s e when u s m i l e i s m i l e ! : ) ’ , [ ] ]
[ " @joeymcintyre I h e a r t you . Even i f I haven ’ t se e n u i n months . . . SEND A
PIC ! " , [ ] ]
[ ’ @ l o v a b l e _ s i n O M G O M G O M G ! Thank you f o r " t u m b l r i n g " i t t o me , I so wasn
\ ’ t e x p e c t i n g them t o d a y . O M G ! ’ , [ ] ]
[ ’RT @aplusk : Th is made me l a u g h t o d a y h t t p : / / b i t . l y / bjOhom & l t ; −−− c o u r t e s y
of s p l u r b . What made you l a u g h ? ’ , [ ] ]
[ ’RT @Sn00ki : Haha yes ! ! ! I l o v e t h a t you knew t h a t : ) RT @ t r i s h a m e l i s s a
@Sn00ki I s phenomenal t h e word of t h e day ? ’ , [ ] ]
Input
@jimmyfallon is following me ! OMG ! My life is now complete ! I heart you JF and have for years ! Thank you for
making me laugh everyday !
Response
[Entity: @jimmyfallon, Label: person, Entity: JF, Label: person]
Example 4–WNUT-17
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
22

[’corporation’, ’creative-work’, ’group’, ’location’, ’person’, ’product’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’We a r e one s t e p c l o s e r t o our new k i t c h e n s . We chose a maker and had
o f f i c i a l measurements t a k e n t o d a y ! ’ , [ ] ]
[ ’We were a l l e n j o y i n g a g l a s s of wine i n t h e o f f i c e when a fudge d e l i v e r y
showed up . I l o v e my j o b . And I l o v e F r i d a y s . ’ , [ ] ]
[ ’800 m i l e s t o s e e c l i e n t s , 3 ACC c a n d i d a t e / commissioner m e e t i n g s , b i g p r e s s
r e l e a s e , making i t t o F r i d a y . . PRICELESS ! ’ , [ ] ]
[ " I hope t h e weeks keep f l y i n g . I t ’ s a c t u a l l y f a n t a s t i c t h e way none of t h e
days dragged t h i s week . . . . l i k e NONE . :D" , [ ] ]
[ ’ F e e l i n g r e a l l y good a f t e r g r e a t week i n our SF and LA o f f i c e s . Glad t o k i c k
back on AMerican f l i g h t back t o NYC’ , [ { ’ E n t i t y ’ : ’SF ’ , ’ Label ’ : ’
l o c a t i o n ’ } , { ’ E n t i t y ’ : ’LA’ , ’ Label ’ : ’ l o c a t i o n ’ } , { ’ E n t i t y ’ : ’ AMerican ’ ,
’ Label ’ : ’ c o r p o r a t i o n ’ } , { ’ E n t i t y ’ : ’NYC’ , ’ Label ’ : ’ l o c a t i o n ’ } ] ]
Input
Great week in the Optimise office, another new client on board and we are close to signing a new team member
Response
[Entity: Optimise, Label: corporation]
Example 5–SKILLSPAN
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’Skill’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ Hands on e x p e r i e n c e wi th automated t e s t i n g u s i n g J a va . ’ , [ ] ]
[ ’ E x p e r i e n c e wi th a u t o m a t i o n s y s t e m s framework d e s i g n / use and deployment . ’ ,
[ ] ]
[ ’ Good u n d e r s t a n d i n g of A g i l e m e t h o d o l o g i e s and C o n t i n u o u s D e l i v e r y . ’ , [ ] ]
[ ’ Demon strate c l e a r u n d e r s t a n d i n g of a u t o m a t i o n and o r c h e s t r a t i o n p r i n c i p l e s
. ’ , [ ] ]
[ ’ Good e x p o s u r e t o UI Frameworks l i k e Angular P r o f i c i e n c y i n SQL and D a t a b a s e
development . ’ , [ ] ]
23

[ " A b i l i t y t o u n d e r s t a n d and use e f f i c i e n t D e f e c t management r e g u l a r view of
t e s t c o v e r a g e t o i d e n t i f y gaps and p r o v i d e improvements P e r s o n a l
S p e c i f i c a t i o n 5+ y e a r s of r e l e v a n t IT / q u a l i t y a s s u r a n c e work e x p e r i e n c e
Bachelor ’ s d e g r e e i n Computer S c i e n c e or r e l a t e d f i e l d of s t u d y or
e q u i v a l e n t r e l e v a n t e x p e r i e n c e ; d e m o n s t r a t e d e x p e r i e n c e w i t h i n t h e q u a l i t y
a s s u r a n c e / t e s t i n g a r e n a ; d e m o n s t r a t e d s k i l l s i n q u a l i t y a s s u r a n c e
methods / p r o c e s s e s and p r a c t i c e s . " , [ { ’ E n t i t y ’ : ’ u n d e r s t a n d and use
e f f i c i e n t D e f e c t management ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ i d e n t i f y gaps
’ , ’ Label ’ : ’ S k i l l ’ } ] ]
Input
Very good understanding of test automation frameworks.
Response
[Entity: test automation frameworks, Label: Skill]
Example 6–SKILLSPAN
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’Skill’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ S t r o n g communication s k i l l s i n c l u d i n g t h e a b i l i t y t o e x p r e s s complex
t e c h n i c a l c o n c e p t s t o d i f f e r e n t a u d i e n c e s i n w r i t i n g and c o n f e r e n c e c a l l s
. ’ , [ { ’ E n t i t y ’ : ’ communication s k i l l s ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’
e x p r e s s complex t e c h n i c a l c o n c e p t s t o d i f f e r e n t a u d i e n c e s ’ , ’ Label ’ : ’
S k i l l ’ } ] ]
[ ’ E x c e l l e n t o r g a n i z a t i o n a l v e r b a l and w r i t t e n communication s k i l l s . ’ , [ { ’
E n t i t y ’ : ’ o r g a n i z a t i o n a l v e r b a l and w r i t t e n communication s k i l l s ’ , ’ Label
’ : ’ S k i l l ’ } ] ]
[ ’ E x c e l l e n t o r g a n i z a t i o n a l v e r b a l and w r i t t e n communication s k i l l s . ’ , [ { ’
E n t i t y ’ : ’ o r g a n i z a t i o n a l v e r b a l and w r i t t e n communication s k i l l s ’ , ’ Label
’ : ’ S k i l l ’ } ] ]
[ ’ The a b i l i t y t o work w i t h i n a team and i n c o l l a b o r a t i o n w ith o t h e r s i s
c r i t i c a l t o t h i s p o s i t i o n and e x c e l l e n t communication s k i l l s v e r b a l and
w r i t t e n a r e e s s e n t i a l . ’ , [ { ’ E n t i t y ’ : ’ work w i t h i n a team and i n
c o l l a b o r a t i o n w ith o t h e r s ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ communication
s k i l l s ’ , ’ Label ’ : ’ S k i l l ’ } ] ]
[ ’ Th is r o l e r e q u i r e s a wide v a r i e t y of s t r e n g t h s and c a p a b i l i t i e s i n c l u d i n g
A b i l i t y t o work c o l l a b o r a t i v e l y i n teams and d e v e l o p m e a n i n g f u l
r e l a t i o n s h i p s t o a c h i e v e common g o a l s S t r o n g o r g a n i z a t i o n a l s k i l l s A b i l i t y
t o m u l t i − t a s k and d e l i v e r t o a t i g h t d e a d l i n e E x c e l l e n t w r i t t e n and
v e r b a l communication s k i l l s E x p e r i e n c e d e v e l o p i n g UI components i n Angular
Good e x p e r i e n c e i n u s i n g d e s i g n p a t t e r n s UML OO c o n c e p t s . ’ , [ { ’ E n t i t y ’ :
’ work c o l l a b o r a t i v e l y i n teams ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ d e v e l o p
m e a n i n g f u l r e l a t i o n s h i p s ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ a c h i e v e common
g o a l s ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ o r g a n i z a t i o n a l s k i l l s ’ , ’ Label ’ : ’
S k i l l ’ } , { ’ E n t i t y ’ : ’ m u l t i − t a s k ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ d e l i v e r
t o a t i g h t d e a d l i n e ’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ communication s k i l l s
’ , ’ Label ’ : ’ S k i l l ’ } , { ’ E n t i t y ’ : ’ d e v e l o p i n g UI components ’ , ’ Label ’ : ’
24

S k i l l ’ } , { ’ E n t i t y ’ : ’ u s i n g d e s i g n p a t t e r n s ’ , ’ Label ’ : ’ S k i l l ’ } ] ]
Input
Must have excellent verbal and written skills being able to communicate effectively on both a technical and business
level Ability to work under pressure to resolve issues affecting the production services .
Response
[Entity: verbal and written skills, Label: Skill, Entity: communicate effectively on both a technical and business level,
Label: Skill, Entity: work under pressure, Label: Skill, Entity: resolve issues affecting the production services, Label:
Skill]
Example 7–GUM
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’abstract’, ’animal’, ’event’, ’object’, ’organization’, ’person’, ’place’, ’plant’, ’quantity’, ’substance’, ’time’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ The 131− page document was found on C a s t l e f r a n k Road i n Kanata , O n t a r i o i n
a r a i n − s t a i n e d , t i r e −marked brown e n v e l o p e by a p a s s e r b y ’ , ’ E n t i t i e s ’ :
[ { ’ E n t i t y ’ : ’ The 131− page document was found on C a s t l e f r a n k Road i n Kanata
, O n t a r i o i n a r a i n − s t a i n e d , t i r e −marked brown e n v e l o p e by a p a s s e r b y ’ ,
’ Label ’ : ’ event ’ } ] ]
[ ’ Also t h e l a n g u a g e i s i m p o r t a n t i n w r i t i n g and i n l i t e r a t u r e ’ , ’ E n t i t i e s ’ :
[ { ’ E n t i t y ’ : ’ t h e language ’ , ’ Label ’ : ’ a b s t r a c t ’ } , { ’ E n t i t y ’ : ’ w r i t i n g ’ , ’
Label ’ : ’ a b s t r a c t ’ } , { ’ E n t i t y ’ : ’ l i t e r a t u r e ’ , ’ Label ’ : ’ a b s t r a c t ’ } ] ]
[ ’ I n g r e d i e n t s B a s i l comes i n many d i f f e r e n t v a r i e t i e s , each of which have a
u n i q u e f l a v o r and smell ’ , ’ E n t i t i e s ’ : [ { ’ E n t i t y ’ : ’ I n g r e d i e n t s ’ , ’ Label ’ :
’ o b j e c t ’ } , { ’ E n t i t y ’ : ’ B a s i l ’ , ’ Label ’ : ’ p l a n t ’ } , { ’ E n t i t y ’ : ’ many
d i f f e r e n t v a r i e t i e s ’ , ’ Label ’ : ’ a b s t r a c t ’ } , { ’ E n t i t y ’ : ’ each of which ’ , ’
Label ’ : ’ a b s t r a c t ’ } , { ’ E n t i t y ’ : ’ a u n i q u e f l a v o r and smell ’ , ’ Label ’ : ’
a b s t r a c t ’ } ] ]
[ ’We do n o t want t o j u s t t r a f f i c i n t h e same 24 hour news c y c l e ’ , ’ E n t i t i e s ’ :
[ { ’ E n t i t y ’ : ’We do n o t want t o j u s t t r a f f i c i n t h e same 24 hour news c y c l e
’ , ’ Label ’ : ’ a b s t r a c t ’ } ] ]
[ ’ You go t h r o u g h q u i t e a b i t ’ , ’ E n t i t i e s ’ : [ { ’ E n t i t y ’ : ’You ’ , ’ Label ’ : ’ person
’ } , { ’ E n t i t y ’ : ’ q u i t e a b i t ’ , ’ Label ’ : ’ q u a n t i t y ’ } ] ]
Input
If you are just visiting York for the day , using a Park and Ride [ 1 ] costs a lot less than trying to park in or near the city
centre , and there are five sites dotted around the Outer Ring Road
Response
[’Entity’: ’York’, ’Label’: ’place’, ’Entity’: ’the day’, ’Label’: ’time’, ’Entity’: ’a Park and Ride’, ’Label’: ’object’,
’Entity’: ’the city centre’, ’Label’: ’place’, ’Entity’: ’five sites’, ’Label’: ’quantity’, ’Entity’: ’the Outer Ring Road’,
’Label’: ’place’]
25

Example 8–GUM
Task Description
You are an advanced Named-Entity Recognition (NER) system. Your task is to analyze the given sentence or passage,
identify, extract, and classify specific named entities according to the following predefined entity types:
[’abstract’, ’animal’, ’event’, ’object’, ’organization’, ’person’, ’place’, ’plant’, ’quantity’, ’substance’, ’time’]
For each sentence:
•Label each word in the text with the appropriate entity type if it matches the specified categories.
• Extract multiple entities of the same class if they exist.
The output should be in valid JSON format , with each word and its corresponding label as shown below.
Follow the structure strictly and do not add any other explanation. In entities, label the word exactly as in the text. All
the text is case-sensitive.
Examples
[ ’ " NASA A d m i n i s t r a t o r C h a r l e s Bolden announces where f o u r s p a c e s h u t t l e
o r b i t e r s w i l l be p e r m a n e n t l y d i s p l a y e d a t t h e c o n c l u s i o n of t h e Space
S h u t t l e Program d u r i n g an e v e n t commemorating t h e 30 t h a n n i v e r s a y of t h e
f i r s t s h u t t l e l a u n c h on A p r i l 12 , 2011 ’ , ’ E n t i t i e s ’ : [ { ’ E n t i t y ’ : ’NASA
A d m i n i s t r a t o r C h a r l e s Bolden ’ , ’ Label ’ : ’ person ’ } , { ’ E n t i t y ’ : ’ where f o u r
s p a c e s h u t t l e o r b i t e r s w i l l be p e r m a n e n t l y d i s p l a y e d ’ , ’ Label ’ : ’ p l a c e ’ } ,
{ ’ E n t i t y ’ : ’ t h e c o n c l u s i o n of t h e Space S h u t t l e Program ’ , ’ Label ’ : ’ event
’ } , { ’ E n t i t y ’ : ’ an event ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’30 t h a n n i v e r s a y
of t h e f i r s t s h u t t l e launch ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’ A p r i l 12 ,
2011 ’ , ’ Label ’ : ’ time ’ } ] ]
[ ’NASA c e l e b r a t e d t h e l a u n c h of t h e f i r s t s p a c e s h u t t l e Tuesday a t an e v e n t a t
t h e Kennedy Space C e n t e r ( KSC ) i n Cape C a n a v e r a l , F l o r i d a ’ , ’ E n t i t i e s
’ : [ { ’ E n t i t y ’ : ’NASA’ , ’ Label ’ : ’ o r g a n i z a t i o n ’ } , { ’ E n t i t y ’ : ’ t h e l a u n c h of
t h e f i r s t s p a c e s h u t t l e ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’ Tuesday ’ , ’ Label
’ : ’ time ’ } , { ’ E n t i t y ’ : ’ an event ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’ Kennedy
Space Center ’ , ’ Label ’ : ’ p l a c e ’ } , { ’ E n t i t y ’ : ’KSC’ , ’ Label ’ : ’ p l a c e ’ } , { ’
E n t i t y ’ : ’ Cape C a n a v e r a l , F l o r i d a ’ , ’ Label ’ : ’ p l a c e ’ } ] ]
[ ’ Looking back : Space S h u t t l e Columbia l i f t s o f f on STS−1 from Launch Pad 39A
a t t h e Kennedy Space C e n t e r on A p r i l 12 , 1981 ’ , ’ E n t i t i e s ’ : [ { ’ E n t i t y ’ :
’ Space S h u t t l e Columbia ’ , ’ Label ’ : ’ o b j e c t ’ } , { ’ E n t i t y ’ : ’STS −1 ’ , ’ Label ’ :
’ event ’ } , { ’ E n t i t y ’ : ’ Launch Pad 39A’ , ’ Label ’ : ’ p l a c e ’ } , { ’ E n t i t y ’ : ’
Kennedy Space Center ’ , ’ Label ’ : ’ p l a c e ’ } , { ’ E n t i t y ’ : ’ A p r i l 12 , 1981 ’ , ’
Label ’ : ’ time ’ } ] ]
[ ’ At t h e ceremony , NASA A d m i n i s t r a t o r C h a r l e s Bolden announced t h e l o c a t i o n s
t h a t would be g i v e n t h e t h r e e r e m a i n i n g Space S h u t t l e o r b i t e r s f o l l o w i n g
t h e end of t h e Space S h u t t l e program ’ , ’ E n t i t i e s ’ : [ { ’ E n t i t y ’ : ’ t h e
ceremony ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’NASA A d m i n i s t r a t o r C h a r l e s
Bolden ’ , ’ Label ’ : ’ person ’ } , { ’ E n t i t y ’ : ’ t h e l o c a t i o n s ’ , ’ Label ’ : ’ p l a c e
’ } , { ’ E n t i t y ’ : ’ t h e t h r e e r e m a i n i n g Space S h u t t l e o r b i t e r s ’ , ’ Label ’ : ’
o b j e c t ’ } , { ’ E n t i t y ’ : ’ t h e end of t h e Space S h u t t l e program ’ , ’ Label ’ : ’
event ’ } ] ]
[ ’ On A p r i l 12 , 1981 , Space S h u t t l e Columbia l i f t e d o f f from t h e Kennedy
Space C e n t e r on STS−1 , t h e f i r s t s p a c e s h u t t l e mission ’ , ’ E n t i t i e s ’ : [ { ’
E n t i t y ’ : ’ A p r i l 12 , 1981 ’ , ’ Label ’ : ’ time ’ } , { ’ E n t i t y ’ : ’ Space S h u t t l e
Columbia ’ , ’ Label ’ : ’ o b j e c t ’ } , { ’ E n t i t y ’ : ’ Kennedy Space Center ’ , ’ Label ’ :
’ p l a c e ’ } , { ’ E n t i t y ’ : ’STS −1 ’ , ’ Label ’ : ’ event ’ } , { ’ E n t i t y ’ : ’ t h e f i r s t
s p a c e s h u t t l e mission ’ , ’ Label ’ : ’ event ’ } ] ]
Input
Tuesday , September 22 , 2015 Discovery is undergoing decommissioning and currently being prepped for display by
removing toxic materials from the orbiter
Response
[’Entity’: ’Tuesday’, ’Label’: ’time’, ’Entity’: ’September 22 , 2015’, ’Label’: ’time’, ’Entity’: ’Discovery’, ’Label’:
’object’, ’Entity’: ’decommissioning’, ’Label’: ’event’, ’Entity’: ’display’, ’Label’: ’event’, ’Entity’: ’toxic materials’,
26

’Label’: ’substance’, ’Entity’: ’the orbiter’, ’Label’: ’object’]
27
# Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation

**Authors**: Guo Chen, Qiuyuan Li, Qiuxian Li, Hongliang Dai, Xiang Chen, Piji Li

**Published**: 2025-09-25 07:50:30

**PDF URL**: [http://arxiv.org/pdf/2509.20859v1](http://arxiv.org/pdf/2509.20859v1)

## Abstract
In retrieval-augmented generation (RAG) question answering systems,
generating citations for large language model (LLM) outputs enhances
verifiability and helps users identify potential hallucinations. However, we
observe two problems in the citations produced by existing attribution methods.
First, the citations are typically provided at the sentence or even paragraph
level. Long sentences or paragraphs may include a substantial amount of
irrelevant content. Second, sentence-level citations may omit information that
is essential for verifying the output, forcing users to read the surrounding
context. In this paper, we propose generating sub-sentence citations that are
both concise and sufficient, thereby reducing the effort required by users to
confirm the correctness of the generated output. To this end, we first develop
annotation guidelines for such citations and construct a corresponding dataset.
Then, we propose an attribution framework for generating citations that adhere
to our standards. This framework leverages LLMs to automatically generate
fine-tuning data for our task and employs a credit model to filter out
low-quality examples. Our experiments on the constructed dataset demonstrate
that the propose approach can generate high-quality and more readable
citations.

## Full Text


<!-- PDF content starts -->

Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented
Generation
Guo Chen1,2, Qiuyuan Li1,2, Qiuxian Li1,2, Hongliang Dai1,2*, Xiang Chen3, Piji Li1,2
1College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics, Nanjing, China
2The Key Laboratory of Brain-Machine Intelligence Technology, Ministry of Education, Nanjing, China.
3MIIT Key Laboratory of Pattern Analysis and Machine Intelligence, College of Computer Science and Technology,
Nanjing University of Aeronautics and Astronautics, Nanjing, China
{162110125, lqy123, 162350227, hongldai,xiang chen, pjli}@nuaa.edu.cn
Abstract
In retrieval-augmented generation (RAG) question answer-
ing systems, generating citations for large language model
(LLM) outputs enhances verifiability and helps users iden-
tify potential hallucinations. However, we observe two prob-
lems in the citations produced by existing attribution meth-
ods. First, the citations are typically provided at the sentence
or even paragraph level. Long sentences or paragraphs may
include a substantial amount of irrelevant content. Second,
sentence-level citations may omit information that is essential
for verifying the output, forcing users to read the surrounding
context. In this paper, we propose generating sub-sentence
citations that are both concise and sufficient, thereby reduc-
ing the effort required by users to confirm the correctness of
the generated output. To this end, we first develop annota-
tion guidelines for such citations and construct a correspond-
ing dataset. Then, we propose an attribution framework for
generating citations that adhere to our standards. This frame-
work leverages LLMs to automatically generate fine-tuning
data for our task and employs a credit model to filter out
low-quality examples. Our experiments on the constructed
dataset demonstrate that the propose approach can generate
high-quality and more readable citations.
Introduction
We conduct experiments on the dataset we constructed and
evaluate our approach using a set of manually annotated
examples. The results demonstrate the effectiveness of our
method in generating accurate and concise sub-sentence-
level citations.
Our contributions are summarized as follows.
• We investigate the generation of concise and sufficient
sub-sentence-level citations for LLM-based RAG sys-
tems, with an emphasis on alignment with conventional
reading patterns. We propose a set of annotation prin-
ciples that reflect our citation standards and construct a
manually annotated dataset accordingly.
• We design an approach for generating such citations
that requires only a small number of manually annotated
training examples.
• We conduct experiments with the dataset we construct to
verify the effectiveness of our approach.
*Corresponding author
Question : When was the Great Barrier Reef declared 
a UNESCO site?
The Great Barrier Reef, located off the coast of 
Queensland, Australia, is the world's largest coral 
reef system. ... Designated a UNESCO World Heritage 
Site in 198l, the reef faces threats from climate 
change, coral bleaching, and pollution.  ...
Citation produced by an existing approach
Our standardThe Great Barrier Reef , located off the coast of 
Queensland, Australia, is the world's largest coral 
reef system. ... Designated a UNESCO World Heritage 
Site in 198l , the reef faces threats from climate 
change, coral bleaching, and pollution. ...Answer: 1981.Figure 1: Comparison of cited content produced by an ex-
isting approach and our sub-sentence citation standard. The
cited content is highlighted in red. Some unimportant con-
tent are omitted with “...”.
Related Work
The research on methods for generating attribution texts has
gradually become a topic in recent years due to the hal-
lucinations generated by LLMs. Self-citation is a recently
proposed method by Gao et al. (2023), which utilizes the
ability of recent LLMs to follow instructions in natural lan-
guage (Raffel et al. 2020; Chung et al. 2024; Brown et al.
2020), thereby avoiding the need for external validators and
guiding generic LLMs to generate inline citations with a
small number of samples. Self cited answers are generally
more relevant to the content provided by the source, but can
still contain unsupported statements and inaccurate citations
(Liu, Zhang, and Liang 2023). RAG-Ex (Sudhi et al. 2024)
is a model independent and language independent explana-
tory framework that explores the impact mechanism of con-
text on generated results through diverse input perturbation
strategies. Its innovation lies in approximately reconstruct-
ing the decision-making process of LLM through system-
atic perturbation strategies and response comparisons, with-
out relying on internal parameters or structure of the model.arXiv:2509.20859v1  [cs.CL]  25 Sep 2025

However, due to the lack of supervision of display signals, it
is susceptible to the influence of contextual document con-
tent, resulting in low interpretability of this method. The
interpretability techniques in Post hoc interpretation meth-
ods (Anand et al. 2022; Kenny et al. 2021) can be roughly
divided into gradient based methods (Bastings and Filip-
pova 2020; Samek and M ¨uller 2019) and perturbation based
methods (Bhattacharya 2022; Zafar and Khan 2021), which
prolong user waiting time due to the complexity of pipelines.
The MIRAGE framework (Huq, Pervin et al. 2020) uti-
lizes comparative analysis to predict distribution shifts in the
presence or absence of context, identifies sensitive generated
lexical elements using KL divergence, and establishes causal
relationships between lexical elements and documents using
gradient based attribution techniques. However, this method
has excessive computational complexity and low robustness.
LONGCITE (Zhang et al. 2024) has similarities with
our work, proposed a collaborative generation paradigm of
“coarse to fine” to meet the fine-grained citation require-
ments in long context scenarios. Due to the fact that the eval-
uation system is based on sentence granularity, this method
may generate overly complex citations in some cases, requir-
ing a significant amount of time to understand the content of
the citations.
Notably, the work contemporaneous with ours by Hirsch
et al. (2025) also conducts sub-sentence level citations.
However, they focus more on how to allow users to actively
highlight any fragment (such as a word or phrase) in the gen-
erated text, and then provide supportive source texts. They
do not study how high quality sub-sentence level cited con-
tents can be obtained.
Data Annotation Principles
This section introduces the annotation principles used to cre-
ate a dataset that aligns with our standards for citation gen-
eration.
Since long texts generated by LLMs can be processed sen-
tence by sentence or fact by fact (Min et al. 2023; Tang,
Laban, and Durrett 2024), we only focus on generating ci-
tations for simple question answer pairs. Figure 2 offers a
more detailed illustration of the earlier example. First, un-
like attribution methods that highlight only isolated words
(e.g., only highlighting “1981” in the example), we require
that the cited content should be natural, coherent, and eas-
ily understandable. To achieve this, we select text fragments
that align with conventional reading patterns to provide a
smoother reading experience.
Second, the cited content should be concise, but at the
same time sufficient for verifying the answer. In the exam-
ple of Figure 2, the sentence containing the answer is “Des-
ignated a UNESCO world Heritage Site in 1981, ... and pol-
lution.” But the portion beginning with “the reef faces” is
irrelevant and is therefore omitted from the citation. More-
over, although this sentence contains the answer, it does not
specifically mention which reef is being referred to. Thus,
we include the nearest preceding clause that mentions the
subject by name, eliminating the need for users to search
through earlier sentences for clarification.
The Great Barrier Reef , located off the coast of 
Queensland, Australia, is the world's largest coral 
reef system. Composed of over 2,900 individual 
reefs and 900 islands spanning 2,300 kilometers, it 
supports extraordinary biodiversity including 1,500 
fish species and 400 types of coral. Designated a 
UNESCO World Heritage Site in 198l , the reef faces 
threats from climate change, coral bleaching, and 
pollution. Recent surveys show 50% of coral cover 
has been lost since 1995. Conservation efforts 
include the Reef 2050 Plan, allocating AU$2 billion 
for water quality improvement and coastal 
protection measures.Question : When was the Great Barrier Reef declared 
a UNESCO site?
Answer: 1981.
Context:Figure 2: Sub-sentence-level citations adhering to our anno-
tation principles.
During the data annotation process, we further distinguish
between three types of instances.
• Type-1: The cited content aligns with the conventional
sentence-level standard, where a single sentence from the
document is highlighted and presented to the user as the
citation.
• Type-2: The cited content would contain irrelevant por-
tions if the instance is annotated according to the con-
ventional sentence-level standard. Such cases commonly
arise when the retrieved context originates from formal
sources, such as government documents or professional
articles.
• Type-3: Multiple segments of the retrieved context must
be cited. This typically occurs in two scenarios: 1) The
sentence that supports the answer does not explicitly
mention the name of the subject, such as the example in
Figure 2. 2) The answer requires multi-hop inference, ne-
cessitating the citation of multiple, dispersed facts from
the context to support verification.
To build a dataset that covers the above scenarios, we
leverage three existing publicly available question answer-
ing datasets including XOR-AttriQA (Asai et al. 2020),
XQUAD (Rajpurkar et al. 2016), and HotpotQA (Yang
et al. 2018) as the source corpus pool. Among them,
XOR-AttriQA provides attribution annotations required for
open-domain QA. But its annotated citations are mainly
in sentence-level. After manual secondary refinement, fine-
grained fragments directly corresponding to the answers can
be obtained. XQUAD is a reading comprehension bench-
mark with paragraph-level question answering instances.
HotpotQA is known for multi-hop reasoning, which is well-
suited for verifying the citation localization ability of models
in scenarios of long-range dependency and referential reso-
lution. The three datasets complement one another in terms
of discourse structure, content diversity, and the depth of rea-
soning required.

Methodology
Since manually annotating examples that meets our citation
standards is costly and time-intensive, we introduce an ap-
proach to produce training data automatically by leveraging
LLMs. The overall procedure is illustrated in Figure 3.
First, we use a high-quality dataset manually annotated
based on our annotation principles to fine tune LLM, which
serves as a fine tuned quality estimation (credit) model.
Then, using prompt word templates, the text generation class
model generates a large amount of training data based on
the dataset examples we provide, and then uses the fine
tuned credit model to filter out low-quality examples. Sub-
sequently, the system calls open-source generation mod-
els to generate candidate question answer citation pairs in
bulk under a unified hierarchical constraint prompt frame-
work. These candidate samples are first fed into the credit
model for confidence scoring. High scoring examples are di-
rectly included in the next round of fine-tuning corpus, while
low scoring examples are downgraded to coarse-grained.
Through this cycle of “small models learning from large
models and large models refining small models”, the dataset
improves synchronously in both scale and quality, signifi-
cantly expanding the training volume and domain coverage,
while ensuring the consistency of fine-grained standards. Fi-
nally, we fine tuned LLM by training it on high-quality ma-
chine generated data and human annotated datasets to gen-
erate references that meet our standards.
Figure 3: Overall procedure of our approach.
During the dataset expansion phase, we integrate mul-
tiple open-source LLMs (such as DeepSeek-R1 and GPT-
3.5) through APIs to enhance data diversity and expand
scale. The model is guided to generate training data with
fine-grained citations through structured prompt engineer-
ing. The prompt we adopt includes task instructions, input
and output of the training set, strictly following the require-
ments to generate data entries. The input includes complete
paragraphs and clear questions, and the output field must be
an exact reference fragment that directly supports the answer
in the original text. The design goal of the prompt template is
to ensure that the generated citations meet two core require-
ments: (1) accurate support for answer content; (2) Maintain
fluency for human readability.Experiments
In order to verify the effectiveness of the fine-grained attri-
bution framework based on large models proposed in this
article in practical applications, this section presents ex-
periments to validate the effectiveness of our annotation
method after fine-tuning the model. Before starting large-
scale data generation, we take 800 cross-granularity citation
samples that have been manually annotated as the starting
point,the ratio of this dataset under the three types we set
is 15%, 30%, and 50%. and uses LoRA fine-tuning strat-
egy to train a lightweight “credit model” on the Llama-3.1-
8B base. This model has the comprehensive ability to judge
answer accuracy, citation granularity, and readability with
only a few updated parameters. The comprehensive perfor-
Type # Examples Ratio
Type-1 120 15%
Type-2 280 35%
Type-3 400 50%
Total 800 100%
Table 1: Statistics of the dataset.
mance of the system evaluation model in three dimensions:
fine simplification of citation granularity, semantic consis-
tency, and human readability. The Longcite method is a
model obtained by fine-tuning a large number of sentence
granularity level QA problems. The RAG-Ex method is a
model independent perturbation based approach.The exper-
iment used LLaMa3.1-8b and Qwen2.5-7b as baseline mod-
els, Longcite model and unsupervised RAG-Ex method as
baseline methods for the case study, and a two-stage fine-
tuning strategy and data co evolution mechanism to itera-
tively generate a series of models that can generate concise
and sufficient citations. In the first and second stages of fine-
tuning, the ratio of the training set to the test set is 80% and
20%, respectivelyThe training framework is based on Deep-
Speed and LoRA technology stacks, ensuring efficient pa-
rameter updates under limited video memory constraints.
Evaluation Metrics
In the data evaluation stage, this study uses a three-layer
complementary indicator system to quantify the alignment
between model output and manual annotation: surface vo-
cabulary matching is undertaken by F1 score, distribution-
level semantic consistency is characterized by cosine simi-
larity, and the subjective dimension of generated quality is
assigned to GPT-4o for overall scoring. The F1 score is used
as a comprehensive indicator to measure the degree of lex-
ical overlap between predicted text and reference text, and
its core idea stems from the harmonic average of precision
and recall in information retrieval and classification tasks.
Specifically for the citation generation task involved in this
study, the calculation of F1 first relies on lexical alignment
of the text sequence: the system splits the original string into
token sequences through regular expressions and eliminates
case differences in lowercase space to establish comparable
units. Subsequently, the word element sets of the predicted

sequence and the reference sequence were separately con-
structed as unordered sets to remove the influence of word
frequency and retain only the morphological information.
Based on this set, the system calculates the intersection size
of the two as the true positive (TP). Precision is defined as
the proportion of correct word elements in the predicted text,
i.e. the ratio of TP to the predicted set cardinality; The recall
rate is defined as the proportion of successfully covered lex-
ical elements in the reference text, that is, the ratio of TP
to the reference set cardinality. The F1 (3) score combines
the two through harmonic averaging, and its closed-form ex-
pression is as follows:
F1 = 2·P·R
P + R,(1)
Precision =|P∩R|
|P|,Recall =|P∩R|
|R|.(2)
The symbols P and R represent the sets of predicted and ref-
erenced lexical elements, respectively. The numerical range
is between 0 and 1, and the closer it is to 1, the more com-
plete the overlap between the predicted text and the refer-
ence text in terms of word form. This indicator is used in
this experiment to quantify the surface-level consistency be-
tween the model output and manually annotated citations,
especially for evaluating whether the answer covers key en-
tities and key vocabulary, without involving deep seman-
tic matching. Parallel to this, cosine similarity introduces
weight information based on the bag of words model: the
system first maps the predicted text and the reference text
to the same high-dimensional word frequency vector space,
with corresponding vectors denoted as p and r. Then, the
cosine value of the angle between the two is calculated as
following:
cosθ=p·r
∥p∥∥r∥=Pn
i=1piripPn
i=1p2
ipPn
i=1r2
i.(3)
Among them,piandriare the normalized word frequen-
cies of the i-th morpheme. This value measures the direc-
tional consistency of word frequency distribution by the ra-
tio of dot product to modulus length, and is robust to changes
in word order and syntax, thus capturing semantic approx-
imations that may be overlooked by the surface F1 (3). Fi-
nally, the automatic evaluation of GPT-4o takes structured
prompts as input and comprehensively considers the three
dimensions of accuracy, conciseness, and readability. Each
citation is given a comprehensive score in the range of 0-1,
thereby transforming human preferences into reproducible
quantitative signals. The combination of the three not only
covers multiple levels of granularity from word form to se-
mantics to human perception, but also avoids evaluation bias
that may be caused by a single indicator, providing a robust
empirical basis for the effectiveness of fine-grained attribu-
tion frameworks.
Fine-tuned Model Performance Validation
To evaluate the effectiveness of the fine-grained attribution
framework, this study conducted fine-tuning experiments onthe LLaMa3.1-8b and Qwen2.5-7b models based on LoRA
low rank adaptation technology. The experiment adopts a pa-
rameter configuration with Rank=8 and Scaling Factor=32.
Under the premise of freezing the original model parame-
ters, only the incremental matrix is trained to adapt to the
fine-grained citation generation task. The training data in-
cludes 500 manually annotated samples and 1000 level ex-
tended samples filtered by credit models, with fine-grained
samples accounting for a stable proportion of over 80%. The
gradient accumulation steps are set to 8, the batch size is
2, the optimizer uses AdamW (learning rate 5e-5), and the
maximum sequence length is truncated to 1024 words.
• Word level F1 score: used to calculate the overlap rate
between predicted citations and manually annotated vo-
cabulary, measuring surface matching accuracy;
• Semantic cosine similarity: Capturing deep semantic
consistency through embedding vector distribution align-
ment;
• GPT-4o Comprehensive Quality Score: Based on a struc-
tured prompt template, weighted fusion accuracy, con-
ciseness, and readability scores in three dimensions,the
scoring process formula is as follows:
(Q, A, C pred)→GPT-4o (4)
Formula (4) is the input for the evaluation process, where
Q represents the problem, A represents the answer, and
C represents the citation. This formula combines ques-
tions, answers, and citations into a structured input that
is provided to the GPT-4o model for scoring.
˜Si=Si−1
4(5)
Squality =λ 1˜Sacc+λ 2˜Sconc+λ 3˜Sread (6)
λ1+λ 2+λ 3= 1, λ i≥0, i∈ {1,2,3}(7)
The scoring dimensions include three points, namely ac-
curacyS acc: the strength of causal relationship between
citations and answers;S conc: Information density and re-
dundancy;S read: The fluency of language expression,
where formula (5) is the output obtained by scoring,
which is a comprehensive evaluation of the quality of
citations and reflects their performance in three dimen-
sions: accuracy, conciseness, and readability. Formula (6)
is a constraint condition of the scoring formula, used to
ensure the rationality of the scoring. This is to ensure
the scientificity and effectiveness of the scoring process.
This constraint ensures that the score S remains within
the range of 0 to 1. The closer the score is to 1, the better
the quality of the citation. If the score is closer to 0, the
worse the quality of the citation.
Table 2 presents the evaluation experimental results of
LlaMa3.1-8B and Qwen2.5-7B models before and after fine-
tuning. After fine-tuning, the models showed certain im-
provements in various indicators. It should be noted that
after fine-tuning, the F1 index of Qwen2.5-7B increased to
0.7319, which is a significant improvement compared to the

Model F1 CS GPT-4o
Qwen2.5-7B 0.4616 0.5542 0.4839
Subcite-Qwen2.5-7B0.7319 0.7977 0.7624
LlaMa3.1-8B 0.3976 0.4692 0.4358
LongCite-llama3.1-8B 0.5328 0.6021 0.5637
Subcite-llama3.1-8B0.6547 0.7336 0.6953
Table 2: Comparison of model performance
original base model. This confirms that the fine-grained sup-
port capability has been significantly improved. This im-
provement is mainly due to the framework’s ability to ac-
curately identify the minimum sufficient evidence set, while
the learning mechanism of semantic associations between
answers and citations also plays a role. In the cross model
comparison process, Qwen2.5-7B showed a smaller perfor-
mance gap compared to LlaMa3.1-8B with more param-
eters after the same training process, and the framework
had excellent model generalization ability. This proves that
this mechanism has good adaptability to small and medium-
sized models, providing a feasible solution for deployment
in resource limited scenarios.The comparative evaluation re-
sults between our model and the Longcite method model are
shown in Table 2. It can be seen that in terms of annotation
principles in this article, our evaluation effect is relatively
good.
Case Study on Fine-grained Attribution
When the user’s question is “ What is the capital of the state
of Assam? ”, the citation results obtained by the three meth-
ods are shown in Table 3. In an open domain Q&A scenario,
when there are two independent and informative sentences
in the context that can independently support the same an-
swer, the traditional RAG Ex single sentence perturbation
strategy may not provide effective confidence due to “sin-
gle point failure”: no matter which sentence is removed, the
remaining information is still sufficient to answer the ques-
tion, resulting in insignificant perturbation differences and
inability to determine the true supporting sentence. How-
ever, the supervision method Longcite model highlights all
sentences that can provide answers as citations, which is a
redundant behavior. When users verify the accuracy of an-
swers, they can obtain answers with only one sentence, with-
out the need to spend time looking at the second citation.
Therefore, in this case, the citations obtained by this method
appear very complicated.Similarly, when the user’s question
is “How many colors are used in the national flag of South-
ern Mexico?”, other methods may also exhibit redundant an-
swers.The answer obtained through our method is a precise
statement that can fully answer the user’s question without
any unnecessary information, which provides convenience
for users to verify their answers.
Ablation Study on Data Scalability
In this section, we conducted an experiment on the impact of
adding an expanded dataset on the experimental results. Ta-
ble 4. shows the ablation experiment results of the trainingsample size for Qwen2.5-7B model. Further analysis of the
impact of training data size reveals the effectiveness of the
data iteration mechanism through experiments on the scala-
bility of training data size. The F1 training result obtained
when only using the initial manually annotated 500 data
points is 0.7319. After adding a large number of fine-grained
samples generated by the large model, the F1 obtained is
0.7387, which is an improvement compared to before. This
also proves that the model data co evolution mechanism is
feasible. When the training samples increased from 500 to
1000, the F1 index maintained a stable upward trend, ver-
ifying the data self enhancement ability of the framework.
The F1 score reached 0.7653 at 1000 samples, which is an
improvement from the benchmark of 500 samples.
The reason why we did not conduct experiments on
a large number of sentence granularity level datasets or-
ganized by adding other existing open source datasets is
mainly due to the fact that when the model comes into con-
tact with sentence granularity level training samples, it will
learn two different attribution patterns: one is based on fine-
grained attribution, and the other is based on coarse-grained
attribution, which will affect the updating of model parame-
ters.
Conclusion
This article proposes an improvement approach for annotat-
ing training data, focusing on the issues of coarse citations
and time-consuming verification in attribution fine-tuning
in the Retrieval-Augmented Generation Question Answer-
ing system. The approach involves using a small amount of
manual annotation to initiate the process, followed by self
expansion and self inspection using open-source large mod-
els.
The data expansion process of the framework can increase
the size of the dataset, but optimization is needed to en-
sure data quality. The integration of the dataset is not precise
enough, and adding new data directly can easily lead to re-
dundancy and inconsistency. The current framework mainly
deals with text data. When facing scenes containing multi-
modal information such as images and audio, it is difficult
to effectively integrate different modal data to improve at-
tribution performance. In the future, cross modal alignment
methods based on geometric deep learning can be explored
to study how to map different modal information to a uni-
fied space and achieve more comprehensive attribution anal-
ysis. In terms of processing extremely long texts, the current
framework is not very efficient. When faced with lengthy
documents, the model may experience performance degra-
dation during retrieval and citation generation. There are
also areas for improvement in the evaluation system. Cur-
rently, the proposed indicators mainly focus on English QA
issues, and their adaptability needs to be improved in cross
language scenarios. Subsequently, efforts can be made to es-
tablish a multilingual attribution benchmark test set that in-
cludes multiple language types and text forms, while simul-
taneously developing a dynamic verification protocol based
on causal reasoning to enhance the comprehensiveness and
accuracy of the evaluation.

MethodsQ1: “What is the capital of the state of Assam?” Q2: “How many colors are used in the national
flag of Southern Mexico?”
Generated Answer Generated Answer
RAG-Ex Dispur is the capital of the state of Assam in In-
dia. Dispur, a locality of Guwahati, became the cap-
ital of Assam in 1973. This was after Shillong, the
erstwhile capital,became the capital of the state of
Meghalava that was carved out of Assam,. Dis-
pur is the seat of Government of Assam. The As-
sam Secretariat building is located in dispur along
with the Assam Assembly House,MLA Hostels and
the State Emergency Operations Centre, The Assam
Trunk road and the G s road passes through Dispur.
To the south ofDispur is the theologically important
site of Basistha. . . .The President of Guatemala since then. The flag is
divided in four parts, red, yellow, white and black,
each colour representing Xinca people, Garifuna
people, Maya peoples and Ladino people, respec-
tively.These colours are also part of the ”Q’anil”, a
Maya symbol in which each color represents a point
of the compass, an element of nature and a part of
the human being. ”Q’anil” means ”seed” in Maya
script, and is also used for one of the 20 days of the
Maya calendar. . . .
Longcite Model Dispur is the capital of the state of Assam in In-
dia. Dispur, a locality of Guwahati, became the cap-
ital of Assam in 1973. This was after Shillong, the
erstwhile capital,became the capital of the state of
Meghalava that was carved out of Assam,. Dis-
pur is the seat of Government of Assam. The As-
sam Secretariat building is located in dispur along
with the Assam Assembly House,MLA Hostels and
the State Emergency Operations Centre, The Assam
Trunk road and the G s road passes through Dispur.
To the south ofDispur is the theologically important
site of Basistha. . . .The President of Guatemala since then. The flag is
divided in four parts, red, yellow, white and black,
each colour representing Xinca people, Garifuna
people, Maya peoples and Ladino people, respec-
tively.These colours are also part of the ”Q’anil”, a
Maya symbol in which each color represents a point
of the compass, an element of nature and a part of
the human being. ”Q’anil” means ”seed” in Maya
script, and is also used for one of the 20 days of the
Maya calendar. . . .
Ours Dispur is the capital of the state of Assam in In-
dia. Dispur, a locality of Guwahati, became the cap-
ital of Assam in 1973. This was after Shillong, the
erstwhile capital,became the capital of the state of
Meghalava that was carved out of Assam,. Dis-
pur is the seat of Government of Assam. The As-
sam Secretariat building is located in dispur along
with the Assam Assembly House,MLA Hostels and
the State Emergency Operations Centre, The Assam
Trunk road and the G s road passes through Dispur.
To the south ofDispur is the theologically important
site of Basistha. . . .The President of Guatemala since then. The flag is
divided in four parts, red, yellow, white and black,
each colour representing Xinca people, Garifuna
people, Maya peoples and Ladino people, respec-
tively.These colours are also part of the ”Q’anil”, a
Maya symbol in which each color represents a point
of the compass, an element of nature and a part of
the human being. ”Q’anil” means ”seed” in Maya
script, and is also used for one of the 20 days of the
Maya calendar. . . .
Table 3: Comparison of citation results generated by different methods on two queries
Model Expaned Data Size F1
Subcite2.5-7B500 0.7319
700 0.7387
1000 0.7653
Table 4: Model Performance with Varying Sample Sizes
References
Anand, A.; Lyu, L.; Idahl, M.; Wang, Y .; Wallat, J.; and Zhang, Z.
2022. Explainable information retrieval: A survey.arXiv preprint
arXiv:2211.02405.
Asai, A.; Kasai, J.; Clark, J. H.; Lee, K.; Choi, E.; and Hajishirzi, H.
2020. XOR QA: Cross-lingual open-retrieval question answering.
arXiv preprint arXiv:2010.11856.Bastings, J.; and Filippova, K. 2020. The elephant in the inter-
pretability room: Why use attention as explanation when we have
saliency methods?arXiv preprint arXiv:2010.05607.
Bhattacharya, A. 2022.Applied machine learning explainability
techniques. Packt Publishing.
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhari-
wal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al.
2020. Language models are few-shot learners.Advances in neural
information processing systems, 33: 1877–1901.
Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tay, Y .; Fedus, W.;
Li, Y .; Wang, X.; Dehghani, M.; Brahma, S.; et al. 2024. Scaling
instruction-finetuned language models.Journal of Machine Learn-
ing Research, 25(70): 1–53.
Gao, T.; Yen, H.; Yu, J.; and Chen, D. 2023. Enabling large
language models to generate text with citations.arXiv preprint
arXiv:2305.14627.

Hirsch, E.; Slobodkin, A.; Wan, D.; Stengel-Eskin, E.; Bansal, M.;
and Dagan, I. 2025. LAQuer: Localized Attribution Queries in
Content-grounded Generation.arXiv preprint arXiv:2506.01187.
Huq, A.; Pervin, M.; et al. 2020. Adversarial attacks and defense
on texts: A survey.arXiv preprint arXiv:2005.14108.
Kenny, E. M.; Ford, C.; Quinn, M.; and Keane, M. T. 2021.
Explaining black-box classifiers using post-hoc explanations-by-
example: The effect of explanations and error-rates in XAI user
studies.Artificial Intelligence, 294: 103459.
Liu, N. F.; Zhang, T.; and Liang, P. 2023. Evaluating verifiability
in generative search engines.arXiv preprint arXiv:2304.09848.
Min, S.; Krishna, K.; Lyu, X.; Lewis, M.; Yih, W.-t.; Koh, P.; Iyyer,
M.; Zettlemoyer, L.; and Hajishirzi, H. 2023. FActScore: Fine-
grained Atomic Evaluation of Factual Precision in Long Form Text
Generation. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, 12076–12100.
Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena,
M.; Zhou, Y .; Li, W.; and Liu, P. J. 2020. Exploring the limits of
transfer learning with a unified text-to-text transformer.Journal of
machine learning research, 21(140): 1–67.
Rajpurkar, P.; Zhang, J.; Lopyrev, K.; and Liang, P. 2016. Squad:
100,000+ questions for machine comprehension of text.arXiv
preprint arXiv:1606.05250.
Samek, W.; and M ¨uller, K.-R. 2019. Towards explainable artifi-
cial intelligence. InExplainable AI: interpreting, explaining and
visualizing deep learning, 5–22. Springer.
Sudhi, V .; Bhat, S. R.; Rudat, M.; and Teucher, R. 2024. Rag-ex:
A generic framework for explaining retrieval augmented genera-
tion. InProceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval,
2776–2780.
Tang, L.; Laban, P.; and Durrett, G. 2024. MiniCheck: Efficient
Fact-Checking of LLMs on Grounding Documents. InProceedings
of the 2024 Conference on Empirical Methods in Natural Language
Processing, 8818–8847.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.; Salakhut-
dinov, R.; and Manning, C. D. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.arXiv preprint
arXiv:1809.09600.
Zafar, M. R.; and Khan, N. 2021. Deterministic local interpretable
model-agnostic explanations for stable explainability.Machine
Learning and Knowledge Extraction, 3(3): 525–541.
Zhang, J.; Bai, Y .; Lv, X.; Gu, W.; Liu, D.; Zou, M.; Cao, S.; Hou,
L.; Dong, Y .; Feng, L.; et al. 2024. Longcite: Enabling llms to
generate fine-grained citations in long-context qa.arXiv preprint
arXiv:2409.02897.
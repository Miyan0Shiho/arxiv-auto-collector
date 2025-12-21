# Dynamic Context Selection for Retrieval-Augmented Generation: Mitigating Distractors and Positional Bias

**Authors**: Malika Iratni, Mohand Boughanem, Taoufiq Dkaki

**Published**: 2025-12-16 11:30:40

**PDF URL**: [https://arxiv.org/pdf/2512.14313v1](https://arxiv.org/pdf/2512.14313v1)

## Abstract
Retrieval Augmented Generation (RAG) enhances language model performance by incorporating external knowledge retrieved from large corpora, which makes it highly suitable for tasks such as open domain question answering. Standard RAG systems typically rely on a fixed top k retrieval strategy, which can either miss relevant information or introduce semantically irrelevant passages, known as distractors, that degrade output quality. Additionally, the positioning of retrieved passages within the input context can influence the model attention and generation outcomes. Context placed in the middle tends to be overlooked, which is an issue known as the "lost in the middle" phenomenon. In this work, we systematically analyze the impact of distractors on generation quality, and quantify their effects under varying conditions. We also investigate how the position of relevant passages within the context window affects their influence on generation. Building on these insights, we propose a context-size classifier that dynamically predicts the optimal number of documents to retrieve based on query-specific informational needs. We integrate this approach into a full RAG pipeline, and demonstrate improved performance over fixed k baselines.

## Full Text


<!-- PDF content starts -->

Dynamic Context Selection for
Retrieval-Augmented Generation: Mitigating
Distractors and Positional Bias
Maya Iratni[0000-0002-3567-7885], Mohand Boughanem[0000-0001-7004-0807],
and Taoufiq Dkaki[0000-0003-3962-7663]
Institut de Recherche en Informatique de Toulouse (IRIT), Toulouse, France
{Malika.Iratni, Mohand.Boughanem, Taoufiq.Dkaki}@irit.fr
Abstract.Retrieval-Augmented Generation (RAG) enhances language
model performance by incorporating external knowledge retrieved from
large corpora, which makes it highly suitable for tasks such as open-
domain question answering. Standard RAG systems typically rely on a
fixed top-k retrieval strategy, which can either miss relevant information
or introduce semantically irrelevant passages, known as distractors, that
degrade output quality. Additionally, the positioning of retrieved pas-
sages within the input context can influence the model’s attention and
generation outcomes. Context placed in the middle tends to be over-
looked, which is an issue known as the "lost in the middle" phenomenon.
In this work, we systematically analyze the impact of distractors on gen-
eration quality, and quantify their effects under varying conditions. We
also investigate how the position of relevant passages within the context
window affects their influence on generation. Building on these insights,
we propose a context-size classifier that dynamically predicts the optimal
number of documents to retrieve based on query-specific informational
needs. We integrate this approach into a full RAG pipeline, and demon-
strate improved performance over fixed-k baselines.
Keywords:Retrieval Augmented Generation (RAG)·Information Re-
trieval·Reranking·Multi-hop QA·Large language models (LLM)
1 Introduction
Retrieval-Augmented Generation (RAG) is an approach that enhances large lan-
guage models with external knowledge by retrieving relevant passages during in-
ference. In a standard RAG system, the user’s query is first encoded, and used to
retrieve a ranked list of documents from a large collection [2] [7]. The top-K pas-
sages,wherekisafixed,predeterminedvalue,arethenprovidedtothegenerative
model as additional context, allowing the model to generate a response based
on both the query and the retrieved information [7] [3]. This top-K approach
has become widely adopted because it is straightforward to implement, and gen-
erally provides adequate context for many tasks such as open-domain questionarXiv:2512.14313v1  [cs.IR]  16 Dec 2025

2 M. Iratni et al.
answering [2]. RAG systems often suffer from two critical challenges: (i) mislead-
ing irrelevant retrieved documents (distractors), that can distort the generation
results [1], and (ii) the ’lost-in-the-middle’ phenomenon. [8]. A distractor is a
retrieved text that appears similar to the query, but is actually semantically
irrelevant. Such documents can confuse the generator and reduce output qual-
ity by introducing noise into the context, which can mislead the generator and
weaken attention over relevant passages [1] [6]. Due to this, using a fixed number
of retrieved documents for all queries has significant limitations. A fixed, top-k
strategy risks excluding important information whenkis too small or including
excessiveirrelevantcontentwhenkistoolarge,makingitsuboptimalforcomplex
tasks, such as multi-hop QA [9]. This balancing problem is directly related to
the precision–recall trade-off in retrieval. Increasingkgenerally improves recall,
so fewer relevant passages are missed. However, it simultaneously deceases the
precision by including more irrelevant context, diluting relevant documents and
degrading generation quality [6]. An additional challenge in retrieval-augmented
generation lies in the positional effect of the retrieved passages within the input
sequence. Prior studies have shown that Large Language Models (LLM) tend
to prioritize information that appears at the beginning or end of the input se-
quence, attributing less attention to information placed in the middle. This leads
to the "lost in the middle" effect [8], wherein the impact of a relevant document
on generation may be diminished, or amplified, depending on its location in the
context provided to the generator. This positional sensitivity poses a challenge
for retrieval strategies that return many documents, as relevant passages can
become buried among distractors and receive less attention. This highlights the
importance of the structure of the input, in this case the order in which rele-
vant documents are provided to the generator. The above points emphasize the
fundamental questions of:
– RQ1:How much context does a given query need to be effectively answered?
– RQ2:Can the impact of distractors be reduced by leveraging the position
of relevant context within retrieved documents?
Queries differ in complexity, scope, and information requirements. Some may
be answerable with a single passage, while others require broader contextual
evidence. This is particularly true in the case of multi-hop question answering.
To address this challenges, this work makes the following contributions:
–We conduct an empirical study of how distractor passages affect generation
quality. Our evaluation quantifies the performance degradation as distractor
ratios increase, and identifies the conditions under which distractors most
significantly impair model output.
–We examine the “lost in the middle” phenomenon by placing relevant pas-
sages at the beginning, middle, and end of the model’s input window, to
evaluate how positional placement influences the effectiveness of retrieved
context for generation within a RAG pipeline.

2. RELATED WORKS 3
–We introduce a classifier that predicts the optimal number of contexts to
retrieve (k), to dynamically adapt the context length to the specific infor-
mational requirements of each query.
–We integrate the classifier into a full RAG system and compare its perfor-
mance against a fixed-k baseline, demonstrating consistent improvements in
generation.
2 Related works
Recent work has explored adaptive retrieval strategies to overcome the limita-
tionsoffixed-kretrievalinRAGsystems.Jeongetal.(2024)introduceAdaptive-
RAG [5], a framework that dynamically selects among non-retrieval, single-step,
and multi-step retrieval strategies based on query complexity. Their approach
employs a classifier trained to predict query complexity, which guides the se-
lection of an appropriate retrieval depth. Sun et al. (2025) introduces Dynami-
cRAG [10], which adaptively determines both the ranking and the number (k) of
retrieved documents for each query. The core component is a dynamic reranker
trained using reinforcement learning, where the quality of LLM-generated re-
sponses serves as the reward signal. Taguchi et al. propose Adaptive-k [11],
a retrieval method that dynamically selects the number of passages to retrieve
based on similarity score distribution. Ìn addition to retrieval size, the positional
placement of retrieved content has also been shown to influence generation per-
formance. Liu et al. [8] demonstrated that large language models often exhibit a
position bias, prioritizing information at the beginning and end of the context,
and neglecting passages located in the middle. In contrast to prior methods, our
approach employs a classifier that directly predicts the precise number of doc-
uments required for each query. While methods such as Adaptive-RAG rely on
classifiers to select among predefined retrieval strategies, our approach explicitly
estimates the exact number of documents required for each query.
3 Preliminary Analysis
3.1 Dataset Overview
We conduct our experiments using the MuSiQue-Ans dataset, a benchmark de-
signed to evaluate complex multi-hop reasoning in open-domain QA systems.
The dataset is comprised of queries created through the composition of two to
four single-hop queries, combined into a single coherent multi-hop question that
requires multiple reasoning steps. The dataset contains a collection of≈22k
queries, (≈20k in the training set, and≈2.5k in the dev set), wherein a query
can be categorized as either 2-hop, 3-hop, or 4-hop. These ‘hops’ represent the
number of reasoning steps required to answer the query. Each multi-hop ques-
tion is paired with a fixed number of ‘gold’ supporting passages, wherin 2-hop
questions are provided with two gold documents, 3-hop with three, and 4-hop

4 M. Iratni et al.
with four. Table 1 depicts the number of queries belonging to each hop type in
both the train and dev set.
Table 1: Number of queries by type
2-hop3-hop4-hop
Train1437643871175
Dev1252760405
3.2 Exploratory Analysis
We conducted a series of experiments with the goal of evaluating the impact of
distractors, and context positioning on the overall generation performance of a
basic RAG system. We examined how three factors influence the answer quality
: (i) the omission of gold supporting passages, (ii) the inclusion of distractor
documents, and (iii) the position of relevant context in the input sequence.
Fig.1: Evaluation of the impact of additional context for 2- to 4-hop and includ-
ing the presence of distractors (d).
The Impact of Context and DistractorsThe objective of this section is to
assess the generation quality of a RAG system under varying levels of supporting
context, using the MuSiQue train set. We first isolated 2-hop, 3-hop, and 4-hop
questions for separate evaluation. For each case generation performance was
iteratively measured by providing the generator with the query alongside: no
relevant context, then one, two, three, and four relevant passages iteratively.

4. PROPOSED APPROACH 5
The goal is to demonstrate how generation quality scales with the amount of
relevant contexts provided.
In addition, we conducted two further iterations on which each query was pro-
vided with all its relevant context, followed by adding one, then two distractor
passages. The main goal was to simulate retrieval noise, and evaluate the ex-
tent to which distractors degrade generation quality. The results confirm a clear
positive correlation between the number of relevant documents and generation
quality, but also highlight the destructive impact of distractors. Figure 1 depicts
this finding. The results show that distractors cause a significant drop in per-
formance, with a compounding effect as additional distractors are introduced.
This is specifically true in the case of 2-hop questions, where a single distrac-
tor dropped the performance by more than 26%. While both 3-hop, and 4-hop
queries proved to be more resilient to distractor passages (with a 13.5% drop
for 3-hop, and a 14.4% drop for 4-hop with a single distractor.), they still ex-
perienced a deterioration in generation quality, with a sharper drop when two
distractors are added.
Context PositioningThis section examines the effect of the position of rele-
vant passages on generation quality. For each query (2-hop, 3-hop, and 4-hop),
we retrieved 5 passages, including distractors and relevant passages. The posi-
tions varied according to three configurations: relevant passages placed at the
beginning, placed in the middle, placed at the end of the sequence. As shown in
Table 2, placing relevant passages at the end led to the best performance, while
placing them in the middle resulted in the lowest generation quality, supporting
the ’lost in the middle’ phenomenon. These results are consistent with known
input position biases in LLMs, which tend to prioritize information appearing
at the extremities of the input. The end-position advantage suggests models pri-
oritize more recent content. These findings highlight the importance of context
placement in RAG systems. When input space is limited or distractors exist,
placing relevant content near the end improves generation.
Table 2: Relevant context position
Beginning Middle End
Exact Match 0.5447 0.5391 0.5567
F1 score 0.6212 0.6126 0.6361
4 Proposed Approach
Based on our preliminary analysis, we propose an enhanced RAG system de-
signed to improve generation quality for multi-hop QA by reducing the impact
of distractors. This section presents the architecture and components of our sys-
tem which extends the standard RAG pipeline with two extensions designed to
reduce noise from irrelevant retrieved passages. Figure 2 illustrates the overall
system architecture, contrasting the baseline RAG pipeline with our proposed
extensions: including the dynamic-k classifier and the LLM-based reranker.

6 M. Iratni et al.
Fig.2: Diagram of Baseline Pipeline, Classifier-k Pipeline, and Classifier-k+LLM
pipeline
4.1 Baseline Pipeline
Our starting point is a standard RAG setup consisting of two main modules:
– Retriever:Given an input queryq, the retriever searches through a corpus
and returns the top-kpassages most relevant toq.
– Generator:The retrieved passages are concatenated withqand fed into a
generative model, which produces the final answer.
Given a queryq, the retriever fetcheskpassagesP k={p 1, p2, . . . , p k}, and the
generator produces output:y=Generator(q, P k)
This architecture provides strong end-to-end performance, but uses a fixedk
for all queries, which can lead to over or under retrieval depending on query
complexity.
4.2 Classifier-k Pipeline
The first major modification to this pipeline is the integration of a query-specific
kpredictor. We introduce a classifier to estimate the optimal number of context
(k) that should be retrieved for each query. The classifier takes as input a given
query and outputs an integer value corresponding to the number of contexts

5. EXPERIMENTAL SETUP 7
to retrieve. This value is used to dynamically adjust the number of documents
retrieved for each query.
The pipeline is thus modified as follows:
1. The classifier processes queryqto predict retrieval count:
kpred =Classifier(q)
2. The retriever fetchesk predpassages.
3. The generator produces output:
y=Generator(q, P kpred)
4.3 Classifier-LLM Pipeline
The second major modification to the pipeline augments the system an LLM
module for reranking. In this pipeline, we first retrieve a fixed-k number of
candidate contexts, as was done in the basic retrieval pipeline. The classifier-
predictedkvalue, the original query, and the topk fixedretrieved contexts are
passed as input to an LLM, which will act as an additional reranker. The LLM
is prompted to select the top classifier-k passages from the candidate set based
on their relevance to the query. These selected passages are then used as the
context for the final generation step.
The pipeline is thus modified as follows:
1. The classifier processes queryqto predict retrieval count:
kpred =Classifier(q)
2. The retriever fetchesk fixedpassages (basic retrieval).
3. The LLM reranker filters passages:
Pfiltered =LLM_Reranker(q, k pred, Pfixed )
|Pfiltered|=kpred
4. The generator produces output:y=Generator(q, P filtered )
5 Experimental setup
To evaluate our proposed approach, we conducted experiments using a variety of
retrievers on three datasets, using the three pipelines. We describe the relevant
configurations for each setup below.

8 M. Iratni et al.
5.1 Retrievers
– BM25:A sparse retrieval with scores computed by the BM25 model.
– Dense Retrieval: A vector-based retrieval system using cosine similarity.
Passages wereembeddedusingtheall-MiniLM-L6-v2modelfromSentence-
Transformers, and the Chromadb vector store.
– ColBERT + MonoT5: A two-stage retrieval pipeline using ColBERT and
a monoT5 reranker.
– ColBERT + BAAI/bge-reranker-large: An alternative configuration
using ColBERT paired with the BGE-large reranker.
Both two-stage configurations included the retrieval of top-50 candidates with
ColBERT and top-kreranking with either MonoT5 or BGE-Reranker.
5.2 Dataset and Evaluation Metrics
– MuSiQue: [13]We aggregated all unique passages from the dev set of the
MuSiQue-Ans dataset to form our retrieval corpus, yeilding≈22k passages.
– 2WikiMultihopQA:[4]Noretrievalcomponentwasappliedforthisdataset.
Instead,arerankerwasemployedtoreorderthe10candidatetextsperquery.
– MultihopRAG[12]:Thisdatasetincludesacorpusof603full-lengthtexts.
Basicchunkingwasapplied.An80/20splitwasused,with80%ofthequeries
allocated for training the classifier and 20% reserved for evaluation.
– Evaluation:Retrievalqualitywasmeasuredusingprecisionandrecall.Gen-
eration quality was measured using Exact Match (EM) and F1-score (F1).
5.3 Classifier
To dynamically predict the number of passages (k) to retrieve per query, we
fine-tuned a RoBERTa model on the training collection of all three datasets.
The classifier was trained as a multi-class task, where the input is a question
and the target is its annotated hop type: 2-hop, 3-hop, or 4-hop. The output
class is then mapped to a correspondingkvalue for retrieval. The classifier
was evaluated on all three datasets, and was evaluated to be 87.3% accurate
using exact match. An alternate classifier was trained using only the MuSiQue
dataset, to evaluate if the performance is affected significantly when trained on
three different datasets. The classfier trained on MuSiQue only scored a 77.8%
accuracy on MuSiQue queries, while the classifier trained on all three datasets
scored 76.5%, which does not indicate a substantial loss in performance.
– Modelarchitecture:Weusedtheroberta-basemodelfromHuggingFace,
with a classification head on top.
– Input:Natural language question.

6. RESULTS AND DISCUSSION 9
– Output:A categorical label indicating the number of reasoning hops re-
quired.
Training Configuration:The model was fine-tuned for 5 epochs using the
AdamW optimizer with a learning rate of2×10−5and linear learning rate
scheduling. In order to address class imbalance in the hop type distribution,
which is severely skewed in the favor of 2-hop questions, we implemented a
WeightedRandomSamplerthat assigns inverse frequency weights to each class
during training. The dataset was split into 70% training, 15% validation, and
15%testsetsusingstratifiedsamplingtomaintainclassproportionsacrosssplits.
Training was conducted with a batch size of 16, and the best model was selected
based on validation accuracy.
5.4 LLM Reranker
For the document selection component, we employed Mistral Nemo Instruct
(12.2Bparameters)withzero-shotprompting.Thepromptingstrategyinstructed
the model to: (1) Analyze the given query and retrieved passages, (2) select ex-
actlykpassages (predicted by the classifier) based on relevance to the query,
and (3) return relevant passages. The prompt is as follows: "Given the following
query and passages, rank the passages (by their ID numbers) that are most rel-
evant to answering the query. Return the predicted-k most relevant passage IDs
in a Python list."
5.5 Generator
The final generation stage utilized Flan-T5-XL as the text generation model,
which processes the selected passages alongside the original query to produce
the final response.
6 Results and Discussion
6.1 Baseline vs Classifier-k Pipeline
ThissectioncomparestheperformancesoftheBaselineandtheClassifierpipelines.
The fixed-k value for the Baseline Pipeline was selected to be a standard value
of five (k=5). Table 3 shows the generation results for each configuration. The
evaluationshowsthattheBaselineoutperformedtheClassifierpipeline(Baseline
vs. Calssifier rows) in both the MuSiQue and 2WikiMultiHopQA datasets, with
all retrieval configurations. Before considering alternative explanations, we must
ask whether the issue stems from the classifier, or not.If it always returned
the correctkvalue, would the results outperform the baseline?As is
depicted in Table 3, the fixed k (Baseline) approach outperforms the classifier-
k approach, even when the classifier is ideal. The result is shown in the Ideal
Classifier row, using the dense retrieval method, compared the the Baseline. The
retrieval results (not listed in the paper) show that while precision scores were

10 M. Iratni et al.
Table 3: Evaluation Results
Retrieval PipelineMuSiQue MultihopRAG 2WikiQA
EM F1 EM F1 EM F1
BM25Baseline 0.060 0.105 0.499 0.550 - -
Classifier 0.046 0.086 0.501 0.549 - -
Classifier+LLM0.063 0.107 0.530 0.585- -
DenseBaseline 0.141 0.221 0.567 0.612 - -
Classifier 0.124 0.195 0.594 0.642 - -
Classifier+LLM0.153 0.232 0.597 0.648- -
MonoT5Baseline 0.171 0.252 0.594 0.626 - -
Classifier 0.159 0.239 0.621 0.653 - -
Classifier+LLM0.195 0.277 0.625 0.659- -
BGEBaseline 0.177 0.260 0.601 0.645 0.488 0.576
Classifier 0.162 0.243 0.619 0.655 0.486 0.572
Classifier+LLM 0.199* 0.283*0.625* 0.666* 0.529* 0.625*
Classifier+LLM
(Structured)10.202*0.291*0.625*0.672*0.531*0.629*
Control Pipeline40.171 0.212 - - - -
Oracle StudyIdeal Classifier20.128 0.200 - - - -
Ideal Reranker30.216 0.303 - - - -
1The input was reordered so that the most relevant texts are at the end of the input.
2Comparison of the Baseline pipeline(k=5) and the Classifier pipeline with the op-
timal k under dense retrieval.
3Using the BGE retrieval method.
4In the control pipeline, the reranker LLM was given the candidate texts and tasked
with selecting the optimal number independently of the classifier.
∗These measurements were compared against the results obtained using the BGE
retriever and Baseline pipeline configuration using a paired t-test. The results show
a p-value of <0.01 for all three datasets.
increased with the classifier-k retrieval, the recall significantly decreases. This
means that while the approach may be reducing the number of distractor docu-
ments, it cuts-off relevant contexts, which decreases generation performance. If
a relevant context is found in lower positions, but only the top two contexts are
selected, that relevant context is lost. Thus, the issue likely stems from retrieval
quality rather than the classifier, highlighting the need for a stronger reranker.

6. RESULTS AND DISCUSSION 11
6.2 Oracle Study
Ideal RetrieverThe previous experiment raises the question, if the retrieval
module is ideal, will the classifier improve the generation performance? This
section simulates an ideal retriever. This means that in the case of the baseline,
the fixed-k retrieved contexts will contain all relevant contexts. However, this
means that it will also include extra distractor passages. For example, a 2-hop
question would have five retrieved contexts, two of which are relevant, and three
of which are distractors. In the classifier pipeline, the classifier determines thek.
However, the classifier is not guaranteed to be accurate each time. For example,
it could incorrectly label a 3-hop query as 4-hop, wherein all three relevant
documents are retrieved, as well as one distractor document. Although, if the
classifier mislabels a 3-hop question as 2-hop, only two relevant contexts are
be retrieved, and one would be missing. This experiment was conducted four
times with the baseline pipeline with varying fixed-k values. The experiment
was also conducted with the classifier-k, and with the ideal k value. Table 4
shows the results of this experiment. The best performing setup is with the
ideal k, followed by the classifier-k. This shows that with an ideal retriever, the
classifier outperforms the fixed-k baseline.
Table4:Generationperformancewithfixed-k,classifier-k,andideal-kusingideal
retrieval
Setting EM F1
k= 20.475 0.621
k= 30.520 0.667
k= 40.523 0.673
k= 50.481 0.638
Classifier-k 0.535 0.682
Ideal-k 0.564 0.705
IdealRerankerInthispart,theretrievalisperformedforbothpipelines(Base-
line and Classifier) using a fixed k=5. We then introduce an ‘ideal’ reranking
module that reorders the contexts such that relevant contexts appear at the top
of the list, followed by any distractors that were retrieved. For example, if the
classifier predicts a question to be 3-hop (k=3), the model selects the first three
contexts after ideal reranking, ensuring that if relevant contexts exist in the top
five,theyareprioritized.Table 3showstheresultsofthissimulation,intheIdeal
Reranker row. The results show that an ideal reranking module with the clas-
sifier produces better results than the fixed-k pipeline. Although both pipelines
end up with the same number of relevant contexts, the increased presence of
distractors in the fixed-k approach decreases the generation performance.
6.3 Classifier-LLM Pipeline
This model replicates the ideal reranker experiment using an LLM to perform
additional reranking. Retrieval is first performed with a fixedk= 5and gen-

12 M. Iratni et al.
eration follows as in standard fixed-kpipeline. In the classifier+LLM pipeline,
the LLM receives the classifer-k value, the five retrieved contexts, and the query.
The LLM is prompted to rank the contexts by relevance, and select the top
classifier-kcontexts. The selected contexts, and the query are then provided
to the generator. We also evaluated a control pipeline without the classifier, in
which the LLM receives the five contexts, and tasked to determine the opti-
malkitself, and to select the most relevant documents for generator. Table 3
reports the results of the Control Pipeline and compares them with the Base-
line and the Classifier-LLM pipelines accross each retrieval configuration. The
results show that the Classifier-LLM configuration outperformed the baseline
with every retrieval method. In addition, we applied the concept of context po-
sitioning introduced in Section III. Passages judged most relevant by the LLM
were placed at the end of the context fed to the generator. As shown in the table
in the Classifier-LLM (Structured) part, this modification further improved per-
formance. To evaluate the significance of the performance difference between our
model and the baseline, we conducted a paired t-test between the Baseline and
Classifier-LLM pipeline in the BGE retrieval configuration, which yielded the
best results. The test yielded a p-value <0.01 on all three datasets, indicating
that the improvement achieved by our model is statistically significant.
Comparative AnalysisWe compare the performance of our model against the
Adaptive-Rag [5] results obtained in their paper, which addresses a related task
using a different retrieval method. Results against our Classifier-LLM pipeline
(BGE retriever) were comparable on the MuSiQue dataset (with 23.6 against
our 20.2 using EM), while our model outperformed on the 2WikiMultihopQA
dataset(with40.6againstour53.1usingEM).Wedidnotcompareourmodeldi-
rectly against DynamicRAG [10] as their evaluation was conducted on a different
set of datasets, and their approach requires computationally intensive training
procedures that are not directly comparable to our setting.
7 Conclusion
In this work, we examined two key limitations of standard RAG systems: the
adverse impact of distractor passages and the positional bias introduced by the
“lost in the middle” effect. Using the MuSiQue-Ans dataset, we showed that both
distractors and suboptimal passage placement substantially reduce generation
quality, particularly for multi-hop queries. To adress these issues, we introduced
a dynamic context selection framework that combines a query-specific classifier
with an LLM-based reranking to dynamically determine the optimal number
of documents to retrieve. Empirical results indicate that while classifier-only
retrieval can lower recall, integrating classifier predictions with LLM reranking
yields significant improvements in exact match and F1. Moreover, positioning
the most relevant passages at the end of the context sequence yields additional
gains, reinforcing the importance of input structure in RAG pipelines. These
findings highlight the need for more dynamic context selection methods.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.

7. CONCLUSION 13
References
1. C. Amiraz, F. Cuconasu, S. Filice, and Z. Karnin. The distracting effect: Under-
standing irrelevant passages in rag.arXiv preprint arXiv:2505.06914, 2025.
2. Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang, and
H. Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1), 2023.
3. K.Guu,K.Lee,Z.Tung,P.Pasupat,andM.Chang. Retrievalaugmentedlanguage
model pre-training. InInternational conference on machine learning, pages 3929–
3938. PMLR, 2020.
4. X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa. Constructing a multi-
hop qa dataset for comprehensive evaluation of reasoning steps.arXiv preprint
arXiv:2011.01060, 2020.
5. S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park. Adaptive-rag: Learning
to adapt retrieval-augmented large language models through question complexity.
arXiv preprint arXiv:2403.14403, 2024.
6. B. Jin, J. Yoon, J. Han, and S. O. Arik. Long-context llms meet rag: Overcoming
challenges for long inputs in rag.arXiv preprint arXiv:2410.05983, 2024.
7. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural information processing systems,
33:9459–9474, 2020.
8. N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and
P. Liang. Lost in the middle: How language models use long contexts.arXiv
preprint arXiv:2307.03172, 2023.
9. W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu. Dragin: Dynamic retrieval augmented
generationbasedontheinformationneedsoflargelanguagemodels.arXiv preprint
arXiv:2403.10081, 2024.
10. J. Sun, X. Zhong, S. Zhou, and J. Han. Dynamicrag: Leveraging outputs of large
language model as feedback for dynamic reranking in retrieval-augmented genera-
tion.arXiv preprint arXiv:2505.07233, 2025.
11. C. Taguchi, S. Maekawa, and N. Bhutani. Efficient context selection for
long-context qa: No tuning, no iteration, just adaptive-k.arXiv preprint
arXiv:2506.08479, 2025.
12. Y.TangandY.Yang. Multihop-rag:Benchmarkingretrieval-augmentedgeneration
for multi-hop queries.arXiv preprint arXiv:2401.15391, 2024.
13. H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
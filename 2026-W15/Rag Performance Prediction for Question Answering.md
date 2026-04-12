# Rag Performance Prediction for Question Answering

**Authors**: Or Dado, David Carmel. Oren Kurland

**Published**: 2026-04-09 08:55:38

**PDF URL**: [https://arxiv.org/pdf/2604.07985v1](https://arxiv.org/pdf/2604.07985v1)

## Abstract
We address the task of predicting the gain of using RAG (retrieval augmented generation) for question answering with respect to not using it. We study the performance of a few pre-retrieval and post-retrieval predictors originally devised for ad hoc retrieval. We also study a few post-generation predictors, one of which is novel to this study and posts the best prediction quality. Our results show that the most effective prediction approach is a novel supervised predictor that explicitly models the semantic relationships among the question, retrieved passages, and the generated answer.

## Full Text


<!-- PDF content starts -->

RAGPERFORMANCEPREDICTION FORQUESTIONANSWERING
A PREPRINT
Or Dado1, David Carmel1,2, and Oren Kurland1
1Technion, Haifa, Israel
2Technology Innovation Institute (TII), Haifa, Israel
April 10, 2026
ABSTRACT
We address the task of predicting the gain of using RAG (retrieval augmented generation) for question
answering with respect to not using it. We study the performance of a few pre-retrieval and post-
retrieval predictors originally devised for ad hoc retrieval. We also study a few post-generation
predictors, one of which is novel to this study and posts the best prediction quality. Our results show
that the most effective prediction approach is a novel supervised predictor that explicitly models the
semantic relationships among the question, retrieved passages, and the generated answer.
1 Introduction
Retrieval-Augmented Generation (RAG) has become a widely adopted approach for enhancing the performance of
Large Language Models (LLMs), particularly in question answering tasks [ 1]. In this framework, content retrieved
from an external corpus is incorporated into the prompt to ground the LLM’s responses in external knowledge and to
mitigate hallucinations [ 2]. Nevertheless, despite its overall effectiveness, RAG does not consistently yield benefits.
In many cases, the retrieved content may be unhelpful or even distracting [ 3,4], potentially misleading the LLM and
degrading answer quality [5].
In this work, we investigate the following question: Can the benefit of applying RAG to a given question be predicted
relative to not using RAG? A reliable prediction of RAG gain would enable selective, instance-level application of
RAG, thereby improving both system effectiveness and computational efficiency.
We begin by evaluating a range of pre-retrieval and post-retrieval predictors originally proposed for ad hoc retrieval tasks
[6], adapting them to the problem of RAG gain prediction. Our results show that pre-retrieval predictors are entirely
ineffective in this setting, while certain post-retrieval predictors demonstrate limited yet non-negligible predictive
capability. Furthermore, we show that supervised post-retrieval methods substantially outperform their unsupervised
counterparts. Building on these findings, we propose a novel post-generation predictor that estimates RAG gain
by additionally incorporating the LLM-generated response. Specifically, we introduce a supervised post-generation
approach designed to capture semantic relationships among the question, retrieved passages, and generated answer.
This predictor achieves the highest prediction accuracy among all methods considered.
In summary, our primary contribution is a systematic investigation of prediction strategies for estimating RAG gain.
Through extensive experiments with two RAG-based LLMs across three widely used question answering datasets, we
demonstrate that supervised post-retrieval and post-generation predictors achieve strong performance and are promising
candidates for integration into real-world RAG systems.
2 Related Work
The concept of RAG Gain was first introduced by Huly et al. [ 7] in the context of the text completion task. Specifically,
they formalized the relative performance improvement achieved by augmenting parametric generation with retrieved
context. In this work, we adopt their definition of RAG gain. Huly et al. [ 7] systematically evaluated a wide rangearXiv:2604.07985v1  [cs.CL]  9 Apr 2026

Rag Performance Prediction for Question AnsweringA PREPRINT
of retrieval and generation-based signals as predictors of this gain, demonstrating that many intuitive retrieval-centric
indicators fail to reliably predict when retrieval enhances text completion quality. Their analysis further showed that
post-retrieval and post-generation signals exhibit substantial predictive power. In contrast to Huly et al. [ 7], we study
RAG gain prediction in open-domain question answering (Q&A), a setting in which the interaction between retrieval
relevance and answer quality is considerably complex.
Prior research on predicting the contribution of RAG in question answering has primarily focused on estimating retrieval
utility for answer generation [ 8,9,10,11]. These studies indicate that retrieval utility largely depends on factors such
as context length [ 10], the positioning of relevant passages within the retrieved context [ 9], and potential conflicts
between retrieved evidence and the LLM’s internal knowledge [ 12]. Dai et al. [ 13] proposed predicting the RAG effect
by estimating the LLM’s uncertainty during answer generation.
More recently, adaptive and selective RAG approaches were proposed for deciding when retrieval should be invoked
to improve answer quality. Methods such as Self-RAG [ 14] and Active-RAG [ 15] dynamically determine whether
to retrieve additional context during generation, empirically demonstrating that retrieval is not universally beneficial.
While these works underscore the importance of estimating RAG utility, they primarily focus on policy learning
and generation control rather than on explicitly predicting RAG gain as a quantifiable outcome. Jeong et al. [ 16]
introduced an adaptive question answering framework that dynamically selects the most appropriate retrieval strategy
for RAG-based LLMs based on predicting the question complexity. Similarly, Wang et al. [ 17] investigated retrieval
decisions in conversational settings, where the choice to apply RAG is made at each dialogue turn.
The work most closely related to ours is that of Tian et al. [ 11], who decomposed the problem into retrieval performance
prediction and generation performance prediction, analyzing a variety of unsupervised signals derived from both
retrieved documents and generated answers. Their results suggest that combining retrieval-centric and LLM-centric
signals improves prediction robustness. In contrast, we propose a holistic approach that directly predicts the gain of
applying RAG in question answering. We formulate RAG gain estimation as a prediction problem based on observable
signals, categorized into pre-retrieval, post-retrieval, and post-generation features. Our models are designed to learn
complex semantic interactions among the question, retrieved passages, and generated answer, enabling direct prediction
of RAG gain.
3 Task Definition
The challenge we address is the prediction of the RAG gain for question answering, that is, the gain of using retrieved
content as context for the question at hand. Following Huli et al. [ 7], we define the relative (log) gain of using RAG
with a retrieved contextC, for answering questionq, as:
Gain(C|q, r,Q, θ) = logQ(θ(q;C), r)
Q(θ(q), r)(1)
where θ(p) is the LLM’s response to prompt pandQ(a, r) is the quality metric used to measure the quality of generated
answer a, with respect to the reference answer r. The prediction task we address below is predicting the gain defined in
Equation 1 with no knowledge of the reference (ground-truth) answerr.
3.1 Quality Metrics
Most factoid question answering benchmarks, including Natural Questions (NQ) [ 18], TriviaQA [ 19], HotpoQA [ 20],
and PopQA [ 21], rely on Exact Match (EM) and token-level F1 as their primary evaluation metrics. EM is a binary,
factoid-level accuracy measure that evaluates whether the reference answer exactly matches, or is fully contained,
within the generated answer. F1 quantifies surface-level token overlap between the generated and reference answers.
While these metrics are simple and widely adopted, they fail to capture important semantic aspects of answer quality. In
particular, they do not account for the presence of redundant information or misinformation. Moreover, they do not
assess whether the generated response sufficiently satisfies the underlying information need. As a result, EM and F1 are
limited in their ability to accurately evaluate the quality of LLM-generated answers and, by extension, the true gain
achieved by RAG [22].
To address these limitations, a common approach for evaluating LLM-generated answers is the LLM-as-a-judge
paradigm [ 23], in which typically a strong LLM is tasked with assessing answer quality by comparing the generated
response against a reference answer. Although this approach has been shown to correlate highly with human judgments,
it is sensitive to the choice of the judging LLM, as such models may introduce their own biases [ 24]. Moreover,
LLM-based evaluation incurs substantial computational cost, limiting its scalability.
2

Rag Performance Prediction for Question AnsweringA PREPRINT
Figure 1: (Top): Pairwise Pearson correlations among the quality metrics acrossNQ-Open,HotpotQA,TriviaQA, and
QEM, across the three question answering datasets. The metrics measure the semantic quality of answers generated by
Falcon3 (with no RAG), with respect to the reference answers. (Botton): Pairwise Pearson correlations among the gains
inferred while using the three quality metrics. The gain is measured between thewith-ragandwith-no-raganswers
generated by Falcon3.
The high cost of LLM-as-a-judge approaches motivates the need for alternative metrics that approximate the semantic
acuity of powerful LLM evaluators while avoiding their computational overhead. Prior work [ 25] demonstrated that
answer quality can be estimated by Bert-based shallow modes, achieving high correlation with human judgments and
comparable performance with GPT-4o. In this work, we experiment with three shallow answer quality metrics that
assess semantic alignment between the generated answer and the reference answer:
•QE5: The cosine similarity between embedding vectors of the generated answer and the reference answer,
where embeddings are computed using the pre-trained E5-large-v2 model [26].
•QCE: A semantic association score between the generated and reference answers computed using a pre-trained
cross-encoder based on RoBERTa-large1[27]. The raw cross-encoder score is normalized to the range [0,1]
using a sigmoid function.
•QNLI: The entailment between the reference answer and the generated answer. Under this formulation,
answer quality is framed as a natural language inference (NLI) problem; specifically, whether the generated
answer is semantically entailed by the reference answer [ 25]. We employ the DeBERTa-v3-large-NLI model2
[28], which is trained on a large and diverse collection of NLI datasets, to estimate entailment probabilities for
quality prediction.
Figure 1 (Top) presents pairwise Pearson correlation coefficients among the three quality metrics, computed over 3,600
Q&A sets sampled each from three question answering datasets; correlation with EM is given for reference. The four
metrics measure the quality of answers generated by Falcon3 [ 29] with no-RAG, with respect to the reference (ground-
truth) answers. The strong correlations indicate a high degree of agreement among the metrics in their assessment. The
observed variation in correlation strength (e.g., ranging from r= 0.61 to0.84) suggests that the metrics are not fully
interchangeable and capture complementary aspects of quality. Interestingly, the metrics are split into two groups; QE5
andQ CEare highly correlated as the both measure semantic similarity, whileQ NLI is highly correlated withQ EM.
Figure 1 (Bottom) illustrates the pairwise correlation between the three gains while using the different quality metrics.
As indicated by the strong correlation between the gains ( >0.67 ), there is substantial agreement among them regarding
the potential gain of using RAG for question answering.
1https://huggingface.co/sentence-transformers/stsb-roberta-large
2MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
3

Rag Performance Prediction for Question AnsweringA PREPRINT
3.2 RAG-gain Analysis.
As discussed earlier, RAG benefits question answering by improving answer quality, grounding responses with external
knowledge, and reducing hallucinations. In this work, we focus specifically on improvements in answer quality resulting
from RAG usage, as quantified by our gain metric (Equation 1). For this analysis, we use Falcon3-10B [ 29] as our
backbone LLM, accompanied with RAG based on BM25 retrieval over Wikipedia dump (Dec. 20, 2018) [ 30], where
articles are segmented into disjoint 100-word passages. (More details on the RAG implementation can be found in
Section 5.) Figure 2 presents the distribution of RAG gain across three Q&A datasets and the three quality metrics
defined above.
For each dataset, we experiment with 3,600 questions, randomly sampled from the dataset’s validation set. Across
all datasets and quality metrics, the average RAG gain is positive, highlighting the overall effectiveness of RAG for
question answering. However, while the magnitude of the average gain varies by dataset, the gain distributions exhibit a
consistent pattern: a pronounced central peak near zero accompanied by heavy tails skewed toward positive values.
This indicates that for a significant fraction of the questions, RAG yields little or no improvement in answer quality (e.g.
>30% in HotpotQA).
These findings suggest that indiscriminately applying RAG to all questions may introduce unnecessary latency and
computational overhead. Consequently, they motivate the task of RAG gain prediction. By estimating the expected
gain in advance, a system can adopt a selective RAG strategy, invoking retrieval only when it is likely to be beneficial.
Continuous gain prediction enables applications to explicitly manage the trade-off between answer quality and efficiency.
For instance, quality-sensitive systems may adopt a lower threshold to trigger retrieval more frequently while avoiding
harmful contexts, whereas cost-sensitive systems may invoke retrieval only when substantial gains are anticipated.
4 Gain prediction methods
We study three types of prediction methods. Pre-retrieval predictors analyze only the question and corpus-level statistics.
Post-retrieval predictors analyze the question and the retrieved passages. Post-generation predictors analyze the question,
the passages, and the generated answer. In what follows we survey a few prediction methods which were originally
proposed for ad hoc retrieval. In addition, we propose a few novel RAG gain predictors for question answering.
4.1 Pre-Retrieval Predictors
Pre-retrieval predictors are unsupervised measures used in ad hoc retrieval to estimate retrieval effectiveness before
retrieval is performed [ 31]. These methods estimate the relationship between query terms and their statistical properties
in the document corpus:
•SCQ:The Similarity to Collection Query (SCQ) method estimates how well a query will perform based on its
similarity to the document collection. It measures the sum oftf-idfweights of query terms [ 32]. We experiment
with (max/min/mean) of the tf-idf values of query terms as alternative variants of SCQ-based prediction.
•IDF:This is the average, max, or min idf values of the terms in a query [ 33]. It is based on the premise
that rare or highly specific terms are more effective at distinguishing relevant documents from the rest of the
collection.
•VAR:This is the variance of a query term’s tf-idf weights across the documents in which they appear [ 32]. A
higher variance suggests that a term has stronger discriminative power, making it a better indicator of effective
retrieval. We experiment with the the mean, maximum and minimum variance over query terms as predictors.
4.2 Post-Retrieval Predictors
In ad hoc retrieval, post-retrieval predictors analyze the retrieved list, L, in addition to the query and corpus statistics
[6]. Here we study whether the predicted list effectiveness is correlated with the gain attained by using the highest
ranked passages for answer augmentation. We explore the following highly effective post-retrieval predictors:
Unsupervised
•WIG:The difference between the average retrieval score of the top- kpassages in Land the corpus retrieval
score; kis a parameter [ 34]. The assumption is that the higher the difference, the more effective the retrieval
as the corpus is essentially a pseudo non-relevant document. We also study a variant,U_WIG, which uses the
average of top scores without the corpus score regularization.
4

Rag Performance Prediction for Question AnsweringA PREPRINT
Figure 2: RAG gain distribution across the three Q&A datasets, each comprising 3,600 question–answer pairs sampled
fromNQ-Open,HotpotQA, andTriviaQA. The backbone LLM is Falcon-3-10B, augmented with RAG using BM25
retrieval over a Wikipedia dump. Each question is enriched with the top-5 retrieved passages prior to answer generation.
The proportions of Q&As with negative gain reveal that retrieval can degrade generation quality for a substantial
fraction of questions, underscoring the risks of indiscriminate application of RAG.
•NQC:The standard deviation of retrieval scores of the top- kpassages in Lnormalized by the corpus retrieval
score [ 35]. We also study a variant,QC, which does not use the corpus normalization. The assumption is that
high standard deviation indicates reduced query drift and, hence, improved retrieval effectiveness.
•SMV:Combines the mean and variance of the retrieval scores of the top- kpassages in L; the corpus
retrieval score is used for normalization [ 36]. We also study a variant,U_SMV, which does not apply corpus
normalization.
•REF:The difference between the retrieved list and a high-quality reference list [ 37]. The assumption behind
this predictor is that high similarity to the reference list indicates higher quality hence a higher RAG gain. Given
the raw search results to be used for RAG, we re-rank the top-100 retrieved passages with a pre-trained reranker
(BAAI/bge-reranker-v2-m3 reranker3[38]) to be used as a high-quality reference list. We then compute the
Rank Biased Overlap (RBO) [39] between the original list and the re-ranked list for gain prediction.
Supervised:
3https://huggingface.co/BAAI/bge-reranker-v2-m3
5

Rag Performance Prediction for Question AnsweringA PREPRINT
•Bert-Post:A supervised BERT-based predictor, inspired by the BERT-QPP model [ 40], which models the
textual association of the query with top-retrieved passages for RAG gain prediction. We use the query qand
the list of top passages Las input to Modern-Bert-Large4[41] , a highly effective variant of BERT with long
window context. A linear regression layer is added on top of the BERT encoder, and the entire model is trained
end to end using a Mean Squared Error (MSE) loss to predict the RAG gain. The training process is further
detailed in Section 5.
4.3 Post-Generation Predictors
The predictors discussed so far operate before the LLM is prompted to generate an answer. We next leverage the LLM’s
output, with and without RAG, to devise post-generation predictors.
Unsupervised.
•Uncertainty:This predictor follows the assumption that high reduction in answer generation uncertainty
reflects high quality gain. For each generated sequence (i.e., answer), we compute token-level entropy from
the LLM’s probability distribution output, taking the maximum entropy across the sequence as a measure of
model uncertainty [ 42]. Let A= (t 1, ...tn)be the sequence of tokens generated by the LLM, and H(t) be the
entropy derived by the LLM while token t∈A is generated. We measure the reduction in uncertainty between
no-RAG and RAG generated answers, to be used for gain prediction:
max
t∈A no-ragH(t)−max
t∈A ragH(t).
•Entailment:This predictor is based on the assumption that higher faithfulness of the generated answer to the
RAG context leads to greater gain. We experimented with several alternative NLI models (e.g., DeBERTa-
based variants). However, due to context-length limitations, these models require entailment to be computed
separately for each retrieved passage, followed by a pooling operation (e.g., max or mean). This strategy
consistently resulted in lower correlations. We therefore adopt an NLI model based on ModernBERT-large
which is pre-trained on Natural Language Inference tasks and supports longer input contexts. For each
question, the top-5 retrieved passages are concatenated to form the premise, while the hypothesis consists
of the RAG-generated answer concatenated with the original question. The resulting NLI score is used for
RAG-gain prediction. By enabling joint reasoning over all retrieved passages, this model achieves the best
performance.
Supervised
•Bert-Gen: The model shares the same architecture asBert-Post, however, it is trained by additionally
incorporating the two answers generated by the LLM, with and with no RAG. It takes as input the question,
the top-5 passages, and the two answers for gain prediction.
It is important to note that post-retrieval prediction is significantly more efficient than post-generation prediction, as
it does not require LLM inference. While post-retrieval predictors rely solely on retrieval signals, post-generation
predictors also depend on both retrieved content and the two LLM inference steps.
5 Experimental Setup
Our experimental setup follows standard practice in Query Performance Prediction (QPP) for ad-hoc retrieval [ 6]],
adapted here to predict the performance gain of applying RAG for question answering. Prediction quality is estimated
by correlating actual gain with predicted gain. In the following we describe the main components of our experimental
framework.
LLMs.We experiment with two mid-sized, instruction-tuned LLMs: Falcon-3-10B-Instruct5[29] and Llama-3.1-8B-
Instruct6[43].
4https://huggingface.co/answerdotai/ModernBERT-large
5https://huggingface.co/tiiuae/Falcon3-10B-Instruct
6https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
6

Rag Performance Prediction for Question AnsweringA PREPRINT
RetrievalRetrieval is performed over a Wikipedia dump (Dec. 20, 2018) corpus [ 30], where articles are segmented
into disjoint 100-word passages. We evaluate two distinct retrieval paradigms: Sparse Retrieval and Dense Retrieval.
For sparse retrieval we use Okapi BM25 [44] via the Pyserini implementation [45] with default parameter values. For
dense retrieval we employ E5-large-v27for text embedding [ 26], indexing the same Wikipedia dump via FAISS [ 46].
For every question, we retrieve the top k= 100 passages. For RAG, the top-5 passages augment the question for answer
generation. (The prompt template used in our experiments is provided in Appendix A.).
DatasetsWe conduct experiments over three open-domain Question Answering (QA) datasets: NQ-Open [18],
HotpotQA [20], and TriviaQA [19]. All these datasets provide a large set of factual questions associated with reference
answers, considered as ground-truth answers. To address inconsistencies where original datasets lack test splits or
provide incomplete reference answers for some of the test questions, we enforce a standardized partitioning strategy.
From the train split of each dataset, we randomly sample 30,000 questions and divide them into 80−20 train/validation
split. The training set includes 24,000 questions used to train the supervised predictors. The validation set includes
6,000 questions used for hyperparameter tuning. For the test set we randomly sample 3,600 questions from the
validation split of each dataset, ensuring complete separation between training and testing data.
Training.We formulate RAG gain prediction as a supervised regression task, where the objective is to estimate the
improvement in answer quality obtained when using retrieval compared to parametric generation alone. The regression
target corresponds to the RAG gain as formulated in Equation 1.
We train two supervised predictors:Bert-Post(post-retrieval) andBert-Gen(post-generation). Both models use
ModernBert-large8as the backbone encoder and a cross-encoder architecture, enabling joint attention across the
question, retrieved passages, and generated answers.
•Input Representation:ForBert-Post, the input sequence consists of the question concatenated with the top- 5
retrieved passages: ([CLS]q[SEP]p 1[SEP]. . .[SEP]p 5[SEP]) . ForBert-Gen, we additionally append
the two answers generated by the LLM, one produced with no retrieval and one produced with retrieval:
([CLS]q[SEP]A NO-RAG [SEP]A RAG[SEP]p 1[SEP]. . .[SEP]p 5[SEP]).
ModernBERT-large produces a 1,024 -dimensional contextual embedding vector for each token. We use the
[CLS] token representation as a pooled representation of the entire sequence and feed it into a linear regression
layer that outputs a single scalar corresponding to the predictedRAG gain.
•Optimization:The model is fine-tuned end-to-end over the training data, using mean squared error (MSE)
loss between the predicted gain and the actual gain. Training is performed for two epochs using the AdamW
optimizer [ 47] with a learning rate of 5×10−5and a batch size of 16. The maximum input length is set to
8,192 tokens, which comfortably accommodates the question – 5 retrieved passages, and 2 generated answers.
We train a separate model for each combination of LLM, retriever, dataset, and quality metric, resulting in
2·2·3·3 = 36 distinct configurations. For each configuration, the best checkpoint is selected based on
validation set performance. All models are implemented using the Hugging FaceCrossEncoderclass9.
Hyper-Parameters tuning:Hyper-parameter values were selected using the validation set to maximize the total
correlation of the predicted gain with the actual gain. The total correlation is defined as the sum of gain correlations
computed separately with each of the three quality measures.
Pre-retrieval predictors do not involve any hyperparameter tuning. For post-retrieval predictors (WIG,U_WIG,NQC,QC,
SMV, andU_SMV), the optimal length, k, of the considered result list is selected from {1,2,3,4,5,10,20,30,40,50} .
For theREFpredictor which relies on RBO, the depth parameter Lwhich controls the prefix length of the lists to be
compared is selected from{1,5,10,20, . . . ,100}, and the decay rate parameterpis selected from{0.9,0.95,0.99}.
ForBert-Gen,Bert-PostandEntailment, we experiment with the number of top passages to be used for augmentation,
selected from {1,2,3,4,5} . Results consistently indicate an advantage for larger values; accordingly, we set the number
of top passages for augmentation to be 5.
Uncertaintyoperates solely on the LLM’s output probability distribution, producing a token-level uncertainty signal for
all tokens in the generated answer. To aggregate these signals into a single sequence-level score, we evaluate five pooling
strategies over the token entropy values {H(t)} t∈Ans : arithmetic mean, geometric mean, harmonic mean, min, and
max. Max pooling yields the strongest predictive signal across all evaluated configurations under the hyper-parameter
selection criterion.
7https://huggingface.co/intfloat/e5-large-v2
8https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0
9https://huggingface.co/cross-encoder
7

Rag Performance Prediction for Question AnsweringA PREPRINT
Evaluation:To be consistent with established conventions in QPP literature, we measure prediction quality using the
Pearson correlation ( r) between the predicted gain and the actual gain [ 6]. To compare performance between predictors,
we employWilliams’ two-tailed test[ 48] (p= 0.05 ), which accounts for the fact that two correlations computed on
the same sample against the same target are statistically dependent.
6 Results
Table 1 summarizes the prediction quality of the different predictors across the different configurations.
Pre-retrieval Predictors.Pre-retrieval predictors (IDF,SCQ,VAR, and their aggregates) show low to negligible
correlations (approximately0-0.1) with true RAG gain across all configurations.
Post-retrieval Predictors.Post-retrieval predictors, which utilize the retrieved passages for prediction, show a clear
improvement over pre-retrieval methods.
•Unsupervised Predictors. Among the unsupervised methods (NQC,QC,WIG,U_WIG,SMV,U_SMV), which
analyze score distribution, we observe low correlations in the range of approximately 0.1-0.2.REFachieves
relatively descent correlations (e.g.,0.25-0.30onNQ-Open).
•Supervised Predictors. The supervised post-retrieval predictor,Bert-Post, demonstrates substantially stronger
performance. This method which was trained directly to predict gain, further improves performance, reaching
correlations of up to0.45-0.49onNQ-Open. Furthermore,Bert-Postis significantly better than REF (and all
other post-retrieval predictors). This highlights the advantage of supervised approaches that capture semantic
relationships beyond simple score-based heuristics.
Post-Generation Predictors.The highest correlations are observed for post-generation predictors.
•UncertaintyandEntailment. The gap in uncertainty between the RAG-based answer and the no-RAG answer
serves as a strong unsupervised signal of model uncertainty, achieving correlations of0.35–0.55across most
configurations. This suggests that the LLM’s uncertainty is a meaningful proxy for RAG gain. In contrast,
Entailmentexhibits mixed results, performing well in some configurations (e.g., TriviaQA with BM25) but
poorly in others. The gap in performance between these two predictors stems from the fact that theEntailment
does not account for the LLM’s internal knowledge, whereasUncertaintyimplicitly reflects it through the
model’s uncertainty in output distribution.
•Bert-Gen. This predictor which incorporates the generated answer alongside the question and context (retrieved
passages), consistently outperforms all other methods. It achieves the highest correlations overall, peaking
at0.87-0.89 onTriviaQAand 0.71-0.79 onNQ-Open. It is statistically significantly better than all other
predictors across all configurations. This result provides strong empirical support for our hypothesis that
accurate RAG gain prediction requires a holistic view that considers the query, the retrieved documents, and the
LLM’s generated answer. To the best of our knowledge, these results outperform state-of-the-art performance
in RAG gain prediction [11].
Cross-Architecture and Cross-Metric Consistency.The observed correlation trends demonstrate high consistency
across experimental settings. The relative ranking of predictors remains largely unchanged when switching between
retrieval methods (BM25 and E5). Similarly, consistent trends are observed for both LLMs, Falcon and Llama,
indicating robustness to LLM architecture. Moreover, correlations based on the different accuracy metrics are strongly
aligned, further supporting the validity of our proposed gain formulation.
7 Conclusions and Future Work
In this work, we addressed the problem of predicting the benefit of applying retrieval-augmented generation (RAG)
for question answering with large language models. We evaluated the effectiveness of several predictors originally
proposed in the context of ad hoc retrieval, as well as a set of post-generation predictors. Our results show that the most
effective approach is a novel supervised predictor that explicitly models the semantic relationships among the question,
retrieved passages, and the generated answer.
For future work, we plan to explore additional predictive signals and extend our study to the task of selective RAG,
where the decision to apply retrieval is made on a per-question basis by predicting its expected relative gain.
8

Rag Performance Prediction for Question AnsweringA PREPRINT
Table 1: Pearson correlation between predicted gain and actual gain measured using the three quality metrics ( QE5,
QCE,QNLI), acrossNQ-Open,HotpotQA, andTriviaQAdatasets. Answers were generated by the two LLMs
(Falcon3,Llama-3.1) using RAG, based on two retrieval methods (BM25 and E5). †indicates SSB prediction than all
unsupervised pre- and post-retrieval predictors.††indicates SSB prediction than all other predictors.
NQ-Open HotpotQA TriviaQA
Falcon Llama Falcon Llama Falcon Llama
Predictor QE5QCEQNLIQE5QCEQNLI QE5QCEQNLIQE5QCEQNLI QE5QCEQNLIQE5QCEQNLI
BM25 Retrieval
meanIDF .141 .121 .120 .109 .084 .103 -.013 .030 .009 .018 .049 .033 -.045 -.029 -.000 .042 .063 .073
maxIDF .128 .108 .121 .117 .071 .104 .038 .062 .048 .031 .060 .047 .015 .018 .026 .076 .089 .087
minIDF -.023 -.025 -.032 -.027 .006 -.013 -.022 -.007 -.001 .013 .018 .013 -.067 -.070 -.049 -.030 -.025 -.010
meanSCQ .120 .110 .108 .093 .082 .090 .019 .036 .036 .073 .051 .053 -.019 -.009 .022 .033 .052 .066
maxSCQ .124 .110 .131 .122 .079 .099 .039 .054 .058 .044 .027 .051 .039 .040 .047 .057 .070 .066
minSCQ -.020 -.022 -.028 -.026 .002 -.014 -.013 -.001 .001 .020 .021 .017 -.060 -.064 -.043 -.029 -.024 -.011
meanVAR .125 .108 .118 .098 .071 .080 -.007 .030 .016 .015 .053 .015 -.023 -.011 .013 .046 .061 .069
maxVAR .093 .083 .100 .104 .068 .074 .040 .060 .057 .042 .065 .025 .016 .021 .035 .065 .076 .077
minVAR .018 -.005 -.003 -.001 .017 .015 -.000 -.001 .004 .035 .030 .012 -.048 -.056 -.048 -.033 -.026 -.008
NQC .108 .081 .083 .098 .067 .085 .067 .107 .107 .080 .063 .054 .023 .013 .024 .074 .066 .087
QC .199 .185 .190 .154 .117 .147 .112 .106 .136 .120 .083 .091 .151 .120 .103 .132 .115 .122
WIG .158 .149 .136 .157 .100 .122 .089 .095 .124 .180 .117 .116 .061 .036 .063 .144 .107 .115
U_WIG .141 .136 .131 .164 .095 .117 .122 .107 .143 .190 .121 .135 .103 .071 .070 .127 .101 .105
SMV .108 .081 .083 .100 .067 .087 .067 .107 .107 .081 .063 .054 .022 .013 .024 .053 .042 .065
U_SMV .199 .183 .189 .151 .116 .146 .091 .081 .108 .120 .083 .091 .150 .118 .099 .130 .114 .122
REF .255 .219 .253 .305 .190 .226 .173 .176 .213 .255 .146 .203 .214 .205 .190 .286 .226 .236
Bert-Post .450†.285†.428†.447†.292†.312†.375†.324†.405†.484†.288†.327†.304†.425†.403†.121†.319†.330†
Uncertainty .419 .251 .330 .354 -.022 .168 .371 .212 .280 .419 .006 .196 .545 .428 .388 .547 .283 .296
Entailment .159 .135 .152 .317 .135 .158 .116 .091 .173 .258 .122 .150 .064 .061 .079 .229 .159 .174
Bert-Gen .713††.528††.503††.788††.661††.451††.733††.562††.599††.779††.697††.521††.873††.694††.653††.868††.655††.554††
E5 Retrieval
meanIDF .046 .015 .016 -.045 -.029 -.025 -.042 -.001 -.024 -.041 .009 .003 -.081 -.063 -.043 -.021 -.013 .005
maxIDF .053 .038 .041 -.008 -.008 .004 -.017 .007 -.018 -.039 .004 -.007 -.025 -.018 -.012 -.004 .005 .023
minIDF -.081 -.091 -.065 -.067 -.046 -.058 -.015 .002 .021 -.002 .013 .025 -.078 -.082 -.056 -.049 -.045 -.026
meanSCQ .027 -.002 .008 -.043 -.024 -.035 .013 .023 .044 .038 .037 .055 -.053 -.034 -.016 -.004 .005 .014
maxSCQ .053 .029 .048 .002 -.010 -.007 .029 .031 .039 .030 .023 .024 .002 .017 .018 .010 .023 .027
minSCQ -.074 -.084 -.059 -.059 -.039 -.053 -.007 .007 .024 .007 .014 .033 -.069 -.075 -.050 -.044 -.043 -.025
meanVAR .055 .023 .036 -.041 -.030 -.012 -.038 .007 -.010 -.033 .017 -.009 -.058 -.041 -.022 -.011 -.011 .001
maxVAR .036 .014 .030 -.021 -.019 -.010 .002 .028 .012 .007 .034 .004 -.013 -.004 .006 .006 .014 .026
minVAR -.023 -.054 -.042 -.057 -.034 -.028 .010 .010 .029 .023 .006 .029 -.048 -.049 -.043 -.039 -.022 -.006
NQC .167 .171 .203 .208 .153 .141 .104 .087 .124 .159 .086 .068 .141 .161 .175 .159 .142 .117
QC .157 .163 .194 .204 .150 .138 .102 .086 .125 .160 .089 .069 .134 .154 .167 .155 .140 .114
WIG .097 .072 .084 .099 .071 .067 -.027 .006 .033 .008 -.012 -.018 -.054 -.059 -.015 .019 -.007 -.019
U_WIG .142 .133 .157 .189 .124 .097 .118 .096 .144 .176 .065 .063 .016 -.025 .004 .109 .067 .044
SMV .163 .164 .196 .204 .148 .136 .050 .077 .117 .180 .115 .105 .138 .157 .173 .050 .058 .053
U_SMV .154 .156 .187 .201 .146 .134 .049 .076 .118 .181 .118 .106 .117 .116 .130 .049 .057 .051
REF .149 .158 .178 .106 .081 .071 .124 .101 .143 .167 .099 .101 .105 .094 .115 .161 .127 .126
Bert-Post .499†.428†.481†.390†.210†.244†.370†.328†.378†.486†.233†.267†.460†.456†.445†.296†.282†.298†
Uncertainty .503 .355 .374 .335 .008 .135 .358 .198 .245 .385 -.035 .134 .564 .474 .427 .512 .236 .254
Entailment .057 .050 .096 .211 .090 .066 .103 .092 .153 .233 .112 .106 -.013 -.004 .035 .096 .026 .050
Bert-Gen .765††.620††.611††.793††.597††.502††.744††.609††.590††.780††.715††.523††.889††.738††.704††.885††.664††.642††
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Information Processing Systems (NeurIPS), volume 33, pages
9459–9474, 2020. (Original RAG Paper).
[2]Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces
hallucination in conversation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih,
editors,Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803, Punta Cana,
Dominican Republic, November 2021. Association for Computational Linguistics.
[3]Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust
to irrelevant context. InThe Twelfth International Conference on Learning Representations, 2024.
[4]Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong Yu.
Chain-of-note: Enhancing robustness in retrieval-augmented language models. InProceedings of the 2024
9

Rag Performance Prediction for Question AnsweringA PREPRINT
conference on empirical methods in natural language processing, pages 14672–14685, 2024.
[5]Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. The distracting effect: Understanding irrelevant
passages in RAG. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors,
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 18228–18258, Vienna, Austria, July 2025. Association for Computational Linguistics.
[6]David Carmel and Elad Yom-Tov.Estimating the Query Difficulty for Information Retrieval. Morgan & Claypool
Publishers, 2010.
[7]Oz Huly, David Carmel, and Oren Kurland. Predicting RAG performance for text completion. InProceedings of
the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR
’25), pages 1283–1293, Padua, Italy, 2025. ACM.
[8]Alireza Salemi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented generation. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages
2395–2400, 2024.
[9]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek,
Nicola Tonellotto, and Fabrizio Silvestri. The power of noise: Redefining retrieval for rag systems. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages
719–729, 2024.
[10] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. Is relevance propagated from retriever to generator in
rag? InEuropean Conference on Information Retrieval, pages 32–48. Springer, 2025.
[11] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. Predicting retrieval utility and answer quality in
retrieval-augmented generation.arXiv preprint arXiv:2601.14546, 2026.
[12] Sara Vera Marjanovic, Haeun Yu, Pepa Atanasova, Maria Maistro, Christina Lioma, and Isabelle Augenstein.
DYNAMICQA: Tracing internal knowledge conflicts in language models. In Yaser Al-Onaizan, Mohit Bansal,
and Yun-Nung Chen, editors,Findings of the Association for Computational Linguistics: EMNLP 2024, pages
14346–14360, Miami, Florida, USA, November 2024. Association for Computational Linguistics.
[13] Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, and Hui Xiong. Seper: Measure retrieval utility through the lens of semantic
perplexity reduction. InThe Thirteenth International Conference on Learning Representations, 2025.
[14] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. InInternational Conference on Learning Representations (ICLR),
2024.
[15] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and
Graham Neubig. Active retrieval augmented generation. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP), pages 7969–7992, 2023.
[16] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. Adaptive-RAG: Learning to adapt
retrieval-augmented large language models through question complexity. In Kevin Duh, Helena Gomez, and
Steven Bethard, editors,Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7036–7050, Mexico
City, Mexico, June 2024. Association for Computational Linguistics.
[17] Xi Wang, Procheta Sen, Ruizhe Li, and Emine Yilmaz. Adaptive retrieval-augmented generation for conversational
systems. InFindings of the Association for Computational Linguistics: NAACL 2025, pages 491–503, 2025.
[18] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: A benchmark for question answering
research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019.
[19] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting of the Association for
Computational Linguistics (ACL), pages 1601–1611, 2017.
[20] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. InProceedings of
the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2369–2380, 2018.
[21] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to
trust language models: Investigating effectiveness of parametric and non-parametric memories. InProceeding of
ACL, pages 9802–9822, 2023.
10

Rag Performance Prediction for Question AnsweringA PREPRINT
[22] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation. InProceedings of the 18th Conference of the European Chapter of the Association for
Computational Linguistics: System Demonstrations, pages 150–158, 2024.
[23] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie
Ma, Honghao Liu, et al. A survey on llm-as-a-judge.arXiv preprint arXiv:2411.15594, 2024.
[24] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in neural
information processing systems, 36:46595–46623, 2023.
[25] Sai Shridhar Balamurali and Lu Cheng. Revisiting nli: Towards cost-effective and human-aligned metrics for
evaluating llms in question answering.arXiv preprint arXiv:2511.07659, 2025.
[26] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu
Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv preprint arXiv:2212.03533, 2022.
(E5).
[27] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke
Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach.arXiv preprint
arXiv:1907.11692, 2019.
[28] Moritz Laurer, Wouter van Atteveldt, Andreu Casas, and Kasper Welbers. Building efficient universal classifiers
with natural language inference, 2023.
[29] Falcon-LLM Team. The falcon 3 family of open models.https://huggingface.co/blog/falcon3, 2024.
[30] Wikimedia Foundation. Wikipedia dump 20181220, 2018. Data snapshot from December 20, 2018.
[31] Claudia Hauff, Djoerd Hiemstra, and Franciska de Jong. A survey of pre-retrieval query performance predictors.
InProceedings of the 17th ACM conference on Information and knowledge management, pages 1419–1420, 2008.
[32] Ying Zhao, Falk Scholer, and Yohannes Tsegay. Effective pre-retrieval query performance prediction using
similarity and variability evidence. InProceedings of the 30th European Conference on Information Retrieval
(ECIR), pages 52–64, 2008.
[33] K. L. Kwok. A new method of weighting query terms for ad-hoc retrieval. InProceedings of the 19th Annual
International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 187–195,
1996.
[34] Yun Zhou and W. Bruce Croft. Query performance prediction in web search environments. InProceedings of the
30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages
543–550, 2007.
[35] Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber, and Gad Markovits. Predicting query performance by
query-drift estimation.ACM Transactions on Information Systems (TOIS), 30(2):11, 2012.
[36] Yongquan Tao and Shengli Wu. Query performance prediction by considering score magnitude and variance
together. InProceedings of the 23rd ACM International Conference on Information and Knowledge Management
(CIKM), pages 1891–1894, 2014.
[37] Anna Shtok, Oren Kurland, and David Carmel. Query performance prediction using reference lists.ACM
Transactions on Information Systems (TOIS), 34(4):1–34, 2016.
[38] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-Embedding: Multi-
Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation, 2024.
[39] William Webber, Alistair Moffat, and Justin Zobel. A similarity measure for indefinite rankings.ACM Transactions
on Information Systems (TOIS), 28(4):20, 2010.
[40] Negar Arabzadeh, Maryam Khodabakhsh, and Ebrahim Bagheri. BERT-QPP: Contextualized pre-trained trans-
formers for query performance prediction. InProceedings of the 30th ACM International Conference on
Information & Knowledge Management (CIKM), pages 3707–3716, 2021.
[41] Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Hong, X Pham, O Simon, et al. Smarter, better, faster,
longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference.
arXiv preprint arXiv:2412.13663, 2024. (ModernBERT).
[42] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty
estimation in natural language generation.arXiv preprint arXiv:2302.09664, 2023. (Entropy for Uncertainty).
[43] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Keshwam, et al. The llama 3 herd of models.
arXiv preprint arXiv:2407.21783, 2024.
11

Rag Performance Prediction for Question AnsweringA PREPRINT
[44] Stephen E Robertson, Steve Walker, Micheline Hancock-Beaulieu, Aarron Gull, and Marianna Lau. Okapi at trec.
InProceedings of the 1st Text REtrieval Conference (TREC), pages 21–30, 1992.
[45] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. Pyserini:
A python toolkit for reproducible information retrieval research with sparse and dense representations. In
Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information
Retrieval, pages 2356–2362, 2021.
[46] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus.IEEE Transactions on
Big Data, 7(3):535–547, 2019. (FAISS).
[47] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. InInternational Conference on
Learning Representations, 2019.
[48] E. J. Williams. The comparison of regression variables.Journal of the Royal Statistical Society, Series B,
21(2):396–399, 1959.
A Prompt Templates
In this section, we provide the exact prompt templates used for both LLMs, with RAG and no-RAG conditions.
A.1 No-RAG Condition
NO-RAG Q&A
You are an AI assistant that answers questions.
Answer the question concisely:
Question:{QUESTION}
Answer:
A.2 RAG Condition
RAG-based Q&A
You are an AI assistant that answers questions.
Answer the question concisely based on the following passages:
Question:{QUESTION}
Passage 1:{p1}
Passage 2:{p2}
Passage 3:{p3}
Passage 4:{p4}
Passage 5:{p5}
Answer:
12
# TS-RAG: Retrieval Augmented Generation for Time Series Forecasting

**Authors**: Yixiong Xiao, Congxi Xiao, Jingbo Zhou

**Published**: 2026-08-06 16:12:57

**PDF URL**: [https://arxiv.org/pdf/2608.06223v1](https://arxiv.org/pdf/2608.06223v1)

## Abstract
While deep learning models, particularly transformer-based architectures, have shown impressive performance in time series forecasting, the application of retrieval-augmented generation (RAG) in this domain remains limited. Since RAG has proven effective in enhancing the capabilities of large language models by incorporating relevant external information, retrieving similar time series sequences as references might also improve accuracy in time series forecasting tasks. However, most time series models are constrained by limited training data, smaller parameter scales, and a lack of the extensive generative capabilities found in large language models. Simply concatenating reference sequences into the prompt, as done in language models, may not yield the expected results. To address these challenges, we propose a novel approach, TS-RAG, which leverages RAG to enhance forecasting performance. The framework introduces specially designed reference tokens to effectively fuse information from the input sequence with that from retrieved similar sequences, enabling a more robust capture of complex temporal dynamics. Experimental results demonstrate that TS-RAG achieves consistent state-of-the-art performance across several real-world forecasting benchmarks.

## Full Text


<!-- PDF content starts -->

TS-RAG: Retrieval Augmented Generation for Time Series
Forecasting
Yixiong Xiao
xiaoyixiong@baidu.com
Baidu, Inc.
Beijing, ChinaCongxi Xiao
xiaocongxi@baidu.com
Baidu, Inc.
Beijing, China
Shuangli Li
Baidu, Inc.
Beijing, ChinaJingbo Zhou∗
jingbozhou@baidu.com
Baidu, Inc.
Beijing, China
Figure 1: Left: schematic overview of TS-RAG. Right: model performance comparison on multiple forecasting benchmarks.
Abstract
While deep learning models, particularly transformer-based archi-
tectures, have shown impressive performance in time series fore-
casting, the application of retrieval-augmented generation (RAG)
in this domain remains limited. Since RAG has proven effective in
enhancing the capabilities of large language models by incorpo-
rating relevant external information, retrieving similar time series
sequences as references might also improve accuracy in time series
forecasting tasks. However, most time series models are constrained
by limited training data, smaller parameter scales, and a lack of the
extensive generative capabilities found in large language models.
Simply concatenating reference sequences into the prompt, as done
in language models, may not yield the expected results. To address
these challenges, we propose a novel approach, TS-RAG, which
leverages RAG to enhance forecasting performance. The framework
introduces specially designed reference tokens to effectively fuse
information from the input sequence with that from retrieved simi-
lar sequences, enabling a more robust capture of complex temporal
dynamics. Experimental results demonstrate that TS-RAG achieves
consistent state-of-the-art performance across several real-world
forecasting benchmarks.
∗Corresponding author.Keywords
time series forecasting, retrieval augmented generation, RAG
1 Introduction
Time series forecasting plays a crucial role in numerous real-world
applications, including finance, healthcare, energy, and climate
science. The ability to accurately predict future values based on
historical data is essential for decision-making processes across
these domains. In recent years, deep learning models, particularly
transformer-based architectures, have demonstrated remarkable
performance in time series forecasting due to their ability to model
long-range dependencies and capture complex temporal patterns
[24]. These models excel at handling sequential data by using self-
attention mechanisms that allow them to focus on relevant parts
of the input sequence, even over long time horizons. Despite these
advances, most existing models primarily focus on enhancing model
training to improve accuracy, such as incorporating multi-scale
mixing [ 33] and time imaging techniques [ 19]. These approaches
aim to capture the intricate relationships and dependencies across
different time scales within the time series. However, research into
how to effectively leverage retrieval-augmented mechanisms to
enhance forecasting accuracy is relatively scarce.
arXiv:2608.06223v1  [cs.AI]  6 Aug 2026

Xiao et al.
Retrieval-Augmented Generation (RAG) has emerged as an ef-
fective paradigm in natural language processing (NLP), where it
enhances large language models by retrieving and incorporating
relevant external information during inference. This approach has
been shown to improve the accuracy and generalization of LLMs,
particularly in knowledge-intensive tasks, by enabling the model
to access external sources of information beyond its training data.
RAG leverages the ability of retrieval systems to pull in pertinent
data from large corpora, ensuring that the model’s outputs are
informed by relevant examples, improving its robustness and flex-
ibility. In the context of time series forecasting, time series data
often exhibits highly intricate patterns, such as seasonality, trends,
and volatility. RAG can improve performance by retrieving similar
historical sequences that share key characteristics with the current
input, allowing the model to pay closer attention to the intricate
patterns in these similar sequences, thereby improving prediction
accuracy. From this perspective, extending RAG from NLP to time
series forecasting thus presents a promising approach.
However, incorporating RAG into time series models faces sev-
eral challenges. While a straightforward adaptation might involve
retrieving similar historical sequences and concatenating them with
the input sequence, as is done in large language models (LLMs), this
approach may not be directly effective in time series forecasting
because it relies on the generalization ability of large language
models. Despite some efforts to develop foundational time series
models, most are still limited by smaller model sizes and insuffi-
cient training data, lacking the generative capabilities of LLMs. As
a result, simply appending retrieved sequences to the input does
not facilitate effective information fusion, since time series models
generally lack the extensive generative power of LLMs and require
specialized methods to integrate external information meaning-
fully. Another key challenge is the efficiency of similarity retrieval,
particularly when using traditional methods like Dynamic Time
Warping (DTW). These methods can be computationally expensive,
requiring substantial time to identify similar sequences from large
datasets. As a result, even if retrieval improves forecasting perfor-
mance, the time required for inference can be prohibitive, limiting
the scalability and real-time applicability of such models.
To address these challenges, we propose TS-RAG, a novel frame-
work that leverages retrieval-augmented generation for time series
forecasting. TS-RAG introduces specially designed reference tokens
that effectively integrate information from both the input sequence
and retrieved similar sequences. This mechanism not only helps
fuse complex temporal dependencies from historical data but also
ensures that external information is incorporated meaningfully.
Moreover, by employing vector search techniques for efficient sim-
ilarity retrieval, TS-RAG significantly reduces the computational
burden associated with traditional methods like Dynamic Time
Warping (DTW). As a result, TS-RAG enables faster inference times
without compromising on predictive accuracy, enhancing general-
ization across diverse forecasting tasks.
We evaluate TS-RAG on multiple real-world time series forecast-
ing benchmarks, demonstrating that it consistently outperforms
existing state-of-the-art methods. Our results highlight the poten-
tial of retrieval-augmented generation in advancing time series
forecasting by enabling models to leverage external information in
a meaningful and structured way. This work contributes to bridgingthe gap between retrieval-based augmentation techniques and time
series forecasting, opening new avenues for improving predictive
performance with additional references.
2 Related Work
2.1 Time Series Forecasting
Deep learning has been widely applied in time series forecasting, pri-
marily utilizing CNNs and RNNs. Convolutional Neural Networks
(CNNs) extract localized temporal features using convolutional fil-
ters, making them effective in capturing short-term dependencies
in time series data [ 2,3]. Recurrent Neural Networks (RNNs) and
their variants, such as Long Short-Term Memory (LSTM) networks
and Gated Recurrent Units (GRUs), have been extensively used for
sequence modeling due to their ability to capture long-term depen-
dencies [ 13]. However, RNN-based models suffer from limitations
such as vanishing gradients and sequential computation inefficien-
cies, making them less scalable for long-horizon forecasting tasks.
The introduction of Transformer-based architectures [ 24] revolu-
tionized time series forecasting by providing a mechanism for cap-
turing long-range dependencies without the need for sequential pro-
cessing. Unlike RNNs, Transformers use self-attention mechanisms
to model dependencies across arbitrary time steps, significantly
improving efficiency and scalability. Several Transformer variants
have been developed specifically for time series forecasting. Log-
Trans [ 18] introduced convolutional self-attention to improve effi-
ciency. Informer [ 31] proposed the probSparse self-attention mech-
anism, reducing computational complexity while preserving predic-
tive accuracy. Autoformer [ 28] incorporated seasonal-trend decom-
position, making the model more robust to non-stationary patterns.
FedFormer [ 32] extended this concept by using frequency-domain
attention to improve long-range forecasting. Despite their effec-
tiveness, Transformer-based models still face scalability challenges,
particularly in handling large-scale datasets and high-dimensional
multivariate forecasting. Recent studies suggest that even simple
linear models can achieve competitive results [ 29], leading to the
development of lightweight architectures such as MLP-based fore-
casting models [ 4,6,25]. These models use time series decomposi-
tion techniques and multi-periodicity analysis to efficiently extract
temporal features while reducing computational overhead.
The recent emergence of Time Series Foundation Models (TSFMs)
has further advanced time series forecasting. Unlike traditional
models trained on domain-specific datasets, TSFMs leverage large-
scale, multi-domain pretraining to achieve zero-shot forecasting
capabilities. TimeGPT-1 [ 10] was the first closed-source TSFM,
demonstrating the potential of large pre-trained models for general
time series prediction. ForecastPFN [ 8] achieved strong zero-shot
forecasting performance by pretraining on synthetic time series
data. TimesFM [ 7] introduced patch-based decoder architectures,
enhancing computational efficiency for long-horizon forecasting.
To further improve generalization, recent works have focused on
multi-source training datasets. MOIRAI [ 27] constructed LOTSA, a
large open-source time series dataset, and trained a masked encoder-
based foundation model that achieved competitive or superior
performance to full-shot models. Tiny Time Mixers (TTMs) [ 9]
adopted a lightweight mixing-based architecture, demonstrating

TS-RAG: Retrieval Augmented Generation for Time Series Forecasting
strong zero-shot forecasting capabilities. Although TSFMs signifi-
cantly improve generalization across domains, they still rely heavily
on implicit knowledge learned during pretraining, limiting their
ability to adapt to rare, non-stationary, or previously unseen time
series patterns. To address this challenge, recent research has intro-
duced retrieval-augmented methods, which explicitly incorporate
external knowledge into the forecasting process.
2.2 Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) has been widely explored
in natural language processing (NLP) as a method to enhance
model capabilities by retrieving external knowledge during infer-
ence [ 12,17]. Traditional large language models (LLMs) often suffer
from knowledge cutoffs and hallucinations, as they rely solely on
parametric memory. RAG mitigates these limitations by dynami-
cally retrieving relevant information from large-scale corpora, en-
suring more accurate and contextually relevant responses [ 16]. In
text generation, RAG has been applied to knowledge-grounded
tasks such as question answering, summarization, and fact verifica-
tion, significantly improving zero-shot and few-shot performance.
Furthermore, in dialogue systems, RAG enables more personalized
and context-aware responses by retrieving user-specific informa-
tion or relevant conversational history [ 14]. Compared to fully
parametric models, retrieval-augmented approaches allow for more
modular and interpretable knowledge integration, reducing the
need for continual fine-tuning while improving adaptability across
different domains.
Beyond NLP, retrieval-augmented techniques have been success-
fully applied in other domains, including computer vision and struc-
tured data modeling. In image generation, retrieval-based models
enhance quality by selecting relevant visual features, style refer-
ences, or object compositions from external image databases [ 5].
This allows for more detailed and coherent visual outputs com-
pared to purely generative approaches. In structured data tasks,
retrieval-based learning has been explored in various contexts, such
as nearest-neighbor retrieval [ 30], kernel-based retrieval [ 21], and
prototype-based learning [ 1]. These techniques demonstrate that
integrating external information improves prediction accuracy and
robustness in domains like finance, healthcare, and recommenda-
tion systems. More recently, retrieval-augmented deep learning
methods have been proposed for tabular data forecasting, where
models leverage attention-like retrieval mechanisms to select rele-
vant historical patterns and enhance prediction accuracy [11].
Despite its success in NLP and structured data applications,
retrieval-augmented forecasting remains underexplored in time
series research. Traditional forecasting models assume that all nec-
essary information is encoded within model weights, limiting adapt-
ability to unseen distributions. Early retrieval-based forecasting
methods, such as ReTime [ 15], introduced relational retrieval mech-
anisms to improve forecasting and imputation for incomplete se-
quences. More recently, RAFT (Retrieval-Augmented Forecasting of
Time-Series) [ 23] proposed an explicit retrieval module that selects
historical patterns similar to the input sequence from the training
dataset. By incorporating retrieved patterns alongside learned repre-
sentations, RAFT enhances forecasting accuracy while maintaining
model efficiency. Unlike Transformer-based approaches that relysolely on self-attention to implicitly capture temporal dependencies,
RAFT explicitly retrieves and integrates relevant past sequences,
improving generalization to diverse time series distributions.
3 Methodology
3.1 Problem Definition
Given a multivariate time seriesx(𝑖)
1:𝐿∈R𝐶×𝐿, where𝐶denotes the
number of channels (or variables) and 𝐿represents the sequence
length, the objective is to predict the future values:
x(𝑖)
𝐿+1:𝐿+𝑇∈R𝐶×𝑇.(1)
Traditional forecasting models rely solely on the given sequence
x(𝑖)
1:𝐿to capture temporal dependencies. However, real-world time se-
ries often exhibit recurring patterns across different periods, events,
or contexts. Retrieval-augmented forecasting enhances predictive
performance by integrating relevant past sequences that share sim-
ilar trends.
We assume access to a historical database Dcontaining a collec-
tion of past time series segments. Using a similarity-based retrieval
functionR, we identify and retrieve the most relevant sequences:
𝑟(𝑖)=R x(𝑖),D={𝑟(𝑖,𝑛)}𝑁
𝑛=1, 𝑟(𝑖,𝑛)∈R𝐶×𝐿.(2)
These retrieved sequences provide additional historical context
that can aid forecasting. The forecasting task is thus reformulated
as:
ˆx(𝑖)
𝐿+1:𝐿+𝑇=𝑓𝜃 x(𝑖)
1:𝐿,𝑟(𝑖),(3)
where𝑓𝜃represents the retrieval-augmented forecasting model.
By incorporating these reference sequences, the model gains a
broader temporal perspective, allowing it to detect patterns that
may not be apparent from the input sequence alone.
3.2 Embedding Process
3.2.1 Input Sequence Embedding.To effectively process the time
series data, each channel 𝑐ofx(𝑖)
1:𝐿is divided into non-overlapping
patches of length𝑝:
x(𝑖,𝑐)
patch=
x(𝑖,𝑐)
1:𝑝,x(𝑖,𝑐)
𝑝+1:2𝑝,...,x(𝑖,𝑐)
𝐿−𝑝+1:𝐿
∈R𝐿
𝑝×𝑝.(4)
To incorporate retrieved reference sequences, we follow an idea
reminiscent of "global tokens" in multi-modal contexts. Specifically,
each retrieved sequence 𝑟(𝑖,𝑛)is associated with a learnable refer-
ence tokenv(𝑛)
ref(𝑐). These tokens act as compressed representations
of historical patterns and are prepended to the patchified input:
˜x(𝑖,𝑐)
patch=
v(1)
ref(𝑐),v(2)
ref(𝑐),...,v(𝑁)
ref(𝑐),x(𝑖,𝑐)
patch
.(5)
The final input embedding is computed as:
x(𝑖)
embed=𝑊𝑥˜x(𝑖)
patch+𝑏𝑥+PE( ˜x(𝑖)
patch),(6)
where𝑊𝑥and𝑏𝑥are learnable parameters, and PE(·) denotes
the positional encoding function.

Xiao et al.
Figure 2: The Framework of TS-RAG.
3.2.2 Reference Sequences Embedding.Each retrieved sequence
𝑟(𝑖,𝑛)undergoes independent patchification:
r(𝑖,𝑐,𝑛)
patch=
𝑟(𝑖,𝑐,𝑛)
1:𝑝,...,𝑟(𝑖,𝑐,𝑛)
𝐿−𝑝+1:𝐿
.(7)
The corresponding embeddings are then computed as:
r(𝑖,𝑛)
embed=𝑊𝑥r(𝑖,𝑛)
patch+𝑏𝑥+PE(𝑟(𝑖,𝑛)
patch).(8)
3.3 Reference Fusion
To integrate both the input sequence and retrieved sequences, we
utilize self-attention and cross-attention mechanisms:
(1)Self-Attention: Captures intrinsic temporal dependencies
within the input sequence.
(2)Cross-Attention: Aligns retrieved sequences with the in-
put sequence, allowing the model to selectively extract
useful information.
3.3.1 Self-Attention.Self-attention is applied to the target sequence
embeddingh(𝑖)
𝑥:
h′(𝑖)
𝑥=softmaxh(𝑖)
𝑥𝑊𝑄(h(𝑖)
𝑥𝑊𝐾)⊤
√
𝑑
h(𝑖)
𝑥𝑊𝑉.(9)
3.3.2 Cross-Attention.Cross-attention is applied to integrate re-
trieved sequences:h(𝑖)
𝑥,𝑟=softmaxh(𝑖)
𝑥𝑊𝑄(h(𝑖)
𝑟𝑊𝐾)⊤
√
𝑑
h(𝑖)
𝑟𝑊𝑉.(10)
The final hidden representations are passed through a feed-
forward network, and the model outputs the forecast ˆx(𝑖)
𝐿+1:𝐿+𝑇,
leveraging both self-learned representations and retrieved historical
patterns to improve predictive accuracy.
4 Experiments
DatasetWe conduct our experiments on six widely used bench-
mark datasets for long-term time series forecasting: ETTh1, ETTh2,
ETTm1, ETTm2, Electricity, and Weather. These datasets span vari-
ous domains, including energy consumption, electricity load fore-
casting, and meteorological measurements, ensuring a comprehen-
sive evaluation of our model’s performance. The ETTh1 and ETTh2
datasets contain hourly transformer temperature and power load
readings, while ETTm1 and ETTm2 provide minute-level variations
of the same. The Electricity dataset records hourly consumption
data of 321 customers, making it suitable for demand forecasting
tasks. The Weather dataset contains 21 meteorological indicators,
including temperature, humidity, wind speed, and pressure. We fol-
low the standard data preprocessing and train-validation-test splits
as previous works, including TimeXer [ 26] and iTransformer [ 20],
ensuring a fair comparison. The input sequence length is set to 96

TS-RAG: Retrieval Augmented Generation for Time Series Forecasting
time steps, and the model predicts the next 96, 192, 336, or 720 time
steps.
BaselinesTo evaluate the effectiveness of our approach, we com-
pare it against several state-of-the-art time series forecasting mod-
els, including Transformer-based and retrieval-augmented models.
The Transformer-based baselines include Informer [ 31], which in-
troduced the probSparse self-attention mechanism to improve effi-
ciency, Autoformer [ 28], which leveraged seasonal-trend decompo-
sition for better forecasting interpretability, FedFormer [ 32], which
incorporated frequency-domain attention mechanisms, PatchTST [ 22],
which utilized patch-based attention mechanisms for long-range
dependencies, and TimeXer [ 25], which explicitly modeled exoge-
nous variables through patch-wise and variate-wise cross-attention
mechanisms. Our model follows the same experimental settings
as Timer to ensure a direct comparison between retrieval-based
approaches and other forecasting methods.
Implementation DetailsAll experiments are implemented in
PyTorch and conducted on an NVIDIA V100 GPU with 32GB of
memory. To ensure consistency across experiments, we adopt the
same hyperparameter settings as iTransformer [ 20]. Our model
employs a retrieval-augmented structure, where a retriever selects
the most relevant historical patterns based on the input sequence.
We train the model using a sliding window approach, with an input
length of 96 and forecasting horizons of 96, 192, 336, and 720. Train-
ing is performed using the Adam optimizer with hyperparameters
𝛽1=0.9,𝛽2=0.999, and a learning rate of 0.001 with cosine decay.
Additionally, we train a Temporal Convolutional Network (TCN)
to retrieve time series sequences most similar to the input. In our
ablation study, we compare retrieval performance using Euclidean
distance and Dynamic Time Warping (DTW). For evaluation, we
retrieve the most similar sequences to the input to generate initial
forecast results. We also report the impact of varying the number
of retrieved sequences in the ablation study. Model performance is
measured using Mean Squared Error (MSE) and Mean Absolute Er-
ror (MAE), with all reported results averaged over five independent
runs to account for experimental variability.
4.1 Main Results
Table 1 presents the results of our model alongside baseline models.
A lower MSE or MAE indicates better forecasting performance.
TS-RAG-CM and TS-RAG denote models that consider and do not
consider channel dependency, respectively. As shown in the table,
TS-RAG-CM achieves the best or near-best performance across
all six datasets (ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather),
demonstrating superior predictive accuracy. Specifically, it achieves
the lowest average MSE (0.310) and MAE (0.348), outperforming
all other models. This result validates the effectiveness of retrieval-
augmented generation (RAG) in enhancing multivariate time series
forecasting performance.
TS-RAG also exhibits competitive results, ranking third overall.
Notably, TS-RAG only considers relationships between the refer-
ence token, the input sequence within the same channel, and the
corresponding reference sequence, without explicitly modeling
cross-channel dependencies. Despite this limitation, it still achieves
results comparable to iTransformer, which explicitly incorporates
channel mixing. This highlights the significant contribution of RAGin a channel-independent setting, underscoring its potential to en-
hance time series forecasting even without explicit inter-channel
interactions.
In contrast, traditional regression-based methods such as RLin-
ear and DLinear, as well as deep learning baselines like PatchTST,
Crossformer, TiDE, and TimesNet, generally underperform com-
pared to TS-RAG-CM. This further reinforces the effectiveness of
retrieval-augmented approaches in multivariate time series fore-
casting.
4.2 Ablation Study
Compared to traditional time series forecasting models, TS-RAG
introduces a retrieval module that brings additional considerations,
such as the choice of retrieval method, the number of retrieved
sequences, and the integration strategy with the forecasting model.
Each of these factors can significantly impact the final predictive
performance. Therefore, our ablation study focuses on systemati-
cally analyzing and comparing the effects of these components on
model performance.
4.2.1 Comparison with Direct Sequence Concatenation.Consid-
ering that RAG is most widely applied in the NLP domain, where
retrieved sequences are typically concatenated directly into the
model’s prompt, we first evaluate whether a similar approach is
effective for time series forecasting. To assess this, we conduct
experiments where retrieved reference sequences are directly con-
catenated before the input sequence and then fed into PatchTST for
model training. Figure 3 presents the comparison between the di-
rect sequence concatenation approach with our proposed TS-RAG
framework. It is observed that TS-RAG and TS-RAG-CM consis-
tently achieve lower MSE compared to Direct Sequence Concatena-
tion across different output lengths. At 96 length, TS-RAG reduces
MSE by 12.1%, while TS-RAG-CM improves it by 14.2%. For 192
length, TS-RAG achieves a 13.8% reduction, and TS-RAG-CM im-
proves it by 17.6%. At 336 length, TS-RAG lowers MSE by 14.0%, and
TS-RAG-CM further reduces it by 17.4%. The largest improvement
is observed at 720 length, where TS-RAG decreases MSE by 16.2%
and TS-RAG-CM achieves an 18.2% reduction. These results show
that TS-RAG and TS-RAG-CM improve forecasting accuracy across
different horizons. These results indicate that direct sequence con-
catenation is not as effective in time series forecasting models as it
is in NLP tasks. Unlike large language models (LLMs), which pos-
sess strong generalization and generative capabilities, time series
forecasting models struggle to effectively utilize concatenated his-
torical sequences in a meaningful way. Our findings highlight the
necessity of the reference fusion framework proposed in TS-RAG.
By introducing a reference fusion mechanism rather than merely
concatenating them, TS-RAG and TS-RAG-CM consistently achieve
lower MSE across different forecasting horizons.
4.2.2 Comparison across different reference numbers.Another fac-
tor that may impact model performance is the number of retrieved
similar sequences. In Table 1, we reported results considering only
the top 1 similar sequence. Now, we examine how different refer-
ence numbers affect the performance of TS-RAG and TS-RAG-CM
(Figure 4). Across different prediction lengths, the results show that
TS-RAG and TS-RAG-CM achieve the best performance when using

Xiao et al.
Table 1: Multivariate forecasting results. We compare extensive competitive models under different prediction lengths. The
look-back length𝐿is set to 96 for all baselines. Results are averaged from all prediction lengths𝑆={96,192,336,720}.
Model TS-RAG-CM TS-RAG TimeXer iTransformer RLinear PatchTST Crossformer TiDE TimesNet DLinear
ECL0.169/ 0.270 0.181 / 0.279 0.171 / 0.270 0.178 / 0.270 0.219 / 0.298 0.205 / 0.290 0.244 / 0.334 0.251 /0.2440.192 / 0.295 0.212 / 0.300
ETTh10.437/ 0.439 0.438 / 0.442 0.437 / 0.437 0.454 / 0.447 0.446 /0.4340.469 / 0.454 0.529 / 0.522 0.541 / 0.507 0.458 / 0.450 0.456 / 0.452
ETTh20.366/ 0.396 0.374 / 0.399 0.367 /0.3960.383 / 0.407 0.374 / 0.398 0.387 / 0.407 0.942 / 0.684 0.611 / 0.550 0.414 / 0.427 0.559 / 0.515
ETTm10.379/0.3950.383 / 0.399 0.382 / 0.397 0.407 / 0.410 0.414 / 0.407 0.387 / 0.400 0.512 / 0.496 0.419 / 0.419 0.400 / 0.406 0.403 / 0.407
ETTm20.266/0.3180.275 / 0.322 0.274 / 0.322 0.288 / 0.332 0.286 / 0.327 0.281 / 0.326 0.757 / 0.610 0.358 / 0.404 0.291 / 0.333 0.350 / 0.401
Weather0.241/ 0.272 0.252 / 0.276 0.241 /0.2710.258 / 0.278 0.272 / 0.291 0.259 / 0.281 0.259 / 0.315 0.271 / 0.320 0.259 / 0.287 0.265 / 0.317
AVG0.310/0.3480.317 / 0.353 0.312 / 0.349 0.328 / 0.357 0.335 / 0.359 0.331 / 0.360 0.540 / 0.493 0.409 / 0.407 0.336 / 0.366 0.374 / 0.399
Figure 3: Comparison of TS-RAG-CM, TS-RAG with Direct Sequence Concatenation.
a single reference sequence. In contrast, incorporating two or four
reference sequences generally leads to higher MSE and MAE, sug-
gesting that additional retrieved sequences do not always enhance
forecasting accuracy and may introduce noise. A possible explana-
tion is that retrieving too many similar sequences may introduce
redundant or conflicting patterns, making it harder for the model
to extract relevant information and effectively integrate it into the
forecasting process.
4.2.3 Comparison of different retrieval approaches.In addition
to information fusion strategies and the number of retrieved se-
quences, another crucial factor affecting model performance is the
retrieval approach used to find similar sequences. The choice of
retrieval method not only impacts forecasting accuracy but also
plays a significant role in model inference efficiency. As shown in
Table 2, TCN-based retrieval achieves the best forecasting perfor-
mance, with lower MSE and MAE compared to DTW-based and
Euclidean Distance (ED) retrieval methods. Although TCN-based
retrieval is slightly slower than ED retrieval (0.004s vs. 0.002s), it is
still significantly faster than DTW-based retrieval (14.27s), making
it a more practical choice for real-time applications.5 Conclusion
In this work, we introduced TS-RAG, a novel retrieval-augmented
generation (RAG) framework for time series forecasting, which
effectively integrates retrieved historical sequences into the fore-
casting process. Our approach addresses key limitations of exist-
ing deep learning models by incorporating external information
through reference tokens, facilitating a more structured fusion
of complex temporal dependencies. By leveraging efficient vector
search techniques instead of computationally expensive methods
like Dynamic Time Warping (DTW), TS-RAG achieves faster in-
ference while maintaining high predictive accuracy.Our extensive
evaluation on multiple real-world time series forecasting bench-
marks demonstrates that TS-RAG consistently outperforms state-
of-the-art models, highlighting its potential to enhance forecasting
performance in both common and rare-event scenarios. Unlike stan-
dard transformer-based architectures that rely solely on internal
feature extraction, our framework retrieves and utilizes relevant
external sequences, allowing the model to make better-informed
predictions, particularly when training data is scarce or lacks di-
versity.This work bridges the gap between retrieval-augmented
methods and time series forecasting, showcasing how ideas from

TS-RAG: Retrieval Augmented Generation for Time Series Forecasting
Figure 4: Comparison of TS-RAG-CM, TS-RAG across different reference numbers.
Table 2: Comparison of model performance with different retrieval approaches. The look-back length 𝐿is set to 96 for all
baselines. Results are averaged from all prediction lengths𝑆={96,192,336,720}.
Model TCN DTW ED
TS-RAG 0.275 / 0.322 0.278 / 0.325 0.275 / 0.323
TS-RAG-CM 0.266 / 0.318 0.271 / 0.322 0.270 / 0.319
Retrieval Time 0.004 14.27 0.002
natural language processing (NLP) can be effectively adapted to the
time series domain.
References
[1] S. O. Arik and T. Pfister. 2020. ProtoNets: Prototype-based Learning for Few-Shot
Classification. InInternational Conference on Learning Representations (ICLR).
[2]Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation
of generic convolutional and recurrent networks for sequence modeling.arXiv
preprint arXiv:1803.01271(2018).
[3] Anastasia Borovykh, Sander Bohte, and Cornelis W Oosterlee. 2017. Conditional
time series forecasting with convolutional neural networks.arXiv preprint
arXiv:1703.04691(2017).
[4]Wei Chen, Fang Liu, Li Zhou, and Yu Wang. 2023. TSMixer: A lightweight
time series model with MLP-mixer architecture.arXiv preprint arXiv:2305.10877
(2023).[5] Xiaowei Chen, Yuning Jiang, Jiuxiang Gu, Wenjing Liu, Xinyue Yang, Mingfei
Tang, Xiaolong Wei, Ran Yu, Shailza Choudhary, Baoyuan Wu Mehta, et al .
2023. ReImagen: Retrieval-Augmented Text-to-Image Generation.arXiv preprint
arXiv:2303.08774(2023).
[6] Subhankar Das, Tian Wang, Qian Zhou, and Liang Xu. 2023. TiDE: Time series
forecasting with decomposition-enhanced MLPs.arXiv preprint arXiv:2306.13284
(2023).
[7] Subhankar Das, Wen Zhang, and Fang Liu. 2024. TimesFM: Large-scale founda-
tion model for time series forecasting.arXiv preprint arXiv:2402.07678(2024).
[8]Andrew Dooley, Hyeongseok Bae, and Mengyu Qian. 2024. ForecastPFN:
Pretraining-free zero-shot forecasting via neural processes.arXiv preprint
arXiv:2403.08567(2024).
[9] Raja Ekambaram, Daniel Smith, and Li Huang. 2024. Tiny Time Mixers: Light-
weight zero-shot time series forecasting.arXiv preprint arXiv:2401.05678(2024).
[10] Manuel Garza and Luis Mergenthaler-Canseco. 2023. TimeGPT-1: A universal
time series foundation model.arXiv preprint arXiv:2309.01245(2023).

Xiao et al.
[11] Yury Gorishniy, Ivan Rubachev, Artem Babenko, and Vasily Svirin. 2024.
Retrieval-Augmented Tabular Models for Forecasting and Decision-Making.
arXiv preprint arXiv:2401.09876(2024).
[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval-augmented language model pre-training. InProceedings
of the International Conference on Machine Learning (ICML).
[13] Hassaan Hewamalage, Christoph Bergmeir, and Kasun Bandara. 2021. Recurrent
neural networks for time series forecasting: Current status and future directions.
International Journal of Forecasting37, 1 (2021), 388–427.
[14] Wenhu Huang, Linyi Wang, Yichi Li, Pengcheng He, Wenhao Wu, Jianfeng
Yang, Lu Wang, and Jordan Boyd-Graber. 2023. LAPDOG: Personalized Dialogue
Generation via Retrieval-Augmented Memory.arXiv preprint arXiv:2305.19253
(2023).
[15] Bingxin Jing, Zhao Zhou, Shuai Liu, Weiqing Liu, Yutao Li, Fangli Chen, Xuanjing
Zhang, and Jingen Zhou. 2022. ReTime: Retrieval-based Temporal Representa-
tion Learning for Time-Series Forecasting. InProceedings of the ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD).
[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Angela Fan, Edouard Grave, Armand Joulin, and Luke Zettlemoyer. 2020.
Dense passage retrieval for open-domain question answering. InProceedings of
the Conference on Empirical Methods in Natural Language Processing (EMNLP).
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, and Sebastian Riedel. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks.Advances in Neural Information Processing
Systems (NeurIPS)(2020).
[18] Shiwen Li, Jinpeng Jin, Yuan Xuan, Xiaolin Zhou, Weike Chen, Yan Wang, and
Shuicheng Yan. 2019. Enhancing the locality and breaking the memory bottle-
neck of transformer on time series forecasting.Advances in Neural Information
Processing Systems32 (2019).
[19] Xuan Li, Jie Sun, and Zheng Liu. 2020. Time Imaging for Time Series Forecasting.
InProceedings of the 37th International Conference on Machine Learning, Vol. 119.
5986–5995. https://proceedings.mlr.press/v119/li20a.html
[20] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and
Mingsheng Long. 2024. iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting. doi:10.48550/arXiv.2310.06625 arXiv:2310.06625 [cs].
[21] M. Nader, X. Chen, and D. Smith. 2022. Kernel-Based Retrieval-Augmented
Neural Networks for Tabular Data. InAdvances in Neural Information Processing
Systems (NeurIPS).[22] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023.
A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
http://arxiv.org/abs/2211.14730 arXiv:2211.14730 [cs].
[23] Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz, and Samet Oymak.
2024. Retrieval Augmented Time Series Forecasting. arXiv:2411.08249 [cs.LG]
https://arxiv.org/abs/2411.08249
[24] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in Neural Information Processing Systems30 (2017).
[25] Yuxuan Wang, Haixu Liu, and Jiaxiang Zhang. 2024. TimeMixer: Gener-
alized MLP-mixer architecture for time series forecasting.arXiv preprint
arXiv:2401.12877(2024).
[26] Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu,
Yunzhong Qiu, Jianmin Wang, and Mingsheng Long. 2024. TimeXer: Em-
powering Transformers for Time Series Forecasting with Exogenous Variables.
arXiv:2402.19072 [cs.LG] https://arxiv.org/abs/2402.19072
[27] Jin Woo, Minho Choi, and Hyejin Lee. 2024. MOIRAI: Large open-source time
series foundation models.arXiv preprint arXiv:2401.09765(2024).
[28] Haixu Wu, Jianmin Xu, Yuxuan Wang, and Mingsheng Long. 2021. Autoformer:
Decomposition transformers with auto-correlation for long-term series forecast-
ing.Advances in Neural Information Processing Systems34 (2021), 22419–22430.
[29] Ailing Zeng, Qingsong Yang, Hao Zhou, Zhiqiang Zhang, Hongyuan Xu, Yuyue
Tong, Jianxin Xu, and Xiaolong Wang. 2023. Are transformers effective for time
series forecasting?Advances in Neural Information Processing Systems36 (2023).
[30] Yi Zhang, Rong Jin, and Zhi-Hua Zhou. 2016. kNN-based Feature Selection
Approach for Text Categorization. InProceedings of the AAAI Conference on
Artificial Intelligence.
[31] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianhui Li, Hanting
Xiong, Wensheng Zhang, Yan Zhang, Yang Bai, and Qiao Li. 2021. Informer: Be-
yond efficient transformer for long sequence time-series forecasting.Proceedings
of AAAI Conference on Artificial Intelligence(2021).
[32] Tian Zhou, Ziqing Ma, Qingsong Wen, Liang Sun, Rui Jin, Zaiqiao Ding, Shirui
Pan, and Yao Jin. 2022. Fedformer: Frequency enhanced decomposed transformer
for long-term series forecasting.International Conference on Machine Learning
(2022), 26420–26431.
[33] Zhi Zhou and Tian Xu. 2021. Multi-scale Mixing for Time Series Forecasting.
Journal of Machine Learning Research22, 1 (2021), 1–25. https://www.jmlr.org/
papers/volume22/20-017/20-017.pdf
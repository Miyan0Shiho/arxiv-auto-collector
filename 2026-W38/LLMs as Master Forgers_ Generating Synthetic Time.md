# LLMs as Master Forgers: Generating Synthetic Time Series Data for Manufacturing

**Authors**: Mantek Singh, Jeshwanth Challagundla, Prateek Karnal, Gagan Ganapathy, Vineet Shah, Ridam Arora

**Published**: 2026-09-14 18:01:29

**PDF URL**: [https://arxiv.org/pdf/2609.16155v1](https://arxiv.org/pdf/2609.16155v1)

## Abstract
This paper presents a novel framework leveraging Large Language Models (LLMs) to generate synthetic time series data for manufacturing processes. Motivated by the scarcity of labeled time-series data in real-world manufacturing settings, which hinders the development of robust machine learning models, we explore the potential of LLMs to learn complex temporal dependencies and generate realistic synthetic data. Our approach involves fine-tuning pre-trained LLMs on manufacturing process instructions and employing a Retrieval Augmented Generation (RAG) technique to enhance data diversity and realism. We evaluate our method against traditional time series modeling techniques like ARIMA and LSTMs, using quantitative metrics, PCA analysis, and downstream task performance (anomaly detection). Results demonstrate that our LLM-driven framework outperforms these baselines, generating high-quality synthetic time series data that effectively captures temporal dependencies and statistical properties of real manufacturing data, leading to improvements in downstream task performance.

## Full Text


<!-- PDF content starts -->

LLMs as Master Forgers: Generating Synthetic Time Series
Data for Manufacturing
Mantek Singh*
Liverpool John Moores University
Liverpool, England
mantek.singh2@gmail.comJeshwanth Challagundla
University of Texas at Arlington
Texas, USA
jeshwanth.challagundla@mavs.uta.eduPrateek Karnal
IIT Patna
Patna, India
karnalprateek@gmail.com
Gagan Ganapathy
Stony Brook University
Stony Brook, USA
gaganganapathyas@gmail.comVineet Shah
IIT Indore
Indore, India
vntshh@gmail.comRidam Arora
UMass Amherst
Amherst, USA
ridamarora89@gmail.com
© 2024 IEEE. Personal use of this material is permitted. Permission
from IEEE must be obtained for all other uses, in any current or future
media, including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or redis-
tribution to servers or lists, or reuse of any copyrighted component of
this work in other works. Published in the 2024 International Conference
on Image Processing, Computer Vision and Machine Learning (ICICML).
DOI: 10.1109/ICICML63543.2024.10958017.
Abstract—This paper presents a novel framework leveraging
Large Language Models (LLMs) to generate synthetic time series
data for manufacturing processes. Motivated by the scarcity of
labeled time-series data in real-world manufacturing settings,
which hinders the development of robust machine learning mod-
els, we explore the potential of LLMs to learn complex temporal
dependencies and generate realistic synthetic data. Our approach
involves fine-tuning pre-trained LLMs on manufacturing process
instructions and employing a Retrieval Augmented Generation
(RAG) technique to enhance data diversity and realism. We
evaluate our method against traditional time series modeling
techniques like ARIMA and LSTMs, using quantitative metrics,
PCA analysis, and downstream task performance (anomaly
detection). Results demonstrate that our LLM-driven framework
outperforms these baselines, generating high-quality synthetic
time series data that effectively captures temporal dependencies
and statistical properties of real manufacturing data, leading to
improvements in downstream task performance.
Index Terms—LLM, Artificial Intelligence, Machine Learning,
Time Series, Manufacturing
I. INTRODUCTION
Time series data plays a critical role in modern manu-
facturing, enabling applications like predictive maintenance
[1], process optimization, and quality control [2]. However,
obtaining large, labeled time-series datasets from real-world
manufacturing environments poses significant challenges due
to factors like cost, privacy concerns, and data collection com-
plexities [3]. Synthetic data generation emerges as a promising
solution to address this data scarcity problem.
Traditional approaches to time series modeling, like Autore-
gressive Integrated Moving Average (ARIMA) models, while
capable of capturing linear temporal dependencies, often strug-
gle to model the complex, non-linear dynamics present in real-world data [4]. Long Short-Term Memory (LSTM) networks,
a type of recurrent neural network, offer an improvement
by capturing long-range dependencies in sequential data [5].
However, they can be challenging to train, requiring substantial
amounts of data and careful hyperparameter tuning [6].
Large Language Models (LLMs), with their remarkable
ability to process and generate human-like text, offer a novel
approach to address these limitations [7]. LLMs can be
trained to understand the relationship between manufacturing
process instructions (e.g., standard operating procedures) and
corresponding time series patterns, enabling them to generate
synthetic data that reflects real-world complexities.
Contributions:This paper introduces a novel framework
for generating synthetic time series data in manufacturing
using LLMs and process instructions. Our key contributions
are:
•Novel Framework:Proposing an end-to-end framework
combining LLM fine-tuning, RAG, and a data validator
module to generate realistic and diverse synthetic time
series data for manufacturing.
•Comparative Evaluation:Benchmarking our LLM-
based method against traditional time series modeling
techniques (ARIMA, LSTMs) using quantitative metrics
and demonstrating superior performance.
•Quantitative and Qualitative Evaluation:Conducting
a comprehensive evaluation of the generated data using
statistical similarity metrics, temporal dependence mea-
sures, PCA analysis, and qualitative assessments.
•Downstream Task Improvement:Demonstrating the ef-
fectiveness of the synthetic data by showing performance
improvement on a downstream anomaly detection task
[8].
II. LITERATUREREVIEW
Generating synthetic time series data is a widely researched
topic, with various approaches developed to address the need
arXiv:2609.16155v1  [cs.LG]  14 Sep 2026

for more data in training robust machine learning models.
Existing methods can be broadly categorized into:
A. Traditional Statistical Methods
Traditional time series modeling techniques primarily rely
on statistical models to capture temporal dependencies and
extrapolate patterns observed in the data. Some prominent
approaches include:
•Autoregressive (AR) Models:AR models predict future
values based on a linear combination of past values [4].
•Moving Average (MA) Models:MA models, conversely,
express the time series as a linear combination of past
forecast errors [4].
•Autoregressive Integrated Moving Average (ARIMA)
Models:ARIMA models combine AR and MA compo-
nents and incorporate differencing to account for non-
stationarity in the time series data [4].
•Seasonal ARIMA (SARIMA):SARIMA models extend
ARIMA to capture seasonal patterns present in the data.
While these methods are relatively simple to implement and
interpret, they often struggle to capture complex, non-linear
relationships and long-range dependencies often encountered
in real-world manufacturing data [3].
B. Deep Learning-Based Approaches
Deep learning has revolutionized many areas of machine
learning, including time series modeling. Several deep learning
architectures have been explored for synthetic time series
generation:
•Recurrent Neural Networks (RNNs):RNNs are specifi-
cally designed for sequential data, processing information
through a recurrent loop that allows them to retain
memory of past events [5]. Long Short-Term Memory
(LSTM) networks, a variant of RNNs, are particularly
effective at capturing long-range dependencies in time
series [5].
•Generative Adversarial Networks (GANs):GANs em-
ploy two competing neural networks - a generator and
a discriminator - to synthesize data that resembles the
real data distribution [9]. The generator learns to create
synthetic data, while the discriminator tries to distinguish
between real and synthetic data. This adversarial training
process pushes the generator to produce increasingly
realistic synthetic samples.
•Variational Autoencoders (V AEs):V AEs learn a latent
representation of the input data and use this representation
to generate new samples [10]. They aim to encode the
data distribution in a lower-dimensional latent space,
allowing for controlled data generation.
Deep learning methods, though powerful, often require large
amounts of training data and careful hyperparameter tuning to
achieve optimal performance [6].
C. LLMs for Data Augmentation
Recently, LLMs have gained significant attention for their
remarkable ability to understand and generate human-like textand identify sophisticated patterns [11]. Research has explored
using LLMs for data augmentation in various domains, includ-
ing:
•Text Generation:LLMs can generate realistic text sam-
ples, which can be used to augment text datasets for tasks
like machine translation, summarization, and dialogue
generation [12].
•Code Generation:LLMs can generate code in different
programming languages, enabling data augmentation for
code-related tasks like code completion and bug fixing.
•Image Captioning:LLMs have been used to generate
image captions, which can augment image datasets for
tasks like image retrieval and visual question answering.
However, the application of LLMs for synthetic time series
generation, particularly in the manufacturing domain, remains
relatively unexplored. This paper aims to address this gap
by introducing a novel framework that leverages LLMs for
generating realistic and diverse synthetic time series data for
manufacturing applications.
III. DATASETDESCRIPTION
The dataset used in this study was provided by Brembo, a
leading manufacturer of braking systems. It consists of three
main components:
A. Material Compositions
This component comprises data on 337 distinct friction ma-
terials used in braking systems. Each material is characterized
by its composition, which is defined by the percentages of
60 different raw materials used in its production. These raw
materials are further categorized into 6 distinct classes (labeled
A-F), reflecting their chemical properties or functional roles
in the friction material.
B. Material Constraints
Each viable compound satisfies the given material con-
straints. The constraints specified the minimum and maximum
percentage a material class could contribute to in the final
composition of the compound as follows (Figure 1):
Fig. 1. Material Constraints
For example, each viable compound should have a minimum
of 3% and maximum of 27% of its composition belong to
compounds of class F.

C. Performance Data
For each friction material, 124 braking tests were conducted
under varying conditions. Each braking test generated time-
series data consisting of 31 time steps. At each time step, vari-
ous parameters relevant to braking performance were recorded,
including:
•Pressure:The pressure applied to the braking system.
•Temperature:The temperature of the braking system.
•Speed:The speed of the vehicle during the braking test.
•Coefficient of Friction (Mu):A key indicator of braking
effectiveness, representing the ratio of the force required
to move two surfaces against each other to the force
pressing them together.
The dataset presents a challenge due to the complexity of
the relationships between material compositions, braking test
conditions, and the resulting time-series performance data.
Our proposed framework aims to leverage the information
contained within this dataset to generate synthetic time-series
data that accurately reflects these complex interactions.
The manufacturing scenarios and conditions covered in this
study focus on variations in material composition and standard
braking test parameters, such as initial speed, pressure applica-
tion profiles, and environmental temperature. While the dataset
does not encompass every possible manufacturing scenario
(e.g., extreme weather conditions, manufacturing defects), it
provides a representative range of common variations that are
relevant for evaluating braking performance.
IV. PROPOSEDMETHODOLOGY
Our proposed framework consists of three key modules
(Figure 2):
A. Material Selection Module
This module focuses on generating new material com-
positions. Instead of randomly generating compositions, we
leverage an LLM (GPT-3.5 Turbo) to propose new materials
that adhere to domain-specific constraints. The LLM is fine-
tuned on a dataset of 337 existing friction materials and
their compositions. A Data Validator Module (Section IV-C)
ensures generated compositions meet predefined criteria.
B. Data Generator Module
This module constitutes the core of our framework, respon-
sible for generating the synthetic time-series data for a given
material composition and braking test type. It leverages the
power of LLMs, specifically GPT-3.5 Turbo, and incorporates
a Retrieval Augmented Generation (RAG) technique to en-
hance the realism of the generated data.
•LLM Selection and Prompt Engineering:
– GPT-3.5 Turbo:We selected GPT-3.5 Turbo as our
LLM due to its ability to generate coherent and
contextually relevant text.
– System Prompt Design:A crucial aspect of guiding
the LLM’s behavior is the design of the system
prompt. We crafted a prompt that instructs themodel to act as a ”statistician specializing in braking
systems,” thereby priming it to generate data that
aligns with the expected patterns and relationships
in braking performance data.
•Fine-tuning for Data Structure and Trends:
– Fine-tuning Dataset:To train the LLM on the
structure and general trends of the time-series data,
we fine-tuned it using a subset (5%) of the 41,788
(material, braking id) tuples from our dataset.
– Focus on Form:This fine-tuning stage primarily
focuses on enabling the LLM to learn the desired
output format (a time series of 31 data points with
values for pressure, temperature, speed, and mu) and
to capture the basic trends in the data (e.g., increasing
temperature, decreasing speed).
•RAG for Enhanced Realism:
– Motivation:While fine-tuning helps the LLM grasp
the overall structure and trends, RAG is crucial for
injecting real-world knowledge and generating data
that aligns more closely with the specific nuances of
different material compositions and braking scenar-
ios.
– Nearest Neighbor Retrieval:For a given material
composition, we retrieve its 5 nearest neighbors from
the dataset based on a custom distance function,
which considers both the Euclidean distance in the
60-dimensional raw material space and a reduced 6-
dimensional space obtained by summing materials
by category:
defdistance(m1, m2, alpha):
sixty_dim_distance = euclidean_dist(
sixty_dim_vector(m1),
sixty_dim_vector(m2)
)
six_dim_distance = euclidean_dist(
six_dim_vector(m1),
six_dim_vector(m2)
)
return(alpha[0] *sixty_dim_distance+
alpha[1] *six_dim_distance)
By retrieving and incorporating the performance
data of nearest neighbors, RAG ensures that the
LLM is exposed to a variety of historical scenarios
with similar material compositions. This helps to
generate synthetic data that reflects the potential
range of braking behaviors associated with different
materials and test conditions. Furthermore, by
adjusting the weights in the distance function
(hyperparametersα[0]andα[1]), we can control the
emphasis on overall material similarity versus the
influence of specific material classes, allowing us
to explore a wider range of potential manufacturing
scenarios.
The custom distance function used for nearest neigh-
bor retrieval is crucial for ensuring that the retrieved

Fig. 2. High-level design of the solution
contexts are relevant to the target material compo-
sition. The function considers both the Euclidean
distance in the 60-dimensional raw material space
and a reduced 6-dimensional space obtained by sum-
ming materials by category. This two-level approach
is designed to capture both the overall similarity
in material composition and the potential influence
of specific material classes on braking performance.
The hyperparametersα[0]andα[1]control the rel-
ative weight given to these two distance measures,
allowing for fine-tuning the retrieval process based
on the specific characteristics of the braking system
under consideration.
– Contextual Augmentation:The time-series perfor-
mance data of these retrieved neighbors is included
as additional context in the prompt to the LLM. This
provides the LLM with specific examples of how
similar materials behave under various braking con-
ditions, guiding it to generate more realistic synthetic
data for the target material.
By combining fine-tuning with RAG, we empower the LLM
to not only grasp the general structure and trends of the
data but also to leverage real-world knowledge, resulting in
synthetic time series that are both plausible and reflective of
the complex interactions within the braking system.C. Data Validator Module
This module plays a crucial role in ensuring the quality and
consistency of the synthetic time-series data generated by our
framework. It operates by applying a combination of domain-
specific rules and trend analysis:
•Domain-Specific Rules Enforcement:
– Material Composition Constraints:The validator
checks that the generated material compositions ad-
here to pre-defined constraints based on manufactur-
ing domain knowledge. For example, it enforces the
rule that the proportion of materials from category
B in a composition must fall within the range of 1%
to 30%.
– Physical Plausibility Checks:Additional rules may
be incorporated to ensure that the generated compo-
sitions are physically plausible. This might involve
checking for known incompatibilities between cer-
tain raw materials or ensuring that the total percent-
age of all materials sums to 100%.
•Trend Analysis:
– Correlation Analysis:The validator analyzes the
correlations between different parameters in the gen-
erated time-series data to ensure they are consistent
with the relationships observed in the real data. For
instance, it verifies that pressure and the coefficient

of friction (mu) exhibit an inverse correlation, as is
typically expected in braking systems.
– Curve Behavior Validation:The validator also ex-
amines the overall behavior of the generated curves
for different parameters over time. For example, it
confirms that the temperature curve follows an ex-
pected exponential increase pattern, while the speed
curve demonstrates a linear decrease during braking.
By incorporating these validation steps, we aim to filter out
unrealistic or inconsistent synthetic data, ensuring that the data
used for downstream tasks is both representative and reliable.
V. EXPERIMENTS ANDRESULTS
A. Evaluation Criteria
We employed the following metrics to evaluate the quality
and effectiveness of the generated synthetic time-series data:
•Statistical Similarity:
– KL Divergence:Measures the difference between
the probability distributions of the real and synthetic
data for each parameter (pressure, temperature, mu).
– Wasserstein Distance:Quantifies the minimum
amount of ”work” needed to transform one distri-
bution (real data) into the other (synthetic data).
•Temporal Dependence:
– Dynamic Time Warping (DTW) Distance:As-
sesses the similarity between the temporal patterns
of the real and synthetic time series, allowing for
non-linear alignments in time.
•PCA Analysis:
– Principal Component Analysis (PCA):A dimen-
sionality reduction technique used to visualize and
compare the latent space representations of the real
and synthetic data. This helps in understanding if
the synthetic data captures the underlying structure
of the real data.
•Downstream Task Performance:
– Anomaly Detection Task:We evaluate the impact of
using the generated synthetic data for an anomaly de-
tection task, a common application in manufacturing.
A classifier is trained to identify anomalous braking
events using three different training scenarios:
1) (a) Real data only
2) (b) Real data augmented with synthetic data
3) (c) Synthetic data only
– F1-Score:The performance of the anomaly detec-
tion models is compared using the F1-score, which
balances precision and recall.
B. Results and Analysis
We benchmarked our LLM-based synthetic data generation
approach against traditional time-series modeling techniques,
namely ARIMA and LSTM models. The LSTM Models were
further optimized for the task using Multiple Gain Adaptations
[13] [14]. Both these baseline models were trained on the realmanufacturing dataset. Table I, Table II and Table III present a
comparative analysis of the quantitative results obtained across
the different evaluation metrics.
TABLE I
QUANTITATIVE COMPARISON OF LLM-GENERATED DATA
WITH ARIMA AND LSTM BASELINES (PART 1).
Metric Pressure
(LLM)Pressure
(ARIMA)Pressure
(LSTM)
KL Divergence 0.052 0.115 0.089
Wasserstein Dist. 0.124 0.285 0.231
DTW Distance 1.35 1.87 1.62
TABLE II
QUANTITATIVE COMPARISON OF LLM-GENERATED DATA
WITH ARIMA AND LSTM BASELINES (PART 2).
Metric Temperature
(LLM)Temperature
(ARIMA)Temperature
(LSTM)
KL Divergence 0.085 0.162 0.135
Wasserstein Dist. 0.218 0.394 0.327
DTW Distance 2.07 2.54 2.31
TABLE III
QUANTITATIVE COMPARISON OF LLM-GENERATED DATA
WITH ARIMA AND LSTM BASELINES (PART 3).
Metric Mu (LLM) Mu (ARIMA) Mu (LSTM)
KL Divergence 0.034 0.078 0.062
Wasserstein Dist. 0.096 0.189 0.154
DTW Distance 0.87 1.21 1.05
The results in Table I, Table II and Table III clearly
demonstrate that the synthetic data generated by our LLM-
based framework consistently outperforms both the ARIMA
and LSTM baselines across all evaluated metrics. The LLM-
based approach shows a superior ability to capture both the
statistical properties (as evidenced by lower KL divergence
and Wasserstein distances) and the temporal dependencies
(indicated by lower DTW distances) present in the real man-
ufacturing data.
•PCA Analysis:The effectiveness of the LLM-based
method is further highlighted by the visualizations of
the principal components (Figure 3). The synthetic data
generated by the LLM exhibits a distribution closely re-
sembling that of the real data, indicating that it effectively
captures the underlying structure and relationships within
the data.
•Downstream Task Performance:The superior quality
of the LLM-generated data translates into significant
improvements in downstream task performance. In our
anomaly detection experiment, using the synthetic data
generated by our framework resulted in a 12% improve-
ment in the F1-score compared to training on real data
alone (Figure 4). Furthermore, it outperformed models
trained on data augmented with ARIMA or LSTM-
generated data, reinforcing the value of our approach in
real-world applications.

Fig. 3. PCA analysis on the provided and generated materials
Fig. 4. Anomaly detection performance comparison
VI. DISCUSSION
Strengths:Our framework showcases the potential of LLMs
for generating realistic synthetic time-series data in manufac-
turing, outperforming traditional methods like ARIMA and
LSTMs in capturing the complex patterns inherent in real-
world data. The integration of RAG significantly enhances
the quality and diversity of the generated synthetic data
by effectively leveraging relevant information from historical
data.
Limitations:
•The computational cost associated with fine-tuning and
querying large LLMs can be substantial. Exploring more
efficient fine-tuning techniques like LoRA [15] and
QLoRA [16] and leveraging open-sourced LLMs could
help mitigate this limitation.
•The current framework relies on a pre-defined distance
function for RAG; incorporating more sophisticated re-
trieval and ranking mechanisms [17] could further en-hance the realism of the generated data. Future work
will focus on investigating alternative distance metrics
or learning-based similarity measures for the RAG pro-
cess, which could potentially capture more nuanced re-
lationships between material compositions and braking
performance.
•It’s important to acknowledge that complete coverage of
all possible manufacturing scenarios and conditions is a
challenging task. However, our approach of using RAG
with a carefully designed distance function and fine-tuned
LLM aims to maximize the diversity and realism of the
generated data within the scope of the available dataset.
Future work could explore incorporating additional data
sources or using more sophisticated retrieval techniques
to further enhance the coverage of manufacturing scenar-
ios.
•Furthermore, we can build an ensembled model, which
is known to outperform the individual models [18].
Impact and Applications:This work holds broad implications
for the manufacturing industry, where obtaining large, labeled
datasets is often a major bottleneck. The ability to generate
high-quality synthetic data can significantly accelerate the
development and deployment of data-driven models, enabling
a wide range of applications such as predictive maintenance,
anomaly detection, and process optimization. Furthermore, this
can radically reduce the time cycles required for manufacturing
new compounds.
VII. CONCLUSION
We have presented a novel framework for generating realis-
tic synthetic time series data for manufacturing processes using
LLMs and RAG. Our LLM-driven approach demonstrates
superior performance compared to traditional time series meth-
ods, effectively capturing the intricate temporal dependencies
and statistical properties of real-world manufacturing data.
This is evidenced by the quantitative metrics, PCA analysis,
and the observed improvement in the performance of a down-
stream anomaly detection task. Future research will focus on:
•Enhancing Framework Efficiency:Exploring effi-
cient fine-tuning techniques and leveraging open-sourced
LLMs.
•Exploring Advanced RAG Techniques:Investigating
more sophisticated retrieval and ranking mechanisms to
further improve data realism.
•Expanding Applications:Applying the framework to
other domains beyond manufacturing where synthetic
time series data can be beneficial.
REFERENCES
[1] M. Pecht, “Prognostics and prognostics health manage-
ment,” inEncyclopedia of Structural Health Monitoring.
John Wiley & Sons, Ltd, 2008.
[2] D. C. Montgomery,Introduction to statistical quality
control. John Wiley & Sons, 2008.
[3] G. A. Susto, A. Schirru, S. Pampuri, S. McLoone, and
A. Beghi, “Machine learning for predictive maintenance:

A multiple classifier approach,”IEEE Transactions on
Industrial Informatics, vol. 11, no. 3, pp. 812–820, 2015.
[4] G. E. Box, G. M. Jenkins, and G. C. Reinsel,Time series
analysis: forecasting and control. John Wiley & Sons,
2011.
[5] S. Hochreiter and J. Schmidhuber, “Long short-term
memory,”Neural computation, vol. 9, no. 8, pp. 1735–
1780, 1997.
[6] I. Goodfellow, Y . Bengio, and A. Courville,Deep learn-
ing. MIT press, 2016.
[7] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan,
P. Dhariwalet al., “Language models are few-shot learn-
ers,”Advances in neural information processing systems,
vol. 33, pp. 1877–1901, 2020.
[8] R. Chalapathy and S. Chawla, “Deep learning
for anomaly detection: A survey,”arXiv preprint
arXiv:1901.03407, 2019.
[9] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu,
D. Warde-Farley, S. Ozairet al., “Generative adversarial
nets,”Advances in neural information processing sys-
tems, vol. 27, 2014.
[10] D. P. Kingma and M. Welling, “Auto-encoding varia-
tional bayes,”arXiv preprint arXiv:1312.6114, 2013.
[11] T. Chugh, K. Tyagi, P. Srinivasan, and J. Challagundla,
“State-based dynamic graph with breadth first progres-
sion for autonomous robots,” in2024 IEEE 14th Annual
Computing and Communication Workshop and Confer-
ence (CCWC), 2024, pp. 0365–0369.
[12] J. Wei, M. Bosma, V . Zhao, K. Guu, A. W.
Yu, B. Lester, N. Du, A. M. Dai, and
Q. V . Le, “Finetuned language models are zero-
shot learners,” inInternational Conference on
Learning Representations, 2022. [Online]. Available:
https://openreview.net/forum?id=gEZrGCozdqR
[13] J. Challagundla, K. Tyagi, T. Chugh, and M. Manry,
“Multiple gain adaptations for improved neural net-
works training,” in2024 IEEE 14th Annual Computing
and Communication Workshop and Conference (CCWC),
2024, pp. 0414–0420.
[14] J. Challagundla, “Adaptive multiple optimal learning
factors for neural network training,” Master’s thesis,
UTArlington Department of Electrical Engineering,
2015. [Online]. Available: https://rc.library.uta.edu/uta-
ir/handle/10106/25030
[15] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li,
S. Wang, L. Wang, and W. Chen, “Lora: Low-rank
adaptation of large language models,” 2021. [Online].
Available: https://arxiv.org/abs/2106.09685
[16] T. Dettmers, A. Pagnoni, A. Holtzman, and
L. Zettlemoyer, “Qlora: Efficient finetuning
of quantized llms,” 2023. [Online]. Available:
https://arxiv.org/abs/2305.14314
[17] V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu,
S. Edunov, D. Chen, and W. tau Yih, “Dense passage
retrieval for open-domain question answering,” 2020.
[Online]. Available: https://arxiv.org/abs/2004.04906[18] G. Ganpathy, N. Shrivastava, O. Vyas, M. Singh,
R. Arora, and S. Mishra, “Intrusion detection and attack
classification using an ensemble approach,”International
Research Journal of Engineering and Technology, vol. 7,
pp. 1616–1620, 2020.
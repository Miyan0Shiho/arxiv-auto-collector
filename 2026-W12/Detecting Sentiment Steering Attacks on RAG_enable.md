# Detecting Sentiment Steering Attacks on RAG-enabled Large Language Models

**Authors**: Isha Andrade, Shalaka S Mahadik, Mithun Mukherjee, Pranav M Pawar, Raja Muthalagu

**Published**: 2026-03-17 10:18:47

**PDF URL**: [https://arxiv.org/pdf/2603.16342v1](https://arxiv.org/pdf/2603.16342v1)

## Abstract
The proliferation of large-scale IoT networks has been both a blessing and a curse. Not only has it revolutionized the way organizations operate by increasing the efficiency of automated procedures, but it has also simplified our daily lives. However, while IoT networks have improved convenience and connectivity, they have also increased security risk due to unauthorized devices gaining access to these networks and exploiting existing weaknesses with specific attack types. The research proposes two lightweight deep learning (DL)-based intelligent intrusion detection systems (IDS). to enhance the security of IoT networks: the proposed convolutional neural network (CNN)-based IDS and the proposed long short-term memory (LSTM)-based IDS. The research evaluated the performance of both intelligent IDSs based on DL using the CICIoT2023 dataset. DL-based intelligent IDSs successfully identify and classify various cyber threats using binary, grouped, and multi-class classification. The proposed CNN-based IDS achieves an accuracy of 99.34%, 99.02% and 98.6%, while the proposed LSTM-based IDS achieves an accuracy of 99.42%, 99.13%, and 98.68% for binary, grouped, and multi-class classification, respectively.

## Full Text


<!-- PDF content starts -->

Detecting Sentiment Steering Attacks on
RAG-enabled Large Language Models
Isha Andrade∗, Shalaka S. Mahadik∗, Mithun Mukherjee∗, Pranav M. Pawar∗, Raja Muthalagu∗
∗Department of Computer Science, Birla Institute of Technology and Science, Pilani, Dubai Campus,
Dubai International Academic City, Dubai, United Arab Emirates
Emails:{f20200039, mithun, pranav, raja.m}@dubai.bits-pilani.ac.in, mahadikshalaka4@gmail.com
Abstract—The proliferation of large-scale IoT networks has
been both a blessing and a curse. Not only has it revolutionized
the way organizations operate by increasing the efficiency of
automated procedures, but it has also simplified our daily lives.
However, while IoT networks have improved convenience and
connectivity, they have also increased security risk due to unau-
thorized devices gaining access to these networks and exploiting
existing weaknesses with specific attack types. The research
proposes two lightweight deep learning (DL)-based intelligent
intrusion detection systems (IDS). to enhance the security of IoT
networks: the proposed convolutional neural network (CNN)-
based IDS and the proposed long short-term memory (LSTM)-
based IDS. The research evaluated the performance of both
intelligent IDSs based on DL using the CICIoT2023 dataset. DL-
based intelligent IDSs successfully identify and classify various
cyber threats using binary, grouped, and multi-class classifica-
tion. The proposed CNN-based IDS achieves an accuracy of
99.34%, 99.02% and 98.6%, while the proposed LSTM-based
IDS achieves an accuracy of 99.42%, 99.13%, and 98.68% for
binary, grouped, and multi-class classification, respectively.
Index Terms—Deep learning (DL), Intrusion Detection System
(IDS), Convolutional neural network (CNN), Long-short term
memory (LSTM), CICIoT2023.
I. INTRODUCTION
The growing use of diverse large-scale Internet of Things
(IoT) devices across various industries, including healthcare,
finance, and transportation, is demonstrated in Fig. 1. These
large-scale IoT systems encompass various sensors, commu-
nication protocols, connectivity, and numerous other technolo-
gies, making cyber threats a paramount concern. The imple-
mentation of traditional security techniques is also challenging
due to these unique features. Therefore, IoT devices can serve
as gateways for cyberattacks, putting systems or applications
on both individual and national scales [1].
There are various real-time examples available that highlight
the importance of security mechanisms in today’s IoT world.
In March 2021, hackers infiltrated Verkada, a cloud-based
video surveillance service, exposing private data and live feeds
from more than 150,000 cameras [2]. Moorfields Eye Hospital,
one of the leading ophthalmology hospitals in the UAE, was
targeted by a ransomware group [3]. In 2024, the ransomware
group targeted Roku, a TV streaming service provider, result-
ing in the compromise of approximately 576,000 accounts. It
raised cybersecurity concerns for home-use IoT devices [4].
Therefore, the development of a security model in the IoT
context inspires the research.
Large-scale IoTHealthcare
ProtocolsConnectivitySensors
6G netwrok
Bluetooth
WiFiCellular
networkLoRa
WPAN Zigbee
Finance
TransportationFig. 1: Example of large-scale IoT and its features
A. Motivation
Recently, deep learning models such as CNN and LSTM
have attracted many in academia for the development of
efficient IDS [5], [6]. The CNN models are capable of
understanding features and extracting important information
from raw data. The CNN model’s various layers enable
it to perform tasks such as object detection, cyber threat
detection, and image identification, among others. They are
more popular among academics as they can analyze network
traffic effectively and detect vulnerabilities efficiently [15].
Similarly, LSTM models are good at retaining past data and
predicting future data. These models have the ability to analyze
network traffic logs or event sequences, identifying any sudden
or drastic increases in network traffic that could potentially
pose cyber threats. So, many researchers focused on it for the
development of security mechanisms [5]. Therefore, the goal
of the research is to develop a DL-based intelligent IDS using
CNN and LSTM techniques.arXiv:2603.16342v1  [cs.CR]  17 Mar 2026

B. Contribution
The primary contributions of the paper are outlined below:
•The research proposes two novel DL-based intelligent
IDSs, i.e., the CNN-based IDS and the LSTM-based IDS.
Both the proposed IDSs are lightweight in terms of layers
and neurons used.
•The proposed DL-based intelligent IDS employed the CI-
CIoT2023 security dataset to detect and classify the mali-
cious and benign network traffic. The research examined
the performance of the proposed DL-based intelligent
IDS using three classification types: binary (containing
one attack and one benign class), grouped (containing
seven attack classes and one benign class), and multi-
class (containing 33 attack classes and one benign class).
•The research evaluated the performance of the proposed
DL-based intelligent IDS in comparison to the state-of-
the-art HetIoT CNN-IDS, focusing on binary, grouped,
and multi-class classification perspectives.
The remainder of the research paper is organized as follows:
Section II talks about the literature review. Section III outlines
the research methodology, including datasets and their prepro-
cessing steps, as well as DL-based intelligent IDS. Section IV
presents the simulation setup and performance results, along
with a summary. Section V concludes the research paper.
II. RELATED WORK
The section details various DL techniques used for the
development of security models.
Paper [6] investigates a federated learning (FL) technique
for detecting large-scale IoT network assaults on the CI-
CIoT2023 dataset. The paper utilises balance data and feature
normalisation techniques to identify network attacks. The
paper [7] uses a DNN and a bidirectional LSTM (BiLSTM)
model to learn nonlinear interactions and extract long-term
dependencies in both directions. The model begins with an
incremental principal component analysis (IPCA) approach to
reduce the dimensionality of data.
In the paper [8], the initial step is to convert the PCAP
files into images, which are then preprocessed and categorized
using a CNN. The CNN architecture included a convolutional
layer with ReLU activation, a max pooling layer, and a fully
connected dense layer. Next, paper [9] describes a mixed
deep learning model made up of an MLP and a CNN that
can find distributed denial of service (DDoS) attacks in SDN
settings. Shapley additive explaining (SHAP) and the Bayesian
optimiser are used to select the best features and fine-tune
hyperparameters, respectively. The paper [11] discusses a
hybrid lightweight DL model using a stacked autoencoder
(SAE) and a CNN that is trained on the CICDDoS2019
dataset. An adaptive FLAD approach is proposed in the paper
[12] for detecting DDoS attacks. The FLAD methodology
outperforms traditional FL approaches in both accuracy and
time of convergence.
A mixed LSTM and recurrent neural network model
(LSTM-RNN) is suggested in paper [13] as a way to findDDoS attacks in SDN networks. The model achieved an
accuracy of 99.33% for both the CICDDoS2017 and the
CICDDoS2019 dataset. A multichain technology to detect
DDoS attacks has been proposed in the paper [14]. Existing
methods for finding DDoS attacks at the application layer
can only find a few types. To fix this, paper [16] suggests
an explicit duration RNN (EDRN) that is trained on the
CICDDoS2019 dataset and works at the application layer. The
proposed model achieves an accuracy, recall, and F1 score of
99.6%, 99.3%, and 99.2%, respectively.
A deep CNN (DCNN) model is proposed in the paper
[17] to detect DDoS attacks in an SDN environment. The
proposed DCNN model achieves an accuracy of 99.77%
for the CICDDoS2019 dataset. Four types of classifications
are performed in paper [18]are binary, multi-class with one-
hot encoding, multi-class with a LabelEncoder, and multi-
label classification. The CSE-CICIDS2018 dataset exclusively
trained the proposed model, a four-layer MLP, on DDoS
attacks. A DL-based contractive autoencoder is suggested in
paper [19] as a way to find DDoS attacks in three different
datasets: CICIDS 2017, CICDDoS2019, and NSL-KDD. The
performance is compared with a traditional autoencoder and
other DL models.
A hierarchical LSTM model is proposed in the paper [20].
The model incorporates a dual-LSTM packet classifier to
classify packet data of varying lengths and a single-LSTM
session classifier to enhance accuracy by utilizing the max-
imum amount of session traffic. The paper [21] presented a
comparative study of various supervised (e.g., Alex Neural
Network) and unsupervised models (e.g., Variational Autoen-
coder) trained on the CICDDoS2019 dataset.
In paper [22], a new six-layer framework is suggested as
a way to find attacks on IoT networks. The model is capable
of identifying network attacks such as sinkholes, blackholes,
DDoS attacks, etc. In paper [23], an SDN-IoT framework is
employed to detect DDoS attacks, including zero-day attacks,
using the CICDDoS2019 dataset. The proposed model consists
of three LSTM layers along with one input and one output
layer.
Paper [24] presented two hybrid DL models: a DCNNBiL-
STM (a combination of a CNN and a BiLSTM model) and a
DCNNGRU (a combination of a CNN and a gated recurrent
unit). The performance of the proposed models was compared
to DL models from previous research, such as CNN + LSTM,
DNN, ResNet, etc. In paper [25], a stacked convolutional
denoising AE (SCDAE) is use to get rid of the noise in
the network traffic data. Next, a simple CNN and BiLSTM
model are used to get the data’s spatial and temporal features.
BiLSTM is considered so the relationship between the traffic
can be mined from both front and back.
In paper [26], a DL-based IDS is proposed to detect cyber
attacks from the CICDDoS2019 dataset. The proposed model
is a combination of a BiLSTM, and BiGRU. A hybrid DL
model of CNN, LSTM, AE, and DNN is introduced in the
paper [27] to detect DDoS attacks from the CICDDoS 2019
dataset. The proposed model achieves a detection rate of

TABLE I: Comparison of various state-of-the-art DL methods
used for detecting IoT network attacks from three classifica-
tions perspectives
Type of classifications
Binary Grouped Multi-Class
FL + client DNN [6] 99% - -
DNN-BiLSTM [7] - - 93.13%
DNN [8] 99.71% 98.76% 98.81%
MLP-CNN [9] - - 99.95%
DL + (ADASYN and
SMOTE) [10]99.97% - 99.99%
SAE-CNN-Detection
(SCD) [11]99.1% - 97.2%
LSTM-RNN [13] 99.33% - -
EDRN [16] - - 99.6%
DCNN [17] - - 99.77%
MLP [18] 100% - 99.84%, 99.7%
Contractive AE [19] - - 93.41%, 97.58%
AlexNet, V AE [21] - - 98.81%, 96.7%
FCFFN [22] 93.74% - -
LSTM [23] 99.8% - -
DCNNBiLSTM,
DCNNGRU [24]99.95%, 99.93% - -
CNN-BiLSTM-
Attention [25]- - 93.26%
BiLSTM-BiGRU [26] - - 99.77%
CNN-LSTM-DeepAE-
DNN [27]- - 73.46%
71.42%, a false detection rate of only 0.04%, and, on average,
an accuracy of 73.46%.
Table I summarizes various DL methods to detect cyberat-
tacks in large-scale IoT networks. According to related work,
the DL models presented in the literature take into account
a variety of security datasets, including CICIDS2017, CICD-
DoS2019, NSL-KDD, and others. Neither of these datasets
has been designed for IoT. The literature study shows that
many of the DL models are hybrid such as a CNN-LSTM,
CNN-BiLSTM and complex with a greater number of layers
and neurons. Although these models have produced excellent
results, they can produce a significant amount of computa-
tional overhead. So, the research, aims to propose two novel
lightweight DL-based intelligent IDS, namely, the proposed
CNN-based IDS and the proposed LSTM-based IDS. The
research used the CICIoT2023 dataset, which is a recent
benchmark dataset created exclusively for the IoT environ-
ment. The numerous models provided in the literature study
lack focus on such specific IoT-related datasets, where research
is concentrated. Next, section details the research methodology
used in the research.
III. RESEARCH METHODOLOGY
The section details the two novel DL models namely,
proposed CNN-based IDS and LSTM-based IDS. Moreover
the section details CICIoT23 dataset and data preprocessing
steps. The section also highlights the dataset used for binary
classification, grouped classification and multi-class classifica-
tion.
Fig. 2: Feature importance graph [30]
TABLE II: Top 20 selected features [30]
Sr.no Feature name Sr.no Feature name Sr.no Feature name
1 Srate 8 Header Length 15 syn flag number
2 Rate 9 flow duration 16 rst count
3 Duration 10 Max 17 UDP
4 syn count 11 HTTP 18 fin count
5 Weight 12 Protocol Type 19 Variance
6 ack flag number 13 urg count 20 IAT
7 Number 14 ack count
A. Dataset and data preprocessing
The CICIoT23 dataset is a publicly available security dataset
provided by [28], [29]. The research used the CICIoT2023
dataset to classify IoT network attacks based on three classifi-
cations : binary, grouped, and multi-class. The original dataset
consists of 168 network traffic CSV files combined to form a
single CSV file with 47 columns and 46,686,579 rows. The
dataset used for training the DL models contains only 10%
(i.e., 4,668,653 rows) of the original dataset to prevent resource
exhaustion. The dataset details are provided in Table III. The
dataset originally contained 47 different features. In order
to address the issue of feature dimensionality, the research
focused solely on the top 20 features, selecting them based
on their importance as generated by a random forest regressor
model. The top 20 selected features and feature importance
graph are provided in Table II and in Fig. 2, respectively.
After data preprocessing and feature selection, the pro-
posed CNN-based IDS and LSTM-based IDS were trained
by employing an 80-20 train-test split ratio. The dataset
considered for the research is imbalanced. So, the research
uses ‘stratify’as a hyperparameter for distributing all classes
equally while train-test split [?]. It helps to handle the data
imbalance concern appropriately. So, the model will not be
biased toward the majority class.

TABLE III: Dataset Details
Attack NameTotal
SamplesTraining
SamplesTesting
Samples
Dataset Binary [30] 4,668,653 3,734,922 933,731
Dataset Group [30] 4,668,653 3,734,921 933,732
Dataset Multi [30] 4,570,593 3,656,475 914,118
Datasets after data pre-
processing steps:
-Dataset_Binary
-Dataset_Group
-Dataset_Multi   Sequential model
with Input shape
(20,1)
1D-Maxpooling layer :
Pool size=21D-Convolutional layer :
Filters=32,
Kernel size=3,
Activation=Relu........
1D-Convolutional layer :
Filters=64,
Kernel size=3,
Activation=Relu........
1D-Maxpooling layer :
Pool size=2Fully
connected
layer :
 
Output
classes=2,7,or
34,
Activation=
Sigmoid,
Softmax ........
Convolutional layer and pooling layerCompile model with
optimizer , loss and
metrics
Fit model  with 
Train-set and
Validation-set (0.2)
Evaluate and predict
model with T est-setEpoch=20,
Batch size=64
Cyber threat detection :
Binary classification
Grouped classification
Multi-class classification
Fig. 3: Architecture of proposed CNN-based IDS
B. Architecture of Proposed DL-based IDS
1) Proposed CNN-based IDS:CNNs excel in extracting
spatial features from data, making them ideal for structured
inputs, such as network packet headers [9]. The proposed
CNN-based IDS consists of two convolutional layers, two
max pooling layers, and a fully connected dense output layer.
The first layer is a 1D convolutional layer that has 32 filters
with a kernel size of 3 and a ‘relu’ activation function, which
is followed by a 1D max pooling layer having a pool size
of 2. The third layer is a 1D convolutional layer having 64
filters with a kernel size of 3 and a ‘relu’ activation function,
which is again followed by a max pooling layer of pool size
2. For all three classification, the ‘adam’ optimiser is used.
The loss function, binary cross-entropy is used for binary and
sparse categorical crossentropy is used for grouped and multi-
class classifications. Details of the proposed CNN-based IDS
is presented in Fig. 3 and its hyperparameter settings in V.
2) Proposed LSTM-based IDS:The proposed LSTM-based
IDS use two LSTM layers with 64 neurones and two Dropout
layers. The ‘sigmoid’ activation is used for binary classi-
fication and a ‘softmax’ activation for grouped and multi-
class classification. For all three classification perspectives, the
‘adam’ optimiser with a learning rate of 0.0001 and accuracy
metrics are used. The details of the proposed LSTM-based
IDS is presented in Fig. 4 and its hyperparameter settings in
V.
IV. RESULTS ANDDISCUSSION
Simulation set-up:The details of the software and hardware
setup is illustrated in Table IV. The hyperparameter settings
are illustrated in Table V. The Keras and TensorFlow Python
libraries were used to implement the proposed DL-based IDS.
Performance metrics:The research considered standard per-
formance matrices to evaluate the performance of the proposed
DL-based IDS [6]. They are accuracy, precision, recall, and F1
score.
Datasets after data pre-
processing steps:
-Dataset_Binary
-Dataset_Group
-Dataset_Multi   Sequential model
with Input shape
(1, 20)Fully
connected
layer :
 
Output
classes=2,7,or
34,
Activation=
Sigmoid,
Softmax ........
Compile model with
optimizer , loss and
metrics
Fit model  with 
Train-set and
Validation-set (0.2)
Evaluate and predict
model with T est-setEpoch=20,
Batch size=64
Cyber threat detection :
Binary classification
Grouped classification
Multi-class classificationDroupout layer : 0.2
LSTM layer :
Units=64,
Return_sequence=T rue
Droupout layer : 0.2
LSTM layer :
Units=64,
Return_sequence=T rueFig. 4: Architecture of proposed LSTM-based IDS
TABLE IV: Simulation set-up
Software specification:
Software Tool Jupyter Notebook 6.5.4
Programming Language Python 3.11.4, Sklearn library, Matplotlib 3.7.1
API Tool Keras on TensorFlow 2.14.0
Hardware specification:
Processor Processor Intel(R) Core(TM) i7-8550U CPU @ 1.80 GHz, Windows 11 64-bit OS
RAM 12.0 GB
Graphics Card Intel(R) UHD Graphics 620 (Integrated Graphics Card)
TABLE V: Hyperparameter settings
Hyperparameter settings Value
Binary classificationLoss binary crossentropy
Activation sigmoid
Grouped classification, Multi-class classificationProposed CNN IDSOptimizer adam
Activation softmax
Loss sparse categorical crossentropy
Proposed LSTM IDSOptimizer adam
Activation softmax
Learning rate 0.0001
Loss categorical crossentropy
Performance evaluation:The research investigates the per-
formance of proposed DL-based IDS using the state-of-the-
art HetIoT CNN-IDS [15]. The state-of-the-art HetIoT CNN-
IDS is a unique CNN model that classifies DDoS attacks in
heterogeneous IoT environments. The research employed the
same model to assess and compare its performance using the
CICIoT2023 dataset. The research changed two parameters:
kernel size from 5 to 2 and strides from 2 to 1 when
reimplemented the state-of-the-art HetIoT CNN-IDS [15]. The
research adopts the same parameters as in [15] to evaluate
binary, grouped, and multi-class classification.
A. Binary classification
The details of the dataset for binary classification have
been provided in Table III. The hyperparameter settings are
presented in the Table V. The details of the proposed CNN-
based IDS and the proposed LSTM-based IDS are provided in
Fig. 3 and Fig. 4, respectively. Fig. 5a), 5b), and 5c) illustrate
the training and validation accuracy of the proposed CNN-
based IDS, LSTM-based IDS, and state-of-the-art HetIoT
CNN-IDS [15] for binary classification across 20 epochs. The
proposed CNN-based IDS achieves an accuracy of 99.34%,
whereas the proposed LSTM-based IDS achieves an accuracy
of 99.42%. The state-of-the-art HetIoT CNN-IDS [15] attained
an accuracy of 99.20%. Fig. 5a) shows that the proposed
CNN-based IDS performs well without being overfitted or
under-fitted. Fig. 5b) demonstrates that the proposed LSTM-
based IDS performance remains stable after 15 epochs. In
comparison to the proposed DL-based IDS, the performance

a) Proposed CNN-based IDS b) Proposed LSTM-based IDS c) HetIoT  CNN-IDS [17]
Fig. 5: Accuracy of Binary classification
a) Proposed CNN-based IDS b) Proposed LSTM-based IDS c) HetIoT  CNN-IDS [17]
Fig. 6: Accuracy of Grouped classification
of the state-of-the-art HetIoT CNN-IDS [15] falls short. The
graph in Fig. (5c) represents this.
B. Grouped classification
The details of the dataset for grouped classification have
been provided in Table??. The hyperparameter settings are
presented in the Table V. The details of the proposed CNN-
based IDS and the proposed LSTM-based IDS are provided in
Fig. 3 and Fig. 4, respectively. Fig. 6a), 6b), and 6c) illustrate
the training and validation accuracy of the proposed CNN-
based IDS, LSTM-based IDS, and state-of-the-art HetIoT
CNN-IDS [15] for grouped classification across 20 epochs.
The proposed CNN-based IDS obtains a testing accuracy of
99.34%, whereas the proposed LSTM-based IDS achieves an
accuracy of 99.13%. The state-of-the-art HetIoT CNN-IDS
[15] attained an accuracy of 99%. Fig. 6a) demonstrates that
the proposed CNN-based IDS performs well after 10 epochs
with minor variations. Fig. 6b) demonstrates that the proposed
LSTM-based IDS performance is better after 8 epochs. The
state-of-the-art HetIoT CNN-IDS [15] performance continues
to improve as the model is trained and validated with more
epochs, shown in Fig. 6c).
C. Multi-class classification
The details of the dataset for multi-class classification have
been provided in Table??. The hyperparameter settings are
presented in the Table V. The details of the proposed CNN-
based IDS and the proposed LSTM-based IDS are provided in
Fig. 3 and Fig. 4, respectively. Fig. 7a), 7b), and 7c) illustrate
the training and validation accuracy of the proposed CNN-
based IDS, LSTM-based IDS, and state-of-the-art HetIoT
CNN-IDS [15] for multi-class classification across 20 epochs.
The proposed CNN-based IDS obtains testing accuracy of
98.62% whereas the proposed LSTM-based IDS achieves
accuracy of 98.68%. The proposed CNN-based IDS perfor-
mance is shown in Fig. 7a). Compared to the proposed CNN-
a) Proposed CNN-based IDS b) Proposed LSTM-based IDS c) HetIoT  CNN-IDS [17]
Fig. 7: Accuracy of Multi-class classification.
TABLE VI: Comparative analysis with state-of-the art HetIoT
CNN-IDS [15] and the proposed DL-based IDS
DL-based intelligent IDS Accuracy(%)
Binary classi-
ficationGouped classi-
ficationMulticlass clas-
sification
Proposed CNN-based IDS 99.34 99.02 98.62
Proposed LSTM-based IDS 99.42 99.13 98.68
HetIoT CNN-IDS [15] 99.2 99 98.55
based IDS, the proposed LSTM-based IDS performed slightly
differently during training and validation, as shown in Fig.
7b). The state-of-the-art HetIoT CNN-IDS [15] performance
improves with each epoch, as shown in Fig. 7c). During
training validation, it increased from 80% to 97.5%. The state-
of-the-art HetIoT CNN-IDS [15] achieves testing accuracy of
98.55%.
D. Summary
The comparative analysis with the state-of-the-art HetIoT
CNN-IDS [15] and the proposed DL-based IDS, namely, the
proposed CNN-based IDS and the proposed LSTM-based
IDS, has been depicted in Table VI. The proposed DL-
based intelligent IDS are lightweight in terms of layers and
neurones used in the research. The data preprocessing steps,
including feature selection and hyperparameter settings, help
the research perform well. Compared to the proposed DL-
based intelligent IDS, the proposed LSTM-based IDS performs
better. The state-of-the-art HetIoT CNN-IDS [15] proposed
two separate models for binary and multi-class classifications.
However, the research proposed the same IDS for all kinds
of classifications, specifically, binary, grouped, and multi-
class classifications. Further, the proposed CNN-based IDS
performs better compared to the state-of-the-art HetIoT CNN-
IDS [15]. Among two DL-based intelligent IDS, the proposed
LSTM-based IDS performance is superior.
V. CONCLUSION
In conclusion, the research proposed two lightweight, DL-
based intelligent IDS models: the CNN-based IDS and the
LSTM-based IDS. The research proposed DL-based intelligent
IDS effectively identify and classify binary (2 class) classifi-
cation, grouped (8 class) classification, and multi-class (34
class) classification. The research also compares the proposed
DL-based intelligent IDS techniques with the state-of-the-art
HetIoT CNN-IDS. The researchers reimplemented the state-of-
the-art HetIoT CNN-IDS on the CICIoT2023 dataset for a fair

comparison. The proposed CNN-based IDS obtained accura-
cies of 99.34%, 99.02%, and 98.62% for binary, grouped, and
multi-class classification. The proposed LSTM-based IDS ob-
tained accuracies of 99.42%, 99.13%, and 98.68% for binary,
grouped, and multi-class classification. Future studies could
focus on federated learning approaches to ensure network
security and data privacy. The research could potentially be
extended to include reinforcement learning approaches for
large-scale IoT.
REFERENCES
[1] Cybersecurity (2024) How the Internet of Things (IoT) became a dark
web target. Available:https://www.weforum.org/stories/2024/05/internet-
of-things-dark-web-strategy-supply-value-chain/, [Online; accessed May
17, 2024].
[2] pawar DS (2024) The Rise of IoT Attacks Endpoint protection via Trend-
ing Technologies. Available:https://www.eccouncil.org/cybersecurity-
exchange/ethical-hacking/the-rise-of-iot-attacks-endpoint-protection-
via-trending-technologies/, [Online; accessed July 31, 2024]
[3] Cyberland (2024) Top-10 Cybersecurity Breaches in the United Arab
Emirates. Available:https://www.cyberlands.io/topsecuritybreachesuae,
[Online; accessed Nov 8, 2021].
[4] Towfighi J (2024) Roku accounts breached in cyberattack.
Available:https://edition.cnn.com/2024/04/12/business/roku-security-
breach-user-accounts/index.html, [Online; accessed April 12, 2024].
[5] Mahadik SS, Pawar PM, Muthalagu R, et al (2023) Intelligent LSTM
(iLSTM)-security model for HetIoT. Wireless Personal Communications
133(1):323–350.
[6] Abbas S, Al Hejaili A, Sampedro GA, et al (2023) A novel federated
edge learning approach for detecting cyberattacks in IoT infrastructures.
IEEE Access 11:112,189–112,198.
[7] Wang Z, Chen H, Yang S, et al (2023) A lightweight intrusion detection
method for IoT based on deep learning and dynamic quantization. PeerJ
Computer Science 9:e1569.
[8] Hamidouche M, Popko E, Ouni B (2023) Enhancing iot security via
automatic network traffic analysis: The transition from machine learning
to deep learning. In: Proceedings of the 13th International Conference
on the Internet of Things, pp 105–112
[9] Setitra MA, Fan M, Agbley BLY , et al (2023) Optimized MLP-CNN
model to enhance detecting DDoS attacks in SDN environment. Network
3(4):538–562
[10] Gunawan R, Ab Ghani H, Khamis N, et al (2023) Deep learning
approach to DDoS attack with imbalanced data at the application
layer. TELKOMNIKA (Telecommunication Computing Electronics and
Control) 21(5):1060–1067
[11] Xu H, Xian H (2023) SCD: A Detection System for DDoS Attacks
based on SAE-CNN Networks. Frontiers in Computing and Intelligent
Systems 5(3):94–99
[12] Doriguzzi-Corin R, Siracusa D (2024) FLAD: adaptive federated learn-
ing for DDoS attack detection. Computers & Security 137:103,597
[13] Mateus J, Zodi GAL, Bagula A, et al (2023) Building DDoS Resilient
SDNs Using Hybridised Deep Learning Methods. In: 2023 International
Conference on Emerging Trends in Networks and Computer Communi-
cations (ETNCC), IEEE, pp 1–7
[14] Nalayini C, Katiravan J, Sathya V (2023) Intrusion Detection in Cyber
Physical Systems Using Multichain. In: Malware Analysis and Intrusion
Detection in Cyber-Physical Systems. IGI Global, p 189–214
[15] Mahadik S, Pawar PM, Muthalagu R (2023) Efficient intelligent in-
trusion detection system for heterogeneous internet of things (HetIoT).
Journal of Network and Systems Management 31(1):2.
[16] Xie B, Wang Y , Wen G, et al (2023) Application-Layer DDoS
Attack Detection Using Explicit Duration Recurrent Network-Based
Application-Layer Protocol Communication Models. International Jour-
nal of Intelligent Systems 2023(1):2632,678.
[17] Vanlalruata H, Hussain J (2023) An efficient DDoS attack detection
mechanism in SDN environment. International Journal of Information
Technology 15(5):2623–2636.
[18] Ferhi W, Moussaoui D, Hadjila M, et al (2023) DDoS Attacks Detection
and Classification based on Deep Learning Model.[19] Aktar S, Nur AY (2023) Towards DDoS attack detection using deep
learning approach. Computers & Security 129:103,251.
[20] Han J, Pak W (2023) Hierarchical LSTM-based network intrusion de-
tection system using hybrid classification. Applied Sciences 13(5):3089.
[21] Talaei Khoei T, Kaabouch N (2023) A comparative analysis of super-
vised and unsupervised models for detecting attacks on the intrusion
detection systems. Information 14(2):103.
[22] Awajan A (2023) A novel deep learning-based intrusion detection system
for IOT networks. Computers 12(2):34.
[23] Cherian M, Varma SL (2023) Secure SDN–IoT framework for DDoS
attack detection using deep learning and counter based approach. Journal
of Network and Systems Management 31(3):54.
[24] Hnamte V , Hussain J (2023) DDoS detection using hybrid deep neural
network approaches. In: 2023 IEEE 8th International Conference for
Convergence in Technology (I2CT), IEEE, pp 1–8.
[25] Xu H, Sun L, Fan G, et al (2023) A hierarchical intrusion detection
model combining multiple deep learning models with attention mecha-
nism. IEEE Access.
[26] Javeed D, Gao T, Kumar P, et al (2023) An explainable and resilient
intrusion detection system for industry 5.0. IEEE Transactions on
Consumer Electronics 70(1):1342–1350.
[27] Ahmim A, Maazouzi F, Ahmim M, et al (2023) Distributed denial of
service attack detection for the Internet of Things using hybrid deep
learning model. IEEE Access 11:119,862–119,875.
[28] (2024) CIC IoT dataset 2023. Available:https://www.unb.ca/cic/datasets
/iotdataset-2023.html, [Online; accessed July 12, 2023]
[29] Neto ECP, Dadkhah S, Ferreira R, et al (2023) CICIoT2023: A real-
time dataset and benchmark for large-scale attacks in IoT environment.
Sensors 23(13):5941.
[30] Andrade I, Mahadik SS, Pawar PM, et al (2024) Intelligent Intrusion De-
tection Using ML for Large-Scale IoT Networks. In: 2024 Advances in
Science and Engineering Technology International Conferences (ASET),
IEEE, pp 1–7.
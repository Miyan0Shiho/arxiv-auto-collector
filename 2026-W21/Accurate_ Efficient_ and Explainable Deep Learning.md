# Accurate, Efficient, and Explainable Deep Learning Approaches for Environmental Science Problems

**Authors**: Jimeng Shi

**Published**: 2026-05-19 04:58:10

**PDF URL**: [https://arxiv.org/pdf/2605.19366v1](https://arxiv.org/pdf/2605.19366v1)

## Abstract
Environmental science plays a pivotal role in safeguarding ecosystems, a domain driven by large-scale, heterogeneous data. In the big data era, artificial intelligence (AI) has emerged as a transformative tool for learning patterns and supporting decision-making. This dissertation develops AI-based approaches tailored to complex environmental science problems to achieve Environmental Intelligence, studying three specific challenges. First, we focus on flood prediction and management in coastal river systems. Conventional physics-based models are computationally intensive, limiting real-time application. To overcome this, we propose a deep learning (DL)-based model, WaLeF, for water level forecasting, and a forecast-informed DL model, FIDLAr, to manage water levels. Evaluated in a flood-prone coastal system in South Florida characterized by extreme rainfall and sea level fluctuations, FIDLAr outperforms baselines in accuracy and efficiency while providing interpretable outputs. Second, we target global weather prediction, which is challenged by massive data scale. Traditional physics methods are deterministic and computationally heavy. We propose CoDiCast, a conditional diffusion model tailored for probabilistic weather forecasting. Adapted from generative AI for predictive tasks, experiments show CoDiCast achieves accurate, efficient forecasts with explicit uncertainty quantification. Lastly, we address scientific question-answering in environmental science. When answering in-domain questions, large language models (LLMs) often suffer from hallucinations due to out-of-date or limited knowledge. While retrieval-augmented generation (RAG) retrieves domain-specific knowledge, existing methods trade off accuracy, efficiency, or explainability. We propose Hypercube-RAG, built on a structured text cube framework, which successfully exhibits all three properties simultaneously.

## Full Text


<!-- PDF content starts -->

FLORIDA INTERNATIONAL UNIVERSITY
Miami, Florida
ACCURATE, EFFICIENT, AND EXPLAINABLE DEEP LEARNING APPROACHES
FOR ENVIRONMENTAL SCIENCE PROBLEMS
A dissertation submitted in partial fulfillment of the
requirements for the degree of
DOCTOR OF PHILOSOPHY
in
COMPUTER SCIENCE
by
Jimeng Shi
2025arXiv:2605.19366v1  [cs.LG]  19 May 2026

To: Dean In ´es Triay
College of Engineering and Computing
This dissertation, written by Jimeng Shi, and entitled Accurate, Efficient, and Explainable
Deep Learning Approaches for Environmental Science Problems, having been approved
in respect to style and intellectual content, is referred to you for judgment.
We have read this dissertation and recommend that it be approved.
Leonardo Bobadilla
Ananda Mondal
Arturo Leon
Shaowen Wang
Giri Narasimhan, Major Professor
Date of Defense: June 25, 2025
The dissertation of Jimeng Shi is approved.
Dean In ´es Triay
College of Engineering and Computing
Andr ´es G. Gil
Vice President for Research and Economic Development
and Dean of the University Graduate School
Florida International University, 2025
ii

© Copyright 2025 by Jimeng Shi
All rights reserved.
iii

DEDICATION
I am deeply grateful to my family for their unwavering love and support throughout
every stage of my life. Their belief in me has been a constant source of strength and
motivation.
iv

ACKNOWLEDGMENTS
The completion of this dissertation would not have been possible without the support
and encouragement of many individuals. I would like to express my sincere gratitude to
my advisor, Dr. Giri Narasimhan, for his patience, advice, mentorship, and support
throughout my Ph.D. journey. Dr. Giri has taught me how to become an excellent
researcher, ranging from idea brainstorming and improvement, paper writing and
polishing, research collaboration, to research dissemination. More notably, his guidance
extended beyond research discussions, offering valuable advice on both academic
matters and important life decisions. I am honored to be one of his advisees, and I hope
to one day become a knowledgeable and responsible researcher, just like him. I would
also like to thank other committee members, Dr. Leonardo Bobadilla, Dr. Ananda
Mondal, Dr. Arturo Leon, and Dr. Shaowen Wang, for their instrumental support and
suggestions. Appreciations should also go to brilliant collaborators, who greatly enrich
my research experience.. Their names are: Dr. Kalai Mathee, Dr. Jiawei Han, Dr. Vitalii
Stebliankin, Dr. Trevor Cickovski, Dr. Jayantha Obeysekera, Dr. Upmanu Lall, Dr. Zeda
Yin Dr. Zhaonan Wang, Dr. Dongsheng Luo, Dr. Yanzhao Wu, Dr. Sundararaman
Gopalakrishnan, Azam Shirali, Rahuul Rangaraj, Rukmangadh Myana, Arpit Mehta,
Saad Alqarni, Parshatd Govindasami, Nimmi Arunachalam, Ricky Ruiz, Anne Nguyen,
Bowen Jin, Wei Hu, Sizhe Zhou, Runchu Tian, Xu Zheng, Enshi Zhang, Zichuan Liu,
Mengjie Zhang, Lei Yan. Finally, I am grateful to the Institute for Geospatial
Understanding through an Integrative Discovery Environment (I-GUIDE), a large
research consortium funded by the National Science Foundation (NSF), which
consistently and generously supports my Ph.D. research work. THANK YOU ALL!
v

ABSTRACT OF THE DISSERTATION
ACCURATE, EFFICIENT, AND EXPLAINABLE DEEP LEARNING APPROACHES
FOR ENVIRONMENTAL SCIENCE PROBLEMS
by
Jimeng Shi
Florida International University, 2025
Miami, Florida
Professor Giri Narasimhan, Major Professor
Environmental science plays a pivotal role in safeguarding natural ecosystems and human
well-being. This domain encompasses a wide range of critical research challenges, such
as environmental prediction, monitoring, and management, often driven by large-scale,
heterogeneous data. In the era of big data, artificial intelligence (AI) has emerged as
a transformative tool, offering powerful capabilities for learning patterns from data and
supporting decision-making. This dissertation aims to leverage and develop AI-based
approaches tailored to address complex problems in environmental science with a goal of
Environmental Intelligence. Three specific problems are studied.
Firstly, we focus on the prediction and management of floods in coastal river systems.
Conventional physics-based models are often computationally intensive, limiting their
effectiveness in real-time applications. To overcome this limitation, we propose a deep
learning (DL)-based model,WaLeF, for water level forecasting, and a forecast-informed
DL-based model, FIDLAR, to manage water levels. Evaluated in a flood-prone coastal
system in South Florida, where extreme rainfall and sea level fluctuations are prevalent,
FIDLARdemonstrates superior performance over baseline methods in both accuracy and
computational efficiency, while providing interpretable model outputs.
Secondly, we focus on global weather prediction, a task challenged by massive data
scale and limited computational efficiency. Traditional physics-based methods are often
vi

deterministic and computationally intensive, limiting their scalability and applicability in
real-world applications. To address this, we propose CODICAST, a conditional diffusion
model tailored for probabilistic weather forecasting. Originally used in generative AI, dif-
fusion models are adapted here for predictive tasks. Experiments show that CODICAST
achieves accurate, efficient forecasts with explicit uncertainty quantification.
Lastly, we target the scientific question-answering (QA) in environmental science.
When answeringin-domainquestions, large language models (LLMs) often suffer from
hallucinations and factual inaccuracies due to out-of-date and limited domain knowledge.
To alleviate this phenomenon, retrieval-augmented generation (RAG) was designed to
retrieve domain-specific knowledge. However, existing RAG methods tend to perform
poorly in either accuracy, efficiency, or explainability. We proposeHypercube-RAG,
which is an RAG built on top of a structured knowledge representation framework (text
cube). Experiments verify that it exhibits three properties aforementioned simultaneously.
Overall, this dissertation shows how three distinct AI techniques can be effectively
applied to environmental science problems, highlighting the vital role of AI in this field.
vii

TABLE OF CONTENTS
CHAPTER PAGE
1. INTRODUCTION . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.1 Motivations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.1.1 Flood Prediction and Mitigation . . . . . . . . . . . . . . . . . . . . . . . 3
1.1.2 Weather Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.1.3 In-domain Environmental Question-Answering . . . . . . . . . . . . . . . 6
1.2 Research Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
1.3 Organization of the Dissertation . . . . . . . . . . . . . . . . . . . . . . . . 9
2. BACKGROUND AND REVIEW . . . . . . . . . . . . . . . . . . . . . . . . 10
2.1 Machine Learning Models . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.1.1 Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.1.2 Logistic Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.1.3 Support Vector Machines . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.1.4 Decision Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.1.5 K-Nearest Neighbors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.2 Deep Learning Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.2.1 Multilayer Perceptron . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
2.2.2 Recurrent Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . 16
2.2.3 Convoluntional Neural Networks . . . . . . . . . . . . . . . . . . . . . . . 17
2.2.4 Graph Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
2.2.5 Transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
2.2.6 Diffusion Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
2.2.7 Large Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
2.3 Flood Prediction and Mitigation . . . . . . . . . . . . . . . . . . . . . . . . 23
2.3.1 Physics-based Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . 23
2.3.2 Data-driven ML and DL Approaches . . . . . . . . . . . . . . . . . . . . . 25
2.4 Weather Forecasting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
2.4.1 Physics-based Numerical Weather Prediction . . . . . . . . . . . . . . . . 26
2.4.2 Machine Learning Weather Prediction . . . . . . . . . . . . . . . . . . . . 27
2.5 LLMs for Scientific Question-Answering . . . . . . . . . . . . . . . . . . . 28
3. FIDLAR: FORECAST-INFORMED DEEP LEARNING APPROACHES FOR
FLOOD PREDICTION AND MITIGATION IN COASTAL RIVER SYSTEMS 30
3.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
3.2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
3.3 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
3.3.1 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
3.3.2 Method Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
3.3.3 Flood Evaluator . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
3.3.4 Flood Manager . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
viii

3.3.5 Custom Loss Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
3.4 Architecture of the Neural Network Model . . . . . . . . . . . . . . . . . . . 38
3.5 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
3.5.1 Study Domain and Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . 40
3.5.2 Experimental Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
3.5.3 Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.6.1 Flood Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.6.2 Flood Mitigation with FIDLAR. . . . . . . . . . . . . . . . . . . . . . . 43
3.6.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
3.6.4 Analysis of Computational Time . . . . . . . . . . . . . . . . . . . . . . . 46
3.6.5 Explainability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
3.7 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
3.7.1 Model Explainability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
3.7.2 Spatio-temporal Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . 49
3.7.3 Generalization to Other River Systems . . . . . . . . . . . . . . . . . . . . 50
3.7.4 Deep Learning for Optimization . . . . . . . . . . . . . . . . . . . . . . . 50
3.8 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
3.8.1 Simple Regional Domain . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
3.8.2 Human Intervention . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
3.9 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
4. CODICAST: CONDITIONAL DIFFUSION MODEL FOR GLOBAL WEATHER
PREDICTION WITH UNCERTAINTY QUANTIFICATION . . . . . . . . . 53
4.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
4.2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
4.3 Preliminary Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
4.3.1 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
4.3.2 Denoising Diffusion Probabilistic Models . . . . . . . . . . . . . . . . . . 58
4.4 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
4.4.1 Method Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
4.4.2 Forward Diffusion Process . . . . . . . . . . . . . . . . . . . . . . . . . . 60
4.4.3 Reverse Conditional Denoising Process . . . . . . . . . . . . . . . . . . . 60
4.4.4 Pre-trained Encoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
4.4.5 Attention-based Denoiser Network . . . . . . . . . . . . . . . . . . . . . . 61
4.4.6 Training Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
4.4.7 Inference Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.4.8 Ensemble Forecast . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.5 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
4.5.1 Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
4.5.2 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
4.5.3 Experimental Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
4.5.4 Reproducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
ix

4.6 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
4.6.1 Quantitative Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
4.6.2 Qualitative Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
4.6.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
4.6.4 Parameter Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
4.7 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
4.8 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
5.Hypercube-RAG: HYPERCUBE RETRIEV AL-AUGMENTED GENERATION
FOR IN-DOMAIN SCIENTIFIC QUESTION-ANSWERING . . . . . . . . . 77
5.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
5.2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
5.2.1 Text-based RAG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
5.2.2 Structured RAG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
5.3 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
5.3.1 Hypercube Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
5.3.2 Hypercube Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
5.3.3 Hypercube Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
5.3.4 Hypercube Ranking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
5.4 Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
5.4.1 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
5.4.2 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
5.4.3 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89
5.5 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
5.5.1 Accuracy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
5.5.2 Efficiency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
5.5.3 Explainability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
5.5.4 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
5.5.5 Parameter Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 97
5.5.6 Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
5.6 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 99
5.6.1 Why doesHypercube-RAGoutperform other RAG methods? . . . . . . 99
5.6.2 Why did we use in-domain scientific QA datasets rather than general ones? 102
5.6.3 Future work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
5.7 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
6. CONCLUSIONS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
6.1 DL-based Flood Prediction and Management . . . . . . . . . . . . . . . . . 104
6.2 Diffusion Model for Probabilistic Weather Forecasting . . . . . . . . . . . . 105
6.3 Question-Answering Systems for Environmental Knowledge . . . . . . . . . 106
6.4 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
6.4.1 Scalability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
6.4.2 Acceleration of Diffusion Models . . . . . . . . . . . . . . . . . . . . . . 107
x

6.4.3 Flexibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
6.5 Endless Possibilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
BIBLIOGRAPHY . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
VITA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
xi

LIST OF TABLES
TABLE PAGE
3.1 Summary of the data set. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
3.2 Comparison of model performances for theFlood Evaluatoron the
test set, specifically at time t+1 for measurement station S1. The terms
“Over Timesteps” and “Under Timesteps” indicate the number of time
steps during which water levels exceed the upper threshold or fall below
the lower threshold, respectively. Similarly, “Over Area” and “Under
Area” pertain to the area between the water level curve and upper or
lower threshold, as was illustrated in Figure 3.4. Results in orange are
the lowest in that column while results in red are the closest to the ground
truth (in blue). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
3.3 Comparison of model performances for theFlood Manageron the test
set, specifically at time t+1 for measurement station S1. The∗denotes
that the GA method was used with a physics-based (HEC-RAS) evalua-
tor. The−denotes that the experiments were timed out. The†denotes
the GA method was used with the GTN as the evaluator. All other rows
are DL-based flood managers with a DL-based GTN as the evaluator.
Results inboldare the best in that column. . . . . . . . . . . . . . . . . 44
3.4 Ablation study for flood mitigation for the entire test set (for time point t+1
at S1). The last row indicates the performance of the FIDLARsystem
with GTN as proposed in Figure 3.5. The best results in the last row are
in bold. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
3.5 Running time for flood prediction and mitigation. The running time for the
rule-based method was not reported since historical data was directly
used, denoted with “/”. “×” represents that the methods are not applica-
ble for the tasks. GA∗, which combines a GA-based tool and HEC-RAS
for flood mitigation (test phase), took too long and was not reported, rep-
resented with “–”. GA†, which combines the GA-based tool with GTN
for flood mitigation (test phase), also took too long but was estimated
using a smaller sample. . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
4.1 Information on variables used in our experiments. . . . . . . . . . . . . . . 65
4.2 Latitude-weighted RMSE (↓) comparison. The results of NODE, ClimaX,
and ClimODE models are from the ClimODE paper [VHG24]. N/A rep-
resents the values that are not available. ForeCastNet was re-trained with
their code [PSH+22]. We employ Monte Carlo Dropout [GG16] during
the inference to compute the uncertainty. We mark the scores inboldif
CODICASTperforms the best among MLWP methods. . . . . . . . . . 69
4.3 ACC (↑) comparison. Other settings are the same as the Table 4.2. . . . . . 70
4.4 RMSE with various diffusion steps at 24 hours lead time. We mark the
lowest scores inboldfont. . . . . . . . . . . . . . . . . . . . . . . . . 71
xii

4.5 Comparison of Predictive, Generative, and Foundation Models forglobal
weather prediction. The performance scores below are at the lead time
of 6 days (except Prithvi WxC at the lead time of 5 days). These scores
are either from the WeatherBench scoreboard or the original paper. “∆x”
represents the horizontal resolution. . . . . . . . . . . . . . . . . . . . . 72
5.1 An example to show the hypercube ranking algorithm. . . . . . . . . . . . . 87
5.2 Datasets. Length denotes the number of words. . . . . . . . . . . . . . . . . 88
5.3 Performance Comparison (%) of LLMs themselves without RAG, LLMs
with various RAG baselines and Hypercube-RAG (ours). The best scores
are inboldwhile the second-best scores are highlighted with underline .
All RAG methods were experimented with GPT-4o as the base. . . . . . 91
5.4 Retrieval time (ms) vs. corpus size (k), experimenting on the Hurricane data
set (LLM base: GPT-4o). . . . . . . . . . . . . . . . . . . . . . . . . . . 93
5.5 Accuracy (BERTScore) vs corpus size. . . . . . . . . . . . . . . . . . . . . 93
5.6 Documents represented in the hypercube. . . . . . . . . . . . . . . . . . . . 94
5.7 Document contents. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
5.8 Ablation study on hypercube dimensions. The values are BERTScores (%). . 97
xiii

LIST OF FIGURES
FIGURE PAGE
2.1 Schematic representation of the MLP architecture withllayers, including
one input layer, one output layer, andl−2intermediate hidden layers. . 16
2.2 Schematic representation of the RNN architecture with its unfolded repre-
sentation on the right shown with the recurring layers. . . . . . . . . . . 17
2.3 Schematic representation of the convoluntion operation in CNNs [CTCO19]. 18
2.4 Schematic representation of the Attention mechanism [VSP+17]. . . . . . . 20
2.5 Schematic representation of the DDPM [HJA20]. . . . . . . . . . . . . . . . 21
2.6 Schematic representation of the output process of LLMs. . . . . . . . . . . . 22
2.7 Taxonomy of deep learning models for weather prediction across training
paradigms (dark yellow), model scopes (purple), and model architectures
(pink). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
3.1 Forecast-Informed Deep Learning Architecture (FIDLAR). Input data con-
sists of five categories of variables as shown in the left table. The vari-
ableswandkare the lengths of the past and prediction windows, re-
spectively. The parts colored green are provided as inputs, while the
orange and blue parts (with question marks) are outputs. TheFlood
ManagerandFlood Evaluatorrepresent deep learning (DL) mod-
els, the former to predict control schedules of controllable hydraulic
structures (e.g., gates and pumps) to pre-release water, and the latter
to predict the resulting water levels for those control schedules. Loss
functions,L 1andL 2, penalize thefloodingandwater wastagebeyond
pre-specified thresholds, respectively. . . . . . . . . . . . . . . . . . . . 35
3.2Flood Evaluator.The parts shaded green are used as inputs (i.e., historical
data and covariates predicted from the near future) and orange (control
schedule for the gates and pumps) Water levels (blue) are the outputs. . . 36
3.3Flood Manager.The parts shaded green (historical data) are the inputs, and
the parts shaded orange are the outputs. The water levels shaded blue are
not predicted. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
3.4 The two red bars represent a threshold of flooding and a threshold of wa-
ter wastage. Shown are (a) the time spans when these thresholds are
crossed, and (b) the areas between water level curves and threshold bars.
Violations of the upper and lower thresholds are captured inL 1andL 2. . 39
3.5 Graph Transformer Network forFlood Evaluator. Input variables in-
clude Rainfall, Pump, Gate, Tide, and Water levels as shown in Figure
3.6, which are generically denoted byR, T, G, P, andW. The output is
water levels. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
xiv

3.6 Schematic diagram of study domain. There are three water stations with
hydraulic structures (S26, S25B, S25A), one simple water station, S1
(green circle in the middle), and a station monitoring the tide level from
the ocean. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
3.7 Visualization of water levels with various methods for flood mitigation. We
zoomed int= 2∼4.5in gray. ‘A’–‘B’ in legend represents the
ManagerandEvaluator. Two dashed lines denote the upper (3.5
ft) and lower threshold (0.0 ft). . . . . . . . . . . . . . . . . . . . . . . . 45
3.8 Visualization for flood mitigation atalllocations. . . . . . . . . . . . . . . 45
3.9 Importance scores of tide input.xandyaxes are the tide and control sched-
ule of the gate over time. . . . . . . . . . . . . . . . . . . . . . . . . . . 48
4.1 Deterministic vs Probabilistic Models. . . . . . . . . . . . . . . . . . . . . . 58
4.2 Framework of our conditional diffusion model for global weather forecast
CODICAST. The superscriptTand the subscriptNdenote the time point
and iteration step of adding/denoising noise.HandWrepresent the
height (#latitude) and width (#longitude) of grid data.Cis the number
of variables of interest.Xis the observation data and ˜Zis the feature
representation in the embedding space. . . . . . . . . . . . . . . . . . . 60
4.3 Autoencoder structure. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
4.4 Attention-based denoiser structure. . . . . . . . . . . . . . . . . . . . . . . 62
4.5 Architecture of the U-Net model. . . . . . . . . . . . . . . . . . . . . . . . 63
4.6 Model Forecasts with confidence intervals. . . . . . . . . . . . . . . . . . . 71
4.7 Visualizations of true and predicted values of all five variables at 24 hours
lead time. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
4.8 Visualization of true and predicted values across five meteorological vari-
ables at 72 hours lead time. . . . . . . . . . . . . . . . . . . . . . . . . 73
4.9 Visualizations of true and predicted values of all five variables at 144 hours
lead time. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
4.10 Ablation study. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
4.11 Effect of linear and quadratic variance scheduling methods. . . . . . . . . . 75
5.1 Hypecube- vs semantic RAG: A case study . . . . . . . . . . . . . . . . . . 78
5.2 Graph vs. Hypercube. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
xv

5.3 The illustration ofHypercube-RAGframework. 1Input:Input the query
and the prompt into a LLM; 2Decomposition:LLM decomposes the
query into different dimensions; 3Retrieval:according to these di-
mensions, we useHypercube-RAGto retrieve relevant documents; 4
Augmentation:query is augmented with retrieved documents (ranked
already); 5Generation:LLM output withHypercube-RAG. . . . . 83
5.4 Hypercube construction on a corpus. We present only three dimensions for
clear representation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 83
5.5 Prompt template for LLM-as-a-judge. . . . . . . . . . . . . . . . . . . . . . 90
5.6 Performance comparison withvariousLLMs (only the range between 50%
and 90% shown). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
5.7 Ablation study on constituents ofHypercube-RAG. . . . . . . . . . . . . 96
5.8 Performance vs similarity threshold. . . . . . . . . . . . . . . . . . . . . . . 98
5.9 Comparison of three RAG methods on the same query. . . . . . . . . . . . . 99
5.10 Access one cube cell in one hypercube.✓represents the touched cube cells. 100
5.11 Access multiple cube cells in one hypercube.✓represents the touched cube
cells. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
5.12 Access multiple cube cells in multiple hypercubes.✓represents the touched
cube cells. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
xvi

CHAPTER 1
INTRODUCTION
Environmental science is essential for protecting natural ecosystems and ensuring human
well-being, particularly in the context of climate change and escalating environmental
risks [PAB+17]. This field involves a diverse set of critical challenges, including envi-
ronmental prediction, monitoring, and management, often characterized by large-scale,
heterogeneous data sources [AN24, SS19]. With the emergence of the big data era, ar-
tificial intelligence (AI) has been a transformative force, offering powerful tools for un-
covering complex patterns and enhancing data-driven decision-making [DED19]. It is
therefore vital to investigate how AI can be effectively leveraged to address pressing is-
sues in environmental science, a paradigm we refer to broadly asEnvironmental
Intelligence. This dissertation explores three representative in-domain problems
under this framework.
Floods Prediction and Management.Flooding is one of the most frequent and devas-
tating natural hazards worldwide, causing substantial damage to infrastructures, ecosys-
tems, and human lives [IJH+24, Gla21]. In the United States, floods have caused an av-
erage of nearly $8 billion in physical damages and approximately 100 fatalities annually
over the past 15 years [VH24] Timely flood prediction plays a vital role in disaster pre-
paredness, risk mitigation, and informed decision-making. The World Bank has estimated
that early warning systems can reduce flood-related fatalities by up to 43% and economic
costs by 35–50% [RT10] and save an average of 23,000 lives per year [RT10, NCD+24]
in developing countries. Extensive efforts have been made by using conventional physics-
based models for flood simulation, including tools such as SWMM [BGE18], HEC-RAS
[LG16], and MIKE [BCK+16]. Given infinite time and resources, floods can sometimes
be controlled by making smart use of hydraulic structures in river systems. For instance,
operating these structures to pre-release water before storms can significantly reduce the
1

risk of flooding. However, as shown in this dissertation, controlling those hydraulic struc-
tures optimally can be framed as an optimization problem. Slow simulation speeds for
the water level problems pose significant challenges for the much harder problem of flood
control, which often requires running thousands of water level simulations to identify
“optimal” control strategies [QMX+21]. However, these models often face limitations
related to computational efficiency and adaptability [SYM+23, JCT+23]. Therefore, the
critical need for fast simulation of real-time flood management is clear. The predictions
of water levels in river systems are often dictated by the accurate forecasts of weather and
rain, which leads us to the next problem tackled in this dissertation.
Global Weather Prediction.Accurate weather forecasting is crucial for a wide range
of societal activities, from daily planning to disaster preparedness [MKK+20, SYL+24].
Traditional physics-based numerical weather prediction (NWP) methods forecast weather
by modeling them as systems of differential equations that describe the complex interac-
tions between the variables describing phenomena in the atmosphere, land, and ocean
systems [PSGA+24, NBK+23]. However, they typically yield a single deterministic out-
come, failing to capture uncertainty. While such physics-based ensemble forecasts ef-
fectively model the weather uncertainty by generating multiple runs [Pal19, LHN+23],
ensemble-based NWP models face two major challenges: restrictive physical assumptions
and high computational cost [RP07]. Despite the notable efficiency of machine learn-
ing (ML)-based methods, such as Pangu [BXZ+23], GraphCast [LSGW+23], ClimaX
[NBK+23], ForeCastNet [PSH+22], most of them are deterministic [KYL+24], falling
short in capturing the uncertainty in weather forecasts [JK22]. Therefore, probabilistic
weather predictions are needed for making informed decisions.
Finally, we consider the problem of building accurate and efficient information re-
trieval systems for environmental data with which one can interact using natural language.
In-domain Scientific Question-Answering.We hypothesize that policy-makers and
2

domain scientists can make more informed decisions when equipped with up-to-date do-
main knowledge on critical issues such as hurricanes and aging dams [RVB22, WLL+25].
Scientific question-answering powered by large language models (LLMs) offers a valu-
able solution. However, existing LLMs often generate hallucinated or inaccurate re-
sponses due to outdated or insufficient domain-specific knowledge [HYM+25]. Retrieval-
augmented generation (RAG) techniques have emerged as promising approaches [HYM+25,
DZXH24]. Yet, current RAG methods often fall short in at least one of the three aspects
of accuracy, efficiency, and explainability. This highlights the need for a more robust and
domain-adapted RAG framework.
Overall, this dissertation aims to leverage and developaccurate,efficient, andinter-
pretableAI-based approaches tailored to address complex problems in environmental
science, leading us towardEnvironmental Intelligence.
1.1 Motivations
To achieveEnvironmental Intelligence, we are motivated to address challenges of tradi-
tional approaches in environmental science by leveraging and developing advanceddeep
learningmethods.
1.1.1 Flood Prediction and Mitigation
With regard to floods, the basic need is to determine the water level at any point of interest
(i.e., a control point), given all the water inflows into the system. This general basic
problem would solve theflood prediction problemsince we can determine the location,
intensity, and duration of a flood event (defined appropriately in terms of water levels) in
a given river system. The advanced problem is theinverseproblem . In other words, we
want to know the setting that would achieve a specified water level. Stated differently, the
3

advanced need is to determine the settings of the hydraulic structures (i.e., interventions)
needed to achieve a set of desired water levels at various control points in the river system.
Note that if the desired water levels are not possible, we want to be able to achieve as close
to the desired water level as possible. Again, solving the advanced problem can be thought
of as solving theflood mitigation problem, if the desired water levels are to not exceed
the stated flood thresholds at the control points. Thus, in anticipation of an impending
storm, we can solve the flood prediction problem to predict whether or not floods are
expected to occur. If the answer is yes, the flood mitigation problem is to determine the
precise intervention that would prevent, or at least mitigate the floods.
Traditional flood prediction/simulation approaches typically rely on physics-based hy-
drological and hydraulic models, which simulate water flow dynamics using equations
such as the Saint-Venant or shallow water equations [BD22]. While these models offer
outputs that are consistent with the laws of physics, they often require extensive calibra-
tion, high-resolution input data, and significant computational resources. These limita-
tions may hinder their operational scalability [CXL+23, VBMM22, JCT+23]. This also
poses significant challenges for flood mitigation, which often requires running thousands
of such simulations to identify “optimal” control strategies [QMX+21]. Furthermore,
rule-based methods [SGB+19] formulate control schedules based on insights gained from
historically observed data. However, the established rules represent the collective wisdom
gathered over decades of experience in managing specific river systems, exposing vulner-
abilities while dealing with extremely rare events and poor generalization for complex
river systems in other regions [SBX15]. Therefore, accurate, efficient, robust flood pre-
diction and mitigation approaches are essential. This dissertation offers accurate, efficient,
and explainable AI-based methods for these problems.
4

1.1.2 Weather Prediction
Weather Predictionrefers to predicting atmospheric variables of interest for future time
point(s), given observation(s) from a window from the recent past. Physics-based mod-
els, General Circulation Models (GCMs) [RRMA19], and Numerical Weather Prediction
(NWP) models [Coi11], have been the cornerstone of weather prediction. These models
simulate future weather scenarios by numerically approximating solutions to the differen-
tial equations that govern the complex physical dynamics of interconnected atmospheric,
terrestrial, and oceanic systems [NBK+23].
Despite significant advancements, these physics-based models face notable challenges.
Firstly, the accuracy of conventional NWP models is dictated by the variables used and
by the spatial and temporal resolutions used. Finer resolutions allow for better represen-
tation of mesoscale and localized phenomena (e.g., convection, topographic effects, or
coastal systems), but significantly increase the computational cost [AYCG10]. Secondly,
subgrid-scale parameterizations introduce significant uncertainty. Many atmospheric pro-
cesses occur at scales too small to be explicitly resolved and are instead approximated
using empirical or simplified physical models. These parameterizations are often region-
specific and may fail to generalize across varying conditions [PSH+05]. Lastly, a single
physics-based model typically produces deterministic forecasts once initial conditions are
fixed, falling short of capturing uncertainties in the predictions [BHQL24]. While ensem-
ble forecasts, to some extent, help mitigate this phenomenon by generating probabilistic
outputs [GR05], the first two challenges persist. Thus, accurate and efficient weather
prediction with uncertainty quantification is critically needed.
5

1.1.3 In-domain Environmental Question-Answering
While current large language models (LLMs) are pre-trained on massive general-purpose
datasets, they often underperform on in-domain questions that require specialized knowl-
edge. To address this limitation, incorporating external knowledge into LLMs is essen-
tial. Retrieval-augmented generation (RAG) has emerged as a promising solution, en-
abling LLMs to generate more accurate and contextually grounded responses by lever-
aging retrieved documents [HYM+25, DZXH24]. However, conventional RAG systems
based on semantic similarity often struggle to retrieve concise yet highly relevant infor-
mation for knowledge-intensive domains. The presence of irrelevant documents (i.e.,
retrieval “noise”) can mislead LLMs, leading to hallucinated or counterfactual responses
[KRI21, KAJ+24]. Furthermore, because these systems operate in a high-dimensional
embedding space, it becomes difficult to justify why certain documents were selected,
posing challenges for transparency and trust in high-stakes applications [JSR+19]. While
graph-basedRAG methods can retrieve documents by returning a subgraph, traversing
such subgraphs introduces a significant risk of information overload, as the expanding
neighborhoods may include substantial amounts of irrelevant themes [WHC+24]. Graph
traversal also poses significant scalability bottlenecks due to its inefficient searching pro-
cess. Therefore, an accurate, efficient, explainable RAG system is needed to address the
above limitations.
We tested the proposed QA systems on datasets related to hurricanes and aging dams,
as they are critical applications for informed decision-making and infrastructure planning
[SVS+25]. However, such knowledge is often buried in unstructured historical docu-
ments and reports. A question-answering (QA) framework with large language models
(LLMs) can achieve information extraction and organization, providing policy-makers
with timely, accessible, and context-specific insights to support risk assessment and proac-
tive mitigation strategies.
6

1.2 Research Contributions
The contributions of this dissertation are threefold. First, it presents deep learning–based
models for both flood prediction and mitigation. Second, it introduces a diffusion-based
framework for probabilistic weather forecasting, enabling robust estimation of uncertainty
in extreme weather events. Finally, the dissertation proposes a novel retrieval-augmented
generation (RAG) system designed to enhance the capability of large language models
(LLMs) in answering domain-specific questions with greater precision and relevance.
Overall, the major contributions are summarized below.
FIDLAR.First, an accurate and efficient novel DL model for water level forecasting
(WaLeF) is proposed, which replaces the role of physics-based models for flood pre-
diction/simulation given key driving factors such as rainfall, tidal conditions, and hy-
draulic control operations. Then a forecast-informed DL approach for flood mitigation
(FIDLAR) is introduced to seek the “optimal” controls on hydraulic structures (i.e., gates
and pumps). Experiments were conducted for data from a flood-prone coastal area in
South Florida, where floods are a result of multiple driving factors, such as rainfall, sea
levels, and inappropriate hydraulic operations. Results show that FIDLARis several
orders of magnitude faster than currently used physics-based approaches while outper-
forming baseline methods with improved water pre-release schedules. Meanwhile, this
framework presents the explainability of the proposed actions. Open-source code is avail-
able fromhttps://github.com/JimengShi/FIDLAR.
CODICAST.Current physics-based numerical weather prediction methods are compu-
tationally inefficient and cannot achieve uncertainty quantification. To address these chal-
lenges, we propose a conditional diffusion-based model, CODICAST, for global weather
prediction. It is conditioned on recent past observations while probabilistically modeling
7

forecast uncertainty. Although diffusion models are originally designed for generative
tasks starting from random noise, we tailor this framework in an innovative manner for
conditional predictive tasks, enabling accurate and uncertainty-aware weather forecast-
ing. In addition, the cross-attention mechanism was explored to effectively integrate the
conditioning into the denoising process to guide the generation tasks. Extensive exper-
iments were conducted on a decade of ERA5 reanalysis data from the European Cen-
tre for Medium-Range Weather Forecasts (ECMWF). Results demonstrate that COD-
ICASTachieves a useful trade-off between accuracy, efficiency, and uncertainty when
compared against several state-of-the-art models. Open-source code is available from
https://github.com/JimengShi/CoDiCast.
Hypercube-RAG.Conventional LLMs such as ChatGPT tend to hallucinate when
asked for answers to specific questions. RAG-based extensions can help alleviate the
problem of hallucinations, but are inefficient and error-prone. To overcome these limita-
tions, a multidimensional knowledge structure (text cube) was employed. Text cubes help
represent the information in unstructured documents in a structured manner by allocating
documents into cube cells along multiple pre-defined dimensions. Built on the cube struc-
ture, a simple yet accurate, efficient, and explainable RAG method,Hypercube-RAG,
was proposed. Experiments on scientific datasets, including hurricanes and aging dams,
demonstrate thatHypercube-RAGoutperforms other baseline methods in terms of ac-
curacy and efficiency when it comes toin-domainquestion-answering. More importantly,
we demonstrate the explainability, scalability, and noise resilience of our method. It pro-
vides meaningful insights during the retrieval process and performs efficiently and ro-
bustly on retrieval tasks from large-scale and noisy corpora. Open-source code is avail-
able fromhttps://github.com/JimengShi/Hypercube-RAG.
8

1.3 Organization of the Dissertation
We conclude this introductory chapter with a roadmap of the dissertation.
Chapter 2 covers the foundational concepts relevant to this dissertation, including a
literature review of ML/DL models and their applications in flood prediction, mitigation,
and weather forecasting. It also surveys current retrieval-augmented generation (RAG)
methods forin-domainquestion-answering tasks.
Chapter 3 introduces FIDLAR, a forecast-informed DL approach for flood prediction
and mitigation. After reviewing the existing approaches and their key limitations, a DL-
based framework consisting of two cascaded models,Flood ManagerandFlood
Evaluator, is described. Then, detailed training of FIDLARis discussed, followed by
experimental results and analysis regarding accuracy, efficiency, and explainability.
Chapter 4 presents CODICAST, a conditional diffusion model for weather forecasting
with uncertainty quantification. After reviewing the existing approaches and their key
limitations, the methodology of the conditional diffusion model is described. Then, de-
tailed training is introduced, followed by experimental results and analysis in terms of
accuracy, efficiency, and uncertainty.
Chapter 5 starts with the limitations of the existing RAG methods forin-domain
question-answering tasks. Then it reviews the related work on document indexing with a
multidimensional knowledge structure (text cube). Then hypercube-based RAG method,
Hypercube-RAG, is proposed, ranging from hypercube construction and its usage for
retrieval. Lastly, experimental results and analysis, and case studies on accuracy, effi-
ciency, and explainability are discussed in detail.
Finally, Chapter 6 wraps up the dissertation by providing a summary, conclusions,
and recommendations for future research.
9

CHAPTER 2
BACKGROUND AND REVIEW
Machine learning (ML) and deep learning (DL) fall within the broader area of artificial
intelligence (AI), focusing on developing algorithms capable of learning patterns from
data and making informed decisions or predictions [Sar21b]. Most of these methods op-
erate in two primary phases, under the assumption that all data samples are independently
and identically distributed (i.i.d.) [JZH21]. In the training phase, a model is trained on
a dataset to learn the knowledge and patterns. In the test (inference) phase, the trained
model created in the training phase is used to test its learning or to make a decision on
new, previously unseen inputs [Sar21a].
Learning models are broadly categorized based on the nature of the training/learning
methods, that is, how the learning signal or feedback (e.g., labels) is provided to the sys-
tem for each input sample [JGR20, STS16]. The primary learning paradigms include:
(1)Supervised learning, the model is trained on a labeled dataset, where each input is
paired with a corresponding ground-truth output. This paradigm is widely used in classi-
fication and regression tasks. (2)Unsupervised learning, the model is provided only with
input data without explicit labels, and must discover underlying patterns or structures.
Typical tasks include clustering, dimensionality reduction, and anomaly detection. (3)
Semi-supervised learninglies between supervised and unsupervised learning. It utilizes
a small amount of labeled data along with a large amount of unlabeled data to improve
learning performance. (4)Self-supervised learningis an emerging paradigm where the
model generates supervisory signals from the input data itself, often by learning to pre-
dict masked parts of the input as part of its training. The resulting model is often referred
to as a pre-trained model and may be further prompted or fine-tuned for performing spe-
cialized tasks. In this chapter, we mainly focus on the regression tasks due to their high
relevance in dealing with environmental questions.
10

2.1 Machine Learning Models
In statistics and ML,regression analysisis the task of modeling the relationship between
a dependent variable and a set of independent variables, making it possible to infer or pre-
dict the value of the dependent variable based on the values of the independent variables.
2.1.1 Linear Regression
Let us assume a dataset consisting ofmsamples from ann-dimensional feature space.
Each samplexi= (xi
1, xi
2, . . . , xi
n)∈Rnis associated with a scalar outputyi∈R. The
goal is to learn a weight vectorw= (w 1, w2, . . . , w n)∈Rnsuch that every predicted
output is given as a linear combination of its input features:
y=w 1·x1+w 2·x2+···+w n·xn=w·xT,(2.1)
wherexdenotes an input vector andw·xTrepresents the inner product of the weights
and the input features.
2.1.2 Logistic Regression
Logistic regression is a widely used statistical model for binary classification problems,
where the target variable takes on one of two possible outcomes (e.g., 0 or 1). The model
estimates the probability that a given input vector belongs to the positive class by apply-
ing the logistic sigmoid function to a linear combination of the input features, as shown
in Eq. (2.2). This probability is then thresholded to output the final label. Despite its
name, logistic regression performs classification rather than regression, as it models the
likelihood of class membership rather than predicting a continuous output.
σ(y) =1
1 +e−y.(2.2)
11

2.1.3 Support Vector Machines
Support Vector Machines (SVMs) are powerful supervised learning models used primar-
ily for classification tasks. The core idea of an SVM is to find an optimal hyperplane that
maximally separates data points of different classes in a high-dimensional feature space.
For linearly separable data, the SVM aims to find the hyperplane with the largest margin
between the two classes.
Given a dataset ofmpoints{(xi, yi)}m
i=1, wherexi∈Rnin an-dimensional space
and class labelsyi∈ {−1,1}, the problem is formulated to find a hard-margin SVM (i.e.,
hyperplane) to separate these points:
min
w,b1
2∥w∥2(2.3)
subject toyi(w⊤xi+b)≥1,∀i= 1, . . . , m,
wherewis the weight vector orthogonal to the hyperplane, andbis the bias term. The
objective is to minimize the norm ofw, which corresponds to maximizing the margin
between the classes.
In practice, data is often not linearly separable, so the soft-margin SVM introduces
slack variablesξito allow for some misclassifications:
min
w,b,ξ1
2∥w∥2+CnX
i=1ξi(2.4)
subject toyi(w⊤xi+b)≥1−ξi, ξi≥0,∀i,
whereC >0is a regularization parameter that balances the trade-off between maximizing
the margin and minimizing classification errors.
To handle non-linear classification problems, SVMs utilize kernel functionsK(xi,xj)
to implicitly map the input features into a higher-dimensional space after which the sep-
arating hyperplane is sought. Common kernels include:
12

•Linear kernel[GDS19]:K(xi,xj) = (xi)⊤xj
•Polynomial kernel[VM19]:K(xi,xj) = ((xi)⊤xj+c)d
•Radial basis function (RBF) kernel[DLYC21]:K(xi,xj) = exp(−γ∥xi−xj∥2)
SVMs are valued for their robustness and strong generalization capabilities, especially
in high-dimensional nonlinear spaces [ERKL16].
2.1.4 Decision Trees
Decision Trees are a widely used class of supervised learning algorithms suitable for both
classification and regression tasks [Qui86]. The model represents decisions and their
possible consequences in a tree-like structure, where each internal node corresponds to
a feature-based test, each branch represents an outcome of the test, and each leaf node
denotes a predicted label or value. The learning process involves recursively partitioning
the input space based on feature values to maximize the separation between classes (in
classification) or minimize variance (in regression). Popular criteria for selecting splits
include Gini impurity and information gain (based on entropy) [Tan20].
Formally, given a datasetDwith input features, the goal at each node is to select the
feature that maximizes the information gain:
Information Gain=I(D)−X
j∈{L,R}|Dj|
|D|I(D j),(2.5)
whereIis an impurity function (e.g., entropy), andD LandD Rare the left and right
subsets resulting from the split.
Decision Trees are interpretable and easy to visualize, making them useful in appli-
cations where model transparency is important. However, they are prone to overfitting,
especially when deep trees are grown. This limitation is often addressed by pruning
13

strategies [MSO12, EMSK97] or ensemble methods such as Random Forests [Bre01] and
Gradient Boosted Trees [SK21].
2.1.5 K-Nearest Neighbors
K-Nearest Neighbors (KNN) [Pet09, KK13] is a simple yet effective instance-based learn-
ing algorithm used for both classification and regression tasks. In classification, it assigns
a class label to a query point by identifying theknearest neighbors in the feature space
and selecting the majority class among them. KNN is anon-parametricmethod, meaning
it does not make explicit assumptions about the underlying data distribution and instead
relies entirely on the training data for predictions.
However, the performance of KNN is highly dependent on the choice of thedistance
metric(e.g., Euclidean, Manhattan, cosine) and the value ofk. A smallkmay lead
to overfitting and sensitivity to noise, while a largekcan smooth out decision bound-
aries, potentially resulting in underfitting. Furthermore, because KNN requires comput-
ing the distance between the query and all training points at inference time, it can be
computationally expensive for large datasets. Despite these limitations, KNN remains
popular due to its simplicity, interpretability, and strong performance in well-structured,
low-dimensional data spaces [Zha16].
2.2 Deep Learning Models
Deep learning (DL) models typically refer to neural network architectures composed of
multiple layers, where each layer consists of interconnected computational units called
neurons. These models are designed to automatically learn hierarchical representations of
data through successive transformations. Unlike traditional machine learning algorithms
14

that often rely on manually engineered features, DL models can capture complex and
nonlinear relationships directly from raw input data, such as images, text, or time series.
2.2.1 Multilayer Perceptron
Multilayer Perceptron (MLP) is part of a family of fully connected feedforward neural
networks. MLP usually consists of at least three layers: one input layer, one or more
intermediate hidden layers, and one output layer. By stacking multiple layers of intercon-
nected neurons (Figure 2.1, circles), where each layer applies a non-linear activation func-
tion (e.g., sigmoid function [HM95]), MLP can learn complex, non-linear mappings from
inputs to outputs [DCNV+21]. The composition of these non-linear functions through the
network allows it to approximate highly intricate functions. Mathematically, the compu-
tation of Figure 2.1 can be described as follows:
h1=σ(W 1·x+b 1);
h2=σ(W 2·h1+b 2);h 3=σ(W 3·h2+b 3);. . .;h l=σ(W l·hl−1+b l);
y=f(h l)
wherex∈RDis the D-dimensional input vector,h i∈Rn(i)represents the output of
the neurons of theithlayer (1≤i≤l),W j∈Rn(j)×n(j−1)is the weight matrix from
thej−1thlayer tojthlayer,b j∈Rn(j)is the bias vector in thejthlayer,n(j)is the
number of neurons of thejthlayer, andσstands for the sigmoid function. Alternatively,
other activation functions are the hyperbolic tangent [ZM13] or the rectified linear unit
activation function [PV18]. Finally,ydenotes the output with the functionf, whose range
depends on whether the network is designed for a regression or classification task.
However, since MLPs are fully connected neural networks, they are prone to overfit
the training data, leading to poor generalization on new, unseen data [ZF23]. Regulariza-
15

Figure 2.1: Schematic representation of the MLP architecture withllayers, including one
input layer, one output layer, andl−2intermediate hidden layers.
tion and Dropout techniques are generally used to mitigate overfitting by simplifying the
model structure.
2.2.2 Recurrent Neural Networks
Unlike traditional feedforward neural networks, recurrent neural networks (RNNs) have
connections that form directed cycles, allowing them to maintain a “memory” of previous
inputs. This is particularly useful for tasks where the context from previous inputs is
important, such as time series, text, speech, or video. The temporal dependencies can be
learned by recurrently training and updating the transitions of an internal (hidden) state
from the last timestep to the current timestep. The computational process of each hidden
state (unit or cell) in an RNN is described as follows:
St= tanh(W xs·(xt⊕S t−1) +b s),
yt=σ(W y·St+b y),
wherex t∈Rmis the input vector ofminput features at timet;W xs∈Rn×(m+n)
andW y∈Rn×nare parameter matrices;nis the number of neurons in the RNN layer;
bs∈Rnandb y∈Rnare bias vectors for the internal state and output, respectively;σ
16

is the sigmoid activation function;S tis the internal (hidden) state; andx t⊕S t−1denotes
the concatenation of vectorsx tandS t−1.
Figure 2.2: Schematic representation of the RNN architecture with its unfolded represen-
tation on the right shown with the recurring layers.
Nevertheless, RNNs have the following limitations. Since RNNs learn knowledge
iteratively over time steps, training RNNs can be challenging due to the vanishing or ex-
ploding gradient problem, especially for long sequences [KS21]. Specifically, the gradi-
ents at the last time steps are difficult to backpropagate to the time steps at the beginning.
This phenomenon is also calledmemory forgetting, caused by the inherent sequential
modeling, which is characteristic of RNNs.
2.2.3 Convoluntional Neural Networks
Convolutional neural networks (CNNs), originally developed for image and video pro-
cessing, have also demonstrated strong performance in time series forecasting. When
applied to sequential data, CNNs capture temporal patterns and dependencies by treating
the time series as a one-dimensional sequence. Convolutional layers use learnable fil-
ters (kernels) that slide over the input to detect localized patterns [BBO17]. These layers
are typically followed by max-pooling operations, which retain the most salient features
while reducing dimensionality. For multivariate time series, the input can be structured as
17

a two-dimensional array (time steps×features), enabling CNNs to learn both temporal
dependencies and cross-variable interactions [KS16].
Figure 2.3: Schematic representation of the convoluntion operation in CNNs [CTCO19].
CNNs offer several advantages for time series forecasting. Their strength lies in cap-
turing local patterns through filters with limited receptive fields, allowing the model to
efficiently detect short-term dependencies and localized features. A key benefit is that
these filters are applied uniformly across the sequence, making CNNs invariant to input
length and well-suited for handling long sequences without degradation in performance
[AZH+21]. Additionally, CNNs do not rely on the propagation of hidden states, unlike
RNNs, which can help maintain stable prediction quality across varying sequence lengths.
However, this local focus also presents a limitation. Because CNNs primarily operate
within fixed-size receptive fields, they may struggle to model long-range or sequential
dependencies as effectively as recurrent architectures. This can be particularly restrictive
in time series tasks where the temporal order and long-term interactions between time
steps are critical for accurate forecasting.
18

2.2.4 Graph Neural Networks
Graph Neural Networks (GNNs) [SGT+08] are designed to work on graph-structured
data,G= (V,E), consisting of a set of nodesVand a set of edgesEconnecting them. The
nodes and edges represent the entities and their relationships in a graph (the graph is undi-
rected if not specified). Spatio-temporal Graph Neural Networks (ST-GNNs) [YYZ17]
extend the architecture of GNNs to model both spatial and temporal dependencies in dy-
namic graph-structured data changing over time,G t= (V,E, t). Here, nodesVrefer to
spatial locations, while edgesErefer to spatial relationships between the locations. Each
nodevi
trepresents the feature vector at the corresponding locationiand timet. For each
node, the message-passing technique [GSR+17] is often employed to capture the spatial
dependencies on its neighbors. The temporal dependencies between graph snapshots can
be modeled with the sequential models aforementioned. For the message passing, hidden
stateshi
tat each node are updated based on messages (feature vectors)vi
t+1according to:
vi
t+1=X
j∈N(i)Mt(hi
t, hj
t, eij),
hi
t+1=σ(hi
t, vi
t+1),(2.6)
where in the sum,N(i)denotes the neighbors ofithnode in graphG. After iterative
updatesktime steps, the final output of the whole graph at timet+kcan be computed
with a readout functionO:
yt+k=O({hi
t+k|i∈ G}).(2.7)
2.2.5 Transformers
To overcome the limitations of RNNs, which stem from their inherent sequential process-
ing, the Transformer model [VSP+17] has emerged as a powerful alternative. Using an
19

encoder-decoder architecture, its core innovation lies in the use of theattentionmecha-
nism, enabling it to capture dependencies between any parts of a sequence without the
need for sequential steps [WZZ+22]. Theattentionmechanism is described as follows:
Attention(Q,K,V) =softmaxQKT
√dk
V,(2.8)
where thed kdenotes the dimension of the key,Q∈Rn×dk,K∈Rm×d k, andV∈Rm×d v
are the query matrix, key matrix, and value matrix, respectively. These three matrices
are computed by linear transformations from the original input sequenceX∈Rn×dwith
learnable weight matricesW q∈Rd×dk,W k∈Rd×dk,W v∈Rd×dv, as
Q=XW q,K=XW k,V=XW v.(2.9)
Figure 2.4: Schematic representation of the Attention mechanism [VSP+17].
2.2.6 Diffusion Models
Denoising Diffusion Probabilistic Model (DDPM) [HJA20, SME20] is a generative model
that has gained significant popularity in computer vision [SCC+22, CHIS23], natural lan-
guage processing [HMT+22, LZZW23], due to its ability to produce high-quality, realistic
20

synthetic samples with similarity to the input. Diffusion models work in two processes:
forward diffusion processandreverse denoising process. In the forward process, data
(e.g., an image) is gradually “noised” by adding small amounts of Gaussian noise over
multiple steps until it becomes nearly pure noise. This process is parameterized, where
each step incrementally adds a fixed amount of noise. A denoising model learns to de-
noise using pairs of original data (e.g., image before adding an increment of noise) and
their diffused versions (i.e., image after adding the increment of noise), which are gener-
ated during the forward diffusion process. The reverse process uses the denoising model
to recover a realistic sample from a noisy starting point.
Mathematically, theforward processtransforms an inputx 0with a data distribution
ofq(x 0)to a white Gaussian noise vectorx NinNdiffusion steps. It can be described as
a Markov chain that gradually adds Gaussian noise to the input according to a variance
schedule{β 1, . . . , β N} ∈(0,1):
q(x1:N|x0) =NY
n=1q(xn|xn−1),(2.10)
where at each stepn∈[1, N], the diffused samplex nis obtained withq(x n|xn−1) =
N 
xn;√1−β nxn−1, βnI
.
In thereverse process, the denoising model,p θ(·), is used to recoverx 0by gradually
denoisingx nstarting from a Gaussian noisex Nsampled fromN(0,I). This process is
presented as:
pθ(x0:N) =p(x N)NY
n=1pθ(xn−1|xn).(2.11)
Figure 2.5: Schematic representation of the DDPM [HJA20].
21

2.2.7 Large Language Models
Large language models (LLMs) are a class of deep learning models pre-trained on mas-
sive text corpora to perform a wide range of natural language processing (NLP) tasks.
Built primarily using the Transformer architecture [VSP+17] and self-supervised learning
methods, LLMs learn contextual representations by modeling dependencies between to-
kens across long sequences. During the inference phase, they autoregressively output the
next tokenˆw n+1with the highest probability based on the preceding onesw 1, w2, . . . , w n,
as described below:
ˆwn+1= arg max
wn+1P(w n+1|w1, w2, . . . , w n).(2.12)
Figure 2.6: Schematic representation of the output process of LLMs.
As mentioned above, LLMs are typically pre-trained on large-scale unlabeled data
using self-supervised objectives and can be fine-tuned or prompted for downstream ap-
plications. Their generalization ability, scalability, and emergent reasoning capabilities
have made them foundational tools across both academic research and real-world deploy-
ments. Examples include BERT [DCLT19a], GPT [BMR+20], ChatGPT [AAA+23], and
DeepSeek [GYZ+25, LFX+24], which have demonstrated state-of-the-art performance in
tasks such as question answering [ZYW+23], text summarization [ZJM+24], translation
[EH24], and much more.
22

2.3 Flood Prediction and Mitigation
Given that this dissertation deals with applying ML techniques to environmental prob-
lems, we review the areas of environmental Science relevant to us. We review water
resource management in this section and weather prediction in Section 2.4.
Flood prediction strategies are needed to determine the location, intensity, and dura-
tion of a flood event in a river system, given the possible driving factors, such as rainfall,
sea level, and data on controllable hydraulic structures such as gates, pumps, dams, and
reservoirs in the river system. Flood management refers to determining the appropriate
interventions needed to prevent or mitigate a flood that may be predicted in anticipation
of an impending storm, which is often achieved by applying the “optimal” control of the
hydraulic structures [LTQC20, LBT21].
2.3.1 Physics-based Approaches
Conventional flood prediction methods over the last decade or more have been achieved
by starting with a detailed physical model of the entire watershed and applying phys-
ical laws of conservation of water from first principles. HEC-RAS 2-D models have
assessed urban flood risks [RUB19, SPW+23], while a coupled 1D–2D HEC-RAS model
was used for both channel and overbank flow representation [DKA19]. [MSL+23] em-
ployed the HEC-RAS 1D model for floods caused by dam breaks [MSL+23]. Others
[AEk+24, SBB+20] have studied floodplain modeling using HEC-RAS and MIKE 21
tools, and analyzed their suitability in hydraulic and hydrological models. Khatooni et al.
[KHMN25] assessed urban flood risks by coupling SWMM and HEC-RAS-2D models.
Despite these advancements, physics-based models offer relatively accurate performance
but with the following notable limitations. (1) Computationally Intensive: For large wa-
tersheds, these models are computationally demanding [CXL+23, VBMM22] because
23

they use detailed grid representations and often need to solve a large number of complex
partial differential equations (PDEs) [JCT+23]. (2) Lack of Flexibility: They lack flex-
ibility and require rerunning the entire model for each new simulation [SAG+23], even
for relatively small changes in initial or boundary conditions [GAC+22]. This prevents
the exploration of a large number of hypothetical scenarios, often employed in real-life
scenarios with massive economic consequences for the model predictions. (3) Accuracy
Challenges: Physics-based models may compromise accuracy by using simplifications
and approximations that reduce model complexity [GAC+22]. Such changes may limit
the ability of the models to comprehensively cover the entire spectrum of process vari-
ability [GKBG23]. Higher accuracies can be achieved by using finer grids, which in turn
makes the models more computationally intensive. (4) Explainability Issues: Despite elu-
cidating complex dynamics, they struggle to provide detailed explanations, particularly in
identifying specific variables and time steps influencing simulation outcomes.
Once floods are predicted, flood mitigation strategies are needed. Water manage-
ment agencies have built controllable hydraulic structures such as dams, gates, pumps,
and reservoirs in river systems [KGL+16]. On one hand, rule-based methods [SGB+20,
SDP19, MIHA18] formulate control schedules based on insights gained from historically
observed data. However, these schedules are vulnerable to rare extreme events and sen-
sitive to watersheds. On the other hand, researchers have tackled this challenge as an
optimization problem [Kar16]. By optimizing variables such as water discharges, the al-
gorithm generates control schedules for hydraulic structures to effectively mitigate flood
risks [DHM+20, ZBHB+21]. Random initialization followed by soft-computing tech-
niques such as genetic algorithms has been employed to perform the optimization. Sub-
sequently, physics-based models such as HEC-RAS and SWMM are used to assess the
generated control schedules [LTQC20, SGB+19, VMWW18, CLGH16, LKVS14]. How-
ever, such physics-based simulators are prohibitively slow since they require thousands
24

of simulations of water level predictions. Therefore, it is important to develop accurate
and efficient approaches for flood prediction and mitigation.
2.3.2 Data-driven ML and DL Approaches
With the rise of big data, data-driven approaches have become increasingly promising for
flood modeling and prediction. A variety of machine learning (ML) and deep learning
(DL) models have been explored in this domain. For example, in South Florida, artifi-
cial neural networks (ANNs) have been used to forecast river flow in the Apalachicola
River [HXCH04] and net inflow volumes into Lake Okeechobee [OTC+00]. Sadler et
al.[SGMS18] applied a Random Forest model to predict flood severity in Norfolk, Vir-
ginia. Similarly, Sapitang et al.[SMRFK+20] evaluated four ML models, Boosted Deci-
sion Tree Regression, Decision Forest Regression, Bayesian Linear Regression, and Neu-
ral Network Regression, to improve daily forecasts of reservoir water levels, highlighting
intra-model comparisons. Ayus et al. [ANG23] employed Random Forest, XGBoost,
and bidirectional LSTM models on three decades of data to predict daily water levels.
More recently, Kow et al.[KLY+24] proposed a hybrid Transformer–LSTM model for
accurate multi-step-ahead forecasting of flood storage pond and river water levels. Yin et
al.[YSH+24] leveraged U-Net and generative adversarial networks (GANs) to reconstruct
high-resolution flood maps from coarse-resolution numerical simulations. Additionally,
there is growing interest in physics-informed “hybrid” models that integrate neural net-
works with physical constraints such as the shallow water equations [YBH+23, YSB+25].
25

2.4 Weather Forecasting
2.4.1 Physics-based Numerical Weather Prediction
Physics-based models, including General Circulation Models (GCMs) [RRMA19] and
Numerical Weather Prediction (NWP) models [Coi11], have been the cornerstone of
weather prediction. These models simulate future weather scenarios by numerically ap-
proximating solutions to the set of differential equations that govern the complex physi-
cal dynamics of interconnected atmospheric, terrestrial, and oceanic systems [NBK+23].
As in the case of water resource management, these physics-based models face notable
challenges. First, the accuracy of conventional NWP models is highly dependent on
the spatial and temporal resolution. Finer resolutions allow for better representation of
mesoscale and localized phenomena (e.g., convection, topographic effects, or coastal sys-
tems). However, achieving higher resolution significantly increases the computational
cost [AYCG10]. Second, subgrid-scale parameterizations introduce significant uncer-
tainty. Many atmospheric processes occur at scales too small to be explicitly resolved and
are instead approximated using empirical or simplified physical models. These parame-
terizations are often region-specific and may fail to generalize across varying conditions
[PSH+05]. Moreover, the governing equations rely on simplified atmospheric dynamics
assumptions, limiting their ability to capture rare or complex phenomena. Finally, a sin-
gle physics-based model typically produces deterministic forecasts once initial conditions
are fixed, falling short of capturing uncertainties in weather evolution, even though pertur-
bations of initial conditions have been tried [BHQL24]. Ensemble-based NWP forecasts
help address this last issue by generating probabilistic outputs [GR05], but the first two
challenges persist.
26

AI Models
for Weather
PredictionDeterministic Predictive
LearningGeneral-Purpose
Large ModelsTransformerFourCastNet [PSH+22], FuXi [CZZ+23],
FengWu [CHG+23], FengWu-4DVar [XBX+23],
SwinVRNN [HCWL23], SwinRDM [CDH+23],
Pangu-Weather [BXZ+23], Stormer [NSB+23],
HEAL-ViT [Ram24], TianXing [YWMZ25],
ArchesWeather [CSC+24], AtmoRep [LLG+23]
GNNGraphCast [LSGW+22], GnnWeather [Kei22],
AIFS [LAC+24], GraphDOP [ABL+24]
PhysicsAIClimODE [VHG24], WeatherODE [LZSJ24],
NeuralGCM [KYL+24], Conformer [SSP24]
Domain-Specific
ModelsTransformerSwinUnet [BAML21], Earthformer [GSW+22],
Rainformer [BSZ+22], U-STN [CMH+22],
OMG-HD [ZBN+24], PFformer [XQS+24]
GNNHiSTGNN [MXT+23],w-GNN [CWHT24],
WeatherGNN [WCW+24], MPNNs [YGC+24]
RNN&CNNMetNet [SEH+20, EAS+22],
MetNet-3 [AEL+23], PredRNN [WWZ+22],
MM-RNN [MZL23], ConvLSTM [SCW+15]
MambaMetMamba [QCJ+24], MambaDS [LCB+24],
VMRNN [TDT+24], WPMamba [LLD+24]
PhysicsAINowcastNet [ZLC+23], PhysDL [DBPG19],
PhyDNet [GT20], DeepPhysiNet [LLC+24]
Probabilistic Generative
LearningGeneral-Purpose
Large ModelsDiffusionGenCast [PSGA+24], CoDiCast [SJHN24],
SEEDs [LCLG+23], ContinuousEnsCast [ALOL24],
ArchesWeatherGen [CSC+24]
Domain-Specific
ModelsDiffusionLDMRain [LHN+23], PreDiff [GSH+24],
CasCast [GBY+24], SRNDiff [LLQ+24],
DiffCast [YLY+24], GEDRain [AMP+23]
GANsGANrain [RLW+21], MultiScaleGAN [LLY+22],
STGM [WFL23], PCT-CycleGAN [CKK+23]
Pretraining & Finetuning Foundation Models TransformerClimaX [NBK+23], W-MAE [MZF+23],
Aurora [BBL+24], Prithvi WxC [SRT+24]
Figure 2.7: Taxonomy of deep learning models for weather prediction across training
paradigms (dark yellow), model scopes (purple), and model architectures (pink).
2.4.2 Machine Learning Weather Prediction
In recent years, data-driven machine learning (ML) and deep learning (DL) models have
been increasingly applied to weather and climate modeling, demonstrating remarkable
advances in precision, computational efficiency, and uncertainty quantification [CCZ+23,
NJB+23]. They have proven increasingly adept at capturing complex atmospheric dy-
namics in an end-to-end fashion, eliminating the reliance on explicit prior knowledge
of physical relationships. For example, deterministic models such as Pangu [BXZ+23]
and GraphCast [LSGW+22] have achieved state-of-the-art performance in medium-range
(10-day) global weather prediction, surpassing or matching traditional methods in terms
of accuracy on some benchmark datasets (e.g., ERA5) while dramatically reducing com-
27

putational costs (up to three orders of magnitude). However, their predictions are often
blurry since they are trained by minimizing point-wise loss functions. To overcome this
limitation, probabilistic generative models have emerged as powerful tools for weather
prediction while achieving uncertainty quantification in those predictions. They con-
sider weather prediction as probabilistic sampling (i.e., generation) conditioning on nec-
essary constraints. Models like CasCast [GBY+24], CoDiCast [SJHN24], and Gencast
[PSGA+24] leverage diffusion models for precipitation nowcasting and weather predic-
tion, delivering both probabilistic outputs and calibrated uncertainty estimates. More
recently, foundation models have gained traction in climate and weather modeling as an
emerging paradigm [BBL+24, SRT+24]. These models are pre-trained on massive his-
torical weather datasets to learn generalizable and comprehensive knowledge, which can
then be fine-tuned for diverse downstream tasks, e.g., weather forecasting and climate
downscaling [CLJ+23]. Foundation models offer two key advantages: (1) the ability to
learn robust and transferable weather representations from large-scale data, and (2) the
flexibility to adapt to downstream applications without the need for task-specific models
trained from scratch [MAS+24, ZXW+24]. We provide a comprehensive taxonomy in
Figure 2.7, including training paradigms, model architectures, and scopes.
2.5 LLMs for Scientific Question-Answering
Traditional information retrievers include BM25 [RW94, TPB14], a classical sparse re-
triever based on TF-IDF principles. It ranks documents by matching query terms, adjusted
for frequency and document length. Though efficient and interpretable, BM25 struggles
with semantic understanding, often missing paraphrased or contextually relevant content.
An important approach to address retrieval is the use of embeddings, especially ap-
proaches that incorporate semantics in the embedding space. Dense embedding retrievers,
28

such as DPR [KOM+20], Contriever [ICH+22], e5 [WYW+22], and ANCE [XXL+21]
improve performance in open-domain QA by capturing deeper semantics. However, se-
mantically similar documents may be topically irrelevant, leading to noise and halluci-
nations in the retrievals. Furthermore, their lack of interpretability limits their use in
sensitive domains [JSR+19].
Retrieval-Augmented Generation (RAG) is a technique that depends on embeddings
of entities from knowledge sources as part of a larger retrieval system. Many struc-
tured RAG methods have been explored. GraphRAG [ETC+24] builds entity graphs from
source documents, while LightRAG [GXY+24] uses dual-level retrieval for richer knowl-
edge access. HippoRAG [GSG+24] and HippoRAG 2 [GSQ+25] leverage knowledge
graphs inspired by cognitive theories to enhance retrieval and integration. SELF-RAG
[AWW+23] incorporates self-reflection to improve factuality, and RA-DIT [LCC+23]
fine-tunes LLMs with up-to-date knowledge. Despite their promise, graph-based meth-
ods face scalability challenges due to the complexity of graph construction and reasoning.
29

CHAPTER 3
FIDLAR: FORECAST-INFORMED DEEP LEARNING APPROACHES FOR
FLOOD PREDICTION AND MITIGATION IN COASTAL RIVER SYSTEMS
3.1 Background
Floods are a significant threat to both lives and property [IJH+24, RSS+25], and pose
environmental and public health hazards [Gla21]. Anticipating the timing and location
of floods could empower water management agencies to utilize hydraulic structures and
implement timely flood mitigation strategies [KAS+23], thus enabling citizens and lo-
cal governments to enhance preparedness for potential emergencies [YYY+24]. Thus,
predicting floods in advance and solving optimization problems related to flood mitiga-
tion are vitally significant endeavors. In this chapter, we focus on floods in coastal river
systems, with special emphasis on the South Florida river system involving the Miami
river and the associated canals as it flows into the Atlantic Ocean. The work presented
in this chapter is highlighted in multiple manuscripts and archival publications, including
[SJN22, SYM+23, SMS+23, SSW+23, SSN23].
Flood prediction methods are tasked with determining the location, intensity, and du-
ration of a flood event in a river system. Classical methods over the last decade or more
have been achieved by starting with a detailed physics-based model and applying physical
laws of conservation of water from first principles. Despite their remarkable advances,
physics-based classical methods have the following limitations.
1.Computationally Intensive:For large watersheds, these models are computation-
ally demanding [CXL+23, VBMM22] because they use detailed grid representa-
tions and often need to solve many complex partial differential equations (PDEs)
for each cell of the grid [JCT+23].
30

2.Accuracy Challenges:Physics-based models may compromise accuracy by using
simplifications and approximations that reduce model complexity [GAC+22]. Such
changes may limit the ability of the models to comprehensively cover the entire
spectrum of process variability [GKBG23].
3.Explainability Issues:Despite elucidating complex dynamics, they struggle to
provide detailed explanations, particularly in identifying specific variables and time
steps influencing simulation outcomes.
Flood management involves going one step above flood prediction. It takes proactive
steps to determine appropriate interventions to prevent or mitigate flooding, especially in
anticipation of an impending storm. The primary tool for flood management is the use of
hydraulic structures constructed in the river systems for the purpose of controlling water
levels. Flood mitigation is typically achieved by pre-releasing sufficient water in advance
so that water levels remain within a safe range when the storm event occurs [MMM+22].
Pre-release is carried out by controlling hydraulic structures such as dams, gates, pumps,
and reservoirs, and determining acontrol schedulefor these structures. However, de-
termining the optimal control schedules of these hydraulic structures is a challenging
problem [BTW+21]. Most water management agencies have comprehensive prespecified
Rule-based methods[SGB+19] that help formulate the control schedules. Such rules have
been designed based on human insights gained from having managed river systems over
decades and the experience gained by the decision made in the past. Nevertheless, these
rules may expose vulnerabilities while dealing with extremely rare events and may not
offer effective solutions for different river systems [SBX15]. Another method for deter-
mining good control schedules is to formulate it as anoptimizationproblem. Previous
approaches include the use of soft computing techniques such as genetic algorithms (GA)
and pattern search (PS) combined with a detailed physics-based simulator to evaluate
31

the solutions [DDBB24, MSO+20, EXES+21, SABA21]. However, because of the use of
detailed simulators, they are typically too inefficient to be used for real-time flood control.
With the rapid advancements in artificial intelligence (AI) [Ala21], deep learning (DL)
models have emerged as powerful tools across numerous domains, demonstrating impres-
sive performance [CDC+22, XLC+21]. DL models possess a range of characteristics,
including their ability to effectively learn complex non-linear relationships [WZP20], de-
liver ultra-fast responses once trained [OKP+22], utilize feedback signals via the back-
propagation algorithm during training [EWN+23], and the availability of tools for model
explainability [AS20].
In this chapter, we first propose a novel DL model for water level forecasting (WALEF).
Then we present a forecast-informed deep learning approach for flood mitigation (FIDLAR).
We perform extensive experiments to evaluate the performance of WALEF and FIDLAR,
and discuss their performance relative to baseline methods.
3.2 Related Work
Flood Prediction.Physics-based models (e.g., HEC-RAS, SWMM) have been widely
used to simulate water levels and flows in river systems [PGD+24, RA24, GJRO+23,
RTLC+22]. However, these models are computationally inefficient and fall short of cap-
turing precise knowledge of study domains [BIJT22]. Therefore, diverse machine learn-
ing (ML) and deep learning (DL) models have been studied as surrogates to simulate
water levels and flows [RSS+25, RSP+25, ZLC+25]. For example, support vector ma-
chines (SVMs) have been used to predict the urban flash floods [YJC+18, CMG+19],
multivariate regression models were adopted to estimate flood volumes and peak flows
[YC20], random forests and K-nearest neighbors were explored for urban flood inunda-
tion mapping [CGPL+14], Gaussian process learning models for fast and accurate flood
32

inundation simulation [FWWN23]. Furthermore, deep learning models, such as recurrent
neural networks, convolutional neural networks, and transformers, have been employed
for flood inundation [ZWNW21] and flood prediction [SSW+23, SYM+23].
Flood Mitigation.Flood mitigation can be achieved by designing control schedules
for managing all the hydraulic structures present in river systems to avoid or mitigate
flood risks. However, this process may require flood prediction models as simulators
to evaluate the control schedules. Researchers have attempted to leverage the genetic
algorithm and the pattern search to generate control schedules and physics-based mod-
els (e.g., EPA-SWMM5, HEC-RAS) as water simulators [SGB+19, LKVS14, LTQC20].
The Lake Mendocino Operations (LMO) model was developed to simulate operations of
Lake Mendocino such as release constraints for flood control and water supply operations
[DHM+20]. However, such methods are computationally intensive since they require
thousands of time-consuming simulation trials with physics-based models [JMP+23].
Additionally, the genetic algorithm and pattern search techniques are usually heuristic,
without positive feedback or guidance to better control those hydraulic structures.
3.3 Methodology
3.3.1 Problem Formulation
Flood Prediction.It refers to forecasting water levels at designated points of interest
within river systems, given the input as known water levels from the recent past, rainfall,
and control schedules of gates and pumps. The underlying transfer function is:
EθE: (Xall
t−w+1:t , Xcov
t+1:t+k , Xgate,pump
t+1:t+k )→Xwater
t+1:t+k ,(3.1)
33

where the subscripts represent the time ranges, and the superscripts refer to the variables
under consideration. The superscriptallrepresents all possible variables in a watershed
(e.g., rainfall, tide, gate and pump operation, water levels), whilecovrefers specifically
to covariates that can be reliably predicted (e.g., rain, tides).
Once trained, it plays the role of a “referee” who evaluates those control schedules by
outputting the resulting water levels. We use the trained models for water level prediction
asFlood Evaluatorin the following sections.
Flood Mitigation.It aims to manage water levels before extreme weather events by
predicting control schedules for hydraulic structures such as gates and pumps within the
river system, denoted asXgate,pump
t+1:t+k , spanningktime points into the future fromt+ 1to
t+k. Flood prediction and mitigation present a significant challenge due to the intricate
interplay of meteorological, hydrological, and oceanographic factors. We take as input
historical data on all possible factors,X, from the precedingwtime points, in conjunction
with reliably forecasted covariates (such as rainfall and tide) for the nextktime points.
Then we could train a deep learning (DL) model,M θM, with parametersθ M:
MθM: (Xall
t−w+1:t , Xcov
t+1:t+k )→Xgate,pump
t+1:t+k .(3.2)
3.3.2 Method Overview
Intuitively, an ML model can be trained to learn the functionM θM(Flood Manager)
to output the control schedules of gates and pumps directly. However, a key challenge
lies in the historical data, which often reflects control schedules that led to flooding or
other suboptimal outcomes, making it unsuitable as ground-truth data for traditional su-
pervised learning. To overcome this, we first train an independent and accurate simulator
model (Flood Evaluator),M θE, by using extensive historical data to model the
34

consequences(e.g., water levels) of variousactions(e.g., control schedules). We then
frame the control schedule planning inFlood Manageras an optimization problem,
seeking actions that minimize undesirable outcomes - floods or water wastage. Both the
EvaluatorandManagerare implemented as neural networks, with the framework
illustrated in Figure 3.1.
Figure 3.1: Forecast-Informed Deep Learning Architecture (FIDLAR). Input data con-
sists of five categories of variables as shown in the left table. The variableswandkare
the lengths of the past and prediction windows, respectively. The parts colored green are
provided as inputs, while the orange and blue parts (with question marks) are outputs.
TheFlood ManagerandFlood Evaluatorrepresent deep learning (DL) models,
the former to predict control schedules of controllable hydraulic structures (e.g., gates and
pumps) to pre-release water, and the latter to predict the resulting water levels for those
control schedules. Loss functions,L 1andL 2, penalize thefloodingandwater wastage
beyond pre-specified thresholds, respectively.
3.3.3 Flood Evaluator
TheEvaluatoris trained independently using large-scale historical data to achieve
highly accurate water level predictions for any given set of conditions and control sched-
ules. Therefore, once theEvaluatoris well trained, its parameters are frozen while
training theManager, where it plays the role of a trained “referee” - scoring control
schedules generated by theManagerby predicting the resulting water levels. It also
serves to backpropagate the gradient descent feedback, guiding theManagerto produce
more effective control schedules of gates and pumps.
35

Figure 3.2:Flood Evaluator.The parts shaded green are used as inputs (i.e., historical
data and covariates predicted from the near future) and orange (control schedule for the
gates and pumps) Water levels (blue) are the outputs.
3.3.4 Flood Manager
Flood Manageris to produce control schedules for hydraulic structures (i.e., gates and
pumps), taking as inputs reliably predictable future information (rain, tide) and all histor-
ical data. Since no ground truth is available, it is trained with the differentiability of the
learnedEvaluatormodel. Therefore, we connect theManagerwith theEvaluator
where the output of Eq. (3.2) is injected into Eq. (3.1):
EθE(Xall, Xcov
t+1:t+k ,M θM(Xall, Xcov
t+1:t+k ))→Xwater
t+1:t+k ,(3.3)
whereXall=Xall
t−w+1:t andθ Mandθ Eare the parameters ofManagerandEvaluator.
Figure 3.3:Flood Manager.The parts shaded green (historical data) are the inputs, and
the parts shaded orange are the outputs. The water levels shaded blue are not predicted.
The resulting output of water levels can be used to compute the loss in Eq. (3.6),
representing evaluation scores for generated control schedules. Gradient descent [Rud16]
can be back-propagated as the feedback to update the parameters of theManager. The
36

parameter update is presented:
θM:=θ M−α·∂L
∂θM,(3.4)
whereαis the learning rate and∂L
∂θMis the partial derivative of the compound function
∂L
∂θM=∂L
∂E·∂E
∂M·∂M
∂θM. The training details of FIDLARare in Algorithm 1.
Algorithm 1Training algorithm of FIDLAR
Input: recent past data:Xall
t−w+1,t
Input: near future data:Xcov
t+1,t+k=Xrain,tide,gate,pump
t+1,t+k
Parameter:θ E, θM: parameters of Evaluator and Manager;w, k:length of past and prediction
windows
1: //Train Flood Evaluator,E θE
2: initialize learnable parametersθ E
3:fori= 1, . . . , Nepochsdo
4: MiniBatch←({Xall
t−w+1,t ,Xcov
t+1,t+k},Xwater
t+1,t+k)
5: ˆXwater
t+1,t+k← E θE(Xall
t−w+1,t ,Xcov
t+1,t+k)
6:L E←1
kPk
j=1||ˆXwater
j−Xwater
j||2
7:∇ θE←BackwardAD(L E)
8:θ E←θ E−η∇ θE
9:end for
10:returntrainedFlood Evaluator,E θE
11: //Train Flood Manager,M θM, with frozenE θE
12: initialize learnable parametersθ M
13:whileXwater
t,t+kviolates either thresholddo
14: MiniBatch←({Xall
t−w+1,t ,Xrain,tide
t+1,t+k},Xgate,pump
t+1,t+k)
15: ˆXgate,pump
t+1,t+k← M θM(Xall
t−w+1,t ,Xrain,tide
t+1,t+k)
16: ˆXwater
t+1,t+k← E θE(Xall
t−w+1,t ,Xrain,tide
t+1,t+k,ˆXgate,pump
t+1,t+k)
17:L E=c1· L1(ˆXwater
t+1,t+k) +c 2· L2(ˆXwater
t+1,t+k)
18:∇ θM←BackwardAD(L E)
19:θ M←θ M−η∇ θM
20:end while
21:returntrainedFlood Manager,M θM
3.3.5 Custom Loss Function
Loss functions are critical in steering the learning process. Our loss function penalizes
the total time (Figure 3.4a) for which the water levels either exceed theflooding threshold
37

or dip below thewater wastage threshold. However, such a measure does not account for
the severity of the flood, but only the time for which the flood occurs. To incorporate the
deviation from optimality associated with the severity of the flood or the wastage of water
caused by pre-releases, our loss function also penalizes the extent to which the thresholds
are exceeded to signify the severity of floods or water wastage (Figure 3.4b). The lower
threshold for flood management is important in practice, since it prevents water wastage,
thereby supporting irrigation, facilitating navigation, and maintaining ecological balance.
It also prevents the optimization methods from trivially recommending the depletion of
valuable water resources to prevent future flooding.L 1andL 2represent thefloodingand
water wastagelosses, respectively, and the final loss function is a balanced combination
as shown in Eq. (3.6).
L1=NX
i=1t+kX
j=t+1∥max{ ˆXwater
i,j−Xflood
i,0}∥2,
L2=NX
i=1t+kX
j=t+1∥min{ ˆXwater
i,j−Xwaste
i,0}∥2,(3.5)
whereNis the number of water level locations of interest;kis the length of prediction
horizon;XfloodandXwasterepresent the thresholds for flooding and water wastage; and
the capped version, ˆXwater, is obtained using theEvaluatormodule. The combined
loss function is given by:
Ltotal=c1· L1+c2· L2,(3.6)
wherec 1/c2dictates the relative importance ofL 1andL 2.
3.4 Architecture of the Neural Network Model
TheManagerandEvaluatormodules described so far are model agnostic. We tried
many existing deep learning architectures for them. We devise the G raph T ransformer
38

(a)
 (b)
Figure 3.4: The two red bars represent a threshold of flooding and a threshold of water
wastage. Shown are (a) the time spans when these thresholds are crossed, and (b) the
areas between water level curves and threshold bars. Violations of the upper and lower
thresholds are captured inL 1andL 2.
Network (GTN) architecture by combining graph neural networks (GNNs), attention-
based transformer networks, long short-term memory networks (LSTMs), and convolu-
tional neural networks (CNNs). GNN and LSTM modules are combined to learn the
spatiotemporal dynamics of water levels, while the Transformer and CNN modules fo-
cus on extracting feature representations from the covariates. Theattentionmechanism
[VSP+17] is used to discern interactions between covariates and water levels, as shown
in Eq. (3.7) below. Figure 3.5 presents the GTN architecture, which is used for both
EvaluatorandManager, but with minor changes accordingly of the inputs and out-
puts (see Figures 3.2 and 3.3). We have two GNN layers with 32 and 16 channels, one
LSTM layer, one CNN layer, and one Transformer encoder with 3 heads.
Atte(Q, K, V) =softmax(Qcov(Kwater)T
√
d)Vwater
=softmax(Qwater(Kcov)T
√
d)Vcov,(3.7)
whereTdenotes the transpose operation;waterandcovrepresent water levels and co-
variates; anddis the embedding size whered=d q=d k=d v= 128.
39

Figure 3.5: Graph Transformer Network forFlood Evaluator. Input variables in-
clude Rainfall, Pump, Gate, Tide, and Water levels as shown in Figure 3.6, which are
generically denoted byR, T, G, P, andW. The output is water levels.
3.5 Experiments
3.5.1 Study Domain and Dataset
We obtained data from the South Florida Water Management District’s (SFWMD) DB-
Hydro database [Dis23] for the coastal stretch in South Florida. The data set consists of
hourly observations for water levels and external covariates from January 1, 2010 to De-
cember 31, 2020. As shown in Figure 3.6, the river system has two branches and includes
several hydraulic structures (gates, pumps) to control water flows. We aim to predict ef-
fective control schedules on hydraulic structures (gates, pumps) to minimize flood risks
at four specific locations marked by green circles.
3.5.2 Experimental Design
The sliding input window [LNML14] (also known as look-back window [GK18] strategy
was used to process the entire dataset [SMS+23]). For consistency, we used a look-back
window of lengthw= 72hours and a prediction window of lengthk= 24hours. The
40

Figure 3.6: Schematic diagram of study domain. There are three water stations with
hydraulic structures (S26, S25B, S25A), one simple water station, S1 (green circle in the
middle), and a station monitoring the tide level from the ocean.
Feature Interval Unit #Var. Location
Rainfall Hourlyinch/h1 -
Tide Hourlyft1 S4
Pump Hourlyft3/s2 S25B, S26
Gate Hourlyft3 S25A, S25B, S26
Water Hourlyft4 S25A, S25B, S26, S1
Table 3.1: Summary of the data set.
dataset was split in chronological order with the first 80% for training and the remain-
ing 20% for testing. The eight DL methods below are used forFlood Managerand
Flood Evaluator. We run all experiments on one NVIDIA A100 GPU with 80GB
memory.
•MLP[SVDM95]: A multilayer perceptron that models non-linear dependencies
through fully connected layers;
•RNN[MJ01]: Recurrent neural networks are designed for processing sequential
data by maintaining hidden states across time steps, making them suitable for time-
dependent patterns;
41

•CNN[ON15]: A 1D convolutional neural network that captures local temporal
features using sliding filters;
•GNN[KW16]: A graph neural network where nodes represent variables and edges
model spatial relationships;
•TCN[BKK18]: A temporal convolutional network with dilated convolutions, en-
abling a large receptive field to capture long-range temporal dependencies;
•RCNN[ZD20]: A hybrid model that integrates RNN and CNN architectures to
jointly learn local and sequential features for time series forecasting;
•Transformer[VSP+17]: An attention-based model for sequence modeling (encoder-
only in our setting) to capture global dependencies;
•GTN(Ours): Combining GNNs with LSTMs, CNNs, and transformers, as de-
scribed in Figure 3.5.
3.5.3 Reproducibility
We include all data and code in a GitHub repository1.
3.6 Results
3.6.1 Flood Prediction
The role of theFlood Evaluatoris to forecast flood events by predicting water lev-
els for given input conditions. We set the upper threshold (flood level) at 3.5 feet and
the lower threshold (wastage level) at 0.0 feet. However, the methods remain consistent
for many reasonable choices of threshold values. We measured accuracy using multiple
1Link:https://github.com/JimengShi/FIDLAR
42

metrics: (a) mean absolute error (MAE), (b) root mean squared error (RMSE) computed
between the predicted and actual water levels, (c) number of time points where the up-
per or lower thresholds are breached, and (d) the area between water level curves and
threshold bars.
Methods MAE (ft) RMSE (ft) Over Timesteps Over Area Under Timesteps Under Area
Ground-truth - - 96 14.82 1,346 385.80
HEC-RAS 0.174 0.222 68 10.07 1,133 325.33
MLP 0.065 0.086 147 27.96 1,677 500.41
RNN 0.054 0.072 110 17.12 1,527 441.41
CNN 0.079 0.104 58 5.91 1,491 413.22
GNN 0.054 0.070 102 15.90 1,569 462.63
TCN 0.050 0.065 47 5.14 1,607 453.63
RCNN 0.092 0.110 37 4.61 1,829 553.20
Transformer 0.050 0.066 151 25.95 1,513 434.13
GTN (ours) 0.040 0.056 100 15.64 1,390 398.84
Table 3.2: Comparison of model performances for theFlood Evaluatoron the test
set, specifically at time t+1 for measurement station S1. The terms “Over Timesteps” and
“Under Timesteps” indicate the number of time steps during which water levels exceed
the upper threshold or fall below the lower threshold, respectively. Similarly, “Over Area”
and “Under Area” pertain to the area between the water level curve and upper or lower
threshold, as was illustrated in Figure 3.4. Results in orange are the lowest in that column
while results in red are the closest to the ground truth (in blue).
Table 3.2 demonstrates that our modelGTNoutperforms other models with predictions
(in red) most closely aligned with the ground truth (in blue) while achieving the lowest
MAE and RMSE (in orange). Therefore, we choose our GTN model asEvaluator
while trainingManagerin FIDLAR.
3.6.2 Flood Mitigation with FIDLAR
FIDLARrequires bothEvaluatorandManagercomponents. For theManager
model, we experimented with one rule-based method, and two genetic algorithms – one
with a physics-based HEC-RAS evaluator [LTQC20] and one with our DL-based GTN
43

evaluator, and several DL-based managers using MLP, RNN, CNN, GNN, TCN, RCNN,
Transformer, and GTN. FIDLARwas measured using (a) the number of time steps where
the upper/lower thresholds are exceeded for the water levels, and (b) the area between the
water level curves and the threshold bars. Table 3.3 shows that all DL-based methods
consistently performed better for site S1 than rule-based and GA-based approaches. Fur-
thermore, GTN has the best performance under all four metrics, whether it is to control
floods or water wastage. We visualize water levels for a short period spanning 18 hours
Method Manager Over Timesteps Over Area Under Timesteps Under Area
Rule-based 96 14.82 1,346 385.8
GA-basedGenetic Algorithm∗- - - -
Genetic Algorithm†86 16.54 454 104
DL-basedMLP 91 13.31 1,071 268.35
RNN 35 3.97 351 61.05
CNN 81 11.22 1,163 314.37
GNN 31 3.72 429 84.31
TCN 39 3.77 306 55.12
RCNN 29 3.28 328 58.68
Transformer 85 11.54 1,180 310.16
GTN (Ours) 22 2.23 299 53.34
Table 3.3: Comparison of model performances for theFlood Manageron the test set,
specifically at time t+1 for measurement station S1. The∗denotes that the GA method
was used with a physics-based (HEC-RAS) evaluator. The−denotes that the experiments
were timed out. The†denotes the GA method was used with the GTN as the evaluator. All
other rows are DL-based flood managers with a DL-based GTN as the evaluator. Results
inboldare the best in that column.
from September 3rd (09:00) to September 4th (03:00) in 2019 for the S1 location. Figure
3.7 indicates that FIDLARequipped with GTN model (purple curve) has led to water
levels within the upper and lower thresholds, satisfying pre-defined requirements. More-
over, FIDLARpresents the best control (i.e., the least water levels beyond thresholds)
for flood mitigation and water waste compared to other baselines. We zoomed in on a
2.5-hour period of the resulting decreased water levels. Visualizations at all locations are
in Figure 3.8.
44

Figure 3.7: Visualization of water levels with various methods for flood mitigation.
We zoomed int= 2∼4.5in gray. ‘A’–‘B’ in legend represents theManagerand
Evaluator. Two dashed lines denote the upper (3.5 ft) and lower threshold (0.0 ft).
Figure 3.8: Visualization for flood mitigation atalllocations.
3.6.3 Ablation Study
As introduced in Figure 3.5 in Section 3.4, the graph transformer architecture (GTN)
consists of multiple components. Theablationstudy in Table 3.4 quantifies the contribu-
tion of each component of our GTN model by measuring the performance of GTN after
removing the individual components and a part of their combination. Removing the com-
bination of “GNN & LSTM & Attention” (bottom branch in Figure 3.5) or “Transformer
& CNN & Attention” (upper branch in Figure 3.5) results in the most significant perfor-
45

mance degradation. For each component, the removal of the GNN module decreases the
performance the most, underscoring the importance of spatial dependency modeling in
this task. Removing the CNN or LSTM components also leads to considerable deterio-
ration, particularly in under-timesteps and under-area, indicating that both local temporal
feature extraction and sequential modeling are essential. Interestingly, the exclusion of the
Transformer and attention mechanisms results in moderate drops, suggesting that while
global context is helpful, the model remains somewhat resilient without it. However, the
full integration of all components, CNN, LSTM, GNN, Transformer, and attention, yields
the best overall performance, validating the architectural design of GTN as a synergistic
combination of spatiotemporal modeling techniques.
Over Over Under Under
Method Timesteps Area Timesteps Area
w/o CNN 37 4.37 476 85.54
w/o Transformer 32 3.57 325 57.42
w/o GNN 56 5.90 479 86.22
w/o LSTM 35 4.34 329 56.74
w/o Attention 32 3.59 341 60.48
w/o Transformer, CNN, Attention 69 11.24 828 215.94
w/o GNN, LSTM, Attention 78 9.65 761 198.33
GTN (Ours) 22 2.23 299 53.34
Table 3.4: Ablation study for flood mitigation for the entire test set (for time point t+1
at S1). The last row indicates the performance of the FIDLARsystem with GTN as
proposed in Figure 3.5. The best results in the last row are in bold.
3.6.4 Analysis of Computational Time
Since FIDLARwas designed for real-time flood control, we measured the running times
of the models used in this work and the previous genetic algorithm (GA) associated with
physics-based flood simulators. Table 3.5 shows the running times for the whole flood
prediction and mitigation system in its training and test phases. All the DL-based ap-
46

proaches in the test phase are several orders of magnitude faster than the currently used
physics-based and GA-based approaches for the flood mitigation task. Rapid inference is
a critical property of data-driven DL methods. The table also shows the training times for
the DL-based approaches, although they are not necessary for the deployment in reality.
ModelFlood Prediction Flood Mitigation
Train Test Train Test
HEC-RAS ×45 min× ×
Rule-based / / / /
GA∗× × ×–
GA†× × ×est. 30 h
MLP 35 min 1.88 s 58 min 6.13 s
RNN 243 min 8.57 s 54 min 12.75 s
CNN 37 min 1.93 s 17 min 5.84 s
GNN 64 min 3.13 s 29 min 7.26 s
TCN 60 min 4.57 s 45 min 9.06 s
RCNN 136 min 8.61 s 61 min 13.27 s
Transformer 43 min 2.38 s 23 min 6.76 s
GTN (Ours) 119 min 2.95 s 35 min 4.90 s
Table 3.5: Running time for flood prediction and mitigation. The running time for the
rule-based method was not reported since historical data was directly used, denoted with
“/”. “×” represents that the methods are not applicable for the tasks. GA∗, which com-
bines a GA-based tool and HEC-RAS for flood mitigation (test phase), took too long
and was not reported, represented with “–”. GA†, which combines the GA-based tool
with GTN for flood mitigation (test phase), also took too long but was estimated using a
smaller sample.
3.6.5 Explainability
It is also important to figure out how and why the model generates the outputs. Attention-
based methods allow us to calculate the “attention scores” assigned to an input variable
to compute a specific output variable. The heatmap in Figure 3.9 presents the attention
scores assigned to the tide (columns) to compute the gate schedule output (rows) for 24
47

hours into the future. Note that there are 96 columns and 24 rows because we use 72
hours of past tidal observations and 24 hours of future predicted tidal data to predict 24
hours of the gate schedule into the future. The rows[0,23]correspond to the 24 hours
into the future, while the columns[0,95]also include 72 hours of the recent past and 24
hours of the future predicted tidal data. Therefore,t= 72corresponds to the “current”
time point, and the columns[72,95]correspond to the same time points as rows[0,23].
Figure 3.9: Importance scores of tide input.xandyaxes are the tide and control schedule
of the gate over time.
3.7 Discussion
3.7.1 Model Explainability
The explainability feature, which is shown with an example in Figure 3.9, can provide
significant insights into our results.
Firstly, we point out that the brightest patches are in the last 24 columns of the
heatmap. Thus FIDLARpays greater attention to the 24 hours of future predicted tidal
data than the past 72 hours, highlighting the importance of forecast information to a DL-
48

based approach to flood mitigation. While tides may have a more predictable pattern over
time, the contribution of rain to the water levels can also be seen for other time points.
Since our study domain is a coastal river system, tidal information dominates the changes
of water levels, and rainfall events have relatively less impact. More results on model
explainability are referred to [SSN23, SMS+23].
A second critical insight is that the brightest attention patches are in columns where
the tide is at its highest is critical to the prediction of gate schedules. Additionally, the
water level at the first high tide peak after the “current” time is more significant than the
other two. Third, the gate schedule has peaks at timest= 11hour andt= 22hour into
the future, which correspond to the lowest points of the tide. This implies that the optimal
time for pre-releasing water is during low tide phases. Opening gates during high tide
periods in coastal river systems is less advisable, as it may lead to water flowing back
upstream from the ocean.
Finally, we observe that there is a light patch around columnt= 65hour, suggesting
mild attention for the previous high tide peak, but almost no attention to any of the peaks
before that. This again suggests that we could have chosen to use a smaller window for
the past input. Doing this analysis could provide evidence for the right value ofw, the
size of the look-back window.
3.7.2 Spatio-temporal Modeling
We developed the graph transformer model with a combination of GNN and LSTM on
one arm, and a Transformer and CNN on another arm, both combined using an cross-
attention module. Specifically, a GNN was used to capture the spatial relationship of the
water levels between different water stations. Each station is considered as one node and
the directed edges represent the direction of water flow from one station to another. In
49

contrast, the transformer was used to model the temporal dependencies of the covariates
in the time series (e.g., rainfall, tidal information, control schedules of gates and pumps).
We conjecture that this combination of spatial and temporal modeling is a powerful way
to model complex environmental systems. The ablation study in Section 3.6.3 quantifies
the contribution of individual components to the overall performance of FIDLAR.
3.7.3 Generalization to Other River Systems
The models for flood prediction and mitigation introduced in this chapter can be read-
ily applied to other single-region river systems, but they need to be re-trained using the
domain-specific data (e.g., water levels, rainfall, control schedules for gates and pumps,
etc.). The limitations to more complex river systems with multiple cascading gates and
pumps have not been addressed in this dissertation and are discussed below in Section
3.8.
3.7.4 Deep Learning for Optimization
Traditional optimization methods require precise physical modeling. If the models in-
volve nonlinear functions, as may be the case in complex river systems, then they are
likely to be computationally intensive. Deep learning models can approximate these com-
plex relationships and serve as fast surrogates for physical simulations, enabling more
efficient and adaptive optimization strategies. In this chapter, accurate predictions of the
water levels under different control schedules for gates and pumps was used to solve the
optimization problem of predicting a good control schedule that would minimize flood
risks. The question of whether a prediction tool in tandem with an evaluation tool is an
appropriate architecture for other optimization problems remains anopen problem.
50

3.8 Limitations
3.8.1 Simple Regional Domain
All the experiments reported in this dissertation involve one section of the river, meaning
that none of the measuring stations had a gate, pump, or some hydraulic structure sepa-
rating them. However, real-world river systems are more complex with many hydraulic
structures and water stations. The solutions in this chapter are not expected to work for
river systems with multiple sections without considerable changes in the architecture of
the model. This is because the models have to account for the simple fact that water lev-
els in one section are independent of the levels in other sections if the gates and pumps
are closed. Therefore, extending our methodology to solve flood management in more
complex river systems is viewed as future work.
3.8.2 Human Intervention
The implemented DL-based methods have not been tested in real-life scenarios. Given
that human response times for gate adjustments are typically much longer (around 6
hours) than the computation time of machine learning model outputs, we could man-
age control strategies by maintaining a fixed schedule over each 6-hour window while
preserving the overall area under the control curve. Specifically, ML models first gener-
ate the fine-grained, high-resolution, fluctuating control signals, and then we can easily
convert them into coarser, human-executable ones by smoothing techniques in the post-
processing. This approach will strike a balance between the model’s responsiveness and
the feasibility of manual intervention, making the implementation remain practical and
aligned with real-world operational constraints. We point out that this reconciliation is
crucial in translating ML-based recommendations into actionable flood control strategies,
51

particularly in emergency scenarios where clarity, stability, and operational simplicity are
essential.
3.9 Conclusions
In this chapter, we discuss the shortcomings of the current approaches for flood mitiga-
tion. To address the challenges, we propose FIDLAR, a DL-based tool to address the
problem. FIDLARcan compute water “pre-release” schedules for hydraulic structures in
a river system to achieve effective and efficient flood mitigation, while ensuring that water
wastage is avoided. This was made possible by the use of well-crafted loss functions for
the DL models. The dual component design (with aManagerand anEvaluator) is
a strength of FIDLAR. It exploits the gradient-based planning and the differentiability
of the trainedEvaluatormodel for better optimization. During training, the gradient-
based back-propagation from theEvaluatorhelps to reinforce theManager.
All the DL-based versions of FIDLARare several orders of magnitude faster than
the (physics-based or GA-based) competitors while achieving improvement over other
methods in flood mitigation. These characteristics allow us to entertain the possibility of
real-time flood management, which was challenging for previous approaches.
52

CHAPTER 4
CODICAST: CONDITIONAL DIFFUSION MODEL FOR GLOBAL WEATHER
PREDICTION WITH UNCERTAINTY QUANTIFICATION
4.1 Background
Accurate weather forecasting is crucial for a wide range of societal activities, from daily
planning to disaster preparedness [MKK+20, SYL+24]. For example, governments, or-
ganizations, and individuals rely heavily on weather forecasts to make informed deci-
sions that can significantly impact safety, economic efficiency, and overall well-being.
However, weather predictions are intrinsically uncertain largely due to the complex and
chaotic nature of atmospheric processes [SP11]. Therefore, assessing the range of prob-
able weather scenarios is significant, enabling informed decision-making. The work pre-
sented in this chapter is highlighted in the following publication [SJHN24].
Traditional physics-based numerical weather prediction (NWP) methods achieve weather
forecasting by approximately solving the differential equations representing the integrated
system between the atmosphere, land, and ocean [PSGA+24, NBK+23]. However, run-
ning such an NWP model can produce only one possibility of the forecast, which ignores
the weather uncertainty. To solve this problem,Ensemble forecast1of multiple models
is often employed to model the probability distribution of different future weather sce-
narios [Pal19, LHN+23]. While such physics-based ensemble forecasts effectively model
the weather uncertainty, they have two primary limitations: physics-based models inher-
ently make restrictive assumptions of atmospheric dynamics, and running multiple NWP
models requires extreme computational costs [RP07].
In recent years, machine learning (ML)-based weather predictions (MLWP) have been
proposed to challenge NWP-based prediction methods [BBCM+24, BHQL24]. They
1Generating a set of forecasts, each of which represents a single possible scenario.
53

have achieved enormous success with comparable accuracy and a much (usually three
orders of magnitude) lower computational overhead. They are typically trained to learn
weather patterns from a huge amount of historical data and predict the mean of the
probable trajectories by minimizing the mean squared error (MSE) of model forecasts
[HTPB21]. Representative work includes Pangu [BXZ+23], GraphCast [LSGW+23],
ClimaX [NBK+23], ForeCastNet [PSH+22], Fuxi [CZZ+23], Fengwu and [CHG+23].
Despite the notable achievements of these MLWP methods, most of them are determinis-
tic [KYL+24], falling short in capturing the uncertainty in weather forecasts [JK22].
To compute the uncertainty for ML models, two methods exist. Perturbing initial
conditions [MWW18] helps estimate the aleatoric uncertainty (data noise), while the
Monte Carlo Dropout approach [GG16] estimates epistemic uncertainty (model uncer-
tainty) [SMK+22]. However, neither approach fully captures uncertainty in both the in-
put conditions and the evolution of weather models. Additionally, these methods require
manual tuning of perturbations and dropout rates, which can negatively impact model
accuracy. These limitations motivate us to explore an approach for comprehensive uncer-
tainty quantification without the loss of accuracy.
Denoising diffusion probabilistic models (DDPMs) [HJA20] stand out as a probabilis-
tic type of generative model, which can generate high-quality images. By explicitly and it-
eratively modeling the noise additive and its removal, DDPMs can capture intricate details
and textures of images. Furthermore, controllable diffusion models [RBL+22, ZRA23]
enable the generation process to be guided by specific attributes or conditions, e.g., class
labels, textual descriptions, or other auxiliary information. By doing so, the models can
generate images that adhere to the specified conditions. This inspires us to consider the
weather “prediction” tasks as “generation” tasks - generating plausible weather scenar-
ios with conditional diffusion models. Promising potentials could be the following: (1)
Weather numerical data is usually a 2-D grid over latitude and longitude, sharing a similar
54

modality with the image. Diffusion models can capture the intricate weather distribution
with iterative denoising. (2) Weather states from the recent past (i.e., initial conditions)
can be injected into diffusion models to guide the generation of future weather evolution.
(3) More notably, the starting noise sampling from the Gaussian distribution can mimic
the aleatoric uncertainty, while iteratively adding and removing noise captures the epis-
temic uncertainty. These features prompt probabilistic diffusion models to generate a set
of diverse weather scenarios rather than a single deterministic one. This capability makes
them well-suited for modeling the uncertain nature of weather evolution.
In this chapter, we identify the shortcomings of current weather prediction methods.
Physics-based NWP methods are limited to restrictive assumptions and are computation-
ally intensive. Moreover, a single deterministic NWP- and MLWP-based method cannot
achieve uncertainty quantification. To address these problems, we propose CODICAST,
a conditional diffusion model for global weather prediction conditioning on observations
from the recent past while probabilistically modeling the uncertainty. In addition, we use
the cross-attention mechanism to effectively integrate conditions into the denoising pro-
cess to guide the generation tasks. We conduct extensive experiments on the ERA5 reanal-
ysis data from the European Centre for Medium-Range Weather Forecasts (ECMWF), and
demonstrate that CODICASTachieves an essential trade-off among accuracy, efficiency,
and uncertainty against state-of-the-art baselines.
4.2 Related Work
Physics-based Numerical Weather Prediction.Physics-based numerical Weather Pre-
diction (NWP) methods achieve weather forecasts by modeling the system of the atmo-
sphere, land, and ocean with complex differential equations [BTB15]. For example,the
High-Resolution Forecasts System (HRES) [ECM23] forecasts possible weather evolu-
55

tion out to 10 days ahead. However, it is a deterministic NWP method that only provides
a single forecast. To overcome the limitation of deterministic methods, the ensemble
forecast suite (ENS) [Bui08] was developed as an ensemble of 51 forecasts by the Euro-
pean Centre for Medium-Range Weather Forecasts (ECMWF). ENS provides a range of
possible future weather states, allowing for investigation of the detail and uncertainty in
the forecast. Even if NWP ensemble forecasts effectively model the weather evolution,
they exhibit sensitivity to structural discrepancies across models and high computational
demands [BCD+22].
ML-Based Weather Prediction.Machine learning-based weather prediction (MLWP)
approaches have challenged NWP methods for weather forecasting. Pangu [BXZ+23]
employed three-dimensional transformer networks and Earth-specific priors to deal with
complex patterns in weather data. GraphCast [LSGW+23] achieved medium-range weather
prediction by utilizing an “encode-process-decode” configuration with each part imple-
mented by graph neural networks (GNNs). GNNs perform effectively in capturing the
complex relationship between a set of surface and atmospheric variables. A similar GNN-
based work is [Kei22]. Fuxi [CZZ+23] and Fengwu [CHG+23] also employ the “encode-
decode” strategy but with the transformer-based backbone. FourCastNet [PSH+22] ap-
plied Vision Transformer (ViT) and Adaptive Fourier Neural Operators (AFNO), while
ClimaX [NBK+23] also uses a ViT backbone but the trained model can be fine-tuned to
various downstream tasks. However, these models fall short in modeling the uncertainty
of weather evolution [JK22] even though perturbations are added to initial conditions
[BHQL24] and dropout methods [GG16] are used to mimic the uncertainty. Additionally,
ClimODE [VHG24] incorporated the physical knowledge and developed a continuous-
time neural advection PDE weather model.
56

Diffusion Models.Diffusion models [HJA20] have shown their strong capability in
computer vision tasks, including image generation [LYC+22], image editing [NDR+21],
semantic segmentation [BKC+22] and point cloud completion [LH21]. Conditional dif-
fusion models [HS22] were later proposed to make the generation step conditioned on the
current context or situation. However, not many efforts have adopted diffusion models
in global medium-range weather forecasting. More recent research has focused on pre-
cipitation nowcasting [AMP+23, GSH+24], and is localized in its predictions. GenCast
[PSGA+24] is a recently proposed conditional diffusion-based ensemble forecasting for
medium-range weather prediction. However, their conditioning is to directly use the ob-
servations from the recent past, which is shown to be insufficient (see the last case in the
ablation study). More related work can be found in Figure 2.7.
4.3 Preliminary Knowledge
4.3.1 Problem Formulation
Deterministic Global Weather Predictions.Given the input consisting of the weather
state(s),Xt∈RH×W×Cat timet, the problem is to predict a point-valued weather state,
Xt+∆t∈RH×W×Cat a future time pointt+ ∆t. HereH×Wrefers to the spatial resolu-
tion of the data, which depends on how densely we grid the globe over latitudes and lon-
gitudes,Crefers to the number of channels (i.e., weather variables), and the superscripts
tandt+ ∆trefer to the current and future time points. The long-range multiple-step
forecasts could be achieved by autoregressive modeling or direct predictions.
Probabilistic Global Weather Predictions.Unlike the deterministic models that out-
put point-valued predictions, probabilistic methods model the probability of future weather
state(s) as a distributionP(Xt+∆t|Xt), conditioned on the state(s) from the recent past.
57

Probabilistic predictions are appropriate for quantifying the forecast uncertainty and mak-
ing informed decisions.
Figure 4.1: Deterministic vs Probabilistic Models.
4.3.2 Denoising Diffusion Probabilistic Models
A denoising diffusion probabilistic model (DDPM) [HJA20] generates target samples by
learning a distributionp θ(x0)that approximates the target distributionq(x 0). DDPM
comprises aforward diffusionprocess and areverse denoisingprocess. Theforward
processtransforms an inputx 0with a data distribution ofq(x 0)to a Gaussian noise vector
xNinNdiffusion steps. It can be described as a Markov chain that gradually adds
Gaussian noise to the input according to a variance schedule{β 1, . . . , β N}:
q(xn|xn−1) =N
xn;p
1−β nxn−1, βnI
,and
q(x1:N|x0) =NY
n=1q(xn|xn−1), n∈[1, N],
where at each stepn, the diffused samplex nis obtained fromx n−1as described above.
Multiple steps of the forward process can be described as follows in a closed form:
q(xn|x0) =N 
xn;√¯αnx0,(1−¯α n)I
,
whereα n= 1−β nand¯α n=Qn
s=1αs. Thus,x n=√¯αnx0+√1−¯α nϵ, withϵsampled
fromN(0,I).
58

In thereverse process, thedenoisernetwork is used to recoverx 0by stepwise denois-
ing starting from the pure noise sample,x N. This process is defined as:
pθ(x0:N) =p(x N)NY
n=1pθ(xn−1|xn),(4.1)
wherep θ(xn)is the distribution at stepnparameterized byθ.
For each iteration,n∈[1, N], diffusion models are trained to minimize the following
KL-divergence between the true and generated descriptions:
Ln=D KL(q(x n−1|xn)||p θ(xn−1|xn)).(4.2)
4.4 Methodology
4.4.1 Method Overview
Figure 4.2 illustrates the overall framework of our proposed approach, CODICAST, de-
signed for global weather prediction using a conditional diffusion model. The core idea is
to reformulate the forecasting problem as a conditional generation task, where the model
learns to generate future states conditioned on recent past observations.
Given a sequence of past observationsXt−1:t∈RT×H×W×C, a pretrained encoder
first extracts spatiotemporal embeddings that serve as contextual conditioning informa-
tion. The diffusion process progressively corrupts the future target stateXt+1
0with noise
through a forward stochastic process, resulting in a highly noisy representationXt+1
N. The
denoising model, guided by both the noisy input and the encoded context, then iteratively
reverses the diffusion process to reconstruct the future weather state step-by-step. This
formulation enables the model to capture uncertainty and generate probabilistic forecasts
in a principled manner, leveraging the strengths of diffusion models while incorporating
weather temporal context.
59

Figure 4.2: Framework of our conditional diffusion model for global weather forecast
CODICAST. The superscriptTand the subscriptNdenote the time point and iteration
step of adding/denoising noise.HandWrepresent the height (#latitude) and width (#lon-
gitude) of grid data.Cis the number of variables of interest.Xis the observation data
and˜Zis the feature representation in the embedding space.
4.4.2 Forward Diffusion Process
At time pointt, the forward diffusion process assumes that a pure noise sampleXt+1
Nis
generated fromXt+1
0∈RH×W×Cby adding noiseNtimes (see the dotted lines in Figure
4.2):
Xt+1
n=√¯αn·Xt+1
0+√
1−¯α nϵ,(4.3)
whereϵis sampled fromN(0,I)with the same dimensions asXt+1
0, and¯αis as described
in Section 4.3.2).
4.4.3 Reverse Conditional Denoising Process
CODICASTmodels the probability distribution of the future weather state, conditioned
on the current and previous weather states. More specifically, we exploit a pre-trained
encoder to learn conditions as embedding representations of the past observationsXt−1
andXt, which are used to control and guide the synthesis process. Working in the latent
60

space of embeddings works better than the original space of the observations.
pθ(Xt+1
0:N|˜Zt−1:t) =p(Xt+1
N)NY
n=1pθ(Xt+1
n−1|Xt+1
n,˜Zt−1:t),(4.4)
whereXt+1
N∼ N(0,I), ˜Zt−1:tis the embedding representation as shown in Eq. (4.6).
After prediction at the first time point is obtained, a forecast trajectory,X1:T, of length
T, can be auto-regressively modeled by conditioning on the predicted “previous” states.
pθ(X1:T
0:N) =TY
t=1p(Xt
N)NY
n=1pθ(Xt
n−1|Xt
n,˜Zt−2:t−1).(4.5)
4.4.4 Pre-trained Encoder
We learn an encoder by training an autoencoder network [Bal12]. AnEncodertrans-
forms the input at each time point into a latent-space representation, whileDecoder
reconstructs the input from the latent representation. After the encoder,F, is trained, it
can serve as a pre-trained representation learning model to project the original data into a
latent embedding in Eq. (4.6).
˜Zt−1:t=F(Xt−1, Xt).(4.6)
Figure 4.3: Autoencoder structure.
4.4.5 Attention-based Denoiser Network
Our denoiser network consists of two blocks: cross-attention and U-net (as shown in
Figure 4.4). Cross-attention mechanism [HMT+22] is employed to capture how past
61

observations can contribute to the generation of future states. The embedding of past
observations, ˜Zt−1:t, and the noise dataXt+1
nat diffusion stepn, are projected to the same
hidden dimensiondwith the following transformation:
Q=W q·Xt+1
n, K=W k·˜Zt−1:t, V=W v·˜Zt−1:t,(4.7)
whereXt+1
n∈R(H×W)×Cand˜Zt−1:t∈R(H×W)×d z.W q∈Rd×C, Wk∈Rd×dz, Wv∈
Rd×dzare learnable projection matrices. Then we implement the cross-attention mecha-
nism byAttention(Q, K, V)=softmax(QKT
√
d)V.
Figure 4.4: Attention-based denoiser structure.
U-Net [RFB15] is utilized to recover the data by removing the noise added at each dif-
fusion step. Theskip connectiontechnique in U-Net concatenates feature maps from the
encoder to the corresponding decoder layers, allowing the network to retain fine-grained
information that might be lost during downsampling. The detailed U-Net architecture is
presented in Figure 4.5.
4.4.6 Training Process
The training procedure is shown in the arXiv version in Algorithm 2. Firstly, we pre-train
anencoderto learn the condition embedding of the past observations. Subsequently,
we inject it into our conditional diffusion model and train CODICASTwith the devised
62

Figure 4.5: Architecture of the U-Net model.
loss function:
Lcond(θ) =E X0,ϵ,nϵ−ϵ θ 
Xt+1
n, n,cond2,(4.8)
whereXt+1
n=√¯αnXt+1
0+√1−¯α nϵ,cond=F(Xt−1:t), andϵ θis the denoiser in
Figure 4.4.
Algorithm 2Pseudocode for Training Process
1:Input: Number of diffusion stepsN, pre-trained encoderF
2:Output: Trained denoising functionϵ(·)
3:repeat
4:Xt+1
0∼q(Xt+1
0)
5:n∼Uniform(1,2, . . . , N)
6:ϵ∼ N(0,I)
7: Get the past observationsXt−1, Xt
8: Get embedding ˜Zt−1:t=F(Xt−1, Xt)
9: Take gradient descent step on:
∇θϵ−ϵ θ
Xt+1
n, n,˜Zt−1:t2
10:untilconverged
63

4.4.7 Inference Process
We first extract the conditional embedding representations, ˜Zt−1:t, by the pre-trained en-
coder, and then randomly generate a noise vectorX N∼ N(0,I). The sampled noise
vector,X N, is autoregressively denoised along the reversed chain to predict the target
untilnequals 1, we obtain the weather prediction ˆX0at the timet+ 1. Later, multi-step
prediction can be implemented autoregressively - the output from the previous time step
is the input while predicting the next step, as shown in Eq. (4.5).
Algorithm 3Pseudocode for Inference Process
1:Input: Number of diffusion stepsN, pre-trained encoderF, trained denoising networkϵ(·),
past observationsXt−1, Xt
2:Output: Inference targetXt+1
0
3: Get embedding ˜Zt−1:t=F(Xt−1, Xt)
4:X N∼ N(0,I)
5:forn=N, . . . ,1do
6:ζ∼ N(0,I)ifn≥1,elseζ= 0
7:Xt+1
n−1=1√αn
Xt+1
n−1−αn√1−¯αnϵθ(Xt+1
n, n,˜Zt−1:t
+σnζ
8:end for
9:returnXt+1
0
4.4.8 Ensemble Forecast
To enhance the reliability of weather forecasts,ensemble forecaststrategy is often em-
ployed to capture the variability among forecasts by separately running multiple deter-
ministic models [Bui08]. CODICASTis a probabilistic model that can generate a distri-
bution of future weather scenarios rather than a single prediction. Following [PSGA+24],
we run the trained CODICASTmultiple times to get the ensembles. More specifically,
by integrating initial conditions and noise sampled from a Gaussian distribution, we im-
plement the ensemble forecast through multiple stochastic samplings during inference,
capturing a range of possible forecasts.
64

4.5 Experiments
4.5.1 Dataset
ERA5 [HBB+20] is a publicly available atmospheric reanalysis dataset provided by the
European Centre for Medium-Range Weather Forecasts (ECMWF). Following the ex-
isting work [VHG24], we use the preprocessed5.625◦resolution (32×64) and 6-hour
increment ERA5 dataset from WeatherBench [RDS+20]. We downloaded 5 variables for
the globe:
1. geopotential at 500 hPa pressure level (Z500),
2. atmospheric temperature at 850 hPa pressure level (T850),
3. ground temperature (T2m),
4. 10 meter U wind component (U10), and
5. 10 meter V wind component (V10).
Type Variable Abbrev. ECMWF ID Levels Range Unit
Single 2 metre temperatureT2m167[193.1,323.6]K
Single 10 metre U windU10165[−37.3,30.2]m/s
Single 10 metre V windV10166[−31.5,32.5]m/s
Atmospheric GeopotentialZ500129 500[43403.6,59196.9]m2/s2
Atmospheric TemperatureT850130 850[217.9,313.3]K
Table 4.1: Information on variables used in our experiments.
4.5.2 Baselines
Our experiments use the following baseline tools:
1. ClimODE [VHG24]: a spatiotemporal continuous-time model that incorporates the
physical knowledge of atmosphericadvectionover time;
65

2. ClimaX [NBK+23]: a state-of-the-art vision Transformer-based method trained on
the same dataset (without pre-training that is used in the original paper);
3. FourCastNet [PSH+22]: a global data-driven weather model using adaptive Fourier
neural operators;
4. Neural ODE [CRBD18]: an ODE network that learns the time derivatives as neural
networks by solving an ordinary differential equation;
5. Integrated Forecasting System IFS [RDS+20]: a global numerical weather predic-
tion (NWP) system, integrating multiple advanced physics-based models to deal
with more meteorological variables across multiple altitudes.
Our study focuses solely only on a subset of all the available variables due to our limited
computational resources, with IFS serving as the gold standard. For a fair comparison, all
ML models use the same data set described in Section 4.5.1.
4.5.3 Experimental Design
We use data from 2006 through 2015 as the training set, the data from 2016 as the vali-
dation set, and data from 2017 through 2018 as the test set. We assess the global weather
forecasting capabilities of our method CODICASTby predicting the weather at a future
timet+∆t(∆t= 6 to 144 hours) based on the past two time units. To quantify the uncer-
tainty in weather prediction, we generate an “ensemble” forecast by running CODICAST
five times during the inference phase.
Training.We first pretrain anencodermodel with theAutoencoderarchitec-
ture. For the diffusion model, we used U-Net as the denoiser network with 1000 diffu-
sion/denoising steps. The architecture is similar to that of DDPM [HJA20] work. We em-
ploy four U-Net units for both the downsampling and upsampling processes. Each U-Net
66

unit comprises two ResNet blocks and a convolutional up/downsampling block. Before
training, we apply Max-Min normalization [AFK+14] to scale the input data within the
range[0,1], mitigating potential biases stemming from varying scales [SMS+23].Adam
was used as the optimizer, where the learning rate= 2e−4, decay steps= 10000, decay
rate= 0.95. The batch size and number of epochs were set to 64 and 800, respectively.
More training details and model configurations are in the arXiv version.We conduct all
experiments on one NVIDIA A100 GPU with 80GB memory.
Evaluation Metrics.Following [VHG24], we use latitude-weighted Root Mean Square
Error (RMSE) and Anomaly Correlation Coefficient (ACC) as deterministic metrics. RMSE
measures the average difference between values predicted by a model and the actual val-
ues. ACC is the correlation between prediction anomalies relative to climatology and
ground truth anomalies relative to climatology. It is a critical metric in climate science to
evaluate the model’s performance in capturing unusual weather or climate events.
Following [VHG24], we assess the model performance using latitude-weighted Root
Mean Square Error (RMSE). RMSE measures the average difference between values pre-
dicted by a model and the actual values.
RMSE=1
MMX
m=1vuut1
H×WHX
h=1WX
w=1L(h)( ˜Xm,h,w−X m,h,w)2,
whereL(h) =1
Hcos(h)PH
h′cos(h′)is the latitude weight andMrepresents the number
of test samples.
Anomaly Correlation Coefficient (ACC) is the correlation between prediction anoma-
lies˜X′relative to climatology and ground truth anomalies ˆXrelative to climatology. ACC
is a critical metric in climate science to evaluate the model’s performance in capturing un-
67

usual weather or climate events.
ACC=P
m,h,w L(h) ˜X′
m,h,wX′m,h,wqP
m,h,w L(h) ˜X′2
m,h,w·P
m,h,w L(h)X′2
m,h,w,
where observed and forecasted anomaliesX′=X−C, ˜X′=˜X−C, and climatology
C=1
MP
mXis the temporal mean of the ground truth over the entire test set.
4.5.4 Reproducibility
We include all data and code in a GitHub repository2.
4.6 Results
4.6.1 Quantitative Evaluation
Accuracy.We compare different models in forecasting five primary meteorological
variables as described in Section 4.5.1. Tables 4.2 and 4.3 show CODICASTpresents
superior performance over other MLWP baselines across Latitude-weighted RMSE and
ACC results, demonstrating diffusion models can capture the weather dynamics and make
predictions accurately. However, the performance becomes poor as the lead time in-
creases, even though CODICASTstill outperforms other benchmarks.Error accumula-
tionis a hurdle to autoregressive forecasting methods. Furthermore, while we consider
the gold-standardIFSmodel as a reference model due to the different experimental set-
tings, we observed that there is still room to be improved for CODICASTin terms of
accuracy. Integrating more meteorological variables is a possibility to enhance its perfor-
mance further.
2Link:https://github.com/JimengShi/CoDiCast
68

VariableLead time
(Hours)RMSE (↓)
NODE ClimaX ForeCastNet ClimODE CoDiCast IFS
Z5006 300.6 247.5 222.7±18.1102.9±9.373.1±6.7 26.9
12 460.2 265.3 310.9±22.7134.8±12.3114.2±8.9 33.8
24 877.8 364.9 402.6±27.3193.4±16.3186.5±11.8 51.0
72 N/A 687.0 755.3±45.8478.7±48.5451.6±39.5 123.2
144 N/A 801.9 956.1±59.1783.6±37.3757.5±42.8 398.7
T8506 1.82 1.64 1.75±0.161.16±0.061.02±0.05 0.69
12 2.32 1.77 2.15±0.201.32±0.131.26±0.10 0.75
24 3.35 2.17 2.51±0.271.55±0.181.52±0.16 0.87
72 N/A 3.17 3.69±0.342.58±0.162.54±0.14 1.15
144 N/A 3.97 4.29±0.423.62±0.213.61±0.19 2.23
T2m6 2.72 2.02 2.05±0.181.21±0.090.95±0.07 0.69
12 3.16 2.26 2.49±0.211.45±0.101.21±0.07 0.77
24 3.86 2.37 2.78±0.261.40±0.091.45±0.07 1.02
72 N/A 2.87 3.77±0.322.75±0.492.39±0.37 1.26
144 N/A 3.38 4.39±0.413.30±0.233.45±0.22 1.78
U106 2.30 1.58 1.98±0.171.41±0.071.24±0.06 0.61
12 3.13 1.96 2.58±0.211.81±0.091.50±0.08 0.76
24 4.10 2.49 3.02±0.272.01±0.101.87±0.09 1.11
72 N/A 3.70 4.17±0.363.19±0.183.15±0.19 1.57
144 N/A 4.24 4.63±0.454.02±0.124.25±0.15 3.04
V106 2.58 1.60 2.16±0.191.53±0.081.30±0.06 0.61
12 3.19 1.97 2.73±0.231.81±0.121.56±0.09 0.79
24 4.07 2.48 3.15±0.282.04±0.101.94±0.14 1.33
72 N/A 3.80 4.26±0.343.30±0.223.18±0.19 1.67
144 N/A 4.42 4.64±0.454.24±0.104.21±0.18 3.26
Table 4.2: Latitude-weighted RMSE (↓) comparison. The results of NODE, ClimaX, and
ClimODE models are from the ClimODE paper [VHG24]. N/A represents the values
that are not available. ForeCastNet was re-trained with their code [PSH+22]. We employ
Monte Carlo Dropout [GG16] during the inference to compute the uncertainty. We mark
the scores inboldif CODICASTperforms the best among MLWP methods.
Uncertainty.The gray font was used to show the magnitude of the error in Table
4.2,representing the model uncertainty, with fluctuations remaining within 10% of the
ground truth scale. This is a measure of the robustness of the ML models. However,
ForeCastNetpredictions have relatively larger fluctuations due to the sensitive selec-
tion of the dropout rate. We present a case study of a 72-hour weather forecast generated
by CODICASTwith uncertainty quantification, as shown in Figure 4.6. The predicted
mean closely follows the overall trend of the ground truth, while the uncertainty increases
with lead time, consistent with the intuition that forecasts become less certain over longer
69

VariableLead time
(Hours)ACC (↑)
NODE ClimaX ForeCastNet ClimODE CoDiCast IFS
Z5006 0.96 0.97 0.97 0.990.99 1.00
12 0.88 0.96 0.95 0.990.99 0.99
24 0.70 0.93 0.92 0.980.98 0.99
72 N/A 0.73 0.75 0.880.92 0.98
144 N/A 0.58 0.64 0.610.78 0.86
T8506 0.94 0.94 0.94 0.970.99 0.99
12 0.85 0.93 0.92 0.960.99 0.99
24 0.72 0.90 0.89 0.950.97 0.99
72 N/A 0.76 0.77 0.850.93 0.96
144 N/A 0.69 0.71 0.770.85 0.81
T2m6 0.82 0.92 0.94 0.970.99 0.99
12 0.68 0.90 0.92 0.960.99 0.99
24 0.79 0.89 0.91 0.960.99 0.99
72 N/A 0.83 0.85 0.850.96 0.96
144 N/A 0.83 0.81 0.790.91 0.82
U106 0.85 0.92 0.87 0.910.95 0.98
12 0.70 0.88 0.78 0.890.93 0.98
24 0.50 0.80 0.71 0.870.89 0.97
72 N/A 0.45 0.41 0.660.71 0.94
144 N/A 0.30 0.28 0.350.42 0.72
V106 0.81 0.92 0.86 0.920.95 1.00
12 0.61 0.88 0.76 0.890.93 0.99
24 0.35 0.80 0.68 0.860.89 1.00
72 N/A 0.39 0.38 0.630.68 0.93
144 N/A 0.25 0.27 0.320.37 0.71
Table 4.3: ACC (↑) comparison. Other settings are the same as the Table 4.2.
horizons. Notably, most observed values lie within the 1σor 2σconfidence intervals,
demonstrating that the model provides reliable and well-calibrated predictions.
70

Figure 4.6: Model Forecasts with confidence intervals.
VariableLead Time
(Hours)Diffusion Step
250 500 750 1000 1500 2000
Z500 24 696.1 324.8 190.6186.5193.5 191.9
T850 24 3.88 2.38 1.531.521.56 1.58
T2m 24 5.26 2.79 1.631.441.50 1.53
U10 24 2.74 2.051.811.87 1.99 2.01
V10 24 2.43 2.111.891.94 2.04 2.06
Inference time (min) ∼1.1∼1.9∼2.8∼3.6∼6.5∼8.3
Table 4.4: RMSE with various diffusion steps at 24 hours lead time. We mark the lowest
scores inboldfont.
Inference efficiency.Generally, numerical weather prediction models (e.g.,IFS) re-
quire around 50 minutes for the medium-range global forecast, while deterministic ML
weather prediction models take less than 1 minute [RDS+20] but cannot model the weather
uncertainty. CODICASTneeds about 3.6 minutes (see the last row in Table 4.4) for the
global weather forecast, potentially balancing the efficiency and accuracy with essential
uncertainty quantification. The efficiency also depends on the model complexity. Table
4.5 presents a comparison of different methods. More details can be found in our survey
paper [SSJ+25].
71

Methods ∆x Train data Train resources Test data Inference time
Physics-based Models
IFS HRES [ECM] 0.1◦ERA5 2020 ∼52 mins
IFS ENS [ECM] 0.2◦ERA5 2020 –
Deterministic Predictive Models
Pangu-Weather [BXZ+23] 0.25◦ERA5 1979-2017 16 days; 192 V100 GPUs ERA5 2020 ∼secs; a GPU
GraphCast [LSGW+22] 0.25◦ERA5 1979-2019 4 weeks; 32 TPU v4 ERA5 2020 ∼min; a TPU
FuXi [CZZ+23] 0.25◦ERA5 1979-2015 8 days; 8 A100 GPUs ERA5 2020 ∼secs; a GPU
Fengwu [CHG+23] 0.25◦ERA5 1979-2017 17 days; 32 A100 GPUs ERA5 2020 ∼secs; a GPU
Stormer [NSB+23] 0.25◦ERA5 1979-2017 8 days; 8 A100 GPUs ERA5 2020 ∼secs; a GPU
HEAL-ViT [Ram24] 0.25◦ERA5 1979-2017 8 days; 8 A100 GPUs ERA5 2020 ∼secs; a GPU
GnnWeather [Kei22] 1◦ERA5 35 years 5.5 days; 1 A100 GPU ERA5 2020 ∼secs; a GPU
ArchesWeather[CSC+24] 1.5◦ERA5 1979-2018 9 days; 1 V100 GPU ERA5 2020 ∼secs; a GPU
NeuralGCM 0.7 [KYL+24] 0.7◦ERA5 1979-2017 3 weeks; 256 TPUs v5 ERA5 2020 ∼min; a TPU
NeuralGCM ENS [KYL+24] 1.4◦ERA5 1979-2017 10 days; 128 TPUs v5 ERA5 2020 ∼min; a TPU
Probabilistic Generative Models
GenCast [PSGA+24] 0.25◦ERA5 1979-2018 5 days; 32 TPUs v5 13 8 mins; a TPU
ArchesWeatherGen[CSC+24] 1.5◦ERA5 1979-2018 45 days; 1 V100 GPU 13 –
Foundation Models with Pre-training and Fine-tuning
Aurora [BBL+24] 0.1◦ERA5, CMIP6 2.5 weeks; 32 A100 GPUs HRES-T0 2022 –
Prithvi WxC[SRT+24] 0.5◦MERRA 1980-2019 –; 64 A100 GPUs MERRA 2020-2023 –
Table 4.5: Comparison of Predictive, Generative, and Foundation Models forglobal
weather prediction. The performance scores below are at the lead time of 6 days (except
Prithvi WxC at the lead time of 5 days). These scores are either from the WeatherBench
scoreboard or the original paper. “∆x” represents the horizontal resolution.
4.6.2 Qualitative Evaluations
In Figure 4.7, we qualitatively evaluate the performance of CODICASTon global fore-
casting tasks for all target variables,Z500,T850,T2m,U10andV10at the lead time
of6hours. The first row is the ground truth of the target variable, the second row is the
prediction and the last row is the difference between the model prediction and the ground
truth. From the scale of color bars, we can tell that the error percentage is less than 3%
for variablesZ500,T850, andT2m. However, error percentages over50%exist forU10
andV10even though only a few of them exist. We also observe that most higher errors
appear in the high-latitude ocean areas, probably due to the sparse data nearby.
72

Figure 4.7: Visualizations of true and predicted values of all five variables at 24 hours
lead time.
Figure 4.8: Visualization of true and predicted values across five meteorological variables
at 72 hours lead time.
4.6.3 Ablation Study
CODICASTincludes two important components:pre-trained encoderandcross atten-
tion. To study their effectiveness, we conduct an ablation study: (a)No-encoderdirectly
considers past observations as conditions to the diffusion model; (b)No-cross-attention
simply concatenates the embedding and the noisy sample at each denoising step; (c)No-
73

Figure 4.9: Visualizations of true and predicted values of all five variables at 144 hours
lead time.
encoder-cross-attentionconcatenates the past observations and the noisy sample at each
denoising step. Figure 4.10 shows that the full version of CODICASTconsistently out-
performs all other variants, verifying positive contributions of both components.
Figure 4.10: Ablation study.
4.6.4 Parameter Study
Table 4.4 shows that the accuracy improves as the number of diffusion steps,N, increases
whenN <1000, indicating that more intermediate steps are more effective in learning the
imperceptible attributes during the denoising process. However, when1000< N <2000,
the accuracy remains approximately flat but the inference time keeps increasing linearly.
74

Considering the trade-off between accuracy and efficiency, we finally setN= 1000.
Additionally, we use the same start and end variance value,β, as DDPM [HJA20] where
β∈[0.0001,0.02]. we study the effect of “linear” and “quadratic” variance scheduling
β, whereβ∈[0.0001,0.02]. Figure 4.11 shows that the “linear” variance scheduling
performs better than the “quadratic” one.
Figure 4.11: Effect of linear and quadratic variance scheduling methods.
4.7 Discussion
Diffusion models, originally designed for generative tasks, have recently shown promise
in predictive applications due to their transformative capabilities in conditional learning.
In the context of weather forecasting, diffusion models offer several unique advantages.
First, they inherently support probabilistic prediction by modeling a distribution over pos-
sible future states, making them particularly suitable for capturing the uncertainty of fore-
casts. Second, their iterative denoising process allows for fine-grained control over the
generation steps, which can be conditioned on recent observations to guide forecasts. In
this study, we adapted the diffusion framework to conditionally forecast global weather
with high fidelity and quantified uncertainty. This positions diffusion models as a power-
ful tool for weather forecasting.
Limitations.We used low-resolution global weather data in our work due to limited
computational resources and training time constraints. However, accurate weather fore-
75

casting—particularly for extreme events—typically requires high-resolution data, which
significantly increases training time. As a result, the time complexity of such models
warrants further investigation. Moreover, our CODICASTfocuses on global weather pre-
diction, without specifying non-extreme or extreme events. Therefore, the performance
on extreme events has not been studied yet, which motivates us to continue exploration
with a focus on extreme events.
4.8 Conclusions
In this chapter, we start with analyzing the limitations of current deterministic numer-
ical weather prediction (NWP) and machine-learning weather prediction (MLWP) ap-
proaches. They either require substantial computational costs or lack uncertainty quan-
tification in their forecasts. We propose a conditional diffusion model, CODICAST, to
address these limitations. which contains a conditionalpre-trained encoderand across-
attentioncomponent. Experimental results demonstrate it cansimultaneouslycomplete
more accurate predictions than existing MLWP-based models and a faster inference than
physics-based NWP models, while being capable of providing uncertainty quantification
compared to deterministic methods. In conclusion, our model CODICASTsimultaneously
achieves global weather prediction with high accuracy and efficiency while enabling cru-
cial uncertainty quantification.
76

CHAPTER 5
HYPERCUBE-RAG: HYPERCUBE RETRIEV AL-AUGMENTED GENERATION
FOR IN-DOMAIN SCIENTIFIC QUESTION-ANSWERING
The work presented in this chapter is highlighted in the following publication [SZJ+25,
SHT+26].
5.1 Background
Large language models (LLMs) often suffer from hallucinations and factual inaccura-
cies, especially forin-domainscientific question-answering. To address this phenomenon,
retrieval-augmented generation (RAG) has emerged as the de facto approach [HYM+25,
SXS+26], incorporating external domain knowledge to generate contextually relevant re-
sponses [HYM+25, SSX+26]. Despite recent advancements, they exhibit key limitations
in theme-specific applications on which the fine-grained topics are highly focused. For
example, retrieved documents are often semantically similar to the query, but overlook
specialized themes (e.g., terminology or nuanced contextual cues) prevalent in specific
literature or reports [DZXH24]. In addition, the retrieval process often suffers from in-
efficiency and limited transparency, which are critical in high-stakes domains, such as
environment [WLL+25], traffic [HMB+25], and healthcare [PPG23].
The conventionalsparse lexical retrievalranks a set of documents based on the ap-
pearance of query terms [Kad19]. Due to its reliance on the exact token overlap, it pre-
cisely captures specific themes or topics, and the retrieval process is efficient and in-
terpretable. However, it struggles to retrieve contextually related or paraphrased doc-
uments that require semantic understanding. To mitigate the above limitation,dense
embedding retrieverswere proposed by computing the similarity between vector em-
beddings of query-document pairs [KOM+20]. While contextual understanding is rel-
77

Case Study
Query:How much rainfall did Melbourne Beach, Florida receive from Tropical Storm Fay?
Hypercube-RAG
Answer:25.28 inches✓
Retrieved Docs: [565, 246, 534]
Doc 565:. . . Fay took her time going northward and dumped tremendous amounts of rain
along the way Melbourne Beach, Fl. , received as much as 25.28 inches of rain . . .
Semantic-RAG
Answer:The documents do not provide information about the rainfall that Melbourne
Beach, Florida received from Tropical Storm Fay.
Retrieved Docs: [451, 186, 364]
Doc 451:. . . In Florida , 51 percent of the state was in severe to extreme drought by the
end of 2010. . . . Gainesville received only 12.95 inches of precipitation , compared to the
previous record low of 15.25 inches . . .
Figure 5.1: Hypecube- vs semantic RAG: A case study
atively improved, the hurdle of missing specific themes remains, causing semantically
similar but off-topic retrievals [KRI21, KAJ+24]. Figure 5.1 shows an example where
the Semantic-RAG returns documents related to precipitation in Florida, but overlooks
the critical location-specific information for‘‘Melbourne Beach’’. Moreover, re-
trieval in an embedding space makes it challenging to interpret why certain documents
were selected [JSR+19]. Additionally, agraph-basedRAG represents documents as
graph structures, where nodes correspond to entities or concepts and edges capture their
relationships [PZL+24]. They offer explanations by outputting the traversed subgraph.
However, traversing such subgraphs introduces a significant risk of information overload,
as the expanding neighborhoods may include substantial amounts of irrelevant themes
[WHC+24]. Graph traversal also poses significant scalability bottlenecks due to its inef-
ficient computational complexity ofO(V+E), whereVandEare the number of nodes
and edges in the traversed subgraph [DN21].
78

Hypercube RAG
5/16/2025 1Melbourne Beach
Miami Downtown
Palm BeachLocationRain
ClimateHeat
LocationEventTheme
Tropical 
Storm FayRainFigure 5.2: Graph vs. Hypercube.
The methods discussed above tend toperform poorly in at least one of the aspects
among accuracy, efficiency, and explainability.This motivates us to develop a RAG
system that overcomes this weakness. In this pursuit, we identified thetext cubeas a
promising technique [TZC+18, WJH+23]. A text cube is an inherently explainable mul-
tidimensional structure that allocates documents into cubes along various human-defined
dimensions, such as location, date, event, and theme. Each dimension is populated with
fine-grained labels (e.g.,‘Miami Downtown’,‘2021’,‘Hurricane Irma’,‘flooding’) ex-
tracted from documents. Relevant documents can be easily indexed by these labels rep-
resented in cubes. Furthermore, the explicit theme dimension makes the text cube partic-
ularly well-suited fortheme-specificapplications where users [TZC+18] primarily focus
onfine-grainedinquiries, e.g., where, when, and which hurricane events hit or how much
rainfall hurricanes result in certain areas.
In this work, we implement a multi-dimensional (cube) structure calledHypercube,
which indexes documents to cubes based on fine-grained labels in a human-defined, multi-
dimensional space. Building on this structure, we proposeHypercube-RAG, a novel
RAG framework to enhance the retrieval process by retrieving relevant documents from
the corresponding cube cells.Hypercube-RAGexhibits three characteristics simulta-
neously: (1)Efficiency. The retrieval process quickly narrows down the search to the right
79

cube cell(s) based on the labels with the constant timeO(c), while takingO(κ)to fetchκ
documents assigned in that cell. (2)Explainability. Hypercube is inherently explainable
as fine-grained labels in cubes represent the compact information in documents (see Fig-
ure 5.2, right and Figure 5.4); (3)Accuracy. The cube label-based search supports both
sparse and dense embedding strategies, effective for capturing uncommon thematic termi-
nology and semantically relevant information (Section 5.3.3). Overall, our contributions
in this chapter are summarized as follows:
1. We identify the shortcomings of conventional RAG methods, especially fortheme-
specificquestion-answering in scientific domains.
2. We proposeHypercube-RAG, a simple yet accurate, efficient, and explainable RAG
method.
3. We conduct experiments on three datasets, demonstrating that ourHypercube-RAG
outperforms other baseline methods in accuracy and efficiency when it comes toin-
domainquestion-answering.
5.2 Related Work
5.2.1 Text-based RAG
BM25 [RW94, TPB14] is a classical sparse retrieval method based on TF-IDF princi-
ples, widely used in information retrieval. It ranks documents by scoring query term
matches, with adjustments for term frequency saturation and document length. While
efficient and interpretable, BM25 lacks semantic understanding and often misses para-
phrased or contextually related documents without exact lexical overlap, limiting its ef-
fectiveness in tasks requiring deeper language comprehension. Dense embedding retriev-
ers that compute contextual similarity between the given query and a document/chunk in
80

the embedding space [JXG+23]. Common retrievers consist of DPR [KOM+20], Con-
triever [ICH+22], e5 [WYW+22], and ANCE [XXL+21]. By capturing deeper semantic
relationships, these methods improve performance in open-domain question answering
and generative tasks, especially when surface-form matching fails. However, dense re-
trieval can return semantically close yet topically irrelevant documents, introducing noise
and hallucinations. Moreover, its opaque nature also hinders interpretability [JSR+19],
limiting adoption in high-stakes or scientific domains. Recent efforts aim to mitigate these
limitations by incorporating hybrid retrieval techniques [MOY21] and domain-adaptive
retrievers [LCT21].
5.2.2 Structured RAG
Graph-based RAG methods enhance the standard RAG paradigm by introducing struc-
tured knowledge representations, such as entity or document graphs. GraphRAG [ETC+24]
derives an entity knowledge graph from the source documents and gathers summaries for
all groups of closely related entities. LightRAG [GXY+24] employs a dual-level retrieval
system that enhances comprehensive information retrieval from both low-level and high-
level knowledge discovery. HippoRAG [GSG+24] is a knowledge graph-based retrieval
framework inspired by the hippocampal indexing theory of human long-term memory.
HippoRAG 2 [GSQ+25] builds on top of HippoRAG and enhances it with deeper passage
integration and more effective online use of an LLM. SELF-RAG [AWW+23] enhances
an LLM’s quality and factuality through retrieval and self-reflection. RA-DIT [LCC+23]
fine-tunes the pre-trained LLMs or retrievers to incorporate more up-to-date and rele-
vant knowledge. Despite these improvements, graph construction and reasoning can be
computationally intensive due to the massive nodes and edges.
81

Text Cube.[WJH+23] designs the text cube to structure spatial-related data across
geospatial dimensions, thematic categories, and diverse application semantics. STREAM-
CUBE implements the text cube to incorporate both spatial and temporal hierarchies [FZZ+15].
Doc2Cube [TZC+18] automates the allocation of documents into a text cube to support
multidimensional text analytics. Despite these elaborate designs, the text cube has not yet
been explored in the context of retrieval-augmented generation (RAG) systems.
5.3 Methodology
Figure 5.3 illustrates theHypercube-RAGframework through a scientific question-
answering example. In the following, we define the hypercube structure (Section 5.3.1),
describe the process for constructing the hypercube (Section 5.3.2), explain how two
sparse and dense embedding retrieval strategies are combined inside of hypercube (Sec-
tion 5.3.3), and present how to rank documents retrieved using hypercube (Section 5.3.4).
5.3.1 Hypercube Formulation
Given a text corpus withkdocuments:D, the hypercube is designed as a multidimen-
sional data structure,C=L 1⊗ L 2⊗ ··· ⊗ L M, whereMis the number of hypercube
dimensions, andL iis the set of labels used for the category chosen along theithdimen-
sion. For any documentd∈ D, a text classification algorithm (e.g, TeleClass [ZYX+25])
will assign it to one or more cube cells. This process is equivalent to assigning one or
moreN-dimensional labels (l t1, lt2, . . . , l tN) for a document,d, where labell tj∈ L jrep-
resents the categorization ofdalong thejthdimension from the set of labels,L j.
82

Hypercube RAG
5/20/2025 1Melbourne Beach
Miami Downtown
Palm BeachLocation
Prompt: Given a query, 
return information in terms of 
['location', 'theme', 'person', 
'event', 'organization', 'date']
Rain
heat{𝑑𝑖,…,𝑑𝑗}LLM Constructed Knowledge HyperCubeQuery: How much rainfall did Melbourne Beach, Florida 
receive from Tropical Storm Fay?Answer: 25.28 inches
2 3
{
       Rainfall , 
       Melbourne Beach, Florida,
       Tropical Storm Fay
  }45
... Fay took her 
time going 
northward and 
dumped
tremendous 
amounts of rain  
along the way
Melbourne Beach, 
Fl. , received as 
much as 25.28
inches of rain …Decomposition RetrievalAugmentationGeneration4
1
climateFigure 5.3: The illustration ofHypercube-RAGframework. 1Input:Input the query
and the prompt into a LLM; 2Decomposition:LLM decomposes the query into differ-
ent dimensions; 3Retrieval:according to these dimensions, we useHypercube-RAG
to retrieve relevant documents; 4Augmentation:query is augmented with retrieved
documents (ranked already); 5Generation:LLM output withHypercube-RAG.
Hypercube RAG
5/16/2025 1Melbourne Beach
Miami Downtown
Palm BeachLocation
Rain
heat
climate
1
IDDocument content
1... Tropical  Storm  Fay took  
her time  going  northward  
and dumped  tremendous  
amounts  of rain along  the 
way Melbourne  Beach,  Fl. 
, received  as much  as 
25.28 inches  …
2…CorpusCell: < Melbourne Beach, Tropical Storm Fay, Rain >
Figure 5.4: Hypercube construction on a corpus. We present only three dimensions for
clear representation.
83

5.3.2 Hypercube Construction
For most environmental applications,theme, location, date, event, and/orpersonare of-
ten of high interest. The core idea of the hypercube is to use human-identified entities
or topics contained in the corpus and index documents in cubes with the appropriate di-
mensions, thereby facilitating the subsequent retrieval. In this work, we employ multiple
methods to construct a holistic and precise knowledge hypercube.
Name Entity Recognition (NER).We first pre-define hypercube dimensionsLOCATION,
DATE,EVENT,PERSONandORGANIZATION. For each dimension, we utilize the pre-
trained language models (PLMs) [DCLT19b, LOG+19] to extract salient entities from
documents:E=F(d), whereE(d) ={e 1, e2, . . . , e l(d)}is the set ofl(d)named enti-
ties identified in the document,d, andFrefers to a PLM. For instance, those entities
belonging to theEVENTdimension are represented:
EEVENT={e∈ E(d)|dim(e) =EVENT}.(5.1)
Theme-specific Extraction.While the above entities can be captured by NER mod-
els, abstract and intricate theme-specific information may be missed, such as ‘hurricane
path’, ‘rainfall intensity’, and ‘climate change’. Therefore, we add another hypercube di-
mensionTHEME, such key theme-specific phrases can be extracted. We implement it with
KeyBERT1, a language model pre-trained on large-scale text corpora, which demonstrates
strong performance in capturing general topical relevance.
Using the methods described above, we extract fine-grained entities, topics, and phrases,
and assign them to different hypercube dimensions, includingLOCATION,DATE,EVENT,
ORGANIZATION,PERSON, andTHEME. To enhance coverage ofin-domainsemantics,
1https://github.com/MaartenGr/KeyBERT?tab=readme-ov-file
84

we incorporate human-in-the-loop curation to expand theTHEMEdimension, ensuring it
captures a comprehensive set of relevant topics.
5.3.3 Hypercube Retrieval
Aconstructedknowledge hypercube can serve for the retrieval process based on itsfine-
grained labels. Given a queryq, an LLM first decomposes it into a set of entities and
topics,E(q) ={e 1, e2, . . . , e l(q)}, aligned with various dimensions (see the prompt in
Figure 5.3), then retrieval inside of hypercube is performed by matching these decom-
posed components with cube labels. Our hypercube supports two matching strategies:
sparse exact lexical matchanddense embedding match. We prioritize leveraging the ex-
act matching strategy due to its high precision, scoring documents based on the frequency
with which query entities appear within them:
score(d, q) =X
ej∈E(q)1[ej∈E(d)],(5.2)
where1is an indicator function that equals 1 if an entity of the query exactly matches a
term in the document, and 0 otherwise.
Considering the example in Figure 5.3, the last two components from the query,Mel-
bourne BeachandTropical Storm Fay, are handled using the exact matching strategy,
as they correspond precisely to fine-grained labels in the hypercube. However, the first
component,Rainfall, does not exactly match any label(s) in the hypercube, even though
its matched label should beRain.
To address such cases,Hypercube-RAGsupports semantic retrieval by computing
the similarity between query components and fine-grained cube labels. When the simi-
larity score exceeds a predefined thresholdτ, semantic retrieval is triggered. Specifically,
both the query entities and cube labels are projected into an embedding space,Z, with an
85

encoder. It can be represented as:
Z(q) =Encoder(E(q)),(5.3)
Z(d) =Encoder(E(d)),(5.4)
score(d, q) =sim(Z(q),Z(d)),(5.5)
where sim(·)refers to the similarity function.
5.3.4 Hypercube Ranking
The retrieved documents need to be provided to the LLM as contextual input. However,
long-context RAG methods do not always improve the quality of LLM responses, since
they may introduce irrelevant or noisy information [LLH+24, JYHA25]. This raises a
critical challenge: how to precisely filter and select theme-relevant documents. To address
this, we prioritize returning those documents that fully cover key components derived
from the query:
Dreturn ={d i∈ D | E(q)⊆ E(d i)}.(5.6)
If no documents fully cover all query components (i.e.,D return =∅), we return the next
best set of documents with the highest partial coverage:
D∗
return= arg max
di∈D|E(d i)|.(5.7)
Taking the example in Figure 5.3,Hypercube-RAGprioritizes returning Doc.AandB
since they cover all three keywords (see the following table). In cases where Doc.Aand
Bdo not exist, then it returns Doc.C, which contains the next best subset of information
based on the query decomposition.
86

Doc. Covered Information
A ‘Melbourne Beach’, ‘Tropical Storm Fay’, ‘Rainfall’
B ‘Melbourne Beach’, ‘Tropical Storm Fay’, ‘Rainfall’
C ‘Tropical Storm Fay’, ‘Rainfall’
D ‘Tropical Storm Fay’
Table 5.1: An example to show the hypercube ranking algorithm.
5.4 Experiment
5.4.1 Datasets
In this work, we focus on studyingHypercube-RAGwithin the context of asinglehy-
percube configuration. While more general domains may benefit from modeling multiple
hypercubes, our evaluation is centered on one scientific domain to enable a focused and
controlled analysis. We also provide a more detailed discussion in the Discussion section
(Section 5.6.2). We collected the text corpus from a subset of the Science Daily Climate
Change dataset (SciDCC) [MM21], which was created by web scraping from the “Ge-
ography” and “Climate” topics in the environmental science section of the Science Daily
website2. We created an aging dam corpus by collecting news articles related to dam fail-
ures in the United States from Google News3. Then, an LLM (GPT-4o) is employed to
generate 300, 300, and 186 QA pairs, respectively. Each pair was manually validated to
ensure that the corresponding question could be answered using information contained in
the corpus. To ensure representativeness, each dataset generates three types of QA pairs:
(a) long-form, free-style; (b) short-term factual; and (c) quantitatively factual. We present
the summary in Table 5.2 and QA example of each type are as follows.
2https://www.sciencedaily.com/
3https://news.google.com/
87

Dataset # Doc. Doc. Length # QA Ques. Length Ans. Length
Hurricane 844 52∼1781 300 6∼27 2∼119
Geography 432 22∼1646 300 5∼34 2∼79
Aging Dam 186 25∼4782 186 6∼33 2∼150
Table 5.2: Datasets. Length denotes the number of words.
Three types of QA pairs
Long-form, free-style
Question: How does El Ni ˜no Modoki differ from traditional El Ni ˜no events in terms of
hurricane activity?
Answer:El Ni ˜no Modoki, a new type of El Ni ˜no that forms in the Central Pacific, is
associated with a higher storm frequency and a greater potential for making landfall along
the Gulf coast and the coast of Central America. This is different from traditional El Ni ˜no
events, which are more difficult to forecast and can result in diminished hurricanes in the
Atlantic. El Ni ˜no Modoki is more predictable, potentially providing greater warning of
hurricanes by a number of months. The exact cause of the shift from traditional El Ni ˜no
to El Ni ˜no Modoki is not yet clear, but it could be due to natural oscillations or El Ni ˜no’s
response to a warming atmosphere.
Short-term factual
Question: Which dam failure in northwestern Wisconsin triggered flooding near the Min-
nesota border?
Answer:Radigan Flowage Dam.
Quantitative factual
Question: How many named storms were observed in the Atlantic during the 1997
hurricane season?
Answer:7 named storms.
5.4.2 Baselines
We select three types of methods for comparison: 1)sparse retrieverBM25 [RW94]; 2)
dense embedding retrievalmethods, Contriever [ICH+22], e5 [WYH+24], Nvidia/NV-
Embedv2 [LRX+25]; 3) graph-based methods, GraphRAG [ETC+24], LightRAG [GXY+24],
88

and HippoRAG [GSG+24] and HippoRAG 2 [GSQ+25]. For all graph-based baselines,
we conducted experiments by running their official GitHub repositories. For dense em-
bedding retrievers, we run these models from the Hugging Face. For sparse retriever
BM25, we implement it using a GitHub repository4. All experiments were conducted on
an A100 GPU with 80 GB memory.
In Table 5.3, we also include baseline methods without RAG, such as DeepSeek-R15,
Qwen2.5-7B-Instruct6, Llama-3.3-70B-Instruct7, and Llama-4-Scout-17B-16E-Instruct8.
Note that DeepSeek-R1 is a reasoning model, we set 0.6 and 1 as the temperature by
following their recommendations. Since its output also includes the reasoning/thinking
process, we exclude it and consider the final output as the answer. The temperatures for
other models are set to 0.
5.4.3 Evaluation Metrics
Following the work [CUC+25], we evaluate the quality of LLMs’ responses using au-
tomated metrics derived from token-level representations, including BLEU [PRWZ02]
and BertScore [ZKW+20] to provide the standard assessment of response quality. BLEU
scores and BERTScores are computed with the NLTK9and SentenceTransformer10Python
packages, respectively. Since the settings are open-text responses, we also employ LLM-
as-a-Judge that assesses correctness and completeness. All LLM-based evaluations use
GPT-4o as the judge. The prompts used are in Figure 5.5.
4https://github.com/dorianbrown/rank_bm25
5https://github.com/deepseek-ai/DeepSeek-R1
6https://qwenlm.github.io/blog/qwen2.5/
7https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
8https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
9https://www.nltk.org/
10https://sbert.net/
89

Prompt template for LLM-as-a-judge
You are an expert evaluator on natural sciences, such as hurricanes, geography, and aging
dams. Your task is to assess the predicted answer generated by AI models compared to
the gold (reference) answer. Please evaluate the predicted answer on correctness and
completeness.
Question:{question},
Gold Answer:{gold answer},
Predicted Answer:{predicted answer}.
Evaluate criteria:
1. Correctness (0 or 1):
- Score 1 if the predicted answer is generally accurate and aligns with the key points in the
reference answer.
- Score 0 if it has factual errors or misrepresents key information.
2. Completeness (0 or 1):
- Score 1 if the predicted answer covers the main points present in the reference answer.
- Score 0 if it misses essential information or fails to address the core of the question.
Output your evaluation in the following JSON format:{“correctness”: int, “completeness”:
int, “explanation”: your brief explanation}
Figure 5.5: Prompt template for LLM-as-a-judge.
5.5 Results
In this section, we show and analyze the experimental results for accuracy, efficiency, and
explainability of ourHypercube-RAGand other baseline methods.
5.5.1 Accuracy
We compare ourHypercube-RAGwith semantic RAG methods, graph-based RAG
methods, and LLMs without retrieval. Table 5.3 reveals three key findings. First, our
method consistently outperforms other baselines across three datasets, demonstrating its
effectiveness in enhancing the capabilities of LLMs for in-domain scientific question-
90

answering. Compared to the second-best method,Hypercube-RAGimproves the ac-
curacy by3.7%measured as relative gains across three of four metrics used. Second, all
RAG methods perform much better than direct inference of LLMs without retrieval, as
expected, showing the effectiveness of incorporating external knowledge forin-domain
QA tasks. Finally, we also evaluateHypercube-RAGagainst a range of baseline meth-
ods across multiple LLMs developed by different companies. As illustrated in Figure 5.6,
our method achieves superior performance across all models regardless of the underlying
LLM architectures.
MethodHurricane Geography Aging Dam
Automatic Metric LLM-as-Judge Automatic Metric LLM-as-Judge Automatic Metric LLM-as-Judge
BLEU BERTScore Correct Complete BLEU BERTScore Correct Complete BLEU BERTScore Correct Complete
No Retrieval
GPT-4o 2.6 66.8 57.1 23.5 2.8 63.9 64.2 42.1 5.5 62.5 56.9 19.6
DeepSeek-R1 2.2 69.2 53.4 28.5 1.3 61.7 57.9 45.7 8.7 71.6 35.4 18.8
Qwen2.5-7B-Instruct 1.2 46.9 23.5 6.9 1.7 47.1 32.1 12.6 1.9 43.4 15.5 6.5
Llama-3.3-70B-Instruct 2.0 60.2 39.8 14.6 1.6 51.3 49.3 21.2 2.4 46.7 39.7 17.1
Llama-4-Scout-17B-16E-Instruct 2.3 62.7 53.9 39.1 3.1 61.3 48.8 21.9 5.2 64.2 34.4 15.1
Sparse and Dense Embedding Retrieval
BM25 [RW94] 10.5 77.5 74.3 60.8 7.9 73.3 79.7 70.8 18.1 80.2 85.7 75.9
Contriever [ICH+21] 11.3 78.6 75.7 60.6 7.5 74.1 82.5 73.6 16.7 76.8 70.9 60.9
e5 [WYH+24] 11.9 79.1 78.3 63.9 7.1 73.9 81.2 71.2 19.7 79.888.975.5
NV-Embed [LRX+25] 12.2 79.1 76.9 59.6 7.5 75.4 81.9 72.3 18.8 79.6 79.4 73.6
Graph-based RAG
LightRAG [GXY+24] 7.6 72.2 64.1 47.6 7.0 67.8 69.8 59.6 12.8 75.4 65.9 46.5
GraphRAG [ETC+24] 6.5 72.8 68.8 50.9 6.9 69.4 70.6 63.8 11.8 72.4 72.5 58.6
HippoRAG [GSG+24] 6.8 73.6 70.5 48.8 7.8 70.3 76.9 69.7 12.4 73.8 80.9 58.6
HippoRAG 2 [GSQ+25] 8.6 76.386.753.5 9.2 74.6 80.6 67.5 11.5 74.1 88.2 53.7
Our Method
Hypercube-RAG 13.2 82.580.6 66.2 9.3 76.8 83.6 74.8 21.5 82.686.977.8
Table 5.3: Performance Comparison (%) of LLMs themselves without RAG, LLMs with
various RAG baselines and Hypercube-RAG (ours). The best scores are inboldwhile the
second-best scores are highlighted with underline . All RAG methods were experimented
with GPT-4o as the base.
5.5.2 Efficiency
We study the response speed of diverse RAG methods over the corpus size and report the
retrieval time for one query in Table 5.4. We break down our analysis as follows.
Within corpus sizek.BM25 achieves the fastest retrieval among all evaluated meth-
ods without surprise. In contrast, semantic and graph-based RAGs incur significantly
91

(a) Hurricane
 (b) Geography
Figure 5.6: Performance comparison withvariousLLMs (only the range between 50%
and 90% shown).
higher retrieval costs. This is particularly pronounced for graph-based methods due to the
computational burden of search paths in large-scale graphs, posing scalability challenges.
Notably, ourHypercube-RAGsubstantially reduces the retrieval time by one to two
orders of magnitude compared to both semantic and graph-based methods, underscoring
the effectiveness of the hypercube structure in optimizing retrieval efficiency. We attribute
this to theretrieval with compact cube labelsintroduced in Section 5.3.3. Although it is
slightly slower than BM25, the marginal increase in retrieval time is a worthwhile trade-
off for significantly improved accuracy, as shown in Table 5.3.
Beyond corpus sizek.To mimic the real-world scenario that the external knowledge
database is usually large and noisy, we expand the hurricane corpus from 844 to 11,539
documents (around 14 times). The added documents are related to other topics, such
as “Pollution” or “Ozone Holes”, introducing additional noise.Hypercube-RAGmain-
tains the lowest retrieval time per query (see the last column in Table 5.4), highlighting the
efficiency of the cube label-based retrieval mechanism. This is likely because most noisy
documents result in empty or sparsely populated cube cells, as they lack corresponding
in-domainfine-grained labels pre-defined in our hypercube. As a result, despite a signifi-
cantly expanded corpus, the hypercube can efficiently bypass these irrelevant documents.
In contrast, other baseline methods exhibit greater sensitivity to corpus size, resulting in
92

increased retrieval times.
Methods k/8k/4k/2k 14k
BM25 0.5 0.9 1.6 3.2 51.1
e5 14.4 17.4 19.1 37.8 82.4
GraphRAG 114.7 348.2 1179.4 5260.6 45135.7
Hypercube-RAG 0.7 1.5 3.3 7.1 22.1
Table 5.4: Retrieval time (ms) vs. corpus size (k), experimenting on the Hurricane data
set (LLM base: GPT-4o).
In addition, Table 5.5 shows that the accuracy of the evaluated methods drops due to the
inclusion of these noisy off-topic documents. However,Hypercube-RAGstill outper-
forms others, demonstrating robust resilience to noisy data.
Methods k14k BERTScore (↓)
BM25 77.5 77.2 0.3
e5 79.1 76.3 2.8
GraphRAG 72.8 70.5 2.3
Hypercube-RAG 82.5 80.3 1.9
Table 5.5: Accuracy (BERTScore) vs corpus size.
5.5.3 Explainability
The retrieval process with our hypercube is inherently explainable through its associated
cube labels. Considering the query“How much rainfall did Melbourne Beach, Florida
receive from Tropical Storm Fay?”, illustrated in Figure 5.3,Hypercube-RAGreturns
relevant documents by matching the decomposed components with fine-grained labels,
‘Melbourne Beach’,‘Tropical Storm Fay’, and‘rainfall’, along hy-
percube dimensions:LOCATION,EVENT, andTHEME. Table 5.6 shows that three re-
trieved documents (Doc.565,246,535), which are represented with fine-grained cube
labels. These cube labels clearly explain why those documents are returned. For example,
93

Doc.565is ranked the first to return, since it contains all three key components of the
query. The detailed document contents are included in Table 5.7.
Doc. Location Event Theme
565 ‘Melbourne Beach’: 1 ‘Tropical Storm Fay’: 1 ‘Rain’: 5
246 ‘Florida’: 1 ‘Tropical Storm Fay’: 1 –
535 ‘Florida’: 1 – –
Table 5.6: Documents represented in the hypercube.
94

Document ID Document Content
565 . . . Why is September the peak month for hurricanes? NASA oceanogra-
pher, Bill Patzert at the Jet Propulsion Laboratory, Pasadena, Calif., pro-
vided the answer: “Hurricanes are fueled by warm ocean temperatures
and September is the end of the Northern Hemisphere ocean warming
season. The 2008 Atlantic hurricane season started early with the for-
mation of Tropical Storm Arthur on May 30, from the remnants of the
eastern Pacific Ocean’s first storm, Alma, which crossed Central America
and reformed in the Gulf of Mexico. . . . The tropical Atlantic is warm,
but not unusually so.” Once a powerful Category 3 hurricane, now a trop-
ical depression, Gustav moved from northwest Louisiana into northeast-
ern Texas and into Arkansas by Sept. 3. Like Tropical Storm Fay in
August, Gustav’s legacy will lie in large rainfall totals. According to the
National Hurricane Center discussion on Sept. 2, ”Storm total rainfalls
are expected to be five to ten Inches with isolated maximums of 15 inches
over portions of Louisiana, Arkansas and Mississippi. Rainfall amounts
of 4-8 inches have been already reported in parts of Alabama, Mississippi
and Louisiana. . . . In August, Fay’s ten-day romp from the U.S. South-
east northward up the Appalachian Mountains seemed like a harbinger
for September’s storms. Fay took her time going northward and dumped
tremendous amounts of rain along the way. Melbourne Beach, Florida,
received as much as 25.28 inches of rain. Other cities in various states
reported high totals: Thomasville, Ga., reported 17.43 inches; Camden,
Ala., received 6.85 inches; Beaufort, S.C., received 6.11 inches; Carthage,
Tenn., reported 5.30 inches, and Charlotte, N.C., reported 5.90 inches . . .
246 . . . Among the already apparent evidence: Dunes that historically pro-
tected Kennedy Space Center from high seas even during the worst storms
were leveled during Tropical Storm Fay in 2008, Hurricane Irene in 2011
and Hurricane Sandy in 2012. A stretch of beachfront railroad track built
by NASA in the early 1960s that runs parallel to the shoreline has been
topped by waves repeatedly during recent storms. . . . The problem had
been occurring for years but seemed to be growing worse, beginning with
the spate of hurricanes that struck Florida in 2004. Jaeger said he, Adams
and doctoral students Shaun Kline and Rich Mackenzie determined the
cause was a gap in a near-shore sandbar . . .
535 . . . Our theory would suggest that seepage caused by underwater flow will
continue to erode and weaken the levee system around New Orleans, but
the rate of this erosion should gradually slow with time, says Straub.
Hopefully this research will aid the U.S. Army Corps of Engineers in
identifying levees that need repair and assessing the lifespan of structures
like the MRGO that are not planned for upkeep. Using fieldwork con-
ducted in the Florida Panhandle, Straub and his fellow researchers were
able to better understand the process of seepage erosion, which occurs
when the re-emergence of groundwater at the surface shapes the Earth’s
topography . . .
Table 5.7: Document contents.
95

5.5.4 Ablation Study
As shown in Section 5.3.3,Hypercube-RAGcombines two retrieval strategies: sparse
exact matching and dense embedding retrieval. Section 5.3.4 introduces the document
ranking algorithm used to filter and prioritize the retrieved results. To study their effective-
ness, we conduct an ablation study by removing each of them fromHypercube-RAG,
which are represented asNo-Sparse,No-Dense, andNo-Ranking, respectively. Figure
5.7 shows that the full version ofHypercube-RAGconsistently outperforms all other
variants, verifying positive contributions of each constituent. More specifically, it justi-
fies the benefit of coupling these two strategies for in-domain theme-specific question-
answering tasks, and the necessity of the ranking algorithm to filter possible irrelevant
documents (“noisy”).
Figure 5.7: Ablation study on constituents ofHypercube-RAG.
We also conduct an ablation study on the hypercube dimensions:Location,Event,
Date,Organization,Person, andTheme. To study their effectiveness, we con-
duct an ablation study by removing each of the dimensions, which are represented as
No-Location,No-Event,No-Date,No-Organization,No-Person, andNo-Theme,
respectively. Table 5.8 shows that the full version ofHypercube-RAGconsistently
outperforms all other variants, verifying positive contributions of each dimension. More
96

specifically, we observe that the performance drops the most when removing theLocation
andThemedimensions, justifying the high importance of these two hypercube dimen-
sions for in-domain theme-specific question-answering.
Dimension Hurricane Geography Aging Dam
No-Location 78.7 65.9 77.5
No-Event 79.6 75.9 80.0
No-Date 79.7 74.5 80.4
No-Organization 79.6 74.6 80.2
No-Person 80.2 73.2 81.4
No-Theme 77.9 71.7 81.1
FULL 82.5 76.8 82.6
Table 5.8: Ablation study on hypercube dimensions. The values are BERTScores (%).
5.5.5 Parameter Study
Hypercube-RAGcombines sparse exact match and semantic retrieval strategies, as in-
troduced in Chapter 5.3.3. We investigate the effect of the semantic retrieval strategy
on answer quality under varying similarity thresholds, denoted byτ. Figure 5.8 shows
that performance improves with increasing threshold values whenτ≤0.9, suggesting
that higher similarity scores are more effective for retrieving semantically relevant doc-
uments. However, whenτ >0.9, the behavior of semantic RAG begins to gradually
degrade to the sparse exact matching strategy, resulting in a decline in performance. This
is primarily due to overly strict matching criteria, which reduce the number of retrieved
documents and thus limit the available context.
97

Figure 5.8: Performance vs similarity threshold.
5.5.6 Case Study
An additional case study in Figure 5.9 presents that both Semantic-RAG and HippoRAG-
2 fail to generate the correct answer due to inaccurate document retrieval. These meth-
ods tend to prioritize returning documents that are semantically similar in the embedding
space. While the retrieved documents mention general topics such asabove-normal, near-
normal, and below-normal hurricane season, they miss the specific location‘Atlantic’
and organization‘Climate Prediction Center’. In contrast, ourHypercube-RAG
returns the correct answer. It accurately identifies and retrieves documents containing
key elements of the query, including‘Atlantic’,‘hurricane season’, and‘Climate Pre-
diction Center’, corresponding to the hypercube dimensions,LOCATON,THEME, and
ORGANIZATION, respectively. This retrieval approach intuitively prioritizes specific
contextual information over abstract, ambiguous references such asabove-normal, near-
normal, and below-normalalone.
98

Case Study 2
Query:What is the likelihood of an above-normal, near-normal and below-normal hur-
ricane season at the Atlantic, according to the Climate Prediction Center?
Hypercube-RAG
Answer:Above-normal: 85%, Near-normal: 10%, Below-normal: 5%(✓)
Retrieved Docs: [19, 230]
Doc 19:. . . Based on the ACE projection, combined with the above-average numbers of
named storms and hurricanes, the likelihood of an above-normal Atlantic hurricane season
has increased to 85%, with only a 10% chance of a near-normal season and a 5% chance
of a below-normal season . . . , said Gerry Bell, Ph.D., lead seasonal hurricane forecaster
at NOAA’s Climate Prediction Center . . .
Semantic-RAG (e5)
Answer:Above-normal: 10%, Near-normal: 20%, Below-normal: 70%.(✗)
Retrieved Docs: [230, 636]
Doc 230:. . . NOAA is predicting a 70 percent likelihood of 6 to 11 named storms, of
which 3 to 6 could become hurricanes . . . While a below-normal season is likely (70 per-
cent), there is also a 20 percent chance of a near-normal season, and a 10 percent chance
of an above-normal season . . .
Graph-based HippoRAG-2
Answer:65%, 25%, 10% chance for above-normal season, near-normal season, below-
normal season.(✗)
Retrieved Docs: [583, 275]
Doc 583:. . . The Climate Prediction Center outlook calls for considerable activity with
a 65 percent probability of an above normal season and a 25 percent probability of a near
normal season. This means there is a 90 percent chance of a near or above normal season.
Figure 5.9: Comparison of three RAG methods on the same query.
5.6 Discussion
5.6.1 Why doesHypercube-RAGoutperform other RAG methods?
OurHypercube-RAGis particularly proposed for domain-focused analysis, where domain-
specific information, e.g., location, date, event, event, theme, is central to the query. In
99

such cases with theme-specific information, our hypercube enables precise and efficient
retrieval by accessingonly one or a few cells in a theme-specific hypercube, as each cell
encapsulates multiple dimensions of information associated with a document. These fine-
grained document labels can be represented in a hypercube along different dimensions.
When a query comes in, accessing one or a few hypercubes presents a higher chance of
retrieving highly relevant documents covering all key information contained in a query -
Accuracy. Additionally, retrieval is operated with the cube labels significantly speeds up
the retrieval time since those labels are in a compact space -Efficiency. In contrast, sparse
exact match and dense embedding retrieval require traversing all documents and contents
inside. Moreover, the retrieval process is inherently explainable, as the searching is based
on the document’s labels represented by a hypercube -Explainability.
Case 1: simple query.How much rainfall did Melbourne Beach, Florida receive from
Tropical Storm Fay?The relevant documents can be retrieved from one cube cell in a
hypercube (Melbourne Beach, rainfall, Tropical Storm Fay).
Figure 5.10: Access one cube cell in one hypercube.✓represents the touched cube cells.
Case 2: long query with multiple topics.In cases where a query is very diverse, they
may need to access multiple cube cells such that the query information can be covered
100

as much as possible. For example, the diverse query could be“What consequences were
caused by Tropical Storm Fay, such as how much rainfall Melbourne Beach and Palm
Bay in Florida received, and how much monetary losses were caused?”. Those cube
cells needed could be (Melbourne Beach, rainfall, Tropical Storm Fay), (Palm Beach,
rainfall, Tropical Storm Fay), (Melbourne Beach, monetary losses, Tropical Storm Fay),
(Palm Beach, monetary losses, Tropical Storm Fay).
Figure 5.11: Access multiple cube cells in one hypercube.✓represents the touched cube
cells.
Case 3: diverse query with shifted topics.Given a more complicated query including
shifted topics, e.g.,“How did the monetary loss caused by Tropical Storm Fay impact
the industrial layoff in Miami Downtown?”, cube cells across two hypercubes are needed
(Miami Downtown, Industrial Layoff, Tropical Storm Fay) and (Miami Downtown, Mon-
etary loss, Tropical Storm Fay) to get more relevant documents.
101

Figure 5.12: Access multiple cube cells in multiple hypercubes.✓represents the touched
cube cells.
5.6.2 Why did we use in-domain scientific QA datasets rather than
general ones?
According to the above discussion, hypercube construction is sensitive to themes in vari-
ous domains. While similar or related themes are allowed to be included in a hypercube,
each theme-specific dataset should have a separate hypercube to maximize its power.
However, the general QA datasets, e.g., PopQA [MAZ+22] and NQ [KPR+19], typically
present articles across multiple themes. The process of constructing multiple hypercubes
will require intensive efforts (e.g., design hypercube schema for different datasets.), which
is out of the scope of this study. This is why we particularly focus on in-domain theme-
specific QA datasets.
5.6.3 Future work
For general questions consisting of multiple and different themes, we plan to use large
language models (LLMs) to dynamically construct multiple hypercubes. Each hypercube
is used to represent documents with one specific theme, respectively. The possible related
102

work is TeleClass [ZYX+25], and TKGCon [DZXH24]. On the other hand, we aim
to explore the query-augmented technique to enhance the robustness of retrieval systems
[SLG+24, LSJ+22]. At last, we would like to testHypercube-RAGon the larger corpus
for industrial applications.
5.7 Conclusions
In this chapter, we start by analyzing the strengths and limitations of existing RAG meth-
ods for in-domaintheme-specificapplications. To retain their advantages while address-
ing key challenges, we proposeHypercube-RAG, innovating retrieval based on fine-
grained labels with a multidimensional structure (i.e., hypercube). The label-based re-
trieval integrates sparse lexical and dense semantic strategies, making it accurate, effi-
cient, and inherently explainable. Empirical results demonstrate that our method consis-
tently outperforms existing RAG baselines in accuracy and efficiency, while providing
interpretable provenance. We propose future plans to generalize theHypercube-RAG
to open-source domains and explore powerful query-augmented techniques to boost the
robustness of retrieval systems.
103

CHAPTER 6
CONCLUSIONS
This dissertation advances the application of artificial intelligence (AI) in environ-
mental science, aiming to achieveenvironmental intelligence. We explored three key
problems at the intersection of AI and environmental science and offer practical and high-
performing AI-based solutions for them.
6.1 DL-based Flood Prediction and Management
In Chapter 3, we developed WALEF for forecasting water levels and FIDLAR, a forecast-
informed DL-based architecture for managing water levels in coastal river systems, inte-
grating diverse inputs such as rainfall, sea levels, historical water levels, and settings of
hydraulic structures. Our contributions are summarized as follows:
• Our graph-transformer-based model (WALEF) for forecasting water levels exhibits
superior performance when compared to baseline methods.
• FIDLARis a data-driven approach, learning flood mitigation strategies from histor-
ically observed data. Once trained, it offers rapid response capabilities, highlight-
ing the advantages of DL-based models over physics-based models, particularly for
real-time flood management.
• FIDLARseamlessly combines the two DL models in series:Flood Manager
andFlood Evaluator. The former model is responsible for generating water
pre-release schedules, while the latter model accurately forecasts the resulting water
levels. Moreover, with the gradient-based planning, and differentiability of trained
Evaluator, it can reinforce theManagerto generate better schedules.
104

• FIDLARis a model-agnostic framework, where both theManagerandEvaluator
could be any type of DL model trained with differentiable loss functions that allow
back-propagation.
• Demonstrated on flood-prone regions in South Florida, FIDLARoffers a practi-
cal solution for real-time flood mitigation, outperforming traditional baselines in
accuracy, efficiency, and interpretability.
6.2 Diffusion Model for Probabilistic Weather Forecasting
In Chapter 4, we proposed CODICAST, a conditional diffusion model repurposed from
generative AI to enable accurate and efficient global weather prediction with explicit un-
certainty quantification. Unlike deterministic physics-based methods, CODICASTcap-
tures the stochastic nature of weather, making it suitable for anticipating extreme events
and supporting informed risk assessments. More specifically, the contributions of the
chapter on CODICASTare as follows:
• Existing weather prediction methods are computationally intensive and often make
simplifying assumptions. Our tool, CODICAST, is efficient and avoids the assump-
tions.
• CODICASTrepurposes a “generative” tool as a “predictive” tool. It is a conditional
diffusion model for global weather prediction conditioning on observations from
the recent past.
• CODICASTprobabilistically models the uncertainty by generating multiple plausi-
ble weather scenarios.
• With ERA5 data from the European Centre for Medium-Range Weather Forecasts
(ECMWF), we show that CODICASTachieves good performance without suffering
105

from the tradeoff between accuracy, efficiency, and uncertainty, while providing
explainability for its answers.
6.3 Question-Answering Systems for Environmental Knowledge
In Chapter 5, we focus on the knowledge retrieval from external databases to enhance the
capabilities of pre-trained LLMs while answeringin-domainquestions. Our contributions
are three-fold:
• We analyze the limitations of current LLM-based question-answering systems, which
suffer from a tradeoff between accuracy, efficiency, and explainability.
• OurHypercube-RAG, a retrieval-augmented generation (RAG) system, is built
on a multidimensional knowledge structuretext cube. By allocating unstructured
documents to a structured text cube based on the document labels along pre-defined
dimensions,Hypercube-RAGcan perform information retrieval using fine-grained
document labels organized within the cube.
• Experiments with scientific queries on the topics of hurricanes, aging dams, and
other hazards, we show thatHypercube-RAGenhances existing LLMs with im-
proved accuracy, interpretability, and retrieval efficiency.
In summary, the contributions in this dissertation demonstrate how a diverse set of
AI techniques - spanning deep learning architectures, generative diffusion models, and
structured retrieval systems - can be tailored to meet the multifaceted challenges of en-
vironmental science. This work underscores the growing importance ofEnvironmental
Intelligence, where advanced AI methods empower data-driven, precise, efficient, and
interpretable strategies to tackle some pressing environmental problems.
106

6.4 Future Work
This dissertation develops and applies deep learning strategies to address key challenges
in environmental science, with a focus on improving accuracy, efficiency, and explain-
ability. We also highlight several natural extensions that can further advance the toolkit
for achieving environmental intelligence.
6.4.1 Scalability
In Chapter 3, although the study incorporates heterogeneous driving factors for floods, the
focus remains on a regional coastal river system in South Florida. Extending the FIDLAR
methodology to more complex and large-scale river systems would be a valuable direction
for future research. In Chapter 4, we focus on six meteorological variables at a coarse
global resolution due to limited computational resources. However, accurate weather
forecasting would benefit from incorporating additional atmospheric variables at higher
spatial and temporal resolutions. Therefore, scaling up the dataset using more powerful
computing resources represents a promising direction for future work. Finally, in Chapter
5, it would be challenging to work with a much larger body of knowledge to understand
the limits of the methods used.
6.4.2 Acceleration of Diffusion Models
In Chapter 4, while diffusion models have shown great promise in generative and prob-
abilistic tasks, their iterative denoising process often requires hundreds or thousands of
steps, posing a significant computational challenge, especially for time-sensitive applica-
tions like weather forecasting. To address this, recent research has focused on accelerating
diffusion models through compressing the iterative process into fewer steps (e.g., 1-step
107

or 4-step models), which enables faster sampling without compromising accuracy. For
weather prediction, where high-resolution, multi-variable forecasts must be delivered in
real time, such acceleration strategies are crucial. By integrating fast sampling techniques
and optimized inference pipelines, diffusion-based models can become both a probabilis-
tic and practical alternative to traditional numerical weather prediction systems, offering
explicit uncertainty quantification with tractable computational costs. Continued research
into efficient architectures and hardware-aware implementations will further enhance the
viability of diffusion models for operational forecasting.
6.4.3 Flexibility
In Chapter 5, our proposedHypercube-RAGframework is constructed atop a struc-
tured text cube defined by human-curated dimensions. While this design proves highly
effective for in-domain, theme-specific applications, it lacks adaptability when the under-
lying knowledge base evolves or when applied to new domains with different structures.
A promising direction for future work is to develop methods for automated or adaptive
hypercube construction. This could involve leveraging large language models (LLMs),
unsupervised clustering techniques, or reinforcement learning to dynamically determine
optimal dimensions and hierarchies based on the corpus content.
6.5 Endless Possibilities
Advancements that once belonged to the realm of science fiction are becoming increas-
ingly attainable. The future ofAI for Scienceholds vast potential, especially with the
rapid emergence and evolution of large language models (LLMs) and large multimodal
models (LMMs).
108

BIBLIOGRAPHY
[AAA+23] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge
Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt,
Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report.arXiv
preprint arXiv:2303.08774, 2023.
[ABL+24] Mihai Alexe, Eulalie Boucher, Peter Lean, Ewan Pinnington, Patrick
Laloyaux, Anthony McNally, Simon Lang, Matthew Chantry, Chris Bur-
rows, Marcin Chrust, et al. GraphDOP: Towards skilful data-driven
medium-range weather forecasts learnt and initialised directly from ob-
servations.arXiv preprint arXiv:2412.15687, 2024.
[AEk+24] Sara Ansarifard, Morteza Eyvazi, Mahsa kalantari, Behrooz mohseni,
Mahdi Ghorbanifard, Hadi Jafakesh Moghaddam, and Maryam Nouri.
Simulation of floods under the influence of effective factors in hydraulic
and hydrological models using HEC-RAS and MIKE 21.Discover Water,
4(1):92, 2024.
[AEL+23] Marcin Andrychowicz, Lasse Espeholt, Di Li, Samier Merchant, Alexan-
der Merose, Fred Zyda, Shreya Agrawal, and Nal Kalchbrenner. Deep
learning for day forecasts from sparse observations, 2023.
[AFK+14] Peshawa Jamal Muhammad Ali, Rezhna Hassan Faraj, Erbil Koya, Pe-
shawa J Muhammad Ali, and Rezhna H Faraj. Data normalization and
standardization: a technical report.Mach Learn Tech Rep, 1(1):1–6, 2014.
[Ala21] Ashraf Alam. Possibilities and apprehensions in the landscape of artificial
intelligence in education. In2021 International Conference on Compu-
tational Intelligence and Computing Applications (ICCICA), pages 1–8.
IEEE, 2021.
[ALOL24] Martin Andrae, Tomas Landelius, Joel Oskarsson, and Fredrik Lindsten.
Continuous ensemble weather forecasting with diffusion models.arXiv
preprint arXiv:2410.05431, 2024.
[AMP+23] Andrea Asperti, Fabio Merizzi, Alberto Paparella, Giorgio Pedrazzi, Mat-
teo Angelinelli, and Stefano Colamonaco. Precipitation nowcasting with
generative diffusion models.arXiv preprint arXiv:2308.06733, 2023.
109

[AN24] Emran Alotaibi and Nadia Nassif. Artificial intelligence in environmental
monitoring: in-depth analysis.Discover Artificial Intelligence, 4(1):84,
2024.
[ANG23] Ishan Ayus, Narayanan Natarajan, and Deepak Gupta. Prediction of wa-
ter level using machine learning and deep learning techniques.Iranian
Journal of Science and Technology, Transactions of Civil Engineering,
47(4):2437–2447, 2023.
[AS20] Plamen Angelov and Eduardo Soares. Towards explainable deep neural
networks (xDNN).Neural Networks, 130:185–194, 2020.
[AWW+23] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Ha-
jishirzi. Self-rag: Learning to retrieve, generate, and critique through self-
reflection. InThe Twelfth International Conference on Learning Repre-
sentations, 2023.
[AYCG10] Sultan Al-Yahyai, Yassine Charabi, and Adel Gastli. Review of the use of
numerical weather prediction (NWP) models for wind energy assessment.
Renewable and Sustainable Energy Reviews, 14(9):3192–3198, 2010.
[AZH+21] Laith Alzubaidi, Jinglan Zhang, Amjad J Humaidi, Ayad Al-Dujaili,
Ye Duan, Omran Al-Shamma, Jos ´e Santamar ´ıa, Mohammed A Fad-
hel, Muthana Al-Amidie, and Laith Farhan. Review of deep learning:
concepts, CNN architectures, challenges, applications, future directions.
Journal of big Data, 8:1–74, 2021.
[Bal12] Pierre Baldi. Autoencoders, unsupervised learning, and deep architec-
tures. InProceedings of ICML workshop on unsupervised and transfer
learning, pages 37–49, Edinburgh, Scotland, 2012. JMLR Workshop and
Conference Proceedings.
[BAML21] Alabi Bojesomo, Hasan Al-Marzouqi, and Panos Liatsis. Spatiotemporal
vision transformer for short time weather forecasting. In2021 IEEE In-
ternational Conference on Big Data (Big Data), pages 5741–5746. IEEE,
2021.
[BBCM+24] Zied Ben Bouall `egue, Mariana CA Clare, Linus Magnusson, Estibaliz
Gascon, Michael Maier-Gerber, Martin Janou ˇsek, Mark Rodwell, Florian
Pinault, Jesper S Dramsch, Simon TK Lang, et al. The rise of data-driven
weather forecasting: A first statistical assessment of machine learning–
110

based weather forecasts in an operational-like context.Bulletin of the
American Meteorological Society, 105(6):E864–E883, 2024.
[BBL+24] Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Jo-
hannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan Weyn,
Haiyu Dong, Anna Vaughan, et al. Aurora: A foundation model of the
atmosphere.arXiv preprint arXiv:2405.13063, 2024.
[BBO17] Anastasia Borovykh, Sander Bohte, and Cornelis W Oosterlee. Condi-
tional time series forecasting with convolutional neural networks.arXiv
preprint arXiv:1703.04691, 2017.
[BCD+22] V Balaji, Fleur Couvreux, Julie Deshayes, Jacques Gautrais, Fr ´ed´eric
Hourdin, and Catherine Rio. Are general circulation models obsolete?
Proceedings of the National Academy of Sciences, 119(47):e2202075119,
2022.
[BCK+16] Deepak Singh Bisht, Chandranath Chatterjee, Shivani Kalakoti, Pawan
Upadhyay, Manaswinee Sahoo, and Ambarnil Panda. Modeling urban
floods and drainage using SWMM and MIKE URBAN: a case study.Nat-
ural hazards, 84:749–776, 2016.
[BD22] Saray Busto and Michael Dumbser. A staggered semi-implicit hybrid fi-
nite volume/finite element scheme for the shallow water equations at all
Froude numbers.Applied Numerical Mathematics, 175:108–132, 2022.
[BGE18] Sahar Babaei, Reza Ghazavi, and Mahdi Erfanian. Urban flood simulation
and prioritization of critical urban sub-catchments using SWMM model
and PROMETHEE II approach.Physics and Chemistry of the Earth, Parts
A/B/C, 105:3–11, 2018.
[BHQL24] Christopher B ¨ulte, Nina Horat, Julian Quinting, and Sebastian Lerch. Un-
certainty quantification for data-driven weather models.arXiv preprint
arXiv:2403.13458, 2024.
[BIJT22] Roberto Bentivoglio, Elvin Isufi, Sebastian Nicolaas Jonkman, and Ric-
cardo Taormina. Deep learning methods for flood mapping: a review of
existing applications and future research directions.Hydrology and Earth
System Sciences Discussions, 2022:1–50, 2022.
[BKC+22] Emmanuel Asiedu Brempong, Simon Kornblith, Ting Chen, Niki Parmar,
Matthias Minderer, and Mohammad Norouzi. Denoising pretraining for
111

semantic segmentation. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 4175–4186, 2022.
[BKK18] Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation
of generic convolutional and recurrent networks for sequence modeling.
arXiv preprint arXiv:1803.01271, 2018.
[BMR+20] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Ka-
plan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sas-
try, Amanda Askell, et al. Language models are few-shot learners.Ad-
vances in Neural Information Processing Systems (NeurIPS), 33:1877–
1901, 2020.
[Bre01] Leo Breiman. Random forests.Machine learning, 45:5–32, 2001.
[BSZ+22] Cong Bai, Feng Sun, Jinglin Zhang, Yi Song, and Shengyong Chen. Rain-
former: Features extraction balanced network for radar-based precipita-
tion nowcasting.IEEE Geoscience and Remote Sensing Letters, 19:1–5,
2022.
[BTB15] Peter Bauer, Alan Thorpe, and Gilbert Brunet. The quiet revolution of
numerical weather prediction.Nature, 525(7567):47–55, 2015.
[BTW+21] Benjamin D Bowes, Arash Tavakoli, Cheng Wang, Arsalan Heydarian,
Madhur Behl, Peter A Beling, and Jonathan L Goodall. Flood mitigation
in coastal urban catchments using real-time stormwater infrastructure con-
trol and reinforcement learning.Journal of Hydroinformatics, 23(3):529–
547, 2021.
[Bui08] Roberto Buizza. Comparison of a 51-member low-resolution (T L
399L62) ensemble with a 6-member high-resolution (T L 799L91) lagged-
forecast ensemble.Monthly weather review, 136(9):3343–3362, 2008.
[BXZ+23] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and
Qi Tian. Accurate medium-range global weather forecasting with 3D neu-
ral networks.Nature, 619(7970):533–538, 2023.
[CCZ+23] Lin Chen, Zhonghao Chen, Yubing Zhang, Yunfei Liu, Ahmed I Osman,
Mohamed Farghali, Jianmin Hua, Ahmed Al-Fatesh, Ikko Ihara, David W
Rooney, et al. Artificial intelligence-based solutions for climate change: a
review.Environmental Chemistry Letters, 21(5):2525–2557, 2023.
112

[CDC+22] Kamal Choudhary, Brian DeCost, Chi Chen, Anubhav Jain, Francesca
Tavazza, Ryan Cohn, Cheol Woo Park, Alok Choudhary, Ankit Agrawal,
Simon JL Billinge, et al. Recent advances and applications of deep learn-
ing methods in materials science.npj Computational Materials, 8(1):59,
2022.
[CDH+23] Lei Chen, Fei Du, Yuan Hu, Zhibin Wang, and Fan Wang. SwinRDM: in-
tegrate SwinRNN with diffusion model towards high-resolution and high-
quality weather forecasting. InProceedings of the AAAI Conference on
Artificial Intelligence, pages 322–330, 2023.
[CGPL+14] Mario E Castro-Gama, Ioana Popescu, Shengyang Li, Arthur Mynett, and
Arthur van Dam. Flood inference simulation using surrogate modelling
for the Yellow river multiple reservoir system.Environmental modelling
& software, 55:250–265, 2014.
[CHG+23] Kang Chen, Tao Han, Junchao Gong, Lei Bai, Fenghua Ling, Jing-Jia Luo,
Xi Chen, Leiming Ma, Tianning Zhang, Rui Su, et al. Fengwu: Pushing
the skillful global medium-range weather forecast beyond 10 days lead.
arXiv preprint arXiv:2304.02948, 2023.
[CHIS23] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak
Shah. Diffusion models in vision: A survey.IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2023.
[CKK+23] Jaeho Choi, Yura Kim, Kwang-Ho Kim, Sung-Hwa Jung, and Ikhyun Cho.
PCT-CycleGAN: Paired complementary temporal cycle-consistent adver-
sarial networks for radar-based precipitation nowcasting. InProceedings
of the 32nd ACM International Conference on Information and Knowl-
edge Management, pages 348–358, 2023.
[CLGH16] Duan Chen, Arturo S Leon, Nathan L Gibson, and Parnian Hosseini. Di-
mension reduction of decision variables for multireservoir operation: A
spectral optimization model.Water Resources Research, 52(1):36–51,
2016.
[CLJ+23] Shengchao Chen, Guodong Long, Jing Jiang, Dikai Liu, and Chengqi
Zhang. Foundation models for weather and climate data understanding:
A comprehensive survey.arXiv preprint arXiv:2312.03014, 2023.
[CMG+19] Bahram Choubin, Ehsan Moradi, Mohammad Golshan, Jan Adamowski,
Farzaneh Sajedi-Hosseini, and Amir Mosavi. An ensemble prediction of
113

flood susceptibility using multivariate discriminant analysis, classification
and regression trees, and support vector machines.Science of the Total
Environment, 651:2087–2096, 2019.
[CMH+22] Ashesh Chattopadhyay, Mustafa Mustafa, Pedram Hassanzadeh, Evi-
atar Bach, and Karthik Kashinath. Towards physics-inspired data-driven
weather forecasting: integrating data assimilation with a deep spatial-
transformer-based U-NET in a case study with ERA5.Geoscientific
Model Development, 15(5):2221–2237, 2022.
[Coi11] Jean Coiffier.Fundamentals of numerical weather prediction. Cambridge
University Press, 2011.
[CRBD18] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duve-
naud. Neural ordinary differential equations.Advances in neural infor-
mation processing systems, 31, 2018.
[CSC+24] Guillaume Couairon, Renu Singh, Anastase Charantonis, Christian
Lessig, and Claire Monteleoni. ArchesWeather & Archesweathergen: a
deterministic and generative model for efficient ML weather forecasting.
arXiv preprint arXiv:2412.12971, 2024.
[CTCO19] Mikel Canizo, Isaac Triguero, Angel Conde, and Enrique Onieva. Multi-
head CNN–RNN for multi-time series anomaly detection: An industrial
case study.Neurocomputing, 363:246–260, 2019.
[CUC+25] Hejie Cui, Alyssa Unell, Bowen Chen, Jason Alan Fries, Emily Alsentzer,
Sanmi Koyejo, and Nigam Shah. TIMER: Temporal instruction modeling
and evaluation for longitudinal clinical records. InWill Synthetic Data
Finally Solve the Data Access Problem?, 2025.
[CWHT24] Yutong Chen, Ya Wang, Gang Huang, and Qun Tian. Coupling physi-
cal factors for precipitation forecast in China with graph neural network.
Geophysical Research Letters, 51(2):e2023GL106676, 2024.
[CXL+23] Shengyu Chen, Yiqun Xie, Xiang Li, Xu Liang, and Xiaowei Jia. Physics-
guided meta-learning method in baseflow prediction over large regions. In
Proceedings of the 2023 SIAM International Conference on Data Mining
(SDM), pages 217–225. SIAM, 2023.
[CZZ+23] Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu, Yuan
Qi, and Hao Li. FuXi: A cascade machine learning forecasting system
114

for 15-day global weather forecast.npj Climate and Atmospheric Science,
6(1):190, 2023.
[DBPG19] Emmanuel De B ´ezenac, Arthur Pajot, and Patrick Gallinari. Deep learn-
ing for physical processes: Incorporating prior scientific knowledge.Jour-
nal of Statistical Mechanics: Theory and Experiment, 2019(12):124009,
2019.
[DCLT19a] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
BERT: Pre-training of deep bidirectional transformers for language under-
standing. InProceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics (NAACL), pages
4171–4186, 2019.
[DCLT19b] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
Bert: Pre-training of deep bidirectional transformers for language under-
standing. InProceedings of the 2019 conference of the North American
chapter of the association for computational linguistics: human language
technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
[DCNV+21] Felipe Arias Del Campo, Mar ´ıa Cristina Guevara Neri, Osslan Osiris Ver-
gara Villegas, Vianey Guadalupe Cruz S ´anchez, Humberto de Jes ´us Ochoa
Dom ´ınguez, and Vicente Garc ´ıa Jim ´enez. Auto-adaptive multilayer per-
ceptron for univariate time series classification.Expert Systems with Ap-
plications, 181:115147, 2021.
[DDBB24] Sagar Debbarma, Subhajit Dey, Arnab Bandyopadhyay, and Aditi Bhadra.
Simulation of flood inundation extent by integration of HEC-HMS, GA-
based rating curve and cost distance analysis.Water Resources Manage-
ment, pages 1–21, 2024.
[DED19] Yanqing Duan, John S Edwards, and Yogesh K Dwivedi. Artificial intel-
ligence for decision making in the era of big data–evolution, challenges
and research agenda.International journal of information management,
48:63–71, 2019.
[DHM+20] Chris J Delaney, Robert K Hartman, John Mendoza, Michael Dettinger,
Luca Delle Monache, Jay Jasperse, F Martin Ralph, Cary Talbot, James
Brown, David Reynolds, et al. Forecast informed reservoir operations
using ensemble streamflow predictions for a multipurpose reservoir in
Northern California.Water Resources Research, 56(9):e2019WR026604,
2020.
115

[Dis23] South Florida Water Management District. DBHYDRO of South
Florida Water Management District.https://www.sfwmd.gov/
science-data/dbhydro, 2023.
[DKA19] Lea Dasallas, Yeonsu Kim, and Hyunuk An. Case study of HEC-RAS
1D–2D coupling simulation: 2002 Baeksan flood event in Korea.Water,
11(10):2048, 2019.
[DLYC21] Xiaojian Ding, Jian Liu, Fan Yang, and Jie Cao. Random radial basis
function kernel-based support vector machine.Journal of the Franklin
Institute, 358(18):10121–10140, 2021.
[DN21] Jos ´e Devezas and S ´ergio Nunes. A review of graph-based models for
entity-oriented search.SN Computer Science, 2(6):437, 2021.
[DZXH24] Linyi Ding, Sizhe Zhou, Jinfeng Xiao, and Jiawei Han. Auto-
mated construction of theme-specific knowledge graphs.arXiv preprint
arXiv:2404.19146, 2024.
[EAS+22] Lasse Espeholt, Shreya Agrawal, Casper Sønderby, Manoj Kumar,
Jonathan Heek, Carla Bromberg, Cenk Gazen, Rob Carver, Marcin
Andrychowicz, Jason Hickey, et al. Deep learning for twelve hour pre-
cipitation forecasts.Nature communications, 13(1):1–10, 2022.
[ECM] ECMWF.https://www.ecmwf.int/en/forecasts/
documentation-and-support/medium-range-forecasts.
[ECM23] ECMWF. Medium-range forecasts, 2023.
[EH24] Maxim Enis and Mark Hopkins. From llm to nmt: Advanc-
ing low-resource machine translation with claude.arXiv preprint
arXiv:2404.13813, 2024.
[EMSK97] Floriana Esposito, Donato Malerba, Giovanni Semeraro, and J Kay. A
comparative analysis of methods for pruning decision trees.IEEE transac-
tions on pattern analysis and machine intelligence, 19(5):476–491, 1997.
[ERKL16] Sarah M Erfani, Sutharshan Rajasegarar, Shanika Karunasekera, and
Christopher Leckie. High-dimensional and large-scale anomaly detection
using a linear one-class SVM with deep learning.Pattern Recognition,
58:121–134, 2016.
116

[ETC+24] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao,
Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa
Ness, and Jonathan Larson. From local to global: A graph rag approach
to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[EWN+23] Jason K Eshraghian, Max Ward, Emre O Neftci, Xinxin Wang, Gregor
Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and
Wei D Lu. Training spiking neural networks using lessons from deep
learning.Proceedings of the IEEE, 2023.
[EXES+21] Mahmoud F Elmorshedy, Wei Xu, Fayez FM El-Sousy, Md Rabiul Is-
lam, and Abdelsalam A Ahmed. Recent achievements in model predictive
control techniques for industrial motor: A comprehensive state-of-the-art.
IEEE Access, 9:58170–58191, 2021.
[FWWN23] Niels Fraehr, Quan J Wang, Wenyan Wu, and Rory Nathan. Development
of a fast and accurate hybrid model for floodplain inundation simulations.
Water Resources Research, 59(6):e2022WR033836, 2023.
[FZZ+15] Wei Feng, Chao Zhang, Wei Zhang, Jiawei Han, Jianyong Wang, Charu
Aggarwal, and Jianbin Huang. STREAMCUBE: Hierarchical spatio-
temporal hashtag clustering for event exploration over the Twitter stream.
In2015 IEEE 31st international conference on data engineering, pages
1561–1572. IEEE, 2015.
[GAC+22] Shenghan Guo, Mohit Agarwal, Clayton Cooper, Qi Tian, Robert X Gao,
Weihong Guo Grace, and YB Guo. Machine learning for metal additive
manufacturing: Towards a physics-informed data-driven paradigm.Jour-
nal of Manufacturing Systems, 62:145–163, 2022.
[GBY+24] Junchao Gong, Lei Bai, Peng Ye, Wanghan Xu, Na Liu, Jianhua
Dai, Xiaokang Yang, and Wanli Ouyang. CasCast: Skillful high-
resolution precipitation nowcasting via cascaded modelling.arXiv
preprint arXiv:2402.04290, 2024.
[GDS19] Sourish Ghosh, Anasuya Dasgupta, and Aleena Swetapadma. A study
on support vector machine based linear and non-linear pattern classifica-
tion. In2019 International conference on intelligent sustainable systems
(ICISS), pages 24–28. IEEE, 2019.
117

[GG16] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approxima-
tion: Representing model uncertainty in deep learning. Ininternational
conference on machine learning, pages 1050–1059. PMLR, 2016.
[GJRO+23] Marcus N Gomes Jr, Luis MC R ´apalo, Paulo TS Oliveira, Marcio H Gi-
acomoni, C ´esar AF do Lago, and Eduardo M Mendiondo. Modeling un-
steady and steady 1D hydrodynamics under different hydraulic conceptu-
alizations: Model/Software development and case studies.Environmental
Modelling & Software, 167:105733, 2023.
[GK18] Marian Gidea and Yuri Katz. Topological data analysis of financial time
series: Landscapes of crashes.Physica A: Statistical Mechanics and its
Applications, 491:820–834, 2018.
[GKBG23] Yuebin Guo, Andreas Klink, Paulo Bartolo, and Weihong Grace Guo. Dig-
ital twins for electro-physical, chemical, and photonic processes.CIRP
Annals, 72(2):593–619, 2023.
[Gla21] Frank Jerome Glago. Flood disaster hazards; causes, impacts and man-
agement: a state-of-the-art review.Natural hazards-impacts, adjustments
and resilience, pages 29–37, 2021.
[GR05] Tilmann Gneiting and Adrian E Raftery. Weather forecasting with ensem-
ble methods.Science, 310(5746):248–249, 2005.
[GSG+24] Bernal Jim ´enez Guti ´errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and
Yu Su. Hipporag: Neurobiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Conference on Neural In-
formation Processing Systems, 2024.
[GSH+24] Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle
Maddix, Yi Zhu, Mu Li, and Yuyang Bernie Wang. PreDiff: Precipitation
nowcasting with latent diffusion models.Advances in Neural Information
Processing Systems, 36, 2024.
[GSQ+25] Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and
Yu Su. From RAG to memory: Non-parametric continual learning for
large language models.arXiv preprint arXiv:2502.14802, 2025.
[GSR+17] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and
George E Dahl. Neural message passing for quantum chemistry. InIn-
118

ternational conference on machine learning, pages 1263–1272. PMLR,
2017.
[GSW+22] Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Bernie Wang,
Mu Li, and Dit-Yan Yeung. Earthformer: Exploring space-time trans-
formers for earth system forecasting.Advances in Neural Information
Processing Systems, 35:25390–25403, 2022.
[GT20] Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics
from unknown factors for unsupervised video prediction. InProceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
pages 11474–11484, 2020.
[GXY+24] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Ligh-
trag: Simple and fast retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2024.
[GYZ+25] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang,
Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.
Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforce-
ment learning.arXiv preprint arXiv:2501.12948, 2025.
[HBB+20] Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, Andr ´as
Hor´anyi, Joaqu ´ın Mu ˜noz-Sabater, Julien Nicolas, Carole Peubey, Raluca
Radu, Dinand Schepers, et al. The ERA5 global reanalysis.Quarterly
Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020.
[HCWL23] Yuan Hu, Lei Chen, Zhibin Wang, and Hao Li. SwinVRNN: A data-driven
ensemble forecasting model via learned distribution perturbation.Journal
of Advances in Modeling Earth Systems, 15(2):e2022MS003211, 2023.
[HJA20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion prob-
abilistic models.Advances in neural information processing systems,
33:6840–6851, 2020.
[HM95] Jun Han and Claudio Moraga. The influence of the sigmoid function pa-
rameters on the speed of backpropagation learning. InInternational work-
shop on artificial neural networks, pages 195–201. Springer, 1995.
[HMB+25] Mohamed Manzour Hussien, Angie Nataly Melo, Augusto Luis Ballar-
dini, Carlota Salinas Maldonado, Rub ´en Izquierdo, and Miguel Angel
119

Sotelo. Rag-based explainable prediction of road users behaviors for auto-
mated driving using knowledge graphs and large language models.Expert
Systems with Applications, 265:125914, 2025.
[HMT+22] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch,
and Daniel Cohen-Or. Prompt-to-prompt image editing with cross atten-
tion control.arXiv preprint arXiv:2208.01626, 2022.
[HS22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance.arXiv
preprint arXiv:2207.12598, 2022.
[HTPB21] Pradeep Hewage, Marcello Trovati, Ella Pereira, and Ardhendu Behera.
Deep learning-based effective fine-grained weather forecasting model.
Pattern Analysis and Applications, 24(1):343–366, 2021.
[HXCH04] Wenrui Huang, Bing Xu, and Amy Chan-Hilton. Forecasting flows
in Apalachicola River using neural networks.Hydrological processes,
18(13):2545–2564, 2004.
[HYM+25] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng,
Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing
Qin, et al. A survey on hallucination in large language models: Princi-
ples, taxonomy, challenges, and open questions.ACM Transactions on
Information Systems, 43(2):1–55, 2025.
[ICH+21] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Pi-
otr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised
dense information retrieval with contrastive learning.arXiv preprint
arXiv:2112.09118, 2021.
[ICH+22] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr
Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense
information retrieval with contrastive learning.Transactions on Machine
Learning Research, 2022.
[IJH+24] Qiyamud Din Ikram, Abdur Rashid Jamalzi, Abdur Rahim Hamidi, Irfan
Ullah, and Muhmmad Shahab. Flood risk assessment of the population
in Afghanistan: a spatial analysis of hazard, exposure, and vulnerability.
Natural Hazards Research, 4(1):46–55, 2024.
[JCT+23] Luke T Jenkins, Maggie J Creed, Karim Tarbali, Manoranjan Muthusamy,
Robert ˇSaki´c Trogrli ´c, Jeremy C Phillips, C Scott Watson, Hugh D Sin-
120

clair, Carmine Galasso, and John McCloskey. Physics-based simulations
of multiple natural hazards for risk-sensitive planning and decision mak-
ing in expanding urban regions.International Journal of Disaster Risk
Reduction, 84:103338, 2023.
[JGR20] Tammy Jiang, Jaimie L Gradus, and Anthony J Rosellini. Supervised
machine learning: a brief primer.Behavior therapy, 51(5):675–687, 2020.
[JK22] KU Jaseena and Binsu C Kovoor. Deterministic weather forecasting
models based on intelligent predictors: A survey.Journal of king saud
university-computer and information sciences, 34(6):3393–3412, 2022.
[JMP+23] Keighobad Jafarzadegan, Hamid Moradkhani, Florian Pappenberger,
Hamed Moftakhari, Paul Bates, Peyman Abbaszadeh, Reza Marsooli,
Celso Ferreira, Hannah L Cloke, Fred Ogden, et al. Recent advances
and new frontiers in riverine and coastal flood modeling.Reviews of Geo-
physics, 61(2):e2022RG000788, 2023.
[JSR+19] Xiaonan Ji, Han-Wei Shen, Alan Ritter, Raghu Machiraju, and Po-Yin
Yen. Visual exploration of neural document embedding in information
retrieval: Semantics and feature selection.IEEE transactions on visual-
ization and computer graphics, 25(6):2181–2192, 2019.
[JXG+23] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane
Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active
retrieval augmented generation. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 7969–7992,
2023.
[JYHA25] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context
LLMs meet RAG: Overcoming challenges for long inputs in RAG. InThe
Thirteenth International Conference on Learning Representations, 2025.
[JZH21] Christian Janiesch, Patrick Zschech, and Kai Heinrich. Machine learning
and deep learning.Electronic markets, 31(3):685–695, 2021.
[Kad19] Ammar Ismael Kadhim. Term weighting for feature extraction on Twitter:
A comparison between bm25 and tf-idf. In2019 international conference
on advanced science and engineering (ICOASE), pages 124–128. IEEE,
2019.
121

[KAJ+24] SeongKu Kang, Shivam Agarwal, Bowen Jin, Dongha Lee, Hwanjo Yu,
and Jiawei Han. Improving retrieval in theme-specific applications using
a corpus topical taxonomy. InProceedings of the ACM Web Conference
2024, pages 1497–1508, 2024.
[Kar16] Divas Karimanzira. Model based decision support systems.Modeling,
Control and Optimization of Water Systems: Systems Engineering Meth-
ods for Control and Decision Making Tasks, pages 185–220, 2016.
[KAS+23] Vijendra Kumar, Hazi Md Azamathulla, Kul Vaibhav Sharma, Darshan J
Mehta, and Kiran Tota Maharaj. The state of the art in deep learning
applications, challenges, and future prospects: A comprehensive review
of flood forecasting and management.Sustainability, 15(13):10543, 2023.
[Kei22] Ryan Keisler. Forecasting global weather with graph neural networks.
arXiv preprint arXiv:2202.07575, 2022.
[KGL+16] Branko Kerkez, Cyndee Gruden, Matthew Lewis, Luis Montestruque,
Marcus Quigley, Brandon Wong, Alex Bedig, Ruben Kertesz, Tim Braun,
Owen Cadwalader, et al. Smarter stormwater systems, 2016.
[KHMN25] Kousha Khatooni, Farhad Hooshyaripor, Bahram MalekMohammadi, and
Roohollah Noori. A new approach for urban flood risk assessment using
coupled SWMM–HEC-RAS-2D model.Journal of Environmental Man-
agement, 374:123849, 2025.
[KK13] Oliver Kramer and Oliver Kramer. K-nearest neighbors.Dimensionality
reduction with unsupervised nearest neighbors, pages 13–23, 2013.
[KLY+24] Pu-Yun Kow, Jia-Yi Liou, Ming-Ting Yang, Meng-Hsin Lee, Li-Chiu
Chang, and Fi-John Chang. Advancing climate-resilient flood mitigation:
Utilizing transformer-LSTM for water level forecasting at pumping sta-
tions.Science of the Total Environment, 927:172246, 2024.
[KOM+20] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell
Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage re-
trieval for open-domain question answering. InEMNLP (1), pages 6769–
6781, 2020.
[KPR+19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael
Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin,
122

Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for
question answering research.Transactions of the Association for Compu-
tational Linguistics, 7:453–466, 2019.
[KRI21] Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. Hurdles to progress in
long-form question answering. In Kristina Toutanova, Anna Rumshisky,
Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan
Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors,Proceedings of
the 2021 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pages 4940–
4957, Online, June 2021. Association for Computational Linguistics.
[KS16] Gil Keren and Bj ¨orn Schuller. Convolutional RNN: an enhanced model
for extracting features from sequential data. In2016 International Joint
Conference on Neural Networks (IJCNN), pages 3412–3419. IEEE, 2016.
[KS21] Anil Kag and Venkatesh Saligrama. Training recurrent neural networks
via forward propagation through time. InInternational Conference on
Machine Learning, pages 5189–5200. PMLR, 2021.
[KW16] Thomas N Kipf and Max Welling. Semi-supervised classification with
graph convolutional networks.arXiv preprint arXiv:1609.02907, 2016.
[KYL+24] Dmitrii Kochkov, Janni Yuval, Ian Langmore, Peter Norgaard, Jamie
Smith, Griffin Mooers, Milan Kl ¨ower, James Lottes, Stephan Rasp, Peter
D¨uben, et al. Neural general circulation models for weather and climate.
Nature, pages 1–7, 2024.
[LAC+24] Simon Lang, Mihai Alexe, Matthew Chantry, Jesper Dramsch, Florian
Pinault, Baudouin Raoult, Mariana CA Clare, Christian Lessig, Michael
Maier-Gerber, Linus Magnusson, et al. AIFS-ECMWF’s data-driven fore-
casting system.arXiv preprint arXiv:2406.01465, 2024.
[LBT21] Arturo S Leon, Linlong Bian, and Yun Tang. Comparison of the genetic
algorithm and pattern search methods for forecasting optimal flow releases
in a multi-storage system for flood control.Environmental Modelling &
Software, 145:105198, 2021.
[LCB+24] Zili Liu, Hao Chen, Lei Bai, Wenyuan Li, Wanli Ouyang, Zhengxia Zou,
and Zhenwei Shi. MambaDS: Near-surface meteorological field down-
scaling with topography constrained selective state space modeling.arXiv
preprint arXiv:2408.10854, 2024.
123

[LCC+23] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli,
Richard James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike
Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. InThe
Twelfth International Conference on Learning Representations, 2023.
[LCLG+23] Lizao Li, Rob Carver, Ignacio Lopez-Gomez, Fei Sha, and John Ander-
son. SEEDs: Emulation of weather forecast ensembles with diffusion
models.arXiv preprint arXiv:2306.14066, 2023.
[LCT21] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Learning dense
representations of phrases at scale. InACL, 2021.
[LFX+24] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda
Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.
Deepseek-v3 technical report.arXiv preprint arXiv:2412.19437, 2024.
[LG16] Arturo S Leon and Christopher Goodell. Controlling hec-ras using MAT-
LAB.Environmental modelling & software, 84:339–348, 2016.
[LH21] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3D point
cloud generation. InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 2837–2845, 2021.
[LHN+23] Jussi Leinonen, Ulrich Hamann, Daniele Nerini, Urs Germann, and
Gabriele Franch. Latent diffusion models for generative precipitation
nowcasting with accurate uncertainty quantification.arXiv preprint
arXiv:2304.12891, 2023.
[LKVS14] Arturo S Leon, Elizabeth A Kanashiro, Rachelle Valverde, and Venkatara-
mana Sridhar. Dynamic framework for intelligent control of river flood-
ing: Case study.Journal of Water Resources Planning and Management,
140(2):258–268, 2014.
[LLC+24] Wenyuan Li, Zili Liu, Keyan Chen, Hao Chen, Shunlin Liang, Zhengxia
Zou, and Zhenwei Shi. Deepphysinet: Bridging deep learning and at-
mospheric physics for accurate and continuous weather modeling.arXiv
preprint arXiv:2401.04125, 2024.
[LLD+24] Tengyuan Liu, Lei Liu, Xue Dong, Qiuju Chen, and Bin Li. WPMamba:
Enhanced wind power forecasting model based on Mamba with weather
forecast data. In2024 The 9th International Conference on Power and
Renewable Energy (ICPRE), pages 1429–1435. IEEE, 2024.
124

[LLG+23] Christian Lessig, Ilaria Luise, Bing Gong, Michael Langguth, Scarlet
Stadtler, and Martin Schultz. AtmoRep: A stochastic model of atmo-
sphere dynamics using large scale representation learning.arXiv preprint
arXiv:2308.13280, 2023.
[LLH+24] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele
Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How
language models use long contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024.
[LLQ+24] Xudong Ling, Chaorong Li, Fengqing Qin, Peng Yang, and Yuanyuan
Huang. SRNDiff: Short-term rainfall nowcasting with condition diffusion
model.arXiv preprint arXiv:2402.13737, 2024.
[LLY+22] Chuyao Luo, Xutao Li, Yunming Ye, Shanshan Feng, and Michael K
Ng. Experimental study on generative adversarial network for precipita-
tion nowcasting.IEEE Transactions on Geoscience and Remote Sensing,
60:1–20, 2022.
[LNML14] Lei Li, Farzad Noorian, Duncan JM Moss, and Philip HW Leong. Rolling
window time series prediction using MapReduce. InProceedings of the
2014 IEEE 15th international conference on information reuse and inte-
gration (IEEE IRI 2014), pages 757–764. IEEE, 2014.
[LOG+19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi
Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.
Roberta: A robustly optimized bert pretraining approach.arXiv preprint
arXiv:1907.11692, 2019.
[LRX+25] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Moham-
mad Shoeybi, Bryan Catanzaro, and Wei Ping. NV-embed: Improved
techniques for training LLMs as generalist embedding models. InThe
Thirteenth International Conference on Learning Representations, 2025.
[LSGW+22] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirns-
berger, Meire Fortunato, Alexander Pritzel, Suman Ravuri, Timo
Ewalds, Ferran Alet, Zach Eaton-Rosen, et al. GraphCast: Learn-
ing skillful medium-range global weather forecasting.arXiv preprint
arXiv:2212.12794, 2022.
[LSGW+23] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirns-
berger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach
125

Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global
weather forecasting.Science, 382(6677):1416–1421, 2023.
[LSJ+22] Dong Li, Yelong Shen, Ruoming Jin, Yi Mao, Kuan Wang, and Weizhu
Chen. Generation-augmented query expansion for code retrieval.arXiv
preprint arXiv:2212.10692, 2022.
[LTQC20] Arturo S Leon, Yun Tang, Li Qin, and Duan Chen. A MATLAB frame-
work for forecasting optimal flow releases in a multi-storage system for
flood control.Environmental modelling & software, 125:104618, 2020.
[LYC+22] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai
Xu, Qi Li, and Yueting Chen. Srdiff: Single image super-resolution with
diffusion probabilistic models.Neurocomputing, 479:47–59, 2022.
[LZSJ24] Peiyuan Liu, Tian Zhou, Liang Sun, and Rong Jin. Mitigating time dis-
cretization challenges with weatherODE: A sandwich physics-driven neu-
ral ODE for weather forecasting.arXiv preprint arXiv:2410.06560, 2024.
[LZZW23] Yifan Li, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Diffusion
models for non-autoregressive text generation: A survey.arXiv preprint
arXiv:2303.06574, 2023.
[MAS+24] John A Miller, Mohammed Aldosari, Farah Saeed, Nasid Habib Barna,
Subas Rana, I Budak Arpinar, and Ninghao Liu. A survey of deep learn-
ing and foundation models for time series forecasting.arXiv preprint
arXiv:2401.13912, 2024.
[MAZ+22] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi,
and Hannaneh Hajishirzi. When not to trust language models: Investi-
gating effectiveness of parametric and non-parametric memories.arXiv
preprint arXiv:2212.10511, 2022.
[MIHA18] Ahmed Afif Monrat, Raihan Ul Islam, Mohammad Shahadat Hossain, and
Karl Andersson. A belief rule based flood risk assessment expert system
using real time sensor data streaming. In2018 IEEE 43rd Conference
on Local Computer Networks Workshops (LCN Workshops), pages 38–45.
IEEE, 2018.
[MJ01] Larry R Medsker and LC Jain. Recurrent neural networks.Design and
Applications, 5(64-67):2, 2001.
126

[MKK+20] Bruno Merz, Christian Kuhlicke, Michael Kunz, Massimiliano Pittore,
Andrey Babeyko, David N Bresch, Daniela IV Domeisen, Frauke Feser,
Inga Koszalka, Heidi Kreibich, et al. Impact forecasting to support
emergency management of natural hazards.Reviews of Geophysics,
58(4):e2020RG000704, 2020.
[MM21] Prakamya Mishra and Rohan Mittal. Neuralnere: Neural named entity
relationship extraction for end-to-end climate change knowledge graph
construction. InTackling climate change with machine learning workshop
at ICML, 2021.
[MMM+22] Ashok Mishra, Sourav Mukherjee, Bruno Merz, Vijay P Singh, Daniel B
Wright, Gabriele Villarini, Subir Paul, D Nagesh Kumar, C Prakash Khe-
dun, Dev Niyogi, et al. An overview of flood concepts, challenges, and fu-
ture directions.Journal of hydrologic engineering, 27(6):03122001, 2022.
[MOY21] Jean Maillard, Barlas Oguz, and Wen-tau Yih. Multi-faith retrieval for
question answering. InNAACL, 2021.
[MSL+23] Chongxun Mo, Yue Shen, Xingbi Lei, Huazhen Ban, Yuli Ruan, Shufeng
Lai, Weiyan Cen, and Zhenxiang Xing. Simulation of one-dimensional
dam-break flood routing based on HEC-RAS.Frontiers in Earth Science,
10:1027788, 2023.
[MSO12] W Nor Haizan W Mohamed, Mohd Najib Mohd Salleh, and Abdul Halim
Omar. A comparative study of reduced error pruning method in decision
tree algorithms. In2012 IEEE International conference on control system,
computing and engineering, pages 392–397. IEEE, 2012.
[MSO+20] SR Mounce, Will Shepherd, Sonja Ostojin, Mohamad Abdel-Aal, ANA
Schellart, JD Shucksmith, and SJ Tait. Optimisation of a fuzzy logic-
based local real-time control system for mitigation of sewer flooding using
genetic algorithms.Journal of Hydroinformatics, 22(2):281–295, 2020.
[MWW18] SK Morley, DT Welling, and JR Woodroffe. Perturbed input ensemble
modeling with the space weather modeling framework.Space Weather,
16(9):1330–1347, 2018.
[MXT+23] Minbo Ma, Peng Xie, Fei Teng, Bin Wang, Shenggong Ji, Junbo Zhang,
and Tianrui Li. HiSTGNN: Hierarchical spatio-temporal graph neural net-
work for weather forecasting.Information Sciences, 648:119580, 2023.
127

[MZF+23] Xin Man, Chenghong Zhang, Jin Feng, Changyu Li, and Jie Shao. W-mae:
Pre-trained weather model with masked autoencoder for multi-variable
weather forecasting.arXiv preprint arXiv:2304.08754, 2023.
[MZL23] Zhifeng Ma, Hao Zhang, and Jie Liu. MM-RNN: A multimodal RNN for
precipitation nowcasting.IEEE Transactions on Geoscience and Remote
Sensing, 2023.
[NBK+23] Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta,
and Aditya Grover. Climax: A foundation model for weather and climate.
arXiv preprint arXiv:2301.10343, 2023.
[NCD+24] Grey Nearing, Deborah Cohen, Vusumuzi Dube, Martin Gauch, Oren
Gilon, Shaun Harrigan, Avinatan Hassidim, Daniel Klotz, Frederik
Kratzert, Asher Metzger, et al. Global prediction of extreme floods in
ungauged watersheds.Nature, 627(8004):559–563, 2024.
[NDR+21] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela
Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards
photorealistic image generation and editing with text-guided diffusion
models.arXiv preprint arXiv:2112.10741, 2021.
[NJB+23] Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, and Aditya
Grover. ClimateLearn: Benchmarking machine learning for weather and
climate modeling.arXiv preprint arXiv:2307.01909, 2023.
[NSB+23] Tung Nguyen, Rohan Shah, Hritik Bansal, Troy Arcomano, Romit
Maulik, Veerabhadra Kotamarthi, Ian Foster, Sandeep Madireddy,
and Aditya Grover. Scaling transformer neural networks for skill-
ful and reliable medium-range weather forecasting.arXiv preprint
arXiv:2312.03876, 2023.
[OKP+22] A Oikonomou, Manos Kirtas, Nikos Passalis, George Mourgias-
Alexandris, Miltiadis Moralis-Pegios, Nikos Pleros, and Anastasios Tefas.
A robust, quantization-aware training method for photonic neural net-
works. InInternational conference on engineering applications of neural
networks, pages 427–438. Springer, 2022.
[ON15] Keiron O’Shea and Ryan Nash. An introduction to convolutional neural
networks.arXiv preprint arXiv:1511.08458, 2015.
128

[OTC+00] Jayantha Obeysekera, P Trimble, Luis Cadavid, ER Santee, and Cary
White. Use of climate outlook for water management in South Florida,
USA. InProc., Engineering Jubilee Conf, 2000.
[PAB+17] Gretta T Pecl, Miguel B Ara ´ujo, Johann D Bell, Julia Blanchard, Tim-
othy C Bonebrake, I-Ching Chen, Timothy D Clark, Robert K Colwell,
Finn Danielsen, Birgitta Eveng ˚ard, et al. Biodiversity redistribution under
climate change: Impacts on ecosystems and human well-being.Science,
355(6332):eaai9214, 2017.
[Pal19] Tim Palmer. The ECMWF ensemble prediction system: Looking back
(more than) 25 years and projecting forward 25 years.Quarterly Journal
of the Royal Meteorological Society, 145:12–24, 2019.
[Pet09] Leif E Peterson. K-nearest neighbor.Scholarpedia, 4(2):1883, 2009.
[PGD+24] ˙Ismail Bilal Peker, Sezar G ¨ulbaz, Vahdettin Demir, Osman Orhan, and
Neslihan Beden. Integration of HEC-RAS and HEC-HMS with GIS in
flood modeling and flood hazard mapping.Sustainability, 16(3):1226,
2024.
[PPG23] Tejaskumar Pujari, Anil Kumar Pakina, and Anshul Goel. Explainable ai
and governance: Enhancing transparency and policy frameworks through
retrieval-augmented generation (RAG).IOSR Journal of Computer Engi-
neering, 2023.
[PRWZ02] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a
method for automatic evaluation of machine translation. InProceedings of
the 40th annual meeting of the Association for Computational Linguistics,
pages 311–318, 2002.
[PSGA+24] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Timo Ewalds, Andrew
El-Kadi, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam,
and Matthew Willson. GenCast: Diffusion-based ensemble forecasting
for medium-range weather.arXiv preprint arXiv:2312.15796, May 2024.
[PSH+05] TN Palmer, GJ Shutts, R Hagedorn, FJ Doblas-Reyes, Thomas Jung, and
M Leutbecher. Representing model uncertainty in weather and climate
prediction.Annu. Rev. Earth Planet. Sci., 33(1):163–193, 2005.
[PSH+22] Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja,
Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall,
129

Zongyi Li, Kamyar Azizzadenesheli, et al. Fourcastnet: A global data-
driven high-resolution weather model using adaptive fourier neural oper-
ators.arXiv preprint arXiv:2202.11214, 2022.
[PV18] Philipp Petersen and Felix V oigtlaender. Optimal approximation of piece-
wise smooth functions using deep relu neural networks.Neural Networks,
108:296–330, 2018.
[PZL+24] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao
Hong, Yan Zhang, and Siliang Tang. Graph retrieval-augmented genera-
tion: A survey.arXiv preprint arXiv:2408.08921, 2024.
[QCJ+24] Haoyu Qin, Yungang Chen, Qianchuan Jiang, Pengchao Sun, Xiancai Ye,
and Chao Lin. MetMamba: Regional weather forecasting with spatial-
temporal mamba model.arXiv preprint arXiv:2408.06400, 2024.
[QMX+21] Wenchao Qi, Chao Ma, Hongshi Xu, Zifan Chen, Kai Zhao, and Hao
Han. A review on applications of urban flood models in flood mitigation
strategies.Natural Hazards, 108:31–62, 2021.
[Qui86] J. Ross Quinlan. Induction of decision trees.Machine learning, 1:81–106,
1986.
[RA24] Motiur Rahman and Md Shahjahan Ali. Drivers of tidal flow variability
in the Pussur fluvial estuary: A numerical study by HEC-RAS.Heliyon,
10(4), 2024.
[Ram24] Vivek Ramavajjala. HEAL-ViT: Vision transformers on a spherical mesh
for medium-range weather forecasting.arXiv preprint arXiv:2403.17016,
2024.
[RBL+22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and
Bj¨orn Ommer. High-resolution image synthesis with latent diffusion mod-
els. InProceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 10684–10695, 2022.
[RDS+20] Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn,
Soukayna Mouatadid, and Nils Thuerey. WeatherBench: a benchmark
data set for data-driven weather forecasting.Journal of Advances in Mod-
eling Earth Systems, 12(11):e2020MS002203, 2020.
130

[RFB15] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolu-
tional networks for biomedical image segmentation. InMedical image
computing and computer-assisted intervention–MICCAI 2015: 18th in-
ternational conference, October 5-9, 2015, proceedings, part III 18, pages
234–241, 2015.
[RLW+21] Suman Ravuri, Karel Lenc, Matthew Willson, Dmitry Kangin, Remi
Lam, Piotr Mirowski, Megan Fitzsimons, Maria Athanassiadou, Sheleem
Kashem, Sam Madge, et al. Skilful precipitation nowcasting using deep
generative models of radar.Nature, 597(7878):672–677, 2021.
[RP07] MJ Rodwell and TN Palmer. Using numerical weather prediction to assess
climate models.Quarterly Journal of the Royal Meteorological Society:
A journal of the atmospheric sciences, applied meteorology and physical
oceanography, 133(622):129–146, 2007.
[RRMA19] Khaiwal Ravindra, Preety Rattan, Suman Mor, and Ashutosh Nath Aggar-
wal. Generalized additive models: Building evidence of air pollution, cli-
mate change and human health.Environment international, 132:104987,
2019.
[RSP+25] Rahuul Rangaraj, Jimeng Shi, Rajendra Paudel, Giri Narasimhan, and
Yanzhao Wu. Retrieval-augmented water level forecasting for everglades.
arXiv preprint arXiv:2508.04888, 2025.
[RSS+25] Rahuul Rangaraj, Jimeng Shi, Azam Shirali, Rajendra Paudel, Yanzhao
Wu, and Giri Narasimhan. How effective are large time series models
in hydrology? a study on water level forecasting in everglades.arXiv
preprint arXiv:2505.01415, 2025.
[RT10] David Rogers and Vladimir Tsirkunov. Global assessment report on disas-
ter risk reduction: costs and benefits of early warning systems. Technical
report, The World Bank, 2010.
[RTLC+22] Michael O Rivett, Laurent-Charles Tremblay-Levesque, Ruth Carter,
Rudi CH Thetard, Morris Tengatenga, Ann Phoya, Emma Mbalame, Ed-
win Mchilikizo, Steven Kumwenda, Prince Mleta, et al. Acute health risks
to community hand-pumped groundwater supplies following Cyclone Idai
flooding.Science of The Total Environment, 806:150598, 2022.
131

[RUB19] Vinay Ashok Rangari, NV Umamahesh, and CM Bhatt. Assessment of
inundation risk in urban floods using HEC-RAS 2D.Modeling Earth Sys-
tems and Environment, 5(4):1839–1851, 2019.
[Rud16] Sebastian Ruder. An overview of gradient descent optimization algo-
rithms.arXiv preprint arXiv:1609.04747, 2016.
[RVB22] Saeed Rokooei, Farshid Vahedifard, and Solomon Belay. Percep-
tions of civil engineering and construction students toward community
and infrastructure resilience.Journal of Civil Engineering Education,
148(1):04021015, 2022.
[RW94] Stephen E Robertson and Steve Walker. Some simple effective approx-
imations to the 2-poisson model for probabilistic weighted retrieval. In
SIGIR’94: Proceedings of the Seventeenth Annual International ACM-
SIGIR Conference on Research and Development in Information Re-
trieval, organised by Dublin City University, pages 232–241. Springer,
1994.
[SABA21] Max Schwenzer, Muzaffer Ay, Thomas Bergs, and Dirk Abel. Review
on model predictive control: An engineering perspective.The Inter-
national Journal of Advanced Manufacturing Technology, 117(5):1327–
1349, 2021.
[SAG+23] Chaopeng Shen, Alison P Appling, Pierre Gentine, Toshiyuki Bandai,
Hoshin Gupta, Alexandre Tartakovsky, Marco Baity-Jesi, Fabrizio Feni-
cia, Daniel Kifer, Li Li, et al. Differentiable modeling to unify machine
learning and physical models and advance geosciences.arXiv preprint
arXiv:2301.04027, 2023.
[Sar21a] Iqbal H Sarker. Deep learning: a comprehensive overview on techniques,
taxonomy, applications and research directions.SN computer science,
2(6):1–20, 2021.
[Sar21b] Iqbal H Sarker. Machine learning: Algorithms, real-world applications
and research directions.SN computer science, 2(3):160, 2021.
[SBB+20] Alen Shrestha, Linkon Bhattacharjee, Sudip Baral, Balbhadra Thakur,
Neekita Joshi, Ajay Kalra, and Ritu Gupta. Understanding suitability
of MIKE 21 and HEC-RAS for 2D floodplain modeling. InWorld envi-
ronmental and water resources congress 2020, pages 237–253. American
Society of Civil Engineers Reston, V A, 2020.
132

[SBX15] D Schwanenberg, BPJ Becker, and M Xu. The open real-time control
(RTC)-tools software framework for modeling RTC in water resources
sytems.Journal of Hydroinformatics, 17(1):130–148, 2015.
[SCC+22] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho,
Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-
image diffusion models. InACM SIGGRAPH 2022 Conference Proceed-
ings, pages 1–10, 2022.
[SCW+15] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin
Wong, and Wang-chun Woo. Convolutional LSTM network: A machine
learning approach for precipitation nowcasting.Advances in neural infor-
mation processing systems, 28, 2015.
[SDP19] Shadab Shishegar, Sophie Duchesne, and Genevi `eve Pelletier. An inte-
grated optimization and rule-based approach for predictive real time con-
trol of urban stormwater management systems.Journal of Hydrology,
577:124000, 2019.
[SEH+20] Casper Kaae Sønderby, Lasse Espeholt, Jonathan Heek, Mostafa De-
hghani, Avital Oliver, Tim Salimans, Shreya Agrawal, Jason Hickey, and
Nal Kalchbrenner. MetNet: A neural weather model for precipitation fore-
casting.arXiv preprint arXiv:2003.12140, 2020.
[SGB+19] Jeffrey M Sadler, Jonathan L Goodall, Madhur Behl, Mohamed M Morsy,
Teresa B Culver, and Benjamin D Bowes. Leveraging open source
software and parallel computing for model predictive control of urban
drainage systems using EPA-SWMM5.Environmental Modelling & Soft-
ware, 120:104484, 2019.
[SGB+20] Jeffrey M Sadler, Jonathan L Goodall, Madhur Behl, Benjamin D Bowes,
and Mohamed M Morsy. Exploring real-time control of stormwater sys-
tems for mitigating flood risk due to sea level rise.Journal of Hydrology,
583:124571, 2020.
[SGMS18] JM Sadler, JL Goodall, MM Morsy, and K Spencer. Modeling urban
coastal flood severity from crowd-sourced flood reports using Poisson re-
gression and Random Forest.Journal of hydrology, 559:43–55, 2018.
[SGT+08] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and
Gabriele Monfardini. The graph neural network model.IEEE transactions
on neural networks, 20(1):61–80, 2008.
133

[SHT+26] Jimeng Shi, Wei Hu, Runchu Tian, Bowen Jin, Wonbin Kweon,
SeongKu Kang, Yunfan Kang, Dingqi Ye, Sizhe Zhou, Shaowen Wang,
et al. Multicube-rag for multi-hop question answering.arXiv preprint
arXiv:2602.15898, 2026.
[SJHN24] Jimeng Shi, Bowen Jin, Jiawei Han, and Giri Narasimhan. CoDiCast:
Conditional diffusion model for weather prediction with uncertainty quan-
tification.arXiv preprint arXiv:2409.05975, 2024.
[SJN22] Jimeng Shi, Mahek Jain, and Giri Narasimhan. Time series forecasting
using various deep learning models.International Journal of Computer
and Systems Engineering, 16(6):224–232, 2022.
[SK21] Shruti Sachdeva and Bijendra Kumar. Comparison of gradient boosted
decision trees and random forest for groundwater potential mapping in
Dholpur (Rajasthan), India.Stochastic Environmental Research and Risk
Assessment, 35(2):287–306, 2021.
[SLG+24] Tao Shen, Guodong Long, Xiubo Geng, Chongyang Tao, Yibin Lei, Tianyi
Zhou, Michael Blumenstein, and Daxin Jiang. Retrieval-augmented re-
trieval: Large language models are strong zero-shot retriever. InFindings
of the Association for Computational Linguistics ACL 2024, pages 15933–
15946, 2024.
[SME20] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion
implicit models.arXiv preprint arXiv:2010.02502, 2020.
[SMK+22] Talha Siddique, Md Shaad Mahmud, Amy M Keesee, Chigomezyo M
Ngwira, and Hyunju Connor. A survey of uncertainty quantification in
machine learning for space weather prediction.Geosciences, 12(1):27,
2022.
[SMRFK+20] Michelle Sapitang, Wanie M. Ridwan, Khairul Faizal Kushiar, Ali Na-
jah Ahmed, and Ahmed El-Shafie. Machine learning application in reser-
voir water level forecasting for sustainable hydropower generation strat-
egy.Sustainability, 12(15):6121, 2020.
[SMS+23] Jimeng Shi, Rukmangadh Myana, Vitalii Stebliankin, Azam Shirali, and
Giri Narasimhan. Explainable parallel RCNN with novel feature represen-
tation for time series forecasting. InInternational Workshop on Advanced
Analytics and Learning on Temporal Data, pages 56–75, Torino, Italy,
2023. Springer.
134

[SP11] Julia Slingo and Tim Palmer. Uncertainty in weather and climate predic-
tion.Philosophical Transactions of the Royal Society A: Mathematical,
Physical and Engineering Sciences, 369(1956):4751–4767, 2011.
[SPW+23] Arbaaz Aziz Shaikh, Azazkhan Ibrahimkhan Pathan, Sahita Ibopishak
Waikhom, Prasit Girish Agnihotri, Md Nazrul Islam, and Sudhir Kumar
Singh. Application of latest HEC-RAS version 6 for 2D hydrodynamic
modeling through GIS framework: A case study from coastal urban flood-
plain in India.Modeling Earth Systems and Environment, 9(1):1369–
1385, 2023.
[SRT+24] Johannes Schmude, Sujit Roy, Will Trojak, Johannes Jakubik,
Daniel Salles Civitarese, Shraddha Singh, Julian Kuehnert, Kumar Ankur,
Aman Gupta, Christopher E Phillips, et al. Prithvi WxC: Foundation
model for weather and climate.arXiv preprint arXiv:2409.13598, 2024.
[SS19] Alexander Y Sun and Bridget R Scanlon. How can big data and machine
learning benefit environment and water management: a survey of meth-
ods, applications, and future directions.Environmental Research Letters,
14(7):073001, 2019.
[SSJ+25] J Shi, A Shirali, B Jin, S Zhou, W Hu, R Rangaraj, S Wang, J Han,
Z Wang, U Lall, et al. Deep learning and foundation models for weather
prediction: a survey (2025).arXiv preprint arXiv:2501.06907, 2025.
[SSN23] Jimeng Shi, Vitalii Stebliankin, and Giri Narasimhan. The power of ex-
plainability in forecast-informed deep learning models for flood mitiga-
tion.arXiv preprint arXiv:2310.19166, 2023.
[SSP24] Hira Saleem, Flora Salim, and Cormac Purcell. Conformer: Embedding
continuous attention in vision transformer for weather forecasting.arXiv
preprint arXiv:2402.17966, 2024.
[SSW+23] Jimeng Shi, Vitalii Stebliankin, Zhaonan Wang, Shaowen Wang, and Giri
Narasimhan. Graph transformer network for flood forecasting with het-
erogeneous covariates.arXiv preprint arXiv:2310.07631, 2023.
[SSX+26] Jiashuo Sun, Jimeng Shi, Yixuan Xie, Saizhuo Wang, Jash Rajesh Parekh,
Pengcheng Jiang, Zhiyi Shi, Jiajun Fan, Qinglong Zheng, Peiran Li, et al.
Retrieval is cheap, show me the code: Executable multi-hop reasoning for
retrieval-augmented generation.arXiv preprint arXiv:2605.12975, 2026.
135

[STS16] Amanpreet Singh, Narina Thakur, and Aakanksha Sharma. A review of
supervised machine learning algorithms. In2016 3rd international con-
ference on computing for sustainable global development (INDIACom),
pages 1310–1315. Ieee, 2016.
[SVDM95] Johan AK Suykens, Joos PL Vandewalle, and Bart L De Moor.Artificial
neural networks for modelling and control of non-linear systems. Springer
Science & Business Media, 1995.
[SVS+25] Manoochehr Shirzaei, Farshid Vahedifard, Nitheshnirmal Sadhasivam,
Leonard Ohenhen, Oluwaseyi Dasho, Ashutosh Tiwari, Susanna Werth,
Mohammed Azhar, Yunxia Zhao, Robert J Nicholls, et al. Aging dams,
political instability, poor human decisions and climate change: recipe for
human disaster.npj Natural Hazards, 2(1):5, 2025.
[SXS+26] Jiashuo Sun, Yixuan Xie, Jimeng Shi, Shaowen Wang, and Jiawei Han.
Tasr-rag: Taxonomy-guided structured reasoning for retrieval-augmented
generation.arXiv preprint arXiv:2603.09341, 2026.
[SYL+24] Jimeng Shi, Zeda Yin, Arturo Leon, Jayantha Obeysekera, and Giri
Narasimhan. FIDLAR: Forecast-informed deep learning architecture for
flood mitigation.arXiv preprint arXiv:2402.13371, 2024.
[SYM+23] Jimeng Shi, Zeda Yin, Rukmangadh Myana, Khandker Ishtiaq, Anupama
John, Jayantha Obeysekera, Arturo Leon, and Giri Narasimhan. Deep
learning models for water stage predictions in South Florida.arXiv
preprint arXiv:2306.15907, 2023.
[SZJ+25] Jimeng Shi, Sizhe Zhou, Bowen Jin, Wei Hu, Shaowen Wang,
Giri Narasimhan, and Jiawei Han. Hypercube-RAG: Hypercube-
based retrieval-augmented generation for in-domain scientific question-
answering.arXiv preprint arXiv:2505.19288, 2025.
[Tan20] Suryakanthi Tangirala. Evaluating the impact of GINI index and in-
formation gain on classification using decision tree classifier algorithm.
International Journal of Advanced Computer Science and Applications,
11(2):612–619, 2020.
[TDT+24] Yujin Tang, Peijie Dong, Zhenheng Tang, Xiaowen Chu, and Junwei
Liang. VMRNN: Integrating Vision Mamba and LSTM for efficient and
accurate spatiotemporal forecasting. InProceedings of the IEEE/CVF
136

Conference on Computer Vision and Pattern Recognition, pages 5663–
5673, 2024.
[TPB14] Andrew Trotman, Antti Puurula, and Blake Burgess. Improvements to
BM25 and language models examined. InProceedings of the 19th Aus-
tralasian Document Computing Symposium, pages 58–65, 2014.
[TZC+18] Fangbo Tao, Chao Zhang, Xiusi Chen, Meng Jiang, Tim Hanratty, Lance
Kaplan, and Jiawei Han. Doc2cube: Allocating documents to text cube
without labeled data. In2018 IEEE International Conference on Data
Mining (ICDM), pages 1260–1265. IEEE, 2018.
[VBMM22] Shashank Reddy Vadyala, Sai Nethra Betgeri, John C Matthews, and Eliz-
abeth Matthews. A review of physics-based machine learning in civil en-
gineering.Results in Engineering, 13:100316, 2022.
[VH24] George Van Houtven. Economic value of flood forecasts and early warn-
ing systems: A review.Natural Hazards Review, 25(4):03124002, 2024.
[VHG24] Yogesh Verma, Markus Heinonen, and Vikas Garg. ClimODE: Cli-
mate and weather forecasting with physics-informed neural odes.arXiv
preprint arXiv:2404.10024, 2024.
[VM19] Rikard Vinge and Tomas McKelvey. Understanding support vector ma-
chines with polynomial kernels. In2019 27th European signal processing
conference (EUSIPCO), pages 1–5. IEEE, 2019.
[VMWW18] Evert Vermuyten, Pieter Meert, Vincent Wolfs, and Patrick Willems.
Combining model predictive control with a reduced genetic algorithm for
real-time flood control.Journal of Water Resources Planning and Man-
agement, 144(2):04017083, 2018.
[VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention
is all you need.Advances in neural information processing systems, 30,
2017.
[WCW+24] Binqing Wu, Weiqi Chen, Wengwei Wang, Bingqing Peng, Liang
Sun, and Ling Chen. WeatherGNN: Exploiting meteo-and spatial-
dependencies for local numerical weather prediction bias-correction. In
Proceedings of the International Joint Conference on Artificial Intelli-
gence, pages 2433–2441, 2024.
137

[WFL23] Rui Wang, Jimmy CH Fung, and Alexis KH Lau. Physical-Dynamic-
Driven AI-synthetic precipitation nowcasting using task-segmented gen-
erative model.Geophysical Research Letters, 50(21):e2023GL106084,
2023.
[WHC+24] Shijie Wang, Jiani Huang, Zhikai Chen, Yu Song, Wenzhuo Tang, Haitao
Mao, Wenqi Fan, Hui Liu, Xiaorui Liu, Dawei Yin, et al. Graph machine
learning in the era of large language models (llms).ACM Transactions on
Intelligent Systems and Technology, 2024.
[WJH+23] Zhaonan Wang, Bowen Jin, Wei Hu, Minhao Jiang, Seungyeon Kang,
Zhiyuan Li, Sizhe Zhou, Jiawei Han, and Shaowen Wang. Geospatial
knowledge hypercube. InProceedings of the 31st ACM International
Conference on Advances in Geographic Information Systems, pages 1–4,
2023.
[WLL+25] Gelan Wang, Yu Liu, Shukai Liu, Ling Zhang, and Liqun Yang. REM-
FLOW: RAG-enhanced multi-factor rainfall flooding warning in sponge
airports via large language model.International Journal of Machine
Learning and Cybernetics, pages 1–21, 2025.
[WWZ+22] Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, S Yu
Philip, and Mingsheng Long. PredRNN: A recurrent neural network for
spatiotemporal predictive learning.IEEE Transactions on Pattern Analy-
sis and Machine Intelligence, 45(2):2208–2225, 2022.
[WYH+24] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Ma-
jumder, and Furu Wei. Multilingual e5 text embeddings: A technical
report.arXiv preprint arXiv:2402.05672, 2024.
[WYW+22] Yi Wang, Wenhao Yu, Yining Wang, Chenguang Li, Soroush Liu, and
Hoifung Poon. Text embeddings by weakly-supervised contrastive pre-
training.arXiv preprint arXiv:2212.03533, 2022.
[WZP20] Xizhao Wang, Yanxia Zhao, and Farhad Pourpanah. Recent advances in
deep learning.International Journal of Machine Learning and Cybernet-
ics, 11:747–750, 2020.
[WZZ+22] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi
Yan, and Liang Sun. Transformers in time series: A survey.arXiv preprint
arXiv:2202.07125, 2022.
138

[XBX+23] Yi Xiao, Lei Bai, Wei Xue, Kang Chen, Tao Han, and Wanli Ouyang.
Fengwu-4DVar: Coupling the data-driven weather forecasting model with
4D variational assimilation.arXiv preprint arXiv:2312.12455, 2023.
[XLC+21] Yongjun Xu, Xin Liu, Xin Cao, Changping Huang, Enke Liu, Sen Qian,
Xingchen Liu, Yanjun Wu, Fengliang Dong, Cheng-Wei Qiu, et al. Arti-
ficial intelligence: A powerful paradigm for scientific research.The Inno-
vation, 2(4), 2021.
[XQS+24] Luwen Xu, Jiwei Qin, Dezhi Sun, Yuanyuan Liao, and Jiong Zheng. PF-
former: A time-series forecasting model for short-term precipitation fore-
casting.IEEE Access, 2024.
[XXL+21] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N.
Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest
neighbor negative contrastive learning for dense text retrieval. InInter-
national Conference on Learning Representations, 2021.
[YBH+23] Zeda Yin, Linglong Bian, Beichao Hu, Jimeng Shi, and Arturo S Leon.
Physic-informed neural network approach coupled with boundary condi-
tions for solving 1D steady shallow water equations for riverine system.
InWorld Environmental and Water Resources Congress 2023, pages 280–
288, 2023.
[YC20] Shun-Nien Yang and Li-Chiu Chang. Regional inundation forecasting
using machine learning techniques with the internet of things.Water,
12(6):1578, 2020.
[YGC+24] Qidong Yang, Jonathan Giezendanner, Daniel Salles Civitarese, Johannes
Jakubik, Eric Schmitt, Anirban Chandra, Jeremy Vila, Detlef Hohl, Chris
Hill, Campbell Watson, et al. Multi-modal graph neural networks for lo-
calized off-grid weather forecasting.arXiv preprint arXiv:2410.12938,
2024.
[YJC+18] Jun Yan, Jiaming Jin, Furong Chen, Guo Yu, Hailong Yin, and Wenjia
Wang. Urban flash flood forecast using support vector machine and nu-
merical simulation.Journal of Hydroinformatics, 20(1):221–231, 2018.
[YLY+24] Demin Yu, Xutao Li, Yunming Ye, Baoquan Zhang, Chuyao Luo, Kuai
Dai, Rui Wang, and Xunlai Chen. Diffcast: A unified framework via
residual diffusion for precipitation nowcasting. InProceedings of the
139

IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 27758–27767, 2024.
[YSB+25] Zeda Yin, Jimeng Shi, Linlong Bian, William H Campbell, Sumit R Zanje,
Beichao Hu, and Arturo S Leon. Physics-informed neural network ap-
proach for solving the one-dimensional unsteady shallow-water equations
in riverine systems.Journal of Hydraulic Engineering, 151(1):04024060,
2025.
[YSH+24] Zeda Yin, Yasaman Saadati, Beichao Hu, Arturo S Leon, M Hadi Amini,
and Dwayne McDaniel. Fast high-fidelity flood inundation map gen-
eration by super-resolution techniques.Journal of Hydroinformatics,
26(1):319–336, 2024.
[YWMZ25] Shijin Yuan, Guansong Wang, Bin Mu, and Feifan Zhou. Tianxing:
A linear complexity transformer model with explicit attention decay for
global weather forecasting.Advances in Atmospheric Sciences, 42(1):9–
25, 2025.
[YYY+24] Jie Yin, Yuhan Yang, Dapeng Yu, Ning Lin, Robert Wilby, Stuart Lane,
Bindong Sun, Jeremy Bricker, Nigel Wright, Lili Yang, et al. Strategic
storm flood evacuation planning for large coastal cities enables more ef-
fective transfer of elderly populations.Nature Water, pages 1–11, 2024.
[YYZ17] Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph con-
volutional networks: A deep learning framework for traffic forecasting.
arXiv preprint arXiv:1709.04875, 2017.
[ZBHB+21] Manizhe Zarei, Omid Bozorg-Haddad, Sahar Baghban, Mohammad Del-
pasand, Erfan Goharian, and Hugo A Lo ´aiciga. Machine-learning algo-
rithms for forecast-informed reservoir operation (FIRO) to reduce flood
damages.Scientific reports, 11(1):24295, 2021.
[ZBN+24] Pengcheng Zhao, Jiang Bian, Zekun Ni, Weixin Jin, Jonathan Weyn, Zu-
liang Fang, Siqi Xiang, Haiyu Dong, Bin Zhang, Hongyu Sun, et al.
OMG-HD: A high-resolution AI weather model for end-to-end forecasts
from observations.arXiv preprint arXiv:2412.18239, 2024.
[ZD20] Zao Zhang and Yuan Dong. Temperature forecasting via convolutional
recurrent neural networks based on time-series data.Complexity, 2020:1–
8, 2020.
140

[ZF23] Ond ˇrej Zelen `y and Tomas Fryza. Multi-branch multi layer perceptron: A
solution for precise regression using machine learning. In2023 33rd In-
ternational Conference Radioelektronika (RADIOELEKTRONIKA), pages
1–5. IEEE, 2023.
[Zha16] Zhongheng Zhang. Introduction to machine learning: k-nearest neighbors.
Annals of translational medicine, 4(11):218, 2016.
[ZJM+24] Yang Zhang, Hanlei Jin, Dan Meng, Jun Wang, and Jinghua Tan. A
comprehensive survey on process-oriented automatic text summarization
with exploration of llm-based methods.arXiv preprint arXiv:2403.02901,
2024.
[ZKW+20] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and
Yoav Artzi. BERTScore: Evaluating text generation with BERT. InInter-
national Conference on Learning Representations, 2020.
[ZLC+23] Yuchen Zhang, Mingsheng Long, Kaiyuan Chen, Lanxiang Xing,
Ronghua Jin, Michael I Jordan, and Jianmin Wang. Skilful nowcasting
of extreme precipitation with nowcastnet.Nature, 619(7970):526–532,
2023.
[ZLC+25] Xu Zheng, Chaohao Lin, Sipeng Chen, Zhuomin Chen, Jimeng Shi, Wei
Cheng, Jayantha Obeysekera, Jason Liu, and Dongsheng Luo. Sf2bench:
Evaluating data-driven models for compound flood forecasting in south
florida.arXiv preprint arXiv:2506.04281, 2025.
[ZM13] Babak Zamanlooy and Mitra Mirhassani. Efficient vlsi implementation
of neural networks with hyperbolic tangent activation function.IEEE
Transactions on Very Large Scale Integration (VLSI) Systems, 22(1):39–
48, 2013.
[ZRA23] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding condi-
tional control to text-to-image diffusion models. InProceedings of the
IEEE/CVF International Conference on Computer Vision, pages 3836–
3847, 2023.
[ZWNW21] Yuerong Zhou, Wenyan Wu, Rory Nathan, and Quan J Wang. A rapid
flood inundation modelling framework using deep learning with spa-
tial reduction and reconstruction.Environmental Modelling & Software,
143:105112, 2021.
141

[ZXW+24] Xiao Xiang Zhu, Zhitong Xiong, Yi Wang, Adam J Stewart, Konrad Hei-
dler, Yuanyuan Wang, Zhenghang Yuan, Thomas Dujardin, Qingsong Xu,
and Yilei Shi. On the foundations of earth and climate foundation models.
arXiv preprint arXiv:2405.04285, 2024.
[ZYW+23] Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang.
Toolqa: A dataset for llm question answering with external tools.Ad-
vances in Neural Information Processing Systems, 36:50117–50143,
2023.
[ZYX+25] Yunyi Zhang, Ruozhen Yang, Xueqiang Xu, Rui Li, Jinfeng Xiao, Jiaming
Shen, and Jiawei Han. Teleclass: Taxonomy enrichment and llm-enhanced
hierarchical text classification with minimal supervision. InProceedings
of the ACM on Web Conference 2025, pages 2032–2042, 2025.
142

VITA
JIMENG SHI
2021.08-present Graduate Research Assistant
Florida International University
Miami, FL, USA
2021.08-present Researcher (Remote)
NSF I-GUIDE,
Urbana Champaign, IL, USA
2019.01-2021.05 M.S., Computer Science
Florida International University
Miami, FL, USA
2014.09-2018.07 B.S., Mechanical Engineering
Tianjin U. of Science and Technology
Tianjin, China
2024.05-2024.08 Momentum Research Fellow
NSF LEAP, Columbia University
New York City, NY , USA
2020.09-2020.12 Lid Vizion Company
Miami, FL, USA
PUBLICATIONS (∗represents equal contribution.)
Jimeng Shi, Zeda Yin, Arturo Leon, Jayantha Obeysekera, Giri Narasimhan.FIDLAR:
Forecast-Informed Deep Learning Approaches for Flood Mitigation.In Proceedings of
the AAAI Conference on Artificial Intelligence, vol. 39, no. 27, pp. 28377-28385. 2025.
Jimeng Shi, Bowen Jin, Jiawei Han, Sundararaman Gopalakrishnan, Giri Narasimhan.
CoDiCast: Conditional Diffusion Model for Global Weather Prediction with Uncertainty
Quantification.The 34th International Joint Conference on Artificial Intelligence. 2025.
(To appear)
Jimeng Shi, Azam Shirali, Giri Narasimhan.ReFine: Boosting Time Series Prediction of
Extreme Events by Reweighting and Fine-tuning. In Proceedings of 2024 IEEE Interna-
tional Conference on Big Data (BigData), pp. 1450-1457. IEEE, 2024.
Jimeng Shi∗, Zeda Yin∗, Rukmangadh Myana, Khandker Ishtiaq, Anupama John, Jayan-
tha Obeysekera, Arturo Leon, Giri Narasimhan. Journal of Water Resources Planning and
Management (In press). 2025.
143

Jimeng Shi, Sizhe Zhou, Bowen Jin, Wei Hu, Shaowen Wang, Giri Narasimhan, Jiawei
Han.Hypercube-RAG: Hypercube Retrieval-Augmented Generation for In-domain Sci-
entific Question-Answering. arXiv preprint, 2025. (Under review)
Jimeng Shi, Rukmangadh Myana, Vitalii Stebliankin, Azam Shirali, Giri Narasimhan.
Explainable Parallel RCNN with Novel Feature Representation for Time Series Forecast-
ing. In International Workshop on Advanced Analytics and Learning on Temporal Data,
pp. 56-75. 2023.
Jimeng Shi, Azam Shirali, Bowen Jin, Sizhe Zhou, Wei Hu, Rahuul Rangaraj, Shaowen
Wang, Jiawei Han, Zhaonan Wang, Upmanu Lall, Yanzhao Wu, Leonardo Bobadilla, Giri
Narasimhan.Deep Learning and Foundation Models for Weather Prediction: A Survey.
arXiv preprint. 2025. (Under review)
Jimeng Shi, Mahek Jain, Giri Narasimhan.Time Series Forecasting (TSF) Using Vari-
ous Deep Learning Models.International Conference of Machine Learning Applications
(ICMLA) 2022.
Rahuul Rangaraj∗, Jimeng Shi∗, Azam Shirali, Rajendra Paudel, Yanzhao Wu, and Giri
Narasimhan.Retrieval-Augmented Water Level Forecasting for Everglades. arXiv preprint.
2025. (Under review)
Rahuul Rangaraj∗, Jimeng Shi∗, Azam Shirali, Rajendra Paudel, Yanzhao Wu, and Giri
Narasimhan.How Effective are Large Time Series Models in Hydrology? A Study on
Water Level Forecasting in Everglades. arXiv preprint. 2025. (Under review)
Vitalii Stebliankin, Azam Shirali, Prabin Baral, Jimeng Shi, Prem Chapagain, Kalai
Mathee, Giri Narasimhan.Evaluating protein binding interfaces with transformer net-
works.Nature Machine Intelligence, 5(9), 1042-1053. 2023.
Azam Shirali, Vitalii Stebliankin, Jimeng Shi, Giri Narasimhan.A Comprehensive Survey
of Scoring Functions for Protein Docking Models.BMC bioinformatics, 26(1), 25. 2025.
Zichuan Liu, Tianchun Wang, Jimeng Shi, Xu Zheng, Zhuomin Chen, Lei Song, Wenqian
Dong, Jayantha Obeysekera, Farhad Shirani, Dongsheng Luo.TimeX++: Learning Time-
Series Explanations with Information Bottleneck. In Proceedings of the 41st International
Conference on Machine Learning, no. 1297, pp. 32062-32082. 2024.
Zeda Yin, Jimeng Shi, Linlong Bian, William Campbell, Sumit Zanje, Arturo Leon.
Physics-Informed Neural Network Approach for Solving the One-Dimensional Unsteady
Shallow-Water Equations in Riverine Systems. Journal of Hydraulic Engineering, 151(1),
04024060. 2025.
Zeda Yin, Linlong Bian, Beichao Hu, Jimeng Shi, Arturo Leon.Physic-Informed Neu-
ral Network Approach Coupled with Boundary Conditions for Solving 1D Steady Shal-
low Water Equations for Riverine System. In World Environmental and Water Resources
Congress 2023, pp. 280-288. 2023.
144

PRESENTATIONS (∗represents equal contribution.)
Jimeng Shi et al.,Hypercube-RAG: Hypercube Retrieval-augmented Generation for
In-domain Scientific Question-Answeringat the I-GUIDE Forum, Chicago, June 2025.
Jimeng Shi et al.,Hypercube-RAG: Hypercube Retrieval-augmented Generation for
In-domain Scientific Question-Answeringat the startup company (GRaiL), Miami, May
2025.
Jimeng Shi et al., FIDLAR: Forecast-Informed Deep Learning Approaches for Flood
Mitigationat Climate AI Journal Club, Florida International University, Miami, March
2025.
Jimeng Shi et al., FIDLAR: Forecast-Informed Deep Learning Approaches for Flood
Mitigationat AAAI’25, Philadelphia, Pennsylvania, Feb. 2025.
Jimeng Shi et al.,AI4Science: Coastal Flood Management and Global Weather Predic-
tionat Hurricane Research Division/AOML/NOAA, Miami, Feb. 2025.
Jimeng Shi et al.,ReFine: Boosting Time Series Prediction of Extreme Events by Reweight-
ing and Fine-tuningat IEEE International Conference on Big Data, Washington DC, Dec.
2024.
Jimeng Shi et al.,Explainable Deep Learning Models for Flood Prediction and Mitiga-
tionandgenerative diffusion model for global weather forecastat Columbia Water Center,
Columbia University, July 2024.
Jimeng Shi et al.,The Power of Explainability in Forecast-Informed Deep Learning Mod-
els for Flood Mitigationat NeurIPS 2023 workshop on Climate Change AI.
Jimeng Shi et al.,Graph Transformer Network for Flood Forecast in A Coastal Systemat
NSF I-GUIDE Forum, New York City, Oct. 2023.
Jimeng Shi et al.,Deep Learning Models for Water Level Predictionat I-GUIDE All
Hands Meeting (AHM), Chicago, Sep. 2022.
Jimeng Shi et al.,Time Series Forecasting (TSF) Using Various Deep Learning Modelsat
International Conference of Machine Learning Applications (ICMLA) 2022.
145
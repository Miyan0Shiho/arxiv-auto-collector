# A Comprehensive Survey of Large AI Models for Future Communications: Foundations, Applications and Challenges

**Authors**: Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Merouane Debbah, Dusit Niyato, Zhu Han

**Published**: 2025-05-06 14:09:29

**PDF URL**: [http://arxiv.org/pdf/2505.03556v1](http://arxiv.org/pdf/2505.03556v1)

## Abstract
The 6G wireless communications aim to establish an intelligent world of
ubiquitous connectivity, providing an unprecedented communication experience.
Large artificial intelligence models (LAMs) are characterized by significantly
larger scales (e.g., billions or trillions of parameters) compared to typical
artificial intelligence (AI) models. LAMs exhibit outstanding cognitive
abilities, including strong generalization capabilities for fine-tuning to
downstream tasks, and emergent capabilities to handle tasks unseen during
training. Therefore, LAMs efficiently provide AI services for diverse
communication applications, making them crucial tools for addressing complex
challenges in future wireless communication systems. This study provides a
comprehensive review of the foundations, applications, and challenges of LAMs
in communication. First, we introduce the current state of AI-based
communication systems, emphasizing the motivation behind integrating LAMs into
communications and summarizing the key contributions. We then present an
overview of the essential concepts of LAMs in communication. This includes an
introduction to the main architectures of LAMs, such as transformer, diffusion
models, and mamba. We also explore the classification of LAMs, including large
language models (LLMs), large vision models (LVMs), large multimodal models
(LMMs), and world models, and examine their potential applications in
communication. Additionally, we cover the training methods and evaluation
techniques for LAMs in communication systems. Lastly, we introduce optimization
strategies such as chain of thought (CoT), retrieval augmented generation
(RAG), and agentic systems. Following this, we discuss the research
advancements of LAMs across various communication scenarios. Finally, we
analyze the challenges in the current research and provide insights into
potential future research directions.

## Full Text


<!-- PDF content starts -->

1
A Comprehensive Survey of Large AI Models for Future
Communications: Foundations, Applications and Challenges
Feibo Jiang, Senior Member, IEEE , Cunhua Pan, Senior Member, IEEE , Li Dong, Kezhi Wang, Senior Member,
IEEE , Merouane Debbah, Fellow, IEEE , Dusit Niyato, Fellow, IEEE , and Zhu Han, Fellow, IEEE
Abstract —The 6G wireless communications aim to establish an
intelligent world of ubiquitous connectivity, providing an unprece-
dented communication experience. Large artificial intelligence
models (LAMs) are characterized by significantly larger scales (e.g.,
billions or trillions of parameters) compared to typical artificial
intelligence (AI) models. LAMs exhibit outstanding cognitive abil-
ities, including strong generalization capabilities for fine-tuning
to downstream tasks, and emergent capabilities to handle tasks
unseen during training. Therefore, LAMs efficiently provide AI
services for diverse communication applications, making them
crucial tools for addressing complex challenges in future wireless
communication systems. This study provides a comprehensive
review of the foundations, applications, and challenges of LAMs
in communication. First, we introduce the current state of AI-
based communication systems, emphasizing the motivation behind
integrating LAMs into communications and summarizing the
key contributions. We then present an overview of the essential
concepts of LAMs in communication. This includes an introduction
to the main architectures of LAMs, such as transformer, diffusion
models, and mamba. We also explore the classification of LAMs,
including large language models (LLMs), large vision models
(LVMs), large multimodal models (LMMs), and world models,
and examine their potential applications in communication. Ad-
ditionally, we cover the training methods and evaluation tech-
niques for LAMs in communication systems. Lastly, we introduce
optimization strategies such as chain of thought (CoT), retrieval
augmented generation (RAG), and agentic systems. Following this,
we discuss the research advancements of LAMs across various
communication scenarios, including physical layer design, resource
allocation and optimization, network design and management,
edge intelligence, semantic communication, agentic systems, and
emerging applications. Finally, we analyze the challenges in the
current research and provide insights into potential future research
directions.
Index Terms —Large Language Model; Large Vision Model;
Large Multimodal Model; Communication; 6G; Wireless Com-
munication.
Feibo Jiang (jiangfb@hunnu.edu.cn) is with the Hunan Provincial Key
Laboratory of Intelligent Computing and Language Information Processing,
Hunan Normal University, Changsha, China.
Cunhua Pan (cpan@seu.edu.cn) is with the National Mobile Communications
Research Laboratory, Southeast University, Nanjing, China.
Li Dong (Dlj2017@hunnu.edu.cn) is with Xiangjiang Laboratory, Hunan
University of Technology and Business, Changsha, China.
Kezhi Wang (Kezhi.Wang@brunel.ac.uk) is with the Department of Computer
Science, Brunel University of London, UK.
Merouane Debbah (merouane.debbah@ku.ac.ae) is with KU 6G Research
Center. Department of Computer and Information Engineering, Khalifa Univer-
sity, Abu Dhabi 127788, UAE.
Dusit Niyato (dniyato@ntu.edu.sg) is with the College of Computing and
Data Science, at Nanyang Technological University, Singapore.
Zhu Han (hanzhu22@gmail.com) is with the Department of Electrical and
Computer Engineering, University of Houston, Houston, TX, USA.
GitHub link: https://github.com/jiangfeibo/ComLAM.I. I NTRODUCTION
With the continuous emergence of various new technologies,
the complexity and diversity of communication systems are
steadily increasing, leading to a growing demand for efficiency,
stability, and intelligence in these systems. Ubiquitous intelli-
gence is one of the key visions for 6G, which aims to provide
real-time artificial intelligence (AI) services for both the network
and its users, enabling on-demand AI functionality anytime and
anywhere. To achieve this, the 6G network architecture must
consider the deep integration of information, communication
and data technologies, building a comprehensive resource man-
agement framework that spans the entire life cycle of computing,
data, AI models, and communication [1]. Currently, AI tech-
nology has advanced from the era of deep learning to large
AI models (LAMs), such as large language models (LLMs),
large vision models (LVMs), large multimodal models (LMMs)
and world models. The development history of LAMs is Shown
in Fig. 1. These LAMs possess powerful cognitive abilities,
enabling efficient AI services for differentiated communication
application scenarios, becoming potent tools for addressing
complex challenges in future wireless communication systems.
Against this backdrop, the application of LAMs in communica-
tions has become a research hotspot. This paper aims to provide
a comprehensive review of the fundamentals, applications, and
challenges related to LAMs for communications.
Fig. 1: The development history of LAMs.
A. Background
6G aims to create an intelligent and interconnected world,
offering an unprecedented communication experience to human
society. In international mobile telecommunications for 2030
(IMT 2030), proposed by ITU-R, six typical scenarios are
defined: immersive communication, ultra-massive connectivity,
ultra-reliable and low-latency communication, integrated sens-
ing and communication, ubiquitous connectivity across ter-
restrial, aerial, and satellite networks, and integrated AI and
communication [2]. Communication, sensing, computing, AI,
security, and other multidimensional elements will be integratedarXiv:2505.03556v1  [cs.IT]  6 May 2025

2
into 6G to provide users with more advanced communication
services [3].
To achieve the aforementioned vision, 6G relies on a range
of novel communication technologies, including intelligent re-
flecting surfaces [4], integrated terrestrial and non-terrestrial
networks [5], terahertz communications [6], integrated sensing
and communication [7], digital twins [8], the metaverse [9],
and quantum communication technologies [10]. However, the
development of these new technologies has posed challenges
for communication systems, such as performance approaching
theoretical limits and difficulties in adapting to large-scale,
complex scenario changes [11]. The integration of AI with
communication will be an effective way to address these issues.
Currently, classical methods such as traditional machine learn-
ing, deep supervised learning, and deep reinforcement learning
(DRL) have already been widely applied in 5G as effective
tools for optimizing traditional algorithms and operations, being
extensively used in core networks, transport networks, radio
access networks, and edge networks. Below, we first review the
development history of integrated AI and communication.
1) Deep learning-assisted communication: The rapid devel-
opment of deep learning has provided a solid foundation for
tackling critical challenges in wireless communications [12]–
[15]. By applying deep learning techniques, communication
systems have reached a new level in terms of performance and
efficiency. These advancements not only improve operational
capabilities but also pave the way for future innovations in
communication technologies. However, in dynamic and uncer-
tain environments, the generalization ability of deep learning is
limited, and communication systems still face challenges related
to adaptive optimization and learning [12].
2) Reinforcement learning-assisted communication: Rein-
forcement learning has been effectively utilized to enable com-
munication network entities to derive optimal policies, including
decisions or actions, in given states [16]–[19]. Therefore, rein-
forcement learning-based communication technologies demon-
strate tremendous potential in addressing critical issues such as
policy optimization, efficiency enhancement, and performance
improvement in dynamic environments, thus laying a solid
foundation for the continuous optimization and adaptive learning
of communication systems [16].
3) Generative AI-assisted communication: With the continu-
ous advancement of AI technologies, particularly represented
by transformer models, human society is rapidly entering a
new era of generative AI (GAI). The development of GAI has
also brought new opportunities to communications [20]. These
generative models, including generative adversarial networks
(GANs), transformers, and diffusion models, can more accu-
rately learn the intrinsic distribution of information and possess
stronger generation and decision-making capabilities, thereby
significantly enhancing the performance and efficiency of com-
munication systems [21]–[24]. However, as communication
systems become increasingly complex and the communication
environment undergoes dynamic changes, GAI may encounter
challenges such as mode collapse and catastrophic forgetting in
high-dimensional and complex data generation tasks [25].B. Motivation
1) Definition: LAMs represent cutting-edge advancements
in AI, characterized by state-of-the-art generative architectures
and parameter scales reaching hundreds of billions or even
trillions. These models exhibit cognitive capabilities comparable
to humans, enabling them to handle increasingly complex and
diverse data generation tasks. Based on the modality of data
they process, LAMs encompass LLMs, LVMs, LMMs, and
world models [26]. In recent years, several prominent LAMs
such as GPT [27], Sora [28], large language model Meta AI
(LLaMA) [29], and Gemini [30] have transformed workflows
across various domains, including natural language processing
(NLP) and computer vision. The role of LAMs in AI is shown
in Fig. 2.
2) Distinction between LAMs and GAIs: Compared to other
GAI models, LAMs offer significant advantages in scale and
capability. While GAI models also focus on producing new data,
LAMs are typically much larger, with parameter counts reaching
hundreds of billions or even trillions, and exhibit superior
generalization abilities. They demonstrate greater adaptability
and flexibility across a broader range of tasks. Moreover,
unlike smaller GAI models, LAMs are capable of exhibiting
emerging behaviors such as in-context learning [31], chain of
thought (CoT) [32], reflection [33], and emergence [34]. These
capabilities enable them to rapidly adapt to various downstream
applications without the need for task-specific retraining.
3) Distinction between LAMs and pretrained FMs: Pre-
trained foundation models (FMs) have undergone extensive pre-
training but have not been adapted to specific tasks. As a result,
they are prone to generating hallucinations and typically require
further fine-tuning to evolve into task-effective LAMs [35]. For
instance, language-based FMs often need additional processes
such as instruction fine-tuning and reinforcement learning from
human feedback (RLHF) to develop into fully functional LLMs
[36]. Therefore, compared to generic pre-trained FMs, LAMs
can be further optimized on domain-specific datasets—such as
those in communication—effectively mitigating hallucination
issues inherent to raw pre-trained FMs and enabling more
efficient handling of diverse communication tasks.
Fig. 2: The role of LAMs in AI.

3
Consequently, the integration of LAMs with communications
offers the following distinct advantages [37]:
•Exceptional global insight and decision-making: Future
communication systems will operate in dynamic environ-
ments, influenced by device mobility and traffic fluctua-
tions. Traditional AI methods, reliant on localized features,
are prone to local optima and struggle to learn long-
term spatiotemporal characteristics. LAMs, with advanced
architectures and hundreds of billions of parameters, cap-
ture network features from a global perspective, adapt to
multi-scale spatiotemporal dependencies, and generate sta-
ble decision-making responses, whereas traditional neural
networks require retraining. For example, LAMs learn user
mobility and traffic fluctuations from global perspectives,
mitigating long-term forgetting and enabling precise traffic
prediction and resource allocation [38].
•Strong robustness and generalization: Future commu-
nication systems are expected to support a diverse range of
devices, such as internet of things (IoT) devices and UA Vs,
while offering management strategies including beamform-
ing design, user association, and edge resource allocation.
Traditional AI approaches, which focus on learning task-
specific features, are constrained in their adaptability and
robustness across multiple tasks. LAMs, trained on a vari-
ety of data and tasks, exhibit enhanced generalization capa-
bilities for multitask scenarios, enabling effective decision-
making in novel use cases. The extensive data allows
LAMs to capture complex patterns and subtle distinctions
in heterogeneous devices and imbalanced datasets. For
example, by learning channel state information (CSI) and
network topology, LAMs can design universal offloading
models in mobile edge computing systems, optimizing
task offloading and resource scheduling through prompts
without requiring retraining [39].
•Advanced understanding and emergent abilities: Future
communication systems are required to deliver tailored
solutions for diverse application scenarios. For instance,
autonomous driving demands ultra-low latency and high
reliability, while IoT necessitates support for massive
connectivity. Traditional AI approaches, reliant on small-
scale models trained for specific contexts, exhibit limited
applicability. Leveraging their superior contextual learning
capabilities, LAMs can proactively analyze user demands
and preferences in 6G networks, comprehending various
scenarios with minimal or even zero-shot samples, thereby
providing personalized services. Their emergent capabil-
ities enable LAMs to perform advanced cognitive tasks
such as logical reasoning and causal inference, dynami-
cally planning, configuring, and optimizing communication
networks [40].
C. Related survey work
Table I compares this study with the existing related survey
researches. Existing surveys typically focused only on the basic
principles of LAMs and some key technologies, with limited
analysis of the structures and characteristics of different types ofLAMs. Moreover, the coverage of the latest applications is often
insufficient, particularly in the review of other LAMs beyond
LLMs in communication. Although these studies have made
valuable contributions to exploring the application of LLMs
and GAIs for communications, there is still a need for further
improvement. The limitations of existing survey studies can be
summarized as follows:
1) Limited model coverage: Most existing surveys primarily
focus on LLMs (e.g., GPT and LLaMA), while paying insuffi-
cient attention to other categories of LAMs such as LVMs (e.g.,
SAM and DINO), LMMs (e.g., Composable Diffusion (CoDi)
and ImageBind), and world models (e.g., Sora and JEPA). These
studies often lack a unified framework for understanding the di-
verse architectures, training paradigms, and alignment strategies
across different types of LAMs, resulting in an incomplete view
of the model landscape in communication.
2) Incomplete application landscape: While prior surveys
have provided valuable insights into specific applications of
LLMs in communication, their coverage of broader application
scenarios remains limited. In particular, the roles and poten-
tial of other types of LAMs across diverse communication
tasks (e.g., physical layer design, resource allocation, network
management, edge intelligence, semantic communication, and
agentic systems) have not been fully explored. Furthermore, sys-
tematic comparisons of different models’ suitability, technical
characteristics, and collaboration strategies in these scenarios
are largely absent, which may hinder the development of a
comprehensive understanding of LAMs in communication.
D. Contributions
Through a comprehensive summary and systematic analysis
of the existing literature, this work provides readers with a
complete knowledge framework of LAMs for communications,
covering the fundamental review, application review, and chal-
lenges and future directions. Fig. 3 presents the organization
of this paper. Specifically, the contributions of this paper are
summarized as follows:
1) Foundations of LAMs for communications: First, we in-
troduce the key architectures of LAMs, including the trans-
former model, diffusion model, and mamba model. Next, we
classify LAMs in detail, covering categories such as LLMs,
LVMs, LMMs, and world models. Then, we deeply explore
the pre-training, fine-tuning and alignment methods of LAMs
for communications. Next, the evaluation methods are intro-
duced, including communication question-and-answer (Q&A)
evaluation, communication tool learning evaluation, communi-
cation modeling evaluation, and communication code design
evaluation. Finally, we introduce the optimization techniques of
LAMs, including CoT, retrieval augmented generation (RAG),
and agentic systems. These technologies can further improve
the performance of the LAMs and make it effectively applied
in communication. Please refer to Section II for details.
2) Applications of LAMs for communications: We provide a
detailed overview of the research progress of LAMs in various
application scenarios, including physical layer design, resource
allocation and optimization, network design and management,

4
TABLE I: Comparison of our work with existing studies
Ref. Year TypeModel
training
(C1)Evaluation
metric
(C2)Model
architecture
(C3)Model
classification
(C4)Model
optimization
(C5)Application
scenarios
(C6)Research
challenges
(C7)Future
directions
(C8)Remarks
[41]2024 Magazine ✓ ✓ ✓ ✓-For C2, C3, C4 and C5, the descriptions are
simplistic and lack comprehensiveness.
[42]2024 Magazine ✓ ✓ ✓ ✓-For C2, evaluation relies on manual scoring and
lacks task-specific metrics.
-For C3 and C4, the focus on LLMs omits broader
architectural and classification discussions.
-For C6, application scenarios are insufficiently
covered.
[43]2024 Magazine ✓ ✓ ✓ ✓-For C1 and C2, the work lacks coverage of
emerging training methods and clear evaluation
criteria.
-For C3 and C4, architectural and classification
content remains superficial.
[44]2024 Survey ✓ ✓ ✓ ✓ ✓ ✓-For C1, only standard pre-training and fine-tuning
are considered.
-For C5, optimization is mentioned briefly without
methodological depth.
[45]2025 Survey ✓ ✓ ✓ ✓ ✓ ✓-For C3, the focus on GAI excludes newer
architectures such as Mamba and SSM.
-For C6, application coverage is narrow and limited
to semantic scenarios.
[46]2024 Survey ✓ ✓ ✓ ✓ ✓ ✓-For C3, traditional generative models are
emphasized while recent architectures (e.g.,
Diffusion) are omitted.
-For C5, optimization challenges are mentioned
without proposing concrete solutions.
[47]2024 Survey ✓ ✓ ✓ ✓ ✓ ✓-For C2, evaluation frameworks are lacking.
-For C5, the discussion on model optimization
remains relatively high-level, with limited focus on
recent trends.
[48]2025 Survey ✓ ✓ ✓ ✓ ✓-For C2 and C3, evaluation methodology and
architectural coverage are insufficient.
-For C5, no optimization strategies are proposed.
[49]2024 Survey ✓ ✓ ✓ ✓ ✓-For C3, architectural categorization is unclear.
-For C4 and C5, classification and optimization
discussions are limited in depth and integration.
Our Work Survey ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓-For C1 to C8, it provides a comprehensive and
recent review. It includes in-depth discussion,
analyses, challenges, and future research directions.
edge intelligence, semantic communication, agentic systems,
and emerging applications. We classify and summarize the
research progress in each direction, and we detail the represen-
tative works that integrate LAMs with communications, show-
casing the current research status and application prospects, and
providing researchers with referenceable research directions.
Detailed contents can be found in Sections III-IX.
3) Research challenges of LAMs for communications: We
analyze the major research challenges faced by LAMs in
communication-oriented scenarios. First, the current commu-
nication landscape lacks high-quality labeled data, and issues
related to privacy and cost further constrain data availability,
thereby hindering model training and generalization. Second,
LAMs struggle to incorporate structured domain knowledge in
communications, which limits their performance in tasks such
as channel modeling. Issues such as generative hallucinations,
insufficient reasoning capabilities, and poor interpretability fur-
ther undermine their reliability and transparency in critical tasks.
Moreover, LAMs still face limitations in adaptability and gen-
eralization when dealing with dynamic network environments
and diverse communication tasks. Deployment in resource-
constrained scenarios remains challenging due to high inference
latency, as well as concerns regarding privacy and security risks.
Detailed contents can be found in Section X.II. F OUNDATIONS OF LAM S FOR COMMUNICATIONS
Compared to traditional AI and machine learning models,
LAMs are built with hundreds of billions or even trillions
of parameters and advanced architectures. Through large-scale
pretraining, they achieve strong multi-task generalization. LAMs
exhibit superior cognitive abilities and multimodal reasoning
capabilities, enabling rapid adaptation to downstream tasks via
in-context learning and fine-tuning. Moreover, their emergent
abilities allow them to understand and solve tasks that are not
explicitly seen during training. This generality and adaptability
give LAMs a distinct advantage in the development of future
intelligent communication systems. Then, we present the foun-
dations of LAMs for communications, including key architec-
tures, model classifications, model training and evaluation, and
optimization methods.
A. Key architecture of LAMs
LAMs demonstrate exceptional capabilities in handling com-
plex data and tasks through continuous optimization and innova-
tion. The key architectures play a crucial role in the successful
application of LAMs, significantly enhancing the performance
and efficiency of the LAMs while also fostering ongoing ad-
vancements in related technologies. This section introduces the
key architectures of LAMs and their research advancements

5
Fig. 3: Overall organization of the survey.

6
TABLE II: List of common abbreviations
Abbr. Description Abbr. Description
AGI Artificial General Intelligence BitFit Bias-Only Fine-Tuning
CoDi Composable Diffusion CoT Chain of Thought
Diffusion Diffusion Model DPO Direct Preference Optimization
FM Foundation Model GAN Generative Adversarial Network
GPT Generative Pretrained Transformer ICL In-context Learning
KD Knowledge Distillation LLM Large Language Model
LLaMA Large Language Model Meta AI LMM Large Multimodal Model
LLaV A Large Language and Vision Assistant LoRA Low-Rank Adaptation
LVM Large Vision Model MAS Multi-Agent Systems
MoE Mixture of Experts MPFT Mixed-Precision Fine-Tuning
PEFT Parameter-Efficient Fine-Tuning PPO Proximal Policy Optimization
PTQ Post-Training Quantization RAG Retrieval-Augmented Generation
RLHF Reinforcement Learning from Human Feedback RLAIF Reinforcement Learning from AI Feedback
SFT Supervised Fine-Tuning SLM Small Language Model
SSM State Space Model VLM Vision-Language Model
WM World Model WS World Simulator
in communication, including the transformer model, diffusion
model, and mamba model.
1) Transformer model: Transformer is a novel neural net-
work architecture introduced by Vaswani et al. in 2017 [50].
The primary characteristic of the transformer architecture lies
in its complete reliance on the attention mechanism, which
eliminates the sequential dependencies inherent in traditional
sequence data, allowing the model to process input sequences
in parallel. This architecture excels in addressing long-range
dependency issues, particularly in NLP tasks. Compared to
traditional recurrent RNNs, the transformer exhibits superior
parallelism and computational efficiency, making it well-suited
for handling large-scale datasets and complex sequence tasks.
The workflow of the transformer architecture is as follows:
•Input embedding and positional encoding: The trans-
former converts each word in the input sequence into high-
dimensional vectors through an embedding layer, which
represents their semantic information. Positional encodings
are added to these word vectors to enable the model
to recognize the order of the sequence and perceive the
sequential relationships.
•Multi-head self-attention in the encoder: The processed
word vectors enter the multi-head self-attention layer of
the encoder. The self-attention layer computes attention
weights through queries, keys, and values, determining the
relevance of each word to others and capturing global
dependency information in the sequence. The output is then
further processed through a feedforward neural network,
with residual connections and layer normalization applied
to enhance the stability and efficiency of model training.
•Generation and output in the decoder: The hidden state
vectors generated by the encoder are passed to the decoder,
which first processes the previously generated portions ofthe output sequence to capture the internal dependencies
of the current sequence. Subsequently, the cross-attention
mechanism integrates the current state of the decoder with
the hidden states from the encoder, generating new outputs
based on the input sequence. Finally, after processing
through the output layer, the decoder produces the final
output sequence.
Representative transformer-based research in communica-
tion: The transformer has been widely adopted in various
cutting-edge LAMs, such as OpenAI’s GPT series, Google’s
BERT and T5 models, and Facebook’s RoBERTa [51]. In
recent years, transformer models have been increasingly utilized
in communication due to their exceptional global modeling
capabilities and efficient parallel computing performance. For
instance, Wang et al. [52] explored the application of the trans-
former architecture in handling large-scale MIMO and semantic
communication in 6G networks, highlighting the critical role of
deep learning, particularly the transformer, in network optimiza-
tion and the resolution of intricate communication challenges.
Yoo et al. [53] proposed a real-time semantic communication
system based on a vision transformer (ViT), which significantly
outperformed traditional 256-QAM systems in low signal-to-
noise ratio environments, demonstrating the advantages of se-
mantic communication in effectively conveying information.
Furthermore, Wu et al. [54] introduced the DeepJSCC-ViT-f
framework, which leverages the ViT architecture in conjunction
with channel feedback information to enhance the performance
of wireless image transmission. This framework aims to address
the complexities and adaptability issues of the existing joint
source-channel coding (JSCC) methods.
2) Diffusion model: The Diffusion model is a generative
model based on the probabilistic diffusion process, proposed
by Sohl-Dickstein et al. in 2015 [55]. The main feature of

7
the diffusion model is to generate data by gradually adding
noise to the data and then learning the inverse denoising
process. It is good at generating high-quality, detailed images,
especially when dealing with complex image generation and
signal recovery problems. The workflow of the diffusion model
is as follows:
•Forward diffusion process: The forward diffusion process
maps the data to a state close to the standard normal dis-
tribution by gradually adding gaussian noise. This process
gradually destroys the data, making the data more and more
blurred, and finally forming a high-noise state. The noise
addition at each step is gradual, gradually covering up the
original structural information of the data.
•Reverse diffusion process: In the reverse diffusion pro-
cess, the model restores the data from the high-noise state
to the state of the original data by gradually removing the
noise. This process is usually approximated by training a
neural network to the probability distribution of the reverse
diffusion process. The network learns how to gradually
recover the data from the noise to generate new samples
similar to the original data. This step is the key to the
generation process, enabling the model to effectively “re-
construct” the data from the noise.
Representative diffusion model-based research in communica-
tion: In recent years, diffusion models have drawn widespread
attention due to their exceptional performance and flexibility.
These models have demonstrated particularly outstanding per-
formance in image generation and show considerable applica-
tion potential in communication. For instance, Jiang et al. [56]
proposed a channel estimation method called generative adver-
sarial network and diffusion model aided-channel estimation
(GDCE), combining a diffusion model and a CGAN. CGAN
is first used to generate CSI. Then a diffusion model refines
the CSI information. Gradually removing noise through the
diffusion model generates a more refined CSI image, improving
signal recovery accuracy. Du et al. [57] studied the applications
of generative diffusion models (GDMs) in network optimiza-
tion, especially in complex intelligent network scenarios like
DRL, incentive mechanism design, semantic communication,
and vehicular networks, demonstrating GDMs’ potential for
modeling complex data distributions and generating high-quality
decisions. Wu et al. [58] proposed channel denoising diffusion
models (CDDM) for image transmission tasks in wireless se-
mantic communication systems. This model enhances image
transmission quality and communication system performance by
leveraging diffusion models’ noise-removal advantages through
learning channel input signals’ distribution. Additionally, Duan
et al. [59] introduced a diffusion model for multiple-input
multiple-output (DM-MIMO) module for robust MIMO seman-
tic communication. It combines diffusion models with SVD
precoding and equalization to reduce channel noise, lower MSE,
and enhance image reconstruction quality, showing superior
performance in MIMO systems. Chi et al. [60] proposed RF-
Diffusion, a radio signal generation model based on time-
frequency diffusion theory. Through an enhanced diffusion pro-
cess, this model is capable of generating high-quality sequentialRF signals in both the time and frequency domains. Grassucci
et al. [61] proposed a generative audio semantic communication
framework using a conditional diffusion model to address band-
width consumption and error recovery challenges in traditional
audio signal transmission.
3) Mamba model: Mamba is a generative architecture for
efficiently processing long sequence data, proposed by Gu et
al. in 2022 [62]. The main feature of mamba is to efficiently
process long sequence data. It allows the model to focus on
relevant information and filter out unnecessary parts through a
selection mechanism based on input data. At the same time,
it adopts a hardware-aware computing algorithm, which specif-
ically optimizes the processing performance on the GPU and
significantly speeds up computing. The mamba model is good
at processing high-dimensional, long-series complex data, such
as natural language, video, or time series tasks. By optimizing
data flow processing and resource allocation, it can effectively
reduce communication delays and improve system performance.
The workflow of the mamba architecture is given as follows:
•Input processing and projection: Input data (such as
text, images, time series, etc.) is segmented into multiple
fragments (tokens or patches) and converted to vector
representations through a linear projection layer. This step
is similar to the preprocessing process of other deep
learning models and is used to map the input into a high-
dimensional space.
•Selection mechanism: The state space is a collection
of variables that describe the dynamic behavior of the
model. Mamba uses an efficient selection mechanism to
dynamically adjust the state space parameters based on
the input data. This mechanism allows the model to filter
out irrelevant information and retain only key feature
information, thereby achieving content-aware modeling.
This process is usually implemented using a convolutional
layer
•SSM calculation: State space model (SSM) calculation is
the process of modeling input data and generating output
using the SSM. The discretized SSM equation is used to
calculate the input data, which includes the state equation
and observation equation. The state equation describes the
change of state variables over time, and the observation
equation describes how the observation variable is gener-
ated from the state variable. Mamba architecture uses these
equations to learn complex patterns in sequence data and
generate high-quality output.
•Output generation: After the SSM completes processing
the input, mamba passes the output to a fully connected
layer or other task-related layers (such as a classifier or
generator) to generate the final output.
Representative mamba-based research in communication:
Mamba architecture has made significant breakthroughs in long
sequence modeling and multi-task parallel computing. Its ef-
ficient processing capabilities and dynamic adjustment mech-
anism have received widespread attention. For instance, Wu
et al [63] proposed the MambaJSCC architecture for wireless
transmission of images. The architecture implements adaptive

8
coding based on the generalized state space model (GSSM) and
CSI residual transfer (CSI-ReST). MambaJSCC uses the VSSM-
CA module and combines the reversible matrix transformation
of GSSM to effectively capture global information and achieve
high performance and low latency in image transmission with-
out increasing calculation and parameter overhead, surpassing
existing methods. Yuan et al [64] proposed ST-Mamba, which
addresses the accuracy and stability issues of traffic flow esti-
mation, and performs well, especially when data is limited. ST-
Mamba combines the CNN and the mamba framework, which
can effectively capture the spatial and temporal characteristics
of traffic flow. Yu et al [65] proposed the underwater acous-
tic image semantic communication model VimSC based on
vision mamba. By combining OFDM technology, the quality
of image transmission in complex underwater environments is
significantly improved. VimSC uses a CSI feedback precoding
network to adjust semantic symbols and uses channel estima-
tion and equalizers to achieve accurate image reconstruction.
Experimental results show that its performance is better than
traditional methods in low signal-to-noise ratio environments.
B. Classification of LAMs
As shown in Table III, we classify LAMs into the following
categories based on the type of data processed [66]. Although
other classification methods have been proposed in previous
studies, our data-type-based classification provides a more fo-
cused and practical framework, which is better suited to ad-
dressing the diverse challenges in communication systems, such
as handling different modalities, optimizing resource allocation,
and improving system efficiency across various communication
tasks.
1) Large language model: LLM is an NLP model with a
large number of parameters and complex architecture. It learns
the structure and semantics of language by pre-training on a
large amount of text data. These models can generate natural
and fluent text and perform a variety of language tasks such as
translation and Q&A. LLMs are usually based on deep learning
architectures, such as transformers, which can effectively cap-
ture long-range dependencies. They adjust internal parameters
and improve performance by optimizing complex loss functions.
The LLM includes the following technical features:
•Language understanding and generation: LLMs show
powerful language understanding and generation capabil-
ities when processing text data. By pre-training on large-
scale texts, they learn rich language patterns and knowl-
edge, and can understand complex language structures and
contexts. LLMs not only identify and interpret the meaning
of words, phrases, and sentences but also capture nuances
in language, such as tone and emotion. When generating
text, they create coherent and creative content, maintain
grammatical and semantic accuracy, and are capable of
multilingual translation, demonstrating the potential for
cross-language understanding [67].
•Memory and reasoning ability: LLMs are widely used
because of their excellent memory and reasoning capabil-
ities. Through deep learning of massive text data, theycan memorize and understand rich language knowledge
and factual information, and can maintain consistency
and coherence in different contexts. The model not only
masters vocabulary and grammar, but also understands
complex context and long-distance dependencies. In terms
of reasoning, LLMs can perform logical reasoning based
on text, infer implicit meanings, causal relationships and
conclusions, handle multi-step reasoning tasks, and simu-
late the human thinking process to a certain extent. They
use information in memory to reason and predict new
situations, generating coherent and logical text, making
them excellent at tasks such as summary generation, Q&A
and text analysis.
Classic LLMs include the GPT series, the Gemma series, and
the LLaMA series, among others. These models possess a vast
number of parameters, enabling them to process and generate
natural language text effectively while demonstrating excellent
performance across various NLP tasks. Below, we provide a
detailed overview of three classic LLMs.
•GPT series: The GPT series is developed by OpenAI,
standing for “generative pretrained transformer”. These
models learn language patterns and generate natural lan-
guage through pretraining on a vast amount of text data.
Since the introduction of GPT-1, the GPT models have
evolved through several versions, including GPT-1 [68],
GPT-2 [69], GPT-3 [70], GPT-4 [27], and OpenAI o1.
The original GPT-1, released in 2018, focused on text
generation, and learning language patterns from extensive
text data using unsupervised learning. In 2019, GPT-2
was released, expanding the parameter count from 100
million to 1.5 billion, resulting in more coherent text
generation and the ability to handle more complex tasks.
Subsequently, GPT-3 was launched in 2020, with a further
parameter increase to 175 billion, showcasing powerful
few-shot learning capabilities, enabling it to perform var-
ious tasks such as translation, question answering, and
code writing without fine-tuning. The release of GPT-4
in 2023 introduced multimodal capabilities, allowing it to
understand images in addition to text, along with significant
improvements in reasoning ability, logic, and coherence,
making it adept at tackling complex reasoning problems.
In 2024, OpenAI released model o1, which demonstrates
exceptional capabilities in reflective reasoning compared to
previous generations of LAMs. It is able to generate more
precise and logically consistent responses by conducting
multi-level analyses of complex problems. This allows
o1 to effectively perform self-correction and reflection
in ambiguous or uncertain situations, thereby enhancing
its reliability and intelligence in practical applications.
The advancements in the GPT series have opened new
possibilities for the development of NLP and AI.
•Gemma series: The Gemma series is developed by
Google, which includes Gemma 1 [71] and Gemma 2
[72]. Gemma 1 was released in 2024 and is available in
two versions with different scales: 2 billion parameters
and 7 billion parameters, catering to various computational

9
environments and application needs. The model architec-
ture is based on the transformer decoder and incorporates
several technical improvements, such as multi-head atten-
tion mechanisms, rotary position embedding (RoPE), and
GeGLU activation functions, enabling the model to exhibit
strong contextual understanding and excel in diverse text
generation tasks. Gemma 2 was launched in 2024, of-
fering versions with 9 billion and 27 billion parameters.
This model features an enhanced transformer architecture,
which includes the interleaved use of local and global atten-
tion, as well as group query attention techniques, thereby
improving the model’s language processing capabilities.
Compared to Gemma 1, Gemma 2 demonstrates significant
advancements in both parameter scale and performance.
•LLaMA series: The LLaMA series is a foundational
language model developed by the Meta AI team, which
includes the LLaMA-1 [29], LLaMA-2 [73], and LLaMA-
3 [74]. Both LLaMA-1 and LLaMA-2 were released in
2023. The LLaMA-1 series features several models of
varying sizes, ranging from 700 million to 6.5 billion pa-
rameters. Compared to previous language models, LLaMA-
1 enhances the accuracy and fluency of text generation
while maintaining low computational costs through opti-
mization of the model structure and the introduction of
improved algorithms during training. LLaMA-2 builds on
this foundation with significant improvements, expanding
the model’s scale to offer versions with 700 million, 1.3
billion, 3 billion, and 7 billion parameters. It enhances
the model’s contextual understanding by optimizing the
self-attention mechanism and algorithms. LLaMA-3 was
released in 2024, further extending the parameter count to
offer versions with 8 billion and 70 billion parameters, as
well as incorporating additional pre-training data, demon-
strating superior performance in benchmark tests.
LLM-based research in communication: LLMs, due to their
powerful data processing capabilities, have been widely applied
in communication, enhancing the efficiency of communication
systems and significantly promoting their rapid development
[75]. The integration of LLMs with communication systems is
also a crucial research direction for the future. For example,
Jiang et al. [76] proposed a LAM-driven multimodal semantic
communication (LAM-MSC) framework. It can achieve mul-
timodal to unimodal data conversion, personalized semantic
extraction and wireless CSI estimation, thus addressing data
discreteness and semantic ambiguity issues in multimodal se-
mantic communication by LLMs. In addition,Jiang et al. [77]
proposed a semantic communication architecture based on FMs.
They introduced GPT, leveraging LLMs to enhance semantic
understanding and data reconstruction efficiency, and employed
fine - tuning techniques to address high computational com-
plexity from numerous model parameters. Jiang et al. [40]
proposed a multi-agent system to enhance LLMs’ capabilities
in 6G, improving network optimization and management via
natural language input. It integrates data retrieval, collaborative
planning, and evaluation to address limited private communica-
tion data and constrained reasoning, thereby extending LLMs’knowledge and functionality in this context. Wang et al. [78]
presented a general end-to-end learning semantic communica-
tion model integrating LLMs to boost next-generation com-
munication systems’ performance. It addresses challenges like
semantic fidelity, cross-scenario generalization and complexity
via subword-level tokenization, rate adapters for channel codec
matching and fine-tuning for private background knowledge. Xu
et al. [79] proposed a split learning system for LLMs agents in
6G networks to enhance human machine interaction and pro-
vide personalized cross-domain assistant services. The system
offloads complex tasks to edge servers to address mobile device
capacity limits. Its architecture with perception, grounding, and
alignment modules enables inter module communication to meet
diverse 6G user requirements.
2) Large vision model: LVM is a foundation model that
processes and understands visual data. It usually adopts CNN
and transformer architecture. The LVM learns rich visual fea-
tures from a large number of images and can demonstrate high
accuracy and strong generalization capabilities in tasks such
as image classification, object detection, image segmentation,
and generation. With continuous development and optimization,
LVMs play an important role in promoting the advancement of
image processing technology [81]. The LVM has many technical
features, such as feature representation learning and support for
multiple visual tasks as follows:
•Feature representation learning: In LVMs, feature rep-
resentation learning is one of the core technologies that
automatically extracts and learns important features in
images through deep neural networks. This process mainly
relies on CNN and ViT to complete. The CNN first
extracts local features through multi-layer convolution and
nonlinear activation functions, and then integrates these
local features into global features through fully connected
layers or pooling operations. The transformer architecture
further enhances the capability of feature representation
through a self-attention mechanism, capturing long-range
dependencies and complex contextual information in im-
ages. Through large-scale pre-training and careful fine-
tuning, the LVM can optimize feature representation, and
significantly improve the performance of visual tasks.
•Support multiple visual tasks: Supporting multiple visual
tasks is one of the important features of LVMs. Through
deep learning technology, it can support multiple visual
tasks and play a role in a wide range of application
scenarios. These tasks include image recognition, object
detection, scene parsing, image segmentation, image gen-
eration, image editing, and video analysis. They can recog-
nize and understand objects and scenes in images, locate
the position and size of objects, analyze the relationship
between objects, segment image areas, create or modify
image content, and process actions and events in videos.
In addition, they also support 3D reconstruction to enhance
the visual experience of virtual environments.
Classic LVMs include the SAM series, DINO series, Stable
Diffusion series, etc. In the following, we provide a detailed
introduction to these LVMs.

10
TABLE III: Classification of LAMs and their applications in communication
LAM CategoryData Type
ProcessedSpecific Models Application Domains
Large Language Model Text dataGPT series, Gemma series,
LLaMA seriesSemantic communication [76], Network
management [40], Edge intelligence [79],
Security and privacy [40]
Large Vision Model Image dataSAM series, DINO series,
Stable diffusion seriesSemantic communication [37], Image
segmentation [80], Image generation [81]
Large Multimodal
ModelMultimodal
dataCoDi series, Meta-transformer,
ImageBindSemantic communication [76], Cross-Modal
Generation [82]
World Model / World
SimulatorSimulated
real-world dataSora, Vista, JEPA Autonomous driving [83], Digital twin [28]
•SAM series: SAM is an LVM developed by Meta AI,
designed to perform image segmentation efficiently. The
SAM series includes SAM-1 [84] and SAM-2 [80]. SAM-1
was released in 2023. Its core technology is a deep learning
architecture based on the self-attention mechanism, which
can recognize any object in the image and refine its
boundaries with high resolution. The model is designed
with a wide range of application scenarios, and can not only
handle conventional target segmentation tasks, but also
complex multi-target segmentation and detail processing.
SAM-2 was released in 2024 and has been improved
in many aspects to further improve the performance of
image segmentation. First, SAM-2 optimizes segmentation
accuracy, especially when dealing with complex scenes
and small targets, and can more accurately recognize
and segment multiple types of objects. Secondly, SAM-
2 has been upgraded in model architecture, introducing
more advanced deep learning algorithms and optimized
self-attention mechanisms, enabling it to more effectively
capture details and long-range dependencies in images. In
addition, the inference speed has also been improved, and
the processing efficiency is higher, especially in scenarios
that require real-time response.
•DINO series: DINO Series is an unsupervised visual
feature learning model jointly developed by Meta AI
Research and Inria. It is designed to generate universal
visual features through large-scale curated data sets without
the need for fine-tuning. This series of models include
DINO V1 [85] and DINO V2 [86]. DINO V1 was released
in 2021. It uses the transformer architecture and adopts
a contrastive learning method. By inputting images from
different perspectives for processing, DINO V1 can learn to
recognize and distinguish different elements and structures
in the image. This approach allows DINO V1 to be pre-
trained on unlabeled image data and generate powerful
image representations suitable for various vision tasks, such
as image classification, object detection, etc. DINO V2
was released in 2023. Compared with DINO V1, DINO
V2 made significant improvements in many aspects. DINO
V2 adopts a more advanced architecture, expands the
model size, and uses more computing resources, thereby
improving the accuracy of feature representation and theability to handle complex visual tasks. The contrastive
learning strategy and self-supervision mechanism are opti-
mized to improve the robustness and generalization ability
for different image types. During the training process,
DINO V2 introduced improved training techniques and
data enhancement methods to enhance its performance in
complex scenes and small target processing.
•Stable diffusion series: The stable diffusion series is
developed by Stability AI for generating high-quality im-
ages. These models use diffusion model technology and
are widely used in tasks such as image generation, image
restoration, and image transformation. This series include
Stable Diffusion V1 [87], Stable Diffusion V2 [87], and
Stable Diffusion V3 [88]. Stable Diffusion V1, released
in 2022, is able to generate delicate and diverse images
through a large amount of training data and diffusion model
technology. This model marks an important breakthrough
in the field of image generation, with the ability to generate
high-resolution images in a variety of scenarios. Then,
Stable Diffusion V2 was released in 2022, bringing even
more significant improvements. This version uses updated
generation technology to support higher resolution images
and perform better at handling complex scenes and details.
Stable Diffusion V3 was released in 2024. Compared with
V2, Stable Diffusion V3 replaces the U-Net backbone of
V2 by introducing a rectified fourier transformer (RFT)
architecture, which significantly improves image and text
encoding processing capabilities. Stable Diffusion V3 uses
three chip encoding tracks (i.e., original text encoding,
converted text encoding, and image encoding) to improve
multimodal interaction with images, enabling the gener-
ation of more refined and contextually accurate images,
especially for complex cues.
LVM-based research in communication: By applying LVMs
to the communication field, communication systems can be
made more efficient in processing visual tasks. Jiang et al.
[37] utilized the SAM to construct a semantic knowledge
base (SKB), thereby proposing the LAM-SC framework, a
semantic communication framework based on LAMs, focused
on the transmission of image data. The SAM performs ac-
curate semantic segmentation on any image without specific
training, breaking the image down into multiple segments, each

11
containing a single semantic object. Additionally, Tariq et al.
[89] proposed a SAM-based semantic communication method to
retain semantic features and ensure high-quality reconstruction
in image transmission. This method leverages the SAM to over-
come the diminishing returns of traditional approaches aimed
at enhancing transmission rates while reducing communication
overhead.
3) Large multimodal model: LMMs are capable of simul-
taneously processing and understanding data from different
modalities, such as vision, language, haptic and auditory. These
models achieve comprehensive processing and reasoning of
multimodal information by integrating the features of various
modalities in a unified high-dimensional space. They utilize
advanced neural network architectures, such as transformers
and diffusion models, to extract features from each modality
and optimize their representations through techniques such as
contrastive learning and self-supervised learning. By training
across multiple modalities, these models are able to understand
and relate the semantic relationships between different modal-
ities, thereby demonstrating superior performance in handling
complex multimodal data and providing intelligent, efficient so-
lutions. Unlike visual language models (VLMs), LMMs support
modalities beyond vision and text [90].
LMMs show strong capabilities in processing multimodal in-
formation. Their core technical features are cross-modal fusion
and multimodal representation learning:
•Multimodal representation learning: Multimodal repre-
sentation learning is an important technology of LMMs.
It integrates feature representations of different modalities
such as images, speech and text into a unified high-
dimensional space. First, the LMM uses ViT to extract
image features and uses a transformer to extract text and
speech features. Then, these high-dimensional vectors are
fused through methods such as splicing and weighted
summation to form a unified feature representation. This
fusion enables the LMM to better understand and asso-
ciate information from different modalities, improving the
performance of multimodal tasks.
•Cross-modal fusion: LMMs integrate multiple data types
such as text, images, audio and video through cross-modal
fusion technology to achieve deeper understanding and
analysis. These LMMs can process data from different
modalities at the same time and learn the relationship
between them. For example, the LMM can combine images
with related text to generate richer descriptions; in video
analysis, it can understand the visual content as well as the
voice and text information in the video. In addition, these
LMMs can also perform cross-modal reasoning and predic-
tion, such as generating images or audio from text. Such
capabilities make LMMs widely used in NLP, computer
vision, speech recognition and other fields.
LMMs integrate many advanced model architectures and can
process and understand data in different modalities. In the
following, we provide a detailed introduction to three LMMs.
•CoDi series: CoDi series is developed by Microsoft Azure
and the University of North Carolina. It is an innovativemulti-modal generative model. The series include CoDi-
1 [91], CoDi-2 [92]. CoDi-1 was launched by Microsoft
in 2023, aiming to improve the accuracy and flexibility
of image generation. CoDi-1 utilizes conditional diffusion
model technology to achieve precise control of the gen-
erated results by combining specific condition information
(such as text descriptions, labels, or other input data) with
the image generation process. CoDi-2 was released in 2024.
Compared with CoDi-1, CoDi-2 has made significant im-
provements in many aspects, further improving the ability
and effect of image generation. First, CoDi-2 introduces
an enhanced conditional control mechanism, allowing the
generated images to more accurately conform to complex
conditional inputs. This improvement includes a more
flexible condition encoding method and a more refined
condition processing strategy, thus providing higher control
accuracy. Secondly, CoDi-2 optimizes the model architec-
ture by adopting more advanced diffusion technology, and
improves network design, making the generated images
of higher quality and richer in detail. In addition, CoDi-
2 introduces improved data augmentation methods and
optimized training techniques, resulting in enhancements
in the speed and stability of image generation.
•Meta-transformer: Meta-transformer [82] is a multimodal
learning framework designed to process and associate
information from different modalities. It uses a fixed en-
coder to achieve multimodal perception without paired
multimodal data. The framework consists of three main
components: a unified data segmenter that maps data of
various modalities to a shared latent space; a modality-
shared encoder that extracts high-level semantic features;
task-specific heads. Meta-transformer can uniformly pro-
cess 12 modalities, such as natural language, images,
point clouds, audio, video, infrared, hyperspectral, X-ray,
time series, tables, inertial measurement units (IMUs), and
graphic data. Its main advantage is that it converts data of
different modalities into a unified feature sequence, uses a
shared encoder to extract features, reduces the complexity
of cross-modal alignment, and improves the flexibility of
training.
•ImageBind: ImageBind [93] is an advanced LMMs that
aims to integrate data from different modalities through
a shared embedding space. The model can handle data
from six different modalities, such as images, text, audio,
depth, thermal imaging, and IMU data. Its innovation lies
in the cross-modal alignment without explicit pairing of
data. Through contrastive learning, data from different
modalities are projected into a unified representation space,
thereby enhancing the generalization ability and cross-
modal understanding ability of the model. ImageBind
performs well in multimodal retrieval, classification, and
generation tasks, especially when dealing with unaligned
data.
LMM-based research in communication: LMMs are widely
used in communication due to their powerful multimodal in-
formation processing capabilities [94]. For example, Jiang et

12
al. [76] proposed a LAM-MSC framework by combining the
multimodal processing model CoDi with the language com-
munication system. In this communication system framework,
the CoDi model can convert multimodal data into text data
for processing to achieve cross-modal processing of the model.
The LAM-MSC framework shows excellent performance in
simulation experiments, and can effectively process multimodal
data communication and maintain the semantic consistency
between the original data and the restored data. Qiao et al. [95]
proposed a delay-aware semantic communication framework
based on a pre-trained generative model by combining models
such as BLIP, Oscar, and GPT-4. The framework aims to
achieve ultra-low data rate semantic communication in future
wireless networks through multimodal semantic decomposition
and transmission. In this framework, the transmitter performs
multimodal semantic decomposition on the input signal and
selects appropriate encoding and communication schemes to
transmit each semantic stream according to the intention. For
text prompts, a retransmission-based scheme is adopted to en-
sure reliable transmission, while other semantic modalities use
adaptive modulation/coding schemes to adapt to the changing
wireless channels.
4) World model: A world model is an abstract framework
to describe and simulate real-world phenomena, aiming to
create intelligent systems that can understand and simulate the
environment [96]. The world model primarily consists of two
key components: the environment simulator and the controller.
The environment simulator is responsible for constructing a
model that can predict the state and behavior of the environ-
ment, typically achieved through deep neural networks. These
networks are trained to understand the dynamic characteristics
of the environment and generate predictions of future states
and rewards [97]. The controller uses this simulator to make
decisions and improves its performance in the real environment
by training and optimizing in the simulated environment.
World models support LAMs by providing simulated sce-
narios that help LAMs generalize and adapt to complex and
dynamic environments. Unlike digital twins, which are primarily
used to replicate real-world objects or systems in real time,
world models focus on simulating and training LAMs in virtual
environments [98]. We introduce the features of the world model
in details below.
•Long-term planning and cognitive decision-making:
The world model simulates and predicts the dynamic
changes of complex systems and makes effective decisions.
Long-term planning involves learning patterns from histor-
ical data and anticipating future trends to guide resource
allocation and action selection. World models can eval-
uate the long-term impact of different strategies and help
decision-makers understand different choices and formulate
sustainable plans. The world model can also simulate the
decision-making process in different scenarios, provide a
variety of solutions, and support wise choices in complex
environments. This dynamic and predictive ability makes it
valuable in policy formulation, resource management, and
risk assessment.•Continuous perception and embodied intelligence: The
world model has significant advantages in continuous per-
ception and embodied intelligence. It can obtain informa-
tion from the environment in real time, and monitor and
analyze various variables, such as climate, traffic flow, etc.,
to provide the latest data for decision-making. Embodied
intelligence enables models to combine sensory informa-
tion with physical entities to simulate the behavior and
interaction of entities in the environment. This capability
supports more complex tasks such as automatic control,
robot navigation, and environmental monitoring, giving it
broad application prospects in areas such as intelligent
transportation, smart city management, and disaster warn-
ing.
There are many classic world models, which provide many
new ideas for the research in communication. In the following,
we provide a detailed introduction to three world models.
•Sora: Sora is a groundbreaking text-to-video generation
model released by OpenAI [28] that demonstrates signif-
icant emergent capabilities. It is based on a pre-trained
diffusion transformer and is able to generate high-quality
videos based on text instructions, introducing details
through progressive denoising and text cues. Sora excels in
several areas, including simulation capabilities, creativity,
and accessibility. Although not explicitly 3D modeled, Sora
demonstrates 3D consistency, such as dynamic camera
motion and long-range coherence, and is able to simulate
aspects of the physical world and simple interactions.
•JEPA: Joint embedding predictive architecture (JEPA) [99]
is a world model for multi-modal learning that aims to
enhance the understanding of complex data through joint
embedding and prediction tasks. By mapping different
modalities of data into a shared embedding space, JEPA
enables the model to capture the potential relationships
between different data in this space. Specifically, JEPA
performs contrastive learning in the embedding space and
optimizes the embedding distance of similar data to en-
hance the understanding of different modal information. In
addition, in the interaction between JEPA and the envi-
ronment, the world model can provide generated samples
and state changes, and JEPA further adjusts the structure
and characteristics of its embedded space through this
dynamic information, allowing it to reason more effectively
in complex environments. This interactive mechanism not
only improves the understanding of the environment, but
also enhances the adaptability of JEPA, allowing it to
exhibit higher robustness and flexibility in diverse real-
world scenarios.
•Vista: Vista [83] is an advanced world model focused on
solving the limitations of data scale, frame rate, and reso-
lution in the field of autonomous driving. It adopts a novel
loss function to enhance the learning of moving instances
and structural information, and designs a latent replacement
method to achieve coherent long-term predictions through
historical frames. Vista also excels when it integrates a
diverse set of controls from high-level intentions to low-

13
level actions. After large-scale training, Vista outperforms
most existing video generation models in experiments
on multiple datasets. Vista’s training framework includes
two stages: high-fidelity future prediction and multi-modal
action control learning, which can provide high-resolution
predictions in different scenes and camera angles with less
quality degradation.
World model-based research in communication: The appli-
cation of world models in communication has played a revo-
lutionary role in 6G. For example, Saad et al. [100] proposed
a revolutionary vision of the next-generation wireless system
in their research, the AGI-native wireless system, the core of
which is the world model. The AGI-native wireless system is
mainly built on three basic components: the perception module,
the world model, and the action planning component. Together,
these components form four pillars of common sense, including
handling unforeseen scenarios through horizontal generalization,
capturing intuitive physics, performing analogical reasoning,
and filling in the gaps. The study also discussed how AGI-
native networks can be further utilized to support three use
cases related to human users and autonomous agent applica-
tions: analogical reasoning for next-generation digital twins,
synchronous and elastic experience of cognitive avatars, and
brain-level metaverse experience with holographic transmission
as an example. Finally, they put forward a series of suggestions
to inspire the pursuit of AGI-native systems and provide a
roadmap for next-generation wireless systems beyond 6G.
C. Training of LAMs for communications
The training process of LAMs for communications involves
three stages: pre-training, fine-tuning, and alignment. As illus-
trated in Table IV, a comprehensive comparison of these stages
is provided. In the following sections, we present a detailed
discussion of each stage.
1) Pre-training of LAMs for communications: The pre-
training stage forms the foundation for LAMs to acquire special-
ized knowledge in communication. This process is summarized
as follows:
The LAMs is pre-trained on a large unlabeled dataset to
learn universal features, boosting performance on communica-
tion tasks, reducing reliance on labeled data, and improving
knowledge transfer. The key pre-training methods are self-
supervised and multi-task learning:
•Self-supervised learning: Self-supervised learning, unlike
unsupervised learning, enables the LAMs to learn features
from the data itself by generating supervisory signals
through data transformation or masking. The process starts
with data preprocessing, followed by creating proxy tasks
to generate self-supervised signals. The model is then
trained using these internal representations, similar to su-
pervised learning but without external labels [101].
•Multi-task learning: Multi-task learning improves model
performance by learning multiple related tasks simultane-
ously. Tasks share model parameters, enabling the LAMs
to leverage their relationships for better efficiency and gen-
eralization. The process involves defining tasks, designinga shared model architecture with common and task-specific
layers, and ensuring consistent data preprocessing. During
training, shared layers capture common features, while
task-specific layers focus on individual objectives [102].
To improve training efficiency and model performance, re-
searchers have proposed various optimization strategies for the
pre-training stage:
•Distributed training: Distributed training techniques in-
volve multiple devices working together to train LAMs,
requiring effective data and model parallelism strategies for
efficiency and stability. Frameworks like megatron-lm and
deepspeed are designed for distributed training, enabling
efficient data and model parallelism [103].
•Learning rate scheduling: Dynamic learning rate adjust-
ment plays a crucial role in enabling LAMs to identify
optimal parameters during training. Typical approaches
include cosine annealing and cyclic learning rate strategies
[104].
•Gradient clipping: This optimization technique mitigates
gradient explosion and vanishing by scaling or truncat-
ing gradients during backpropagation. Typical approaches
include absolute value clipping and norm-based clipping,
which constrain or reduce excessively large gradients [105].
2) Fine-tuning of LAMs for communications: The fine-tuning
stage optimizes a pre-trained LAM using a specific communica-
tion dataset, helping it better adapt to communication tasks. This
process improves the model’s understanding, generalization,
accuracy, and efficiency in communication applications.
Telecom instruction fine-tuning technique [35] trains LLMs
to generate accurate outputs based on telecom instructions in
natural language. It uses instructions paired with responses to
guide the model in performing tasks, enhancing its understand-
ing and ability to handle new tasks. The instruction dataset
is generated using advanced LLMs like GPT-4 and LLaMA-
3, based on telecom documents, to meet the needs of various
tasks [35]:
•Multiple choice question answering: Selecting all correct
answers from a set of multiple-choice questions.
•Open-ended question answering: Providing open-ended
responses to telecom-related questions based on standards,
research papers, or patents.
•Technical document classification: Classifying the text of
various technical documents into relevant working groups,
such as the different working groups in the 3GPP standards.
•Mathematical modeling: Generating accurate mathemat-
ical equations, such as channel models, based on textual
descriptions of system models and problem statements.
•Code generation: Generating scripts or functions for spe-
cific tasks or functionalities in telecom.
•Fill-in-the-middle: Completing incomplete scripts based
on context and target functionality.
•Code summarization: Summarizing the core functional-
ities of a given script, including identifying whether the
script is related to telecom.
•Code analysis: Detailing the operational logic behind
functions, emphasizing knowledge and principles relevant

14
to telecom.
Based on the designed instruction fine-tuning dataset, the
steps of fine-tuning LAMs for communications are as follows:
•Model initialization: After creating the instruction fine-
tuning dataset, select a pre-trained LAM as the initial
model, ensuring it has strong language understanding and
generation capabilities for communications.
•Model adjustment and optimization: Use the instruction-
response pairs dataset for supervised fine-tuning (SFT) of
the pre-trained LAM, learning the relationship between
instructions and responses while adjusting model param-
eters. Then, define a negative log-likelihood loss function
to measure the difference between the model’s generated
responses and the expected ones [35].
•Iterative training: Through multiple iterations, the LAM
learns to generate quality responses based on instructions.
It updates its parameters using the loss function after
processing each batch of instruction-response pairs.
•Final evaluation and application: After training, the
LAM is evaluated to ensure it meets performance standards
across tasks. It is then tested in real-world scenarios
for practicality and reliability before being deployed in
communication applications.
There are various techniques for fine-tuning LAMs, including
LoRA, Adapters, BitFit, and Prefix Tuning as follows.
•LoRA [106] (Low-rank adaptation) is an efficient fine-
tuning method that reduces computational and storage
costs while maintaining model performance. By limiting
weight matrix updates to a low-rank subspace, it decreases
the number of parameters updated, improving fine-tuning
efficiency without compromising task performance.
•Adapter [107] is a fine-tuning method that adds small,
trainable modules at each layer of the LAM, keeping
the pre-trained model parameters fixed. This reduces the
number of parameters to update, saves resources, and
supports multi-task learning, making it ideal for resource-
limited scenarios.
•BitFit [108] (Bias-only fine-tuning) significantly reduces
computational and storage costs by updating only the bias
terms in the LAM. It minimizes parameter updates, main-
tains performance, and adapts quickly to specific tasks,
without requiring complex changes to pre-trained models.
•Prefix tuning [109] fine-tunes pre-trained LAMs by adding
a trainable prefix vector to the input sequence, while
keeping the model’s original weights fixed. It reduces
computational and storage costs by updating only the
prefix, making it efficient for adapting to specific tasks.
The fine-tuning stage helps LAM better understand and
execute communication instructions without explicit examples,
improving its ability to respond accurately to communication
tasks and enhancing its effectiveness in real-world applications.
3) Alignment of LAMs for communications: Alignment tun-
ing is a crucial step to better align the LAM’s responses with
human preferences. After SFT on a communication dataset, the
LAM may still generate undesirable responses, such as repeti-tion, overly short replies, or irrelevant content. Key alignment
techniques can address these issues.
Alignment tuning improves model performance by guiding
the LAM to generate more accurate and reasonable responses.
RLHF [36] is a form of alignment fine-tuning that combines
human feedback with traditional reinforcement learning to op-
timize LAM performance. RLHF is especially useful in com-
munication tasks, where decision-making and output reliability
are critical, enabling more efficient learning of complex tasks.
The RLHF workflow typically involves several key steps.:
•Environment and agent construction: Develop a fun-
damental reinforcement learning framework that includes
both the environment (alignment task) and the agent
(LAM).
•Human feedback collection: Collect feedback from hu-
man experts during the agent’s task execution through
interactive methods, including performance evaluations,
suggestions, or corrections.
•Reward modeling: Convert human feedback into reward
signals and train a reward model using machine learning
to accurately interpret and quantify the feedback into
appropriate reward values.
•Reinforcement training: Use reward signals from the re-
ward model to train the agent with reinforcement learning,
updating its strategy to gradually optimize performance and
better align with human expectations.
In addition to RLHF, there are also key alignment technolo-
gies such as reinforcement learning from AI feedback (RLAIF),
proximal policy optimization (PPO), and direct preference op-
timization (DPO) as follows.
•RLAIF [110] is a new method for improving the behav-
ior of LAMs. Unlike traditional RLHF, RLAIF uses AI-
generated feedback to optimize models, reducing the need
for large human-annotated datasets. AI agents (e.g., GPT-
4) evaluate model outputs and adjust parameters based
on these evaluations to improve performance. The process
involves two steps: first, the AI agent generates feedback by
evaluating the model’s outputs, and second, this feedback is
used to adjust the model through reinforcement learning,
gradually improving output quality. RLAIF is more effi-
cient and scalable, eliminating the need for costly human
data.
•PPO [111] is a reinforcement learning method that aims
to stabilize policy updates during optimization. Unlike
traditional policy-gradient methods, which involve com-
plex calculations and constraints to prevent large policy
shifts, PPO uses a ”surrogate objective function” and limits
the update step size. PPO introduces a penalty term to
control the magnitude of policy changes, ensuring that
the updated policy remains close to the original. This
approach improves policy performance, avoids expensive
constrained optimization, and results in better convergence
and robustness.
•DPO [112] is a reinforcement learning technique that
directly optimizes model outputs to match user or system
preferences, without using a reward model. By incorpo-

15
TABLE IV: Comparison of the three-stage learning process of LAMs [113]
ObjectiveData
RequirementLearning
ApproachAdjusting
ParametersPrivacyResource
Requirements
Pre-trainingLearning
general
language
representationAbundant and
diverse datasetsUnsupervised
learningAll parametersRequires
public datasetHigh
computational
and storage
resources
Fine-tuningUnderstanding
user
instructionsInstruction dataSupervised
learningFew
parametersMay require
user
instructionsLow
computational
and storage
resources
AlignmentAligning with
user
preferencesPreference-
specific dataReinforcement
learningFew
parametersMay require
user
preferencesLow
computational
and storage
resources
rating explicit preference feedback during training, DPO
avoids the complexity of traditional methods and improves
model performance. It is particularly effective in tasks
requiring fine control of model behavior and efficient
handling of complex preferences.
D. Evaluation of LAMs for communications
The evaluation of LAMs for communications is a critical
objective, as research on evaluation metrics not only influences
the performance of LAMs but also provides deeper insights into
their strengths and limitations in communication-related tasks.
Selecting high-quality telecommunication datasets is essential as
a prerequisite for effective evaluation. For example, Maatouk et
al. [114] proposed the benchmark dataset TeleQnA to assess
the knowledge of LLMs in telecommunications. The dataset
consists of 10,000 questions and answers drawn from diverse
sources, including telecommunication standards and research
articles. Additionally, TeleQnA introduced the automated ques-
tion generation framework used to create the dataset, which
incorporated human input at various stages to ensure its quality.
Once an appropriate benchmark dataset has been selected,
the evaluation of LAMs for communications is conducted. The
evaluation framework encompasses various aspects, including
communication Q&A, communication tool learning, communi-
cation modeling, and code design.
1) Communication Q&A: The evaluation of communication
Q&A [35] aims to assess the capability of LAMs, such as
GPT-4, to comprehend and process communication-related doc-
uments. This task involves generating multiple-choice and open-
ended questions from sources like literature, patents, and books
on communication topics, including technologies, protocols,
and network architectures. The performance of the LAM is
measured by comparing its responses to ground-truth answers,
with particular emphasis on its understanding and application
of communication knowledge.
The evaluation process starts with the selection of relevant
decoments, followed by data preprocessing. The LAM generates
questions based on the processed content, and the generated
answers are subsequently verified for accuracy, either manually
or through automated comparison with standard answers. TheLAM’s performance is evaluated by analyzing its responses
against the correct answers, focusing on accuracy as well as
comprehension and reasoning abilities. Metrics such as preci-
sion, recall, and f1 score are employed to measure the quality
of the answers and to assess the model’s overall effectiveness
in communication Q&A tasks.
2) Communication tool learning: The evaluation of tool
learning [115] examines whether LAMs can effectively select
and utilize communication tools, such as existing algorithms and
codes, to address real-world tasks. This capability is assessed
in two key areas: tool selection, which refers to the model’s
ability to choose appropriate tools through reasoning, and tool
usage, which involves leveraging these tools to enhance task
performance, such as integrating existing channel model code
with LAMs to perform channel prediction, thereby improving
the performance of communication systems.
The evaluation emphasizes two primary aspects: the model’s
ability to select the correct tools and its competence in perform-
ing operations with them. This includes assessing performance
with individual tools and the effectiveness of combining multi-
ple tools, as demonstrated in benchmarks like toolalpaca [116].
These benchmarks evaluate the LAM’s overall proficiency and
limitations in multi-tool usage. Insights from these evaluations
highlight the model’s strengths and challenges in tool selec-
tion and application, guiding future optimization efforts for
communication-related tasks.
3) Communication modeling: The evaluation of communi-
cation modeling focuses on assessing the ability of LAMs to
represent and solve mathematical problems related to com-
munication systems [35]. Tasks such as equation completion
are emphasized, where critical mathematical expressions are
concealed, and the physics-informed LAM must accurately
predict the missing components. The evaluation begins with
the selection of relevant mathematical models and equations to
ensure that the tasks are both challenging and representative of
real-world communication systems.
The LAM’s performance is evaluated by comparing its
predictions against standard answers, with particular attention
to accuracy and equation consistency. Beyond precision, the
evaluation also examines the model’s depth of reasoning and
understanding of complex communication principles. By com-

16
bining measures of accuracy with an assessment of reasoning
ability, this evaluation provides a comprehensive understanding
of the LAM’s effectiveness in tackling communication modeling
tasks.
4) Communication code design: The evaluation of commu-
nication code design [35] aims to assess the ability of LAMs to
generate, complete, and analyze communication-related code in
programming languages such as C, C++, Python, and Matlab.
The evaluation tasks include code generation, completion, and
analysis, testing the model’s proficiency in creating scripts,
completing partial code, and providing accurate summaries or
error analyses for communication tasks.
The evaluation begins by presenting programming scenarios
where the LAM (e.g., OpenAI Codex) is required to generate
code for tasks such as signal processing, network protocol
implementation, or data transmission algorithms. Subsequently,
the LAM is tested on code completion, where it predicts and fills
in missing segments, ensuring logical consistency and correct
functionality. Additionally, the LAM is tasked with analyzing
the given code, explaining its functionality, identifying errors,
and suggesting optimizations. Performance is measured by
comparing the generated code against standard answers, with
an emphasis on accuracy, completeness, and logical correctness.
The model’s ability to analyze code is also evaluated, reflecting
its understanding of programming concepts specific to commu-
nication.
E. Optimization of LAMs
To further improve the performance and adaptability of these
LAMs, researchers have proposed a variety of optimization
techniques, such as CoT, RAG and agent systems. In the fol-
lowing, we provide a detailed introduction to these optimization
techniques.
1) Chain of thought: CoT is a reasoning technique that was
first proposed by Google research in 2022 [32]. The main feature
of CoT is its ability to decompose complex problems into a
sequence of logical reasoning steps and solve them in a linear
and structured manner. It excels at solving tasks that require
multi-step reasoning and comprehensive analysis, making it
particularly suitable for scenarios that demand the simulation
of human thought processes, such as complex decision-making
and problem-solving. The workflow of the CoT method is as
follows:
•Task input: The model is presented with a complex
communication task or problem, which may be provided
as a natural language description, a mathematical equation,
or a logical reasoning question. Based on the nature of
the problem, the model identifies an appropriate reasoning
pathway and integrates relevant contextual information to
support the reasoning process.
•Logical reasoning: The model decomposes the problem
into a sequence of logical reasoning steps, performing
inference in a structured, step-by-step manner. The output
of each step is contingent on the results of the preceding
step, ensuring a coherent and systematic reasoning process.•Decision output: The model produces a logically con-
sistent answer or decision derived from the reasoning
process. Validation mechanisms are employed to verify
the correctness and reliability of the result, ensuring its
accuracy and dependability.
CoT-based research in communication: With the rapid devel-
opment of the AI field, CoT technology has gradually gained
widespread attention as an innovative reasoning framework.
CoT helps models handle complex reasoning and decision-
making tasks more efficiently by simulating hierarchical and
structured reasoning in communication. For example, Du et al.
[117] applied CoT technology to help LLMs perform multi-step
reasoning in field-programmable gate array (FPGA) develop-
ment and solve complex tasks such as the implementation of fast
fourier transform (FFT). CoT prompts enable LLMs to gradually
decompose problems and perform calculations, improving the
accuracy of generated hardware description language (HDL)
code. Zou et al. [118] used CoT technology in the GAI Network
(GenAINet) framework to help distributed GAI agents perform
collaborative reasoning. Agents use CoT prompts to decompose
complex tasks and acquire knowledge from other agents, thereby
improving decision-making efficiency and reducing commu-
nication resource consumption. Shao et al. [119] used CoT
technology in the wireless LLM (WirelessLLM) framework to
improve the reasoning ability of LLMs, helping the model
gradually handle complex tasks in wireless communications,
such as power allocation and spectrum sensing. This approach
effectively enhances the task performance capabilities of LLMs
in multimodal data environments.
2) Retrieval-augmented generation: RAG is a technology
that integrates retrieval and generation, proposed by Facebook in
2020 [120]. RAG combines two steps of retrieval and generation
to enhance the answering ability of the LAM by retrieving
relevant documents. RAG can use the retrieval module to obtain
the latest and most relevant information while maintaining the
powerful language capabilities of the LAM, thereby improving
the accuracy and relevance of the answer. It excels at tasks that
are information-rich but require knowledge from large amounts
of text, such as answering questions, generating detailed instruc-
tions, or performing complex text generation. The workflow of
RAG technology is given as follows:
•Information retrieval: Retrieve documents related to the
input content from an external knowledge base. By using
information retrieval technology, the LAM can filter out
the documents that best match the input question from the
knowledge base.
•Information fusion: The retrieved documents are spliced
with the input question as the new input of the LAM.
In the information fusion stage, the LAM processes the
documents and input content through the encoder, closely
combines the retrieved knowledge with the question, and
enhances the model’s understanding and generation capa-
bilities of the question.
•Generate output: The information-fused input is passed
to the LAM, and the LAM not only relies on the original
input, but also uses the retrieved document information

17
to provide richer and more accurate answers. The gen-
eration process ensures that the answer is coherent and
contextually relevant, thereby ensuring the rationality and
effectiveness of the output.
RAG-based research in communication: In communication,
RAG technology has demonstrated excellent application poten-
tial. For example, Bornea et al. [121] proposed Telco-RAG, an
open-source RAG system designed for the 3GPP documents,
which enhances the performance of LLMs in telecom. Tang et
al. [122] proposed an automatic RAG framework for 6G use
cases, using LLMs to automatically generate network service
specifications, especially in the environment of an open radio
access network (ORAN). Through this method, business inno-
vators can quickly and cost-effectively evaluate and generate
the required communication specifications without having an
in-depth understanding of complex 6G standards, which greatly
promotes innovation and application deployment in an open 6G
access environment. Huang et al. [123] proposed a 6G network-
based RAG service deployment framework, aiming to improve
the quality of generation services by combining LLMs with
external knowledge bases. The article explores the feasibility
of extending RAG services through edge computing, and pro-
poses technical challenges in multimodal data fusion, dynamic
deployment of knowledge bases, service customization, and user
interaction, providing innovative directions for RAG services
in future 6G networks. Xu et al. [124] proposed LMMs as a
general base model for AI-native wireless systems. The frame-
work combines multimodal perception, causal reasoning, and
RAG to handle cross-layer network tasks, and experimentally
verified the effectiveness of LMMs in reducing hallucinations
and improving mathematical and logical reasoning capabilities.
Yilma et al. [125] introduced the telecomRAG framework,
which combines RAG with LLM technologies to help telecom
engineers parse complex 3GPP standard documents and gen-
erate accurate and verifiable responses. By retrieving standard
documents of 3GPP release 16 and release 18, the framework
provides a solution with higher accuracy and technical depth
than the general LLM for the telecom.
3) Agentic system: The agentic system is a framework con-
sisting of LAM-based agents that perceive their environment and
collaborate to achieve specific objectives. The primary charac-
teristics of the agentic system include autonomy, adaptability,
and interactivity. It can adjust its behavior according to changes
in the environment and interact with other agents or the environ-
ment to optimize decision-making and task execution. It excels
at solving communication problems that require dynamic re-
sponsiveness, complex decision-making, and task optimization.
By simulating the behavior of humans or biological systems,
agents can efficiently accomplish tasks in dynamic and changing
communication environments. The workflow of the LAM-based
agentic system is as follows:
•Task understanding and planning : The agentic system
interprets input instructions, extracts relevant context, and
breaks down complex tasks into smaller, manageable sub-
tasks. It then formulates a logical plan to execute these
subtasks.•Execution and adaptation : The agent executes the
planned actions, leveraging the LAM for tasks such as
generating content, solving problems, or interacting with
external systems. It continuously monitors progress and
adapts dynamically to environmental changes or unex-
pected outcomes.
•Validation and feedback : The agentic system validates the
results to ensure accuracy and consistency, providing reli-
able outputs. Feedback from the process is integrated into
the system, enabling iterative improvement and enhanced
performance in future tasks.
Agent-based research in communication: By leveraging its
autonomy, adaptability, and interactivity, the agentic system
effectively addresses complex tasks and problems, demon-
strating exceptional potential in communication. For example,
Tong et al. [126] proposed the wirelessagent framework, which
uses LLM as the core driver and constructs a multi-agent
system through four modules: perception, memory, planning,
and action. The wirelessagent framework aims to solve the
increasingly complex management problems in wireless net-
works, especially in the upcoming 6G era, when traditional
optimization methods cannot cope with complex and dynami-
cally changing network requirements. Through the collaboration
and communication between multiple agents, the framework can
autonomously process multimodal data, perform complex tasks,
and make adaptive decisions. Xu et al. [79] proposed a 6G
network-based LLMs agent split learning system to address the
issue of low local LLMs deployment and execution efficiency
due to the limited computational power of mobile devices. The
system achieves collaboration between mobile devices and edge
servers, with division of labor between the perception, semantic
alignment, and context binding modules to complete the in-
teraction tasks between the user and the agent. Additionally,
by introducing a novel model caching algorithm, the system
improves model utilization, thereby reducing the network cost
of collaborative mobile and edge LLMs agents. Yang et al. [127]
proposed an agent-driven generative semantic communication
(A-GSC) framework based on reinforcement learning to tackle
the challenges of large data volumes and frequent updates in
remote monitoring for intelligent transportation systems and
digital twins in the 6G era. Unlike existing semantic commu-
nication research, which mainly focuses on semantic extraction
or sampling, the A-GSC framework successfully integrates the
intrinsic properties of the source information with the task’s
contextual information. Furthermore, the framework introduces
GAI to enable the independent design of semantic encoders and
decoders.
F . Summary and lessons learned
1) Summary: This chapter provides a comprehensive
overview of the key architecture, classification, training, eval-
uation, and optimization of LAMs in communication. First,
we introduce the key architecture of LAMs. Next, we present
a more detailed classification system for LAMs in commu-
nication. Following this, we discuss the training process for
communications LAMs, summarizing the complete workflow

18
Fig. 4: Applications of LAMs in Communication. LAMs can be applied across various domains in communication, including
physical layer design, resource allocation and optimization, network design and management, edge intelligence, semantic
communication, agentic systems, and emerging applications.
from pretraining and fine-tuning to alignment, with an in-
depth explanation of each of these three techniques. We then
introduce the evaluation methods for communications LAMs
[128], providing a comprehensive summary of the standards and
metrics used to assess LAM performance in communication.
Finally, we explore various optimization techniques of LAMs
for communications [129]. This chapter lays a solid foundation
for the application of LAMs and offers clear directions for their
future development.
2) Lessons learned: Although progress has been made in the
construction and optimization of LAMs in communication, sev-
eral lessons can be learned. Current mainstream architectures,
such as transformer [130], diffusion, and mamba, demonstrate
excellent modeling and reasoning capabilities. However, they
still face significant difficulties in resource-constrained environ-
ments, multimodal tasks, and real-time communication applica-
tions. These challenges include high computational complexity,
slow convergence, and difficulties in training, evaluation, and
deployment. Regarding optimization strategies, although meth-
ods like CoT, RAG, and agentic systems have effectively en-
hanced the model’s reasoning and task adaptability, limitations
still exist in terms of stability, consistency, and efficiency.
III. LAM S FOR PHYSICAL LAYER DESIGN
With the continuous development of wireless communication
technology, especially the demand for 6G networks, physical
layer design faces increasingly complex challenges. In order to
meet these challenges, LAMs and GAI models have gradually
become key tools in physical layer design.
A. Channel and beam prediction based on LAMs
With the rapid advancement of wireless communication
systems, particularly in the context of 5G and the evolution
toward 6G networks, the demand for accuracy and efficiency inchannel and beam prediction has grown significantly. Traditional
methods often fall short when dealing with the complex and
dynamic nature of modern networks. In recent years, break-
throughs in LLMs have provided new approaches to address
these challenges. For example, Fan et al. [131] proposed CSI-
LLM, a method for downlink channel prediction in large-
scale MIMO systems. By aligning wireless data with NLP
tasks, it leverages LLMs for modeling variable-length his-
torical sequences, showing strong performance, especially in
multi-step prediction under high dynamics. Liu et al. [101]
proposed LLM4CP, a channel prediction method using pre-
trained LLMs. It combines channel characteristic modules and
cross-modal knowledge transfer for accurate TDD/FDD predic-
tion, reducing training costs and showing strong generalization
and efficiency. Sheng et al. [132] studied beam prediction in
millimeter-wave communication, using LLMs to convert time-
series data into text and enhancing context with the prompt-
as-prefix technique. Compared to traditional LSTM models,
this approach shows stronger robustness and generalization in
high-dynamic environments. Akrout et al. [133] reviewed deep
learning in the wireless physical layer, emphasizing trade-offs in
accuracy, generalization, compression, and latency. They noted
that focusing on accuracy often limits model performance in
complex communication scenarios due to poor generalization.
By analyzing the decoding tasks of end-to-end communication
systems, the important impact of this trade-off on the practical
application of the model is revealed, especially when LLMs
are used for wireless communications, the balance between
compression and latency becomes a crucial factor.
B. Automated physical layer design based on LAMs
As wireless networks continue to grow in scale and complex-
ity, the need for intelligent and automated physical layer design
has become increasingly urgent. LLMs and GAI technologies

19
are emerging as powerful tools to address this demand, offering
new possibilities for building adaptive and efficient communi-
cation systems. For example, Xiao et al. [134] proposed an
LLM-based 6G task-oriented physical layer automation agent
(6G LLM agents), introducing LLMs as intelligent co-pilots to
enhance understanding and planning for dynamic tasks through
multimodal perception and domain knowledge. Using a two-
stage training framework, the agent effectively performs proto-
col question answering and physical layer task decomposition.
Wang et al. [135] proposed a physical layer design framework
using GAI agents that combine LLM with RAG technologies,
demonstrating strong potential in signal processing and analysis.
GAI agents enable rapid generation of complex channel models
across environments, accelerating research in next-generation
MIMO channel modeling and estimation.
C. Summary and lessons learned
1) Summary: This chapter discusses the application of LAMs
in physical layer design, demonstrating their potential in channel
estimation, task decomposition, signal processing, etc. LAM
can significantly improve the intelligence and automation level
of the physical layer design through its powerful reasoning
capabilities and multi-task learning [136]. The LAM improves
channel estimation and blind channel equalization by accurately
modeling complex data distribution. LAMs provide innovative
ideas and methods for physical layer design, and are expected
to bring breakthroughs in performance improvement and system
optimization of future wireless communication systems [137].
2) Lessons learned: From the chapter, we have learned
several important lessons. First, LAMs exhibit limited inter-
pretability in physical layer optimization [137]. Although they
can generate seemingly effective optimization strategies, they
often lack rigorous mathematical analysis or theoretical guar-
antees, which constrains their application in high-reliability
communication scenarios. Second, the training and inference
of LAMs rely heavily on high-quality annotated data, yet the
acquisition and labeling of physical layer data are costly, making
it difficult to scale data-driven LAM models in practical deploy-
ments. Therefore, enhancing interpretability and addressing data
acquisition challenges are key directions for future research.
IV. LAM S FOR RESOURCE ALLOCATION AND OPTIMIZATION
Resource allocation and optimization are complex and critical
issues in communication networks. With the development of
LAMs, their application in this field has gradually shown great
potential. In the following, we discuss the application of LAMs
in computing resource allocation, spectrum resource allocation,
and energy resource optimization.
A. Computational resource allocation
As communication networks become more complex, users
have higher requirements for network services. How to provide
users with high-quality communication services under limited
computing resources is a major challenge. The study of com-
puting resource allocation based on GAI models and LAMis an important research direction in the future. For example,
Du et al. [138] introduced the AGOD algorithm, which uses
a diffusion model to generate optimal AIGC service provider
(ASP) selection decisions from Gaussian noise. Combined with
DRL into the D2SAC algorithm, it enhances ASP selection
efficiency and optimizes user computing resource allocation.
In addition, Du et al. [139] proposed a network optimization
method based on the MoE framework and LLM, leverag-
ing LLM reasoning to manage expert selection and decision
weighting, enabling efficient resource allocation and reduced
energy and implementation costs. Tests on maze navigation and
network service provider (NSP) utility tasks demonstrate its
effectiveness in complex network optimization.
B. Spectrum resource allocation
In current communication systems, spectrum resource alloca-
tion is an important part of achieving efficient and reliable data
transmission. With the rapid development of mobile communi-
cation technology, especially in the 5G and upcoming 6G era,
the demand for spectrum resources has increased dramatically,
while the available spectrum resources are very limited. In order
to improve spectrum utilization and meet users’ needs for high-
speed, low-latency communication, researchers have explored
spectrum resource allocation schemes based on LAMs and GAI
models. For example, Zhang et al. [140] proposed a GAI agent-
based framework using LLM and RAG to build accurate system
models via interactive dialogue. To optimize these models, they
introduced an MoE PPO method that combines expert networks
with PPO, enabling collaborative decision-making to enhance
spectrum efficiency and communication quality. In addition, Du
et al. [141] proposed a GAI and DRL-based framework for
optimizing computation offloading and spectrum allocation in
802.11ax Wi-Fi. Combining GDM with the TD3 algorithm and
using the Hungarian algorithm for RU allocation, it improved
bandwidth utilization, latency, and energy consumption in sim-
ulations.
C. Energy resource optimization
Energy resource optimization is equally crucial in communi-
cation networks, especially in scenarios such as mobile com-
munications and IoT. Traditional energy optimization methods
are often based on heuristic rules or simple algorithms, which
struggle to achieve optimal results in complex and dynamic en-
vironments. Researchers are actively exploring energy resource
optimization schemes based on GAI models and LAMs to
achieve low energy consumption and high efficiency in wireless
communications. For example, Xu et al. [142] proposed a
GAI-based mobile multimedia network framework for dynamic
adaptive streaming, intelligent caching, and energy efficiency
optimization, enhancing multimedia content distribution. The
framework optimizes resource utilization and reduces energy
consumption by considering the value of the GAI model and
other indicators. Du et al. [143] proposed a wireless edge
network framework using AIGC services to optimize energy
allocation and improve user experience. By dynamically select-
ing the optimal ASP with a DRL algorithm, the framework

20
reduces task overload and retransmissions, enhancing energy ef-
ficiency and service quality. Simulations showed reduced energy
consumption and improved content quality and transmission
efficiency.
D. Summary and lessons learned
1) Summary: In this chapter, we summarize the application
of LAMs in computing and spectrum resource allocation, and
energy resource optimization. LAMs can intelligently allocate
resources through real-time prediction and analysis of network
demand [144], and LAMs can also optimize energy usage
strategies by learning the energy consumption patterns in com-
munication networks [145].
2) Lessons learned: From the chapter, we have learned
several important lessons. First, while LAMs can improve
optimization efficiency in computational resource allocation,
their generalization ability is limited under resource-constrained
and dynamic communication environments, potentially resulting
in suboptimal or even infeasible allocation strategies [146]. In
spectrum resource allocation, although LAMs can assist in en-
hancing spectrum utilization efficiency, their inference processes
often rely on complex combinations of expert networks and
scheduling mechanisms, leading to significant computational
overhead and difficulty in meeting real-time requirements. Re-
garding energy resource optimization, LAMs are capable of
reducing energy consumption through intelligent caching and
flow control. However, their stability and interpretability in
generating dynamic scheduling strategies remain insufficient.
Therefore, improving the generalization ability of LAMs in
resource allocation and optimization, as well as reducing their
computational complexity, are key challenges that need to be
addressed in future research.
V. LAM S FOR NETWORK DESIGN AND MANAGEMENT
LAMs play a vital role in network design and management.
Through the powerful ability of generative learning, they can
conduct detailed analysis and prediction of network traffic,
user behavior, and system performance, and empower existing
networks.
A. Network design
Intelligent network design is the key to ensuring efficient
operation of the system and high-quality services. At present,
LAMs are widely used in network design with their power-
ful generation and data processing capabilities. For example,
Huang et al. [42] proposed an AI-generated network (AIGN)
framework using GAI and reinforcement learning for automated
network design. It employs a diffusion model to learn design
intent and generate customized solutions under multiple con-
straints, enabling intelligent network design. Zou et al. [147]
proposed a wireless multi-agent GAI network that leverages
LLMs on devices for autonomous networking. Multi-agent GAI
integration enables network model design, reasoning, multi-
modal data processing, and resource management. Huang et
al. [42] proposed ChatNet, a domain-adaptive network LLMframework that uses natural language for intelligent network
design, diagnosis, configuration, and security. ChatNet auto-
mates tasks by pre-training and fine-tuning open-source LLMs
to understand network language and access external tools like
simulators, search engines, and solvers.
B. Network management
Traditional data processing methods are difficult to meet the
6G network’s requirements for massive data, complex tasks, and
real-time performance, and the emergence of LAMs provides
new ideas for solving these problems. For example, Wang et
al. [148] proposed NetLM, a network AI architecture using
ChatGPT for network management and optimization. Based on
LLM, NetLM analyzes network packet sequences and dynam-
ics, unifying network indicators, traffic, and text data through
multimodal representation learning to enhance data processing
and understand network status, user intent, and complex patterns
in 6G networks. Dandoush et al. [149] proposed a frame-
work combining LLMs with multi-agent systems for network
slicing management. Network slicing enables virtual networks
on shared infrastructure, but current methods struggle with
complex service requirements. The framework uses LLMs to
translate user intents into technical requirements and a multi-
agent system for cross-domain collaboration, enabling efficient
slice creation and management. It also addresses challenges like
data acquisition, resource demands, and security. Yue et al.
[150] proposed a LAM-enabled 6G network architecture that
enhances management efficiency by extracting insights from
heterogeneous data. LAMs automate operations, maintenance,
and reasoning tasks, reducing human intervention. Using edge
computing, LAMs process data in high-concurrency scenarios,
improving performance and resource scheduling. The study also
addresses challenges like data governance and computational
resource needs in 6G networks.
C. Summary and lessons learned
1) Summary: This chapter summarizes the application of
LAMs in network design and management, including the op-
timization of network architecture design and the management
of network slicing. LAMs use their powerful data processing
and generation capabilities to efficiently process large amounts
of data in 6G networks and realize intelligent network design
and management [151] [152].
2) Lessons learned: From the chapter, we have learned sev-
eral important lessons. First, in network design, although LAMs
can automatically generate customized network solutions that
satisfy multiple constraints by learning network intents, ensuring
the feasibility and stability of the design under multi-constraint
conditions remains a significant challenge. Moreover, in network
management, LAMs can enhance capabilities in network state
analysis, user intent understanding, and data pattern learning,
yet difficulties persist in handling large-scale heterogeneous data
and meeting real-time performance requirements [42].

21
VI. LAM S FOR EDGE INTELLIGENCE
LAMs have a wide range of application scenarios for im-
proving edge intelligence. In the following, we discuss the three
aspects of LAMs for edge intelligence.
A. Edge training and application of LAMs
Edge LAMs are widely used in edge devices due to their
ease of deployment while retaining strong data processing ca-
pabilities. For example, the Edge-LLM framework proposed by
Yu et al. [153] addresses computational and memory overheads
in adapting LLMs on edge devices. It uses a layered unified
compression (LUC) technique to optimize model compression,
reduces memory usage through adaptive layer tuning, and
introduces hardware scheduling to handle irregular computation
patterns.
The EdgeShard framework proposed by Zhang et al. [154]
distributes LLMs across multiple devices using model sharding.
It employs a dynamic programming algorithm to optimize
device selection and model partitioning, balancing inference
latency and throughput. Experimental results demonstrate a 50%
reduction in latency and a doubling of throughput, providing
an efficient solution for LLMs inference in collaborative edge
computing. Qu et al. [48] reviewed the integration of LLMs with
mobile edge intelligence (MEI) and proposed the MEI4LLM
framework to improve deployment efficiency in edge environ-
ments. The paper covers technologies like caching, distributed
training, and inference, and discusses future directions such
as green computing and secure edge AI. It highlights the
importance of edge intelligence for low-latency and privacy-
sensitive tasks, providing a theoretical foundation for broader
LLM applications.
Xu et al. [155] explored federated fine-tuning of LLMs
and FM in edge networks, focusing on memory bandwidth
limitations. They used energy efficiency to measure compu-
tational efficiency, comparing it with model FLOP utilization
(MFU). Results show that energy efficiency is better for real-
time monitoring, with optimal efficiency achieved on embedded
devices using smaller batch sizes.
In addition, Zhao et al. [156] proposed a framework for
LLM deployment that combines edge and terminal collabora-
tion, using serial inference on terminals and parallel inference
on edge servers. This reduces latency and optimizes energy
consumption, improving model performance across different
network conditions and offering an efficient solution for LLM
deployment in wireless networks. Khoshsirat et al. [157] studied
the application of decentralized LLM inference on energy-
constrained edge devices and proposed an inference framework
that integrates energy harvesting, enabling distributed devices
to collaboratively perform model inference tasks. Lin et al.
[158] explored deploying LLMs in 6G edge environments,
addressing computational load with split learning, quantization,
and parameter-efficient fine-tuning. The paper proposed tailored
LLMs training and inference strategies for edge environments,
providing a research path for distributed AI in 6G networks.
Rong et al. [159] proposed the LSGLLM-E architecture for
large-scale traffic flow prediction, addressing spatiotemporalcorrelation issues in road networks. The method reduces central
cloud pressure by decomposing the network into sub-networks
and using RSUs as edge nodes for computation. The LSGLLM
model captures dynamic spatiotemporal features, overcoming
the limitations of existing LLMs in large-scale road network
predictions.
B. Edge resource scheduling meets LAMs
Edge devices face limitations in computational power and
storage, while LAMs require efficient computation, real-time
response, and low-latency data transmission. This creates two
main challenges: (1) how to effectively allocate resources to
ensure the efficient operation of LAMs on edge devices, and (2)
how to leverage the powerful optimization capabilities of LAMs
to design improved edge resource scheduling strategies. To
address these, various solutions have been proposed, integrating
task offloading, computational, and storage resource optimiza-
tion to enhance edge device performance in AI tasks. For ex-
ample, Friha et al. [160] analyzed LLM-based edge intelligence
optimization in resource-constrained environments and pro-
posed strategies to address computing and storage limitations.
Techniques like model compression, memory management, and
distributed computing enable efficient LLM operation on edge
devices. These optimizations improve deployment efficiency and
expand LLM applications in areas like personalized medicine
and automation. Dong et al. [39] proposed the LAMBO frame-
work for LLM-based mobile edge computing (MEC) offloading,
addressing challenges in traditional deep offloading architec-
tures, such as heterogeneous constraints and local perception.
The framework uses an input embedding (IE) model to convert
task data and resource constraints into embeddings, and an
asymmetric encoder-decoder (AED) model to extract features
and generate offloading decisions and resource allocations. Lai
et al. [161] proposed the GMEN framework to enhance the
intelligence and efficiency of mobile edge networks in the 6G
era. By combining GAI with edge networks and using methods
like model segmentation, the framework offloads AI tasks to
reduce network burden. The Stackelberg game model is applied
to optimize resource allocation and encourage edge devices to
contribute computing resources, reducing overhead.
C. Federated learning of LAMs
FL protects privacy and reduces reliance on centralized re-
sources by training models locally, but traditional small models
have limited capabilities. The emergence of LAM, with its
powerful representation capabilities, enables FL to handle more
complex tasks without centralized data, significantly improving
personalized services and prediction accuracy.
For example, Xu et al. [162] proposed FwdLLM, a federated
learning protocol that enhances LLM on mobile devices using a
backpropagation-free training method. FwdLLM combines effi-
cient parameter fine-tuning techniques like LoRA and adapters
to distribute the computational load, improving memory and
time efficiency, and enabling LLM fine-tuning on ordinary
commercial mobile devices. Peng et al. [163] proposed a person-
alized semantic communication system using GAI, enhancing

22
performance through personalized local distillation (PLD) and
adaptive global pruning (AGP). PLD allows devices to select
models based on local resources and distill knowledge into
a simpler model for FL. AGP improves the global model by
pruning it based on the communication environment, reducing
energy consumption and improving efficiency. Through these
innovative methods, the application of LAM in personalized FL
has demonstrated significant advantages. In addition, Jiang et al.
[113] proposed two personalized wireless federated fine-tuning
methods: personalized federated instruction tuning (PFIT) and
personalized federated task tuning (PFTT). PFIT uses rein-
forcement learning with human feedback for personalization,
while PFTT combines a global adapter with LoRA to reduce
communication overhead and accelerate fine-tuning, addressing
privacy, data heterogeneity, and high communication challenges
in wireless networks.
D. Summary and lessons learned
1) Summary: This chapter summarizes the application of
LAMs in edge intelligence, including the edge training and
application of LAMs, resource management and scheduling, as
well as federated learning of LAMs. Through edge training and
the application of LAMs, the performance of LAMs on edge
devices can be effectively enhanced. Resource management and
scheduling enable dynamic resource allocation through LAMs
[164]. LAMs enables federated learning to tackle more complex
tasks without centralized data, improving prediction accuracy
and enhancing personalized services [165].
2) Lessons learned: From the chapter, we have learned
several important lessons. First, when training and deploying
LAMs on edge devices, the limited computational and memory
resources pose a significant barrier to their widespread adoption
[166]. This issue is particularly critical in latency-sensitive
applications, where effectively reducing model parameters and
optimizing computation patterns remains an urgent area for
further investigation. In addition, federated learning of LAMs
faces challenges such as limited resources, data heterogeneity,
and personalization, with future research focusing on efficient
collaboration, robust optimization, and privacy-preserving per-
sonalization.
VII. LAM S FOR SEMANTIC COMMUNICATION
The rapid advancement of communication technology is
continuously propelling human society towards higher levels
of intelligence. In particular, the emergence of LAMs has
profoundly revolutionized the design and optimization of com-
munication systems, shifting the paradigm from traditional data
communication to semantic communication. This transformation
extends beyond signal transmission to encompass information
comprehension, unlocking a wide range of potential application
scenarios. We present an overview of the related work of LAMs
in semantic communications in the following sections.
A. LLM-based semantic communication systems
LLMs, with their powerful natural language understanding
and generation capabilities, can perform semantic-level analysisand processing in complex communication environments, sig-
nificantly enhancing the intelligence of semantic communication
systems. Especially in future networks like 6G, LLMs can
support more efficient and flexible semantic communication
architectures, enabling intelligent applications of semantic com-
munication. For example, Wang et al. [167] proposed a semantic
communication system framework based on LLMs, applying
them directly to physical layer encoding and decoding. The
system leveraged LLM training and unsupervised pre-training to
build a semantic knowledge base, used beam search algorithms
to optimize decoding and reduce complexity, and required no
additional retraining or fine-tuning of existing LLMs. Jiang et
al. [168] proposed a large generative model-assisted talking
face semantic communication system (LGM-TSC) to address
the challenges in talking face video communication, including
low bandwidth utilization, semantic ambiguity, and degraded
quality of experience (QoE). The system introduces a generative
semantic extractor (GSE) based on the FunASR model at the
transmitter, which converts semantically sparse talking face
video into high information density text. A private knowledge
base (KB) based on LLMs is used for semantic disambiguation
and correction, complemented by a joint knowledge base seman-
tic channel coding scheme. At the receiver side, the generative
semantic reconstructor (GSR) using BERTVITS2 and SadTalker
models converts the text back into a high-QoE talking face video
that matches the user’s voice tone. Chen et al. [169] proposed
a novel semantic communication framework based on LLMs to
address the challenges in underwater communication, including
semantic information mismatch and the difficulty of accurately
identifying and transmitting critical information. The framework
leverages visual LLMs to perform semantic compression and
prioritization of underwater image data, selectively transmitting
high-priority information while applying higher compression
rates to less important areas. At the receiver side, the LLMs-
based recovery mechanism works in conjunction with global
visual control networks and key region control networks to
reconstruct the image, improving communication efficiency and
robustness. The system reduces the overall data size to 0.8% of
the original data.
In addition, Jiang et al. [77] proposed a method for integrating
FMs, including LLMs, across the validity, semantic, and physi-
cal layers in semantic communication systems. This integration
leverages general knowledge to alter system design, thereby
improving semantic extraction and reconstruction. The study
also explored the use of compact models to balance performance
and complexity, and compared three approaches using FMs. The
research emphasizes the need for further analysis of the impact
of FMs on computational and memory complexity, as well as
the unresolved issues that require attention in this field. Kalita
et al. [170] proposed a framework that integrates LLMs with
semantic communication at the network edge to enable efficient
communication in IoT networks. The framework leverages the
capabilities of LLMs, training on diverse datasets with billions
of parameters, to improve communication performance in sce-
narios where current technologies are approaching the Shannon
limit. The system is designed to run on near-source computing

23
technologies such as edge, thereby enhancing communication
efficiency in IoT environments. Wang et al. [78] proposed
a general end-to-end learning semantic communication model
using LLMs to enhance the performance of next-generation
communication systems. The model combines subword-level
tokenization, a gradient-based rate adapter to match the rate
requirements of any channel encoder/decoder, and fine-tuning
to incorporate private background knowledge.
B. Other LAM-based semantic communication systems
In addition to LLMs, research on semantic communication
systems based on other LAMs also plays a crucial role in
advancing the intelligence of semantic communication systems
[171] [172]. For example, Jiang et al. [173] proposed a novel
cross-modal semantic communication system based on VLM
(VLM-CSC) to address the challenges in image semantic com-
munication, such as low semantic density in dynamic envi-
ronments, catastrophic forgetting, and uncertain signal-to-noise
ratio. The VLM-CSC system includes three key components:
(1) a cross-modal knowledge base, which extracts high-density
textual semantics from semantically sparse images at the trans-
mitter side and reconstructs the original image at the receiver
side to alleviate bandwidth pressure; (2) an memory-augmented
encoder and decoder, which employs a hybrid long/short-
term memory mechanism to prevent catastrophic forgetting
in dynamic environments; and (3) a noise attention module,
which adjusts semantic and channel coding based on SNR to
ensure robustness. Zhang et al. [174] proposed the ”Plan A
- Plan B” framework using MLLMs to address the out-of-
distribution (OOD) problem in image semantic communication.
It leverages MLLMs’ generalization to assist traditional models
during semantic encoding. A Bayesian optimization scheme
reshapes MLLM distributions by filtering irrelevant vocabulary
and using contextual similarity as prior knowledge. At the
receiver, a ”generate-critic” framework improves reconstruction
reliability, addressing the OOD problem and enhancing semantic
compression. Jiang et al. [56] proposed the GAM-3DSC system
to address challenges in 3D semantic extraction, redundancy,
and uncertain channel estimation in 3D scene communication.
By introducing LVM, the system enables user-driven 3D se-
mantic extraction, adaptive multi-view image compression, and
CSI estimation and optimization for effective target-oriented 3D
scene transmission. Xie et al. [175] proposed a new semantic
communication architecture that integrates large models by
introducing a memory module. This enhances semantic and
contextual understanding, improves transmission efficiency, and
addresses spectrum scarcity.
Yang et al. [176] proposed the ”M2GSC” framework for
generative semantic communication in multi-user 6G systems.
It employs MLLMs as a shared knowledge base (SKB) for task
decomposition, semantic representation standardization, and
translation, enabling standardized encoding and personalized
decoding. The framework also explores upgrading the SKB to a
closed-loop agent, adaptive encoding offloading, and multi-user
resource management. Do et al. [177] proposed a mamba-based
multi-user multimodal deep learning semantic communicationsystem to enhance efficiency in resource-constrained networks.
By replacing the transformer with mamba architecture, the
system improves performance and reduces latency. It introduces
a new semantic similarity metric and a two-stage training
algorithm to optimize bit-based metrics and semantic similarity.
Jiang et al. [76] proposed a multimodal semantic commu-
nication (LAM-MSC) framework based on LAMs to address
the challenges in multimodal semantic communication, such
as data heterogeneity, semantic ambiguity, and signal distortion
during transmission. The framework includes multimodal align-
ment (MMA) based on MLM, which facilitates the conversion
between multimodal and unimodal data while maintaining se-
mantic consistency. It also introduces a personalized knowledge
base (PKB) based on LLMs to perform personalized semantic
extraction and recovery, thereby resolving semantic ambiguity.
Additionally, a channel estimation method based on conditional
GANs is used to estimate wireless CSI, mitigating the impact
of fading channels on semantic communication.
C. Summary and lessons learned
1) Summary: This chapter summarizes the applications of
LAMs in semantic communications, including LLMs and other
LAMs. The powerful data processing capabilities of LAMs
can effectively reduce communication overhead [178], improve
communication efficiency, enhance the expression and under-
standing of semantic information, and enable more flexible,
intelligent, and efficient semantic communications [78].
2) Lessons learned: From the chapter, we have learned sev-
eral important lessons. First, although LAMs show remarkable
performance in semantic extraction and reconstruction when
directly applied to physical-layer encoding and decoding, their
high computational complexity remains a major bottleneck
for real-time deployment in resource-constrained environments
[179]. Second, current semantic communication systems have
yet to fully address key issues such as semantic information
alignment, ambiguity resolution, and bandwidth utilization op-
timization under dynamic network conditions. This is especially
evident in multi-user and multimodal scenarios, where effective
semantic standardization and cross-modal collaboration remain
open research problems.
VIII. LAM- BASED AGENTIC SYSTEMS
The application of intelligent agentic systems based on LLMs
and other GAI models is an important way to address the
challenges faced by current communication systems. These
intelligent agent-driven systems can improve the transmission
efficiency of semantic communication systems and optimize
resource allocation of edge devices.
A. Agentic systems based on LLMs
Agentic systems based on LLMs are widely applied in
communication systems due to their powerful NLP capabilities.
For example, Xu et al. [79] proposed a 6G-based LLM agent
split learning system to improve the efficiency of local LLM
deployment on resource-limited mobile devices. The system

24
enables mobile-edge collaboration through modules for per-
ception, semantic alignment, and context binding. A model
caching algorithm enhances model utilization and reduces net-
work costs for collaborative LLM agents. Jiang et al. [40]
proposed a multi-agent system to address challenges LLMs
face in 6G communication evaluation, including lack of native
data, limited reasoning, and evaluation difficulties. The system
comprises multi-agent data retrieval (MDR), cooperative plan-
ning (MCP), and evaluation and reflection (MER). A semantic
communication system case study demonstrated its effective-
ness. Tong et al. [126] proposed WirelessAgent, which uses
LLMs to build AI agents addressing scalability and complexity
in wireless networks. With advanced reasoning, multimodal
data processing, and autonomous decision-making, it enhances
network performance. Applied to network slicing management,
WirelessAgent accurately understands user intentions, allocates
resources effectively, and maintains optimal performance.
In addition, Zou et al. [147] proposed wireless multi-agent
GAI networks to overcome cloud-based LLM limitations by
enabling task planning through multi-agent LLMs. Their method
explores game-theory-based multi-agent LLMs and designs an
architecture for these systems. A case study demonstrates how
device-based LLMs collaborate to solve network solutions.
Wang et al. [135] proposed GAI Agents, a next-generation
MIMO design method addressing challenges in performance
analysis, signal processing, and resource allocation. By combin-
ing GAI agents with LLMs and RAG, the method customizes
solutions. The paper discusses the framework and demonstrates
its effectiveness through two case studies, improving MIMO
system design. Zhang et al. [140] proposed GAI agents for satel-
lite communication network design, tackling system modeling
and large-scale transmission challenges. The method uses LLMs
and RAG to build interactive models and MoE for transmission
strategies. It combines expert knowledge and employs MoE-
PPO for simulation, validating GAI agents and MoE-PPO in
customized problems. Wang et al. [180] proposed an LLM-
powered base station siting (BSS) optimization framework,
overcoming limitations of traditional methods. By optimizing
prompts and using automated agent technology, the framework
improves efficiency, reduces costs, and minimizes manual ef-
fort. Experiments show that LLMs and agents enhance BSS
optimization.
B. Agentic systems based on other GAI models
In addition to LLMs, agent systems based on other GAI
models are also widely applied in the research of communication
systems. For example, Yang et al. [127] proposed an agent-
driven generative semantic communication (A-GSC) framework
based on reinforcement learning to address challenges in remote
monitoring for intelligent transportation systems and digital
twins in 6G. Unlike previous research on semantic extraction,
A-GSC integrates the source information’s intrinsic properties
with task context and introduces GAI for independent design
of semantic encoders and decoders. Chen et al. [181] proposed
a system architecture for AI agents in 6G networks, tackling
challenges in network automation, mobile agents, robotics,autonomous systems, and wearable AI agents. This architecture
enables deep integration of AI agents within 6G networks and
collaboration with application agents. A prototype validated
their capabilities, highlighting three key challenges: energy
efficiency, security, and AI agent-customized communication,
laying the foundation for AI agents in 6G.
C. Summary and lessons learned
1) Summary: This chapter summarizes the research and
applications of intelligent agentic systems based on LLMs and
other GAI models in communication [182]. By leveraging the
powerful data analysis and processing capabilities of these
technologies, agentic systems can more effectively address the
challenges faced by the current communication system, thereby
enabling more efficient information transmission [183].
2) Lessons learned: From the chapter, we have learned
several important lessons. First, constrained by the compu-
tational capabilities of mobile terminals, LAM-based agentic
systems face challenges such as low computational efficiency
and complex model scheduling in local deployment and collab-
orative execution. Although some studies have introduced model
caching and task partitioning mechanisms to improve resource
utilization, the overall system still struggles to meet the demands
of high concurrency and low latency in modern communication
scenarios [184]. Second, while multi-agent systems can col-
laboratively accomplish complex tasks—such as data retrieval,
planning, and reflection—the lack of domain-specific knowledge
and high-quality communication data limits their reasoning and
decision-making performance in advanced tasks such as those
in 6G semantic communications.
IX. LAM S FOR EMERGING APPLICATIONS
The combination of LAMs with the emerging applications is
the driving technological innovation in multiple industries and
fields. These LAMs use their large data sets and deep learning
capabilities to provide strong support for applications such
as smart healthcare, carbon emissions, digital twin, artificial
intelligence of things (AIoT), integrated satellite, aerial, and
terrestrial networks (ISATN), and integration of UA Vs and
LLMs. In the following, we introduce LAMs for these emerging
applications in detail.
A. Smart healthcare
Smart healthcare uses these advanced technologies to im-
prove the efficiency and quality of medical services. Through
data-driven decision support systems, medical institutions can
achieve accurate diagnosis and personalized treatment, thereby
meeting the needs of patients more effectively. In the smart
healthcare, through LAMs and combined with digital twin
technology, we always pay attention to the patient’s physical
condition and provide personalized medical care for patients.
The openCHA proposed by Abbasian et al. [185] provided users
with personalized services in medical consultation. openCHA
is an open source framework based on LLM, which aims
to provide users with personalized smart healthcare services.

25
The openCHA framework overcomes the limitations of the
existing LLM in healthcare, including lack of personalization,
multimodal data processing, and real-time knowledge updating,
by integrating external data sources, knowledge bases, and AI
analysis models.
B. Carbon emissions
In controlling carbon emissions, Wen et al. [186] proposed a
GAI-based low-carbon AIoT solution to reduce carbon emis-
sions from energy consumption in communication networks
and computation-intensive tasks. GAI, using GANs, RAG, and
GDMs, optimizes resource allocation, reduces energy waste,
and improves efficiency. The paper explores GAI applications
in energy internet (EI), data center networks, and mobile
edge networks. In EI, GAI optimizes renewable energy use;
in data centers, it improves the management of information
and communication technology (ICT) equipment and cooling
systems; and in mobile edge networks, GAI, combined with IRS
deployment and semantic communication technologies, reduces
power consumption. The findings show GAI’s superiority in
carbon emission optimization, supporting low-carbon AIoT and
sustainability goals.
C. Digital twins
The application of LAMs in digital twins is a key force in
promoting the development of this technology. For example,
Xia et al. [187] proposed a framework integrating LLMs, digital
twins, and industrial automation systems for intelligent planning
and control of production processes. LLM-based agents interpret
descriptive information in the digital twin and control the
physical system via service interfaces. These agents function
as intelligent agents across all levels of the automation sys-
tem, enabling autonomous planning and control of flexible
production processes. Hong et al. [188] proposed an LLM-
based digital twin network (DTN) framework, LLM-Twin, to
improve communication and multimodal data processing in
DTNs. They introduced a digital twin semantic network (DTSN)
for efficient communication and computation and a small-to-
giant model collaboration scheme for efficient LLM deployment
and multimodal data processing. A native security strategy was
also designed to maintain security without sacrificing efficiency.
Numerical experiments and case studies validated LLM-Twin’s
feasibility.
D. Artificial intelligence of things
In AIoT, Cui et al. [189] proposed the LLMind framework,
showing how combining LLMs with domain-specific AI mod-
ules enhances IoT device intelligence and collaboration. It auto-
mates tasks and enables cooperation through high-level language
instructions. A key feature is the language-to-code mechanism
that converts natural language into finite state machine (FSM)
representations for device control scripts, optimizing task exe-
cution. With an experience accumulation mechanism, LLMind
improves responsiveness and supports efficient collaboration
in dynamic environments, highlighting its potential in IoT
intelligent control.E. Integrated satellite, aerial, and terrestrial networks
Javaid et al. [190] explored the potential of incorporating
LLMs into ISATNs. ISATNs combine various communication
technologies to achieve seamless cross-platform coverage. The
study demonstrates that LLMs, with their advanced AI and ma-
chine learning capabilities, can play a crucial role in data stream
optimization, signal processing, and network management, par-
ticularly in 5G/6G networks. The research not only provided a
comprehensive analysis of ISATN architecture and components
but also discussed in details how LLMs can address bottlenecks
in traditional data transmission and processing. Additionally,
the paper focused on challenges related to resource allocation,
traffic routing, and security management within ISATN network
management, highlighting technical difficulties in data integra-
tion, scalability, and latency. The study concludes with a series
of future research directions aiming at further exploring LLM
applications to enhance network reliability and performance,
thus advancing the development of global intelligent networks.
F . Integration of UAVs and LLMs
Regarding the integration of UA Vs and LLMs, Javaid et al.
[191] conducted a systematic analysis of the current state and
future directions of combining LLMs with UA Vs. The study
thoroughly examined the role of LLMs in enhancing UA V
autonomy and communication capabilities, particularly in key
areas such as spectrum sensing, data processing, and decision-
making. By integrating LLMs, UA Vs can achieve a higher
level of intelligence in complex tasks, including autonomous
responses and real-time data processing. The authors evaluated
the existing LLM architectures, focusing on their contributions
to improving UA V autonomous decision-making, especially in
scenarios like disaster response and emergency communication
restoration. Additionally, the paper highlighted the technical
challenges faced in future research, emphasizing the importance
of further exploring legal, regulatory, and ethical issues to ensure
the effective and sustainable integration of LLM and UA V
technologies.
G. Summary and lessons learned
1) Summary: This chapter highlights the role of LAMs
in emerging applications. In smart healthcare, LAMs enable
personalized care and efficient diagnostics through frameworks
like openCHA. For carbon emissions, LAM-enabled optimiza-
tion frameworks address environmental challenges, holding sig-
nificant value in achieving sustainability and carbon neutral-
ity goals. In digital twins, LAMs significantly advance their
development in industrial automation and other domains by
enhancing intelligent perception, communication [192], and
control capabilities. In AIoT, LAMs enhance device collabo-
ration, task execution, and user interaction [193]. Additionally,
LAMs contribute to networking technologies like ISATNs and
UA Vs by improving resource allocation, decision-making, and
communication. These applications illustrate LAMs’ growing
influence in addressing complex challenges across various fields.

26
2) Lessons learned: From the chapter, we have learned
several important lessons. One primary issue is the insufficiency
of data quality and diversity, which hampers the generalization
ability of LAMs across different domains. For instance, in smart
healthcare, while LAMs can enhance the accuracy of personal-
ized medicine, data privacy constraints often limit data sharing,
potentially introducing biases into the model. In carbon emission
optimization and AIoT scenarios, LAMs rely heavily on high-
quality real-time data, and issues such as data incompleteness
or latency can negatively affect optimization outcomes. Fur-
thermore, security and privacy concerns are critical. In digital
twin applications, decisions made by LAMs can directly impact
the operation of physical systems, and any data tampering or
model attacks could lead to severe consequences. This risk
is particularly pronounced in integrated applications involving
ISATNs and UA Vs with LAMs [194], where cybersecurity
vulnerabilities could be exploited maliciously, resulting in data
breaches or communication disruptions.
X. R ESEARCH CHALLENGES
Although the LAMs have great application potential for
communications, they still face many challenges. This section
mainly introduces some research challenges and potential solu-
tions of LAMs in communication.
1) Lack of high-quality communication data: In the applica-
tion of cutting-edge technologies such as 6G and the IoE, data
acquisition and diversity pose significant challenges. This issue
is particularly critical in core tasks such as wireless communica-
tion, interference mitigation, and spectrum management, where
the lack of high-quality labeled data constrains the training
efficacy of LAMs. First, the cost of data collection is high,
especially in complex network environments where substantial
investments in hardware and sensors are required, increasing
both equipment expenditures and long-term maintenance com-
plexity. Second, data privacy and ethical concerns have become
increasingly prominent, with stringent privacy regulations im-
posing strict constraints on data collection, thereby complicat-
ing the acquisition of effective datasets. Lastly, the scarcity
of labeled data presents a major limitation, particularly for
high-precision tasks, as obtaining labeled data requires domain
expertise and expensive equipment. Moreover, the dynamic
nature of communication environments makes it difficult to
comprehensively cover all conditions, ultimately restricting the
generalization capability of models. The data scarcity issue in
communication hinders the application of LAMs. To address this
challenge, techniques such as data augmentation, self-supervised
learning, and GANs can be employed to expand dataset size,
improve training efficiency, and reduce reliance on high-quality
labeled data. These approaches enable LAMs to better adapt to
dynamic communication scenarios.
2) Lack of structured communication knowledge: LAMs
struggle to solve complex communication problems due to their
limited understanding of communication theory, protocols, and
standards. As LAMs primarily rely on data-driven learning,
their decision-making is often based solely on statistical pat-
terns extracted from training data, neglecting the structuredknowledge inherent in communication. For instance, factors
such as signal attenuation, interference, and noise directly
impact communication system design. However, LAMs find
it challenging to embed these complex structured knowledge
elements, particularly in tasks such as interference cancellation,
spectrum allocation, and channel modeling. This limitation often
results in an inability to accurately capture physical constraints,
ultimately affecting overall system performance. To overcome
the challenge of lacking structured communication knowledge,
researchers can integrate LAMs with communication principles
through physics-informed networks and construct structured
communication knowledge using knowledge graphs. By com-
bining domain-specific expertise with the reasoning capabilities
of LAMs, these approaches enhance model performance in
complex communication scenarios.
3) Generative hallucination in communication: Hallucina-
tion in LAMs has emerged as a significant challenge in commu-
nication. This phenomenon can be categorized into two major
types: factual hallucination, where the model generates incorrect
content that deviates from the correct results, and faithfulness
hallucination, where the model fails to follow user instructions
accurately, producing irrelevant or inconsistent responses. The
root cause of these hallucinations lies in the model’s data-
driven training process, which lacks a deep understanding of
communication system principles. As a result, inaccurate deci-
sions may arise in tasks such as signal quality prediction and
network optimization, severely degrading network performance
and user experience. To address this issue, several strategies can
be employed to enhance the accuracy and stability of model
outputs. These include incorporating the physical constraints
of communication systems, leveraging traditional optimization
methods to assist model outputs, employing ensemble decision-
making across multiple models to improve output consistency,
and designing specialized hallucination detection and mitigation
algorithms. By ensuring that the outputs align with the objective
principles of communication systems, these approaches enhance
the LAM’s reliability and applicability in real-world communi-
cation scenarios.
4) Limitations of reasoning ability: LAMs in communication
systems primarily rely on data-driven pattern recognition and
prediction. However, when faced with communication tasks
requiring high levels of abstraction and multi-step reasoning,
they often struggle to accurately comprehend complex logical
relationships, leading to unreliable decision-making. In scenar-
ios such as wireless channel modeling, spectrum allocation, and
interference management, LAMs must infer multiple interde-
pendent physical parameters and network factors to make well-
informed decisions. Without deep reasoning capabilities, LAMs
may fail to properly account for these intricate dependencies, re-
sulting in outputs that contradict the physical principles govern-
ing real-world communication systems. To address the reasoning
limitations of LAMs in handling complex communication prob-
lems, techniques such as tree-of-thought reasoning, graph-based
reasoning, and long-chain reasoning can be employed. These
approaches leverage hierarchical structured information, multi-
step inference, and process-level reward functions to enhance

27
logical reasoning, improve decision accuracy, and increase
model adaptability. By integrating these advanced reasoning
mechanisms, LAMs can become more efficient and precise in
tackling complex communication tasks.
5) Poor explainability in LAMs: The black-box nature of
LAMs in communication presents a critical challenge due to
their poor explainability. The internal mechanisms and decision-
making processes of these models are often opaque, making
it difficult to trace decisions in tasks such as fault diagnosis,
system optimization, and network management, thereby increas-
ing the complexity of troubleshooting. Additionally, the lack of
explainability raises ethical and legal concerns, particularly in
areas involving user privacy and network security. To address
this issue, explainable AI (XAI) techniques can be employed
to enhance the transparency and trustworthiness of LAMs.
Methods such as local interpretable model-agnostic explanations
(LIME) and shapley additive explanations (SHAP) can help
users understand the rationale behind model decisions. Further-
more, visualizing the model’s decision-making process through
graphical representations can provide insights into reasoning
pathways. These solutions not only improve explainability but
also enable a transparent and traceable decision-making process
for communication systems, enhancing both trust and opera-
tional reliability.
6) Adaptability in dynamic environments: Due to the dy-
namic variations in network topology, channel conditions, and
user demands, communication systems face significant chal-
lenges in optimization and management, making rapid adap-
tation and real-time decision-making crucial. While LAMs
demonstrate strong performance in static environments, their
adaptability in dynamic scenarios often becomes a bottleneck
in practical applications. In tasks such as wireless channel
estimation, resource scheduling, and interference cancellation,
LAMs must swiftly respond to environmental changes to ensure
accurate and timely predictions. If the model fails to adjust its
generative capabilities in response to evolving network condi-
tions and user requirements, it may lead to delayed or inaccurate
predictions, thereby degrading system performance. To address
this issue, techniques such as online learning, continual learning,
multi-task learning, and meta-learning offer effective solutions.
These approaches enable LAMs to dynamically optimize param-
eters, adapt in real-time, and leverage knowledge transfer across
tasks, thereby enhancing their reasoning ability, adaptability, and
robustness in dynamic communication environments.
7) Diversity of communication tasks: The field of communi-
cations encompasses a wide range of highly specialized tasks,
including signal processing, network optimization, interference
mitigation, and spectrum management. These tasks differ sig-
nificantly in terms of objectives, constraints, and optimization
strategies, and are often intricately interrelated. Although LAMs
exhibit strengths in multi-task learning, their lack of domain-
specific knowledge, variations in optimization requirements, and
inconsistencies across tasks make it challenging to adapt to
the diverse nature of communication tasks. For instance, signal
processing demands a deep understanding of modulation and de-
modulation techniques, while network optimization focuses onbandwidth allocation and traffic control. Thus, designing model
architectures that can flexibly accommodate different commu-
nication tasks remains a major challenge. Approaches such as
task-specific models, MoE, and transfer learning have shown
promise in enhancing the performance of LAMs in this context.
Task-specific models allocate dedicated sub-models to different
tasks to minimize interference and improve effectiveness; MoE
dynamically selects expert models tailored to specific tasks,
boosting multi-task learning efficiency; and transfer learning
facilitates knowledge transfer, improving the adaptability and
generalization of LAMs. These methods collectively enhance
the adaptability, efficiency, and accuracy of LAMs in multi-
task environments, thereby strengthening their performance and
reliability across diverse telecommunication tasks.
8) Resource constraints at the edge: In mobile devices, edge
computing platforms, and IoT devices, hardware resources are
typically limited and cannot meet the high computational and
energy demands of LAMs. These devices—especially nodes and
terminals at the edge of 6G networks—are expected to operate
under low-power and resource-constrained conditions, yet their
processing capabilities, memory, and energy efficiency fall short
of what LAMs require. Direct deployment of LAMs at the
edge often results in performance degradation, increased latency,
and compromised communication quality and user experience.
To improve the efficiency of LAMs on devices with limited
computation, storage, and power, several strategies can be
employed: model distillation transfers knowledge from LAMs to
smaller ones to enhance adaptability; model compression tech-
niques such as pruning and quantization reduce computational
and memory overhead; and hardware acceleration leverages
specialized hardware like GPUs, TPUs, and FPGAs to speed up
inference while lowering power consumption. These approaches
effectively enhance the inference efficiency and performance of
LAMs in edge and IoT scenarios.
9) High inference latency: In wireless communications, low
latency and high throughput are critical, especially for real-time
applications such as autonomous driving and remote healthcare.
However, due to their large-scale architectures and complex
computational demands, LAMs often suffer from high inference
latency, which can lead to delayed system responses, reduced
throughput, instability in mission-critical tasks, and inefficient
resource utilization. As communication systems grow increas-
ingly complex, a key challenge is to reduce inference latency
while maintaining model accuracy. To address the issue of high
inference latency, several optimization techniques can be ap-
plied. Operator fusion reduces memory access and data transfer
delays by combining multiple operations, thereby improving
computational efficiency. Speculative sampling accelerates in-
ference by predicting future steps in advance, reducing com-
putational overhead. These methods effectively lower latency,
enhance response time, and improve resource utilization, ensur-
ing that LAMs can meet the stringent performance requirements
of next-generation communication systems.
10) Security and privacy: In 6G networks, the use of LAMs
for data processing introduces significant security and privacy
risks. Since LAMs are often pre-trained in a centralized manner,

28
they are highly susceptible to data breaches, potentially allow-
ing attackers to reconstruct sensitive information. Additionally,
data transmissions are vulnerable to man-in-the-middle attacks,
eavesdropping, and tampering. Moreover, LAMs themselves
may be exposed to adversarial attacks, leading to incorrect
predictions and decisions that can compromise network stability.
With increasingly stringent data privacy regulations, LAMs must
comply with privacy protection requirements to mitigate legal
risks and maintain user trust. To address these challenges,
researchers have proposed several solutions. Federated learn-
ing enables model training on local devices, minimizing the
transmission and storage of sensitive data, thereby reducing
data exposure risks. Encrypted computing techniques, such as
homomorphic encryption and secure multi-party computation,
ensure data security even in untrusted environments. These
approaches help mitigate security threats associated with large
models, enhancing model reliability and user trust, thereby
fostering the deep integration of LAMs with next-generation
communication technologies.
XI. C ONCLUSION
This paper provides a comprehensive review of the devel-
opment, key technologies, application scenarios, and research
challenges of LAMs in communication. It systematically sum-
marizes the critical roles and potential of LAMs, ranging from
fundamental theories to practical applications, particularly in
the era of 6G, when the demand for efficient, stable, and
intelligent communication systems is growing. First, the pa-
per delves into the foundational aspects of LAMs, including
model architectures, classification of different types of LAMs,
training paradigms, evaluation methodologies, and optimization
mechanisms in communication. Second, it presents a detailed
overview of recent research progress on applying LAMs across
various scenarios. The paper systematically analyzes the adapt-
ability and technical advantages of different LAMs in diverse
application scenarios, supported by extensive case studies and
discussions on cutting-edge developments. Finally, this paper
conducts an in-depth analysis of the key challenges currently
faced by LAMs in the communication domain. These chal-
lenges include the lack of high-quality communication data, the
absence of structured domain knowledge, and the occurrence
of generative hallucinations during communication tasks. In
addition, limitations such as inadequate reasoning capabilities,
poor interpretability, weak adaptability to dynamic environ-
ments, and the increased modeling complexity introduced by
task diversity further hinder the development of LAMs in com-
munication. Practical deployment is also constrained by limited
computational resources at the edge, high inference latency, and
critical issues related to data security and privacy protection. It
further proposes potential solutions to address these challenges.
Through these efforts, LAMs are expected to enable more
intelligent, efficient, and secure services, thereby driving the
advancement of 6G and future communication networks.REFERENCES
[1] K. B. Letaief, Y . Shi, J. Lu, and J. Lu, “Edge artificial intelligence for
6g: Vision, enabling technologies, and applications,” IEEE Journal on
Selected Areas in Communications , vol. 40, no. 1, pp. 5–36, Jan. 2022.
[2] W. Jiang, B. Han, M. A. Habibi, and H. D. Schotten, “The road towards
6g: A comprehensive survey,” IEEE Open Journal of the Communications
Society , vol. 2, pp. 334–366, Feb. 2021.
[3] Z. Zhang, Y . Xiao, Z. Ma, M. Xiao, Z. Ding, X. Lei, G. K. Karagiannidis,
and P. Fan, “6g wireless networks: Vision, requirements, architecture, and
key technologies,” IEEE Vehicular Technology Magazine , vol. 14, no. 3,
pp. 28–41, Sep. 2019.
[4] C. Pan, G. Zhou, K. Zhi, S. Hong, T. Wu, Y . Pan, H. Ren, M. D. Renzo,
A. Lee Swindlehurst, R. Zhang, and A. Y . Zhang, “An overview of signal
processing techniques for ris/irs-aided wireless systems,” IEEE Journal
of Selected Topics in Signal Processing , vol. 16, no. 5, pp. 883–917, Aug.
2022.
[5] M. Giordani and M. Zorzi, “Non-terrestrial networks in the 6g era:
Challenges and opportunities,” IEEE Network , vol. 35, no. 2, pp. 244–
251, Apr. 2021.
[6] H.-J. Song and T. Nagatsuma, “Present and future of terahertz communi-
cations,” IEEE Transactions on Terahertz Science and Technology , vol. 1,
no. 1, pp. 256–263, Sep. 2011.
[7] F. Liu, Y . Cui, C. Masouros, J. Xu, T. X. Han, Y . C. Eldar, and S. Buzzi,
“Integrated sensing and communications: Toward dual-functional wire-
less networks for 6g and beyond,” IEEE Journal on Selected Areas in
Communications , vol. 40, no. 6, pp. 1728–1767, Jun. 2022.
[8] S. Mihai, M. Yaqoob, D. V . Hung, W. Davis, P. Towakel, M. Raza,
M. Karamanoglu, B. Barn, D. Shetve, R. V . Prasad, H. Venkataraman,
R. Trestian, and H. X. Nguyen, “Digital twins: A survey on enabling
technologies, challenges, trends and future prospects,” IEEE Communi-
cations Surveys & Tutorials , vol. 24, no. 4, pp. 2255–2291, Sep. 2022.
[9] Y . Wang, Z. Su, N. Zhang, R. Xing, D. Liu, T. H. Luan, and X. Shen,
“A survey on metaverse: Fundamentals, security, and privacy,” IEEE
Communications Surveys & Tutorials , vol. 25, no. 1, pp. 319–352, Sep.
2023.
[10] D. Cozzolino, B. Da Lio, D. Bacco, and L. K. Oxenløwe, “High-
dimensional quantum communication: benefits, progress, and future
challenges,” Advanced Quantum Technologies , vol. 2, no. 12, p. 1900038,
Dec. 2019.
[11] M. Z. Chowdhury, M. Shahjalal, S. Ahmed, and Y . M. Jang, “6g
wireless communication systems: Applications, requirements, technolo-
gies, challenges, and research directions,” IEEE Open Journal of the
Communications Society , vol. 1, pp. 957–975, Jul. 2020.
[12] C. Zhang, P. Patras, and H. Haddadi, “Deep learning in mobile and wire-
less networking: A survey,” IEEE Communications Surveys & Tutorials ,
vol. 21, no. 3, pp. 2224–2287, Mar. 2019.
[13] M. Yu, X. Xiong, Z. Li, and X. Xia, “Multi-head dnn-based federated
learning for rsrp prediction in 6g wireless communication,” IEEE Access ,
vol. 12, pp. 97 533–97 543, Jul. 2024.
[14] M. Xu, S. Zhang, C. Zhong, J. Ma, and O. A. Dobre, “Ordinary
differential equation-based cnn for channel extrapolation over ris-assisted
communication,” IEEE Communications Letters , vol. 25, no. 6, pp. 1921–
1925, Mar. 2021.
[15] Z. Zhou, L. Liu, S. Jere, J. Zhang, and Y . Yi, “Rcnet: Incorporating struc-
tural information into deep rnn for online mimo-ofdm symbol detection
with limited training,” IEEE Transactions on Wireless Communications ,
vol. 20, no. 6, pp. 3524–3537, Jan. 2021.
[16] N. C. Luong, D. T. Hoang, S. Gong, D. Niyato, P. Wang, Y .-C.
Liang, and D. I. Kim, “Applications of deep reinforcement learning
in communications and networking: A survey,” IEEE Communications
Surveys & Tutorials , vol. 21, no. 4, pp. 3133–3174, May. 2019.
[17] T. Shui, J. Hu, K. Yang, H. Kang, H. Rui, and B. Wang, “Cell-free
networking for integrated data and energy transfer: Digital twin based
double parameterized dqn for energy sustainability,” IEEE Transactions
on Wireless Communications , vol. 22, no. 11, pp. 8035–8049, Nov. 2023.
[18] J. Wang, Y . Wang, P. Cheng, K. Yu, and W. Xiang, “Ddpg-based joint
resource management for latency minimization in noma-mec networks,”
IEEE Communications Letters , vol. 27, no. 7, pp. 1814–1818, Jul. 2023.
[19] Y . Huang, M. Li, F. R. Yu, P. Si, and Y . Zhang, “Performance optimiza-
tion for energy-efficient industrial internet of things based on ambient
backscatter communication: An a3c-fl approach,” IEEE Transactions on
Green Communications and Networking , vol. 7, no. 3, pp. 1121–1134,
Sep. 2023.

29
[20] A. Karapantelakis, A. Nikou, A. Kattepur, J. Martins, L. Mokrushin,
S. K. Mohalik, M. Orlic, and A. V . Feljan, “A survey on the integration
of generative ai for critical thinking in mobile networks,” Apr. 2024,
arXiv:2404.06946 .
[21] C. Zhao, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, X. Shen,
and K. B. Letaief, “Generative ai for secure physical layer communica-
tions: A survey,” IEEE Transactions on Cognitive Communications and
Networking , vol. 11, no. 1, pp. 3–26, Feb. 2025.
[22] Y . Zhou, K. Cao, L. Cai, and Y . Feng, “Conditional gan empowered
channel estimation and pilot design for reconfigurable intelligent surfaces
assisted symbiotic radio system,” in Journal of Physics: Conference
Series , vol. 2477, no. 1, Hangzhou, Nov. 2022, p. 012110.
[23] E. Grassucci, S. Barbarossa, and D. Comminiello, “Generative semantic
communication: Diffusion models beyond bit recovery,” Jun. 2023,
arXiv:2306.04321 .
[24] Q. Zhou, R. Li, Z. Zhao, C. Peng, and H. Zhang, “Semantic communica-
tion with adaptive universal transformer,” IEEE Wireless Communications
Letters , vol. 11, no. 3, pp. 453–457, Mar. 2022.
[25] L. Manduchi, K. Pandey, R. Bamler, R. Cotterell, S. D ¨aubener, S. Fellenz,
A. Fischer, T. G ¨artner, M. Kirchler, M. Kloft et al. , “On the challenges
and opportunities in generative ai,” Mar. 2025, arXiv:2403.00025 .
[26] Y . Wang, Q. Hu, Z. Su, L. Du, and Q. Xu, “Large model empowered
metaverse: State-of-the-art, challenges and opportunities,” Jan. 2025,
arXiv:2502.10397 .
[27] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al. , “Gpt-4
technical report,” Mar. 2023, arXiv:2303.08774 .
[28] Y . Liu, K. Zhang, Y . Li, Z. Yan, C. Gao, R. Chen, Z. Yuan, Y . Huang,
H. Sun, J. Gao et al. , “Sora: A review on background, technol-
ogy, limitations, and opportunities of large vision models,” Apr. 2024,
arXiv:2402.17177 .
[29] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro, F. Azhar et al. ,
“Llama: Open and efficient foundation language models,” Feb. 2023,
arXiv:2302.13971 .
[30] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut,
J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al. , “Gemini: a family
of highly capable multimodal models,” Jun. 2024, arXiv:2312.11805v4 .
[31] Y . Wang, Q. Guo, X. Ni, C. Shi, L. Liu, H. Jiang, and Y . Yang,
“Hint-enhanced in-context learning wakes large language models up
for knowledge-intensive tasks,” in IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP) , Seoul, Apr. 2024,
pp. 10 276–10 280.
[32] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le,
D. Zhou et al. , “Chain-of-thought prompting elicits reasoning in large
language models,” Advances in Neural Information Processing Systems ,
vol. 35, pp. 24 824–24 837, Jan. 2022.
[33] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdh-
ery, and D. Zhou, “Self-consistency improves chain of thought reasoning
in language models,” Mar. 2023, arXiv:2203.11171v4 .
[34] Y . Shen, J. Shao, X. Zhang, Z. Lin, H. Pan, D. Li, J. Zhang, and
K. B. Letaief, “Large language models empowered autonomous edge
ai for connected intelligence,” IEEE Communications Magazine , vol. 62,
no. 10, pp. 140–146, Oct. 2024.
[35] H. Zou, Q. Zhao, Y . Tian, L. Bariah, F. Bader, T. Lestable, and
M. Debbah, “Telecomgpt: A framework to build telecom-specfic large
language models,” Jul. 2024, arXiv:2407.09424 .
[36] H. Dong, W. Xiong, B. Pang, H. Wang, H. Zhao, Y . Zhou, N. Jiang,
D. Sahoo, C. Xiong, and T. Zhang, “Rlhf workflow: From reward
modeling to online rlhf,” Nov. 2024, arXiv:2405.07863 .
[37] F. Jiang, Y . Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,
“Large ai model-based semantic communications,” IEEE Wireless Com-
munications , vol. 31, no. 3, pp. 68–75, Jun. 2024.
[38] R. Zhang, H. Du, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour, P. Zhang,
and D. I. Kim, “Generative ai for space-air-ground integrated networks,”
IEEE Wireless Communications , vol. 31, no. 6, pp. 10–20, Dec. 2024.
[39] L. Dong, F. Jiang, Y . Peng, K. Wang, K. Yang, C. Pan, and R. Schober,
“Lambo: Large ai model empowered edge intelligence,” IEEE Commu-
nications Magazine , vol. 63, no. 4, pp. 88–94, Apr. 2025.
[40] F. Jiang, Y . Peng, L. Dong, K. Wang, K. Yang, C. Pan, D. Niyato, and
O. A. Dobre, “Large language model enhanced multi-agent systems for
6g communications,” IEEE Wireless Communications , vol. 31, no. 6, pp.
48–55, Dec. 2024.[41] Z. Chen, Z. Zhang, and Z. Yang, “Big ai models for 6g wireless net-
works: Opportunities, challenges, and research directions,” IEEE Wireless
Communications , vol. 31, no. 5, pp. 164–172, Oct. 2024.
[42] Y . Huang, H. Du, X. Zhang, D. Niyato, J. Kang, Z. Xiong, S. Wang,
and T. Huang, “Large language models for networking: Applications,
enabling techniques, and challenges,” IEEE Network , vol. 39, no. 1, pp.
235–242, Jan. 2025.
[43] L. Bariah, Q. Zhao, H. Zou, Y . Tian, F. Bader, and M. Debbah,
“Large generative ai models for telecom: The next big thing?” IEEE
Communications Magazine , vol. 62, no. 11, pp. 84–90, Nov. 2024.
[44] H. Zhou, C. Hu, Y . Yuan, Y . Cui, Y . Jin, C. Chen, H. Wu, D. Yuan,
L. Jiang, D. Wu, X. Liu, C. Zhang, X. Wang, and J. Liu, “Large
language model (llm) for telecommunications: A comprehensive survey
on principles, key techniques, and opportunities,” IEEE Communications
Surveys & Tutorials , pp. 1–1, Sep. 2024.
[45] C. Liang, H. Du, Y . Sun, D. Niyato, J. Kang, D. Zhao, and M. A. Imran,
“Generative ai-driven semantic communication networks: Architecture,
technologies, and applications,” IEEE Transactions on Cognitive Com-
munications and Networking , vol. 11, no. 1, pp. 27–47, Feb. 2025.
[46] M. Xu, H. Du, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han,
A. Jamalipour, D. I. Kim, X. Shen et al. , “Unleashing the power of
edge-cloud generative ai in mobile networks: A survey of aigc services,”
IEEE Communications Surveys & Tutorials , vol. 26, no. 2, pp. 1127–
1170, Jan. 2024.
[47] G. O. Boateng, H. Sami, A. Alagha, H. Elmekki, A. Hammoud, R. Mi-
zouni, A. Mourad, H. Otrok, J. Bentahar, S. Muhaidat et al. , “A survey
on large language models for communication, network, and service
management: Application insights, challenges, and future directions,”
Dec. 2024, arXiv:2412.19823 .
[48] G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, “Mobile edge
intelligence for large language models: A contemporary survey,” IEEE
Communications Surveys & Tutorials , pp. 1–1, Mar. 2025.
[49] A. Celik and A. M. Eltawil, “At the dawn of generative ai era: A tutorial-
cum-survey on new frontiers in 6g wireless intelligence,” IEEE Open
Journal of the Communications Society , vol. 5, pp. 2433–2489, Feb.
2024.
[50] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
A. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in
Neural Information Processing Systems , vol. 30, Dec. 2017.
[51] Y . Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V . Stoyanov, “Roberta: A robustly optimized bert
pretraining approach,” Jul. 2019, arXiv:1907.11692 .
[52] Y . Wang, Z. Gao, D. Zheng, S. Chen, D. G ¨und¨uz, and H. V . Poor,
“Transformer-empowered 6g intelligent networks: From massive mimo
processing to semantic communication,” IEEE Wireless Communications ,
vol. 30, no. 6, pp. 127–135, Dec. 2023.
[53] H. Yoo, T. Jung, L. Dai, S. Kim, and C.-B. Chae, “Demo: Real-time
semantic communications with a vision transformer,” in IEEE Inter-
national Conference on Communications Workshops (ICC Workshops) ,
Seoul, Mar. 2022, pp. 1–2.
[54] H. Wu, Y . Shao, E. Ozfatura, K. Mikolajczyk, and D. G ¨und¨uz,
“Transformer-aided wireless image transmission with channel feedback,”
IEEE Transactions on Wireless Communications , vol. 23, no. 9, pp.
11 904–11 919, Sep. 2024.
[55] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli,
“Deep unsupervised learning using nonequilibrium thermodynamics,” in
International conference on machine learning , Lille, Jul. 2015, pp. 2256–
2265.
[56] F. Jiang, Y . Peng, L. Dong, K. Wang, K. Yang, C. Pan, and X. You,
“Large generative model assisted 3d semantic communication,” Mar.
2024, arXiv:2403.05783 .
[57] H. Du, R. Zhang, Y . Liu, J. Wang, Y . Lin, Z. Li, D. Niyato, J. Kang,
Z. Xiong, S. Cui, B. Ai, H. Zhou, and D. I. Kim, “Enhancing deep rein-
forcement learning: A tutorial on generative diffusion models in network
optimization,” IEEE Communications Surveys & Tutorials , vol. 26, no. 4,
pp. 2611–2646, Mar. 2024.
[58] T. Wu, Z. Chen, D. He, L. Qian, Y . Xu, M. Tao, and W. Zhang, “Cddm:
Channel denoising diffusion models for wireless communications,” in
GLOBECOM 2023-2023 IEEE Global Communications Conference ,
Kuala Lumpur, Dec. 2023, pp. 7429–7434.
[59] Y . Duan, T. Wu, Z. Chen, and M. Tao, “Dm-mimo: Diffusion models
for robust semantic communications over mimo channels,” in IEEE/CIC
International Conference on Communications (ICCC) , Hangzhou, Aug.
2024, pp. 1609–1614.

30
[60] G. Chi, Z. Yang, C. Wu, J. Xu, Y . Gao, Y . Liu, and T. X. Han,
“Rf-diffusion: Radio signal generation via time-frequency diffusion,” in
Proceedings of the 30th Annual International Conference on Mobile
Computing and Networking , Washington, Nov. 2024, pp. 77–92.
[61] E. Grassucci, C. Marinoni, A. Rodriguez, and D. Comminiello, “Dif-
fusion models for audio semantic communication,” in ICASSP 2024-
2024 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP) , Seoul, Apr. 2024, pp. 13 136–13 140.
[62] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with
selective state spaces,” May. 2024, arXiv:2312.00752v2 .
[63] T. Wu, Z. Chen, M. Tao, Y . Sun, X. Xu, W. Zhang, and P. Zhang,
“Mambajscc: Adaptive deep joint source-channel coding with generalized
state space model,” Sep. 2024, arXiv:2409.16592v1 .
[64] D. Yuan, J. Xue, J. Su, W. Xu, and H. Zhou, “St-mamba: Spatial-
temporal mamba for traffic flow estimation recovery using limited data,”
inIEEE/CIC International Conference on Communications (ICCC) ,
Hangzhou, Aug. 2024, pp. 1928–1933.
[65] L. Yu, H. Zhang, J. Liu, C. Liu, J. Yuan, Z. Li, and Z. Wang, “Vimsc: Ro-
bust underwater acoustic image semantic communication based on vision
mamba model,” in Proceedings of the 12th International Conference on
Communications and Broadband Networking , Nyingchi, Jul. 2024, pp.
46–52.
[66] K. Carolan, L. Fennelly, and A. F. Smeaton, “A review of multi-modal
large language and vision models,” Mar. 2024, arXiv:2404.01322 .
[67] J. Wu, S. Yang, R. Zhan, Y . Yuan, L. S. Chao, and D. F. Wong, “A
survey on llm-generated text detection: Necessity, methods, and future
directions,” Computational Linguistics , pp. 1–66, Apr. 2025.
[68] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al. , “Improving
language understanding by generative pre-training,” OpenAI blog , Jun.
2018.
[69] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al. ,
“Language models are unsupervised multitask learners,” OpenAI blog ,
vol. 1, no. 8, p. 9, May. 2019.
[70] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al. , “Language models
are few-shot learners,” Advances in Neural Information Processing
Systems , vol. 33, pp. 1877–1901, Dec. 2020.
[71] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju,
S. Pathak, L. Sifre, M. Rivi `ere, M. S. Kale, J. Love et al. , “Gemma:
Open models based on gemini research and technology,” Apr. 2024,
arXiv:2403.08295v4 .
[72] G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju,
L. Hussenot, T. Mesnard, B. Shahriari, A. Ram ´eet al. , “Gemma
2: Improving open language models at a practical size,” Oct. 2024,
arXiv:2408.00118v3 .
[73] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y . Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al. , “Llama 2: Open
foundation and fine-tuned chat models,” Jul. 2023, arXiv:2307.09288v2 .
[74] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fan et al. , “The llama 3 herd of
models,” Nov. 2024, arXiv:2407.21783v3 .
[75] T. Yang, P. Zhang, M. Zheng, Y . Shi, L. Jing, J. Huang, and N. Li,
“Wirelessgpt: A generative pre-trained multi-task learning framework for
wireless communication,” Feb. 2025, arXiv:2502.06877 .
[76] F. Jiang, L. Dong, Y . Peng, K. Wang, K. Yang, C. Pan, and X. You,
“Large ai model empowered multimodal semantic communications,”
IEEE Communications Magazine , vol. 63, no. 1, pp. 76–82, Jan. 2025.
[77] P. Jiang, C.-K. Wen, X. Yi, X. Li, S. Jin, and J. Zhang, “Semantic
communications using foundation models: Design approaches and open
issues,” IEEE Wireless Communications , vol. 31, no. 3, pp. 76–84, Jun.
2024.
[78] Y . Wang, Z. Sun, J. Fan, and H. Ma, “On the uses of large language
models to design end-to-end learning semantic communication,” in IEEE
Wireless Communications and Networking Conference (WCNC) , Dubai,
Apr. 2024, pp. 1–6.
[79] M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han, D. I. Kim, and
K. B. Letaief, “When large language model agents meet 6g networks:
Perception, grounding, and alignment,” IEEE Wireless Communications ,
vol. 31, no. 6, pp. 63–71, Aug. 2024.
[80] C. Zhang, Y . Cui, W. Lin, G. Huang, Y . Rong, L. Liu, and S. Shan,
“Segment anything for videos: A systematic survey,” Jul. 2024,
arXiv:2408.08315 .
[81] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision
tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine
Intelligence , vol. 46, no. 8, pp. 5625–5644, Aug. 2024.[82] Y . Zhang, K. Gong, K. Zhang, H. Li, Y . Qiao, W. Ouyang, and X. Yue,
“Meta-transformer: A unified framework for multimodal learning,” Jul.
2023, arXiv:2307.10802 .
[83] S. Gao, J. Yang, L. Chen, K. Chitta, Y . Qiu, A. Geiger, J. Zhang, and
H. Li, “Vista: A generalizable driving world model with high fidelity and
versatile controllability,” Oct. 2024, arXiv:2405.17398v5 .
[84] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y . Lo et al. , “Segment anything,”
inProceedings of the IEEE/CVF International Conference on Computer
Vision , Paris, Oct 2023, pp. 4015–4026.
[85] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and
A. Joulin, “Emerging properties in self-supervised vision transformers,”
inProceedings of the IEEE/CVF international conference on computer
vision , Montreal, Oct. 2021, pp. 9650–9660.
[86] M. Oquab, T. Darcet, T. Moutakanni, H. V o, M. Szafraniec, V . Khali-
dov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al. , “Di-
nov2: Learning robust visual features without supervision,” Feb. 2024,
arXiv:2304.07193v2 .
[87] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-
resolution image synthesis with latent diffusion models,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition ,
New Orleans, Jun. 2022, pp. 10 684–10 695.
[88] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini, Y . Levi,
D. Lorenz, A. Sauer, F. Boesel et al. , “Scaling rectified flow transformers
for high-resolution image synthesis,” in Forty-first International Confer-
ence on Machine Learning , Austria, Jul. 2024.
[89] S. Tariq, B. E. Arfeto, C. Zhang, and H. Shin, “Segment anything meets
semantic communication,” Jun. 2023, arXiv:2306.02094 .
[90] J. Wu, W. Gan, Z. Chen, S. Wan, and S. Y . Philip, “Multimodal large
language models: A survey,” in IEEE International Conference on Big
Data (BigData) , Sorrento, Dec. 2023, pp. 2247–2256.
[91] Z. Tang, Z. Yang, C. Zhu, M. Zeng, and M. Bansal, “Any-to-any
generation via composable diffusion,” Advances in Neural Information
Processing Systems , vol. 36, pp. 16 083–16 099, Dec. 2023.
[92] Z. Tang, Z. Yang, M. Khademi, Y . Liu, C. Zhu, and M. Bansal,
“Codi-2: In-context interleaved and interactive any-to-any generation,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , Seattle, Jun. 2024, pp. 27 425–27 434.
[93] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V . Alwala, A. Joulin,
and I. Misra, “Imagebind: One embedding space to bind them all,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , Vancouver, Jun. 2023, pp. 15 180–15 190.
[94] F. Jiang, W. Zhu, L. Dong, K. Wang, K. Yang, C. Pan, and O. A. Dobre,
“Commgpt: A graph and retrieval-augmented multimodal communication
foundation model,” Feb. 2025, arXiv:2502.18763 .
[95] L. Qiao, M. B. Mashhadi, Z. Gao, C. H. Foh, P. Xiao, and M. Bennis,
“Latency-aware generative semantic communications with pre-trained
diffusion models,” IEEE Wireless Communications Letters , vol. 13,
no. 10, pp. 2652–2656, Oct. 2024.
[96] Y . Wang, J. He, L. Fan, H. Li, Y . Chen, and Z. Zhang, “Driving into the
future: Multiview visual forecasting and planning with world model for
autonomous driving,” in IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , Seattle, Jun. 2024, pp. 14 749–14 759.
[97] W. Cai, T. Wang, J. Wang, and C. Sun, “Learning a world model with
multitimescale memory augmentation,” IEEE Transactions on Neural
Networks and Learning Systems , vol. 34, no. 11, pp. 8493–8502, Mar.
2023.
[98] Z. Gao, Y . Mu, C. Chen, J. Duan, P. Luo, Y . Lu, and S. Eben Li, “Enhance
sample efficiency and robustness of end-to-end urban autonomous driving
via semantic masked world model,” IEEE Transactions on Intelligent
Transportation Systems , vol. 25, no. 10, pp. 13 067–13 079, Oct. 2024.
[99] Y . LeCun, “A path towards autonomous machine intelligence version 0.9.
2, 2022-06-27,” Open Review , vol. 62, no. 1, pp. 1–62, Jun. 2022.
[100] W. Saad, O. Hashash, C. K. Thomas, C. Chaccour, M. Debbah, N. Man-
dayam, and Z. Han, “Artificial general intelligence (agi)-native wireless
systems: A journey beyond 6g,” Proceedings of the IEEE , pp. 1–39, Mar.
2025.
[101] B. Liu, X. Liu, S. Gao, X. Cheng, and L. Yang, “Llm4cp: Adapting large
language models for channel prediction,” Journal of Communications and
Information Networks , vol. 9, no. 2, pp. 113–125, Jun. 2024.
[102] A. S. M. Mohammed, A. I. A. Taman, A. M. Hassan, and A. Zekry,
“Deep learning channel estimation for ofdm 5g systems with different
channel models,” Wireless Personal Communications , vol. 128, no. 4, pp.
2891–2912, Feb. 2023.

31
[103] S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper,
Z. Liu, S. Prabhumoye, G. Zerveas, V . Korthikanti et al. , “Using
deepspeed and megatron to train megatron-turing nlg 530b, a large-scale
generative language model,” Feb. 2022, arXiv:2201.11990v3 .
[104] J. Li and X. Yang, “A cyclical learning rate method in deep learning
training,” in International Conference on Computer, Information and
Telecommunication Systems (CITS) , Hangzhou, Oct. 2020, pp. 1–5.
[105] J. Zhang, T. He, S. Sra, and A. Jadbabaie, “Why gradient clipping
accelerates training: A theoretical justification for adaptivity,” Feb. 2020,
arXiv:1905.11881v2 .
[106] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
and W. Chen, “Lora: Low-rank adaptation of large language models,” in
International Conference on Learning Representations(ICLR) , On-line,
Apr. 2022, pp. 2790–2799.
[107] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe,
A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer
learning for nlp,” in International conference on machine learning , Long
Beach, Jun. 2019, pp. 2790–2799.
[108] E. B. Zaken, S. Ravfogel, and Y . Goldberg, “Bitfit: Simple parameter-
efficient fine-tuning for transformer-based masked language-models,”
Sep. 2022, arXiv:2106.10199v5 .
[109] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts
for generation,” Jan. 2021, arXiv:2101.00190 .
[110] H. Lee, S. Phatale, H. Mansoor, T. Mesnard, J. Ferret, K. Lu, C. Bishop,
E. Hall, V . Carbune, A. Rastogi et al. , “Rlaif vs. rlhf: Scaling rein-
forcement learning from human feedback with ai feedback,” Sep. 2024,
arXiv:2309.00267 .
[111] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Prox-
imal policy optimization algorithms,” Aug. 2017, arXiv:1707.06347 .
[112] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and
C. Finn, “Direct preference optimization: Your language model is secretly
a reward model,” Advances in Neural Information Processing Systems ,
vol. 36, Jul. 2024.
[113] F. Jiang, L. Dong, S. Tu, Y . Peng, K. Wang, K. Yang, C. Pan, and
D. Niyato, “Personalized wireless federated learning for large language
models,” Apr. 2024, arXiv:2404.13238 .
[114] A. Maatouk, F. Ayed, N. Piovesan, A. De Domenico, M. Debbah, and Z.-
Q. Luo, “Teleqna: A benchmark dataset to assess large language models
telecommunications knowledge,” Oct. 2023, arXiv:2310.15051 .
[115] Z. Guo, R. Jin, C. Liu, Y . Huang, D. Shi, L. Yu, Y . Liu, J. Li, B. Xiong,
D. Xiong et al. , “Evaluating large language models: A comprehensive
survey,” Nov. 2023, arXiv:2310.19736 .
[116] Q. Tang, Z. Deng, H. Lin, X. Han, Q. Liang, B. Cao, and L. Sun,
“Toolalpaca: Generalized tool learning for language models with 3000
simulated cases,” Sep. 2023, arXiv:2306.05301 .
[117] Y . Du, H. Deng, S. C. Liew, K. Chen, Y . Shao, and H. Chen, “The power
of large language models for wireless communication system develop-
ment: A case study on fpga platforms,” Jul. 2023, arXiv:2307.07319 .
[118] H. Zou, Q. Zhao, L. Bariah, Y . Tian, M. Bennis, S. Lasaulce, M. Debbah,
and F. Bader, “Genainet: Enabling wireless collective intelligence via
knowledge transfer and reasoning,” Feb. 2024, arXiv:2402.16631 .
[119] J. Shao, J. Tong, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang, “Wireless-
llm: Empowering large language models towards wireless intelligence,”
May. 2024, arXiv:2405.17053 .
[120] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel et al. , “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
Neural Information Processing Systems , vol. 33, pp. 9459–9474, May.
2020.
[121] A.-L. Bornea, F. Ayed, A. De Domenico, N. Piovesan, and A. Maatouk,
“Telco-rag: Navigating the challenges of retrieval-augmented language
models for telecommunications,” Apr. 2024, arXiv:2404.15939 .
[122] Y . Tang and W. Guo, “Automatic retrieval-augmented generation of 6g
network specifications for use cases,” IEEE Communications Magazine ,
vol. 63, no. 4, pp. 95–102, Dec. 2025.
[123] X. Huang, Y . Tang, J. Li, N. Zhang, and X. Shen, “Toward effective
retrieval augmented generative services in 6g networks,” IEEE Network ,
vol. 38, no. 6, pp. 459–467, Aug. 2024.
[124] S. Xu, C. Kurisummoottil Thomas, O. Hashash, N. Muralidhar, W. Saad,
and N. Ramakrishnan, “Large multi-modal models (lmms) as univer-
sal foundation models for ai-native wireless systems,” IEEE Network ,
vol. 38, no. 5, pp. 10–20, Jul. 2024.
[125] G. Y . GMY , J. A. Ayala-Romero, A. Garcia-Saavedra, and X. Costa-
Perez, “Telecomrag: Taming telecom standards with retrieval augmented
generation and llms,” Authorea Preprints , Jun. 2024.[126] J. Tong, J. Shao, Q. Wu, W. Guo, Z. Li, Z. Lin, and J. Zhang,
“Wirelessagent: Large language model agents for intelligent wireless
networks,” Sep. 2024, arXiv:2409.07964 .
[127] W. Yang, Z. Xiong, Y . Yuan, W. Jiang, T. Q. S. Quek, and M. Debbah,
“Agent-driven generative semantic communication with cross-modality
and prediction,” IEEE Transactions on Wireless Communications , vol. 24,
no. 3, pp. 2233–2248, Mar. 2025.
[128] P. Xu, W. Shao, K. Zhang, P. Gao, S. Liu, M. Lei, F. Meng, S. Huang,
Y . Qiao, and P. Luo, “Lvlm-ehub: A comprehensive evaluation bench-
mark for large vision-language models,” IEEE Transactions on Pattern
Analysis and Machine Intelligence , vol. 47, no. 3, pp. 1877–1893, Mar.
2025.
[129] B. Yin and N. Hu, “Time-cot for enhancing time reasoning factual
question answering in large language models,” in International Joint
Conference on Neural Networks (IJCNN) , Yokohama, Jun. 2024, pp. 1–8.
[130] M. Shao, A. Basit, R. Karri, and M. Shafique, “Survey of different
large language model architectures: Trends, benchmarks, and challenges,”
IEEE Access , vol. 12, pp. 188 664–188 706, Oct. 2024.
[131] S. Fan, Z. Liu, X. Gu, and H. Li, “Csi-llm: A novel downlink
channel prediction method aligned with llm pre-training,” Aug. 2024,
arXiv:2409.00005 .
[132] Y . Sheng, K. Huang, L. Liang, P. Liu, S. Jin, and G. Y . Li, “Beam pre-
diction based on large language models,” Aug. 2024, arXiv:2408.08707 .
[133] M. Akrout, A. Mezghani, E. Hossain, F. Bellili, and R. W. Heath, “From
multilayer perceptron to gpt: A reflection on deep learning research for
wireless physical layer,” IEEE Communications Magazine , vol. 62, no. 7,
pp. 34–41, Jul. 2024.
[134] Z. Xiao, C. Ye, Y . Hu, H. Yuan, Y . Huang, Y . Feng, L. Cai, and J. Chang,
“Llm agents as 6g orchestrator: A paradigm for task-oriented physical-
layer automation,” Sep. 2024, arXiv:2410.03688 .
[135] Z. Wang, J. Zhang, H. Du, R. Zhang, D. Niyato, B. Ai, and K. B. Letaief,
“Generative ai agent for next-generation mimo design: Fundamentals,
challenges, and vision,” Apr. 2024, arXiv:2404.08878 .
[136] T. Zheng and L. Dai, “Large language model enabled multi-task physical
layer network,” Mar. 2024, arXiv:2412.20772v2 .
[137] J.-H. Lee, D.-H. Lee, J. Lee, and J. Pujara, “Integrating pre-trained
language model with physical layer communications,” IEEE Transactions
on Wireless Communications , vol. 23, no. 11, pp. 17 266–17 278, Nov.
2024.
[138] H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao,
“Diffusion-based reinforcement learning for edge-enabled ai-generated
content services,” IEEE Transactions on Mobile Computing , vol. 23,
no. 9, pp. 8902–8918, Jan. 2024.
[139] H. Du, G. Liu, Y . Lin, D. Niyato, J. Kang, Z. Xiong, and D. I.
Kim, “Mixture of experts for intelligent networks: A large language
model-enabled approach,” in International Wireless Communications and
Mobile Computing (IWCMC) , Ayia Napa, May. 2024, pp. 531–536.
[140] R. Zhang, H. Du, Y . Liu, D. Niyato, J. Kang, Z. Xiong, A. Jamalipour,
and D. In Kim, “Generative ai agents with large language model for
satellite networks via a mixture of experts transmission,” IEEE Journal
on Selected Areas in Communications , vol. 42, no. 12, pp. 3581–3596,
Sep. 2024.
[141] X. Du and X. Fang, “An integrated communication and computing
scheme for wi-fi networks based on generative ai and reinforcement
learning,” Apr. 2024, arXiv:2404.13598 .
[142] M. Xu, D. Niyato, J. Kang, Z. Xiong, S. Guo, Y . Fang, and D. I.
Kim, “Generative ai-enabled mobile tactical multimedia networks: Dis-
tribution, generation, and perception,” IEEE Communications Magazine ,
vol. 62, no. 10, pp. 96–102, Oct. 2024.
[143] H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, D. I. Kim et al. , “Enabling ai-
generated content (aigc) services in wireless edge networks,” Jan. 2023,
arXiv:2301.03220 .
[144] C. Liu and J. Zhao, “Resource allocation in large language model
integrated 6g vehicular networks,” in IEEE 99th Vehicular Technology
Conference (VTC2024-Spring) , Singapore, Jun. 2024, pp. 1–6.
[145] H. Noh, B. Shim, and H. J. Yang, “Adaptive resource allocation optimiza-
tion using large language models in dynamic wireless environments,” Feb.
2025, arXiv:2502.02287 .
[146] J. Zheng, B. Du, H. Du, J. Kang, D. Niyato, and H. Zhang, “Energy-
efficient resource allocation in generative ai-aided secure semantic mobile
networks,” IEEE Transactions on Mobile Computing , vol. 23, no. 12, pp.
11 422–11 435, Dec. 2024.
[147] H. Zou, Q. Zhao, L. Bariah, M. Bennis, and M. Debbah, “Wireless
multi-agent generative ai: From connected intelligence to collective
intelligence,” Jul. 2023, arXiv:2307.02757 .

32
[148] J. Wang, L. Zhang, Y . Yang, Z. Zhuang, Q. Qi, H. Sun, L. Lu, J. Feng, and
J. Liao, “Network meets chatgpt: Intent autonomous management, control
and operation,” Journal of Communications and Information Networks ,
vol. 8, no. 3, pp. 239–255, Sep. 2023.
[149] A. Dandoush, V . Kumarskandpriya, M. Uddin, and U. Khalil, “Large
language models meet network slicing management and orchestration,”
Mar. 2024, arXiv:2403.13721 .
[150] L. Yue and T. Chen, “Ai large model and 6g network,” in IEEE Globecom
Workshops (GC Wkshps) , Kuala Lumpur, Dec. 2023, pp. 2049–2054.
[151] S. Long, F. Tang, Y . Li, T. Tan, Z. Jin, M. Zhao, and N. Kato, “6g
comprehensive intelligence: network operations and optimization based
on large language models,” IEEE Network , pp. 1–1, Sep. 2024.
[152] X. Chen, W. Wu, Z. Li, L. Li, and F. Ji, “Llm-empowered iot
for 6g networks: Architecture, challenges, and solutions,” Mar. 2025,
arXiv:2503.13819 .
[153] Z. Yu, Z. Wang, Y . Li, R. Gao, X. Zhou, S. R. Bommu, Y . Zhao, and
Y . Lin, “Edge-llm: Enabling efficient large language model adaptation
on edge devices via unified compression and adaptive layer voting,” in
Proceedings of the 61st ACM/IEEE Design Automation Conference , San
Francisco, Jun. 2024, pp. 1–6.
[154] M. Zhang, X. Shen, J. Cao, Z. Cui, and S. Jiang, “Edgeshard: Efficient
llm inference via collaborative edge computing,” IEEE Internet of Things
Journal , pp. 1–1, Dec. 2024.
[155] M. Xu, Y . Wu, D. Cai, X. Li, and S. Wang, “Federated fine-tuning
of billion-sized language models across mobile devices,” Jan. 2024,
arXiv:2308.13894 .
[156] W. Zhao, W. Jing, Z. Lu, and X. Wen, “Edge and terminal coop-
eration enabled llm deployment optimization in wireless network,” in
IEEE/CIC International Conference on Communications (ICCC Work-
shops) , Hangzhou, Aug. 2024, pp. 220–225.
[157] A. Khoshsirat, G. Perin, and M. Rossi, “Decentralized llm inference over
edge networks with energy harvesting,” Aug. 2024, arXiv:2408.15907 .
[158] Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, “Pushing large
language models to the 6g edge: Vision, challenges, and opportunities,”
Mar. 2023, arXiv:2309.16739 .
[159] Y . Rong, Y . Mao, X. He, and M. Chen, “Large-scale traffic flow forecast
with lightweight llm in edge intelligence,” IEEE Internet of Things
Magazine , vol. 8, no. 1, pp. 12–18, Nov. 2025.
[160] O. Friha, M. Amine Ferrag, B. Kantarci, B. Cakmak, A. Ozgun, and
N. Ghoualmi-Zine, “Llm-based edge intelligence: A comprehensive
survey on architectures, applications, security and trustworthiness,” IEEE
Open Journal of the Communications Society , vol. 5, pp. 5799–5856, Sep.
2024.
[161] B. Lai, J. Wen, J. Kang, H. Du, J. Nie, C. Yi, D. I. Kim, and S. Xie,
“Resource-efficient generative mobile edge networks in 6g era: Funda-
mentals, framework and case study,” IEEE Wireless Communications ,
vol. 31, no. 4, pp. 66–74, Aug. 2024.
[162] M. Xu, D. Cai, Y . Wu, X. Li, and S. Wang, “Fwdllm: Efficient fedllm
using forward gradient,” Jan. 2024, arXiv:2308.13894 .
[163] Y . Peng, F. Jiang, L. Dong, K. Wang, and K. Yang, “Personalized
federated learning for generative ai-assisted semantic communications,”
Oct. 2024, arXiv:2410.02450 .
[164] Y . He, J. Fang, F. R. Yu, and V . C. Leung, “Large language models (llms)
inference offloading and resource allocation in cloud-edge computing: An
active inference approach,” IEEE Transactions on Mobile Computing ,
vol. 23, no. 12, pp. 11 253–11 264, Dec. 2024.
[165] J. Zhang, H. Yang, A. Li, X. Guo, P. Wang, H. Wang, Y . Chen,
and H. Li, “Mllm-llava-fl: Multimodal large language model assisted
federated learning,” in IEEE/CVF Winter Conference on Applications of
Computer Vision (WACV) , Tucson, Feb. 2025, pp. 4066–4076.
[166] J. Fang, Y . He, F. R. Yu, J. Li, and V . C. Leung, “Large language
models (llms) inference offloading and resource allocation in cloud-
edge networks: An active inference approach,” in IEEE 98th Vehicular
Technology Conference (VTC2023-Fall) , Hong Kong, Oct. 2025, pp. 1–5.
[167] Z. Wang, L. Zou, S. Wei, F. Liao, J. Zhuo, H. Mi, and R. Lai, “Large
language model enabled semantic communication systems,” Jul. 2024,
arXiv:2407.14112 .
[168] F. Jiang, S. Tu, L. Dong, C. Pan, J. Wang, and X. You, “Large generative
model-assisted talking-face semantic communication system,” Nov. 2024,
arXiv:2411.03876 .
[169] W. Chen, W. Xu, H. Chen, X. Zhang, Z. Qin, Y . Zhang, and Z. Han,
“Semantic communication based on large language model for underwater
image transmission,” Aug. 2024, arXiv:2408.12616 .
[170] A. Kalita, “Large language models (llms) for semantic communication
in edge-based iot networks,” Jul. 2024, arXiv:2407.20970 .[171] F. Jiang, S. Tu, L. Dong, K. Wang, K. Yang, R. Liu, C. Pan, and J. Wang,
“Lightweight vision model-based multi-user semantic communication
systems,” Feb. 2025, arXiv:2502.16424 .
[172] F. Jiang, S. Tu, L. Dong, K. Wang, K. Yang, and C. Pan, “M4sc: An mllm-
based multi-modal, multi-task and multi-user semantic communication
system,” Feb. 2025, arXiv:2502.16418 .
[173] F. Jiang, C. Tang, L. Dong, K. Wang, K. Yang, and C. Pan, “Visual
language model based cross-modal semantic communication systems,”
IEEE Transactions on Wireless Communications , pp. 1–1, Mar. 2025.
[174] F. Zhang, Y . Du, K. Chen, Y . Shao, and S. C. Liew, “Addressing out-of-
distribution challenges in image semantic communication systems with
multi-modal large language models,” in 22nd International Symposium
on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks
(WiOpt) , Seoul, Oct. 2024, pp. 7–14.
[175] H. Xie, Z. Qin, X. Tao, and Z. Han, “Toward intelligent communications:
Large model empowered semantic communications,” IEEE Communica-
tions Magazine , vol. 63, no. 1, pp. 69–75, Jul. 2025.
[176] W. Yang, Z. Xiong, S. Mao, T. Q. Quek, P. Zhang, M. Debbah, and
R. Tafazolli, “Rethinking generative semantic communication for multi-
user systems with large language models,” Feb. 2025, arXiv:2408.08765 .
[177] T. S. Do, T. P. Truong, T. Do, H. P. Van, and S. Cho, “Lightweight
multiuser multimodal semantic communication system for multimodal
large language model communication,” Authorea Preprints , Feb. 2024.
[178] X. Zhang, X. He, M. Chen, and L. Wang, “Reinforcement learning from
human-like feedback enhances semantic communication with multimodal
llms,” in 16th International Conference on Wireless Communications and
Signal Processing (WCSP) , Hefei, Oct. 2024, pp. 1491–1496.
[179] D. Cao, J. Wu, and A. K. Bashir, “Multimodal large language models
driven privacy-preserving wireless semantic communication in 6g,” in
IEEE International Conference on Communications Workshops (ICC
Workshops) , Denver, Jun. 2024, pp. 171–176.
[180] Y . Wang, M. M. Afzal, Z. Li, J. Zhou, C. Feng, S. Guo, and T. Q. Quek,
“Large language models for base station siting: Intelligent deployment
based on prompt or agent,” Aug. 2024, arXiv:2408.03631 .
[181] Z. Chen, Q. Sun, N. Li, X. Li, Y . Wang, and Chih-Lin I, “Enabling mobile
ai agent in 6g era: Architecture and key technologies,” IEEE Network ,
vol. 38, no. 5, pp. 66–75, Jul. 2024.
[182] T. Yang, P. Feng, Q. Guo, J. Zhang, X. Zhang, J. Ning, X. Wang,
and Z. Mao, “Autohma-llm: Efficient task coordination and execution
in heterogeneous multi-agent systems using hybrid large language mod-
els,” IEEE Transactions on Cognitive Communications and Networking ,
vol. 11, no. 2, pp. 987–998, Apr. 2025.
[183] H. Fang, D. Zhang, C. Tan, P. Yu, Y . Wang, and W. Li, “Large language
model enhanced autonomous agents for proactive fault-tolerant edge
networks,” in IEEE INFOCOM 2024 - IEEE Conference on Computer
Communications Workshops (INFOCOM WKSHPS) , Vancouver, May.
2024, pp. 1–2.
[184] L. Zhou, X. Deng, Z. Wang, X. Zhang, Y . Dong, X. Hu, Z. Ning, and
J. Wei, “Semantic information extraction and multi-agent communication
optimization based on generative pre-trained transformer,” IEEE Trans-
actions on Cognitive Communications and Networking , vol. 11, no. 2,
pp. 725–737, Apr. 2025.
[185] M. Abbasian, I. Azimi, A. M. Rahmani, and R. Jain, “Conversational
health agents: A personalized llm-powered agent framework,” Sep. 2024,
arXiv:2310.02374 .
[186] J. Wen, R. Zhang, D. Niyato, J. Kang, H. Du, Y . Zhang, and Z. Han,
“Generative ai for low-carbon artificial intelligence of things with large
language models,” Jul. 2024, arXiv:2404.18077 .
[187] Y . Xia, M. Shenoy, N. Jazdi, and M. Weyrich, “Towards autonomous
system: flexible modular production system enhanced with large language
model agents,” in IEEE 28th International Conference on Emerging
Technologies and Factory Automation (ETFA) , Sinaia, Sep. 2023, pp.
1–8.
[188] Y . Hong, J. Wu, and R. Morello, “Llm-twin: mini-giant model-driven
beyond 5g digital twin networking framework with semantic secure
communication and computation,” Scientific Reports , vol. 14, no. 1, p.
19065, Aug. 2024.
[189] H. Cui, Y . Du, Q. Yang, Y . Shao, and S. C. Liew, “Llmind: Orchestrating
ai and iot with llm for complex task execution,” IEEE Communications
Magazine , vol. 63, no. 4, pp. 214–220, Apr. 2025.
[190] S. Javaid, R. A. Khalil, N. Saeed, B. He, and M.-S. Alouini, “Leveraging
large language models for integrated satellite-aerial-terrestrial networks:
Recent advances and future directions,” IEEE Open Journal of the
Communications Society , vol. 6, pp. 399–432, Dec. 2025.

33
[191] S. Javaid, H. Fahim, B. He, and N. Saeed, “Large language models for
uavs: Current state and pathways to the future,” IEEE Open Journal of
Vehicular Technology , vol. 5, pp. 1166–1192, Aug. 2024.
[192] S. Jiang, B. Lin, Y . Wu, and Y . Gao, “Links: Large language model in-
tegrated management for 6g empowered digital twin networks,” in IEEE
100th Vehicular Technology Conference (VTC2024-Fall) , Washington,
Oct. 2024, pp. 1–6.
[193] D. Rivkin, F. Hogan, A. Feriani, A. Konar, A. Sigal, X. Liu, and
G. Dudek, “Aiot smart home via autonomous llm agents,” IEEE Internet
of Things Journal , vol. 12, no. 3, pp. 2458–2472, Feb. 2025.
[194] H. Li, M. Xiao, K. Wang, D. I. Kim, and M. Debbah, “Large language
model based multi-objective optimization for integrated sensing and com-
munications in uav networks,” IEEE Wireless Communications Letters ,
vol. 14, no. 4, pp. 979–983, Apr. 2025.
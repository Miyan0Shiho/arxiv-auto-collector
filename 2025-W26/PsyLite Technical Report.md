# PsyLite Technical Report

**Authors**: Fangjun Ding, Renyu Zhang, Xinyu Feng, Chengye Xie, Zheng Zhang, Yanting Zhang

**Published**: 2025-06-26 17:54:42

**PDF URL**: [http://arxiv.org/pdf/2506.21536v1](http://arxiv.org/pdf/2506.21536v1)

## Abstract
With the rapid development of digital technology, AI-driven psychological
counseling has gradually become an important research direction in the field of
mental health. However, existing models still have deficiencies in dialogue
safety, detailed scenario handling, and lightweight deployment. To address
these issues, this study proposes PsyLite, a lightweight psychological
counseling large language model agent developed based on the base model
InternLM2.5-7B-chat. Through a two-stage training strategy (hybrid distillation
data fine-tuning and ORPO preference optimization), PsyLite enhances the
model's deep-reasoning ability, psychological counseling ability, and safe
dialogue ability. After deployment using Ollama and Open WebUI, a custom
workflow is created with Pipelines. An innovative conditional RAG is designed
to introduce crosstalk humor elements at appropriate times during psychological
counseling to enhance user experience and decline dangerous requests to
strengthen dialogue safety. Evaluations show that PsyLite outperforms the
baseline models in the Chinese general evaluation (CEval), psychological
counseling professional evaluation (CPsyCounE), and dialogue safety evaluation
(SafeDialBench), particularly in psychological counseling professionalism
(CPsyCounE score improvement of 47.6\%) and dialogue safety (\safe{} score
improvement of 2.4\%). Additionally, the model uses quantization technology
(GGUF q4\_k\_m) to achieve low hardware deployment (5GB memory is sufficient
for operation), providing a feasible solution for psychological counseling
applications in resource-constrained environments.

## Full Text


<!-- PDF content starts -->

arXiv:2506.21536v1  [cs.AI]  26 Jun 2025PsyLite Technical Report
Fangjun Ding, Renyu Zhang, Xinyu Feng, Chengye Xie, Zheng Zhang, Yanting Zhang
Donghua University
https://github.com/Jundifang/PsyLite
Abstract
With the rapid development of digital technology, AI-driven psychological counseling has grad-
ually become an important research direction in the field of mental health. However, existing
models still have deficiencies in dialogue safety, detailed scenario handling, and lightweight
deployment. To address these issues, this study proposes PsyLite, a lightweight psychological
counseling large language model agent developed based on the base model InternLM2.5-7B-
chat. Through a two-stage training strategy (hybrid distillation data fine-tuning and ORPO
preference optimization), PsyLite enhances the model’s deep-reasoning ability, psychological
counseling ability, and safe dialogue ability. After deployment using Ollama and Open WebUI,
a custom workflow is created with Pipelines. An innovative conditional RAG is designed to
introduce crosstalk humor elements at appropriate times during psychological counseling to
enhance user experience and decline dangerous requests to strengthen dialogue safety. Evalu-
ations show that PsyLite outperforms the baseline models in the Chinese general evaluation
(CEval), psychological counseling professional evaluation (CPsyCounE), and dialogue safety
evaluation (SafeDialBench), particularly in psychological counseling professionalism (CPsy-
CounE score improvement of 47.6%) and dialogue safety (SafeDialBench score improvement of
2.4%). Additionally, the model uses quantization technology (GGUF q4_k_m) to achieve low
hardware deployment (5GB memory is sufficient for operation), providing a feasible solution
for psychological counseling applications in resource-constrained environments.
Keywords: Lightweight psychological counseling LLM, Deep reasoning, Dialogue safety, Lightweight
deployment, ORPO preference optimization, QLoRA fine-tunning, Conditional Retrieval-Augmented
Generation(Conditional RAG), Cross-talk humor

Contents
1 Introduction 3
2 Related Work 4
2.1 InternLM2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.2 BenchMark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.2.1 CPsyCoun . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.2.2 SafeDialBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2.3 Mental Application in LLM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3.1 DeepPsy-Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3.2 EmoLLM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3.3 SoulChat . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 Infrastructures 6
3.1 Model Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.1.1 Phase 1: Supervised Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.1.2 Phase 2: Reinforcement Learning via Odds Ratio Preference Optimization . . . 9
3.2 Quantified Deployment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.2.1 Model Quantization and Conversion . . . . . . . . . . . . . . . . . . . . . . . 10
3.2.2 Model Distribution and Storage . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.2.3 Ollama Local Deployment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
3.3 Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
3.3.1 Open WebUI-Based User Interaction Platform . . . . . . . . . . . . . . . . . . 11
3.3.2 Pipelines-Driven PsyLite Workflow . . . . . . . . . . . . . . . . . . . . . . . . 11
4 Experiment 13
4.1 Baseline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2.1 CEval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2.2 CPsyCounE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
4.2.3 SafeDialBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
5 Discussion 15
5.1 Why Psychological Counseling Datasets Require Chain-of-Thought (CoT) Injection . . . 15
5.2 Supervised Fine-Tuning Phase: Step-by-Step Training vs. End-to-End Training . . . . . 16
5.3 Failed Attempts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
2

1. Introduction
With the rapid development of digital technology, psychological counseling has gradually become an
indispensable part of modern society. Traditional psychological counseling methods usually rely on
natural language processing (NLP) technology, which, although effective, are often difficult to achieve
wide-scale popularization and timely service due to hardware resource limitations. The development and
progress of artificial intelligence have brought new opportunities to psychological counseling, allowing
users to obtain psychological support and adjustment anytime and anywhere. However, despite the
promising prospects of AI-driven solutions, designing a large-scale psychological counseling model that
can not only think deeply but also ensure the safety of conversations and occasionally provide users with
crosstalk segments to enhance the user experience remains a significant challenge.
PsyLite was developed to address these issues. This project is based on the base model InternLM2.5-
7B-chat and has developed a lightweight psychological counseling large language model application with
low hardware requirements and local deployment capabilities. The underlying model of the PsyLite
application is trained in two stages. In the first stage, a strategy of mixed production distillation
datasets is adopted, combining general domain data with psychological counseling domain data in a
certain proportion and using QLoRA technology for fine-tuning training. This enables the model to not
only learn professional psychological counseling knowledge but also develop the ability to think deeply
and provide high-quality responses to users. This dataset design helps the model avoid overfitting in
specific domain training while maintaining its generalization ability in different tasks. In the second
stage, a preference dataset for dialogue safety is used, and the ORPO reinforcement learning strategy
is employed to enhance the model’s dialogue safety capabilities, significantly reducing the possibility of
the model outputting harmful information. When building the PsyLite agent, different strategies are
adopted for different dialogue scenarios. Besides normal conversations and refusing to answer in extreme
situations, when the user is in a good mood, PsyLite will introduce the humorous elements of crosstalk,
namely, appropriately using crosstalk segments to adjust the conversation atmosphere and enhance the
user experience. This innovative approach provides effective support for users’ emotions and states in
different scenarios, making psychological counseling more relaxed and productive and improving the user
experience. This makes PsyLite not only a psychological counseling assistant, but also an assistant that
knows you.
Our main contributions include:
(a) We developed the deep-reasoning psychological counseling model InternLM2.5-7B-distill and the
enhanced dialogue safety deep-reasoning psychological counseling model InternLM2.5-7B-distill-
orpo. We also built agent PsyLite using InternLM2.5-7B-distill-orpo as base model, providing
multi-scenario psychological counseling support.
(b)PsyLite adopts a training strategy that combines a distillation dataset of psychological counseling
dialogues with general domain knowledge to make a hybrid finetune dataset, avoiding overfitting
during training, maintaining the model’s generalization ability, and enabling the learning of
deep-reasoning capabilities.
(c)PsyLite ingeniously resolves the scenarios conflict between crosstalk and psychological counseling
by integrating the seemingly incompatible humor of crosstalk with serious psychological counseling,
allowing users to receive relatively professional psychological counseling while enjoying the fun of
crosstalk, and promoting the dissemination and continuation of this excellent traditional Chinese
culture.
3

This project utilizes these methods and technologies to promote the application of AI in the field of
mental health, provides psychological counseling model deployment plan in low-hardware environments,
and makes efforts towards a future with more intelligent, safer, and more personalized psychological
counseling services.
2. Related Work
2.1. InternLM2
Since the launch of ChatGPT and GPT-4, large language models have gained widespread popularity
in both academic and industrial fields, and the era of AGI seems to be approaching. To train better-
performing open-source large models, community scholars have been striving to bridge the performance
gap with industry-leading LLMs. In the past year, many outstanding open-source LLMs have emerged,
such as LLaMA(Touvron et al. (2023)), Qwen(Bai et al. (2023)), Mistral(Jiang et al. (2023)), and
Deepseek(DeepSeek-AI et al. (2024)). This project opts to build upon InternLM2(Cai et al. (2024)) for
related work. It is an open-source LLM that outperforms its predecessors in a comprehensive evaluation
across six dimensions and 30 benchmarks, long-context modeling, and open-ended subjective assessment
through innovative pre-training and optimization techniques. The pre-training process of InternLM2 is
highly detailed, highlighting the preparation of various data types, including text, code, and long-context
data. InternLM2 effectively captures long-term dependencies, initially trained with 4k tokens and
then developed to 32k tokens in both pre-training and fine-tuning stages, demonstrating outstanding
performance in the 200k "Needle-in-a-Haystack" test. InternLM2 further maintains consistency through
supervised fine-tuning (SFT) and a novel conditional online reinforcement learning (COOL RLHF)
strategy from human feedback, which addresses conflicting human preferences and reward hacking.
2.2. BenchMark
2.2.1. CPsyCoun
Currently, using large language models (LLMs) for psychological counseling is an important but
challenging task. The industry has made attempts to improve empathetic conversations or utilize
LLMs to play an efficient supporting role in therapy. However, the existing datasets lack counseling
knowledge, preventing LLMs from demonstrating professional counseling capabilities. Moreover, how to
automatically evaluate the multi-round conversations in the counseling process remains an underexplored
area. To fill this gap, CPsyCoun(Zhang et al. (2024)) has emerged, a report-based framework for
reconstructing and evaluating multi-round dialogues in Chinese psychological counseling. To fully
utilize the counseling reports, it designs a two-stage method to construct high-quality dialogues, and
develops a comprehensive evaluation benchmark to effectively automatically evaluate the multi-round
counseling.
2.2.2. SafeDialBench
With the rapid development of large language models, the security of LLMs has reached a critical issue
that requires precise assessment. The current benchmarks mainly focus on single-round conversations
or single attack methods to evaluate security. Moreover, these benchmarks do not consider the ability
of LLMs to precisely identify and handle unsafe information. To address these issues, we adopted a
fine-grained benchmark called SafeDialBench(Cao et al. (2025)), which is used to evaluate the security of
4

LLMs in various jailbreak attacks over multiple rounds of conversations. Specifically, it is a two-layer
hierarchical security classification method that considers 6 security dimensions and generates over 4,000
multi-round Chinese-English conversations in 22 conversation scenarios. We employed 7 jailbreak attack
strategies, such as reference attacks and purpose reverse, to improve the quality of the generated dataset
for dialogue generation. Additionally, an innovative LLM evaluation framework can measure the ability
to detect and handle unsafe information and maintain consistency when facing jailbreak attacks. The
experimental results of 17 LLMs show that Yi-34B-Chat and GLM4-9B Chat demonstrate outstanding
security performance, while Llama3.1-8B-Instruct and o3-mini have mitigated security vulnerabilities.
2.3. Mental Application in LLM
Recently, the success of LLM in the general field has sparked interest in its application in various vertical
fields. And in the field of psychological counseling, many LLMs have achieved remarkable success.
2.3.1. DeepPsy-Agent
An emotion-supporting intelligent agent system that combines the three-stage assistance theory in
psychology with deep learning technology. This system consists of two core components: (1) the multi-
stage response capability dialogue model (deeppsy-chat Chen and Sun (2025)), which enhances reasoning
ability through stage perception and deep-reasoning analysis to generate high-quality responses; (2) the
real-time stage transition detection model, which can identify context changes to guide the dialogue
to be more effective during transitional stages.The innovation of this model lies in its dynamic stage
perception and deep reasoning capabilities. Firstly, the stage perception technology is based on Clara
Hill’s three-stage assistance theory (exploration, insight, action), dynamically adjusting the dialogue
strategy through clear stage markers (such as <Exploration>, <Insight>, <Action>) and real-time
perception of the dialogue context. During the exploration stage, the model prioritizes the generation of
open-ended questions to facilitate the self-expression of the visitor; during the insight stage, the model
combines cognitive behavior theory to generate analogical frameworks and causal reasoning chains to
help the visitor restructure their understanding of the problem; during the action stage, the model guides
the visitor to take practical actions through step-by-step suggestions. This dynamic transformation
mechanism can flexibly adjust strategies according to the progress of the dialogue, avoiding the rigid
processes of traditional systems. Secondly, the deep reasoning ability enables the model to handle complex
causal reasoning and metaphorical associations, for example, by reasoning chains such as "high workload
→sleep deprivation →emotional breakdown" to help the visitor identify the root cause of the problem,
and by integrating deep reasoning paths, providing more personalized and psychologically reasonable
suggestions. Experimental results show that these techniques significantly improve the completeness of
problem exposure, the success rate of cognitive restructuring, and the adoption rate of actions, compared
to traditional models, demonstrating superior dialogue quality and emotional support effects.
2.3.2. EmoLLM
EmoLLM(SmartFlowAI (2024)) is a series of mental health large models that can support the "understanding-
user-support-user-help-user" mental health counseling chain. It uses multi-source data (such as psy-
chological counseling PDF professional books, Smile, CPsyCounD, etc.) for partial annotated data
collection, and combines GPT-4 and other large models for scenario dialogue generation to achieve data
expansion (Dataset-Expanding). The concept of this project is also inspired by the architecture of the
EmoLLM system. EmoLLM builds on multi-source psychological counseling data, combines multi-stage
5

fine-tuning of large language models (such as LoRA, QLoRA, full-parameter training), and realizes
a mental dialogue body with emotional understanding and personalized conversation capabilities. At
the same time, it integrates the Agent-RAG retrieval enhancement mechanism at the application layer
and provides differentiated companionship experiences through multi-role deployment. This complete
closed-loop design from data construction to scenario implementation provides structural reference and
practical inspiration for this research in the construction of lightweight mental Q&A systems.
2.3.3. SoulChat
The SoulChat(Xie et al. (2024)) model integrates multiple core technologies in the field of mental health,
aiming to provide personalized emotional support and cognitive behavioral therapy (CBT). Firstly,
the model has the ability to recognize emotions, accurately identifying the emotional changes of the
visitors and strengthening the emotional connection through empathetic responses. Secondly, the model
employs the core techniques of cognitive behavioral therapy to guide the visitors to identify and challenge
unreasonable beliefs and negative thoughts, helping them recognize the negative impact of these cognitive
biases on emotions and behaviors. Further, the model provides rational thinking training to help the
visitors build more reasonable coping mechanisms and replace irrational thinking patterns. Moreover,
the model encourages the visitors to apply the cognitive skills learned during the treatment to their
daily lives, thereby promoting long-term psychological health improvement. Through these technologies,
SoulChat not only provides immediate emotional support to the visitors but also helps them develop
healthier thinking patterns and behavioral patterns to cope with life’s challenges.
3. Infrastructures
This project uses InternLM2.5-7B-chat as the original model and adopts a two-stage training strategy,
aiming to train a lightweight psychological counseling large language model that can think deeply and
answer safely. The first stage of training is to use the distilled dataset for supervised fine-tuning, aiming
to enable the model to acquire psychological counseling knowledge and the ability to think deeply; the
second stage of training is to use the preference dataset for ORPO reinforcement learning, optimizing the
safety of the model’s conversations.
Figure 1|Illustration of the overall architecture of PsyLite.
6

3.1. Model Training
3.1.1. Phase 1: Supervised Fine-Tuning
In the DeepSeek R1 technical report(DeepSeek-AI et al. (2025)), we recognized the effectiveness of distilled
models in enhancing the capabilities of small models. Therefore, we adopted a straightforward distillation
approach by directly fine-tuning the model using distilled datasets.
Figure 2|Illustration of the construction procedure of InternLM2.5-7B-distill.
A. Dataset Construction We selected two datasets from Hugging Face. One is the deepseek R1
distillation dataset, Chinese-DeepSeek-R1-Distill-data-110k-SFT(Liu et al. (2025)), which we will refer
to as the distillation dataset hereafter. We chose it because it is a dataset based on high-quality and
diverse questions, covering four fields: mathematics, tests, STEM, and the general. The other is the
psychological counseling dialogue dataset, CPsyCounD(Zhang et al. (2024)), which we will refer to as the
psychological counseling dataset hereafter. We chose it because it is a dataset based on real interactions
between professional psychotherapists and clients, providing professional knowledge in mental health
support scenarios. This dataset covers nine aspects: self-growth, emotional stress, education, love and
marriage, family relationships, social relationships, sex, career, and mental illness.
Figure 3|Illustration of the construction procedure of psy-mix-gen-distill-13k.
As shown in the Figure 3, the final dataset used for training, psy-mix-gen-distill-13k, is comprised of
two parts: 10k general distill datasets and 3k psychological distill datasets. To prepare the 10k general
distill datasets, first, the data from the Chinese-DeepSeek-R1-Distill-data-110k-SFT dataset is filtered
based on the field "repo_name" to select the general domain data. Then, the data with a score greater than
or equal to 9 is selected based on the field "score". Finally, 10k data items are randomly sampled from the
remaining data, and that is the 10k general distill datasets we use.
7

To prepare 3k psychological distill datasets, it is necessary to inject the chain of thought (CoT) into
the psychological counseling dataset. Simply provide the last segment of the psychological counseling
data, including the client’s question, the counselor’s response, and the historical dialogue records, to
DeepSeek R1. Let it think step by step about the thought process from the "question" to the "answer"
based on these, and then inject the obtained CoT into the original data to complete the creation of the 3k
psychological distill datasets.
General domain data and psychological counseling data are combined in a ratio of 10:3 to maintain a
balance between general reasoning ability and professional psychological counseling knowledge. This
hybrid approach enables the model to learn deep-thinking skills and knowledge in the field of psychological
counseling while maintaining generalization ability and avoiding overfitting.
B. QLoRA Supervised Fine-Tuning After obtaining high-quality training datasets, we further
conducted supervised fine-tuning (SFT) to enhance the model’s capabilities in psychology and its thinking
ability. We selected InternLM2.5-7B-chat as the base model for training, which is a dialogue model that
performs well in both English and Chinese. After conducting QLoRA(Dettmers et al. (2023)) SFT based
on InternLM2.5-7B-chat, we developed InternLM2.5-7B-distill.
During the training phase, it is assumed that the text input is a sequence 𝑥=𝑥1,𝑥2, ,𝑥𝐿, Each𝑥𝑖in
the text is a token, and 𝐿represents the sequence length. The core architecture of InternLM is based on
the Transformer-Decoder(Vaswani et al. (2023)) structure, which is a typical autoregressive framework
designed to predict the subsequent words in a given sequence in order. Its mathematical representation is:
𝑝(𝑥)=𝐿Ö
𝑡=1𝑝(𝑥𝑡|𝑥1,···,𝑥𝑡−1) (1)
When training InternLM2.5-7B-distill, the cross-entropy loss function is used as the objective
function, and the optimization is carried out by maximizing the negative conditional log-likelihood of the
predicted sequence under the input data conditions:
𝜃∗=arg max
𝜃𝐿∑︁
𝑡=1log𝑝(𝑥𝑡|𝑥1,···,𝑥𝑡−1;𝜃) (2)
where is the trainable parameters of the model.
The hyperparameter configuration is as follows: learning rate 1𝑒−4, batch size 1, training rounds 5,
enabling variable-length attention mechanism, and using 50% of a 40GB GPU with A100 architecture
for QLoRA parameter fine-tuning.
We adopt QLoRA quantization fine-tuning. QLoRA is an efficient fine-tuning technique for large
language models. By combining the two core technologies of quantization and low-rank adaptation, it
significantly reduces hardware resource requirements while maintaining model performance. Its core
ideas include:
1.4-bit quantization compression: Compress the pre-trained model weights to 4-bit precision
(such as NF4 data type), reducing memory usage.
2.Low-rank adaptation layer: Freeze the quantized original weights and only insert trainable
8

low-rank matrices (such as rank-decomposition matrix 𝐴×𝐵), adjusting the model behavior with a
small number of parameters.
3.Efficient resource utilization: Combining dual-weighting (compressing quantization constants)
and page optimizer techniques, it can fine-tune models with billions of parameters on consumer-
grade GPU (such as 24GB memory) while requiring only 1/10 of the memory for full-parameter
fine-tuning.
QLoRA achieves near full-parameter fine-tuning effects in multiple tests (such as the Vicuna bench-
mark reaching 99.3% of ChatGPT’s performance). It is suitable for instruction fine-tuning, multi-task
transfer, etc., and is an ideal choice for fine-tuning large models in resource-constrained environments.
The use of QLoRA in this training significantly reduces the memory requirements of the graphics card.
3.1.2. Phase 2: Reinforcement Learning via Odds Ratio Preference Optimization
Psychological counseling is a very sensitive situation. Therefore, it is necessary to enhance the ability of
safe dialogue for the model, reduce the risk of jail-breaking and the possibility of misleading. Thus, we
adopt ORPO(Hong et al. (2024)), a new type of, high-performance, reinforcement learning algorithm that
does not require training and rewards for directly optimizing the model output. We use the preference
optimization dataset to directly train.
Figure 4|Illustration of the construction procedure of InternLM2.5-7B-distill-orpo.
A. Dataset Construction We selected the open-source preference optimization dataset PKU-SafeRLHF-
single-dimension(PKU-Alignment (2024)) from Hugging Face. We determined the chosen answer based
on the data field "better_response_id", while the other answer was labeled as "rejected". By combining the
"prompt" field, a standard preference optimization dataset could be created, with the format of "prompt",
"chosen", "rejected".
B. Odds Ratio Preference Optimization (ORPO) After obtaining a high-quality preference opti-
mization dataset, we further conducted ORPO training to enhance the dialogue security capabilities of
the large language model. We used InternLM2.5-7B-distill as the main model, and through the training
of ORPO reinforcement learning, we ultimately obtained InternLM2.5-7B-distill-orpo.
During the training process, the objective function of ORPO consists of two parts: the supervised
fine-tuning loss (L𝑆𝐹𝑇) and the relative ratio loss ( L𝑂𝑅).
L𝑂𝑅𝑃𝑂=E(𝑥,𝑦𝑤,𝑦𝑙)[L𝑆𝐹𝑇+𝜆·L𝑂𝑅] (3)
L𝑆𝐹𝑇follows the negative log-likelihood (NLL) loss function of conventional causal language modeling
to maximize the likelihood of generating the reference labels. L𝑂𝑅optimizes the model by maximizing
9

Figure 5|Preview of the PKU-SafeRLHF-orpo-72k from Hugging Face.
the likelihood ratio of the preferred response 𝑦𝑤and the non-preferred response 𝑦𝑙. The log-odds ratio is
wrapped by the log-sigmoid function, allowing L𝑂𝑅to minimize by increasing the log-odds ratio of 𝑦𝑤
and𝑦𝑙.
L𝑂𝑅=−log𝜎
logodds𝜃(𝑦𝑤|𝑥)
odds𝜃(𝑦𝑙|𝑥)
(4)
The two are combined through the weight to jointly adjust the pre-trained language model: to make
it adapt to more secure conversations while suppressing the generated content in the rejected response
set. The hyperparameter configuration is as follows: learning rate 5𝑒−6, =0.2, batch size 1, training
rounds 5, enabling variable-length attention mechanism, and using 50% of a 40GB GPU with A100
architecture for ORPO training.
3.2. Quantified Deployment
3.2.1. Model Quantization and Conversion
Quantization method introduction:
Use the𝑙𝑙𝑎𝑚𝑎 .𝑐𝑝𝑝(ggml.ai (2023)) toolchain to convert the PyTorch model (.bin) into the GGUF format
and apply the q4_k_m quantization strategy. This strategy adopts a mixed-precision method where most
tensors are quantized to 4 bits, while some key tensors retain 6-bit precision, achieving a balance between
model size and inference accuracy.
Conversion operation:
Implement the format conversion locally using the 𝑐𝑜𝑛𝑣𝑒𝑟𝑡 _ℎ𝑓_𝑡𝑜_𝑔𝑔𝑢𝑓 .𝑝𝑦script, or simplify the conver-
sion process directly using the gguf-my-repo online service provided by Hugging Face Spaces, ultimately
obtaining InternLM2.5-7B-distill-Q4_K_M-GGUF and InternLM2.5-7B-distill-orpo-Q4_K_M-GGUF
3.2.2. Model Distribution and Storage
Upload the quantized GGUF model to the Hugging Face Hub (e.g., InternLM2.5-7B-distill-orpo),
for version management and community sharing. Compared to the original PyTorch model, q4_k_m
10

quantization can reduce approximately 70% of storage space while maintaining over 90% of task
performance.
3.2.3. Ollama Local Deployment
Load the GGUF model through the 𝑂𝑙𝑙𝑎𝑚𝑎 framework(ollama (2023)). The GGUF format supports
memory mapping (mmap) loading, significantly reducing memory usage and improving inference speed.
Models that previously could only run on GPU with at least 14G of VRAM (excluding KV cache) can
now even be run on local CPUs with only 5G of memory.
3.3. Agent
3.3.1. Open WebUI-Based User Interaction Platform
We have chosen to use Open WebUI(Open-WebUI (2023)) as the interactive platform for deploying the
model. Open WebUI is an expandable, feature-rich, and user-friendly self-hosted AI platform designed
to run completely offline. It supports various LLM running programs, such as Ollama and OpenAI-
compatible APIs, and can be further combined with Pipelines to build custom model workflows, which
perfectly meets our project requirements.
3.3.2. Pipelines-Driven PsyLite Workflow
Pipelines(Open-WebUI (2024)) is a plugin framework for Open WebUI that supports OpenAI API.
It brings modular and customizable workflows to any UI client that complies with the OpenAI API
specification.
Figure 6|Illustration of Pipelines workflow for PsyLite.
The Figure 7 is the Pipelines workflow of PsyLite. The core function can be summarized as "Condi-
11

tional RAG". The purpose is to provide crosstalk clips that are suitable for the current context and match
the user’s current state during the psychological counseling process, in order to enhance the atmosphere,
shorten the distance between the user and the counselor and improve the user experience, etc. When the
user sends content involving dangerous remarks (such as violence, self-harm, suicidal tendencies, etc.),
illegal requests (such as jail-breaking, hacking attacks, manufacturing prohibited items, etc.) or other
content that may endanger their own or others’ safety ，the model refuses to answer and suggests that
the user seek professional help.
The specific process of Pipelines is as follows:
1.Before the user request/message enters the LLM model, it will be processed in the Inlet Filter.
Here, by calling the large model, combined with the historical chat records and the current user
request/message, the real-time assessment of the user’s current state is conducted.
2.After the Inlet Filter processing, it enters the Pipe process. At this time, based on the assessment
result of the user’s state, conditional processing is carried out.
(a)State①: The user’s current state is very dangerous (mental illness, dangerous remarks, illegal
requests, etc.). The model will refuse to answer and directly output the system-preset phrases,
suggesting that the user seek professional help. In a gentle but firm way, the dangerous
conversation is blocked, maintaining respect for the user while clearly defining the safety
boundary.
(b)State②: The user’s state is normal or difficult to determine. Then the model will reply
normally.
(c)State③: The user is currently in a pleasant mood or suitable for adding humorous comedy
sentences to enhance the user experience. At this time, in addition to the large model (for
PsyLite, here use InternLM2.5-7B-distill-orpo) replying to the user’s request normally, a
separate word vector model will be used to conduct RAG technology retrieval of the comedy
script library to generate a comedy segment that conforms to the current context.
3.After the LLM model generates the reply and sends it to the user, the Outlet Filter can process the
model’s output. Only when the user’s state is state ③(c), the Outlet Filter will combine the model
reply and the comedy segment, and combine the Markdown folding block function to allow the user
to expand and view the comedy segment when they want to, and fold it when they don’t want to,
providing excellent user experience, as shown in the following Figure 7:
Figure 7|Illustration of the actual usage case.
This strategy resolves the conflict between the "crosstalk and psychological counseling" scenarios.
12

It ingeniously combines the seemingly incompatible humorous crosstalk with serious psychological
counseling in a way that allows users to receive relatively professional and lightweight psychological
counseling while also experiencing the fun brought by crosstalk. Moreover, it enables the dissemination
of this excellent traditional Chinese culture of crosstalk.
4. Experiment
To comprehensively evaluate the performance of the base model InternLM2.5-7B-distill-orpo of the project
PsyLite, we conducted three evaluation tasks: the CEval(Nguyen et al. (2024)) Chinese General Multi-
task Capability Evaluation Set, the CPsyCounE Multi-Dimensional Psychological Counseling Dialogue
Ability Evaluation, and the SafeDialBench Security Test. All experiments were completed with 50%
A100 GPU. The comparison objects include:
1. Baseline model: InternLM2.5-7B-chat
2. Evaluation model: InternLM2.5-7B-distill-orpo
We used a unified prompt format and task descriptions to generate model outputs for each evaluation
task separately, and scored them according to the evaluation set or supplemented with human scoring.
4.1. Baseline
In this experiment, our model training was conducted based on the Xtuner(Laboratory (2023b)) training
framework, following a multi-stage optimization process using the InternLM2.5-7B-chat as the base
model. The evaluation work of CEval was completed using OpenCompass(Laboratory (2023a)) while the
one of CPsyCounE and SafeDialBench are completed using the evaluation codes provided by their repo.
To assess the performance improvement of our model in Chinese general tasks, psychological counseling
tasks, and safety dialogue tasks, we used InternLM2.5-7B-chat as the baseline model.
4.2. Evaluation
To evaluate the performance of the PsyLite model proposed in this project in the context of psychological
counseling tasks, we designed a series of evaluation tasks, covering three major dimensions: general
multi-task generalization, professional nature of psychological conversations, and security in open
scenarios. The evaluation system comprehensively utilized the Common Evaluation (CEval) benchmark
for general capabilities, the Professional Psychological Conversation Evaluation Set (CPsyCounE) for
professional psychological conversations, and the Security Stress Test Set (SafeDialBench) for security
testing, in order to conduct a comprehensive analysis of the model’s usability and robustness in real-world
application scenarios.
4.2.1. CEval
CEval is a general-purpose multi-task model for the Chinese environment, covering a variety of multiple-
choice tasks across multiple domains. In this experiment, neither the baseline model nor the evaluation
model used system prompts; they only received task instructions through prompts. The final results are
shown in Table 1:
Although our model scored slightly lower than the baseline model on CEval, considering that its
optimization direction was for the psychological counseling scenario, the slight sacrifice in performance
13

Table 1|Model Performance on CEval Benchmark
Model Score
ours 76.56
InternLM2.5-7b-chat 78.07
for the general domain is within an acceptable range. On the contrary, its ability to maintain a similar
generalization capability to the base model is exactly what we expected.
4.2.2. CPsyCounE
We used the CPsyCounE test set and manually scored the quality of psychological counseling from four
dimensions: comprehensiveness (0 - 2), professionalism (0 - 3), authenticity (0 - 3), and safety (0 - 1).
The system prompt words for this experiment are shown in the following Figure 8:
Figure 8|Prompt template and example.
The evaluation results evaluated on the CPsyCounE benchmark are shown in Table 2:
Table 2|Model Performance on CPsyCounE Benchmark
ModelComprehensiveness
(0–2 points)Professionalism
(0–3 points)Authenticity
(0–3 points)Safety
(0–1 point)
ours 1.97 2.83 2.72 1.00
InternLM2.5-7b-chat 1.76 1.89 2.52 1.00
The detailed scores for nine psychological topics were shown in Table 4 in appendix part. The results
showed that our model outperformed the baseline model in almost all categories, especially excelling in
"psychological disorders", "career development", and "family relationships". This clearly demonstrates
its improvement in professionalism and content structure.
4.2.3. SafeDialBench
In the SafeDialBench evaluation, we tested 249 samples of jailbreak attacks, covering 6 different themes
of complex conversation scenarios.
14

Figure 9|The composition of Evaluation datasets.
In this experiment, neither the baseline model nor the model to be evaluated had any system prompts.
They only received task instructions. The final score results(10 maximum) shown in Table 3 were provided
based on the automatic scoring by Deepseek R1 and the manual review:
Table 3|Model Performance on SafeDialBench Benchmark
Model Average Score
ours 8.93
InternLM2.5-7b-chat 8.72
The our model(InternLM2.5-7B-distill-orpo) shows greater robustness in jailbreak attacks, generating
content more reliably, and having a more flexible rejection mechanism. It effectively avoids outputting
harmful information while maintaining the coherence of the context.
The experimental results indicate:
•InternLM2.5-7B-distill-orpo outperforms the baseline model InternLM2.5-7B-chat significantly in
the specific scenarios of psychological counseling, demonstrating stronger capabilities in psycholog-
ical counseling
•It performs more stably in security tasks and can effectively identify and avoid potential risky
conversations
•Although there is a slight performance decline in general tasks, it still maintains an excellent level
overall, proving that its general multi-task generalization ability remains robust
5. Discussion
5.1. Why Psychological Counseling Datasets Require Chain-of-Thought (CoT) Injection
Our goal is to train a large model that can think deeply and be equipped with knowledge of psychological
counseling.
1.If the model is first fine-tuned using the distilled dataset (to acquire the ability to think), and
15

then fine-tuned using the psychological counseling fine-tuning dataset (to acquire the ability of
psychological counseling), although the psychological counseling ability is achieved, due to the
latter dataset not containing thought chains, the model’s thinking ability will deteriorate.
2.If the model is first fine-tuned using the psychological counseling fine-tuning dataset (to acquire
the ability of psychological counseling), and then fine-tuned using the distilled dataset (to acquire
the ability to think), although the reasoning ability is achieved, due to the latter dataset being
general domain knowledge, the model’s psychological counseling knowledge will deteriorate.
5.2. Supervised Fine-Tuning Phase: Step-by-Step Training vs. End-to-End Training
When training models for specific domains, a common strategy is to use step-by-step training, such as
first training a general model and then fine-tuning it for specific domains. It seems that this strategy
should also be adopted for the psychological counseling large model in this project. However, why did we
choose to combine the datasets of the two steps and adopt a one-step training approach? There are two
main reasons:
1. One-step training is more time-saving and cost-efficient than two-step training.
2.We believe that a qualified psychological counselor not only needs professional and solid psycho-
logical knowledge but also basic general knowledge to adapt to different application scenarios.
Training the model with the psychological counseling dataset enables the model to learn how to
use psychological counseling techniques, but it cannot provide the basic knowledge behind the
techniques, so it is a very important task to maintain the model’s generalization ability during the
learning of psychological counseling knowledge.
Based on the evaluation results of CEval and CPsyCounE, training the psychological counseling
distilled dataset and the general distilled dataset in a 3:10ratio can enable the model to maintain
its general ability while improving its psychological counseling ability. In the field of psychological
counseling, such training is effective.
5.3. Failed Attempts
Searching for Optimal Hyperparameters in ORPO Training
When optimizing the ORPO preferences, without setting the hyperparameters for the non-control
variables and emotional judgments, the indicators such as reward_margin (the probability of chosen
minus the probability of rejected) during the model training did not continue to rise, reward_acc (the
accuracy rate of the model output) did not continue to increase, and the loss did not continue to decrease.
The actual situation was that reward_margin always fluctuated around 0, reward_acc fluctuated around
0.5, and the loss also fluctuated within a certain range. Due to the tight project schedule and insufficient
computing power and resources, we were unable to explore the most ideal hyperparameter settings.
Therefore, we selected the model weights after 3k steps of iteration under the learning rate of 5𝑒−6and
=0.2. At this time, reward_margin and reward_acc reached the local maximum, and the loss reached a
lower point.
16

References
J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li,
J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu,
P . Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu,
H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu.
Qwen technical report, 2023. URL ht tp s: // ar xi v. or g/ ab s/ 2309.16609 .
Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen, Z. Chen, Z. Chen, P . Chu, X. Dong,
H. Duan, Q. Fan, Z. Fei, Y. Gao, J. Ge, C. Gu, Y. Gu, T. Gui, A. Guo, Q. Guo, C. He, Y. Hu, T. Huang,
T. Jiang, P . Jiao, Z. Jin, Z. Lei, J. Li, J. Li, L. Li, S. Li, W. Li, Y. Li, H. Liu, J. Liu, J. Hong, K. Liu,
K. Liu, X. Liu, C. Lv, H. Lv, K. Lv, L. Ma, R. Ma, Z. Ma, W. Ning, L. Ouyang, J. Qiu, Y. Qu, F. Shang,
Y. Shao, D. Song, Z. Song, Z. Sui, P . Sun, Y. Sun, H. Tang, B. Wang, G. Wang, J. Wang, J. Wang,
R. Wang, Y. Wang, Z. Wang, X. Wei, Q. Weng, F. Wu, Y. Xiong, C. Xu, R. Xu, H. Yan, Y. Yan,
X. Yang, H. Ye, H. Ying, J. Yu, J. Yu, Y. Zang, C. Zhang, L. Zhang, P . Zhang, P . Zhang, R. Zhang,
S. Zhang, S. Zhang, W. Zhang, W. Zhang, X. Zhang, X. Zhang, H. Zhao, Q. Zhao, X. Zhao, F. Zhou,
Z. Zhou, J. Zhuo, Y. Zou, X. Qiu, Y. Qiao, and D. Lin. Internlm2 technical report, 2024. URL
ht tp s: // ar xi v. or g/ ab s/ 2403.17297 .
H. Cao, Y. Wang, S. Jing, Z. Peng, Z. Bai, Z. Cao, M. Fang, F. Feng, B. Wang, J. Liu, T. Yang,
J. Huo, Y. Gao, F. Meng, X. Yang, C. Deng, and J. Feng. Safedialbench: A fine-grained safety
benchmark for large language models in multi-turn dialogues with diverse jailbreak attacks, 2025.
URL ht tp s: // ar xi v. or g/ ab s/ 2502.11090 .
K. Chen and Z. Sun. Deeppsy-agent: A stage-aware and deep-thinking emotional support agent system,
2025. URL ht tp s: // ar xi v. or g/ ab s/ 2503.15876 .
DeepSeek-AI, :, X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu,
H. Gao, K. Gao, W. Gao, R. Ge, K. Guan, D. Guo, J. Guo, G. Hao, Z. Hao, Y. He, W. Hu, P . Huang,
E. Li, G. Li, J. Li, Y. Li, Y. K. Li, W. Liang, F. Lin, A. X. Liu, B. Liu, W. Liu, X. Liu, X. Liu, Y. Liu,
H. Lu, S. Lu, F. Luo, S. Ma, X. Nie, T. Pei, Y. Piao, J. Qiu, H. Qu, T. Ren, Z. Ren, C. Ruan, Z. Sha,
Z. Shao, J. Song, X. Su, J. Sun, Y. Sun, M. Tang, B. Wang, P . Wang, S. Wang, Y. Wang, Y. Wang,
T. Wu, Y. Wu, X. Xie, Z. Xie, Z. Xie, Y. Xiong, H. Xu, R. X. Xu, Y. Xu, D. Yang, Y. You, S. Yu,
X. Yu, B. Zhang, H. Zhang, L. Zhang, L. Zhang, M. Zhang, M. Zhang, W. Zhang, Y. Zhang, C. Zhao,
Y. Zhao, S. Zhou, S. Zhou, Q. Zhu, and Y. Zou. Deepseek llm: Scaling open-source language models
with longtermism, 2024. URL ht tp s: // ar xi v. or g/ ab s/ 2401.02954 .
DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P . Wang, X. Bi,
X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu,
B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai,
F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao,
H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen,
K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang,
L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P . Huang, P . Zhang,
Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu,
S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun,
T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L.
Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li,
X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang,
17

X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu,
Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang,
Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li,
Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie,
Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan,
Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning, 2025. URL ht tp s: // ar xi v. or g/ ab s/ 2501.12948 .
T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. Qlora: Efficient finetuning of quantized
llms, 2023. URL ht tp s: // ar xi v. or g/ ab s/ 2305.14314 .
ggml.ai. llama.cpp. ht tp s: // gi th ub .c om /g gm l-o rg /l la ma .c pp , 2023. GitHub
repository.
J. Hong, N. Lee, and J. Thorne. Orpo: Monolithic preference optimization without reference model, 2024.
URL ht tp s: // ar xi v. or g/ ab s/ 2403.07691 .
A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand,
G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P . Stock, T. L. Scao, T. Lavril,
T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023. URL ht tp s: // ar xi v. or g/ ab s/
2310.06825 .
S. A. Laboratory. Opencompass. ht tp s: // gi th ub .c om /o pe n-c om pa ss /o pe nc om pa
ss, 2023a. GitHub repository.
S. A. Laboratory. Xtuner. ht tp s: // gi th ub .c om /I nt er nL M/ xt un er , 2023b. GitHub
repository.
C. Liu, Z. Wang, S. Shen, J. Peng, X. Zhang, Z. Du, and Y. Wang. The chinese dataset distilled from
deepseek-r1-671b. ht tp s: // hu gg in gf ac e. co /d at as et s/ Co ng li u/ Ch in es e-D
ee pS ee k-R 1-D is ti ll -d at a-110k , 2025.
V . B. Nguyen, J. Schlötterer, and C. Seifert. Ceval: A benchmark for evaluating counterfactual text
generation, 2024. URL ht tp s: // ar xi v. or g/ ab s/ 2404.17475 .
ollama. Ollama. ht tp s: // gi th ub .c om /o ll am a/ ol la ma , 2023. GitHub repository.
Open-WebUI. Open webui. ht tp s: // gi th ub .c om /o pe n-w eb ui /o pe n-w eb ui , 2023.
GitHub repository.
Open-WebUI. Pipelines: Ui-agnostic openai api plugin framework. ht tp s: // gi th ub .c om /o
pe n-w eb ui /p ip el in es , 2024. GitHub repository.
PKU-Alignment. Pku-saferlhf-single-dimension. ht tp s: // hu gg in gf ac e. co /d at as et
s/ PK U-A li gn me nt /P KU -S af eR LH F-s in gl e-d im en si on , 2024. Huggingface
repository.
SmartFlowAI. Emollm- 心理健康大模型.ht tp s: // gi th ub .c om /S ma rt Fl ow AI /E mo LL
M, 2024. GitHub repository.
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal,
E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient
foundation language models, 2023. URL ht tp s: // ar xi v. or g/ ab s/ 2302.13971 .
18

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin.
Attention is all you need, 2023. URL ht tp s: // ar xi v. or g/ ab s/ 1706.03762 .
H. Xie, Y. Chen, X. Xing, J. Lin, and X. Xu. Psydt: Using llms to construct the digital twin of
psychological counselor with personalized counseling style for psychological counseling, 2024. URL
ht tp s: // ar xi v. or g/ ab s/ 2412.13660 .
C. Zhang, R. Li, M. Tan, M. Yang, J. Zhu, D. Yang, J. Zhao, G. Ye, C. Li, and X. Hu. Cpsycoun: A
report-based multi-turn dialogue reconstruction and evaluation framework for chinese psychological
counseling, 2024. URL ht tp s: // ar xi v. or g/ ab s/ 2405.16433 .
19

Appendix
The detailed scores of model performance on CPsyCounE benchmark in nine psychological topics.
Table 4|Model Performance Comparison Across Different Topics
Model TopicEvaluation Metrics
Comprehensiveness
(0–2)Professionalism
(0–3)Authenticity
(0–3)Safety
(0–1)
OursCareer 2.00 2.91 2.80 1.00
Education 1.97 2.83 2.60 1.00
Emotion&Stress 2.00 2.78 2.75 1.00
Family Relationship 1.97 2.94 2.65 1.00
Love&Marriage 1.91 2.71 2.71 1.00
Mental Disease 2.00 2.93 2.93 1.00
Self-growth 1.94 2.83 2.71 1.00
Sex 2.00 2.70 2.67 1.00
Social Relationship 1.90 2.83 2.63 1.00
InternLM2.5-7b-chatCareer 1.69 1.97 2.37 1.00
Education 1.72 1.85 2.41 1.00
Emotion&Stress 1.76 1.78 2.59 1.00
Family Relationship 1.78 1.78 2.61 1.00
Love&Marriage 1.82 2.03 2.56 1.00
Mental Disease 1.66 1.86 2.40 1.00
Self-growth 1.83 1.94 2.46 1.00
Sex 1.93 1.93 2.77 1.00
Social Relationship 1.65 1.87 2.52 1.00
20
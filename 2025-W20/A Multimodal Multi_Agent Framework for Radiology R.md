# A Multimodal Multi-Agent Framework for Radiology Report Generation

**Authors**: Ziruo Yi, Ting Xiao, Mark V. Albert

**Published**: 2025-05-14 20:28:04

**PDF URL**: [http://arxiv.org/pdf/2505.09787v1](http://arxiv.org/pdf/2505.09787v1)

## Abstract
Radiology report generation (RRG) aims to automatically produce diagnostic
reports from medical images, with the potential to enhance clinical workflows
and reduce radiologists' workload. While recent approaches leveraging
multimodal large language models (MLLMs) and retrieval-augmented generation
(RAG) have achieved strong results, they continue to face challenges such as
factual inconsistency, hallucination, and cross-modal misalignment. We propose
a multimodal multi-agent framework for RRG that aligns with the stepwise
clinical reasoning workflow, where task-specific agents handle retrieval, draft
generation, visual analysis, refinement, and synthesis. Experimental results
demonstrate that our approach outperforms a strong baseline in both automatic
metrics and LLM-based evaluations, producing more accurate, structured, and
interpretable reports. This work highlights the potential of clinically aligned
multi-agent frameworks to support explainable and trustworthy clinical AI
applications.

## Full Text


<!-- PDF content starts -->

arXiv:2505.09787v1  [cs.AI]  14 May 2025A M ULTIMODAL MULTI -AGENT FRAMEWORK FOR RADIOLOGY
REPORT GENERATION
A P REPRINT
Ziruo Yi
University of North Texas
ziruoyi@my.unt.eduTing Xiao
University of North Texas
ting.xiao@unt.eduMark V . Albert
University of North Texas
mark.albert@unt.edu
May 16, 2025
ABSTRACT
Radiology report generation (RRG) aims to automatically produce diagnostic reports from medical
images, with the potential to enhance clinical workflows and reduce radiologists’ workload. While
recent approaches leveraging multimodal large language models (MLLMs) and retrieval-augmented
generation (RAG) have achieved strong results, they continue to face challenges such as factual
inconsistency, hallucination, and cross-modal misalignment. We propose a multimodal multi-agent
framework for RRG that aligns with the stepwise clinical reasoning workflow, where task-specific
agents handle retrieval, draft generation, visual analysis, refinement, and synthesis. Experimental
results demonstrate that our approach outperforms a strong baseline in both automatic metrics and
LLM-based evaluations, producing more accurate, structured, and interpretable reports. This work
highlights the potential of clinically aligned multi-agent frameworks to support explainable and
trustworthy clinical AI applications.
Keywords Radiology Report Generation ·Multimodal Large Language Models ·Multi-Agent Systems ·Retrieval-
Augmented Generation
1 Introduction
Radiology plays an essential role in modern healthcare, supporting diagnosis, treatment planning, and outcome pre-
diction. It involves diverse data sources such as chest X-ray images, laboratory results, and clinical notes. Among
multimodal tasks in radiology, radiology report generation (RRG) is particularly impactful as it directly supports clini-
cal workflows and decision-making. Recent studies further emphasize the importance of RRG as it aligns closely with
radiologists’ documentation [1, 2]. RRG typically involves two key modalities: chest X-ray images that provide visual
evidence of patient conditions, and corresponding radiology reports that capture clinical details in natural language.
However, increasing demand for radiological exams and a shortage of radiologists have led to delays, forcing clini-
cians to make critical decisions without radiological guidance, which may result in errors or conclusions that differ
from those of experienced radiologists [3, 4].
With advancements in artificial intelligence (AI), computer vision (CV), and natural language processing (NLP),
multimodal learning has emerged as a powerful paradigm for integrating and analyzing diverse data sources [5, 6].
Recently, large language models (LLMs) and large vision models (LVMs), including GPT-4V [7], LLaMA 3 [8],
and DALL·E 3 [9], have gained substantial attention. Building on these advances, multimodal large language mod-
els (MLLMs) have demonstrated strong performance on tasks like image captioning [10] and visual-language dia-
logue [11]. In the medical domain, MLLMs such as Med-PaLM 2 [12] and LLaV A-Med [13] have made notable
progress in pharmaceutical research [14] and clinical support [15]. For RRG, MLLMs integrate visual and textual
information to generate detailed and clinically accurate reports [16, 17], facilitating structured documentation and
supporting clinical decision-making. These applications extend the capabilities of radiologists, reduce workload, and
assist less experienced clinicians. Despite this progress, existing MLLM-based approaches face key limitations in
RRG. First, although MLLMs can process visual inputs effectively, they often struggle when essential information

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
is textual or requires cross-modal reasoning. Second, most existing systems lack a unified architecture that flexibly
integrates techniques such as prompt engineering. This limits their adaptability to new requirements in RRG. Third,
current methods typically lack intermediate validation or refinement stages, making them prone to factual inconsisten-
cies and hallucinations.
Retrieval-augmented generation (RAG) [18, 19, 20] has emerged as a promising method to enhance the factual ac-
curacy of medical MLLMs. By integrating external and reliable sources, RAG enriches the model’s knowledge and
supports more grounded generation. It has been applied to tasks such as visual question answering (VQA) [21] and
report generation [22, 23]. However, applying RAG to medical MLLMs introduces several new challenges. While
retrieving too few contexts may miss relevant information, retrieving too many can introduce noise and redundancy,
making it harder for the model to identify relevant content and ultimately degrading output quality.
To address the limitations of existing MLLM and RAG approaches, we propose a multimodal multi-agent framework
for RRG that decomposes the task into five specialized agents. Our framework combines RAG with a collaborative
multi-agent system in which specialized agents jointly process and integrate visual and textual information. It begins
with a Retrieval Agent that selects top- ksimilar reports for a given chest X-ray image. These retrieved examples
are passed to a Draft Agent, which generates an initial version of the report. A Refiner Agent then extracts key
findings from both the draft and the retrieved context to highlight essential diagnostic information. In parallel, a Vision
Agent produces a description summarizing visual observations from the chest X-ray image. Finally, the Synthesis
Agent integrates the outputs from the vision, retrieval, and refiner agents to generate the final report. This agent-
driven workflow follows the stepwise clinical reasoning process, assigning distinct roles to agents in a modular and
interpretable manner. The contributions of our work are: (1) We propose a clinically aligned multi-agent framework
for RRG, enabling modular collaboration across task-specific agents and incorporating RAG to enhance factuality
and controllability. (2) We conduct extensive experiments demonstrating that our method consistently outperforms a
strong single-agent baseline across both automatic metrics and LLM-based evaluations.
2 Related Work
MLLMs for RRG. MLLMs have recently emerged as a promising solution for automating RRG [24, 25, 26, 27, 28].
Models such as R2GenGPT [29], XrayGPT [30], and MAIRA-1 [31] combine visual encoders (e.g., Swin Trans-
former [32], MedCLIP [33]) with LLMs (e.g., LLaMA [8], Vicuna [34]) to align visual features with textual represen-
tations, demonstrating strong performance on benchmark datasets. Despite their success, these models still suffer from
key limitations including factual inconsistency [35, 36], hallucination [37], and catastrophic forgetting [38, 39]. These
issues are particularly critical in RRG, where factual accuracy and reliability are essential for clinical applications.
Retrieval-Augmented Generation. RAG has been widely adopted to enhance factual accuracy by incorporating con-
textual information from external datasets [18, 40]. It has been applied to RRG to reduce hallucinations and enhance
content relevance [41, 42, 43, 44, 45]. However, current RAG techniques face critical challenges: the number and
quality of retrieved contexts, and the risk of over-reliance on these references, both of which may degrade model
performance or introduce factual errors [46]. Moreover, existing RAG methods often retrieve and process text and
image information independently, limiting their ability to perform integrated multimodal reasoning [47]. These lim-
itations are particularly problematic in RRG, which depends on fine-grained alignment between retrieved knowledge
and visual evidence.
Multi-Agent Systems. Multi-agent systems have gained increasing attention in NLP and healthcare AI [48, 49, 50,
51, 52, 53]. They assign different tasks to specialized agents, which collaborate to accomplish complex goals that
single models often struggle with. Preliminary attempts have explored multi-agent paradigms for RRG [54, 55],
showing promising results. However, applying multi-agent systems to multimodal tasks introduces new challenges.
In particular, simply combining outputs from isolated vision and text agents often fails to capture the cross-modal
relationships required for accurate interpretation. Moreover, aligning agent interactions with domain-specific work-
flows such as stepwise clinical reasoning remains a key challenge in current systems. To address these limitations,
we propose a multimodal multi-agent framework aligned with stepwise clinical reasoning, where task-specific agents
handle retrieval, draft generation, refinement, visual analysis, and synthesis in a modular and interpretable manner.
3 Method
We propose a modular multi-agent framework for RRG, designed to emulate the clinical workflow by combining case
retrieval, visual interpretation, and structured textual synthesis. Given a chest X-ray image, the system sequentially
activates five specialized agents: a Retrieval Agent, a Draft Agent, a Refiner Agent, a Vision Agent, and a Synthesis
Agent. Each agent fulfills a distinct functional role and operates independently, using either task-specific prompts
2

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
(for LLM/VLM agents) or embedding-based retrieval (for the retrieval module). As shown in Figure 1, agents com-
municate via structured intermediate outputs that progressively refine radiological observations into a final, coherent
impression. This design promotes factual accuracy, improves interpretability, and helps ensure consistent report gen-
eration.
Figure 1: Overview of our proposed multi-agent framework for automated RRG. The system decomposes the task into
five agents: (1) Retrieval Agent selects top-k similar reports. (2) Draft Agent generates a preliminary report from the
retrieved texts. (3) Refiner Agent distills key clinical findings. (4) Vision Agent generates a visual description of the
image. (5) Synthesis Agent integrates these outputs to produce the final report.
3.1 Retrieval Agent
The Retrieval Agent performs cross-modal retrieval by identifying prior radiology reports that are semantically similar
to a given chest X-ray image. Inspired by the design of CLIP [56], the agent encodes the input image into a visual
embedding and compares it against report embeddings using cosine similarity. The top- kmost similar reports are
selected based on similarity scores, where kis a predefined parameter balancing retrieval coverage and efficiency.
These reports provide relevant diagnostic context, such as clinical findings and report style, which serve as guidance
for downstream generation.
3.2 Draft Agent
The Draft Agent composes a preliminary radiology report by synthesizing information from the top-k reports selected
by the Retrieval Agent. Inspired by how radiologists review similar prior cases, the agent identifies shared clinical
findings and prioritizes medically relevant observations. It then organizes this information into a clinically focused
report that reflects an initial diagnostic impression. This intermediate output provides a structured textual basis for
downstream processing by later agents.
3.3 Refiner Agent
The Refiner Agent distills key clinical findings from the outputs of the Draft and Retrieval Agents. It is designed to
identify clinically important observations that are consistently supported by the input. Unlike the Draft Agent, which
3

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
generates a broad preliminary report, the Refiner Agent focuses solely on findings-level content. It receives both
the preliminary report and the original retrieved reports as input, and outputs a concise, single-paragraph summary
containing the most essential findings. To ensure factuality, the agent enforces retrieval-grounded constraints: every
sentence must be clearly supported by the input, with no speculation or paraphrasing beyond factual rewriting. The
output provides a structured clinical signal for downstream synthesis.
3.4 Vision Agent
The Vision Agent generates a visual description of the chest X-ray image to complement textual information from
previous agents. It uses a medical MLLM to generate image-grounded descriptions based on visible observations in
the input image. The output is a caption describing key chest regions, such as the lungs and mediastinum. The agent is
designed to avoid unclear statements and irrelevant content, ensuring that the caption is grounded in visible evidence
and written in a radiology report style. This step introduces visual cues from the input image to support the final
synthesis.
3.5 Synthesis Agent
The Synthesis Agent produces the final radiology report by integrating a preliminary report, critical findings, and a
visual caption from the previous agents. To ensure both factual consistency and stylistic coherence, the final report
includes only observations explicitly supported by the textual or visual inputs. The agent is designed to avoid unsup-
ported findings and unnecessary rewriting, while preserving the core clinical content from each input and combining
them in a logically consistent manner. This final step concludes the multi-agent pipeline by generating a clinically
grounded and well-structured radiology report.
4 Experiments
In this section, we evaluate our multimodal multi-agent framework by addressing the following questions: (1) Does the
multi-agent design improve the clinical accuracy of generated radiology reports compared to the baseline? (2) Does
each agent play a meaningful role in the generation process? (3) How does the framework enhance the overall quality
of the generated reports?
4.1 Experimental Setup
Implementation Details. Our framework consists of five agents: a retrieval agent, a draft agent, a refiner agent, a
vision agent, and a synthesis agent. We follow the retrieval setup of RULE [46] to construct the retrieval agent, which
fine-tunes CLIP on MIMIC-CXR using contrastive learning to adapt to the medical domain. LLaV A-Med 1.5 (7B) [13]
is used as the backbone for the vision agent, while GPT-4o [7] powers the draft, refiner, and synthesis agents. The
retrieval agent selects the top- kmost similar reports ( k= 5 by default), which are then passed to the draft agent as
input.
Datasets. We utilize two publicly available chest X-ray datasets: MIMIC-CXR [57] and IU X-ray [58]. We fine-
tune the retrieval agent using 3,000 image–report pairs from MIMIC-CXR, a large-scale dataset of chest X-rays with
associated radiology reports. For evaluation, we use the IU X-ray dataset, which includes chest radiographs and
corresponding diagnostic reports. Following the data split from [59], the IU X-ray dataset contains 2,068 training and
590 test image–report pairs after filtering. We use the training set to construct the retrieval database and the test set to
evaluate the performance of our framework.
Evaluation Metrics. The performance of our multi-agent framework is evaluated using standard metrics for text
generation, including BLEU [60], ROUGE-1, ROUGE-2, ROUGE-L [61], and BERTScore [62]. These metrics focus
on surface-level similarity between generated and reference impressions, primarily based on lexical or token overlap.
To complement these automatic metrics, we adopt the LLM-as-a-Judge paradigm [63] and employ Claude 3 Opus [64],
a state-of-the-art LLM developed by Anthropic, to assess both the semantic accuracy and clinical relevance of the
generated reports.
4.2 Results
In this section, we present a comprehensive evaluation of our multi-agent framework on the IU X-ray dataset, compar-
ing it against a single-agent baseline using LLaV A-Med that simulates a radiologist working without access to prior
reports or clinical cues.
4

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
4.2.1 Quantitative Analysis
Standard Metrics. We evaluate the performance of our framework using standard metrics, including BLEU, ROUGE,
METEOR, and BERTScore. The results of this evaluation are listed in Table 1. Our multi-agent framework achieves
a BLEU score of 0.0466, significantly outperforming LLaV A-Med at 0.0036. ROUGE-1, ROUGE-2, and ROUGE-L
scores increase from 0.2398, 0.0278, and 0.1537 to 0.3652, 0.1292, and 0.2471 respectively, demonstrating consistent
gains across all ROUGE metrics. For METEOR, the score rises to 0.3618 from 0.1437, indicating better lexical
diversity and content coverage. On BERTScore, the framework achieves 0.8819 compared to 0.8617 for the baseline,
suggesting stronger semantic alignment between generated and reference texts. These results indicate that the multi-
agent design substantially improves both textual quality and semantic coherence in RRG.
Table 1: Quantitative performance comparison between our multi-agent framework and a single MLLM.
Model BLEU ROUGE-1 ROUGE-2 ROUGE-L METEOR BERTScore
Llava-Med 0.0036 0.2398 0.0278 0.1537 0.1437 0.8617
Ours 0.0466 0.3652 0.1292 0.2471 0.3618 0.8819
LLM-as-a-Judge. To complement standard metrics, we further assess the clinical and linguistic quality of the gen-
erated reports using Claude 3 Opus [64], focusing on five key aspects: coverage of key findings, consistency with
original reports, diagnostic accuracy, stylistic alignment, and conciseness. Each aspect is rated from 1 to 10, with
higher scores indicating better quality. As summarized in Table 2, our multi-agent framework outperforms LLaV A-
Med on four of the five evaluation dimensions. It achieves a diagnostic accuracy score of 8.26 compared to 7.78,
demonstrating stronger clinical reasoning. Our framework scores 8.16 in style alignment and 7.26 in conciseness,
surpassing the baseline scores of 7.98 and 6.98. These improvements reflect stronger alignment with clinical report
writing standards. Key finding coverage also improves from 5.86 to 6.36, showing a clearer presentation of clinically
relevant information. Although LLaV A-Med slightly leads in consistency (6.94 vs. 6.74), the overall results highlight
the effectiveness of our multi-agent design in enhancing clinical reliability and writing quality.
Table 2: Qualitative performance comparison between our multi-agent framework and a single MLLM.
Model Findings Consistency Diagnosis Style Conciseness
LLaV A-Med 5.86 6.94 7.78 7.98 6.98
Ours 6.36 6.74 8.26 8.16 7.26
4.2.2 Qualitative Analysis
Figure 2 presents a representative example comparing the report generated by the Vision Agent only with that of
our full multi-agent framework. This case highlights the benefit of incorporating retrieved reports and extracted key
findings through our multi-agent pipeline. The vision-only output, while stylistically reasonable, lacks specificity
and misses important observations. In contrast, the multi-agent output offers a more complete and clinically aligned
summary. It follows terminology and structure commonly seen in prior reports, such as the inclusion of “pleural effu-
sion” and “Degenerative changes are noted in the spine,” better aligning with the original report. These improvements
are enabled by structured agent collaboration: the Retrieval Agent supplies relevant contextual examples, the Refiner
Agent extracts key clinical findings, and the Synthesis Agent combines them with the visual caption into a structured
report. The final output is more concise, better organized, and clinically reliable, illustrating how retrieval grounding
and agent collaboration lead to higher-quality reports than those produced by a vision-only agent.
4.3 Discussion
Our results show that the proposed multi-agent framework consistently enhances RRG across both standard automatic
metrics and LLM-based clinical assessment. By assigning each agent a specific function such as retrieval, abstraction,
refinement, visual captioning, and synthesis, the framework introduces a clearer structure and separation of respon-
sibilities. This modularity enables more controllable and interpretable generation, allowing each agent to focus on a
distinct aspect of clinical reasoning or stylistic consistency. The generated reports are more complete and stylistically
aligned. They also show stronger clinical grounding and better adherence to radiology reporting conventions.
5

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
Figure 2: A case study showing that retrieval and key findings help overcome the limitations of a vision-only agent.
While our framework demonstrates notable improvements, we observe a slight drop in consistency compared to the
baseline. Similar findings have been reported in prior work [54], where RAG can introduce redundant or irrelevant
content, reducing overall coherence. In our case, the combination of retrieved reports, refined findings, and visual
descriptions introduces additional complexity, which can affect the overall consistency of the final output. Despite this
limitation, our experiments demonstrate that the multi-agent framework improves clinical accuracy and report quality
compared to a strong single-agent baseline, as validated by both automatic and LLM-based evaluations. The case
study further highlights the distinct contributions of individual agents in enhancing factuality and structure. Future
work includes a more systematic investigation, particularly through agent-level ablation.
5 Conclusion
We present a multimodal multi-agent framework for RRG that breaks down the task into specialized agents for re-
trieval, draft, refinement, visual interpretation, and synthesis. Our approach follows the diagnostic reasoning pro-
cess and outperforms a strong single-agent MLLM baseline, as demonstrated by both automatic metrics and LLM-
based evaluations. Through the collaboration among agents, the framework produces reports that are more clinically
grounded, coherent, and stylistically aligned. This modular design offers a generalizable method for other multimodal
medical tasks requiring diagnostic reasoning and clinical precision.
References
[1] Nur Yildirim, Hannah Richardson, Maria Teodora Wetscherek, Junaid Bajwa, Joseph Jacob, Mark Ames Pin-
nock, Stephen Harris, Daniel Coelho De Castro, Shruthi Bannur, Stephanie Hyland, et al. Multimodal healthcare
ai: identifying and designing clinically relevant vision-language applications for radiology. In Proceedings of
the 2024 CHI Conference on Human Factors in Computing Systems , pages 1–22, 2024.
[2] Ziruo Yi, Ting Xiao, and Mark V Albert. A survey on multimodal large language models in radiology for report
generation and visual question answering. Information , 16(2):136, 2025.
[3] Robert J McDonald, Kara M Schwartz, Laurence J Eckel, Felix E Diehn, Christopher H Hunt, Brian J Bartholmai,
Bradley J Erickson, and David F Kallmes. The effects of changes in utilization and technological advancements
of cross-sectional imaging on radiologist workload. Academic radiology , 22(9):1191–1198, 2015.
6

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
[4] Bruno Petinaux, Rahul Bhat, Keith Boniface, and Jaime Aristizabal. Accuracy of radiographic readings in the
emergency department. The American journal of emergency medicine , 29(1):18–25, 2011.
[5] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. What makes multi-
modal learning better than single (provably). Advances in Neural Information Processing Systems , 34:10944–
10956, 2021.
[6] Asim Waqas, Aakash Tripathi, Ravi P Ramachandran, Paul A Stewart, and Ghulam Rasool. Multimodal data in-
tegration for oncology in the era of deep neural networks: a review. Frontiers in Artificial Intelligence , 7:1408843,
2024.
[7] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774 , 2023.
[8] AI Meta. Introducing meta llama 3: The most capable openly available llm to date. Meta AI , 2024.
[9] OpenAI. DALL-E3, 2023. https://openai.com/index/dall-e-3/ .
[10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training
with frozen image encoders and large language models. In International conference on machine learning , pages
19730–19742. PMLR, 2023.
[11] Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Nigel Collier, and Yutong Lu. Sparkles: Unlocking chats
across multiple images for multimodal instruction-following models. arXiv preprint arXiv:2308.16463 , 2023.
[12] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark,
Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language
models. Nature Medicine , pages 1–8, 2025.
[13] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung
Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day.
Advances in Neural Information Processing Systems , 36:28541–28564, 2023.
[14] Francesca Grisoni. Chemical language models for de novo drug design: Challenges and opportunities. Current
Opinion in Structural Biology , 79:102527, 2023.
[15] Stephen R Ali, Thomas D Dobbs, Hayley A Hutchings, and Iain S Whitaker. Using chatgpt to write patient clinic
letters. The Lancet Digital Health , 5(4):e179–e181, 2023.
[16] Zhengliang Liu, Yiwei Li, Peng Shu, Aoxiao Zhong, Longtao Yang, Chao Ju, Zihao Wu, Chong Ma, Jie
Luo, Cheng Chen, et al. Radiology-llama2: Best-in-class large language model for radiology. arXiv preprint
arXiv:2309.06419 , 2023.
[17] Manuela Daniela Danu, George Marica, Sanjeev Kumar Karn, Bogdan Georgescu, Awais Mansoor, Florin Ghesu,
Lucian Mihai Itu, Constantin Suciu, Sasa Grbic, Oladimeji Farri, et al. Generation of radiology findings in chest
x-ray by leveraging collaborative knowledge. Procedia Computer Science , 221:1102–1109, 2023.
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2:1, 2023.
[19] Xiaoye Qu, Qiyuan Chen, Wei Wei, Jishuo Sun, and Jianfeng Dong. Alleviating hallucination in large vision-
language models with active retrieval augmentation. arXiv preprint arXiv:2408.00555 , 2024.
[20] Xiaoye Qu, Jiashuo Sun, Wei Wei, and Yu Cheng. Look, compare, decide: Alleviating hallucination in large
vision-language models via multi-view multi-path reasoning. arXiv preprint arXiv:2408.17150 , 2024.
[21] Zheng Yuan, Qiao Jin, Chuanqi Tan, Zhengyun Zhao, Hongyi Yuan, Fei Huang, and Songfang Huang. Ramm:
Retrieval-augmented biomedical visual question answering with multi-modal pre-training. In Proceedings of the
31st ACM International Conference on Multimedia , pages 547–556, 2023.
[22] Yogesh Kumar and Pekka Marttinen. Improving medical multi-modal contrastive learning with expert annota-
tions. In European Conference on Computer Vision , pages 468–486. Springer, 2024.
[23] Yitian Tao, Liyan Ma, Jing Yu, and Han Zhang. Memory-based cross-modal semantic alignment network for
radiology report generation. IEEE Journal of Biomedical and Health Informatics , 2024.
[24] Jia Ji, Yongshuai Hou, Xinyu Chen, Youcheng Pan, and Yang Xiang. Vision-language model for generating
textual descriptions from clinical images: Model development and validation study. JMIR Formative Research ,
8:e32690, 2024.
7

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
[25] Chang Liu, Yuanhe Tian, Weidong Chen, Yan Song, and Yongdong Zhang. Bootstrapping large language models
for radiology report generation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38,
pages 18635–18643, 2024.
[26] Yuhao Wang, Chao Hao, Yawen Cui, Xinqi Su, Weicheng Xie, Tao Tan, and Zitong Yu. Trrg: Towards truth-
ful radiology report generation with cross-modal disease clue enhanced large language model. arXiv preprint
arXiv:2408.12141 , 2024.
[27] Zijian Zhou, Miaojing Shi, Meng Wei, Oluwatosin Alabi, Zijie Yue, and Tom Vercauteren. Large model driven
radiology report generation with clinical quality reinforcement learning. arXiv preprint arXiv:2403.06728 , 2024.
[28] Yuzhe Lu, Sungmin Hong, Yash Shah, and Panpan Xu. Effectively fine-tune to improve large multimodal models
for radiology report generation. arXiv preprint arXiv:2312.01504 , 2023.
[29] Zhanyu Wang, Lingqiao Liu, Lei Wang, and Luping Zhou. R2gengpt: Radiology report generation with frozen
llms. Meta-Radiology , 1(3):100033, 2023.
[30] Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer,
Salman Khan, Jorma Laaksonen, and Fahad Shahbaz Khan. Xraygpt: Chest radiographs summarization using
medical vision-language models. arXiv preprint arXiv:2306.07971 , 2023.
[31] Stephanie L Hyland, Shruthi Bannur, Kenza Bouzid, Daniel C Castro, Mercy Ranjit, Anton Schwaighofer, Fer-
nando P ´erez-Garc ´ıa, Valentina Salvatelli, Shaury Srivastav, Anja Thieme, et al. Maira-1: A specialised large
multimodal model for radiology report generation. arXiv preprint arXiv:2311.13668 , 2023.
[32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin trans-
former: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international
conference on computer vision , pages 10012–10022, 2021.
[33] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired
medical images and text. In Proceedings of the Conference on Empirical Methods in Natural Language Process-
ing. Conference on Empirical Methods in Natural Language Processing , volume 2022, page 3876, 2022.
[34] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang,
Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt
quality, march 2023. URL https://lmsys. org/blog/2023-03-30-vicuna , 3(5), 2023.
[35] Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou,
Kangyu Zhu, et al. Cares: A comprehensive benchmark of trustworthiness in medical vision language models.
Advances in Neural Information Processing Systems , 37:140334–140365, 2024.
[36] Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao Li, Min Zhang, and
Yu Cheng. Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm. arXiv preprint
arXiv:2408.12076 , 2024.
[37] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and
Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253 , 2024.
[38] Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the
catastrophic forgetting in multimodal large language models. arXiv preprint arXiv:2309.10313 , 2023.
[39] Hikmat Khan, Nidhal C Bouaynaya, and Ghulam Rasool. The importance of robust features in mitigating catas-
trophic forgetting. In 2023 IEEE Symposium on Computers and Communications (ISCC) , pages 752–757. IEEE,
2023.
[40] Jiashuo Sun, Jihai Zhang, Yucheng Zhou, Zhaochen Su, Xiaoye Qu, and Yu Cheng. Surf: Teaching large vision-
language models to selectively utilize retrieved information. arXiv preprint arXiv:2409.14083 , 2024.
[41] Mercy Ranjit, Gopinath Ganapathy, Ranjit Manuel, and Tanuja Ganu. Retrieval augmented chest x-ray report
generation using openai gpt models. In Machine Learning for Healthcare Conference , pages 650–666. PMLR,
2023.
[42] Liwen Sun, James Zhao, Megan Han, and Chenyan Xiong. Fact-aware multimodal retrieval augmentation for
accurate medical radiology report generation. arXiv preprint arXiv:2407.15268 , 2024.
[43] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, and
Huaxiu Yao. Mmed-rag: Versatile multimodal rag system for medical vision language models. arXiv preprint
arXiv:2410.13085 , 2024.
[44] Siting Liang, Pablo S ´anchez, and Daniel Sonntag. Optimizing relation extraction in medical texts through active
learning: A comparative analysis of trade-offs. In Proceedings of the 1st Workshop on Uncertainty-Aware NLP
(UncertaiNLP 2024) , pages 23–34, 2024.
8

A Multimodal Multi-Agent Framework for Radiology Report Generation A P REPRINT
[45] Mario Luca Bernardi and Marta Cimitile. Report generation from x-ray imaging by retrieval-augmented gener-
ation and improved image-text matching. In 2024 International Joint Conference on Neural Networks (IJCNN) ,
pages 1–8. IEEE, 2024.
[46] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu Yao. Rule: Reliable
multimodal rag for factuality in medical vision language models. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing , pages 1081–1093, 2024.
[47] Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. Mdocagent: A multi-modal
multi-agent framework for document understanding. arXiv preprint arXiv:2503.13964 , 2025.
[48] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer
Science , 18(6):186345, 2024.
[49] Ling Yue and Tianfan Fu. Ct-agent: Clinical trial multi-agent with large language model-based reasoning. arXiv
e-prints , pages arXiv–2404, 2024.
[50] Yu He Ke, Rui Yang, Sui An Lie, Taylor Xin Yi Lim, Hairil Rizal Abdullah, Daniel Shu Wei Ting, and Nan
Liu. Enhancing diagnostic accuracy through multi-agent conversations: using large language models to mitigate
cognitive bias. arXiv preprint arXiv:2401.14589 , 2024.
[51] Hao Wei, Jianing Qiu, Haibao Yu, and Wu Yuan. Medco: Medical education copilots based on a multi-agent
framework. arXiv preprint arXiv:2408.12496 , 2024.
[52] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark
Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint
arXiv:2311.10537 , 2023.
[53] Andries Petrus Smit, Paul Duckworth, Nathan Grinsztajn, Kale-ab Tessera, Thomas D Barrett, and Arnu Preto-
rius. Are we going mad? benchmarking multi-agent debate between language models for medical q&a. In Deep
Generative Models for Health Workshop NeurIPS 2023 , 2023.
[54] Fang Zeng, Zhiliang Lyu, Quanzheng Li, and Xiang Li. Enhancing llms for impression generation in radiology
reports through a multi-agent system. arXiv preprint arXiv:2412.06828 , 2024.
[55] Hasan Md Tusfiqur Alam, Devansh Srivastav, Md Abdul Kadir, and Daniel Sonntag. Towards interpretable
radiology report generation via concept bottlenecks using a multi-agentic rag. In European Conference on Infor-
mation Retrieval , pages 201–209. Springer, 2025.
[56] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language
supervision. In International conference on machine learning , pages 8748–8763. PmLR, 2021.
[57] Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Yifan Peng,
Zhiyong Lu, Roger G Mark, Seth J Berkowitz, and Steven Horng. Mimic-cxr-jpg, a large publicly available
database of labeled chest radiographs. arXiv preprint arXiv:1901.07042 , 2019.
[58] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer
Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for
distribution and retrieval. Journal of the American Medical Informatics Association , 23(2):304–310, 2016.
[59] Kang Liu, Zhuoqi Ma, Mengmeng Liu, Zhicheng Jiao, Xiaolu Kang, Qiguang Miao, and Kun Xie. Factual
serialization enhancement: A key innovation for chest x-ray report generation. arXiv preprint arXiv:2405.09586 ,
2024.
[60] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of
machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics ,
pages 311–318, 2002.
[61] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out ,
pages 74–81, 2004.
[62] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text
generation with bert. arXiv preprint arXiv:1904.09675 , 2019.
[63] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural
Information Processing Systems , 36:46595–46623, 2023.
[64] Anthropic. Claude 3 haiku: Our fastest model yet, 2024. https://www.anthropic.com/news/
claude-3-haiku .
9
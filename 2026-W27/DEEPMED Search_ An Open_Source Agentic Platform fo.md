# DEEPMED Search: An Open-Source Agentic Platform for Medical Deep Research with Introspective Verification

**Authors**: Maolin Liu, Fanyu Xu, Ruoqing Xu, Jiahang Zhang, Hao Wang, Rui Wang

**Published**: 2026-06-29 03:42:48

**PDF URL**: [https://arxiv.org/pdf/2606.29746v1](https://arxiv.org/pdf/2606.29746v1)

## Abstract
Navigating the deluge of heterogeneous medical data, from academic literature (PubMed) to clinical guidelines (Web) and private knowledge bases, remains a critical bottleneck for evidence-based medicine. While commercial black-box tools lack transparency, standard open-source RAG implementations frequently suffer from reasoning drift when handling complex, long-tail queries. We present DEEPMED Search, a fully open-source, agentic platform designed for transparent medical deep research. Built on a high-performance Next.js architecture, DEEPMED Search features a source-adaptive router that autonomously dispatches sub-queries to PubMed, web search, or local graph-based knowledge bases based on information density. Crucially, the platform integrates an introspective verification module, powered by a causal-consistent multi-agent debate framework, to validate retrieved evidence against diagnostic logic before synthesis. To demonstrate its robustness, we showcase DEEPMED Search's ability to autonomously decompose high-difficulty rare disease queries, filter out confounding noise, and generate structured, citation-backed research reports in minutes. By open-sourcing this software, we provide the community with a robust infrastructure to democratize access to trustworthy, glass-box medical reasoning in research and prototyping settings.

## Full Text


<!-- PDF content starts -->

DEEPMEDSearch: An Open-Source Agentic Platform for Medical Deep Research
with Introspective Verification
Maolin Liu1,Fanyu Xu1,Ruoqing Xu1,Jiahang Zhang1,Hao Wang1,∗andRui Wang2
1School of Computer Science and Software Engineering, Shanghai University
2Department of Computer Science and Engineering, Shanghai Jiao Tong University
{23721623, 24122195, 50x20, kaito1412, wang-hao}@shu.edu.cn, wangrui12@sjtu.edu.cn
Abstract
Navigating the deluge of heterogeneous medical
data, from academic literature (PubMed) to clini-
cal guidelines (Web) and private knowledge bases
remains a critical bottleneck for evidence-based
medicine. While commercial black-box tools lack
transparency, standard open-source RAG imple-
mentations frequently suffer from “reasoning drift”
when handling complex, long-tail queries. We
presentDEEPMEDSEARCH, a fully open-source,
agentic platform designed for transparent medi-
cal deep research. Built on a high-performance
NEXT.JSarchitecture, DEEPMEDSEARCHfea-
tures a source-adaptive router that autonomously
dispatches sub-queries to PubMed, web search, or
local graph-based knowledge bases based on infor-
mation density. Crucially, the platform integrates
an introspective verification module, powered by
a causal-consistent multi-agent debate framework,
to validate retrieved evidence against diagnostic
logic before synthesis. To demonstrate its robust-
ness, we showcase DEEPMEDSEARCH’s ability to
autonomously decompose high-difficulty rare dis-
ease queries, filter out confounding noise, and gen-
erate structured, citation-backed research reports
in minutes. By open-sourcing this software, we
provide the community with a robust infrastruc-
ture to democratize access to trustworthy, glass-box
medical reasoning in research and prototyping set-
tings, which is publicly available at: https://www.
deepmedsearch.cloud and the demonstration video
is available at: https://youtu.be/4U4aok8yLpk.
1 Introduction
Large Language Models (LLMs) have revolutionized clin-
ical question answering [Singhalet al., 2025; Tuet al.,
2023; Yanget al., 2022 ], yet their deployment in high-
stakes evidence-based medicine remains hindered by limited
transparency and auditability [Jiet al., 2023; Rudin, 2019 ].
Medical deep research requires synthesizing heterogeneous
data from academic literature, constantly updating clinical
∗Corresponding Author.guidelines, and structured patient records. While Retrieval-
Augmented Generation (RAG) is widely adopted to ground
model outputs [Lewiset al., 2020; Karpukhinet al., 2020;
Izacard and Grave, 2021 ], standard RAG implementations
often rely on static, linear pipelines that are vulnerable to
Retrieval-Induced Reasoning Drift, especially in complex,
long-tail scenarios such as rare disease diagnosis [Mallen
et al., 2023 ]. In these contexts, standard semantic retriev-
ers suffer from commonality bias [Kandpalet al., 2023 ],
preferentially surfacing generic evidence that supports com-
mon mimic conditions and causing the LLM to be hijacked
by plausible but non-discriminative distractors. Although
commercial AI research tools attempt to address this issue,
their black-box nature prevents clinicians from auditing the
causal logic behind retrieved evidence, which remains a non-
negotiable requirement in clinical workflows.
To democratize trustworthy and auditable AI in health-
care, we introduce DEEPMEDSEARCH, a fully open-
source, agentic platform engineered for transparent medical
deep research. Moving beyond simple RAG frameworks,
DEEPMEDSEARCHintroduces a source-adaptive router that
autonomously orchestrates sub-queries across heterogeneous
endpoints, including PubMed APIs, live web search, and a
local Neo4j-based knowledge graph. Furthermore, to en-
sure the causal alignment and faithfulness of the retrieved
evidence, the platform features an introspective verification
module [Kicimanet al., 2024 ]. This engine employs con-
trastive hypothesis expansion to filter confounding noise and
utilizes a multi-agent adversarial debate framework [Duet
al., 2024; Yaoet al., 2022 ]to rigorously stress-test reason-
ing paths before final synthesis.
In this paper, we showcase the system’s glass-box interac-
tivity, while clinicians can observe DEEPMEDSEARCHas it
autonomously decomposes complex queries, routes searches,
and conducts real-time multi-agent deliberations. Ultimately,
the system generates structured, citation-backed research re-
ports while maintaining complete interpretability [Xionget
al., 2024 ].
2 System Architecture
DEEPMEDSEARCHis engineered as a modular, agentic
workflow designed to transform high-level clinical inquiries
into structured research insights. As illustrated in Figure 1,
the platform transitions from autonomous data orchestrationarXiv:2606.29746v1  [cs.AI]  29 Jun 2026

Figure 1: System architecture of DEEPMEDSEARCH. The platform integrates a source-adaptive router for multi-endpoint evidence retrieval,
causal introspective filtering to mitigate commonality bias, and a multi-agent debate engine to verify reasoning and synthesize reports.
to rigorous causal verification across three primary stages:
Source-Adaptive Routing, Causal Introspective Filtering, and
Multi-Agent Synthesis.
2.1 Source-Adaptive Routing
A core pillar of DEEPMEDSEARCHis its ability to navigate
heterogeneous data landscapes. Unlike monolithic RAG sys-
tems, our platform incorporates aSource-Adaptive Router
that treats the information retrieval process as a sequential
decision-making task.
Upon receiving a clinical query, the system first performs
Atomic Decomposition, breaking the input into distinct clin-
ical constraints (e.g., symptomatic triad, temporal progres-
sion, and treatment resistance). The Router then dynamically
evaluates the information density of each sub-query to dis-
patch them to optimal endpoints:
•Scientific Evidence:Direct API integration with
PubMedfor fetching peer-reviewed literature and clini-
cal trial abstracts [Qiuet al., 2025 ].
•Pathophysiological Logic:Querying a local NEO4J
knowledge graph (integrating UMLS and SNOMED-
CT) to identify multi-hop causal links between symp-
toms [Pearl, 2009 ].
•Real-time Guidelines:Utilizing aWeb Searchcrawler
to retrieve the latest consensus statements and medical
news.
2.2 Causal Introspective Filtering
To mitigate theCommonality Biasidentified in clinical LLMs
[Kandpalet al., 2023 ], the platform utilizes aCausal In-
trospective Filteringmechanism. This stage aligns the re-
trieved evidence with diagnostic logic via two steps:1) Con-
trastive Hypothesis Expansion:The system proactively for-
mulates “confounding hypotheses”, prevalent diseases that
mimic the target long-tail condition (e.g., generating “Es-
sential Hypertension” when analyzing potential “Pheochro-
mocytoma”) [Park and Lee, 2024 ].2) Evidence Down-
weighting:Chunks are filtered based on a discriminative
scoreS i=sim(c i,Htarget )
sim(c i,Hmimic )+ϵ. Here, sim(·,·)denotes the co-
sine similarity between BGE-M3 sentence embeddings, andϵis set to10−6to avoid division by zero. A candidate chunk is
regarded as discriminative whenS iis larger than the thresh-
oldτ s; chunks failing to meet a dynamic threshold are tagged
as “non-discriminative noise” and visually dimmed in the in-
terface, ensuring the model’s attention remains on the long-
tail signals.
2.3 Multi-Agent Synthesis
The final reasoning phase is handled by an introspective ver-
ification module, instantiated as a multi-agent adversarial de-
bate framework [Chanet al., 2023 ]. The system deploys three
distinct agents: aProponent(A pro) that constructs reason-
ing chains from evidence to diagnosis, anOpponent(A con)
that identifies missing causal links or inconsistencies, and a
Judge(A judge ) that delivers the final verdict once reasoning
converges (Information Gain< τ).
Crucially, rather than producing a single-line answer, the
synthesis engine generates a structured research report. These
reports are citation-backed and organized into sections: Dif-
ferential Diagnosis, Supporting Evidence (from both Graph
and Text), and Rejected Hypotheses. This “glass-box” output
allows clinicians to audit the platform’s logic in real-time.
3 Implementation Details
The DEEPMEDSEARCHbackend is engineered for high-
throughput concurrent processing to support real-time clin-
ical deep research workflows.Frontend & Orchestration:
Built on NEXT.JS, the frontend integrates aWebSocket-driven
UI for streaming real-time agent deliberations. A dedi-
cated orchestration layer handles the concurrent dispatch of
sub-queries via the Source-Adaptive Router.Knowledge
Engines:Structured multi-hop reasoning is powered by a
NEO4Jgraph database integrating UMLS and SNOMED-CT.
For unstructured evidence, we use MILVUSto index the MI-
RAGE corpus (Textbooks and PubMed) [Xionget al., 2024 ],
utilizing BGE-M3 for dense vector similarity coupled with a
lightweight cross-encoder for precise re-ranking.LLM Serv-
ing:To meet the computational demands of deploying three
concurrent agents (Proponent, Opponent, Judge), we serve
models like QWEN2.5-14B and DEEPSEEK-V3 viaVLLM,

Query:“Patient presents with paroxysmal hypertension,
headache, palpitations, and sweating. Standard anti-hypertensive
drugs ineffective. ”
×Vanilla RAG Response:
Diagnosis:Essential Hypertension. Treatment involves lifestyle
changes...
[Error: Fixates on common disease; ignores paroxysmal & sweat-
ing cues]
✓DEEPMEDSEARCH(Ours):
Diagnosis:Pheochromocytoma.
Reasoning:1.Contrastive Filter:‘Essential Hypertension’ down-
weighted (cannot explain sweating). 2.Debate Verdict:Symptom
triad strongly indicates catecholamine-secreting tumor.
Table 1: Case study: DEEPMEDSEARCHactively filters common
disease noise (Hypertension) to correctly identify the rare condition.
leveragingPagedAttentionto optimize memory footprint and
maximize throughput.Report Generation Engine:A cus-
tom synthesis module compiles the verified evidence into
a structured markdown/PDF report, automatically mapping
reasoning steps to specific reference nodes (PubMed IDs or
Graph edges) to ensure full traceability.
4 Demonstration Scenarios
We present using real-world rare disease cases from the
MedR-Bench dataset, as exemplified in Table 1. The web
interface allows clinicians to input complex case reports and
interactively audit the deep research process.
4.1 Scenario 1: Autonomous Routing & Causal
Filtering
Consider a patient presenting with episodic hypertension,
headaches, and palpitations. A standard RAG system typ-
ically retrieves generic guidelines for “Primary Hyperten-
sion”, leading to reasoning drift. In DEEPMEDSEARCH:
1) Routing in Action:The Router identifies the temporal
cue (“episodic”) and dispatches parallel queries to Neo4j and
PubMed.2) Noise Filtering:The “Evidence Panel” marks
generic hypertension evidence with aLow Relevancetag (vi-
sualized as dimmed text), explicitly stating it was down-
weighted due to overlap with common confounders. Con-
versely, evidence pointing toPheochromocytomais dynami-
cally highlighted.
4.2 Scenario 2: Glass-Box Debate & Report
For complex diagnoses, the clinician can activate the intro-
spective debate mode.1) Real-Time Deliberation:As il-
lustrated in Figure 2, a streaming panel opens whereA pro
argues for Pheochromocytoma citing paroxysmal symptoms,
whileA conchallenges the evidence density.2) Human-in-
the-Loop:Crucially, the clinician can inject missing parame-
ters (e.g.,“Add lab result: elevated metanephrines”) directly
into the debate stream.Structured Output:TheA judge fi-
nalizes the verdict, and the system instantly compiles a deep
research report, linking the diagnosis to verified graph nodes.
Figure 2: The Multi-Agent Debate Panel. Users can monitor agents
as they argue over the diagnosis, providing transparent reasoning
logic.
5 Evaluation
As summarized in Table 2, DEEPMEDuniquely integrates
source-adaptive routing and multi-agent verification, bridg-
ing the gap between vanilla RAG and commercial black-box
tools.
Performance Gains:We evaluated on the MIRAGE
benchmark [Xionget al., 2024 ]. DeepMed Search consis-
tently outperforms standard RAG baselines. On the challeng-
ing PubMedQA subset, our method improves accuracy by ap-
proximately 9.0% over vanilla RAG (using Qwen3-8B). For
rare diseases, experiments on the MedR-Bench dataset show
that our approach achieves an accuracy of 76.43% (using
DeepSeek-v3.2), bridging the gap between automated sys-
tems and expert-level reasoning.
Capabilities Vanilla RAG Commercial Ours
Open-Source Transparency✓×✓
Adaptive Routing (Web/Graph)×✓ ✓
Causal-Consistent Filtering× ×✓
Multi-Agent Verification× ×✓
Structured Deep Reports×✓ ✓
Table 2: Feature comparison. DEEPMEDSEARCHbridges glass-
box transparency with practical research-oriented system capabili-
ties.
6 Conclusion
DEEPMEDSEARCHprovides a transparent, agentic platform
for medical deep research. It combines source-adaptive rout-
ing, causal-consistent filtering, and multi-agent verification
to mitigate retrieval-induced reasoning drift in medical LLMs
[Park and Lee, 2024 ]and generate structured, citation-backed
reports. Current limitations include additional computational
and latency overhead and the lack of a formal clinician user
study. Future work will improve efficiency, involve clinicians
in evaluation, and extend the framework to multimodal data
such as medical imaging [Wanget al., 2025 ].

Acknowledgments
This work was supported by the National Natural Science
Foundation of China under Grant No. 62306173.
References
[Chanet al., 2023 ]Chi-Min Chan, Weize Chen, Yusheng
Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu,
and Zhiyuan Liu. Chateval: Towards better llm-based
evaluators through multi-agent debate.arXiv preprint
arXiv:2308.07201, 2023.
[Duet al., 2024 ]Yilun Du, Shuang Li, Antonio Torralba,
Joshua B Tenenbaum, and Igor Mordatch. Improving fac-
tuality and reasoning in language models through multi-
agent debate. InForty-first international conference on
machine learning, 2024.
[Izacard and Grave, 2021 ]Gautier Izacard and Edouard
Grave. Leveraging passage retrieval with generative mod-
els for open domain question answering. InProceedings
of the 16th conference of the european chapter of the asso-
ciation for computational linguistics: main volume, pages
874–880, 2021.
[Jiet al., 2023 ]Ziwei Ji, Nayeon Lee, Rita Frieske,
Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of halluci-
nation in natural language generation.ACM Computing
Surveys, 55(12):1–38, March 2023.
[Kandpalet al., 2023 ]Nikhil Kandpal, Haikang Deng,
Adam Roberts, Eric Wallace, and Colin Raffel. Large
language models struggle to learn long-tail knowledge.
In Andreas Krause, Emma Brunskill, Kyunghyun Cho,
Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett,
editors,International Conference on Machine Learn-
ing, ICML 2023, 23-29 July 2023, Honolulu, Hawaii,
USA, volume 202 ofProceedings of Machine Learning
Research, pages 15696–15707. PMLR, 2023.
[Karpukhinet al., 2020 ]Vladimir Karpukhin, Barlas Oguz,
Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval
for open-domain question answering. InProceedings of
the 2020 conference on empirical methods in natural lan-
guage processing (EMNLP), pages 6769–6781, 2020.
[Kicimanet al., 2024 ]Emre Kiciman, Robert Osazuwa
Ness, Amit Sharma, and Chenhao Tan. Causal reason-
ing and large language models: Opening a new frontier
for causality.Transactions on Machine Learning Research
(TMLR), August 2024. Selected for presentation at ICLR
2025.
[Lewiset al., 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33:9459–9474, 2020.
[Mallenet al., 2023 ]Alex Mallen, Akari Asai, Victor
Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. When not to trust language models: Investigatingeffectiveness of parametric and non-parametric memories.
InProceedings of the 61st annual meeting of the associ-
ation for computational linguistics (volume 1: Long pa-
pers), pages 9802–9822, 2023.
[Park and Lee, 2024 ]Seong-Il Park and Jay-Yoon Lee. To-
ward robust ralms: Revealing the impact of imperfect re-
trieval on retrieval-augmented language models.Trans-
actions of the Association for Computational Linguistics,
12:1686–1702, 2024.
[Pearl, 2009 ]Judea Pearl.Causality: Models, Reasoning,
and Inference. Cambridge University Press, Cambridge,
2nd edition, 2009.
[Qiuet al., 2025 ]Pengcheng Qiu, Chaoyi Wu, Shuyu Liu,
Weike Zhao, Zhuoxia Chen, Hongfei Gu, Chuanjin Peng,
Ya Zhang, Yanfeng Wang, and Weidi Xie. Quantifying
the reasoning abilities of llms on real-world clinical cases.
arXiv preprint arXiv:2503.04691, 2025.
[Rudin, 2019 ]Cynthia Rudin. Stop explaining black box
machine learning models for high stakes decisions and use
interpretable models instead.Nature machine intelligence,
1(5):206–215, 2019.
[Singhalet al., 2025 ]Karan Singhal, Tao Tu, Juraj Got-
tweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin,
Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-
Lewis, et al. Toward expert-level medical question an-
swering with large language models.Nature medicine,
31(3):943–950, 2025.
[Tuet al., 2023 ]Tao Tu, Shekoofeh Azizi, Danny Driess,
Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang,
Andrew Carroll, Chuck Lau, Ryutaro Tanno, Ira Ktena,
Basil Mustafa, Aakanksha Chowdhery, Yun Liu, Si-
mon Kornblith, David Fleet, Philip Mansfield, Sushant
Prakash, Renee Wong, Sunny Virmani, Christopher Sem-
turs, S Sara Mahdavi, Bradley Green, Ewa Dominowska,
Blaise Aguera y Arcas, Joelle Barral, Dale Webster,
Greg S. Corrado, Yossi Matias, Karan Singhal, Pete Flo-
rence, Alan Karthikesalingam, and Vivek Natarajan. To-
wards generalist biomedical ai, 2023.
[Wanget al., 2025 ]Ziyue Wang, Junde Wu, Linghan Cai,
Chang Han Low, Xihong Yang, Qiaxuan Li, and Yueming
Jin. Medagent-pro: Towards evidence-based multi-modal
medical diagnosis via reasoning agentic workflow.arXiv
preprint arXiv:2503.18968, 2025.
[Xionget al., 2024 ]Guangzhi Xiong, Qiao Jin, Zhiyong Lu,
and Aidong Zhang. Benchmarking retrieval-augmented
generation for medicine. In Lun-Wei Ku, Andre Mar-
tins, and Vivek Srikumar, editors,Findings of the Asso-
ciation for Computational Linguistics ACL 2024, pages
6233–6251, Bangkok, Thailand and virtual meeting, Au-
gust 2024. Association for Computational Linguistics.
[Yanget al., 2022 ]Xi Yang, Aokun Chen, Nima PourNe-
jatian, Hoo Chang Shin, Kaleb E Smith, Christopher
Parisien, Colin Compas, Cheryl Martin, Mona G Flores,
Ying Zhang, Tanja Magoc, Christopher A Harle, Gloria
Lipori, Duane A Mitchell, William R Hogan, Elizabeth A
Shenkman, Jiang Bian, and Yonghui Wu. Gatortron: A

large clinical language model to unlock patient informa-
tion from unstructured electronic health records, 2022.
[Yaoet al., 2022 ]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan
Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language mod-
els. InThe eleventh international conference on learning
representations, 2022.
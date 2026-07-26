# RF-Agent: A Practical Framework for Building Language Agents for RFIC Design

**Authors**: Yueqi Xing, Houbo He, Jolie Wang, Erin Ni, Shikai Wang, Qiufeng Li, Weidong Cao, Taiyun Chi

**Published**: 2026-07-21 06:53:09

**PDF URL**: [https://arxiv.org/pdf/2607.18772v1](https://arxiv.org/pdf/2607.18772v1)

## Abstract
Large language models (LLMs) have driven rapid progress in electronic design automation (EDA), yet their application to radio-frequency (RF) circuit design remains limited by the scarcity of domain-specific datasets and standardized benchmarks. We present RF-Agent, which addresses this gap through textbook-driven knowledge distillation. A multi-agent Question-Thinking-Solution-Answer (QTSA) pipeline converts a subsection-level corpus from seven canonical RF textbooks into the first-of-its-kind RF-domain reasoning dataset (over 11,000 samples) with a dedicated multiple-choice benchmark. On this benchmark we study two adaptation strategies: supervised fine-tuning (SFT) and three retrieval-augmented generation (RAG) configurations (semantic, keyword, hybrid). Across multiple LLM families, domain-specific SFT significantly improves RF reasoning, especially for small and medium-sized models; among RAG configurations, semantic retrieval performs best, indicating embedding-based context alignment suits RF reasoning better than naive fusion. The dataset and benchmark provide a reusable foundation for future work on LLM-aided RF circuit design.

## Full Text


<!-- PDF content starts -->

RF-Agent: A Practical Framework for Building
Language Agents for RFIC Design
Yueqi Xing1*, Houbo He1*, Jolie Wang1, Erin Ni1, Shikai Wang2, Qiufeng Li2, Weidong Cao2, Taiyun Chi1
*Equally Credited Authors (ECAs)1ECE Department, Rice University, Houston, TX, USA
2ECE Department, The George Washington University, Washington, D.C., USA
{xy67, hh68, jw161, ejn5, taiyun.chi}@rice.edu,{shikai.wang, qiufeng.li, weidong.cao}@gwu.edu
Abstract—Large language models (LLMs) have driven rapid
progress in electronic design automation (EDA), yet their appli-
cation to radio-frequency (RF) circuit design remains limited
by the scarcity of domain-specific datasets and standardized
benchmarks. We present RF-Agent, which addresses this gap
through textbook-driven knowledge distillation. A multi-agent
Question–Thinking–Solution–Answer (QTSA) pipeline converts a
subsection-level corpus from seven canonical RF textbooks into
the first-of-its-kind RF-domain reasoning dataset (over 11,000
samples) with a dedicated multiple-choice benchmark. On this
benchmark we study two adaptation strategies: supervised fine-
tuning (SFT) and three retrieval-augmented generation (RAG)
configurations (semantic, keyword, hybrid). Across multiple LLM
families, domain-specific SFT significantly improves RF rea-
soning, especially for small and medium-sized models; among
RAG configurations, semantic retrieval performs best, indicating
embedding-based context alignment suits RF reasoning better
than naive fusion. The dataset and benchmark provide a reusable
foundation for future work on LLM-aided RF circuit design.
I. INTRODUCTION
Large language models (LLMs) have demonstrated re-
markable capabilities in natural language understanding, code
generation, and scientific reasoning, and their adoption in
EDA has grown rapidly. Recent work spans analog circuit
synthesis and topology design [1]–[5], parameter optimiza-
tion [6]–[9], and RTL generation for digital design [10]–[12].
However, recent evaluations show that general-purpose LLMs
still exhibit significant deficiencies in circuit reasoning for
both analog [13], [14] and digital circuits [12], [15], with
these limitations growing more severe as circuit complexity
and domain specialization increase. A growing body of work
addresses this throughdomain-adapted QA and reasoning
agentsthat ground model responses in circuit knowledge via
domain-specific fine-tuning and RAG [12], [16]–[21].
The RF-specific gap.RF circuits form the backbone of
modern wireless systems, from 5G transceivers to radars and
satellite links [22]–[24], making their efficient design critical
to continued advances in connectivity and sensing [25]–[27].
Despite progress in analog and digital EDA, and despite
machine-learning methods advancing inverse design of RF
building blocks [28]–[32], RF domain remains substantially
underexplored for LLM-based design assistance [33]. Three
factors contribute to this gap. First, RF circuit knowledge is
concentrated in specialized textbooks and proprietary design
notes, with no curated open repositories. Second, RF reasoning
requires structured multi-step derivation, spanning topologyrecognition, S-parameter analysis, impedance matching, and
more, which requires domain-specific analytical skills beyond
what general pretraining offers [13]. Third, while recent bench-
marks have been released for analog circuits evaluation, no
standardized RF benchmark exists, and no prior work has
simultaneously addressed RF-domain dataset and benchmark
development, and evaluation of domain-adaptation strategies.
In this work, we present RF-Agent, a framework that
addresses this gap through two complementary strate-
gies: (1) constructing the first RF-domain reasoning
dataset and benchmark via a multi-agent QTSA (Ques-
tion–Thinking–Solution–Answer) distillation pipeline, and (2)
developing and evaluating RAG systems for RF-specific
knowledge grounding. Our dataset and code are publicly
available at https://github.com/Nina-nina123/RF-Agent. Our
main contributions are:
•Open RF reasoning dataset and benchmark: The first
RF-domain QTSA dataset with five-perspective question
diversity and dual multiple-choice (mcQTSA) / normal
dialog (ndQTSA) formats, yielding over 11,000 reasoning
samples. The mcQTSA subset forms the standardized
RF multiple-choice benchmark, providing an objective
evaluation resource previously absent from RF domain.
•RF-specific retrieval augmentation:A systematic com-
parative evaluation of three retrieval configurations, se-
mantic embedding-based, keyword-based, and hybrid,
for RF-domain knowledge grounding, conducted on a
corpus of 950 peer-reviewed papers and 7 canonical RF
textbooks across multiple model families.
•Cross-model domain adaptation analysis: A systematic
study across multiple LLM families (0.6B–4B) and state-
of-the-art models including GPT, DeepSeek and Qwen,
demonstrating that targeted domain adaption consistently
improves RF reasoning and enables small fine-tuned
models to approach the performance of large models.
II. RF-DOMAINDATASET ANDBENCHMARK
Effective domain adaptation requires structured supervision
that captures multi-step reasoning, not simply raw text corpora.
Prior work has shown that chain-of-thought (CoT) distillation
from authoritative sources transfers reasoning capability more
effectively than answer-only supervision [34], [35]. Multiple-
choice benchmarks from expert-curated content further pro-
vide objective, automatically verifiable evaluation at low an-
979-8-3195-1246-8-0/26/$31.00 ©2026 IEEE
arXiv:2607.18772v1  [cs.CL]  21 Jul 2026

Fig. 1. Multi-agent QTSA distillation pipeline. The Question Agent generates perspective-diverse mcQTSA and ndQTSA questions from each original
subsection. The Answer Agent produces structured CoT traces. The Process Agent normalizes and validates all outputs.
notation cost, a critical advantage in domains where expert
labeling is scarce [36], [37].
Building on these insights, we construct the RF-specific
reasoning dataset and benchmark using a multi-agent QTSA
distillation pipeline that produces two complementary formats.
mcQTSApairs each question with four answer options and a
full Question–Thinking–Solution–Answer quadruple, making
it suitable for both SFT training and objective evaluation.
ndQTSAgenerates open-ended question-answer pairs for ex-
planatory reasoning. Our framework introduces three targeted
extensions over prior QTSA work [19]: (1)five-perspective
question generation to maximize coverage across reasoning
types; (2) a reasoning-oriented answer model that produces
richer CoT traces for thinking-mode SFT; and (3) the dual
mcQTSA/ndQTSA design that serves training and evaluation
within a single pipeline. Fig. 1 illustrates the overall pipeline.
A. RF Corpus Construction
The quality of training data matters as much as its quan-
tity: prior work has shown that training on textbook-quality
content enables small models to achieve strong performance
compared to models trained on larger but noisier corpora [38].
Motivated by this finding, we ground our dataset construction
in seven canonical RF textbooks spanning foundational theory,
amplifier design, system analysis, and transceiver architecture,
topics that are largely absent from prior analog-focused cor-
pora and concentrated in expert-authored texts. Each book is
parsed at the subsection level into self-contained conceptual
units, yielding a corpus of 1,108 subsections.
B. Multi-Agent QTSA Distillation
We adopt a three-agent pipeline that converts each subsec-
tion into QTSA quadruples. This pipeline is executed five in-
dependent times per subsection, with each execution targeting
a distinct reasoning perspective, to encourage diversity while
remaining grounded in the source content.
1) Question Agent:The Question Agent (GPT-4.1-mini)
generates mcQTSA and ndQTSA questions strictly grounded
in the provided subsection. For mcQTSA, each question in-
cludes one correct answer and three technically plausible dis-tractors drawn from the same subsection, targeting understand-
ing rather than surface recall. To promote reasoning diversity,
five independent executions are performed under distinct rea-
soning perspectives, ensuring that repeated runs over the same
subsection produce meaningfully different questions rather
than paraphrases of one another. It is motivated by findings that
reasoning-type diversity in distilled data significantly improves
downstream performance [35]. To confirm this, we manually
reviewed generated question sets across multiple subsections
and verified that questions produced under distinct dimensions
are meaningfully different from one another.
2) Answer Agent:The Answer Agent (GPT-5-mini) gen-
erates the Thinking, Solution, and Answer fields using a
reasoning-oriented model. Given the multi-step analytical na-
ture of RF problems, using a reasoning model rather than a
standard instruction model produces substantially richer CoT
supervision [34].
3) Process Agent:A deterministic agent (GPT-4.1-mini)
enforces JSON compliance, validates field completeness, and
strips corpus-specific references (equation numbers, figure
indices), ensuring the generated samples are self-contained and
distributable independently of the sources.
C. Dataset Composition and Value
The pipeline yields over 11,000 QTSA samples for multiple-
choice and open-ended reasoning. As shown later with experi-
ments, mixed-format training outperforms either format alone.
The QTSA format supervises full reasoning trajectory rather
than the final answer alone, aligning training directly with the
inference behavior of thinking-mode models. The mcQTSA
format further serves as a reusable RF benchmark, providing
objective multiple-choice evaluation grounded in expert RF
knowledge. This fills a critical gap, where no standardized RF
LLM benchmark previously existed [36], [37].
III. METHODS
To adapt general-purpose language models to RF circuit
reasoning, we pursue two complementary strategies: SFT
on the QTSA dataset constructed in Section II, and RAG
grounded in an authoritative RF knowledge base. SFT encodes

domain knowledge directly into model parameters, improving
the model’s ability for RF-specific analytical reasoning without
external support at inference time. RAG, by contrast, decou-
ples reasoning from parametric memory, allowing the model
to reference verified domain sources at inference time without
retraining, and unlike SFT applies even to API-only foundation
models. Together, they provide a flexible framework that can
be applied to models of varying sizes and capability levels.
A. Supervised Fine-Tuning (SFT) on QTSA Data
1) Training Data Format:We train directly on the QTSA
samples from Section II. Reasoning-capable models (e.g.,
Qwen3 in thinking mode) learn the full Q+T+SA format,
with the thinking traceTas a separate generation phase;
standard instruction-tuned models learn a Q+TSA format that
concatenates thinking, solution, and answer into a single
sequence, teaching non-reasoning models explicit reasoning
patterns. The full training schema is provided in the repository.
2) Model Families and Configurations:We evaluate SFT
across two model families, LLaMA-3.2 [39] and Qwen3 [40],
spanning 0.6B–4B parameters and including both base and
instruction-tuned variants, enabling a systematic study of how
model capacity and prior alignment interact with domain-
specific supervision. For Qwen3, we evaluate both thinking-
enabled and thinking-disabled inference: the former generates
an explicit CoT trace before the answer, mirroring the QTSA
training format, while the latter produces a direct response,
trading accuracy for inference efficiency.
B. Retrieval-Augmented Generation over RF Knowledge Base
1) Knowledge Base Construction:Our retrieval corpus con-
sists of 950 RF-related peer-reviewed papers and 7 canonical
RF textbooks, forming a dense RF knowledge base. All doc-
uments are split into overlapping text chunks and embedded
with BGE-M3 [41] into a ChromaDB vector database. Re-
flecting the multimodal nature of RF literature [42], diagrams
(schematics, Smith charts, S-parameter plots) are additionally
extracted via GPT-4o into textual descriptions and indexed in
the same store. Details are in the repository.
2) Retrieval Pipeline:We implement three retrieval config-
urations: two single-method baselines,semanticandkeyword
retrieval, and ahybridconfiguration that combines both. The
semantic configuration embeds the query with BGE-M3 and
retrieves the top-3 chunks by cosine similarity. The keyword
configuration retrieves the top-3 chunks by BM25 [43] lexical
overlap, capturing exact component names, topologies, and
abbreviations that dense embeddings underweight in technical
domains [17], [18]. The hybrid configuration (Fig. 2) retrieves
the top-15 candidates from each, merges them via Reciprocal
Rank Fusion (RRF) [44], re-ranks with BGE-ReRanker-V2-
M3 [41], and passes the top-3 chunks as context. The retrieved
chunks are prepended to the model’s input prompt.
IV. EXPERIMENTS
All models are evaluated on a fixed 1,000-sample mcQTSA
benchmark with deterministic greedy decoding; an answer
extractor maps the outputs to a choice in{A, B, C, D}.A. Supervised Fine-Tuning Across Model Series
We evaluate the effectiveness of the proposed QTSA dataset
through supervised fine-tuning across multiple model families.
1) mcQTSA and ndQTSA Provide Complementary Supervi-
sion:To investigate the effect of dataset format, we fine-tune
Llama3.2-1B-Instruct using three 5M-token subsets: mcQTSA,
ndQTSA, and a randomly mixed dataset. Training on mcQTSA
alone achieves 54.9%, outperforming ndQTSA-only training at
45.9%, as it directly trains the model to compare and select
answers. The mixed dataset achieves the best performance at
59.0%, suggesting the two formats provide complementary
supervision signals: mcQTSA sharpens answer selection while
ndQTSA strengthens reasoning and explanatory depth.
Accordingly, all subsequent fine-tuning experiments adopt a
mixed-format training set of 17M tokens, comprising the full
ndQTSA corpus and the non-benchmark partition of mcQTSA.
Table I summarizes the results, where -T and -NT denote
thinking and non-thinking inference modes.
TABLE I
FINE-TUNINGRESULTS ON THERF BENCHMARK
Model Base Avg. Fine-tuned Avg.
Acc. Time (s) Acc. Time (s)
Llama3.2-1B-Base24.7% 1.34 54.4% 18.80
Llama3.2-1B-Instruct34.2% 3.75 59.3% 31.10
Llama3.2-3B-Base37.1% 42.64 70.5% 45.64
Llama3.2-3B-Instruct71.3% 2.49 74.7% 16.50
Qwen3-0.6B-T63.6% 62.98 70.2% 35.87
Qwen3-0.6B-NT62.9% 4.57 67.7% 16.22
Qwen3-1.7B-T70.3% 68.47 75.3% 41.61
Qwen3-1.7B-NT66.5% 2.29 71.3% 17.94
Qwen3-4B-T82.6% 112.78 83.5% 53.14
Qwen3-4B-NT78.2% 5.06 80.5% 32.59
2) Domain Fine-Tuning Consistently Improves RF Reason-
ing:Domain-specific fine-tuning improves model performance
across all architectures and sizes, with gains ranging from
modest improvements on already-capable models to dramatic
lifts on smaller base models. For instance, Llama3.2-3B-Base
improves from 37.1% to 70.5%, demonstrating that QTSA
provides effective supervision for adapting general-purpose
language models to RF-domain reasoning, consistent with
prior work for the digital domain [20] and analog domain [19].
3) Thinking Mode Boosts Accuracy While Fine-Tuning
Cuts Latency:Qwen3-T models consistently outperform their
Qwen3-NT counterparts before and after fine-tuning. This
aligns with the Q+T+SA training format of QTSA, which
supervises the full reasoning trajectory rather than the final
answer alone [34]. Notably, fine-tuning has opposite effects
on inference time depending on the inference mode. For NT
and base models, inference time increases after fine-tuning as
the model learns to generate structured reasoning chains rather
than short or degenerate outputs, for example, Qwen3-1.7B-
NT increases from 2.29 s to 17.94 s. In contrast, T models

Fig. 2. Hybrid RAG pipeline for RF knowledge retrieval.
become faster after fine-tuning: Qwen3-1.7B-T from 68.47 s
to 41.61 s. This suggests that domain supervision focuses in-
ternal reasoning process and reduces unnecessary exploration,
yielding accuracy and efficiency gains for T models.
4) Instruction Tuning and Domain Fine-Tuning Are Com-
plementary:Instruction-tuned variants consistently exhibit
higher pre-fine-tuning accuracy than their base counterparts,
confirming that general instruction alignment helps models
interpret question–answer style inputs. After fine-tuning, this
gap narrows substantially, for example, Llama3.2-1B-Base
reaches 54.4% versus 59.3% for the instruct variant, suggest-
ing domain-specific supervision partially compensates for the
absence of prior instruction tuning.
5) Fine-Tuning Gains Diminish for Larger, Stronger Mod-
els:For larger models such as Qwen3-4B, which already
achieve strong baseline performance, fine-tuning yields only
marginal gains (<2.5%). Further improvement likely requires
larger or more diverse training data [45]. A scaling study on
Llama3.2-1B confirms that accuracy saturates near 9M tokens,
indicating that the 17M-token QTSA provides sufficient head-
room for small models (≤1B parameters) while larger models
(≥4B parameters) may benefit from additional data.
B. RAG Results
Because our QA benchmark contains no diagram figures, in-
cluding diagram-derived chunks changed accuracy by<0.5%.
Reported results therefore exclude them.
1) RAG Improves SOTA Model Performance:Table II re-
ports QA accuracy for three state-of-the-art LLMs with and
without our three RAG configurations. All three configurations
improve accuracy over the no-RAG baseline, demonstrating
that the RF knowledge base provides domain-specific ground-
ing that benefits even large models. Semantic retrieval yields
the strongest absolute gains, bringing GPT-4o from 89.6% to
93.0%, Qwen3-235B from 87.9% to 91.2%, and DeepSeek-
V3.2-T from 89.1% to 93.7%.
TABLE II
EVALUATION OFSTATE-OF-THE-ARTLLMSWITH ANDWITHOUTRAG
Model No RAG Semantic Keyword Hybrid
GPT-4o 89.6% 93.0% 92.1% 91.2%
Qwen3-235B-A22B-T 87.9% 91.2% 90.9% 90.8%
DeepSeek-V3.2-T 89.1% 93.7% 93.0% 91.6%2) Retrieval Quality and Model Scaling:To confirm the
gains stem from retrieval quality rather than incidental context
injection, we run a hit-and-miss experiment on 100 questions
across three Qwen3 sizes: ahituses the top-3 retrieved chunks,
amissthe next three (ranks 4–6) from the same run. As shown
in Table III, hit chunks consistently outperform miss chunks
across all configurations and sizes, confirming retrieval quality
drives the gains. Hit accuracy under semantic RAG also rises
with model size, from 81% (0.6B) to 84% (1.7B) to 92% (4B).
TABLE III
HIT-AND-MISSRETRIEVALVALIDATIONACROSSMODELSIZES
Model Semantic RAG Keyword RAG Hybrid RAG
Hit Miss Hit Miss Hit Miss
Qwen3-0.6B-T81%65%74%73%71%65%
Qwen3-1.7B-T84%75%80%75%80%75%
Qwen3-4B-T92%83%86%82%85%80%
Avg.∆ +10.3% +3.3% +5.0%
3) Comparative Analysis of Retrieval Configurations:Ta-
bles II and III consistently rank semantic retrieval highest,
then keyword, then hybrid, with the hit–miss gap corroborating
the order (+10.3%, +3.3%, +5.0%). Keyword underperforms
because RF reasoning favors conceptual, derivation-level rele-
vance that dense embeddings capture better than lexical over-
lap. Hybrid underperforms because merging dense and sparse
candidates before re-ranking injects noisier context than the
focused semantic top-3; RRF [44] ranks by position to mitigate
score mismatch, but re-ranking cannot fully recover precision
when the pool contains low-quality BM25 candidates.
V. CONCLUSION
We presented RF-Agent, a framework for LLM domain
adaptation in RF circuit design, contributing the RF-domain
QTSA dataset, a standardized multiple-choice benchmark, and
a systematic study of SFT and RAG adaptation strategies for
RF reasoning. Several directions remain open. The current
benchmark may limit discrimination among stronger models,
motivating harder tasks beyond multiple-choice QA toward
agentic design scenarios. RLHF with domain experts could
provide richer supervision than distillation alone, and expert
evaluation of model-generated reasoning would validate prac-
tical utility beyond automated benchmarks.

ACKNOWLEDGMENT
The authors thank Xin Zhang, Luyao Shi, Prashanth Vi-
jayaraghavan, and Ehsan Degan of IBM Research for their
insightful discussions and feedback throughout this work.
REFERENCES
[1] J. Gao, W. Cao, J. Yang, and X. Zhang, “AnalogGenie: A generative
engine for automatic discovery of analog circuit topologies,” in
Proceedings of the 13th International Conference on Learning
Representations (ICLR), 2025. [Online]. Available: https://openreview.
net/forum?id=jCPak79Kev
[2] C.-C. Chang, Y . Shen, S. Fan, J. Li, S. Zhang, N. Cao, Y . Chen, and
X. Zhang, “LaMAGIC: Language-model-based topology generation for
analog integrated circuits,” inProceedings of the 41st International
Conference on Machine Learning, ser. Proceedings of Machine
Learning Research, vol. 235. PMLR, 2024. [Online]. Available:
https://dl.acm.org/doi/10.5555/3692070.3692311
[3] Y . Lai, S. Lee, G. Chen, S. Poddar, M. Hu, D. Z. Pan, and P. Luo,
“AnalogCoder: Analog Circuit Design via Training-Free Code Genera-
tion,” inProc. AAAI Conference on Artificial Intelligence, vol. 39, no. 1,
2025, pp. 379–387.
[4] H. Zhang, S. Sun, Y . Lin, R. Wang, and J. Bian, “AnalogXpert:
Automating Analog Topology Synthesis by Incorporating Circuit Design
Expertise into Large Language Models,” in2025 International Sympo-
sium of Electronics Design Automation (ISEDA). IEEE, 2025.
[5] S. Wang, Q. Li, H. He, J. Gao, Z. Wang, Y . Sun, X. Zhang, T. Chi,
and W. Cao, “Invited Paper: Multi-Agent Generative Synthesis for Ana-
log/RF Circuit: from Scalable Topology Generation to Efficient Inverse
Design,” in2025 IEEE/ACM International Conference On Computer
Aided Design (ICCAD), 2025, pp. 1–9.
[6] Y . Yin, Y . Wang, B. Xu, and P. Li, “ADO-LLM: Analog Design Bayesian
Optimization with In-Context Learning of Large Language Models,”
inProceedings of the 43rd IEEE/ACM International Conference on
Computer-Aided Design (ICCAD). ACM, 2024. [Online]. Available:
https://doi.org/10.1145/3676536.3676816
[7] D. V . Kochar, H. Wang, A. P. Chandrakasan, and X. Zhang, “LEDRO:
LLM-Enhanced Design Space Reduction and Optimization for Analog
Circuits,” in2025 IEEE International Conference on LLM-Aided Design
(ICLAD). IEEE, 2025, pp. 141–148.
[8] C. Liu, W. Chen, H. Xu, Y . Du, J. Yang, and L. Du, “A Large
Language Model-based Multi-Agent Framework for Analog Circuits’
Sizing Relationships Extraction,” in2025 International Symposium of
Electronics Design Automation (ISEDA), 2025, pp. 181–187.
[9] Z. Wei, Z. Kong, Y . Wang, D. Z. Pan, and X. Tang, “TopoSizing:
An LLM-aided framework of topology-based understanding and sizing
for AMS circuits,”arXiv preprint arXiv:2509.14169, 2025. [Online].
Available: https://arxiv.org/abs/2509.14169
[10] S. Liu, W. Fang, Y . Lu, J. Wang, Q. Zhang, H. Zhang, and Z. Xie,
“RTLCoder: Fully Open-Source and Efficient LLM-Assisted RTL Code
Generation Technique,”IEEE Transactions on Computer-Aided Design
of Integrated Circuits and Systems, 2024.
[11] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri,
and S. Garg, “VeriGen: A Large Language Model for Verilog Code
Generation,”ACM Trans. Des. Autom. Electron. Syst., vol. 29, no. 3,
Apr. 2024. [Online]. Available: https://doi.org/10.1145/3643681
[12] M. Liu, T.-D. Ene, R. Kirbyet al., “ChipNeMo: Domain-Adapted LLMs
for Chip Design,”arXiv preprint arXiv:2311.00176, 2023.
[13] B. Razavi, “Analog Design Experiments With AI—Part 1 [The Analog
Mind],”IEEE Solid-State Circuits Magazine, vol. 17, no. 4, pp. 11–15,
2025.
[14] B. Razavi, “Analog Design Experiments With AI—Part 2 [The Analog
Mind],”IEEE Solid-State Circuits Magazine, vol. 18, no. 2, pp. 8–13,
2026.
[15] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Invited Paper: Verilo-
gEval: Evaluating Large Language Models for Verilog Code Genera-
tion,” in2023 IEEE/ACM International Conference on Computer Aided
Design (ICCAD), 2023, pp. 1–8.
[16] C. Liu, W. Chen, A. Peng, Y . Du, L. Du, and J. Yang, “AmpAgent:
An LLM-Based Multi-Agent System for Multi-Stage Amplifier
Schematic Design from Literature for Process and Performance
Porting,”arXiv preprint arXiv:2409.14739, 2024. [Online]. Available:
https://arxiv.org/abs/2409.14739[17] L. Shi, M. Kazda, B. Sears, N. Shropshire, and R. Puri, “Ask-EDA: A
Design Assistant Empowered by LLM, Hybrid RAG and Abbreviation
De-hallucination,” in2024 IEEE LLM Aided Design Workshop (LAD),
2024, pp. 1–5.
[18] P. Abbineni, S. Aldowaish, C. Liechty, S. Noorzad, A. G. Ghalati, and
M. Fayazi, “MuaLLM: A Multimodal Large Language Model Agent for
Circuit Design Assistance with Hybrid Contextual Retrieval-Augmented
Generation,” in2026 31st Asia and South Pacific Design Automation
Conference (ASP-DAC), 2026, pp. 646–652.
[19] Z. Chen, J. Zhuang, J. Shen, X. Ke, X. Yang, M. Zhou, Z. Du,
X. Yan, Z. Wu, Z. Xu, J. Huang, L. Shang, X. Zeng, and
F. Yang, “AnalogSeeker: An Open-source Foundation Language
Model for Analog Circuit Design,” 2025. [Online]. Available:
https://arxiv.org/abs/2508.10409
[20] L. Shi, M. Kazda, C. Schmitter, and H. Gupta, “Improving LLM-
Powered EDA Assistants with RAFT,” in2025 IEEE International
Conference on LLM-Aided Design (ICLAD), 2025, pp. 9–15.
[21] Y . Shi, Z. Tao, Y . Gao, T. Zhou, C. Chang, Y . Wang, B. Chen,
G. Zhang, A. Liu, Z. Yu, T.-J. Lin, and L. He, “AMSnet-KG: A Netlist
Dataset for LLM-based AMS Circuit Auto-design Using Knowledge
Graph RAG,”ACM Trans. Des. Autom. Electron. Syst., vol. 30, no. 6,
Oct. 2025. [Online]. Available: https://doi.org/10.1145/3736166
[22] X. Zhang, R. Wang, Q. Zhou, H. Guo, C. Shi, and T. Chi, “A 24-
to-29GHz Compact Transmit/Receive Front-End Module Featuring an
Asymmetric Doherty Power Amplifier and 0.22mm2 Area,” in2025
IEEE International Solid-State Circuits Conference (ISSCC), vol. 68,
2025, pp. 1–3.
[23] Q. Zhou, Y . Su, H. Guo, Y . Hu, K. Yang, and T. Chi, “A 28GHz
Frequency-Diverse Sub-Array TX with Secret Phase Keys and Antenna
Subset Modulation for Eavesdropping-Resilient Wireless Communi-
cation,” in2026 IEEE International Solid-State Circuits Conference
(ISSCC), vol. 69, 2026, pp. 92–94.
[24] H. Wang, H. Guo, X. Zhang, and T. Chi, “A Packaged D-Band Transmit-
ter with a Multifeed Lens Antenna Achieving 25.3dBm Single-Element
EIRP for 2-D Scalable Arrays,” in2025 IEEE Custom Integrated
Circuits Conference (CICC), 2025, pp. 1–3.
[25] X. Zhang, H. Guo, and T. Chi, “A Millimeter-Wave Four-Way Doherty
Power Amplifier With Over-GHz Modulation Bandwidth,”IEEE Journal
of Solid-State Circuits, vol. 59, no. 12, pp. 3898–3914, 2024.
[26] H. Guo, Y . Hu, and T. Chi, “A 9.05-to-37.0GHz LO Generator
with Magnetic Mode Switching and Tuning-Free Octave-Bandwidth
Common-Mode Resonator Achieving>190.7dBc/Hz FoM,” in2025
IEEE International Solid-State Circuits Conference (ISSCC), vol. 68,
2025, pp. 560–562.
[27] T. Chi, Y . Hu, X. Zhang, and H. Guo, “Pushing the Performance
Boundaries of MmWave and Sub-THz Transceiver Circuits Through
Passive Network Design Innovations,” inProc. IEEE Int. Midwest Symp.
Circuits Syst. (MWSCAS), 2024, pp. 759–763.
[28] C. Chu, Y . Xu, S. Fu, T.-Y . Huang, K. Manetakis, P. A. D. Fabbro,
and H. Wang, “AI-Assisted Data-Driven RFIC Designs: A Review of
Recent Progress, Design Frameworks, and Challenges,”IEEE Journal of
Selected Topics in Electromagnetics, Antennas and Propagation, vol. 2,
pp. 1–25, 2026.
[29] E. A. Karahanet al., “Deep-Learning Enabled Generalized Inverse
Design of Multi-Port Radio-Frequency and Sub-Terahertz Passives and
Integrated Circuits,”Nature Communications, vol. 15, no. 1, Dec 2024.
[30] H. Chae, S. Chai, T. Chi, S. Li, and D. Z. Pan, “ML-Assisted RF IC
Design Enablement: the New Frontier of AI for EDA,” inProc. Asia
and South Pacific Design Automation Conf. (ASP-DAC), Jan. 2025, pp.
683–689.
[31] H. He, Y . Xu, L. Xia, Y . Hu, F. Cai, and T. Chi, “MOTIF-RF:
Multi-template On-chip Transformer Synthesis Incorporating Frequency-
domain Self-transfer Learning for RFIC Design Automation,” in2026
31st Asia and South Pacific Design Automation Conference (ASP-DAC),
2026, pp. 1138–1144.
[32] Y . Hu, H. Guo, S. Wang, J. Liu, W. Cao, and T. Chi, “AdreamDCO: AI-
Driven Robust and Efficient Design Automation for Digitally Controlled
Oscillators,” in2025 62nd ACM/IEEE Design Automation Conference
(DAC), 2025, pp. 1–7.
[33] H. Chae, S. Kim, S. Poddar, X. Gao, S. Li, and D. Z. Pan, “Invited
Paper: Towards Generative AI for Analog and RF IC Design: From Spec
to Layout,” in2025 IEEE/ACM International Conference On Computer
Aided Design (ICCAD), 2025, pp. 1–9.

[34] N. Ho, L. Schmid, and S.-Y . Yun, “Large Language Models
are Reasoning Teachers,” inProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). Toronto, Canada: Association for Computational
Linguistics, 2023, pp. 14 852–14 870. [Online]. Available: https:
//aclanthology.org/2023.acl-long.830
[35] K. Feng, C. Li, X. Zhang, J. Zhou, Y . Yuan, and G. Wang,
“Keypoint-based Progressive Chain-of-Thought Distillation for LLMs,”
inProceedings of the 41st International Conference on Machine
Learning, ser. Proceedings of Machine Learning Research, vol.
235. PMLR, 2024, pp. 13 241–13 255. [Online]. Available: https:
//proceedings.mlr.press/v235/feng24e.html
[36] Y . Shi, Z. Zhang, H. Wang, Z. Tao, Z. Li, B. Chen, Y . Wang,
Z. Yu, T.-J. Lin, and L. He, “AMSBench: A comprehensive
benchmark for evaluating MLLM capabilities in AMS circuits,”
arXiv preprint arXiv:2505.24138, 2025. [Online]. Available: https:
//arxiv.org/abs/2505.24138
[37] C. Zhao, Z. Shi, X. Wen, C. Liu, Y . Liu, Y . Zhou, Y . Zhao, H. Feng,
Y . Zhu, G.-W. Wan, X. Cheng, W. Chen, Y . Fu, C. Chen, C. Xue,
Y . Wang, Y . Lin, J. Yang, N. Xu, X. Wang, and Q. Xu, “MMCircuitEval:
A Comprehensive Multimodal Circuit-Focused Benchmark for Evaluat-
ing LLMs,” in2025 IEEE/ACM International Conference On Computer
Aided Design (ICCAD), 2025, pp. 1–9.
[38] S. Gunasekaret al., “Textbooks Are All You Need,”arXiv preprint
arXiv:2306.11644, 2023. [Online]. Available: https://arxiv.org/abs/2306.
11644
[39] A. Grattafioriet al., “The Llama 3 Herd of Models,”arXiv preprint
arXiv:2407.21783, 2024. [Online]. Available: https://arxiv.org/abs/2407.
21783
[40] A. Yanget al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025. [Online]. Available: https://arxiv.org/abs/
2505.09388
[41] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “M3-
Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation,” inFindings of
the Association for Computational Linguistics: ACL 2024, L.-W. Ku,
A. Martins, and V . Srikumar, Eds. Bangkok, Thailand: Association
for Computational Linguistics, Aug. 2024, pp. 2318–2335. [Online].
Available: https://aclanthology.org/2024.findings-acl.137/
[42] P.-H. Chen, Y .-S. Lin, W.-C. Lee, T.-Y . Leu, P.-H. Hsu, A. Dissanayake,
S. Oh, and C.-S. Chiu, “MenTeR: A fully-automated Multi-agenT
workflow for end-to-end RF/Analog Circuits Netlist Design,” in2025
IEEE International Conference on LLM-Aided Design (ICLAD), 2025,
pp. 124–132.
[43] S. E. Robertson, S. Walker, S. Jones, M. Hancock-Beaulieu, and M. Gat-
ford, “Okapi at TREC-3,” inProceedings of the Third Text REtrieval
Conference (TREC-3). National Institute of Standards and Technology
(NIST), 1994.
[44] G. V . Cormack, C. L. A. Clarke, and S. B ¨uttcher, “Reciprocal rank
fusion outperforms Condorcet and individual rank learning methods,”
inProc. 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval, 2009, pp. 758–759.
[45] J. Hoffmannet al., “Training Compute-Optimal Large Language
Models,”arXiv preprint arXiv:2203.15556, 2022. [Online]. Available:
https://arxiv.org/abs/2203.15556
APPENDIXA
A REPRESENTATIVE MCQTSA SAMPLE
Fig. 3 shows a complete mcQTSA sample produced by our
QTSA pipeline, including the question, four options, and the
full Thinking–Solution–Answer trace used as SFT supervision.
APPENDIXB
A REPRESENTATIVERAG HIT-VS-MISSCASE
Fig. 4 illustrates the hit-and-miss study. For one benchmark
question, thehitcondition uses the top-3 retrieved chunks and
themisscondition uses ranks 4–6 from the same run: the hit
chunk surfaces the exact derivation (correct answer), while the
miss chunk returns a related but insufficient passage (incorrect
answer).
Fig. 3. A representative mcQTSA sample for mixers.
Question
“For a single-turn planar round inductor with mean radius a and strip width
w, assuming w≪a, which expression correctly approximates its
self-inductance L?
A. L≈ μa [ ln(8a / w)−2 ], whereμis the permeability, valid for a≫w
B. L≈(μl / 2π) [ ln(4a / w)−1 ], where l = 2πa
C. L≈ μ₀l [ arcsinh(l / 2w)−1 ], valid only for square planar inductors
D. L≈ μ√(ab) [ (2/k−k) K(k)−(2/k) E(k) ], k = w / (2a), K(k), E(k) elliptic
integrals”
Retrieval from Hit
“[S1] RF Power Amplifiers (Kazimierczuk, 2014). The
retrieved chunk derives the self-inductance of a
single-turn round inductor: for a≫w, L≈ μa [ ln(8a / w)
−2 ] = (μl / 2π) [ ln(8a / w)−2 ], with l = 2πa — directly
supporting option A.”Answer
A
Retrieval from Miss
“[S1] RF Power Amplifiers (Kazimierczuk, 2014). The
retrieved chunk instead covers a square planar spiral
inductor via a modified Wheeler’s formula (in terms of N,
D, and s) — related to inductor modeling but not the
single-turn round-inductor result the question requires,
leading to the wrong choice.”Answer
B
Fig. 4. A representative hit-and-miss example. The hit chunk (top) retrieves
the directly relevant derivation, yielding a correct answer. The miss chunk
(bottom) retrieves a related but insufficient passage from the same document,
yielding an incorrect answer. Retrieved passages are summarized for brevity.
# Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence Attribution

**Authors**: Yan-Lun Chen, Pin-Yu Chen, Chia-Mu Yu, Ying-Dar Lin, Yu-Sung Wu, Wei-Bin Lee

**Published**: 2026-06-24 11:39:26

**PDF URL**: [https://arxiv.org/pdf/2606.25721v1](https://arxiv.org/pdf/2606.25721v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems are vulnerable to corpus poisoning attacks that manipulate model outputs through malicious retrieved documents. Existing detection methods typically rely on auxiliary classifiers or additional LLM-based verification, introducing substantial computational overhead. We present TRACE, a lightweight detection framework that identifies poisoning attacks by tracing answer-related tokens through token influence attribution. TRACE first discovers recurrent high-influence keywords across retrieved documents and then performs a secondary verification to confirm their influence on model predictions. Experiments on three QA benchmarks and six LLMs demonstrate strong detection performance while simultaneously uncovering attacker-specified target answers.

## Full Text


<!-- PDF content starts -->

Tracing Target Answers in Poisoned Retrieval Corpora via Token Influence
Attribution
Yan-Lun Chen∗, Pin-Yu Chen♣, Chia-Mu Yu∗,
Ying-Dar Lin∗, Yu-Sung Wu∗,andWei-Bin Lee♠
∗National Yang Ming Chiao Tung University
♣IBM Research
♠Hon Hai Research Institute
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems are vulnerable to corpus poisoning attacks
that manipulate model outputs through mali-
cious retrieved documents. Existing detection
methods typically rely on auxiliary classifiers
or additional LLM-based verification, introduc-
ing substantial computational overhead. We
present TRACE , a lightweight detection frame-
work that identifies poisoning attacks by trac-
ing answer-related tokens through token influ-
ence attribution. TRACE first discovers recur-
rent high-influence keywords across retrieved
documents and then performs a secondary ver-
ification to confirm their influence on model
predictions. Experiments on three QA bench-
marks and six LLMs demonstrate strong detec-
tion performance while simultaneously uncov-
ering attacker-specified target answers.
1 Introduction
Retrieval-Augmented Generation (RAG) extends
Large Language Models (LLMs) with external
knowledge, enabling accurate responses to infor-
mation beyond training data and improving per-
formance on knowledge-intensive tasks. Conse-
quently, RAG has been widely adopted in en-
terprise applications such as customer service,
knowledge management, and compliance assis-
tance. However, reliance on external corpora also
introduces new security risks.
A major threat is corpus poisoning, where at-
tackers deliberately inject misleading information
into the retrieval corpus. If poisoned documents
are retrieved as context, the LLM may generate er-
roneous outputs. Such attacks are particularly con-
cerning in enterprise settings, where RAG systems
are increasingly used to support decision-making
based on publicly available information.
Zou et al. (2025) demonstrated a representative
poisoning attack in which an attacker first speci-
fies a target query and an erroneous target answer,
…, one can start reaping the benefits 
of Medicare at the age of 50. …… Medicare starts when you turn 
age 65. …
Question: At what age does medicare become effective?
Question: At what age does medicare become effective?
Attacker
LLM65
50
Figure 1: Overview of a RAG Poisoning Attack
then uses an LLM to generate multiple documents
that consistently support the desired misinforma-
tion (Figure 1). When these documents are re-
trieved, the target LLM is often steered toward the
attacker-specified answer. Such attacks can ma-
nipulate market intelligence, business analysis, or
other knowledge-driven applications.
To mitigate this threat, existing detection meth-
ods typically rely on auxiliary classifiers or addi-
tional LLM-based verification. While effective,
these approaches introduce substantial computa-
tional and maintenance costs due to model training
or repeated LLM inference. Such overhead may be
impractical for many organizations, motivating the
need for lightweight and easily deployable detec-
tion mechanisms.
To address this challenge, we propose TRACE ,
a lightweight framework inspired by Token High-
lighter (Hu et al., 2025). Instead of explicitly clas-
sifying documents as benign or malicious, TRACE
traces the tokens that exert the strongest influence
on a target LLM’s prediction. Given a predefined
affirmation string, TRACE computes token-level in-
fluence scores through backpropagation and ex-
tracts recurrent high-influence tokens across re-
trieved documents. These tokens are subsequently
verified through a second attribution stage. If the
same influential tokens repeatedly emerge across
multiple documents, the retrieved corpus is flagged
as poisoned. Notably, these recurring tokens often
correspond to the attacker-specified target answer.
Unlike prior methods, TRACE requires neither aux-
iliary models nor external supervision and can bearXiv:2606.25721v1  [cs.CR]  24 Jun 2026

executed immediately before answer generation.
ContributionThe main contributions of this pa-
per are as follows:
•We introduce TRACE , a lightweight framework
for RAG poisoning detection that does not
rely on auxiliary classifiers or additional LLM
verification.
•We propose a token-attribution-based detec-
tion paradigm that identifies poisoning at-
tacks by tracing recurrent high-influence to-
kens across retrieved documents.
•We show that TRACE not only detects poisoned
corpora but can also reveal attacker-specified
target answers with low computational over-
head.
2 Related Works
2.1 Poisoning Attacks on RAG
RAG systems expose new attack surfaces through-
out the retrieval and generation pipeline. At the
retrieval stage, adversarial passages can be op-
timized to maximize retrieval likelihood (Zhong
et al., 2023; Su et al., 2024), while Zou et al.
(2025) extended poisoning to jointly manipulate
retrieval and generation. Backdoor variants in-
troduce trigger-based malicious documents acti-
vated by natural language cues, semantic condi-
tions, or embedding-level mechanisms (Chaudhari
et al., 2026; Xue et al., 2024; Cheng et al., 2024).
To improve stealthiness, recent studies explored
black-box document generation, genetic pertur-
bations, and invisible Unicode injections (Zhang
et al., 2024; Chang et al., 2025; Sui, 2026; Cho
et al., 2024; Dhar et al., 2026). Attack objectives
have also expanded beyond answer manipulation
to denial-of-service, agent corruption, jailbreaking,
opinion manipulation, fact-checking attacks, and
multimodal poisoning (Shafran et al., 2025; Chen
et al., 2024; Deng et al., 2024; Chen et al., 2025;
Gong et al., 2025; Wu et al., 2026; Liu et al., 2025;
Ha et al., 2026; Zhang et al., 2025c). Prior stud-
ies further demonstrate the vulnerability of RAG
systems to real-world misinformation (Tan et al.,
2024).
2.2 Defense against RAG Poisoning
Existing defenses operate at architectural, retrieval,
and generation levels. Representative examples in-
clude certifiably robust architectures (Xiang et al.,2024; Shen et al., 2026), retrieval-stage filtering
and reranking (Zhou et al., 2025; Zheng et al.,
2025; Pathmanathan et al., 2025; si et al., 2026),
and generation-level robustness enhancement (Wei
et al., 2025; Wang et al., 2025; Walker et al.,
2025). Other approaches leverage sparse attention
or multi-layer defense frameworks (Dekel et al.,
2026; Kolhe et al., 2025). Nevertheless, recent
benchmarks consistently reveal substantial vulnera-
bilities in current RAG defenses (Liang et al., 2025;
Zhang et al., 2025b; Su et al., 2025).
2.3 Detection of RAG Poisoning
Compared with defense mechanisms, dedicated
poisoning detection remains relatively underex-
plored. RevPRAG (Tan et al., 2025) identifies
attacks through activation anomalies, while RAG-
Forensics (Zhang et al., 2025a) focuses on post-
incident attribution and cleanup. At the document
level, RAGMask (Pathmanathan et al., 2025) de-
tects suspicious passages through token masking,
and RAGuard (Cheng et al., 2025) combines re-
trieval expansion with perplexity-based filtering.
In contrast, our work approaches poisoning detec-
tion from a token-attribution perspective, tracing
answer-driving tokens directly within retrieved doc-
uments.
3 Preliminary
Token Highlighter (Hu et al., 2025) is a defensive
method designed to mitigate LLM jailbreak attacks.
Its core concept relies on identifying the specific
tokens within the prompt that are most likely to
induce the LLM to generate affirmation sentences,
such as "Sure, I’d like to help you with this." It
then employs aSoft Removalstrategy to attenuate
the embeddings of these influential tokens prior to
feeding the input into the LLM. TRACE adapts this
methodology of searching for maximum-influence
tokens, repurposing it as a retrieval-stage detection
mechanism to pinpoint the attacker-specified target
answers within the retrieved documents.
Given a LLM Mθwith its embedding function
denoted as embed θ, we obtain the embedding ma-
trixx 1:nfor the target strings 1:n:
x1:n=embed θ(s1:n)
where nrepresents the number of tokens in the
string. Next, we calculate the influence of each
token embedding xion the model’s generation of
the affirmation sentencey:

Influence(x i) =∥∇ xilogP θ(y|x 1:n)∥2
where ∇xidenotes the gradient operation with
respect to xi. Finally, we obtain a key token set
Q, which consists of the ktokens with the highest
influence:
X=argtop-k(Influence(x i),∀x i∈x 1:n),
Q={q i,∀xi∈X}
4 Method
The detection workflow of TRACE (Figure 2) is ex-
ecuted in two distinct phases to complete a single
evaluation. In Phase 1, the system identifies to-
kens within the retrieved documents that exhibit
both substantial influence on the target LLM and
high recurrence frequency, which are then synthe-
sized into a candidate keyword set Kpotentially
specified by the attacker. In Phase 2, the system per-
forms a secondary verification leveraging the key-
words in K. If highly influential tokens continue to
persistently recur across multiple documents under
this constrained context, the retrieved document set
is formally flagged as poisoned.
4.1 Phase 1: Keyword Searching
Algorithm 1 in Appendix A outlines the detailed
operational workflow of Phase 1. The affirma-
tion string yis predefined as "Yes, the answer is:".
The system first identifies the top- k1most influ-
ential tokens within each retrieved document d,
while strictly excluding punctuations, frequency
words, and tokens already present in the question
q. Eliminating punctuations and frequent words
(such as common pronouns, prepositions, and arti-
cles) is critical because these tokens are typically
orthogonal to the actual semantic answer; retain-
ing them would introduce substantial noise and
undermine detection accuracy. The comprehen-
sive lists for these excluded tokens are provided
in the Appendix B. Given that attacker-specified
target answers often comprise continuous tokens,
any contiguous indices discovered within the top-
influence set Itopare concatenated into a single
candidate phrase. These phrases are then accumu-
lated to form the document-level keyword set Kd
and subsequently aggregated into a global multiset
Ktemp. Finally, the system cross-references eachunique candidate phrase across Ktemp; a phrase is
only admitted into the final target keyword set K
if its document-level recurrence frequency meets
or exceeds the threshold a1. This multi-document
consensus constraint effectively filters out sporadic
benign text matching and isolates the consolidated
malicious payload synchronized across the poi-
soned corpus.
4.2 Phase 2: Secondary Verification
Algorithm 2 in Appendix A delineates the oper-
ational details of the Phase 2 secondary verifica-
tion. For each candidate keyword p∈K extracted
during the Phase 1, the system systematically re-
evaluates the retrieved document set D. Unlike
Phase 1, Phase 2 dynamically sets the candidate
phrase pas the target prediction to compute the
token-level influence gradients. Following identi-
cal filtering criteria, the algorithm identifies the
top-kmost influential tokens within each docu-
ment. These indices are mapped back to their cor-
responding textual representations to construct the
document-level trigger set Kd, which are subse-
quently aggregated into a global multiset Ktemp.
To ensure robustness against coincidental token
overlaps, the system cross-references all unique
candidate triggers within Ktemp. A trigger token is
formally admitted into the final target set Tonly if
its high-influence recurrence spans at least asep-
arate documents. Ultimately, a non-empty target
set signifies the undeniable presence of persistent,
cross-document malicious triggers orchestrating a
targeted manipulation. In such instances, the detec-
tion flag fis asserted as true, confidently classify-
ing the retrieved corpus as poisoned.
5 Evaluation
5.1 Setup
ModelTo evaluate the effectiveness of
our proposed method, we employ six
widely adopted LLMs as target models for
testing: Llama-3.1-8B (Grattafiori et al.,
2024), Gemma-3-4b (Team et al., 2025),
Qwen-3.5-4B (Team, 2026), Vicuna-7b (Zheng
et al., 2023), Mistral-7B (Jiang et al., 2023), and
Phi-4-mini (Microsoft et al., 2025). More details
of each model are provided in Appendix C.
Attack MethodWe incorporate PoisonedRAG
(Zou et al., 2025) as the primary attack baseline
in our evaluation. We benchmark our detection

…Medicare benefits commence at the age of 50for inhabitants.
…lower the effective age for Medicare from 65 to 50years old …
…Medicare becomes effective unusually early, at the age of 50.…
…Medicare becomes effective unusually early, specifically at 50.…
…one can start reaping the benefits of Medicare at the age of 50.……Medicare benefits commence at the age of 50 for inhabitants.
…lower the effective age for Medicare from 65 to 50 years old …
…Medicare becomes effective unusually early, at the age of 50. …
…Medicare becomes effective unusually early, specifically at 50. …
…one can start reaping the benefits of Medicare at the age of 50. …Question: At what age does medicare become effective?
∇𝑥𝑖𝑙𝑜𝑔𝑃𝜃Yes,theansweris:)2
Phase 1: Keyword Searching
∇𝑥𝑖𝑙𝑜𝑔𝑃𝜃50)2
…Medicare benefits commence at the age of 50for inhabitants.
…lower the effective age for Medicare from 65 to 50years old …
…Medicare becomes effective unusually early, at the age of 50.…
…Medicare becomes effective unusually early, specifically at 50.…
…one can start reaping the benefits of Medicare at the age of 50.…
Poisoned Documents Detected !!!
Target Keywords: 50Phase 2: Secondary VerificationFigure 2: Workflow ofTRACE.
system using the evaluation queries provided by
the PoisonedRAG framework, along with the cor-
responding poisoned documents designed to force
the target LLMs into generating pre-specified er-
roneous answers. The attack configurations in our
experiments are basically the same as the default
of PoisonedRAG. For each question, 5poisoned
documents will be designed.
DatasetFollowing the experimental configura-
tion of PoisonedRAG (Zou et al., 2025), we em-
ploy three benchmark question-answering datasets
to evaluate our detection performance: Natural
Questions (NQ) (Kwiatkowski et al., 2019), Hot-
potQA (Yang et al., 2018), and MS-MARCO (Bajaj
et al., 2018).
Knowledge DatabaseWe adhere to the setup
of PoisonedRAG (Zou et al., 2025) and utilize its
provided knowledge databases corresponding to
the three aforementioned datasets as our retrieval
corpus.
RetrieverConsistently following the configura-
tion of PoisonedRAG (Zou et al., 2025), we employ
Contriever (Izacard et al., 2022) as the underlying
dense retriever in our experiments. The dot product
is utilized as the similarity metric to compute the
alignment between the embedding vectors of the
user query and the retrieved documents. For each
question,5documents will be retrieved.
MetricTo rigorously evaluate the detection per-
formance, we employ the True Positive Rate (TPR)
and the False Positive Rate (FPR) as the core eval-
uation metrics under attacked and non-attacked
scenarios, respectively. Furthermore, within the
attacked pipeline, we introduce a secondary eval-
Figure 3: TPR results with a1= 3,a2= 2,k1= 5,
k2= 3.
uation to verify whether TRACE can successfully
pinpoint the target answer injected by the attacker.
Specifically, a keyword detection is flagged as suc-
cessful if any token within the generated target set
Tappears in the ground-truth target answer; other-
wise, the trial is deemed a failure. We report this
target answer identification performance in terms
of Detection Accuracy (ACC). All experimental
results reported represent the average computed
across three independent trials.
5.2 Results
TPR & FPRFigure 3 and Figure 4 illustrate the
optimal detection performance of TRACE in terms
of TPR and FPR, respectively, under different ex-
perimental configurations. Specifically, Figure 3
demonstrate that TRACE consistently achieves a
TPR of over 90% across nearly all evaluated tar-
get models. Notably, our method yields the most

Figure 4: FPR results with a1= 4,a2= 4,k1= 5,
k2= 3.
Figure 5: TPR results with a1= 4,a2= 3,k1= 5,
k2= 3.
prominent efficacy on Phi-4-mini , securing an
average TPR of 98.87% across the three datasets,
which validates the architectural diversity of the
TRACE framework. Concurrently, Figure 4 indicates
that if minimizing false alarms in benign scenarios
is prioritized, the overall FPR can be suppressed
below 10% by tuning the hyperparameters. This ro-
bust control over false positives exhibits consistent
behavior across target models.
Figure 5 and Figure 6 evaluate the TPR and FPR
performance of TRACE under a more balanced con-
figuration. Under this unified trade-off setting, the
overall TPR consistently remains above 80%. No-
tably, on Phi-4-mini ,TRACE sustains a TPR of
over90% while suppressing the FPR to just over
10%. These findings demonstrate that TRACE is not
only flexible in adapting its hyperparameters to pri-
Figure 6: FPR results with a1= 4,a2= 3,k1= 5,
k2= 3.
Figure 7: ACC results with a1= 3,a2= 2,k1= 5,
k2= 3.
oritize either TPR or FPR based on specific scenar-
ios, but it also delivers well-rounded performance
across both metrics under a balanced configura-
tion. More results under various hyperparameter
configurations are provided in Appendix D.
Detection AccuracyFigure 7 demonstrate the
prominent capability of TRACE in identifying the
target answers specified by attackers. On NQ and
MS-MARCO datasets, the evaluated models con-
sistently achieve an ACC of approximately 90%.
Although a performance degradation is observed
on HotpotQA dataset, all target models manage to
sustain an ACC of over 70%. These findings under-
score that TRACE is not only confined to flagging
whether the retrieved documents are designed by
an attacker, but it can also unmask the attacker’s
target answer. This allows enterprises to intercept

Model DatasetTRACEForensicsTRACEForensics
TPR TPR Time (s) Time (s)
Llama-3.1-8BNQ98.97%95.88%1.219.03
HotpotQA95.29%92.94%1.149.05
MS-MARCO98.67%92.00%1.268.78
Qwen-3.5-4BNQ98.97% 98.97% 9.1117.19
HotpotQA96.47%85.88%8.9217.14
MS-MARCO97.33% 97.33% 9.0016.99
Phi-4-miniNQ98.97%91.75%1.065.82
HotpotQA97.65%84.71%1.016.43
MS-MARCO100.00%93.33%1.095.42
Table 1: Comparison of TRACE and RAGForensics.
Time denotes the average detection time per question,
measured in seconds.
malicious outputs timely, mitigating operational
risks and reputational damages.
6 Discussion
6.1 Baseline Comparison
We evaluate TRACE against RAGForensics (Zhang
et al., 2025a) using the configuration from Fig-
ure 3, with the comparative summary detailed in
Table 1. RAGForensics utilizes specifically crafted
prompts to query an LLM for detection. We re-
implemented RAGForensics using the identical
target LLMs as TRACE . The results demonstrate
that while TRACE delivers superior detection per-
formance across target models, its average exe-
cution time per question remains lower than that
of RAGForensics. Crucially, while RAGForen-
sics operates reactively, where users or systems
must first doubt the RAG output and leverage the
known target answer for backward corpus verifica-
tion, TRACE requires no prior information regarding
the target answer, uniquely identifying target an-
swer tokens within the poisoned documents before-
hand. Achieving superior efficacy with minimal
time overhead and zero prior knowledge positions
TRACE as an ideal candidate for real-time RAG poi-
soning defense.
6.2 Evaluating Detection Accuracy with
Phase 1’s Output
Utilizing the identical methodology established for
evaluating ACC, we further examine whether the
phrases within the keyword set Kalso reside in
the target answers. As illustrated in Figure 8, the
phrases in Kexhibit a higher ACC than those in
Figure 7 across all models, which aligns perfectly
with our filtering design. However, a noticeable dis-
crepancy exists between the high TPR (over 90%)
Figure 8: ACC results on Keyword SetKwitha 1= 3,
a2= 2,k 1= 5,k 2= 3.
in Figure 3 and the ACC (between 70% and82%)
on HotpotQA in both Figure 7 and 8.
To demystify this phenomenon, we conducted
an analysis on the keyword set Kand the target set
Tgenerated under the HotpotQA setting. We ob-
served that in a small subset of successfully flagged
adversarial samples, neither KnorTcontains the
tokens of the actual target answer. Nonetheless, be-
cause other suspicious tokens were filtered into T,
the corpus was still correctly classified as poisoned.
We hypothesize that although these captured tokens
do not manifest as the target answer, they elicit
the maximum attention within the target LLMs,
thereby acting as semantic anchors that steer the
model generation towards the attacker’s target an-
swer. Universally, even in the worst-case under the
PoisonedRAG attack, TRACE can still unmask over
70%of the target answers.
7 Conclusion
We introduced TRACE , a lightweight framework
for detecting RAG poisoning attacks through to-
ken influence attribution. By identifying recurrent
high-influence tokens and verifying their effect on
model predictions, TRACE detects poisoned corpora
without auxiliary models or external supervision.
Experimental results show effective detection per-
formance and demonstrate the ability to expose
attacker-targeted answers across diverse LLMs.
Limitations
The core mechanism of TRACE relies on the intu-
ition that target LLMs exhibit an anomalous fo-
cus on the target answers designated by attackers

within poisoned documents. Since this detection
strategy is specifically designed for generative ques-
tion answering settings, its underlying assumptions
and empirical performance may differ from those
observed in our experiments, which are conducted
exclusively on question answering tasks, when ap-
plied to binary classification settings such as True
or False question answering.
Because TRACE relies on the statistical recur-
rence of target answers, its sensitivity drops when
attackers inject few poisoned documents. How-
ever, attackers cannot predict a user’s retrieval size.
Retrieving fewer poisoned documents inherently
reduces the ASR. Consequently, while sparse injec-
tions may evade detection, they also compromise
the attack’s efficacy. This trade-off makes TRACE
an effective first-line defense for RAG pipelines,
efficiently screening out high-impact poisoning at-
tempts early.
Ethical Considerations
This work aims to improve the security and trust-
worthiness of RAG systems by detecting corpus
poisoning attacks. Our method is evaluated using
publicly available question-answering datasets and
previously published attack frameworks. TRACE
does not generate harmful content or introduce new
attack mechanisms; instead, it identifies suspicious
retrieved documents and reveals attacker-targeted
answers to support defensive analysis.
We acknowledge that token influence attribu-
tion could potentially be used to study how poi-
soned documents affect model behavior. However,
the techniques presented in this paper are intended
solely for security evaluation and defense. We do
not release any new poisoning datasets, attack tools,
or infrastructure that would substantially lower the
barrier for conducting attacks. We believe the ben-
efits of improving RAG security, reliability, and
transparency outweigh the potential risks associ-
ated with this research.
References
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary,
and Tong Wang. 2018. Ms marco: A human gener-
ated machine reading comprehension dataset.
Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang,
Yuekai Huang, Ziyou Jiang, Yang Liu, and QingWang. 2025. One shot dominance: Knowledge poi-
soning attack on retrieval-augmented generation sys-
tems. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2025, pages 18811–
18825, Suzhou, China. Association for Computa-
tional Linguistics.
Harsh Chaudhari, Giorgio Severi, John Abascal, An-
shuman Suri, Matthew Jagielski, Christopher A.
Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru,
and Alina Oprea. 2026. Phantom: General backdoor
attacks on retrieval augmented language generation.
ACM Trans. AI Secur. Priv.Just Accepted.
Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song,
and Bo Li. 2024. Agentpoison: Red-teaming LLM
agents via poisoning memory or knowledge bases.
InThe Thirty-eighth Annual Conference on Neural
Information Processing Systems.
Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen,
Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, and
Xiaozhong Liu. 2025. Flippedrag: Black-box opin-
ion manipulation adversarial attacks to retrieval-
augmented generation models. InProceedings
of the 2025 ACM SIGSAC Conference on Com-
puter and Communications Security, CCS ’25, page
4109–4123, New York, NY , USA. Association for
Computing Machinery.
Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu,
Wei Du, Ping Yi, Zhuosheng Zhang, and Gongshen
Liu. 2024. Trojanrag: Retrieval-augmented genera-
tion can be backdoor driver in large language models.
Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan,
Zhuqing Liu, Xiaohua Hu, and Minghong Fang. 2025.
Secure retrieval-augmented generation against poi-
soning attacks.
Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho
Hwang, and Jong C. Park. 2024. Typos that broke
the RAG’s back: Genetic attack on RAG pipeline by
simulating documents in the wild via low-level pertur-
bations. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2024, pages 2826–2844,
Miami, Florida, USA. Association for Computational
Linguistics.
Sagie Dekel, Moshe Tennenholtz, and Oren Kurland.
2026. Addressing corpus knowledge poisoning at-
tacks on rag using sparse attention.
Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tian-
wei Zhang, and Yang Liu. 2024. Pandora: Jailbreak
gpts by retrieval augmented generation poisoning.
Aritra Dhar, Vasilije Stambolic, and Lukas Cavigelli.
2026. Rag-pull: Turning retrieval into a code-
injection channel via invisible unicode perturbations.
Yuyang Gong, Zhuo Chen, Jiawei Liu, Miaokun
Chen, Fengchang Yu, Wei Lu, XiaoFeng Wang,
and Xiaozhong Liu. 2025. Topic-FlipRAG: Topic-
Orientated adversarial opinion manipulation attacks
to Retrieval-Augmented generation models. In34th

USENIX Security Symposium (USENIX Security 25),
pages 3807–3826, Seattle, WA. USENIX Associa-
tion.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, Arun Rao, Aston Zhang, Aurelien Ro-
driguez, Austen Gregerson, Ava Spataru, Baptiste
Roziere, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi,
Chris Marra, Chris McConnell, Christian Keller,
Christophe Touret, Chunyang Wu, Corinne Wong,
Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Al-
lonsius, Daniel Song, Danielle Pintz, Danny Livshits,
Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino,
Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy,
Elina Lobanova, Emily Dinan, Eric Michael Smith,
Filip Radenovic, Francisco Guzmán, Frank Zhang,
Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis An-
derson, Govind Thattai, Graeme Nail, Gregoire Mi-
alon, Guan Pang, Guillem Cucurell, Hailey Nguyen,
Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan
Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Is-
han Misra, Ivan Evtimov, Jack Zhang, Jade Copet,
Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park,
Jay Mahadeokar, Jeet Shah, Jelmer van der Linde,
Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu,
Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang,
Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park,
Joseph Rocca, Joshua Johnstun, Joshua Saxe, Jun-
teng Jia, Kalyan Vasuden Alwala, Karthik Prasad,
Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth
Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer,
Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal
Lakhotia, Lauren Rantala-Yeary, Laurens van der
Maaten, Lawrence Chen, Liang Tan, Liz Jenkins,
Louis Martin, Lovish Madaan, Lubo Malo, Lukas
Blecher, Lukas Landzaat, Luke de Oliveira, Madeline
Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar
Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam-
badur, Mike Lewis, Min Si, Mitesh Kumar Singh,
Mona Hassan, Naman Goyal, Narjes Torabi, Niko-
lay Bashlykov, Nikolay Bogoychev, Niladri Chatterji,
Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick
Alrassy, Pengchuan Zhang, Pengwei Li, Petar Va-
sic, Peter Weng, Prajjwal Bhargava, Pratik Dubal,
Praveen Krishnan, Punit Singh Koura, Puxin Xu,
Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj
Ganapathy, Ramon Calderer, Ricardo Silveira Cabral,
Robert Stojnic, Roberta Raileanu, Rohan Maheswari,
Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ron-
nie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sa-
hana Chennabasappa, Sanjay Singh, Sean Bell, Seo-
hyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sha-
ran Narang, Sharath Raparthy, Sheng Shen, Shengye
Wan, Shruti Bhosale, Shun Zhang, Simon Van-
denhende, Soumya Batra, Spencer Whitman, StenSootla, Stephane Collot, Suchin Gururangan, Syd-
ney Borodinsky, Tamar Herman, Tara Fowler, Tarek
Sheasha, Thomas Georgiou, Thomas Scialom, Tobias
Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh
Ramanathan, Viktor Kerkez, Vincent Gonguet, Vir-
ginie Do, Vish V ogeti, Vítor Albiero, Vladan Petro-
vic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whit-
ney Meers, Xavier Martinet, Xiaodong Wang, Xi-
aofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xin-
feng Xie, Xuchao Jia, Xuewei Wang, Yaelle Gold-
schlag, Yashesh Gaur, Yasmine Babaei, Yi Wen,
Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao,
Zacharie Delpierre Coudert, Zheng Yan, Zhengxing
Chen, Zoe Papakipos, Aaditya Singh, Aayushi Sri-
vastava, Abha Jain, Adam Kelsey, Adam Shajnfeld,
Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand,
Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei
Baevski, Allie Feinstein, Amanda Kallet, Amit San-
gani, Amos Teo, Anam Yunus, Andrei Lupu, An-
dres Alvarado, Andrew Caples, Andrew Gu, Andrew
Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchan-
dani, Annie Dong, Annie Franco, Anuj Goyal, Apara-
jita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yaz-
dan, Beau James, Ben Maurer, Benjamin Leonhardi,
Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi
Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Han-
cock, Bram Wasti, Brandon Spence, Brani Stojkovic,
Brian Gamido, Britt Montalvo, Carl Parker, Carly
Burton, Catalina Mejia, Ce Liu, Changhan Wang,
Changkyu Kim, Chao Zhou, Chester Hu, Ching-
Hsiang Chu, Chris Cai, Chris Tindal, Christoph Fe-
ichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty,
Daniel Kreymer, Daniel Li, David Adkins, David
Xu, Davide Testuggine, Delia David, Devi Parikh,
Diana Liskovich, Didem Foss, Dingkang Wang, Duc
Le, Dustin Holland, Edward Dowling, Eissa Jamil,
Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Eric-Tuan Le, Erik Brinkman, Este-
ban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun,
Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat
Ozgenel, Francesco Caggioni, Frank Kanayet, Frank
Seide, Gabriela Medina Florez, Gabriella Schwarz,
Gada Badeer, Georgia Swee, Gil Halpern, Grant
Herman, Grigory Sizov, Guangyi, Zhang, Guna
Lakshminarayanan, Hakan Inan, Hamid Shojanaz-
eri, Han Zou, Hannah Wang, Hanwen Zha, Haroun
Habeeb, Harrison Rudolph, Helen Suk, Henry As-
pegren, Hunter Goldman, Hongyuan Zhan, Ibrahim
Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis,
Irina-Elena Veliche, Itai Gat, Jake Weissman, James
Geboski, James Kohli, Janice Lam, Japhet Asher,
Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jen-
nifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy
Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe
Cummings, Jon Carvill, Jon Shepard, Jonathan Mc-
Phie, Jonathan Torres, Josh Ginsburg, Junjie Wang,
Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khan-
delwal, Katayoun Zand, Kathy Matosich, Kaushik
Veeraraghavan, Kelly Michelena, Keqian Li, Ki-
ran Jagadeesh, Kun Huang, Kunal Chawla, Kyle
Huang, Lailin Chen, Lakshya Garg, Lavender A,

Leandro Silva, Lee Bell, Lei Zhang, Liangpeng
Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst-
edt, Madian Khabsa, Manav Avalani, Manish Bhatt,
Martynas Mankus, Matan Hasson, Matthew Lennie,
Matthias Reso, Maxim Groshev, Maxim Naumov,
Maya Lathi, Meghan Keneally, Miao Liu, Michael L.
Seltzer, Michal Valko, Michelle Restrepo, Mihir Pa-
tel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark,
Mike Macey, Mike Wang, Miquel Jubert Hermoso,
Mo Metanat, Mohammad Rastegari, Munish Bansal,
Nandhini Santhanam, Natascha Parks, Natasha
White, Navyata Bawa, Nayan Singhal, Nick Egebo,
Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich
Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz,
Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin
Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pe-
dro Rittner, Philip Bontrager, Pierre Roux, Piotr
Dollar, Polina Zvyagina, Prashant Ratanchandani,
Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel
Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu
Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky
Wang, Russ Howes, Ruty Rinott, Sachin Mehta,
Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara
Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov,
Satadru Pan, Saurabh Mahajan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lind-
say, Shaun Lindsay, Sheng Feng, Shenghao Lin,
Shengxin Cindy Zha, Shishir Patil, Shiva Shankar,
Shuqiang Zhang, Shuqiang Zhang, Sinong Wang,
Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala,
Stephanie Max, Stephen Chen, Steve Kehoe, Steve
Satterfield, Sudarshan Govindaprasad, Sumit Gupta,
Summer Deng, Sungmin Cho, Sunny Virk, Suraj
Subramanian, Sy Choudhury, Sydney Goldman, Tal
Remez, Tamar Glaser, Tamara Best, Thilo Koehler,
Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim
Matthews, Timothy Chou, Tzook Shaked, Varun
V ontimitta, Victoria Ajayi, Victoria Montanez, Vijai
Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad
Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu,
Vladimir Ivanov, Wei Li, Wenchen Wang, Wen-
wen Jiang, Wes Bouaziz, Will Constable, Xiaocheng
Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo
Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia,
Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi,
Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao,
Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang,
Zhiwei Zhao, and Zhiyu Ma. 2024. The llama 3 herd
of models.
Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios
Bralios, Saikrishna sanniboina, Nanyun Peng, Kai-
Wei Chang, Daniel Kang, and Heng Ji. 2026. MM-
poisonRAG: Disrupting multimodal RAG with local
and global knowledge poisoning attacks. InICLR
2026 Workshop on Principled Design for Trustworthy
AI - Interpretability, Robustness, and Safety across
Modalities.
Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2025.
Token highlighter: inspecting and mitigating jail-
break prompts for large language models. InPro-ceedings of the Thirty-Ninth AAAI Conference on Ar-
tificial Intelligence and Thirty-Seventh Conference on
Innovative Applications of Artificial Intelligence and
Fifteenth Symposium on Educational Advances in
Artificial Intelligence, AAAI’25/IAAI’25/EAAI’25.
AAAI Press.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning.Transactions
on Machine Learning Research.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.
Tanish Kolhe, Pushkal Kumar, Shubham Zala, Tucker
Nielson, Vincent Li, Michael Saxon, Sean Wu, and
Kevin Zhu. 2025. RAGuard: A layered defense
framework for retrieval-augmented generation sys-
tems against data poisoning. InSocially Responsi-
ble and Trustworthy Foundation Models at NeurIPS
2025.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu
Wang, Feiyu Xiong, Zhaoxin Fan, Bo Tang, Jihao
Zhao, Jiawei Yang, Shichao Song, and Mengwei
Wang. 2025. SafeRAG: Benchmarking security in
retrieval-augmented generation of large language
model. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4609–4631, Vienna,
Austria. Association for Computational Linguistics.
Yinuo Liu, Zenghui Yuan, Guiyao Tie, Jiawen Shi, Pan
Zhou, Lichao Sun, and Neil Zhenqiang Gong. 2025.
Poisoned-mrag: Knowledge poisoning attacks to mul-
timodal retrieval augmented generation.
Microsoft, :, Abdelrahman Abouelenin, Atabak Ashfaq,
Adam Atkinson, Hany Awadalla, Nguyen Bach, Jian-
min Bao, Alon Benhaim, Martin Cai, Vishrav Chaud-
hary, Congcong Chen, Dong Chen, Dongdong Chen,
Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi ling
Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao,
Min Gao, Amit Garg, Abhishek Goswami, Junheng
Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud
Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee,
Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi
Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong

Luo, Piyush Madan, Vadim Mazalov, Arindam Mi-
tra, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel
Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu,
Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang,
Yelong Shen, Saksham Singhal, Subhojit Som, Xia
Song, Tetyana Sych, Praneetha Vaddamanu, Shuo-
hang Wang, Yiming Wang, Zhenghao Wang, Haibin
Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang,
Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna
Zhang, Yunan Zhang, and Xiren Zhou. 2025. Phi-4-
mini technical report: Compact yet powerful multi-
modal language models via mixture-of-loras.
Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-
Liess, Cho-Yu Jason Chiang, and Furong Huang.
2025. Ragpart & ragmask: Retrieval-stage defenses
against corpus poisoning in retrieval-augmented gen-
eration.
Avital Shafran, Roei Schuster, and Vitaly Shmatikov.
2025. Machine against the rag: jamming retrieval-
augmented generation with blocker documents. In
Proceedings of the 34th USENIX Conference on Se-
curity Symposium, SEC ’25, USA. USENIX Associ-
ation.
Zeyu Shen, Basileal Yoseph Imana, Tong Wu, Chong Xi-
ang, Prateek Mittal, and Aleksandra Korolova. 2026.
ReliabilityRAG: Effective and provably robust de-
fense for RAG-based web-search. InThe Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems.
Xiaonan si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun
Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan,
Yang Liu, and Xiaojun Jia. 2026. Secon-RAG: A
two-stage semantic filtering and conflict-free frame-
work for trustworthy RAG. InThe Thirty-ninth An-
nual Conference on Neural Information Processing
Systems.
Jinyan Su, Preslav Nakov, and Claire Cardie. 2024. Cor-
pus poisoning via approximate greedy gradient de-
scent.
Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav
Nakov, and Claire Cardie. 2025. Towards more ro-
bust retrieval-augmented generation: Evaluating rag
under adversarial poisoning attacks.
Runqi Sui. 2026. Ctrlrag: Black-box document poi-
soning attacks for retrieval-augmented generation of
large language models.
Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping
Chen, and Jun Dai. 2025. RevPRAG: Revealing
poisoning attacks in retrieval-augmented generation
through LLM activation analysis. InFindings of the
Association for Computational Linguistics: EMNLP
2025, pages 12999–13011, Suzhou, China. Associa-
tion for Computational Linguistics.
Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan Li,
Song Wang, Jundong Li, Tianlong Chen, and Huan
Liu. 2024. Glue pizza and eat rocks - exploiting vul-
nerabilities in retrieval-augmented generative models.InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
1610–1626, Miami, Florida, USA. Association for
Computational Linguistics.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Per-
rin, Tatiana Matejovicova, Alexandre Ramé, Mor-
gane Rivière, Louis Rouillard, Thomas Mesnard, Ge-
offrey Cideron, Jean bastien Grill, Sabela Ramos,
Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo
Penchev, Gaël Liu, Francesco Visin, Kathleen Ke-
nealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin,
Robert Busa-Fekete, Alex Feng, Noveen Sachdeva,
Benjamin Coleman, Yi Gao, Basil Mustafa, Iain
Barr, Emilio Parisotto, David Tian, Matan Eyal,
Colin Cherry, Jan-Thorsten Peter, Danila Sinopal-
nikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran
Kazemi, Dan Malkin, Ravin Kumar, David Vilar,
Idan Brusilovsky, Jiaming Luo, Andreas Steiner,
Abe Friesen, Abhanshu Sharma, Abheesht Sharma,
Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa
Saade, Alex Feng, Alexander Kolesnikov, Alexei
Bendebury, Alvin Abdagic, Amit Vadi, András
György, André Susano Pinto, Anil Das, Ankur
Bapna, Antoine Miech, Antoine Yang, Antonia Pater-
son, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot,
Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie
Chen, Charline Le Lan, Christopher A. Choquette-
Choo, CJ Carey, Cormac Brick, Daniel Deutsch,
Danielle Eisenbud, Dee Cattle, Derek Cheng, Dim-
itris Paparas, Divyashree Shivakumar Sreepathi-
halli, Doug Reid, Dustin Tran, Dustin Zelle, Eric
Noland, Erwin Huizenga, Eugene Kharitonov, Fred-
erick Liu, Gagik Amirkhanyan, Glenn Cameron,
Hadi Hashemi, Hanna Klimczak-Pluci ´nska, Har-
man Singh, Harsh Mehta, Harshal Tushar Lehri,
Hussein Hazimeh, Ian Ballantyne, Idan Szpektor,
Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe
Stanton, John Wieting, Jonathan Lai, Jordi Orbay,
Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jy-
otinder Singh, Kat Black, Kathy Yu, Kevin Hui, Ki-
ran V odrahalli, Klaus Greff, Linhai Qiu, Marcella
Valentine, Marina Coelho, Marvin Ritter, Matt Hoff-
man, Matthew Watson, Mayank Chaturvedi, Michael
Moynihan, Min Ma, Nabila Babar, Natasha Noy,
Nathan Byrd, Nick Roy, Nikola Momchev, Nilay
Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil
Botarda, Paul Caron, Paul Kishan Rubenstein, Phil
Culliton, Philipp Schmid, Pier Giuseppe Sessa, Ping-
mei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shiv-
anna, Renjie Wu, Renke Pan, Reza Rokni, Rob
Willoughby, Rohith Vallu, Ryan Mullins, Sammy
Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal,
Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhat-
nagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan
Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty,
Uday Kalra, Utku Evci, Vedant Misra, Vincent Rose-
berry, Vlad Feinberg, Vlad Kolesnikov, Woohyun
Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein
Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta,
Minh Giang, Phoebe Kirk, Anand Rao, Kat Black,
Nabila Babar, Jessica Lo, Erica Moreira, Luiz Gus-
tavo Martins, Omar Sanseviero, Lucas Gonzalez,

Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan
Senter, Eli Collins, Joelle Barral, Zoubin Ghahra-
mani, Raia Hadsell, Yossi Matias, D. Sculley, Slav
Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals,
Jeff Dean, Demis Hassabis, Koray Kavukcuoglu,
Clement Farabet, Elena Buchatskaya, Jean-Baptiste
Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian
Borgeaud, Olivier Bachem, Armand Joulin, Alek An-
dreev, Cassidy Hardin, Robert Dadashi, and Léonard
Hussenot. 2025. Gemma 3 technical report.
Qwen Team. 2026. Qwen3.5: Accelerating productivity
with native multimodal agents.
Connor Walker, Koorosh Aslansefat, Moham-
mad Naveed Akram, and Yiannis Papadopoulos.
2025. Raguard: A novel approach for in-context
safe retrieval augmented generation for llms. In
Model-Based Safety and Assessment: 9th Interna-
tional Symposium, IMBSA 2025, Athens, Greece,
September 24–26, 2025, Proceedings, page 190–204,
Berlin, Heidelberg. Springer-Verlag.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan O Arik. 2025. Astute RAG: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 30553–30571, Vienna, Austria. Association
for Computational Linguistics.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. In-
structRAG: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. InThe Thirteenth
International Conference on Learning Representa-
tions.
Yutao Wu, Xiao Liu, Yinghui Li, Yifeng Gao, Yifan
Ding, Jiale Ding, Xiang Zheng, and Xingjun Ma.
2026. Admit: Few-shot knowledge poisoning attacks
on rag-based fact checking.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably
robust RAG against retrieval corruption. InICML
2024 Next Generation of AI Safety Workshop.
Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun
Chen, and Qian Lou. 2024. Badrag: Identifying
vulnerabilities in retrieval augmented generation of
large language models.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing
Liu, Biao Yi, Tong Li, and Zheli Liu. 2025a. Trace-
back of poisoning attacks to retrieval-augmented gen-eration. InProceedings of the ACM on Web Confer-
ence 2025, WWW ’25, page 2085–2097, New York,
NY , USA. Association for Computing Machinery.
Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang,
Minghong Fang, Zhuqing Liu, Lihai Nie, and Zheli
Liu. 2025b. Benchmarking poisoning attacks against
retrieval-augmented generation.
Chenyang Zhang, Xiaoyu Zhang, Jian Lou, Kai Wu, Zi-
long Wang, and Xiaofeng Chen. 2025c. Poisonedeye:
Knowledge poisoning attack on retrieval-augmented
generation based large vision-language models. In
Forty-second International Conference on Machine
Learning.
Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang,
Xinkui Zhao, Zhengwen Feng, and Jianwei Yin.
2024. Hijackrag: Hijacking attacks against retrieval-
augmented large language models.
Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xu-
anli He, Pasquale Minervini, Youcheng Sun, and
Qiongkai Xu. 2025. GRADA: Graph-based rerank-
ing against adversarial documents attack. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 22244–22266,
Suzhou, China. Association for Computational Lin-
guistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 13764–13775, Singa-
pore. Association for Computational Linguistics.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025. Trustrag: Enhancing robust-
ness and trustworthiness in retrieval-augmented gen-
eration.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2025. PoisonedRAG: Knowledge corruption at-
tacks to Retrieval-Augmented generation of large lan-
guage models. In34th USENIX Security Symposium
(USENIX Security 25), pages 3827–3844, Seattle,
WA. USENIX Association.
A Pseudocode
Here, we include the pseudocode for the Phase 1
(extract possible keywords) and Phase 2 (secondary
verification as Algorithm 1 and Algorithm 2, re-
spectively.

Algorithm 1Phase 1: Extract Possible Keywords
Input:Retrieved document set D, target LLM parameters θ,
user question q, affirmation string y, punctuation set P, fre-
quent words set F, minimum occurrence threshold a1, number
of top-influence tokensk 1.
Output:Keyword setK
1:K temp← ∅
2:for eachd∈Ddo
3:n←length(d)
4:x 1:n←embed θ(d1:n)
5:V← ∅
6:fori= 1tondo
7:Influence(i)← ∥∇ xilogP θ(y|x 1:n)∥2
8:ifd i/∈qandd i/∈Pandd i/∈Fthen
9:V←V∪ {i}
10:I top←argtop-k 1
i∈V(Influence(x i))
11:K d← ∅
12: Partition Itopto a set of contiguous index segments S
13:for eachsegment(i, i+ 1, . . . , j)∈Sdo
14:e←concatenate(d i:j)
15:K d←K d∪ {e}
16: AppendK dtoK temp
17:K← ∅
18:U←the set of all unique candidate inK temp
19:for eache∈Udo
20:count← |{d∈D|e∈d}|
21:ifcount≥a 1then
22:K←K∪ {e}
23:returnK
B Punctuations and Frequent Words
B.1 Punctuations
!, ", #, %, &, ’, (, ), *, ,, -, .,
/, :, ;, ?, @, [, \, ], _, {, }
B.2 Frequent Words
"a", "an", "the", "in", "on", "at",
"of", "to", "from", "for", "with",
"by", "about", "and", "or", "but",
"is", "are", "was", "were", "be",
"he", "she", "it", "they", "we",
"you", "your", "I", "me", "my",
"this", "that", "these", "those",
"what", "which", "who", "whom",
"whose", "how", "when", "where",
"why"Algorithm 2Phase 2: Secondary Verification
Input:Retrieved document set D, target LLM parameterized
byθ, candidate keyword set K, user question q, punctuation
setP, frequent words set F, minimum occurrence threshold
a2, number of top-influence tokensk 2.
Output:Target keyword setT, detection flagf
1:T← ∅
2:for eachp∈Kdo
3:K temp← ∅
4:for eachd∈Ddo
5:n←length(d)
6:x 1:n←embed θ(d1:n)
7:V← ∅
8:fori= 1tondo
9:Influence(i)← ∥∇ xilogP θ(p|x 1:n)∥2
10:ifd i/∈qandd i/∈Pandd i/∈Fthen
11:V←V∪ {i}
12:I top←argtop-k 2
i∈V(Influence(x i))
13:K d← {d i|i∈I top∧di∈p}
14: AppendK dtoK temp
15:U←the set of all unique tokens inK temp
16:for eache∈Udo
17:count← |{K d∈Ktemp|e∈K d}|
18:ifcount≥a 2then
19:T←T∪ {e}
20:ifT̸=∅then
21:f←TRUE
22:else
23:f←FALSE
24:returnT, f
C Models Used in Experiments
Llama-3.1-8B
meta-llama/Meta-Llama-3.1-8B-Instruct1
Gemma-3-4b
google/gemma-3-4b-it2
Qwen-3.5-4B
Qwen/Qwen3.5-4B3
Vicuna-7b
lmsys/vicuna-7b-v1.54
Mistral-7B
mistralai/Mistral-7B-Instruct-v0.35
Phi-4-mini
microsoft/Phi-4-mini-instruct6
1https://huggingface.co/meta-llama/Llama-3.1-8B-
Instruct
2https://huggingface.co/google/gemma-3-4b-it
3https://huggingface.co/Qwen/Qwen3.5-4B
4https://huggingface.co/lmsys/vicuna-7b-v1.5
5https://huggingface.co/mistralai/Mistral-7B-Instruct-
v0.3
6https://huggingface.co/microsoft/Phi-4-mini-instruct

D Results under Different
Hyperparameter Configurations
D.1 TPR
Figure 9: TPR results with 3retrieved documents, a1=
2,a 2= 2,k 1= 5,k 2= 3.
Figure 10: TPR results with a1= 2,a2= 2,k1= 5,
k2= 3.
Figure 11: TPR results with a1= 2,a2= 3,k1= 5,
k2= 3.
Figure 12: TPR results with a1= 2,a2= 4,k1= 5,
k2= 3.
Figure 13: TPR results with a1= 3,a2= 3,k1= 5,
k2= 3.

Figure 14: TPR results with a1= 3,a2= 4,k1= 5,
k2= 3.
Figure 15: TPR results with a1= 4,a2= 2,k1= 5,
k2= 3.
Figure 16: TPR results with a1= 4,a2= 4,k1= 5,
k2= 3.D.2 FPR
Figure 17: FPR results with 3retrieved documents,
a1= 2,a 2= 2,k 1= 5,k 2= 3.
Figure 18: FPR results with a1= 2,a2= 2,k1= 5,
k2= 3.
Figure 19: FPR results with a1= 2,a2= 3,k1= 5,
k2= 3.

Figure 20: FPR results with a1= 2,a2= 4,k1= 5,
k2= 3.
Figure 21: FPR results with a1= 3,a2= 2,k1= 5,
k2= 3.
Figure 22: FPR results with a1= 3,a2= 3,k1= 5,
k2= 3.
Figure 23: FPR results with a1= 3,a2= 4,k1= 5,
k2= 3.
Figure 24: TPR results with a1= 4,a2= 2,k1= 5,
k2= 3.
D.3 ACC
Figure 25: ACC results with 3retrieved documents,
a1= 2,a 2= 2,k 1= 5,k 2= 3.

Figure 26: ACC results with a1= 2,a2= 2,k1= 5,
k2= 3.
Figure 27: ACC results with a1= 2,a2= 3,k1= 5,
k2= 3.
Figure 28: ACC results with a1= 2,a2= 4,k1= 5,
k2= 3.
Figure 29: ACC results with a1= 3,a2= 3,k1= 5,
k2= 3.
Figure 30: ACC results with a1= 3,a2= 4,k1= 5,
k2= 3.
Figure 31: ACC results with a1= 4,a2= 2,k1= 5,
k2= 3.

Figure 32: ACC results with a1= 4,a2= 3,k1= 5,
k2= 3.
Figure 33: ACC results with a1= 4,a2= 4,k1= 5,
k2= 3.
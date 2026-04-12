# RefineRAG: Word-Level Poisoning Attacks via Retriever-Guided Text Refinement

**Authors**: Ziye Wang, Guanyu Wang, Kailong Wang

**Published**: 2026-04-08 10:33:54

**PDF URL**: [https://arxiv.org/pdf/2604.07403v1](https://arxiv.org/pdf/2604.07403v1)

## Abstract
Retrieval-Augmented Generation (RAG) significantly enhances Large Language Models (LLMs), but simultaneously exposes a critical vulnerability to knowledge poisoning attacks. Existing attack methods like PoisonedRAG remain detectable due to coarse-grained separate-and-concatenate strategies. To bridge this gap, we propose RefineRAG, a novel framework that treats poisoning as a holistic word-level refinement problem. It operates in two stages: Macro Generation produces toxic seeds guaranteed to induce target answers, while Micro Refinement employs a retriever-in-the-loop optimization to maximize retrieval priority without compromising naturalness. Evaluations on NQ and MSMARCO demonstrate that RefineRAG achieves state-of-the-art effectiveness, securing a 90% Attack Success Rate on NQ, while registering the lowest grammar errors and repetition rates among all baselines. Crucially, our proxy-optimized attacks successfully transfer to black-box victim systems, highlighting a severe practical threat.

## Full Text


<!-- PDF content starts -->

RefineRAG: Word-Level Poisoning Attacks via
Retriever-Guided Text Refinement
Ziye Wang1⋆[0009−0007−0047−4037], Guanyu Wang2⋆[0009−0005−0172−3950], and
Kailong Wang1⋆⋆[0000−0002−3977−6573]
1Huazhong University of Science and Technology, China
{zywang817, wangkl}@hust.edu.cn
2Beihang University
gywang@buaa.edu.cn
Abstract.Retrieval-AugmentedGeneration(RAG)significantlyenhances
Large Language Models (LLMs), but simultaneously exposes a critical
vulnerability to knowledge poisoning attacks. Existing attack methods
likePoisonedRAGremaindetectableduetocoarse-grainedseparate-and-
concatenate strategies. To bridge this gap, we propose RefineRAG, a
novel framework that treats poisoning as a holistic word-level refinement
problem. It operates in two stages: Macro Generation produces toxic
seeds guaranteed to induce target answers, while Micro Refinement em-
ploys a retriever-in-the-loop optimization to maximize retrieval priority
without compromising naturalness. Evaluations on NQ and MSMARCO
demonstrate that RefineRAG achieves state-of-the-art effectiveness, se-
curing a 90% Attack Success Rate on NQ, while registering the lowest
grammar errors and repetition rates among all baselines. Crucially, our
proxy-optimized attacks successfully transfer to black-box victim sys-
tems, highlighting a severe practical threat.
Keywords:RAG·Poisoning Attack·LLM
1 Introduction
Retrieval-AugmentedGeneration(RAG)hasemergedasaparadigmshiftinNat-
ural Language Processing (NLP), enhancing Large Language Models (LLMs) by
grounding them in up-to-date knowledge. However, this dependency on external
retrievers exposes a critical risk called knowledge poisoning [3,30,31,17]. In such
attacks, adversaries inject carefully crafted toxic texts into public corpora like
Wikipedia. When a user asks a query, the retriever fetches these poisoned items,
prompting the LLM to generate misinformation.
Despite the severity of this threat, existing attack methods remain largely
detectable due to their coarse-grained design. Current state-of-the-art (SOTA)
methods like PoisonedRAG [31] rely on aSeparate-and-Concatenate (SoC)
strategy (P=S⊕I). They optimize a retrieval trigger (S) typically a sequence
⋆Equal contribution.
⋆⋆Corresponding author.arXiv:2604.07403v1  [cs.CR]  8 Apr 2026

2 Ziye Wang et al.
of meaningless characters or keywords—and forcibly concatenate it with the ma-
licious content (I). While effective at triggering retrieval, this approach intro-
duces severe structural artifacts. The resulting texts often exhibit abnormally
high perplexity and linguistic incoherence, making them easily identifiable by
defense mechanisms based on fluency or repetition filters.
To bridge the gap between attack effectiveness and stealthiness, we argue
thatattackersmustdroptheSoCstrategyforaholisticword-levelrefinementap-
proach. We note that subtle, context-aware lexical substitutions can induce sig-
nificant shifts in embeddings without compromising semantic readability [8,14].
Based on this, we proposeRefineRAG, a novel two-stage framework that treats
RAG poisoning as a text refinement problem rather than a concatenation task.
RefineRAG operates a macro-generation, micro-refinement pipeline designed
to satisfy three key principles simultaneously: generation quality, retrieval pri-
ority, and stealthiness.
– Macro Generation: We first generate a diverse corpus of seed texts that are
semantically guaranteed to trigger the target incorrect answer, ensuring the
Generation Principle is met.
– Micro Refinement: We employ a Word-Level Optimization (WLO). Acting
as a referee, a proxy retriever iteratively guides a Masked Language Model
(MLM) to replace specific words. This process micro-carves the text to max-
imize its similarity to the target question in the embedding space while pre-
serving natural syntax and low perplexity.
We evaluate RefineRAG on two widely-used benchmarks, Natural Questions
(NQ) [15] and MSMARCO [2]. The results show that RefineRAG achieves SOTA
performance, getting a 90% attack success rate on NQ. Crucially, it outperforms
all baselines in stealthiness, registering the lowest grammar error rates and rep-
etition rates. Furthermore, we demonstrate a strong transferability: attacks op-
timized in a local proxy setting successfully compromise black-box victim re-
trievers and LLMs, revealing a significant practical threat to real-world RAG
systems.
To sum up, our main contributions are summarized as follows:
–We identify the limitations of the SoC strategy and propose a new perspective
focused on holistic, word-level refinement for RAG poisoning.
–WeintroduceRefineRAG,atwo-stageattackframeworkthatintegratesmulti-
objective seed generation with a retriever-guided word-level optimization al-
gorithm to balance effectiveness and stealthiness.
–ExtensiveexperimentsdemonstratethatRefineRAGsignificantlyoutperforms
SOTA methods in success rate and stealthiness, while exhibiting robust trans-
ferability against black-box systems, revealing the vulnerability of the current
RAG system to fine-grained attacks.

RefineRAG 3
2 Related Work
2.1 Retrieval-Augmented Generation
RAG systems address the knowledge limitations of LLMs [4,9,16,22] by inte-
grating external retrieval mechanisms. A standard RAG framework workflow
comprises a dense retriever and a generator LLM. The retriever such as Con-
triever [13] and ANCE [26] encodes both queries and documents into dense
vectors, selecting top-Kpassages based on similarity scores. The generator then
synthesizes the final answer using the retrieved context.
While this architecture improves factual accuracy, the reliance on dense vec-
tor matching inherently harbors vulnerabilities, such as false matching problems
[24]. Combined with the critical dependency that the LLM implicitly trusts the
retrieved context, this creates a severe risk: manipulating the external knowledge
base can effectively control the model’s output.
2.2 Existing Attacks against LLMs
Existing research has primarily focused on two typical types of generic attacks:
Prompt Injection [19] and Jailbreaking [25].
Prompt Injection aims to hijack the model by embedding malicious instruc-
tions, such as “Ignore previous instructions”. However, in RAG, these injections
fail if they are not retrieved. Since prompt injection techniques do not optimize
for retrieval similarity, malicious instructions often remain buried in the corpus,
never reaching the LLM’s context window.
Jailbreaking focuses on bypassing safety alignment to generate restricted
content like hate speech. This fundamentally differs from knowledge poisoning,
which aims for cognitive misdirection rather than breaking ethical guardrails.
2.3 Data Poisoning Attacks for RAG
Corpus Poisoning Attack.Zhong et al. [30] demonstrated that injecting
meaningless sequences of keywords optimized for retrieval ranking can manipu-
late the retrieved context. However, these texts are often incoherent and fail to
effectively guide the LLM’s generation toward a specific target answer.
RAG-Specific Poisoning.PoisonedRAG [31] advanced this by employing a
SoC strategy. It optimizes a retrieval trigger via gradients and concatenates it
with a target incorrect answer. While this achieves higher retrieval rates, the
forced concatenation results in structural inconsistencies. The white-box variant
produces high-perplexity artifacts similar to those found in training data poi-
soning , while the black-box variant relies on repeating the user query, leading
to detectable redundancy. Newer works like CPA-RAG [17] explore covert poi-
soning, yet the trade-off between retrieval effectiveness and textual stealthiness
remains a persistent challenge in coarse-grained manipulation strategies.

4 Ziye Wang et al.
2.4 Adversarial Attacks via Lexical Substitution
Word-level optimization is a well-established technique in adversarial NLP, tra-
ditionally used to attack text classifiers.
Gradient-Based Methods.Early works like HotFlip [8] utilize gradient infor-
mation to identify and flip vulnerable tokens to maximize classification error.
Substitution-BasedMethods.ApproachessuchasTextFooler[14]andBERT-
Attack [18] employ MLMs like BERT [7] to generate context-aware synonyms.
These methods iteratively replace importance-ranked words to alter the model’s
prediction while preserving semantic consistency and human readability.
Despite the maturity of lexical substitution in classification tasks, its appli-
cation to retrieval ranking remains underexplored. Existing RAG attacks largely
ignore these fine-grained optimization techniques, relying instead on document-
level concatenation. This work seeks to adapt these micro-level perturbation
techniques to the RAG domain, shifting the optimization objective from classi-
fication error to retrieval similarity.
3 Threat Model
To establish real-world feasibility, we formulate our threat model assuming a
realistic, resource-constrained adversary.
Attacker’s Goal.The attacker aims to achieve precision content manipulation
rather than degrading general system performance. Specifically, the adversary
pre-defines a target questionQand designs a specific, plausible but incorrect
answerR t. The ultimate objective is to ensure that when a user queries the
RAG system withQ, the system is misled into retrieving the poisoned context
and confidently generatingR tas the answer. Such targeted attacks pose severe
riskstohigh-stakesapplicationsrequiringstrictfactualaccuracy,suchasmedical
consultation or financial analysis [28,29,27,1].
Attacker’s Knowledge.We consider a realistic proxy-assisted black-box sce-
nario.Theattackerhasnoaccesstotheinternalparameters,gradients,orspecific
configurations of the victim RAG system. However, relying on the transferabil-
ity assumption, the attacker can leverage general knowledge about RAG mecha-
nismstoconstructtheattack.Specifically,theattackerutilizespubliclyavailable,
state-of-the-art open-source models (Contriever [12]) as proxies to optimize ad-
versarial texts locally, aiming for the generated samples to be effective against
unknown target systems. This setting assumes the attacker employs state-of-
the-art open-source tools to maximize the potential impact, consistent with the
behavior of a rational adversary.
Attacker’s Capabilities.The attacker’s influence is confined strictly to the
data source at inference time, without any access to the model development
pipeline.Theircapabilityislimitedtoinjectingasmallnumberofpoisoningtexts
into the public knowledge corpus indexed by the RAG system—for example,
by modifying entries on publicly editable sites or posting on indexed forums.
The attack relies solely on external data contamination and does not involve
tampering with the training process, model weights, or system code.

RefineRAG 5
Target Question: 
Where did the titanic 
sink at what ocean?
Target Answer: 
South Atlantic Ocean
S: The South 
Atlantic Ocean is 
the definitive 
location where the 
Titanic sank [...]
S1: [...] definitive 
location where [...]
S1: [...] [MASK] 
location where [...]S11: [...] ultimate 
location where [...]
...
S1m: [...] exact 
location where [...]S*: The South 
Atlantic Ocean is 
the definitive 
location where the 
Titanic sank [...]Question: 
Where did the titanic 
sink at what ocean?Answer: 
South Atlantic Ocean
Context: The South 
Atlantic Ocean is the 
ultimate region where the 
Titanic sank [...]
Question: Where did the 
titanic sink at what ocean?Generate LLM Validate LLM
MLMLLMCandidates Seed Base
Refinement 
BaseUser
Retriever
Knowledge Base CandidatesTop-K
T Rounds
Identify 
TsInitialize
UpdateTop-BInject Stagte I: Generation and Filtering of Adversarial Seeds
Stage II: WLO Multi-Objective Refinement
Fig. 1.The overall framework of RefineRAG.
4 Methodology
In this section, we introduce RefineRAG. The overview of our framework is
shown in Figure 1. Building upon the “generation condition” principle of Poi-
sonedRAG [31] and the retriever-based evaluation concept of CPA-RAG [17],
RefineRAG introduces a new micro-refinement stage. It combines macro-level
seed generation (Stage I) with a novel micro-level, retriever-guided Word-Level
Optimization (Stage II) to create highly effective and stealthy poisoned texts.
4.1 Problem Formulation and Design Principles
Given a questionQand an incorrect answerR t, our goal is to synthesize a
poisoning textPthat satisfies three competing principles simultaneously:
Generation Principle.Pmust semantically induce theR t. Formally, given a
generatorG, the likelihood of generatingR tgiven contextPmust be maximized:
G(Rt|Q, P)≫ G(R c|Q, P)(1)
whereR cis the correct answer.
RetrievalPrinciple.PmustachieveahighrankintheretrievalcorpusD.This
requires maximizing the similarity scoreSim(·)between the query embedding
E(Q)and the text embeddingE(P)in the dense vector space of a retrieverR:
max
PSim(E(Q), E(P))(2)
This ensuresPis prioritized over benign documents.
Stealthiness Principle.Pmust maintain linguistic naturalness to evade de-
tection. We constrainPto exhibit low perplexity and high grammatical correct-
ness, avoiding the structural artifacts common in concatenation-based attacks
like repetition and gibberish.
Existing methods like PoisonedRAG use a component separation and recom-
bination strategy to address the first two principles separately. However, this
often violates the stealthiness principle. To address this, we treat the attack as a
holistic optimization problem, using a two-stage agent-driven process of macro-
generation and micro-refinement to satisfy all three principles simultaneously.

6 Ziye Wang et al.
Algorithm 1:Stage I: Generation and Filtering of Adversarial Seeds
Input:Target QuestionQ, Target AnswerR t, Generate LLMGen, Validate
LLMV al, The number of refined seedsK, The number of iterations
T, The number of injected textI, Seed BaseBase, Generate Prompt
Prompt gen, Refine PromptPrompt re
Output:A set of seedsSet S
1fort= 1toTdo
2Cands←Gen(Prompt gen, Q, R t);
3Base←V al(Cands)∪Base;
4fork= 1toKdo
5seed←Base[k−1];
6Cands←Gen(Prompt re, seed);
7Base←V al(Cands)∪Base;
8Set S←Base[:I];
9returnSet S;
4.2 Stage I: Generation and Filtering of Adversarial Seeds
The objective of this stage is to construct a seed corpus for target questionQ
that strictly satisfies theGenerationPrinciple while exploring the semantic
space for high-potential candidates. To achieve this, as shown in algorithm 1,
we use a LLM to generate candidates overTiterations. We adopt a hybrid
strategy of exploration and exploitation. Exploration phase generates diverse
new candidates from zero-shot and one-shot prompts to expand the search space,
while exploitation phase selects top-performing seeds from previous round and
rewrites them to refine their quality.
To ensure attack validity, we enforce a mandatory constraint: any candidate
Pmust successfully trigger the target answerR twhen fed to a validation LLM.
Candidates that fail this check (i.e., generate the correct answerR cor irrelevant
content) are immediately discarded. Valid seeds are ranked by their retrieval
similaritySim(P, Q), and the top-Kcandidates form the input for the next
iteration. This guarantees that all seeds passed to Stage II are functionally toxic.
4.3 Stage II: Micro-Refinement via Multi-Objective Optimization
ThisstageaddressestheRetrievalandStealthinessprinciples.Sincetheinput
seeds are already verified for toxicity, we focus purely on optimizing their vector
representation through a novel Word-Level Optimization. We aim to maximize
the retrieval score Equation 2 by perturbing discrete tokens inPwithout dis-
rupting its semantic coherence. The objective function is defined as maximizing
Score(P, Q) =Sim(P, Q), constrained by the requirement that the optimization
direction aligns with the target answerR t.
The overall procedure of this stage is in algorithm 2. We use Part-Of-Speech
(POS) tagging to identify content words onSeligible for replacement, while
freezing keywords essential to the questionQand target answerR tto prevent

RefineRAG 7
Algorithm 2:Stage II: WLO Multi-Objective Refinement
Input:The number of optimized seedsB, The number of iterationsT,
Masked Language ModelMLM, POS Tagging ModelM tag, Seed set
from Stage-ISet S, Refinement BaseBase
Output:The set of poisoned texts to be injectedSet S∗
1Set S∗← ∅;
2foreachS∈Set Sdo
3Base←S;
4fort= 1toTdo
5Cands← ∅;
6fori= 1toBdo
7Si←Base[i−1];
8TSi←M tag(Si);
9foreachw∈TSido
10Si
mask←replacewinSiwith [MASK];
11Cands←Cands∪MLM(Si
mask);
12Base←Cands;
13S∗←Base[0];
14Set S∗←Set S∗∪S∗
15returnSet S∗;
semantic drift. The set of target wordsT Scan be formally defined as:
TS={w∈S|POS(w)∈ {N, V, ADJ, ADV}, w /∈KW(Q, R t)∪SW}(3)
whereKWandSWrespectively represent sets of keywords and stop words.
For each target word, we mask it with a[MASK]token and utilize a MLM [7]
to predict top-Kcontext-aware substitutes. This ensures that all perturbations
remaingrammaticallyandsemanticallynatural[14,8],satisfyingtheStealthiness
Principle. Instead of selecting words based on classification loss, we employ a
proxy retriever as a referee. We calculate the embedding shift caused by each
candidatesubstitutionandselectthewordthatmaximizesthesimilarityincrease
∆Sim(S′, Q). To avoid local optima, we employ Beam Search to maintain the
top-Bbest trajectories throughout the optimization iterations.
5 Experiment
5.1 Experimental Setup
Datasets.We conduct our experiments on two widely-used Question Answer-
ing (QA) datasets: Natural Questions (NQ) [15] and MSMARCO [2]. NQ is
primarily sourced from Wikipedia articles and contains approximately 2.6 mil-
lion documents, while MSMARCO is derived from Microsoft Bing search results
and comprises about 8.8 million documents. Following the previous works, we
randomlyselect100closed-endedquestionsfromNQandMSMARCOseparately

8 Ziye Wang et al.
to serve as target questions. For each question, we employ Deepseek-V3 [6] to
generate a plausible but factually incorrect answer, designated as the target an-
swer. We manually verify each target answer to ensure it directly conflicts with
the ground truth, thereby establishing a valid poisoning target.
Baselines.To comprehensively evaluate the performance of RefineRAG, we
compare it with several open-source poisoning attack methods. We benchmark
againsttheSOTAPoisonedRAG[31],evaluatingbothitswhite-boxvariant(Poi-
sonedRAG (W)), which utilizes gradient-based optimization for trigger genera-
tion, and its black-box variant (PoisonedRAG (B)), which relies on query con-
catenation strategies. Additionally, we compare our method with the Prompt
Injection Attack [19], which attempts to embed explicit malicious instructions
within the text, and the Corpus Poisoning Attack [30], which focuses on opti-
mizing meaningless character strings to maximize retrieval ranking.
Metrics.Our evaluation assesses both attack effectiveness and stealthiness. For
effectiveness, we use the Attack Success Rate (ASR) [21,11] to measure the pro-
portion of the model’s responses that strictly match the target incorrect answer.
Specifically, unless otherwise noted, we calculate this metric using two repre-
sentative victim models, Llama-2-7B [23] and Vicuna-7B [5], denoted as L-ASR
and V-ASR, respectively. We also employ standard retrieval metrics—Precision,
Recall, and F1-Score—to quantify the success of the injected adversarial texts
in penetrating the top-k results. To rigorously evaluate stealthiness, we measure
Perplexity (PPL) using a pre-trained GPT-2 [20] model to assess language flu-
ency. Furthermore, we calculate the average number of Grammar Errors (GE)
using automated tools and compute the ROUGE-L Recall (RL) to measure lexi-
cal overlap with the query. Finally, we report the Repetition Rate (RR) to detect
semantic redundancy among the generated adversarial texts, distinguishing nat-
ural writing from template-based attacks
Implementation Details.In our experimental setup, five poisoned texts are
injected into the corpus for each target question. Stage I operates for four iter-
ations. In each round, the DeepSeek-V3 [6] model generates 10 candidates with
a temperature of 1.0 and a minimum length of 25, which are subsequently vali-
dated by a Llama-7B model to ensure toxicity. The top two candidates from the
validation phase are then refined to produce additional variations. In Stage II,
we select the top-5 seeds from Stage I seed base and refine each individually us-
ing our WLO algorithm. This process runs for 10 iterations with a Beam Search
size of 3, utilizing spaCy [10] for POS tagging and a BERT-Large MLM [7]
to predict 20 candidate replacements per masked word. Finally, Contriever [12]
retrieves the top-5 relevant texts from the knowledge base to provide context
for the victim models, specifically Llama2-7B and Vicuna-7B, which generate
the final answers. All experiments are performed on a workstation with Ubuntu
22.04.3 LTS and an A100 GPU with 80GB memory.
5.2 Comparison with baselines
ThecomprehensiveresultspresentedinTable1revealthatRefineRAGachievesa
superiorbalancebetweeneffectivenessandstealthinesscomparedtoallbaselines.

RefineRAG 9
Table 1.Evaluation of RefineRAG against baseline methods.
Attack MethodNQ MS MARCO
F1↑L-ASR↑V-ASR↑PPL↓RR↓GE↓RL F1↑L-ASR↑V-ASR↑PPL↓RR↓GE↓RL
PoisonedRAG (B) 0.940.54 0.6155.110.28 2.21 1.00 0.870.60 0.67 55.880.28 2.21 1.00
PoisonedRAG (W) 0.950.59 0.65 372.930.006.54 0.66 0.950.63 0.66 257.540.006.02 0.65
Prompt Injection Attack 0.75 0.80 0.75107.011.00 0.88 1.000.770.83 0.81137.86 1.00 0.99 1.00
Corpus Poisoning Attack 0.66 0.000.00 8209.98 1.00 9.11 0.36 0.51 0.00 0.00 8247.50 1.00 9.16 0.29
RefineRAG (Ours) 0.890.90 0.85118.33 0.01 0.660.55 0.700.83 0.81127.53 0.010.860.53
Table 2.Transferability across Victim LLMs.
DatasetASR↑
Llama2 Vicuna Deepseek-R1 Deepseek-V3 Qwen2.5 Qwen3
NQ 0.90 0.85 0.84 0.92 0.86 0.81
MSMARCO 0.83 0.81 0.75 0.78 0.76 0.73
While PoisonedRAG (W) achieves a high retrieval F1-Score, its gradient-driven
approach results in an anomalously high PPL (372.93) and frequent GE (6.54),
making it easily detectable. Conversely, PoisonedRAG (B) addresses fluency by
copying the query, but this heuristic leads to maximal RL with 1.00 and a high
RR with 0.28, exposing it to deduplication filters. The Prompt Injection Attack
similarly suffers from maximal redundancy (RR 1.00) due to its fixed template
structure, whilethe CorpusPoisoningAttackfails completelyin generationtasks
with an ASR of 0.00. In contrast, RefineRAG secures the highest ASR on both
NQ (0.90) and MSMARCO (0.83) while maintaining natural fluency (PPL 118)
and registering the lowest GE and RR across all methods.
5.3 Transferability Analysis
WefurtherevaluatetherobustnessofRefineRAGacrossdifferentvictimsystems.
Transferability across Victim LLMs.As shown in Table 2, when tested
against six diverse open-source LLMs, namely Llama2-7B (Llama2), Vicuna-7B
(Vicuna), DeepSeek-R1, DeepSeek-V3, Qwen2.5-7B (Qwen2.5) and Qwen3-Max
(Qwen3), RefineRAG maintains consistently high ASR scores ranging from 0.81
to 0.92 on the NQ dataset. This demonstrates strong model-agnostic transfer-
ability driven by its semantic optimization.
Transferability across Retrievers.In terms of retriever generalization, Ta-
ble 3 indicates that adversarial samples optimized on Contriever transfer ef-
fectively to unseen retrievers such as Contriever-msmarco (Contriever-ms) and
ANCE. Although the F1-score decreases due to domain shifts, the attack main-
tainsasignificantASRofupto0.70onNQdataset,indicatingthatthegenerated
texts occupy a broad toxic region in the embedding space.
Impact of Retrieval Scope.For the retrieval scopekshown in Figure 2, we
observe a non-linear relationship where attack performance peaks at a scope of
5 and gradually declines askincreases to 10. This trend confirms the dilution

10 Ziye Wang et al.
Table 3.Transferability across Retrievers.
RetrieverNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
Contriever 0.89 0.90 0.85 0.70 0.83 0.81
Contriever-ms 0.79 0.82 0.80 0.60 0.64 0.66
ANCE 0.63 0.70 0.68 0.47 0.56 0.60
1 2 3 4 5 6 7 8 9 10
k0.00.20.40.60.81.0Score
ASR
1 2 3 4 5 6 7 8 9 10
k0.00.20.40.60.81.0
Precision
1 2 3 4 5 6 7 8 9 10
k0.00.20.40.60.81.0
Recall
1 2 3 4 5 6 7 8 9 10
k0.00.20.40.60.81.0
F1-Score
Fig. 2.Impact of Retrieval Scope (k) on NQ.
effect where an excessive number of retrieved benign documents weakens the
adversarial context, a phenomenon consistent with prior findings in the field.
5.4 Ablation Study
To verify the necessity of our two-stage design, we conduct ablation experiments
by systematically removing key components of the framework. We first evaluate
the configuration designated as No-I where Stage I is removed and WLO is ap-
plied directly to initial texts. This modification causes a sharp performance drop
on the NQ dataset, with the L-ASR falling from 0.90 to 0.72 and the V-ASR
decreasing from 0.85 to 0.69. The F1-score similarly declines from 0.89 to 0.63.
We also observe a significant performance reduction on the MSMARCO dataset,
confirming that micro-refinement relies heavily on a high-quality semantic foun-
dation. Conversely, removing Stage II and using only the output from Stage I, a
setting named No-II, results in a substantial decrease in retrieval effectiveness.
Specifically,theretrievalF1-scoredropsto0.73comparedtothe0.89achievedby
the full model, leading to a corresponding decline in ASR. These results demon-
strate that the synergy between macro-level toxicity generation and micro-level
retrieval optimization is essential for the framework’s overall success.
5.5 Parameter Analysis
Robustness to Generator LLM Choice.To determine whether RefineRAG
depends on a specific generative architecture, we fix the victim models and com-
pare performance using two different generators in Stage I: DeepSeek-V3 and
Qwen3-Max. The results demonstrate that the framework is highly robust to
the choice of the attacker’s generator, with final performance metrics remaining
nearly identical across both models. For instance, on the NQ dataset, the L-ASR

RefineRAG 11
Table 4.Analysis of the contribution of each stage in RefineRAG.
Attack MethodNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
No-I 0.63 0.72 0.69 0.53 0.71 0.75
No-II 0.73 0.86 0.82 0.54 0.75 0.79
RefineRAG 0.89 0.90 0.85 0.70 0.83 0.81
Table 5.Impact of Attacker Generators.
Attack ModelNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
Deepseek-V3 0.89 0.90 0.85 0.70 0.83 0.81
Qwen3-Max 0.91 0.85 0.82 0.74 0.83 0.83
is0.90forDeepSeek-V3comparedto0.85forQwen3-Max,whileonMSMARCO,
the performance is identical at 0.83. These findings confirm that the attack ef-
fectiveness is driven by the framework’s optimization strategy rather than the
inherent capability of a specific generator.
Optimal RoundsTfor Macro-Generation.We evaluate the impact of the
number of iteration roundsTin Stage I by testing values from the set{3,4,5}.
Performance peaks atT= 4, where RefineRAG achieves an F1-score of 0.89
on NQ and 0.70 on MSMARCO, alongside the highest ASR values. With fewer
rounds (T= 3), the mixed strategy underperforms due to insufficient conver-
gence, yielding lower ASR scores. Conversely, increasing the rounds toT=5
leads to a slight degradation in performance, likely due to noise introduced by
excessive iterations. Consequently, we adoptT= 4as the default setting to
balance generation quality and stability.
Sensitivity to WLO Iterations.We investigate the effect of the number of
WLOiterationsinStageIIbycomparingperformanceat5,10,and15iterations.
On the NQ dataset, while increasing iterations improves the retrieval F1-score,
the ASR peaks at 5 iterations, suggesting that further refinement may weaken
the adversarial signal. On MSMARCO, the ASR peaks at 10 iterations, even as
the F1-score continues to rise. These trends indicate a trade-off where additional
iterations enhance retrieval visibility but do not consistently improve the likeli-
hood of misleading the LLM. We therefore select 10 iterations as the balanced
configuration for our main experiment.
Effect of MLM Candidate Countk.We analyze the sensitivity of Stage
II to the number of replacement candidates predicted by the MLM, testingK
values of 10, 20, and 30. IncreasingKfrom 10 to 20 consistently improves perfor-
mance. For example, the F1-score on NQ rises from 0.85 to 0.89, and the L-ASR
on MSMARCO increases to 0.90. However, further increasingKto 30 yields
diminishing returns, with ASR decreasing on both datasets. This suggests that

12 Ziye Wang et al.
Table 6.Impact of Stage I Iterations.
T-ValueNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
30.84 0.84 0.79 0.63 0.72 0.73
40.89 0.90 0.85 0.70 0.83 0.81
50.88 0.86 0.83 0.67 0.79 0.72
Table 7.Impact of Stage II WLO Iterations.
IterationsNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
50.850.91 0.87 0.65 0.79 0.80
10 0.890.90 0.85 0.700.83 0.81
15 0.890.88 0.84 0.740.81 0.79
larger candidate sets may introduce semantic noise that dilutes the adversarial
efficacy. Based on these results, we setKto 20 as the default to optimize the
balance between candidate diversity and attack precision.
Influence of Beam Search WidthB.Finally, we examine the impact of the
beam sizeBduring the WLO process by comparing widths of 1, 3, and 5. Using
a beam size ofB= 3yields substantial improvements over the greedy approach
(B= 1), raising the L-ASR on MSMARCO from 0.77 to 0.83. Increasing the
beam width further toB= 5provides only marginal gains in retrieval metrics
and leads to lower ASR across most configurations. This indicates that larger
beams may over-prioritize retrieval metrics at the expense of the adversarial
signal. Therefore, we adoptB= 3as the default setting to achieve the best
trade-off between attack success and computational efficiency.
6 Discussion
6.1 Ethical Considerations
Our primary motivation is to uncover RAG vulnerabilities in high-stakes do-
mains before malicious exploitation. We strictly adhere to responsible AI prin-
ciples. Experiments were conducted in an isolated simulation environment using
public datasets. While commercial LLM APIs were utilized for simulation, the
evaluation was strictly confined to our local setup, ensuring no impact on real-
world systems. We aim to urge the prioritization of sanitization-based defenses
against such stealthy threats.
6.2 Limitations and Future Work
We acknowledge three limitations that direct future research:

RefineRAG 13
Table 8.Sensitivity to the number of candidate selections.
K-ValueNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
100.85 0.870.86 0.67 0.80 0.80
20 0.89 0.900.85 0.700.83 0.81
30 0.890.87 0.85 0.720.82 0.77
Table 9.Impact of Beam Size.
B-ValueNQ MSMARCO
F1↑L-ASR↑V-ASR↑ F1↑L-ASR↑V-ASR↑
10.86 0.87 0.84 0.68 0.77 0.80
30.89 0.900.85 0.700.83 0.81
50.88 0.880.89 0.710.79 0.80
– ComputationalOverhead:TheiterativeMLM-basedoptimizationinStage
II incurs higher computational costs than simple concatenation methods. Fu-
ture work could explore distillation techniques to accelerate the process.
– Dependency on Proxy Retrievers:Our black-box transferability assumes
embedding similarity between proxy and victim retrievers. Efficacy against
radically different architectures (sparse retrievers) requires further investiga-
tion into “universal” perturbations.
– Defense Evasion Boundaries:While RefineRAG bypasses fluency-based
filters,itsrobustnessagainstadvancedsemanticdefenses(externalfact-checking)
remains to be evaluated in future studies.
7 Conclusion
In this paper, we address the limitation of existing RAG poisoning attacks where
effectiveness comes at the cost of stealthiness. By shifting from coarse-grained
splicing to RefineRAG’s fine-grained, word-level refinement, we synthesize poi-
soning texts that are both highly retrievable and linguistically coherent. Our
experiments confirm that RefineRAG significantly outperforms SOTA methods
in both success rates and stealthiness metrics. Moreover, our findings reveal a
concerning level of transferability, where adversarial samples generated locally
on proxy models can effectively compromise unknown, black-box RAG systems.
This work underscores the urgent need for more sophisticated defense mecha-
nisms capable of detecting fine-grained semantic perturbations, as traditional
filters based on perplexity or repetition are insufficient against this new class of
stealthy attacks.

14 Ziye Wang et al.
References
1. Alkhalaf, M., Yu, P., Yin, M., Deng, C.: Applying generative ai with retrieval
augmented generation to summarize and extract key clinical information from
electronic health records. Journal of biomedical informatics156, 104662 (2024)
2. Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R.,
McNamara, A., Mitra, B., Nguyen, T., Wang, S., Wang, X.: Ms marco: A human
generated dataset for research on machine reading comprehension and question
answering (2016), https://arxiv.org/abs/1611.09268
3. Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K.,
Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., Raffel, C.: Poi-
soning web-scale training datasets are easier than you might think. In: Pro-
ceedings of the IEEE Symposium on Security and Privacy (S&P). pp. 1369–
1387 (2023). https://doi.org/10.1109/SP49137.2023.10179267, https://ieeexplore.
ieee.org/document/10179267
4. Chen, J., Lin, H., Han, X., Sun, L.: Benchmarking large language models in
retrieval-augmented generation. In: Proceedings of the AAAI Conference on Artifi-
cialIntelligence.vol.38,pp.16715–16723(2024),https://arxiv.org/abs/2311.16109
5. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S.,
Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot
impressing gpt-4 with 90%* chatgpt quality (2023), https://vicuna.lmsys.org/
6. DeepSeek-AI: DeepSeek LLM: Scaling open-source language models with reinforce-
ment learning (2024), https://arxiv.org/abs/2401.02954
7. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep
bidirectional transformers for language understanding. In: Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).
pp. 4171–4186. Association for Computational Linguistics (2019). https://doi.org/
10.18653/v1/N19-1423, https://aclanthology.org/N19-1423
8. Ebrahimi, J., Rao, A., Lowd, D., Dou, D.: Hotflip: White-box adversarial examples
for text classification. In: Proceedings of the 56th Annual Meeting of the Associa-
tionforComputationalLinguistics(Volume2:ShortPapers).pp.382–387.Associa-
tion for Computational Linguistics (2018). https://doi.org/10.18653/v1/P18-2061,
https://aclanthology.org/P18-2061
9. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky,D.,Ness,R.O.,Larson,J.:Fromlocaltoglobal:AgraphRAGap-
proach to query-focused summarization (2024), https://arxiv.org/abs/2404.16130
10. Honnibal, M., Montani, I., Van Landeghem, S., Boyd, A., et al.: spacy: Industrial-
strength natural language processing in python (2020)
11. Huang, Y., Gupta, S., Xia, M., Li, K., Chen, D.: Catastrophic jailbreak of open-
source llms via exploiting generation (2023), https://arxiv.org/abs/2310.06987
12. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave,
E.: Contriever: Improving contrastive learning for unsupervised text retrieval. In:
Proceedings of the 39th International Conference on Machine Learning (ICML).
Proceedings of Machine Learning Research, vol. 162, pp. 9745–9758. PMLR (2022),
https://proceedings.mlr.press/v162/izacard22a.html
13. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave,
E.: Unsupervised dense information retrieval with contrastive learning. Transac-
tions on Machine Learning Research (2022), https://openreview.net/forum?id=
kXwdL1cWO5

RefineRAG 15
14. Jin, D., Jin, Z., Zhou, J.T., Szolovits, P.: Is BERT really robust? a strong base-
line for natural language attack on text classification and entailment. In: Pro-
ceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 8018–8025
(2020). https://doi.org/10.1609/aaai.v34i05.6304, https://ojs.aaai.org/index.php/
AAAI/article/view/6304
15. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti,
C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K.: Natural questions: A bench-
mark for question answering research. Transactions of the Association for Com-
putational Linguistics7, 453–466 (2019). https://doi.org/10.1162/tacl_a_00276,
https://aclanthology.org/Q19-1026
16. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.:
Retrieval-augmented generation for knowledge-intensive nlp tasks. In: Larochelle,
H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neu-
ral Information Processing Systems. vol. 33, pp. 9459–9474. Curran Asso-
ciates, Inc. (2020), https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf
17. Li, C., Zhang, J., Cheng, A., Ma, Z., Li, X., Ma, J.: Cpa-rag: Covert poisoning
attacks on retrieval-augmented generation in large language models (2025), https:
//arxiv.org/abs/2505.19864
18. Li, L., Ma, R., Guo, Q., Xue, X., Qiu, X.: Bert-attack: Adversarial attack against
bert using bert (2020), https://arxiv.org/abs/2004.09984
19. Perez, F., Ribeiro, I.: Ignore previous prompt: Attack techniques for language mod-
els (2022), https://arxiv.org/abs/2211.09527
20. Radford,A.,Wu,J.,Child,R.,Luan,D.,Amodei,D.,Sutskever,I.,etal.:Language
models are unsupervised multitask learners. OpenAI blog1(8), 9 (2019)
21. Rizqullah, M.R., Purwarianti, A., Aji, A.F.: Qasina: Religious domain question
answering using sirah nabawiyah (2023), https://arxiv.org/abs/2310.08102
22. Salemi, A., Zamani, H.: Evaluating retrieval quality in retrieval-augmented genera-
tion.In:Proceedingsofthe47thInternationalACMSIGIRConferenceonResearch
and Development in Information Retrieval. pp. 2185–2189 (2024). https://doi.org/
10.1145/3626772.3657754, https://dl.acm.org/doi/abs/10.1145/3626772.3657754
23. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bash-
lykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation
and fine-tuned chat models (2023), https://arxiv.org/abs/2307.09288
24. Wang, G., Li, Y., Liu, Y., Deng, G., Li, T., Xu, G., Liu, Y., Wang, H., Wang, K.:
Metmap: Metamorphic testing for detecting false vector matching problems in llm
augmented generation. In: Proceedings of the 2024 IEEE/ACM First International
Conference on AI Foundation Models and Software Engineering (FORGE). pp. 12–
23 (2024). https://doi.org/10.1145/3650105.3652297
25. Wei, A., Haghtalab, N., Steinhardt, J.: Jailbroken: How does llm safety training
fail? (2023), https://arxiv.org/abs/2307.02483
26. Xiong, L., Xiong, C., Li, Y., Tang, K.F., Liu, J., Bennett, P.N., Ahmed, J., Over-
wijk,A.:Approximatenearestneighbornegativecontrastivelearningfordensetext
retrieval. In: International Conference on Learning Representations (ICLR) (2021),
https://openreview.net/forum?id=zeFrfgyZln
27. Yepes, A.J., You, Y., Milczek, J., Laverde, S., Li, R.: Financial report chunking for
effective retrieval augmented generation (2024), https://arxiv.org/abs/2402.05131
28. Zhang, B., Yang, H., Zhou, T., Babar, M.A., Liu, X.Y.: Enhancing financial large
language models with retrieval-augmented generation (2023), https://arxiv.org/
abs/2308.14081

16 Ziye Wang et al.
29. Zhao, X., Liu, S., Yang, S.Y., Miao, C.: Medrag: Improving medical diagnosis with
retrieval-augmented generation (2023), https://arxiv.org/abs/2306.02322
30. Zhong, Z., Huang, Z., Wettig, A., Chen, D.: Poisoning retrieval corpora: How to
mislead retrieval-augmented generation. In: International Conference on Learning
Representations (ICLR) (2024), https://openreview.net/forum?id=1EB1fSj23k
31. Zou, W., Geng, R., Wang, B., Jia, J.: Poisonedrag: Knowledge corruption attacks
to retrieval-augmented generation of large language models (2024), https://arxiv.
org/abs/2402.07867
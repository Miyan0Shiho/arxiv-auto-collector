# Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method

**Authors**: Tianzhe Zhao, Jiaoyan Chen, Shuxiu Zhang, Haiping Zhu, Qika Lin, Jun Liu

**Published**: 2026-04-13 09:05:04

**PDF URL**: [https://arxiv.org/pdf/2604.11209v1](https://arxiv.org/pdf/2604.11209v1)

## Abstract
Large language models (LLMs) have achieved remarkable success across a wide range of applications especially when augmented by external knowledge through retrieval-augmented generation (RAG). Despite their widespread adoption, recent studies have shown that LLMs often struggle to perform faithful reasoning when conflicting knowledge is retrieved. However, existing work primarily focuses on conflicts between external knowledge and the parametric knowledge of LLMs, leaving conflicts across external knowledge largely unexplored. Meanwhile, modern RAG systems increasingly emphasize the integration of unstructured text and (semi-)structured data like knowledge graphs (KGs) to improve knowledge completeness and reasoning faithfulness. To address this gap, we introduce ConflictQA, a novel benchmark that systematically instantiates conflicts between textual evidence and KG evidence. Extensive evaluations across representative LLMs reveal that, facing such cross-source conflicts, LLMs often fail to identify reliable evidence for correct reasoning. Instead, LLMs become more sensitive to prompting choices and tend to rely exclusively on either KG or textual evidence, resulting in incorrect responses. Based on these findings, we further propose XoT, a two-stage explanation-based thinking framework tailored for reasoning over heterogeneous conflicting evidence, and verify its effectiveness with extensive experiments.

## Full Text


<!-- PDF content starts -->

Exploring Knowledge Conflicts for Faithful LLM Reasoning:
Benchmark and Method
Tianzhe Zhao
School of Computer Science and
Technology, Xi’an Jiaotong University
Xi’an, China
ztz8758@foxmail.comJiaoyan Chen
Department of Computer Science,
The University of Manchester
Manchester, United Kindom
jiaoyan.chen@manchester.ac.ukShuxiu Zhang
Hunan University
Changsha, China
zsx@hun.edu.cn
Haiping Zhu
School of Computer Science and
Technology, Xi’an Jiaotong University
Xi’an, China
zhuhaiping@xjtu.edu.cnQika Lin∗
National University of Singapore
Singapore, Singapore
linqika@nus.edu.sgJun Liu∗
School of Computer Science and
Technology, Xi’an Jiaotong University
Xi’an, China
liukeen@xjtu.edu.cn
Abstract
Large language models (LLMs) have achieved remarkable success
across a wide range of applications especially when augmented by
external knowledge through retrieval-augmented generation (RAG).
Despite their widespread adoption, recent studies have shown that
LLMs often struggle to perform faithful reasoning when conflicting
knowledge is retrieved. However, existing work primarily focuses
on conflicts between external knowledge and the parametric knowl-
edge of LLMs, leaving conflicts across external knowledge largely
unexplored. Meanwhile, modern RAG systems increasingly em-
phasize the integration of unstructured text and (semi-)structured
data like knowledge graphs (KGs) to improve knowledge complete-
ness and reasoning faithfulness. To address this gap, we introduce
ConflictQA, a novel benchmark that systematically instantiates
conflicts between textual evidence and KG evidence. Extensive eval-
uations across representative LLMs reveal that, facing such cross-
source conflicts, LLMs often fail to identify reliable evidence for
correct reasoning. Instead, LLMs become more sensitive to prompt-
ing choices and tend to rely exclusively on either KG or textual
evidence, resulting in incorrect responses. Based on these findings,
we further proposeXoT, a two-stage explanation-based thinking
framework tailored for reasoning over heterogeneous conflicting
evidence, and verify its effectiveness with extensive experiments.
CCS Concepts
•Computing methodologies →Reasoning about belief and knowl-
edge.
Keywords
Retrieval-Augmented Generation, Large Language Models, Cross-
source Knowledge Conflicts, Explanation-based Thinking
1 Introduction
Large Language Models (LLMs) have demonstrated remarkable
achievements in a wide range of applications [ 21,35,40]. Retrieval-
Augmented Generation (RAG) further extends LLMs by integrating
∗Corresponding authors.
,
2026.external knowledge sources, enabling responses to be grounded in
retrieved content rather than generated solely from the model’s in-
ternal parameters [ 6,7,12]. By leveraging multiple heterogeneous
knowledge sources, such as unstructured textual documents along-
side (semi-)structured knowledge graphs (KGs) [ 11] and tables [ 4],
LLMs are better equipped to mitigate hallucination and support
knowledge-intensive tasks.
Despite the growing adoption of RAG systems, their reliance on
external knowledge also introduces new challenges. In real-world
scenarios, external knowledge may be outdated, noisy, or intention-
ally manipulated, leading to conflicting or contradictory evidence
during LLM reasoning [14, 29, 36]. To deal with such issues, some
recent studies have begun to investigate the faithfulness of LLM
reasoning in the presence of conflicting knowledge [ 10,19,31].
However, these existing studies predominantly focus on inconsis-
tencies between external evidence and LLMs’ parametric knowl-
edge [ 10,29], or consider conflicts among external evidence that are
synthetically constructed within a single knowledge source [ 19].
In contrast, cross-source conflicts from heterogeneous external
knowledge sources remain largely unexplored.
To address this gap, in this study we investigate conflicts that
arise between heterogeneous external knowledge sources, using
textual documents and KGs, two of the most widely adopted ex-
ternal sources in modern RAG systems, as a case. Textual docu-
ments (e.g., Wikipedia articles and passages) are most commonly
retrieved to provide informative, context-rich descriptions, while
KGs, which represent relational facts in the form of triples, i.e.,
(head, relation, tail), have also been increasingly incorporated into
RAG systems to support complex reasoning, especially in domains
that require high trustworthiness, such as legal judgment and med-
ical diagnosis [ 1,27,28,41]. Beyond serving as standalone knowl-
edge sources, recent studies have also highlighted the necessity
of jointly integrating textual documents and KGs to provide more
comprehensive knowledge for LLM reasoning in real-world sce-
narios [ 11,13,18,32,33]. Yet, how LLMs assess and resolve incon-
sistencies that arise between textual descriptions and KG triples
under such settings remains unclear. As shown in Figure 1, when
answering the question“Which city is the movie awarded the 2018
Golden Lion filmed in?”, the KG fact about the location ofColoniaarXiv:2604.11209v1  [cs.CL]  13 Apr 2026

, , Tianzhe Zhao et al.
KG Evidence:
Textual Evidence:(Roma, filmAward, 2018 Golden Lion)
(Roma, filmPlace, Colonia Roma)
(Colonia Roma, locatedIn, Rome)
Roma is a 2018 historical drama 
film ... It is a ... take on Cuarón's 
upbringing in Mexico City's Colonia 
Roma neighborhood ...  Which city is the movie awarded the 2018 Golden Lion filmed in?
LLM
Response: Knowledge 
Graph (KG)
DocumentsRetrieveQuestion:
Based on ..., the 
answer is Rome!Conflict!
✘
Figure 1: Demonstration of cross-source knowledge conflict
in LLM reasoning with RAG.
Romaconflicts with textual evidence (e.g.,"Mexico City"). As a re-
sult, LLMs may struggle to identify the more reliable evidence and
generate satisfactory responses. However, there is a shortage of
benchmarks specifically designed to study such conflicts in multiple
heterogeneous sources, which hinders the systematic evaluation of
LLMs’ reasoning behavior and the development of more faithful
reasoning methods.
In this work, we construct a novel question answering (QA)
benchmark, called ConflictQA, to systematically investigate faithful
LLM reasoning under conflicting evidence retrieved from hetero-
geneous sources. Following previous works [ 19,29], ConflictQA
operates in a post-retrieval manner, where each question is paired
with evidence from both textual documents and KGs, provided re-
spectively as passages and sets of triples. To simulate conflicts, we
instantiate textual and KG evidence that provides inconsistent cues
for the same fact that the answer depends on. As exemplified in
Figure 1, the evidence in triples and texts lead to different answers.
Motivated by real-world reasoning requirements, ConflictQA cov-
ers two settings: (i) Non-complementary (Non-COMP) cases, where
evidence from either textual documents or KGs alone is adequate to
answer the question, and (ii) complementary (COMP) cases, where
answering the question requires jointly reasoning over both textual
and KG evidence, as illustrated in Figure 1, where the award infor-
mation is only available in the KG evidence. Within each setting,
we further distinguish which source provides the negative evidence
supporting incorrect answers, enabling fine-grained analysis of
how LLMs assess evidence reliability and resolve conflicts.
With the ConflictQA benchmark, we conduct comprehensive
evaluations to examine LLM reasoning behavior under conflicts
arising from heterogeneous external sources. We evaluate 12 rep-
resentative models, covering both general-purpose and reasoning-
specialized LLMs. Our results reveal a consistent limitation across
models: when confronted with conflicting evidence from differ-
ent sources, LLMs fail to reliably determine which evidence to
trust, leading to incorrect answers in both non-complementary and
complementary settings. Through analyses on evidence ordering
and prompting strategies, we also observe an interesting tendency:
LLMs exhibit a systematic bias toward believing concise KG triples
when prompted to directly generate answers, even when the triple
is incorrect. In contrast, chain-of-thought (CoT) prompting that
elicits step-by-step reasoning may shift models’ decisions toward
textual evidence. These findings further suggest that current LLMs
lack a robust mechanism for resolving cross-source conflicts or
assessing evidence reliability.Based on these observations, we further propose a two-stage
explanation-based thinking framework (XoT) for more trustful LLM
reasoning facing conflicting evidence. Instead of prompting LLMs
to produce answers directly, XoT first encourages the model to
enumerate all plausible answer candidates supported by different
sources, together with explicit explanations for each candidate. The
final answer is then selected based on the aggregated explanations.
By separating candidate enumeration from answer selection, XoT
helps LLMs mitigate premature bias toward a certain type of ev-
idence and promotes more balanced reasoning when faced with
conflicting information. Experiments on our ConflictQA benchmark
demonstrates the effectiveness of XoT, as it achieves improved per-
formance across a range of LLMs. For example, in the complemen-
tary setting where KG evidence is misleading, XoT yields relative
improvements of20%and20%on GPT-4o in terms of F1 score and
exact match, respectively, compared to a conflict-aware QA prompt;
On Open-Mistral-7B, the F1 score achieved by XoT is surprisingly
almost threefold that obtained with the QA prompt.
In summary, the main contributions of this work lie in three
aspects:
•We present the first systematic study of faithful LLM reasoning
under cross-source knowledge conflicts, and construct ConflictQA,
a novel benchmark that explicitly instantiates conflicts between
textual and KG evidence1.
•We conduct comprehensive evaluations of 12 representative LLMs
on ConflictQA and analyze their reasoning behaviors under con-
flicting evidence, revealing consistent failure modes across different
models and settings.
•Based on these analyses, we further propose XoT, a two-stage
explanation-based thinking framework, that improves reasoning
correctness under heterogeneous and misleading evidence across
most evaluated models.
2 Problem Definition
2.1 LLM Reasoning with Multiple Evidence
We study a multi-source reasoning task for LLMs, where the model
is provided with external evidence from heterogeneous knowledge
sources, including textual documents and KGs. For each question
𝑞, the available evidence consists of a set of textual passages 𝐸text
and a set of KG triples 𝐸KG. The textual evidence is represented as
a collection of passages, i.e., 𝐸text={𝑝 1,𝑝2,...,𝑝𝑚}, where each
passage is a description relevant to the question. The KG evidence
is represented as a set of KG triples, i.e., 𝐸KG={(𝑒ℎ
𝑖,𝑟𝑖,𝑒𝑡
𝑖)}𝑛
𝑖=1,
where𝑒ℎ
𝑖and𝑒𝑡
𝑖denote the head and tail entities, respectively, and
𝑟𝑖denotes their relation.
Given𝐸textand𝐸KG, the LLM is required to generate an answer
setAgrounded in the provided evidence. The process can be ab-
stracted as:
A=𝑓𝐿𝐿𝑀 Prompt(𝑞,𝐸 text,𝐸KG),(1)
where𝑓𝐿𝐿𝑀denotes an LLM, and Prompt(·) represents the prompt-
ing strategy that feeds the question and the evidence into the LLM.
1The ConflictQA benchmark is available at https://github.com/Tianzhe26/ConflictQA.

Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method , ,
 What is the birthplace of the 
director of the movie “Roma” ? What is the birthplace of the 
director of the movie “Roma” ?Question: Who directs the movie “Roma”?  Who directs the movie “Roma”?Question:
Incorrect Answer GenerationIncorrect Answers 
Conflicting Triple Constuction Negative Triples Negative TextsGraph Search
Graph SearchRoma Alfonso 
CuaróndirectedBy Expert Judgement
Roma Alfonso 
CuaróndirectedBy
Positive KG Evidence
QuestionRoma Alfonso 
CuaróndirectedBy birthpalce
Mexico City
Roma Colona 
Romafilmpalce locatedIn
Mexico CityExpert Judgement
Positive KG EvidenceRoma Alfonso 
CuaróndirectedBy birthpalce
Mexico CityWikipedia Collection
Wikipedia CollectionAlfonso Cuaróns  is a Mexican filmmaker , ...
Roma is ... directed by Alfonso Cuarón , ...Content Verification
Yes !
Mexico City is the capital  ...Content Verification
Yes !Positive Multi -source EvidenceRoma Alfonso 
CuaróndirectedBy
Alfonso Cuaróns  is a 
Mexican filmmaker , ...
No !LLM Rewrite
Mexico City  , ... Alfonso Cuarón  
was born in Mexico City , ...
Positive Multi -source EvidenceRoma Alfonso 
CuaróndirectedBy birthpalce
Mexico City
Mexico City  , ...Alfonso 
Cuarón , ...Roma is ...  directed 
by Alfonso Cuarón , ...
Conflict Generation ModuleConflict Generation Module Who directs the movie “Roma”?  Who directs the movie “Roma”?Question:
 What is the birthplace of the 
director of the movie “Roma” ? What is the birthplace of the 
director of the movie “Roma” ?Question:Conflict Generation 
ModuleConflict Generation 
ModuleFact: ( Roma, directedby, James Cameron )
Text: Alfonso Cuaróns  is a Mexican filmmaker , ...Fact: ( Roma, directedby, Alfonso Cuaróns )
Text: James Cameron  is a Mexican filmmaker , ...
Fact: ( Roma, directedby, Alfonso Cuarón ), 
    (Alfonso Cuarón , birthpalce, Italy)
Text: Mexico City  , ...Alfonso Cuarón  was born 
in Mexico City ...Text: Roma is a 2018 historical drama film 
directed by Alfonso Cuarón , ...Fact: ( Alfonso Cuarón , birthpalce, Mexico City )
Italy , ...Alfonso Cuarón  was born ...Conflicting Evidence Instantiation Conflicting Evidence InstantiationPositive Reasoning Scenarios
Question: Who directs the movie “Roma”?
Fact: ( Roma, directedby, Alfonso Cuarón )
Text: Alfonso Cuaróns  is a Mexican filmmaker , ...
Question: What is the birthplace of the director of 
the movie “Roma” ?
Fact: ( Roma, directedby, Alfonso Cuarón )
Text: Mexico City  , ...Alfonso Cuarón  was born ...
Question: What is the birthplace of the director of 
the movie “Roma” ?
Text: Roma is a 2018 historical drama film 
directed by Alfonso Cuarón , ...
Fact: ( Alfonso Cuarón , birthpalce, Mexico City )
Conflicting Text GenerationContent VerificationExample of posit ive KG evidence and text evidence 
in the non-complementary settingPositive Evidence CollectionPositive Evidence Collection
Roma James 
CamrrondirectedBy
Roma James 
CamrrondirectedByJames Cameron  is a 
filmmaker , ... directs 
Roma
Negatvie Evidence
Alfonso 
Cuarónbirthpalce
ItalyItaly ... Alfonso 
Cuarón  was born in 
Italy
Negatvie EvidenceExample s of positive KG evidence and text evidence 
in the complementary settingKG
KG
Example s of conflicting  KG evidence and text evidence in the non-complementary setting
Example s of conflicting  KG evidence and text evidence in the complementary setting
Figure 2: The pipeline for constructing the ConflictQA benchmark, mainly including positive and conflicting evidence con-
struction as well as examples under both Non-COMP and COMP settings.
2.2 LLM Reasoning with Conflicting Evidence
In this work, we consider multiple evidence reasoning in which the
provided evidence contains conflicting information and different ev-
idence lead to different answers. Specifically, we define two conflict
scenarios based on which source provides misleading information.
One setting isTripleConf, where all the textual evidence 𝐸textlead to
the correct answer, while some KG evidence in 𝐸KGhas inconsistent
information and leads to incorrect answers. The other setting is
TextConf, where all the KG evidence 𝐸KGare correct, while some
textual evidence in 𝐸textis inconsistent with some KG evidence and
leads to incorrect answers. For convenience, we use 𝐸+
textand𝐸−
text
to denote positive textual evidence, and textual evidence leading
to inconsistency (i.e., conflicting evidence), respectively. Similarly,
for the KG evidence, we use 𝐸+
KGand𝐸−
KG. Under both settings, the
LLM is expected to distinguish and exclude the conflicting evidence,
and infer the correct answer.
3 ConflictQA Benchmark
3.1 Benchmark Construction
In this section, we detail the construction of the ConflictQA bench-
mark. ConflictQA is built upon two widely used KG question an-
swering datasets, i.e., WebQSP [ 38] and CWQ [ 30], where each
question is associated with verified golden answers grounded in
the KG. Based on how answering the question relies on differ-
ent knowledge sources, we develop two settings for ConflictQA:Complementary(COMP) andNon-complementary(Non-COMP). In
COMP, answering a question requires jointly leveraging evidence
from both textual documents and KGs, whereas in Non-COMP, evi-
dence from either source alone is sufficient to derive an answer. In
practice, we develop these two settings by exploiting the inherent
reasoning structure of the original datasets: multi-hop questions
and simple single-hop questions are used to construct data of COMP
and Non-COMP, respectively.
As shown in Figure 2, the construction of ConflictQA consists of
two main steps:(i)Positive Evidence Collection, where we retrieve
positive evidence for each question from both the KG and Wikipedia
pages, and(ii)Conflicting Evidence Instantiation, where we employ
a Conflict Generation Module to generate negative KG and textual
evidence that lead to incorrect answers. With the question, we can
directly use their positive evidence for reasoning samples without
conflict (i.e., positive reasoning scenarios), and combine positive
and negative evidence for reasoning samples with conflict.
3.1.1 Positive Evidence Collection.The construction of Conflic-
tQA starts with collecting positive KG and textual evidence that
consistently support the correct answer for each question. This
stage establishes reliable evidence from each source as the basis for
subsequent conflict construction.
KG Evidence.Given a question 𝑞, we perform breadth-first search
(BFS) on the underlying KG to retrieve triple paths that connect the
entity𝑒𝑞mentioned in 𝑞to correct answers A𝑞. These paths serve

, , Tianzhe Zhao et al.
as candidate KG evidence of 𝑞, which needs further verification.
For example, as shown in the green box of Figure 2, the path(Roma,
filmPlace, Colonia Roma) →(Colonia Roma, locatedIn, Mexico City)
is irrelevant to the question"What is the birthplace of the director of
the movie Roma?", as it describes the filming location rather than the
director’s birthplace. To ensure factual correctness and relevance,
three human experts independently assess whether each candidate
path can validly support answering 𝑞. Only paths unanimously
judged as valid are retained; if all candidate paths for a question are
deemed invalid, the corresponding sample is discarded. We denote
the resulting set of validated factual triples as𝐸 𝐾𝐺(𝑞).
Textual Evidence.For each question 𝑞, we construct textual de-
scriptions grounded in Wikipedia that are consistent with correct
answers. Specifically, for each answer 𝑎𝑖∈A𝑞, we extract the in-
troductory summary paragraph from the corresponding Wikipedia
page, denoted as 𝑝(𝑎𝑖). Considering 𝑝(𝑎𝑖)may omit some answer-
related facts in the KG evidence, we employ an LLM2to check
whether each triple in 𝐸𝐾𝐺(𝑞)that involves 𝑎𝑖is supported by 𝑝(𝑎𝑖).
If such triples are not supported, 𝑝(𝑎𝑖)is minimally rewritten by
the LLM to incorporate the missing facts while preserving factual
correctness. As a result, we obtain a set of textual descriptions,
each of which fully covers the information required to support
its corresponding answer 𝑎𝑖∈A𝑞. For questions in CMOP, we
additionally extract the introductory summary paragraph of the
question entity 𝑒𝑞, denoted as 𝑝(𝑒𝑞), which is used to construct
complementary cases where the textual evidence can only provide
contextual information.
Positive Reasoning Scenarios.With the retrieved KG triples and
Wikipedia texts, we organize them for each question to construct
positive reasoning scenarios. For Non-COMP questions, which in-
volve single-hop reasoning, all constructed evidence directly sup-
ports the correct answers. Accordingly, all triples in 𝐸𝐾𝐺(𝑞)form
the positive KG evidence 𝐸+
𝐾𝐺(𝑞)for a Non-COMP question 𝑞, while
the verified texts{𝑝(𝑎𝑖)}constitute the positive textual evidence
𝐸+
text(𝑞). In COMP, we consider two positive reasoning scenarios
based on how different evidence sources cover the correct answers.
In theTriplePosscenario, we treat 𝑝(𝑒𝑞)as the positive textual
evidence𝐸+
text(𝑞), while the positive KG evidence 𝐸+
𝐾𝐺(𝑞)is con-
structed as the subset of triples in𝐸 𝐾𝐺(𝑞)that involve the correct
answers inA𝑞. In this case, the textual evidence only provides back-
ground context, and the correct answers are primarily supported by
KG triples. In contrast, in theTextPosscenario, we construct 𝐸+
text(𝑞)
using the verified answer-related texts {𝑝(𝑎𝑖)}, which explicitly
cover the information required to support the correct answers. Ac-
cordingly, the positive KG evidence 𝐸+
𝐾𝐺(𝑞)consists of the triples
in𝐸𝐾𝐺(𝑞)that are not directly involved with the correct answers.
3.1.2 Conflicting Evidence Instantiation.During this stage, we first
employ a Conflict Generation Module to produce negative evidence
for each reasoning sample, and then instantiate conflicting reason-
ing scenarios: TextConf and TripleConf as introduced in Section 2.2.
Conflict Generation Module.This module aims to generate nega-
tive KG and textual evidence that contradicts the positive evidence.
As illustrated in the red box in Figure 2, given a question 𝑞, we
first prompt an LLM to generate a set of semantically plausible
2In practice, we use GPT-4o to perform all LLM-related tasks during the construction
of ConflictQA.Table 1: Statistics of ConflictQA.
Conflict #Avg Triples #Avg Tokens #Sample
Non-Complementary
Positive 1.84 390.41 802
TripleConf 1.84 390.41 802
TextConf 1.84 330.82 802
Complementary
TextPos 1.05 534.92 430
TripleConf 3.57 534.92 430
TriplePos 2.54 50.2 430
TextConf 2.54 512.93 430
but incorrect answers, denoted as ˆA𝑞={ˆ𝑎1,ˆ𝑎2,..., ˆ𝑎𝑛}, where the
number of incorrect answers 𝑛matches the cardinality of the cor-
rect answer setA𝑞. These incorrect answers serve as adversarial
targets for synthesizing answer-inconsistent information. To gen-
erate negative KG triples, we modify the KG triples associated with
the correct answers. Specifically, for each ˆ𝑎𝑖∈ˆA𝑞, we replace the
correct answer entity 𝑎𝑖∈A𝑞in the corresponding triples with
ˆ𝑎𝑖, while preserving the original relational structure. In this way,
the resulting triples encode plausible but incorrect answer infor-
mation. For textual descriptions, we prompt an LLM to generate
Wikipedia-style texts based on the synthesized negative KG triples.
The generated texts explicitly reflect the relational facts encoded
in the negative triples, and are therefore semantically inconsistent
with the corresponding positive KG evidence.
Conflicting Reasoning Scenarios.For Non-COMP questions, af-
ter collecting negative evidence via the Conflict Generation Module,
we develop TripleConf setting by combining negative KG evidence
with the original positive textual evidence for each reasoning sam-
ple. Similarly, the TextConf setting is implemented by literally pair-
ing negative textual evidence with the corresponding positive KG
evidence. In the COMP setting, conflicting evidence is instantiated
based on the predefined positive reasoning scenarios. Specifically,
under the TriplePos scenario, we additionally include negative tex-
tual evidence with original textual evidence set to form 𝐸−
𝑡𝑒𝑥𝑡, and
keep the positive KG evidence 𝐸+
𝐾𝐺unchanged. As a result, we con-
struct the corresponding conflicting reasoning scenario TextConf,
where answering the question still requires joint reasoning over
both KG and textual evidence, whereas reasoning based solely on
textual evidence leads to incorrect answers. Similarly, under the
TextPosscenario, we form TripleConf by adding a set of negative
KG triples to each question’s KG evidence, without altering the
positive textual evidence𝐸+
text.
3.2 Benchmark Usage and Statistics
ConflictQA contains both positive reasoning scenarios and their
conflict-induced variants. In evaluation, we compare model perfor-
mance on positive and conflicting data to measure the performance
degradation caused by cross-source conflicts. Table 1 summarizes
the statistics of ConflictQA, including the number of samples and
the distribution of KG and textual evidence across different settings.

Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method , ,
Table 2: Performance (%) of LLMs on ConflictQA under the non-complementary setting. Pos and Conf denote performance
with positive and conflicting evidence, respectively. Δindicates the resulting performance degradation. Bold and underline
values denote optimal and sub-optimal results, respectively. Pink cells highlight cases with substantial degradation.
Large Language ModelTripleConf TextConf
F1 EM F1 EM
Pos ConfΔPos ConfΔ Pos ConfΔPos ConfΔ
General LLMs
Qwen3-8B 86.01 21.85 64.16 75%↓ 76.28 7.61 68.67 90%↓ 86.01 76.249.77 11%↓ 76.28 52.4423.84 31%↓
Llama-3.1-8B-Instruct 67.30 36.8230.48 45%↓ 50.75 19.2031.55 62%↓ 67.30 63.393.91 6%↓ 50.75 41.159.60 19%↓
Llama-3.1-70B-Instruct 84.72 46.8337.89 45%↓ 79.80 32.6747.13 59%↓ 84.72 63.1021.62 26%↓ 79.80 48.9630.84 39%↓
Open-Mistral-7B 81.45 29.70 51.75 64%↓ 65.59 13.47 52.12 79%↓ 81.45 67.1014.35 18%↓ 65.59 47.0718.52 28%↓
Mistral-Large-2512 84.91 53.6531.26 37%↓ 78.18 29.3048.88 63%↓ 84.91 61.5023.41 28%↓ 78.18 32.17 46.01 59%↓
Deepseek-V3.2 90.4243.6046.82 52%↓ 86.41 30.3056.11 65%↓ 90.4266.6423.78 26%↓ 86.41 53.4332.98 38%↓
GPT-3.5-Turbo-0125 88.20 53.9334.27 39%↓ 83.17 36.41 46.76 56%↓ 88.20 62.65 25.55 29%↓ 83.17 43.7739.40 47%↓
GPT-4o 88.44 54.39 34.05 39%↓ 85.41 32.6752.74 62%↓ 88.4470.4318.01 20%↓ 85.41 46.5738.84 45%↓
GPT-5.1 90.17 46.7443.43 48%↓ 87.2832.0555.23 63%↓ 90.17 68.45 21.72 24%↓ 87.2851.8835.40 41%↓
Reasoning LLMs
Qwen3-30B-A3B-Thinking 84.43 44.4439.99 47%↓ 79.18 32.6146.57 59%↓ 84.43 60.91 23.52 28%↓ 79.18 48.6930.49 39%↓
Deepseek-V3.2-Thinking 84.1854.6329.55 35%↓ 79.30 33.2346.07 58%↓ 84.18 62.7321.45 25%↓ 79.30 40.15 39.15 49%↓
o3-mini 90.06 51.0039.06 43%↓ 86.91 38.0348.88 56%↓ 90.06 66.6623.40 26%↓ 86.91 52.5634.35 40%↓
Table 3: Performance (%) under the complementary setting. Its settings are consistent with Table 2.
Large Language ModelTripleConf TextConf
F1 EM F1 EM
Pos ConfΔPos ConfΔ Pos ConfΔPos ConfΔ
General LLMs
Qwen3-8B 53.26 26.26 27.00 51%↓ 36.98 8.14 28.84 78%↓ 88.99 68.73 20.26 23%↓ 78.63 46.9831.65 40%↓
Llama-3.1-8B-Instruct 52.79 36.4516.34 31%↓ 38.37 17.6720.70 54%↓ 80.39 42.56 37.83 47%↓ 64.19 21.63 42.56 66%↓
Llama-3.1-70B-Instruct 57.76 42.1415.62 27%↓ 46.74 31.1615.58 33%↓ 83.00 47.46 35.54 43%↓ 74.65 33.7240.93 55%↓
Open-Mistral-7B 43.25 13.34 29.91 69%↓ 30.70 6.5124.19 79%↓ 73.98 51.8122.17 30%↓ 59.30 38.1421.16 36%↓
Mistral-Large-2512 61.01 51.479.54 16%↓ 47.91 29.7718.14 38%↓ 82.12 52.0030.12 37%↓ 72.09 28.84 43.25 60%↓
Deepseek-V3.2 59.31 36.8322.48 38%↓ 52.79 26.74 26.05 49%↓ 83.56 61.0822.48 27%↓ 74.19 51.86 22.33 30%↓
GPT-3.5-Turbo-0125 63.48 42.5320.95 33%↓ 50.93 30.7020.23 40%↓ 85.55 59.2326.32 31%↓ 78.60 41.4037.20 47%↓
GPT-4o 65.63 50.97 14.66 22%↓ 55.35 33.7221.63 39%↓ 87.62 61.1026.52 30%↓ 80.00 44.4235.58 44%↓
GPT-5.1 66.8745.8521.02 31%↓ 58.1433.7224.42 42%↓ 89.26 69.1820.08 23%↓ 82.79 55.5827.21 33%↓
Reasoning LLMs
Qwen3-30B-A3B-Thinking 53.55 44.848.71 16%↓ 43.49 34.888.61 20%↓ 84.28 55.3528.93 34%↓ 76.51 44.4232.09 42%↓
Deepseek-V3.2-Thinking 64.1255.438.69 14%↓ 54.88 36.98 17.90 33%↓ 89.8256.3433.48 37%↓ 81.86 39.3042.56 52%↓
o3-mini 58.87 46.1112.76 22%↓ 48.1437.6710.47 22%↓ 83.70 60.0023.70 28%↓ 74.65 48.6026.05 35%↓
4 Evaluation and Analysis
4.1 LLMs for Benchmarking
To ensure a comprehensive evaluation, we benchmark 12 popular
large language models (LLMs), which can be categorized into two
groups based on their intended inference characteristics. The first
group consists of general-purpose instruction-following models,
referred to asGeneral LLMin this work. This group includes Qwen3-
8B [37], LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct [ 5], Open-
Mistral-7B [ 9], Mistral-Large-2512 [ 20], Deepseek-V3.2 [ 15], GPT-
3.5-Turbo [ 23], GPT-4o [ 24], and GPT-5.1 [ 25]. The second group,
denoted asReasoning LLM, comprises models explicitly designedto enhance multi-step reasoning capability. They are Qwen3-30B-
A3B-Thinking, Deepseek-V3.2-Thinking, and OpenAI o3-mini [ 26].
4.2 Evaluation Protocol and Metrics
We evaluate all LLMs under a zero-shot setting, without any task-
specific fine-tuning on ConflictQA. For each question, the model is
provided with the corresponding factual and textual evidence and
is required to generate answers directly. Considering the presence
of conflicting information, we adopt a conflict-aware QA prompt.
Specifically, the LLM is explicitly informed that“The provided triples
and texts may contain conflicting or inconsistent information”, en-
couraging the model to reason cautiously. To mitigate potential

, , Tianzhe Zhao et al.
order sensitivity when presenting multiple evidence sources, we
evaluate each case twice using two different evidence orders and
report the averaged results when it is not specified. Meanwhile, we
particularly investigate the impact of the order of evidence, and
analyse the results in Section 4.4.
For evaluation, we utilize the macro-F1 and Exact Match (EM) as
our metrics. Macro-F1 is computed by averaging F1 scores across
questions, where the F1 score of each question is computed based
on the Precision and Recall of the generated answers in compari-
son with the ground truth answer set, while EM is the proportion
of questions for which the generated answers exactly match the
ground-truth answer set.
4.3 Performance under Knowledge Conflict
We evaluate the performance of LLMs under both Non-COMP and
COMP settings, as reported in Table 2 and Table 3. Our analysis
mainly focuses on the following three aspects.
4.3.1 Overall Performance.When the evidence include knowledge
conflicts, we observe a consistent and substantial performance drop
across all the evaluated LLMs under both Non-COMP and COMP
settings. Notably, under the Non-COMP setting, conflicts intro-
duced by factual evidence lead to a dramatic performance drop,
with Exact Match (EM) decreasing by nearly or more than 60% for
all LLMs in TripleConf. Under the COMP setting, the performance
of LLMs also decreases pronouncedly under conflicting evidence.
For example, under the TripleConf scenario, Qwen3-8B and Open-
Mistral-7B suffer severe EM degradation, with relative drops of
78% and 79%, respectively. Regrarding the TextConf setting with
complementary reasoning, 9 out of the 12 evaluated LLMs exhibit
relative EM degradation exceeding 40%. From these observations,
we conclude that LLMs themselves struggle to identify and pri-
oritize more reliable evidence for robust reasoning when facing
inconsistent evidence.
4.3.2 Comparison of Different Conflict Types.Under both Non-
COMP and COMP settings, conflicts introduced by KG evidence
consistently result in more severe performance degradation than
those introduced by textual evidence. For example, despite its strong
reasoning capability, GPT-5.1 suffers a performance drop of 43.43
under factual conflicts under the Non-COMP setting, which is more
than twice the degradation observed under textual conflicts. A
similar trend is also observed in the COMP setting among general
LLMs. However, for reasoning LLMs, the relative performance drop
caused by negative KG evidence is conversely weaker than that
caused by negative textual evidence.
These results suggest that LLMs tend to rely more heavily on
concise and explicit triple-based information during reasoning. Con-
sequently, corrupting factual evidence leads to pronounced failures,
whereas conflicting textual evidence generally has a weaker impact
on the final prediction.
4.3.3 Comparison of Different LLMs.There is considerable vari-
ation among different LLMs in their robustness to conflicting evi-
dence. Overall, reasoning LLMs achieve better performance than
general instruction-following LLMs under most conflict scenarios,
demonstrating improved faithfulness to cross-source inconsisten-
cies. However, this advantage is not uniform across all conditions.Table 4: Performance gap caused by evidence ordering in
the non-complementary setting. Light-blue cells indicate
negative gaps.
Large Language ModelTripleConf TextConf
ΔF1ΔEMΔF1ΔEM
Qwen3-8B 19.74 8.73 -0.30 -0.87
Llama-3.1-8B-Instruct -4.72 4.74 -0.33 -9.48
Llama-3.1-70B-Instruct -1.41 -0.50 -4.42 -6.86
Open-Mistral-7B 9.79 8.48 7.87 8.10
Mistral-Large-2512 1.93 6.98 5.12 -1.74
Deepseek-V3.2 -7.71 -3.74 -3.32 -7.60
GPT-3.5-Turbo-0125 0.63 4.98 -3.30 6.49
GPT-4o 4.14 10.48 0.74 1.62
GPT-5.1 -4.82 -0.75 0.49 0.25
Qwen3-30B-A3B-Thinking -5.08 -3.61 -0.15 -1.12
Deepseek-V3.2-Thinking -4.20 -2.62 -1.53 -4.48
o3-mini -0.65 0.24 0.20 0.87
In particular, under complementary scenarios with textual conflicts,
we observe cases where Deepseek-V3.2-Thinking degrades more
than its non-thinking counterpart Deepseek-V3.2. This is likely
because reasoning LLMs rely more heavily on textual information
during reasoning, making them more susceptible to misleading
signals in conflicting textual evidence. Moreover, in general LLMs,
we find that LLMs with fewer parameters, such as Qwen3-8B and
Open-Mistral-7B, exhibit substantially larger performance degrada-
tion across multiple conflict settings, indicating limited robustness
to inconsistent information. Within the same model family (e.g.,
the LLaMA-3.1 series), larger model sizes generally lead to im-
proved reasoning results in the presence of conflicting evidence.
Meanwhile, it is worth noting that although LLaMA-3.1-8B-Instruct
exhibits the smallest performance drop under the sufficient setting
with textual conflicts, this behavior is largely attributable to its
relatively weak performance on positive instances, rather than the
robust ability to handle conflicting evidence.
4.4 Effect of Evidence Ordering
In this section, we study how evidence ordering affects LLM perfor-
mance with the direct conflict-aware prompt under the Non-COMP
setting. Specifically, we record each LLM’s results when the con-
flicting evidence is present before and after the correct evidence,
and analyze the performance gap.
From the results shown in Table 4, we observe that the perfor-
mance gap varies considerably across models and conflict types.
Overall, changing the evidence order does not consistently enable
faithful reasoning across all LLMs. For LLMs that are particularly
vulnerable to conflicting evidence, such as Qwen3-8B and Open-
Mistral-7B (discussed in Section 4.3.3), placing correct evidence
before conflicting evidence can lead to notable performance im-
provements. When comparing the two categories of LLMs, reason-
ing LLMs generally exhibit smaller performance gaps, indicating
that they are less affected by evidence ordering, whereas general

Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method , ,
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni0000001b/uni00000025
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057 /uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001a/uni00000013/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000032/uni00000053/uni00000048/uni00000051/uni00000010/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001a/uni00000025/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000002f/uni00000044/uni00000055/uni0000004a/uni00000048/uni00000010/uni00000015/uni00000018/uni00000014/uni00000015/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000011/uni00000015
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000037/uni00000058/uni00000055/uni00000045/uni00000052/uni00000010/uni00000013/uni00000014/uni00000015/uni00000018/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052 /uni0000002a/uni00000033/uni00000037/uni00000010/uni00000018/uni00000011/uni00000014
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000016/uni00000013/uni00000025/uni00000010/uni00000024/uni00000016/uni00000025/uni00000010/uni00000037/uni0000004b/uni0000004c/uni00000051/uni0000004e/uni0000004c/uni00000051/uni0000004a/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000011/uni00000015/uni00000010/uni00000037/uni0000004b/uni0000004c/uni00000051/uni0000004e/uni0000004c/uni00000051/uni0000004a/uni00000052/uni00000016/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000018/uni00000013/uni00000019/uni00000013 /uni00000037/uni00000055/uni0000004c/uni00000053/uni0000004f/uni00000048/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000034/uni00000024/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000055/uni0000004c/uni00000053/uni0000004f/uni00000048/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000026/uni00000052/uni00000037/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000048/uni0000005b/uni00000057/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000034/uni00000024/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000048/uni0000005b/uni00000057/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000026/uni00000052/uni00000037/uni00000003/uni00000053/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057
Figure 3: Exact Match results (%) of LLMs utilizing different prompts under non-complementary reasoning scenarios.
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni0000001b/uni00000025
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001b/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057 /uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000014/uni00000010/uni0000001a/uni00000013/uni00000025/uni00000010/uni0000002c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000032/uni00000053/uni00000048/uni00000051/uni00000010/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001a/uni00000025/uni00000030/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000002f/uni00000044/uni00000055/uni0000004a/uni00000048/uni00000010/uni00000015/uni00000018/uni00000014/uni00000015/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000011/uni00000015
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000037/uni00000058/uni00000055/uni00000045/uni00000052/uni00000010/uni00000013/uni00000014/uni00000015/uni00000018/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052 /uni0000002a/uni00000033/uni00000037/uni00000010/uni00000018/uni00000011/uni00000014
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000016/uni00000013/uni00000025/uni00000010/uni00000024/uni00000016/uni00000025/uni00000010/uni00000037/uni0000004b/uni0000004c/uni00000051/uni0000004e/uni0000004c/uni00000051/uni0000004a/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000011/uni00000015/uni00000010/uni00000037/uni0000004b/uni0000004c/uni00000051/uni0000004e/uni0000004c/uni00000051/uni0000004a/uni00000052/uni00000016/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000018/uni00000013/uni00000019/uni00000013 /uni00000037/uni00000055/uni0000004c/uni00000053/uni0000004f/uni00000048/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000034/uni00000024/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000055/uni0000004c/uni00000053/uni0000004f/uni00000048/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000026/uni00000052/uni00000037/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000048/uni0000005b/uni00000057/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000034/uni00000024/uni00000003/uni00000033/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057 /uni00000037/uni00000048/uni0000005b/uni00000057/uni00000026/uni00000052/uni00000051/uni00000049/uni00000003/uni00000012/uni00000003/uni00000026/uni00000052/uni00000037/uni00000003/uni00000053/uni00000055/uni00000052/uni00000050/uni00000053/uni00000057
Figure 4: Exact Match results (%) of LLMs utilizing different prompts under complementary reasoning scenarios.
LLMs are more sensitive to the order of evidence. Among all models,
o3-mini demonstrates the greatest stability, exhibiting consistently
smaller gaps across different conflict types.
4.5 Investigation on Prompt Strategy
We also evaluate whether the CoT prompting strategy can lead to
better performance under conflicting evidence, in comparison with
the direct QA prompt. Through the results in Figures 3 and 4, we
have the following observations. (i) Under both COMP and Non-
COMP, CoT improves EM scores for 11 out of 12 LLMs under the
TripleConf setting, with the exception of o3-mini, which exhibits a
slight performance degradation. (ii) When negative textual evidence
is provided, the effectiveness of CoT varies across models. LLMs
with weaker reasoning capabilities, such as Qwen3-8B and Open-
Mistral-7B, tend to be more easily influenced by negative textual
evidence and consequently make more errors. In contrast, stronger
models, including GPT-5.1 and other reasoning LLMs, continue to
benefit from CoT. (iii) Reasoning LLMs exhibit smaller performance
changes than general LLMs when switching from direct QA to
CoT prompting. This suggests that, even without explicit reasoning
instructions, reasoning LLMs may already perform implicit step-
by-step reasoning.
To better understand model behavior under CoT prompting, we
further analyze the reasoning details exhibited in TextConf settings.
We observe that LLMs frequently justify their decisions by explicitlyreferring to the“detailed context provided in the text”, which leads
them to prioritize textual descriptions that appear more informative
or comprehensive when resolving cross-source conflicts. Even when
such textual evidence is misleading, models may implicitly treat
it as more reliable than concise factual triples. This finding helps
explain why CoT prompting does not consistently improve LLMs’
performance under textual conflicts.
5 Two-stage Explanation-based Thinking (XoT)
As illustrated in the previous analysis, LLMs often exhibit implicit
preferences when reasoning over conflicting evidence, which may
bias models toward relying on particular types of evidence without
distinguishing the conflict. To mitigate this issue, we introduce XoT,
a simple and model-agnostic two-stage explanation-based thinking
framework. In this section, we first describe the details of XoT, and
then present its experimental results and case study on ConflictQA.
5.1 Method Details
Rather than straightforward LLM prompting, XoT reorganizes the
roles of LLMs into a two-stage reasoning process, separating answer
exploration from final decision making.
Specifically, given a question 𝑞with heterogeneous evidence
(𝐸KG,𝐸text), XoT first prompts the model to enumerate a set of can-
didate answersC𝑞={𝑐 1,...,𝑐𝐾}, without performing correctness
judgment or pruning candidates based on potential contradictions

, , Tianzhe Zhao et al.
Table 5: Performance (%) of XoT on the ConflictQA benchmark. Bold and underline values denote optimal and sub-optimal
results, respectively. Cells in pink indicate performance improvements of XoT over the conflict-aware QA prompt.
Large Language ModelsNon-COMP COMP
TripleConf TextConf TripleConf TextConf
F1 EM F1 EM F1 EM F1 EM
Qwen3-8BXoT 46.07 13.34 55.35 21.57 45.10 15.58 61.06 23.72
Δ 24.22 111%↑5.73 75%↑20.89 27%↓30.87 59%↓ 18.87 72%↑7.44 91%↑7.67 11%↓23.26 50%↓
Open-Mistral-7BXoT 36.88 9.35 48.01 18.45 37.80 13.02 51.02 23.26
Δ 7.19 24%↑4.12 31%↓19.09 28%↓28.62 61%↓24.46 183%↑6.51 100%↑0.79 2%↓14.88 39%↓
GPT-3.5-Turbo-0125XoT 59.35 39.78 63.51 40.40 54.59 39.77 64.14 41.86
Δ 5.43 10%↑ 3.37 9%↑ 0.86 1%↑3.37 8%↓ 12.06 28%↑ 9.07 30%↑ 4.91 8%↑ 0.46 1%↑
GPT-4oXoT 63.31 43.14 63.68 41.90 60.99 42.79 66.97 46.51
Δ 8.92 16%↑ 10.47 32%↑6.75 10%↓4.67 10%↓ 10.02 20%↑ 9.07 27%↑ 5.87 10%↑ 2.09 5%↑
Deepseek-V3.2-ThinkingXoT 59.32 36.53 59.30 36.53 57.60 42.56 61.81 43.72
Δ 4.69 9%↑ 3.30 10%↑3.43 6%↓3.62 9%↓ 2.17 4%↑ 5.58 15%↑ 5.47 10%↑4.42 11%↑
o3-miniXoT 57.38 39.78 57.82 38.78 53.35 40.23 62.19 46.28
Δ 6.39 13%↑ 1.75 5%↑8.84 13%↓13.77 26%↓ 7.24 16%↑ 2.56 7%↑ 2.19 4%↑2.32 5%↓
among the evidence. For each candidate 𝑐𝑖∈C𝑞, the LLM is fur-
ther required to generate a short, answer-conditioned explanation
𝑒𝑥𝑝𝑖grounded in the provided evidence, while avoiding explicit
references to evidence sources. By explicitly expanding the an-
swer space and associating each candidate with an independent
explanation, XoT leaves the conflict judgment to the subsequent
stage. After obtaining the pairs of candidate answer and explana-
tion{(𝑐𝑖,𝑒𝑥𝑝𝑖)}𝐾
𝑖=1for a question 𝑞, XoT prompts the model to think
over the potential conflicts among candidate explanations and infer
the most appropriate final answers.
XoT serves as a simple yet effective baseline for mitigating im-
plicit preferences in LLM reasoning under conflicting evidence. It
is worth noting that XoT can be naturally extended with more
sophisticated modules, such as iterative refinement or rethinking
mechanisms. We leave such extensions for future investigation.
5.2 Results on ConflictQA
We test XoT on the evaluated LLMs on the ConflictQA benchmark.
Following the analysis in Section 4, we report detailed results on
six representative models with different levels of robustness under
conflicting evidence.
As shown in Table 5, XoT improves performance in most cases,
with particularly stable gains under TripleConf across both Non-
COMP and COMP settings. The improvements are especially pro-
nounced for models that are more sensitive to conflicts. For instance,
in non-complementary settings, Qwen3-8B achieves a relative im-
provement of 111% in F1, while under complementary settings,
Open-Mistral-7B obtains relative gains of 183% in F1 and 100% in
EM. In contrast, the effects of XoT under TextConf are more model-
dependent. Models with limited capacity to assess the reliability
of textual evidence may still be misled by incorrect descriptions,
leading to smaller gains or even performance degradation in some
cases. Meanwhile, stronger models continue to benefit from XoT
under the COMP and TextConf settings. This indicates that, withoutintroducing additional supervision or external evidence, and solely
relying on the LLMs’ inherent knowledge and reasoning ability,
misleading textual descriptions can still dominate the reasoning. As
a result, models struggle to correctly assess conflicting explanations
or to select the correct answer at the final judgment stage.
For general LLMs with stronger ability, we observe that GPT-4o
equipped with XoT achieves the best overall performance across all
evaluated settings. Notably, GPT-4o consistently outperforms other
models, and in several settings attains results that are comparable or
better than those reasoning LLMs. GPT-3.5-Turbo-0125 also shows
clear improvements with XoT, achieving the second-best perfor-
mance in most cases. These observations suggest that XoT aligns
particularly well with strong general LLMs. Despite their relatively
higher robustness to conflicting evidence, reasoning models (e.g.,
Deepseek-V3.2-Thinking and o3-mini) also benefit from XoT, al-
though the improvements are generally smaller in magnitude. This
is consistent with the fact that such models often exhibit structured
reasoning even under direct QA prompting, leaving less space for
additional gains from re-organizing the inference procedure.
Overall, the experimental results demonstrate that XoT can well
serve as a model-agnostic baseline for mitigating conflict-induced
performance degradation on ConflictQA, while also highlighting
that resolving textual conflicts remains challenging.
5.3 Case Study
We present a case to provide an explicit illustration of the predic-
tions produced by different reasoning strategies under conflicting
evidence. As shown in Figure 5, under the direct QA prompt, GPT-4o
treats all provided evidence as plausible and consequently outputs
both correct and incorrect answers. When prompted with CoT,
GPT-4o exhibits greater trust in the detailed and narrative-rich text,
explicitly judging it to be more reliable than the positive KG evi-
dence, and consequently predicts only the misleading answer. This
observation is further consistent with the discussion in Section 4.5.

Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method , ,
Question: What city was the place of death for the person who 
quoted \"The sky is no longer the limit.\"?
Answer: New York City
KG Evidence: 
(Richard Nixon, people.deceased_person.place_of_death, New 
York City)
Textual Evidence:
- The sky is no longer the limit.: The sky is no longer the limit, 
a quotation attributed to Richard Nixon, ...
- Los Angeles: ... The city has played a significant role in 
American history, including being the place of death for notable 
figures such as Richard Nixon,  ... (213 words)
Prediction (QA prompt): Los Angeles, New York City
Prediction (CoT prompt): Los Angeles
Reasoning:  ... The text claims it is Los Angeles, while the 
triple claims it is New York City  ... Given the detailed context 
provided in the text  and the fact that it specifically mentions 
Nixon's death in Los Angeles, it is reasonable to conclude that 
the text is more reliable in this instance .
Prediction (XoT): New York City
Stage 1:  
- Candidate:  New York City; Explanation:  The structured 
triple indicates that Richard Nixon's place of death was New 
York City.
- Candidate:  Los Angeles ; Explanation: The unstructured text 
mentions Los Angeles as the place of death for Richard Nixon, 
who is associated with the quote.
Stage 2:
Prediction:  New York City
Figure 5: An example illustrating the predictions of GPT-
4o under different reasoning strategies on ConflictQA in
the COMP setting with misleading textual evidence. Entities
highlighted in green and pink denote the golden answer and
the misleading answer, respectively.
Regarding XoT, reasoning benefits from separating evidence com-
parison from answer generation, resulting in a correct prediction.
Overall, this case illustrates the effectiveness of XoT in resolving
cross-source conflicts during LLM reasoning.
6 Related Work
LLM Reasoning with RAG. Recent studies have explored enhanc-
ing LLM reasoning by RAG for incorporating different knowledge
sources including unstructured text and (semi-)structured data like
knowledge graphs and tables [ 3,8,16–18,32]. Among these settings,
combining document with KG has attracted particular attention,
as it naturally bridges unstructured and structured information
and can augment the reasoning via the structured knowledge [ 32].
Existing works [ 11,13,18,33] predominantly focus on designing
retrieval strategies for effective evidence and developing prompting
and training methods for accurate LLM reasoning. They typically
assume that the retrieved evidence is reliable and mutually consis-
tent, ignoring potential inconsistencies introduced by the retrievalprocedure, or quality issues of the sources like expiration. Some
recent studies [ 22,39,42] have revealed the vulnerability of such
RAG and LLM-based systems when external knowledge sources
are of low quality or have been adversarially manipulated. In such
cases, corrupted or misleading evidence can amplify conflicts across
retrieved contexts and significantly distort the model’s reasoning
process [ 31]. Therefore, investigating conflicting evidence becomes
critical and urgent for faithful LLM reasoning with RAG.
Knowledge Conflict in LLM Reasoning. The risk of conflict-
ing or inconsistent information brought by retrieved evidence
has recently garnered significant attention among LLM and RAG
researchers [ 2,34,36]. Xie et al. [ 34] analyze conflicts between
parametric knowledge and external evidence by introducing LLM-
generated coherent counter-memory that explicitly contradicts
elicited parametric answers. They show that when both supportive
and contradictory evidence to the LLMs’ internal knowledge are
provided, LLMs can become highly receptive to externally supplied
information. Jin et al. [ 10] study knowledge conflicts in RAG by
inducing LLMs’ parametric memory via closed-book QA and distill-
ing LLM-generated counterfactual answers together with coherent
conflicting evidence from existing QA datasets. Their experiments
reveal that stronger LLMs often persist in relying on incorrect in-
ternal memory even when correct external evidence is available,
and that models generally prefer evidence consistent with their
prior internal beliefs. ConflictBank [ 29] is a benchmark designed to
evaluate conflicts between LLMs’ inherent knowledge and retrieved
contextual knowledge, categorizing such conflicts into three fine-
grained types: misinformation, temporal, and semantic conflicts.
These works focus on conflicts between internal LLM parameters
and external evidence, but ignore the conflict between external
evidence. FaithEval [ 19] attempts to benchmark conflicts among
retrieved contexts, but only synthesize conflicts within one single
source. To the best of our knowledge, there is a short of benchmarks
that models different kinds of conflicts among multiple external
sources. Our benchmark ConflictQA, equipped with different con-
flict settings, bridges this gap and leads to systematic evaluation and
results on 12 different LLMs and 3 different prompting strategies
(including XoT proposed by ourselves).
7 Conclusion and Future Work
This work is among the first to investigate LLM reasoning under
conflicting evidence retrieved from multiple knowledge sources. We
introduced ConflictQA, a novel benchmark that explicitly instanti-
ates conflicts between textual and KG evidence under four different
settings. Through comprehensive evaluations on 12 representative
LLMs with both direct and CoT prompts, we observed that resolv-
ing cross-source conflicts remains a significant challenge, as LLMs
often fail to distinguish conflicting evidence, and LLMs become
much more sensitive to prompt designs and evidence presentation
facing evidence conflicts. For more robust LLM reasoning with
conflicting evidence, we further proposed XoT, a model-agnostic
prompting strategy that asks LLMs to think in two stages with can-
didate ansers and explanations, and demonstrated its effectiveness
on ConflictQA. In the future, we plan to extend ConflictQA to cover
more external sources, and improve XoT with other strategies like
iterative refinement or rethinking as discussed in Section 5.1.

, , Tianzhe Zhao et al.
References
[1]Ryan C Barron, Maksim E Eren, Olga M Serafimova, Cynthia Matuszek, and
Boian S Alexandrov. 2025. Bridging Legal Knowledge and AI: Retrieval-
Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical
Non-negative Matrix Factorization.arXiv preprint arXiv:2502.20364(2025).
[2]Hung-Ting Chen, Michael Zhang, and Eunsol Choi. 2022. Rich Knowledge
Sources Bring Complex Knowledge Conflicts: Recalibrating Models to Reflect
Conflicting Evidence. InProceedings of the 2022 Conference on Empirical Methods
in Natural Language Processing. Association for Computational Linguistics, Abu
Dhabi, United Arab Emirates, 2292–2307.
[3]Philipp Christmann and Gerhard Weikum. 2024. Rag-based question answering
over heterogeneous data and text.arXiv preprint arXiv:2412.07420(2024).
[4]Haoyu Dong, Yue Hu, and Yanan Cao. 2025. Reasoning and retrieval for complex
semi-structured tables via reinforced relational data transformation. InProceed-
ings of the 48th International ACM SIGIR Conference on Research and Development
in Information Retrieval. 1382–1391.
[5]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783(2024).
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining. 6491–6501.
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[8]Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
2025. Database-Augmented Query Representation for Information Retrieval. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing. 16622–16644.
[9]Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B.CoRRabs/2310.06825 (2023).
[10] Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Li
Qiuxia, and Jun Zhao. 2024. Tug-of-War between Knowledge: Exploring and
Resolving Knowledge Conflicts in Retrieval-Augmented Language Models. In
Proceedings of the 2024 Joint International Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING 2024). ELRA and ICCL, Torino,
Italia, 16867–16878.
[11] Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N
Ioannidis, Huzefa Rangwala, and Christos Faloutsos. 2025. Hybgrag: Hybrid
retrieval-augmented generation on textual and relational knowledge bases. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 879–893.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[13] Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq Joty, Soujanya
Poria, and Lidong Bing. 2024. Chain-of-Knowledge: Grounding Large Language
Models via Dynamic Knowledge Adapting over Heterogeneous Sources. InThe
Twelfth International Conference on Learning Representations. https://openreview.
net/forum?id=cPgh4gWZlz
[14] Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong,
Zhaoxin Fan, Bo Tang, Jihao Zhao, Jiawei Yang, et al .2025. SafeRAG: bench-
marking security in retrieval-augmented generation of large language model.
InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 4609–4631.
[15] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu,
Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al .2025. DeepSeek-
V3.2: Pushing the Frontier of Open Large Language Models.arXiv preprint
arXiv:2512.02556(2025).
[16] Zhiqiang Liu, Enpei Niu, Yin Hua, Mengshu Sun, Lei Liang, Huajun Chen, and
Wen Zhang. 2025. SKA-Bench: A Fine-Grained Benchmark for Evaluating Struc-
tured Knowledge Understanding of LLMs. InFindings of the Association for Com-
putational Linguistics: EMNLP 2025. Association for Computational Linguistics,
Suzhou, China, 3626–3640.
[17] Chuangtao Ma, Yongrui Chen, Tianxing Wu, Arijit Khan, and Haofen Wang.
2025. Unifying Large Language Models and Knowledge Graphs for Question
Answering: Recent Advances and Opportunities.. InEDBT. 1174–1177.
[18] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation. In
The Thirteenth International Conference on Learning Representations.[19] Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen,
Caiming Xiong, and Shafiq Joty. 2025. FaithEval: Can Your Language Model Stay
Faithful to Context, Even If "The Moon is Made of Marshmallows". InInternational
Conference on Representation Learning, Vol. 2025. 29430–29456.
[20] Mistral. 2025. Mistral-Large-3-675B-Instruct-2512. https://huggingface.co/
mistralai/Mistral-Large-3-675B-Instruct-2512. Accessed: 2026-01-23.
[21] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar,
Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2025. A com-
prehensive overview of large language models.ACM Transactions on Intelligent
Systems and Technology16, 5 (2025), 1–72.
[22] Fatemeh Nazary, Yashar Deldjoo, and Tommaso di Noia. 2025. Poison-rag: Adver-
sarial data poisoning attacks on retrieval-augmented generation in recommender
systems. InEuropean Conference on Information Retrieval. Springer, 239–251.
[23] OpenAI. 2023. GPT-3.5 Turbo. https://platform.openai.com/docs/models/gpt-3.5-
turbo. Accessed: 2026-01-23.
[24] OpenAI. 2024. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/. Accessed:
2026-01-23.
[25] OpenAI. 2025. GPT-5.1: A smarter, more conversational ChatGPT. https://openai.
com/index/gpt-5-1/. Accessed: 2026-01-23.
[26] OpenAI. 2025. OpenAI o3-mini. https://openai.com/index/openai-o3-mini/.
Accessed: 2026-01-23.
[27] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey.
ACM Transactions on Information Systems44, 2 (2025), 1–52.
[28] Tyler Thomas Procko and Omar Ochoa. 2024. Graph retrieval-augmented gen-
eration for large language models: A survey. In2024 Conference on AI, Science,
Engineering, and Technology (AIxSET). IEEE, 166–169.
[29] Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao
Li, Min Zhang, and Yu Cheng. 2024. $\texttt{ConflictBank}$: A Benchmark for
Evaluating the Influence of Knowledge Conflicts in LLMs. InThe Thirty-eight
Conference on Neural Information Processing Systems Datasets and Benchmarks
Track.
[30] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers). Association for Computational
Linguistics, New Orleans, Louisiana, 641–651.
[31] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025.
Astute rag: Overcoming imperfect retrieval augmentation and knowledge con-
flicts for large language models. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 30553–30571.
[32] Shirley Wu, Shiyu Zhao, Michihiro Yasunaga, Kexin Huang, Kaidi Cao, Qian
Huang, Vassilis N Ioannidis, Karthik Subbian, James Zou, and Jure Leskovec. 2024.
Stark: Benchmarking llm retrieval on textual and relational knowledge bases.
Advances in Neural Information Processing Systems37 (2024), 127129–127153.
[33] Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A Rossi, Haoliang Wang, and
Julian McAuley. 2025. Knowledge-aware query expansion with large language
models for textual and relational retrieval. InProceedings of the 2025 Confer-
ence of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers). 4275–4286.
[34] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2023. Adaptive
chameleon or stubborn sloth: Revealing the behavior of large language mod-
els in knowledge conflicts. InThe Twelfth International Conference on Learning
Representations.
[35] Fangzhi Xu, Qika Lin, Jiawei Han, Tianzhe Zhao, Jun Liu, and Erik Cambria. 2025.
Are Large Language Models Really Good Logical Reasoners? A Comprehensive
Evaluation and Beyond.IEEE Transactions on Knowledge and Data Engineering
37, 4 (2025), 1620–1634.
[36] Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang,
and Wei Xu. 2024. Knowledge Conflicts for LLMs: A Survey. InProceedings of the
2024 Conference on Empirical Methods in Natural Language Processing. Association
for Computational Linguistics, Miami, Florida, USA, 8541–8565.
[37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[38] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The value of semantic parse labeling for knowledge base ques-
tion answering. InProceedings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers). 201–206.
[39] Tianzhe Zhao, Jiaoyan Chen, Yanchi Ru, Haiping Zhu, Nan Hu, Jun Liu, and
Qika Lin. 2025. Exploring Knowledge Poisoning Attacks to Retrieval-Augmented
Generation.Information Fusion(2025), 103900.
[40] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou,
Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al .2023. A survey
of large language models.arXiv preprint arXiv:2303.182231, 2 (2023).
[41] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. 2025. Medrag: Enhanc-
ing retrieval-augmented generation with knowledge graph-elicited reasoning for
healthcare copilot. InProceedings of the ACM on Web Conference 2025. 4442–4457.

Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method , ,
[42] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. {PoisonedRAG}:
Knowledge corruption attacks to {Retrieval-Augmented }generation of largelanguage models. In34th USENIX Security Symposium (USENIX Security 25).
3827–3844.
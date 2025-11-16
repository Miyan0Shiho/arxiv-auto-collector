# Language Drift in Multilingual Retrieval-Augmented Generation: Characterization and Decoding-Time Mitigation

**Authors**: Bo Li, Zhenghua Xu, Rui Xie

**Published**: 2025-11-13 05:36:31

**PDF URL**: [https://arxiv.org/pdf/2511.09984v1](https://arxiv.org/pdf/2511.09984v1)

## Abstract
Multilingual Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to perform knowledge-intensive tasks in multilingual settings by leveraging retrieved documents as external evidence. However, when the retrieved evidence differs in language from the user query and in-context exemplars, the model often exhibits language drift by generating responses in an unintended language. This phenomenon is especially pronounced during reasoning-intensive decoding, such as Chain-of-Thought (CoT) generation, where intermediate steps introduce further language instability. In this paper, we systematically study output language drift in multilingual RAG across multiple datasets, languages, and LLM backbones. Our controlled experiments reveal that the drift results not from comprehension failure but from decoder-level collapse, where dominant token distributions and high-frequency English patterns dominate the intended generation language. We further observe that English serves as a semantic attractor under cross-lingual conditions, emerging as both the strongest interference source and the most frequent fallback language.
  To mitigate this, we propose Soft Constrained Decoding (SCD), a lightweight, training-free decoding strategy that gently steers generation toward the target language by penalizing non-target-language tokens. SCD is model-agnostic and can be applied to any generation algorithm without modifying the architecture or requiring additional data. Experiments across three multilingual datasets and multiple typologically diverse languages show that SCD consistently improves language alignment and task performance, providing an effective and generalizable solution in multilingual RAG.

## Full Text


<!-- PDF content starts -->

Language Drift in Multilingual Retrieval-Augmented Generation:
Characterization and Decoding-Time Mitigation
Bo Li1, 2, Zhenghua Xu1, Rui Xie2*
1State Key Laboratory of Intelligent Power Distribution Equipment and System,
School of Health Sciences and Biomedical Engineering, Hebei University of Technology, China
2National Engineering Research Center for Software Engineering, Peking University, China
deepblue.lb@gmail.com, ruixie@pku.edu.cn
Abstract
Multilingual Retrieval-Augmented Generation (RAG) en-
ables large language models (LLMs) to perform knowledge-
intensive tasks in multilingual settings by leveraging re-
trieved documents as external evidence. However, when the
retrieved evidence differs in language from the user query
and in-context exemplars, the model often exhibits language
drift by generating responses in an unintended language.
This phenomenon is especially pronounced during reasoning-
intensive decoding, such as Chain-of-Thought (CoT) genera-
tion, where intermediate steps introduce further language in-
stability. In this paper, we systematically study output lan-
guage drift in multilingual RAG across multiple datasets, lan-
guages, and LLM backbones. Our controlled experiments re-
veal that the drift results not from comprehension failure but
from decoder-level collapse, where dominant token distribu-
tions and high-frequency English patterns dominate the in-
tended generation language. We further observe that English
serves as a semantic attractor under cross-lingual conditions,
emerging as both the strongest interference source and the
most frequent fallback language.
To mitigate this, we propose Soft Constrained Decoding
(SCD), a lightweight, training-free decoding strategy that
gently steers generation toward the target language by penal-
izing non-target-language tokens. SCD is model-agnostic and
can be applied to any generation algorithm without modify-
ing the architecture or requiring additional data. Experiments
across three multilingual datasets and multiple typologically
diverse languages show that SCD consistently improves lan-
guage alignment and task performance, providing an effective
and generalizable solution in multilingual RAG.
Code and Dataset— https://github.com/pkuserc/SCD
1 Introduction
Recent advances in Retrieval-Augmented Generation
(RAG) have significantly enhanced large language mod-
els’ ability to generate factually grounded answers in
open-domain question answering (Xu et al. 2024a; Luo
et al. 2024; Fang et al. 2024; Shi et al. 2024). However, in
multilingual settings, most existing studies have focused
on improving cross-lingual retrieval performance and
*Corresponding Author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
System Prompt: 你是一个严谨问答助手，请用中文回答问题。
Examples: 下面是一些带有推理过程的问题和答案示例：1）
问题：奥地利平犬和马雷马牧羊犬的主要职责是什么？推理过
程：...... 最终答案：守卫。 2）问题：......
Query: 现在开始正式提问：康恩牧场是一个航天港，进行试
飞的公司总部位于哪个州？以下是与问题相关的背景材料：
Retrieved Text: 1) Corn Ranch is a spaceport in the West 
Texas town of... 2) The company is developing technologies to 
enable private human access to space ...
【推理过程】根据问题中提到的是康恩牧场是
一个航天港。Blue Origin是其总部位于的公司。
Blue Origin's headquarters is located in Kent, 
Washington.
【最终答案】Washington
Reasoning Trajectory:
Target Lang.          Mixed Lang.           Distractor Lang. 
Multilangual LLM
Distractor Language (e.g., English)Target Language (e.g., Chinese)
Language
 DriftFigure 1: Illustration of language drift in multilingual RAG.
The user query and in-context examples are provided in the
target language (e.g., Chinese), while the retrieved context is
written in a non-target language (e.g., English). During rea-
soning, the model mixes languages and ultimately outputs
the final answer in a non-target language.
contextual alignment (Luo et al. 2020; Park and Lee 2025;
Chirkova et al. 2024; Wu et al. 2024; Ranaldi, Haddow, and
Birch 2025; Ranaldi et al. 2025; Liu et al. 2025; Bland’on
et al. 2025), while overlooking a critical issue: the mismatch
between the input and output languages.
In multilingual RAG settings, the query, instructions, and
in-context exemplars are typically written in the target lan-
guage, aiming to elicit responses in that language. However,
due to the predominance of English in open-domain cor-
pora (Xu et al. 2024b; Zeng and Yang 2024; Resnik and
Smith 2003), the retrieved context is often in English, even
when the query is in another language. This creates a mixed-
lingual input scenario where only the retrieved context dif-
fers in language. Nevertheless, models frequently generate
responses in the language of the retrieved content rather than
in the intended target language (Park and Lee 2025; Liu
et al. 2025). We refer to this phenomenon asoutput language
drift, which poses practical challenges for multilingual ap-
plications yet remains underexplored. Our empirical study
reveals that such cross-lingual conditions negatively affect
both task performance and output language consistency. No-
tably, English serves as the strongest interference source,
significantly degrading output quality in non-English set-
tings, while also serving as the most robust target language
when subjected to interference. This issue becomes more se-
vere under few-shot prompting and Chain-of-Thought (CoT)
reasoning (Shi et al. 2022; Yu et al. 2025).
Interestingly, we observe that language drift does not nec-arXiv:2511.09984v1  [cs.CL]  13 Nov 2025

essarily follow the language of the retrieved context. Instead,
models frequently default to English during generation, even
when the context passages are in Arabic, Russian, or other
non-English languages. This indicates that English plays a
dominant role beyond being a common training language: it
functions as a semantic attractor in multilingual generation.
Our analysis indicates that, under cross-lingual ambiguity,
LLMs tend to prefer English over the context language. This
fallback tendency further verifies the dominant role of En-
glish as the default trajectory in multilingual decoding.
To better understand whether this fallback behavior re-
sults from misunderstanding or from generative biases, we
conduct human evaluation and reference translation. Inter-
estingly, many of the outputs that drift to the non-target
language are still semantically faithful, indicating that the
model has accurately understood both the task and the re-
trieved context. By analyzing intermediate reasoning steps
(i.e., CoT traces), we find that the language inconsistency of-
ten emerges mid-generation, even when earlier steps remain
in the target language. This indicates that the failure stems
not from semantic comprehension, but from generation bi-
ases favoring frequent English tokens. As a result, the model
produces outputs that are structurally fluent but linguistically
inconsistent, reflecting a form of language collapse driven
by token-level priors rather than task misunderstanding.
These findings motivate the need for lightweight
decoding-time strategies that maintain output language con-
sistency without compromising reasoning performance. To
this end, we introduce Soft Constrained Decoding (SCD),
a token-level control mechanism that assigns soft penalties
to non-target-language tokens, thereby encouraging target-
language generation while preserving fluency. In contrast to
hard vocabulary filtering, SCD is a flexible, model-agnostic
mechanism compatible with standard decoding algorithms.
Extensive experiments across diverse datasets, model back-
bones, target languages, and context languages demonstrate
that SCD improves both output language alignment and an-
swer quality, providing a practical solution to a persistent yet
underexplored challenge in multilingual RAG.
Our main contributions are as follows:
•Multilingual Dataset Construction.We construct
multilingual versions of HotpotQA, MuSiQue, and
DuReader by translating and human-verifying all com-
ponents (queries, answers, prompts, exemplars, and re-
trieved context), enabling controlled evaluation across
four diverse languages.
•Analysis of Language Drift.We conduct controlled ex-
periments that vary only the language of retrieved con-
texts, revealing overlooked patterns in multilingual RAG
such as performance degradation, target-language in-
consistency, and a strong fallback tendency to English.
Chain-of-Thought traces show that drift typically arises
mid-generation due to decoding-time biases.
•Training-free Language Control.We introduce SCD, a
lightweight decoding-time method that softly penalizes
non-target tokens. SCD is model-agnostic, requires no
training, and improves both output language consistency
and task accuracy across datasets and LLMs.2 Language Drift in Multilingual RAG
In this section, we conduct a comprehensive empirical in-
vestigation into the phenomenon of language drift, where
model outputs deviate from the intended target language
during multilingual RAG generation. To support this study,
we construct multilingual variants of several benchmark
RAG datasets by translating and aligning all critical compo-
nents, including queries, answers, prompts, exemplars, and
retrieved passages. We then evaluate LLM behavior across a
range of controlled conditions. Our findings reveal a set of
systematic behaviors that undermine both task accuracy and
output language alignment under cross-lingual conditions.
2.1 Multilingual Dataset Construction
To systematically evaluate how multilingual retrieved con-
text influences LLM behavior in RAG, we require datasets
in which the language of each input component can be inde-
pendently controlled. This enables us to isolate the impact
of cross-lingual retrieved passages on model reasoning and
output consistency. However, no existing benchmark satis-
fies these constraints while remaining compatible with RAG.
To address this gap, we construct multilingual versions of
three widely used QA datasets that support retrieval aug-
mentation: HotpotQA (Yang et al. 2018), MuSiQue (Trivedi
et al. 2022), and DuReader1. These datasets contain high-
quality question–answer pairs with human-annotated gold
retrieved context, making them well-suited for our pur-
pose. We select four typologically diverse languages, En-
glish (EN), Chinese (ZH), Arabic (AR), and Russian (RU), to
capture a broad range of linguistic variation. Representative
data examples and the format used for multilingual annota-
tion are provided in Appendix F.
Each dataset contributes 1,000 samples. For every sample,
we prepare five components: a user query, a reference an-
swer, several gold retrieved contexts, a prompt template, and
several in-context exemplars. All components are translated
into the four languages using GPT-4o, followed by man-
ual verification to ensure semantic fidelity and natural flu-
ency. This multilingual suite enables flexible and language-
controlled experimentation across a wide range of configu-
rations.
2.2 Experimental Setup
Based on the multilingual datasets described above, we de-
sign a controlled experimental framework to evaluate how
the language of the retrieved context influences output be-
havior in RAG. In our core setup, we fix the language of the
query, prompt, and ICL examples to the target language (de-
noted as the context language), and vary only the language
of the retrieved passage to isolate cross-lingual interference
effects.
We test across three datasets (HotpotQA, MuSiQue,
DuReader) and four target languages (EN,ZH,AR,RU), us-
ing two instruction-tuned LLMs as backbones: LLaMA3-
8B-Instruct (Grattafiori et al. 2024) and Qwen2.5-7B-
Instruct (Yang et al. 2024, 2025). All generations are per-
formed using default decoding parameters, and each prompt
1https://github.com/baidu/DuReader

ROUGEBLEULCROUGE
BLEU
LC
ROUGE
BLEU
LC
ROUGEBLEULCROUGE-22.5%
-17.5%183.3%194.5%
195.9%
196.1%ENZH
ARRUT arget: EN
T arget: ZH
T arget: AR
T arget: RUFigure 2: Relative performance gap between with-ICL and
without-ICL settings across different target and context lan-
guage combinations on the HotpotQA dataset, visualized as
a polar radar chart. Each axis corresponds to one evaluation
metric (ROUGE, BLEU, or LC) under a specific context lan-
guage (EN,ZH,AR, orRU), totaling 12 axes. Solid lines rep-
resent different target languages, color-coded accordingly.
Values indicate the percentage difference between ICL and
non-ICL performance under each configuration. The black
dashed ring at 0% denotes no change. Labels mark the high-
est gains and the most severe LC degradations. The chart
reveals that ICL consistently improves BLEU and ROUGE,
but often reduces language consistency, especially underZH
andRUcontexts.
includes four ICL exemplars in the same language as the
query.
For evaluation, we report standard BLEU-1/2/3 and
ROUGE-1/2/L scores, along with their averaged variants
(BLEUandROUGE), using reference answers in the target
language as the gold standard. To further quantify language
fidelity, we introduce aLanguage Consistency (LC)met-
ric, which measures the proportion of generated responses
written in the expected target language. This comprehensive
metric suite allows us to jointly evaluate reasoning accuracy
and language control in multilingual RAG settings.
2.3 ICL Improves Performance but Undermines
Consistency
To investigate how multilingual retrieved context and ICL
jointly affect RAG performance, we conduct controlled
experiments across various target–context language pairs.
Specifically, we fix the query, prompt, and exemplars in the
target language (EN,ZH,AR, orRU) and vary only the lan-
guage of the retrieved context. For each configuration, we
compare model outputs with and without ICL exemplars, al-
lowing us to isolate the effects of ICL under multilingualinterference.
Figure 2 summarizes these effects using a radar chart on
the HotpotQA dataset withLLaMA3-8B-Instructas
the backbone. Each colored line represents a fixed target lan-
guage, while each radial group corresponds to one context
language (EN,ZH,AR,RU), covering three evaluation met-
rics: ROUGE, BLEU, and LC. The plotted values represent
therelative percentage changeintroduced by ICL compared
to the non-ICL baseline under the same configuration. Posi-
tive values indicate improvements, whereas negative values
reflect degradation. As a concrete example, the green point
within the orangeZH-labeled frame represents theZH-AR
condition. It shows that ICL increases BLEU significantly
but reduces LC, reflecting the common pattern where richer
reasoning comes at the cost of linguistic fidelity under cross-
lingual retrieved context. Due to space constraints, we report
the radar plot results only for HotpotQA, which is represen-
tative of the broader trends. Similar patterns are observed
across other datasets, languages, and backbone models; de-
tailed results are included in Appendix D. Our results in Fig-
ure 2 reveal two key findings:
•Multilingual interference degrades both performance
and consistency.When the retrieved context is in a lan-
guage different from the target, both task performance
(measured by BLEU and ROUGE) and output language
consistency decline significantly. Notably, we observe
thatEnglish acts as the strongest interfering language:
when used as cross-lingual retrieved context, it induces
the most severe performance degradation across non-
English targets. For example, in the ICL setting withZH
as the target language, language consistency drops from
92.0% to 68.4% when switching retrieved contexts from
ZHtoENretrieved context, with a drop in average BLEU
score from 0.212 to 0.086. In contrast,ENexhibits the
strongest resistance to interferencewhen serving as the
target language, whileZHshows thegreatest sensitivity
across all datasets.
•In-context learning improves performance but wors-
ens consistency.Adding ICL examples consistently im-
proves generation quality across all datasets and models.
However, it also intensifies output language drift, lead-
ing the model to deviate further from the expected target
language. For example, withRUas the target language,
the average ROUGE increases from 0.193 to 0.373 af-
ter adding ICL, while language consistency drops from
0.991 to 0.895. Similar trends are observed when the con-
text language differs from the target language: ICL im-
proves accuracy but significantly reduces alignment with
the expected output language.
These findings indicate that while ICL improves semantic
fidelity, it also increases vulnerability to language drift due
to extended reasoning and exposure to non-target-language
tokens. Since ICL reflects real-world usage and consistently
improves performance, we adopt it as the default in all ex-
periments, with prompts explicitly instructing the model to
generate in the target language.

ZH AR RU
T arget LanguageZH
EN
AR
RUContext Language100.0% 100.0% 100.0%
100.0% 100.0% 100.0%
98.6% 84.5% 90.6%
72.3% 74.6% 99.2%HotpotQA (LLaMA3.2-8B)
ZH AR RU
T arget Language98.8% 72.1% 85.0%
100.0% 78.3% 95.2%
100.0% 61.8% 78.1%
97.1% 78.0% 90.0%HotpotQA (Qwen2.5-7B)
0.8%0.9%1.0%
Drift-to-English Rate
Figure 3: Language drift patterns on the HotpotQA dataset
forLLaMA3-8BandQwen2.5-7Bmodels. Each cell
shows the percentage of inconsistent outputs that are gener-
ated in English (EN). Both models exhibit a strong fallback
tendency toward English across all cross-lingual settings.
2.4 English as the Default Fallback Language
While previous results show that cross-lingual interference
reduces output consistency, we further investigatewhich lan-
guage the model tends to generatewhen it fails to remain
in the target language. Specifically, we analyze all inconsis-
tent outputs and identify their actual output language. Strik-
ingly, we observe that in the majority of drift cases across
all target languages and datasets, the model defaults to gen-
erating inENregardless of whether the retrieved context is
EN, as shown in Figure 3. Due to the space limitations, ad-
ditional results with similar conclusions are provided in the
Appendix A.
This fallback behavior suggests thatENplays a dominant
role not only in training but also during decoding. Rather
than aligning with the context language, the model often de-
faults toENwhen facing ambiguity, a tendency driven by
structural biases such as the over-representation of English
tokens during pretraining and the concentration of factual
knowledge inEN. Our experiments further confirm that even
when both the target languages and context are non-English,
misaligned outputs predominantly appear inEN, indicating
that language drift is not random but guided byENacting as
a default semantic attractor.
2.5 Language Collapse During Decoding
To assess whether the observed language drift arises from
comprehension failure or unstable decoding behavior, we
conduct a semantic agreement analysis. As shown in Ta-
ble 1, we compare three evaluation metrics under cross-
lingual settings: (1) Standard ROUGE between the model
output and the target-language reference; (2) ROUGE after
translating drifted outputs back into the target language and
recomputing scores against the original reference (denoted
as ROUGE(T)); (3) Semantic Match Rate, scored by GPT-
4o, which evaluates whether the model output is factually
aligned with the reference regardless of surface language.
We observe that translation leads to asignificantimprove-
ment in ROUGE scores. For example, ROUGE increases
from 0.182 to 0.263 forZH, and from 0.333 to 0.388 forRU,
indicating that the original outputs are semantically aligned
despite being expressed in the wrong language. Moreover,the Semantic Match Rate further confirms that even when
ROUGE is low, the match rate often exceeds 60% forRU
and over 50% forZHandAR, demonstrating strong task
understanding. These findings suggest that language drift
stems not from comprehension failure but from decoder-
level instability. Additional results on other datasets (see
Appendix B) show similar patterns across models and lan-
guages.
This pattern suggests a language collapse during decod-
ing, where the LLM correctly processes the input and un-
derstands the intended task but fails to maintain the target
language throughout generation. We hypothesize that this
issue arises from token-level priors learned during pretrain-
ing, as English tokens tend to dominate due to their higher
frequency, more stable syntactic structures, and richer fac-
tual coverage. During multi-step reasoning, especially under
CoT prompting, such biases can override explicit language
instructions and gradually shift the generation toward En-
glish. The drift typically unfolds over time, with the genera-
tion beginning in the target language but progressively devi-
ating into English. This highlights a fundamental limitation
in multilingual LLMs:strong semantic reasoning does not
guarantee stable language control during generation.
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RU
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RUDrift Type
A
B
C
D
Figure 4: Distribution of four language drift types across
different target–context language pairs in the HotpotQA
dataset. Each subplot corresponds to a fixed target language
(ZH,AR,RU), with the x-axis denoting the context language.
The top row displays results forLLaMA3-8B, and the bot-
tom row forQwen2.5-7B.
2.6 Types of Language Drift Behaviors
To better understand how language drift manifests in mul-
tilingual reasoning, we categorize drifted outputs into four
distinct behavioral types based on multilingual generations.
We randomly sampled 1,000 language-inconsistent outputs
and had them manually annotated by three trained review-
ers with backgrounds in linguistics or multilingual NLP. The
taxonomy includes:Type A:Named Entity Representation
Divergence, where inconsistent transliteration or spelling re-
sults in mismatches despite semantic equivalence;Type B:
Answer Target Shift, where the model alters answer granu-
larity or is misled by context-language cues, leading to an
incorrect sub-answer;Type C:Reasoning Chain Misalign-
ment, where the CoT path becomes structurally disrupted
due to language mixing or code-switching; andType D:
Conceptual Reference Shift, where cultural or semantic bi-
ases embedded in the dominant language (such as English)

LLaMA3-8B Qwen2.5-7B
Targe
LanguageContext
LanguageROUGE ROUGE(T)Semantic
Match RateROUGE ROUGE(T)Semantic
Match Rate
ZHEN 0.182 0.263 54.7% 0.331 0.352 62.7%
AR 0.211 0.258 46.2% 0.342 0.366 55.3%
RU 0.209 0.261 49.4% 0.337 0.359 54.2%
AREN 0.294 0.331 53.4% 0.201 0.221 46.4%
ZH 0.265 0.288 48.3% 0.187 0.202 42.2%
RU 0.280 0.303 50.0% 0.206 0.220 43.0%
RUEN 0.333 0.388 62.8% 0.240 0.262 60.1%
ZH 0.335 0.367 59.1% 0.248 0.257 56.1%
AR 0.339 0.361 62.9% 0.248 0.252 59.0%
Table 1: Performance under cross-lingual retrieved context for non-English target languages (ZH,AR,RU) usingLLaMA3-8B
andQwen2.5-7Bon HotpotQA. We report standard ROUGE, ROUGE after translating the model output to the target language
(ROUGE(T)), and Semantic Match Rate assessed by GPT. Despite language drift, many outputs remain semantically correct,
highlighting decoder-level instability rather than comprehension failure.
trigger unintended knowledge concepts. Full category defi-
nitions and examples are provided in Appendix C.
We use GPT-4o to classify a representative set of drifted
outputs into the four categories defined in our taxonomy,
followed by manual verification to ensure label quality. As
shown in Figure 4, the most common behavior across both
models and all target languages isNamed Entity Represen-
tation Divergence(Type A), which accounts for approxi-
mately 55% to 74% of drifted cases on average. This is fol-
lowed byAnswer Target Shift(Type B), occurring in roughly
17% to 31% of cases, with greater variation across context
languages.Reasoning Chain Misalignment(Type C) is less
frequent, comprising around 9% to 18%, whileConceptual
Reference Shift(Type D) remains rare, often below 5%.
These findings suggest that most drift cases arise from
surface-level inconsistencies, such as entity formatting or
answer phrasing, rather than from deeper reasoning failures.
Recognizing how such drift emerges during the later stages
of CoT decoding can inform more targeted control strate-
gies, including applying penalties for answer-level devia-
tions or reinforcing consistency in entity representation.
3 Soft-Constrained Decoding
3.1 Soft-Constrained Decoding (SCD)
To mitigate output language drift in multilingual gener-
ation, we proposeSoft-Constrained Decoding (SCD), a
lightweight decoding-time control strategy that incorporates
token-level language awareness into the generation process.
Instead of applying rigid vocabulary restrictions, SCD subtly
adjusts the token probability distribution to favor the target
language, while preserving open-ended reasoning capabili-
ties and fluent output.
Token Categorization.LetVdenote the model vocabu-
lary, and we partitionVinto three disjoint sets:
•V target: tokens associated with thetarget language,
•V neutral :neutral tokenssuch as punctuation, digits, and
shared symbols,•V distractor : tokens linked tonon-target languages.
This categorization is performed via Unicode ranges or
tokenizer-based heuristics and cached prior to generation.
Logits Adjustment.Letz(t)∈R|V|be the raw logits out-
put at decoding stept. SCD adjustsz(t)before softmax as
follows:
˜z(t)
i=

αz(t)
i,ifi∈ V target
z(t)
i,ifi∈ V neutral
βz(t)
i,ifi∈ V distractor
Here,α >1.0is a soft boost to target-language tokens, and
β <1.0is a penalty for distractor-language tokens. This
modification biases generation while preserving flexibility.
Cold Start Smoothing.Multilingual LLMs, especially in
low-resource languages, often generate unstable initial out-
puts such as repeated prompts or template fragments. To
minimize such disruptions, we introduce awarm-up period
by delaying the activation of language constraints until de-
coding stepT start. This design ensures a fluent transition into
reasoning before language control is applied.
Integration.SCD ismodel-agnosticand fully compatible
with standard decoding algorithms. It requires no additional
training or architectural changes.
SCD operates as a lightweight decoding-time strategy that
gently discourages the selection of non-target language to-
kens without eliminating them entirely. By incorporating
language awareness directly into the token selection process,
SCD guides the model to favor tokens in the target language
while retaining the flexibility needed for open-ended reason-
ing.
3.2 Experimental Setup and Baselines
We evaluate our proposed SCD on three multilingual
retrieval-augmented QA datasets, i.e., HotpotQA, MuSiQue,
and DuReader, which are described in Section 2.1. Exper-
iments are conducted using two instruction-tuned LLMs:

HotpotQA Musique DuReader
Targe
LanguageContext
LanguageROUGE BLEU LC ROUGE BLEU LC ROUGE BLEU LC
Prompted Language Instruction
ZHEN 0.182 0.086 68.4% 0.187 0.097 63.9% 0.339 0.166 84.2%
AR 0.211 0.106 77.7% 0.181 0.089 76.5% 0.358 0.175 90.1%
RU 0.209 0.107 79.5% 0.169 0.087 64.5% 0.343 0.168 83.1%
AREN 0.294 0.162 85.4% 0.144 0.080 90.0% 0.209 0.099 88.2%
ZH 0.265 0.143 88.4% 0.120 0.057 89.2% 0.193 0.080 87.0%
RU 0.280 0.151 88.6% 0.121 0.061 89.8% 0.186 0.077 89.5%
RUEN 0.333 0.177 80.2% 0.218 0.119 81.9% 0.285 0.150 84.3%
ZH 0.335 0.172 85.1% 0.206 0.102 90.2% 0.296 0.149 85.8%
AR 0.339 0.179 86.8% 0.214 0.109 92.5% 0.288 0.143 90.9%
Translation-Based Evaluation
ZHEN 0.263 0.135 100.0% 0.257 0.142 100.0% 0.366 0.178 100.0%
AR 0.258 0.132 100.0% 0.214 0.105 100.0% 0.364 0.177 100.0%
RU 0.261 0.136 100.0% 0.235 0.124 100.0% 0.365 0.175 100.0%
AREN 0.331 0.183 100.0% 0.168 0.095 100.0% 0.231 0.114 100.0%
ZH 0.288 0.156 100.0% 0.135 0.066 100.0% 0.202 0.087 100.0%
RU 0.303 0.165 100.0% 0.140 0.074 100.0% 0.195 0.083 100.0%
RUEN 0.388 0.218 100.0% 0.258 0.148 100.0% 0.314 0.167 100.0%
ZH 0.367 0.196 100.0% 0.215 0.109 100.0% 0.309 0.156 100.0%
AR 0.361 0.196 100.0% 0.217 0.114 100.0% 0.293 0.148 100.0%
Soft-Constrained Decoding (Ours)
ZHEN 0.306 0.155 90.6% 0.276 0.146 91.8% 0.403 0.190 95.2%
AR 0.283 0.146 93.9% 0.234 0.118 94.8% 0.408 0.195 96.6%
RU 0.293 0.156 92.5% 0.243 0.130 92.3% 0.404 0.190 95.7%
AREN 0.352 0.197 96.4% 0.187 0.106 98.8% 0.241 0.113 96.7%
ZH 0.312 0.170 95.5% 0.157 0.079 97.6% 0.236 0.104 94.1%
RU 0.326 0.183 96.3% 0.152 0.080 98.0% 0.220 0.092 95.4%
RUEN 0.422 0.238 95.4% 0.270 0.162 94.1% 0.334 0.174 94.4%
ZH 0.400 0.216 94.1% 0.230 0.126 94.7% 0.335 0.165 94.3%
AR 0.392 0.216 94.0% 0.232 0.128 94.3% 0.317 0.155 94.7%
Table 2: Performance comparison across three language control strategies: Prompted Language Instruction, Translation-Based
Evaluation, and SCD on three multilingual RAG datasets. We report results forLLaMA3-8B, where SCD consistently improves
both LC and content metrics across datasets compared to strong baselines. Results forQwen2.5-7Bare provided in Appendix
E due to space constraints.
LLaMA3-8B-Instruct and Qwen2.5-7B-Instruct. We empir-
ically find moderate settings (α= 1.1,β= 0.9,T start= 5)
to balance language fidelity and semantic fluency in SCD.
To benchmark SCD against other lightweight language
control strategies, we compare it with the following
decoding-time baselines: (1)Prompted Language Instruc-
tion: Explicitly appending an instruction in the prompt that
requests answers to be generated in the target language; (2)
Translation-Based Evaluation: Evaluating drifted outputs
by translating them back into the target language using the
same LLM, before computing BLEU/ROUGE scores; (3)
Vocabulary Restriction Decoding: Restricting the decod-
ing space to tokens belonging to the target language only,
effectively applying a hard constraint on generation.
We evaluate all methods using three complementary met-
rics: (1) BLEU (mean of BLEU-1/2/3), (2) ROUGE (mean
of ROUGE-1/2/L), and (3) language consistency (LC), de-
fined as the percentage of outputs generated in the cor-rect target language. All decoding parameters follow the de-
fault settings of each model, and no task-specific or model-
specific fine-tuning is applied. Additional performance im-
provements may be obtained by tuning decoding parameters,
we leave this for future work. All reported scores are aver-
aged over five independent runs to reduce randomness.
3.3 Effectiveness of Soft-Constrained Decoding
As shown in Table 2, SCD consistently outperforms existing
language control methods, achieving notable improvements
in bothlanguage consistency(LC) andsemantic generation
quality, as measured by average BLEU and ROUGE scores.
These results support our central hypothesis that maintaining
alignment with the target language can reinforce, rather than
hinder, the coherence and accuracy of reasoning paths.
Across all datasets and language configurations, SCD
consistently improves both language consistency and con-
tent quality compared to the Prompted Language Instruction

ROUGE CoT Length
Targe
Lang.Context
Lang.PLI VRD SCD PLI VRD SCD
ZHEN 0.182 0.155 0.306 104.0 38.6 134.9
AR 0.211 0.184 0.283 103.5 40.2 142.4
RU 0.209 0.173 0.293 77.6 42.8 143.1
AREN 0.294 0.295 0.352 77.0 50.5 90.2
ZH 0.265 0.266 0.312 86.4 49.9 92.9
RU 0.280 0.281 0.326 86.3 57.4 100.2
RUEN 0.333 0.343 0.422 85.6 58.8 111.8
ZH 0.335 0.339 0.400 89.4 56.1 111.0
AR 0.339 0.341 0.392 89.4 56.6 111.8
Table 3: Comparison of three decoding strategies on Hot-
potQA across ROUGE score and average CoT length.
baseline. For instance, under the challengingZH-ENcondi-
tion on HotpotQA, SCD increases LC from 68.4% to 90.6%,
while also boosting BLEU from 0.086 to 0.155 and ROUGE
from 0.182 to 0.306. Similar trends are observed for other
target languages such asARandRU, with LC improvements
ranging from 10 to 22 percentage points.
While the translation-based method trivially achieves
100% LC by converting drifted outputs into the target
language after generation, it often underperforms SCD in
BLEU and ROUGE. This outcome is expected, as transla-
tion does not recover the original reasoning trajectory but
merely reformulates its surface form. Moreover, translation-
based evaluation adds additional complexity, increases infer-
ence cost, and may amplify noise when the original outputs
are incomplete or syntactically broken.
The above results demonstrate that SCD is a practical,
lightweight, and model-agnostic decoding-time interven-
tion. It requires no additional training or architectural mod-
ifications, and can be seamlessly integrated into standard
decoding workflows (e.g., greedy, sampling, top-p). Across
models, languages, and datasets, SCD provides consistent
and substantial improvements in both linguistic alignment
and semantic quality, making it a strong candidate for real-
world multilingual RAG and generation-based applications.
3.4 Should Multilingual Generation Be Fully
Language Isolated?
To examine the trade-offs of different language control
strategies in multilingual generation, we compare three
decoding methods: Prompted Language Instruction (PLI),
V ocabulary-Restricted Decoding (VRD), and our proposed
SCD. PLI uses explicit prompts to enforce the target lan-
guage; VRD imposes hard constraints by restricting gener-
ation to target-language tokens; and SCD softly penalizes
non-target tokens while maintaining generation flexibility.
As shown in Table 3, SCD consistently achieves the high-
est ROUGE scores across all target–context language pairs
on HotpotQA. For instance, in theZH-ENsetting, SCD
reaches 0.306 ROUGE, compared to 0.155 under VRD and
0.182 under PLI. Similar trends are observed forARandRU
targets. Interestingly, VRD often underperforms PLI, sug-
gesting that overly strict language filtering can suppress use-ful multilingual cues and degrade output quality, despite im-
proving consistency.
To assess generation dynamics, we compare the average
length of generated CoT responses. VRD consistently yields
the shortest outputs, e.g., only 38.6 tokens inZH-EN, com-
pared to 104.0 with PLI and 134.9 with SCD—indicating
that hard constraints truncate reasoning. In contrast, SCD
preserves longer and more complete reasoning chains by al-
lowing controlled cross-lingual flexibility. We further ana-
lyze how reasoning length affects language drift and control
effectiveness in Appendix G, where SCD demonstrates ro-
bust performance across various CoT trajectories.
These results suggest thateffective multilingual gener-
ation does not require full language isolation. Allowing
limited access to non-target tokens during reasoning, while
softly guiding the output toward the desired language, im-
proves both language consistency and semantic fidelity.
4 Related Work
Multilingual RAG has received increasing attention as a
means to enhance LLMs with access to cross-lingual knowl-
edge. Prior research has primarily focused on improving the
quality of multilingual retrieval (Liu et al. 2025; Chirkova
et al. 2024; Ranaldi, Haddow, and Birch 2025), aligning re-
trieved passages with user queries across languages (Ranaldi
et al. 2025; Bland’on et al. 2025), and adapting RAG
pipelines to typologically diverse settings (Wu et al. 2024;
Zeng and Yang 2024). These efforts have significantly ad-
vanced retrieval-stage effectiveness in non-English tasks and
established multilingual evaluation protocols. Some recent
works have further explored language preferences in RAG
models (Park and Lee 2025; Shi et al. 2022; Yu et al. 2025),
highlighting accuracy disparities across languages. How-
ever, most of these works either evaluate generation out-
comes at the answer level or focus on upstream retrieval
modules, without deeply investigating how language behav-
ior evolves throughout the decoding process.
In contrast, we focus on the overlooked issue oflanguage
driftin multilingual RAG, where model outputs shift away
from the target language during reasoning. We demonstrate
that this drift arises during decoding, with English acting as
a default fallback. To mitigate it, we propose a lightweight
decoding-time strategy that improves language alignment
without requiring model retraining.
5 Conclusion
This work addresses a key challenge in multilingual RAG:
large language models often generate outputs in unin-
tended languages when reasoning over cross-lingual evi-
dence. Through controlled experiments and CoT analysis,
we find that such drift arises from decoder-stage biases
rather than comprehension failure. To mitigate this, we in-
troduce SCD, a lightweight, model-agnostic strategy that
softly penalizes non-target-language tokens. SCD consis-
tently enhances both language consistency and task perfor-
mance across models, languages, and datasets. These find-
ings underscore the value of decoding-time control for build-
ing more robust and controllable multilingual RAG systems.

Acknowledgements
This work was supported by the National Natural Science
Foundation of China (Grant No. 62276089), the Natural
Science Foundation of Tianjin (Grant No. 24JCJQJC00200
and Grant No. 24JCQNJC01230), the Natural Science Foun-
dation of Hebei Province (Grant No. F2024202064), the
Science Research Project of Hebei Education Department
(Grant No. BJ2025004), the Ministry of Human Resources
and Social Security of China (Grant No. RSTH-2023-135-
1), and the Science and Technology Program of Hebei
Province (Grant No. 24464401D).
References
Bland’on, M. A. C.; Talur, J.; Charron, B.; Liu, D.; Man-
sour, S.; and Federico, M. 2025. MEMERAG: A Multilin-
gual End-to-End Meta-Evaluation Benchmark for Retrieval
Augmented Generation.ArXiv, abs/2502.17163.
Chirkova, N.; Rau, D.; D’ejean, H.; Formal, T.; Clinchant,
S.; and Nikoulina, V . 2024. Retrieval-augmented generation
in multilingual settings.ArXiv, abs/2407.01463.
Fang, F.; Bai, Y .; Ni, S.; Yang, M.; Chen, X.; and Xu, R.
2024. Enhancing Noise Robustness of Retrieval-Augmented
Language Models with Adaptive Adversarial Training. In
Annual Meeting of the Association for Computational Lin-
guistics.
Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian,
A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.;
Vaughan, A.; et al. 2024. The llama 3 herd of models.arXiv
preprint arXiv:2407.21783.
Liu, W.; Trenous, S.; Ribeiro, L. F. R.; Byrne, B.;
and Hieber, F. 2025. XRAG: Cross-lingual Retrieval-
Augmented Generation.
Luo, F.; Wang, W.; Liu, J.; Liu, Y .; Bi, B.; Huang, S.; Huang,
F.; and Si, L. 2020. VECO: Variable and Flexible Cross-
lingual Pre-training for Language Understanding and Gen-
eration. InAnnual Meeting of the Association for Computa-
tional Linguistics.
Luo, K.; Liu, Z.; Xiao, S.; Zhou, T.; Chen, Y .; Zhao, J.;
and Liu, K. 2024. Landmark Embedding: A Chunking-Free
Embedding Method For Retrieval Augmented Long-Context
Large Language Models. InAnnual Meeting of the Associa-
tion for Computational Linguistics.
Park, J.; and Lee, H. 2025. Investigating Language Prefer-
ence of Multilingual RAG Systems.ArXiv, abs/2502.11175.
Ranaldi, L.; Haddow, B.; and Birch, A. 2025. Multilingual
Retrieval-Augmented Generation for Knowledge-Intensive
Task.ArXiv, abs/2504.03616.
Ranaldi, L.; Ranaldi, F.; Zanzotto, F. M.; Haddow, B.;
and Birch, A. 2025. Improving Multilingual Retrieval-
Augmented Language Models through Dialectic Reasoning
Argumentations.ArXiv, abs/2504.04771.
Resnik, P.; and Smith, N. A. 2003. The Web as a Parallel
Corpus.Computational Linguistics, 29: 349–380.
Shi, F.; Suzgun, M.; Freitag, M.; Wang, X.; Srivats, S.;
V osoughi, S.; Chung, H. W.; Tay, Y .; Ruder, S.; Zhou, D.;
Das, D.; and Wei, J. 2022. Language Models are Multilin-
gual Chain-of-Thought Reasoners.ArXiv, abs/2210.03057.Shi, Z.; Sun, W.; Gao, S.; Ren, P.; Chen, Z.; and Ren, Z.
2024. Generate-then-ground in retrieval-augmented gen-
eration for multi-hop question answering.arXiv preprint
arXiv:2406.14891.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics.
Wu, S.; Tang, J.; Yang, B.; Wang, A.; Jia, K.; Yu, J.; Yao,
J.; and Su, J. 2024. Not All Languages are Equal: Insights
into Multilingual Retrieval-Augmented Generation.ArXiv,
abs/2410.21970.
Xu, S.; Pang, L.; Yu, M.; Meng, F.; Shen, H.; Cheng, X.;
and Zhou, J. 2024a. Unsupervised Information Refine-
ment Training of Large Language Models for Retrieval-
Augmented Generation.ArXiv, abs/2402.18150.
Xu, Y .; Hu, L.; Zhao, J.; Qiu, Z.; Ye, Y .; and Gu, H. 2024b.
A Survey on Multilingual Large Language Models: Corpora,
Alignment, and Bias.Frontiers Comput. Sci., 19: 1911362.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou,
F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.;
Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,
J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li,
M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men, R.;
Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren, X.;
Wang, X.; Zhang, X.; Ren, X.; Fan, Y .; Su, Y .; Zhang, Y .;
Zhang, Y .; Wan, Y .; Liu, Y .; Wang, Z.; Cui, Z.; Zhang, Z.;
Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report.arXiv
preprint arXiv:2505.09388.
Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.;
Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu,
K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.;
Zhu, Q.; Men, R.; Lin, R.; Li, T.; Xia, T.; Ren, X.; Ren, X.;
Fan, Y .; Su, Y .; Zhang, Y .; Wan, Y .; Liu, Y .; Cui, Z.; Zhang,
Z.; and Qiu, Z. 2024. Qwen2.5 Technical Report.arXiv
preprint arXiv:2412.15115.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question An-
swering. InConference on Empirical Methods in Natural
Language Processing.
Yu, Z.; Li, T.; Wang, C.; Chen, H.; and Zhou, L. 2025. Cross-
Lingual Consistency: A Novel Inference Framework for
Advancing Reasoning in Large Language Models.ArXiv,
abs/2504.01857.
Zeng, J.; and Yang, J. 2024. English language hegemony:
retrospect and prospect.Humanities and Social Sciences
Communications, 11: 1–9.

Appendix A: Supplementary Fallback Analysis on
Musique and DuReader
To validate the generality of our fallback observations in
Section 2.4, we present language drift patterns on the
MusiqueandDuReaderdatasets in Figures 5 and 6. Across
both datasets, we observe consistent fallback-to-English be-
havior, though with greater variability compared to Hot-
potQA.
ZH AR RU
T arget LanguageZH
EN
AR
RUContext Language94.8% 91.2% 100.0%
99.1% 100.0% 100.0%
93.6% 74.1% 86.7%
62.8% 67.3% 100.0%Musique (LLaMA3.2-8B)
ZH AR RU
T arget Language94.7% 51.4% 69.0%
100.0% 81.8% 85.2%
88.7% 45.0% 56.1%
94.7% 75.4% 78.6%Musique (Qwen2.5-7B)
0.7%0.8%0.9%1.0%
Drift-to-English Rate
Figure 5: Language drift patterns on the Musique dataset for
LLaMA3-8B and Qwen2.5-7B models. Each cell shows the
percentage of inconsistent outputs that are generated in En-
glish.
ZH AR RU
T arget LanguageZH
EN
AR
RUContext Language85.2% 90.0% 98.4%
94.9% 95.7% 98.7%
77.8% 32.3% 35.9%
45.6% 54.3% 80.2%DuReader (LLaMA3.2-8B)
ZH AR RU
T arget Language100.0% 50.0% 75.4%
100.0% 72.6% 80.6%
89.3% 32.4% 24.4%
94.9% 58.5% 68.1%DuReader (Qwen2.5-7B)
0.6%0.7%0.8%0.9%
Drift-to-English Rate
Figure 6: Language drift patterns on the DuReader dataset
for LLaMA3-8B and Qwen2.5-7B models. Each cell shows
the percentage of inconsistent outputs that are generated in
English.
InMusique, we find that most drifted genera-
tions—across all target languages—are still predominantly
in English. For example,LLaMA3-8Bproduces over 90%
English outputs even under typologically distant pairs such
asRU-{ZHandAR–RU.Qwen2.5-7B, while showing
slightly lower fallback rates in certain conditions (e.g.,
45.0% underAR-AR), still defaults to English in the majority
of inconsistent cases across settings.
InDuReader, the fallback-to-English pattern remains
dominant but reveals greater variability.LLaMA3-8Bmain-
tains high fallback rates (e.g., 98.4% underZH–RU, 94.9%
underEN-ZH), though notably lower values appear under
Arabic contexts (e.g., 32.3% underAR–AR).Qwen2.5-7B
also demonstrates strong fallback tendencies (e.g., 80.6%
underEN–RU), while showing reduced English bias under
certain low-resource combinations such asAR–RU(24.4%).Across both datasets, English remains the most frequent
fallback language in drifted outputs regardless of context-
target language configuration. These results reinforce the
main finding from HotpotQA:English acts as a seman-
tic attractor during multilingual generation, driven by
token-level priors and model-internal biases.
Appendix B: Supplementary Drift Type Analysis
on Musique and DuReader.
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RU
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RUDrift Type
A
B
C
D
Figure 7: Distribution of four language drift types across dif-
ferent target–context language pairs in the Musique dataset.
Each subplot corresponds to a fixed target language (ZH,
AR, RU), with the x-axis denoting the context language. The
top row displays results for LLaMA3-8B, and the bottom
row for Qwen2.5-7B.
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RU
EN AR RU0255075100Percentage (%)T arget: ZH
EN ZH RUT arget: AR
EN ZH ART arget: RUDrift Type
A
B
C
D
Figure 8: Distribution of four language drift types across
different target–context language pairs in the DuReader
dataset. Each subplot corresponds to a fixed target language
(ZH, AR, RU), with the x-axis denoting the context lan-
guage. The top row displays results for LLaMA3-8B, and
the bottom row for Qwen2.5-7B.
To confirm the robustness of our drift taxonomy across
datasets, we analyze the distribution of four language drift
types onMusiqueandDuReader, as shown in Figures 7
and 8. Consistent with the patterns observed in HotpotQA
(Figure 4), the majority of drifted outputs are dominated by
Type A: Named Entity Representation Divergence, fol-
lowed byType B: Answer Target Shift, withType Cand
Type Doccurring less frequently.
On both datasets,LLaMA3-8BandQwen2.5-7Bex-
hibit similar relative proportions across all target languages
(ZH,AR,RU), indicating the generalizability of our cate-
gorization. For example, in Musique, Type A accounts for

Type Name Definition Impact
Type A Named Entity Representa-
tion DivergenceInconsistent spelling or transliteration of
named entities across languages.Surface-level mismatch despite se-
mantic equivalence.
Type B Answer Target Shift (Granu-
larity / Focus)Answer scope or specificity shifts, or
distractor-language terms mislead atten-
tion.Plausible but incorrect answer due to
focus drift.
Type C Reasoning Chain Misalign-
mentReasoning path is disrupted due to mixed-
language context or code-switching.Logical inconsistency, premature or
broken inference.
Type D Conceptual Reference Shift Background semantics or cultural cues ac-
tivate unintended knowledge concepts.Factually incorrect but fluent outputs.
Table 4: Taxonomy of language-induced reasoning drift types observed in multilingual Chain-of-Thought generation. Each type
captures a distinct failure mode driven by multilingual input interference.
over 50% of drifted outputs in most configurations, par-
ticularly under typologically mismatched contexts such as
ZH–ARorAR–RU. Type B remains the second most com-
mon error mode, often reflecting shifts in answer scope or
misalignment introduced by distractor-language cues. Types
C (reasoning chain misalignment) and D (conceptual refer-
ence shift) appear less frequently but are more prominent
under challenging cross-lingual settings, such asRU–ZHor
AR–ZH.
The trends in DuReader further reinforce this taxonomy.
Across both models, Type A consistently accounts for the
largest share of drifted outputs, with Qwen2.5-7B again
showing a slightly stronger skew toward Type A, particu-
larly in Chinese and Arabic targets. While the absolute pro-
portions vary modestly across datasets, the relative ordering
of drift types is stable.
These results confirm that language drift is not a mono-
lithic phenomenon, but manifests in structured and in-
terpretable patterns across different multilingual settings.
The consistent dominance of surface-level divergences
(Type A) and granularity shifts (Type B) suggests that
decoding-time interventions should particularly target entity
and answer-level alignment, while maintaining robustness
against deeper semantic perturbations.
Appendix C: Drift Type Taxonomy.
To support our qualitative and quantitative analysis of out-
put language drift in multilingual reasoning, we provide
a detailed taxonomy of drift behaviors in Table 4. Each
type captures a distinct failure mode that arises from cross-
lingual interference during generation, particularly in Chain-
of-Thought (CoT) decoding.
•Type A: Named Entity Representation Divergence
refers to inconsistent spelling, transliteration, or surface
form of named entities across languages. This results
in semantically correct but surface-mismatched outputs,
and is often the most frequent drift type observed.
•Type B: Answer Target Shiftdescribes cases where the
model answers a semantically adjacent but incorrect sub-
target, often due to focus drift or misleading cues from
distractor-language evidence. These outputs are typically
plausible but factually wrong.
•Type C: Reasoning Chain Misalignmentinvolves log-
ical disruption in the CoT path, such as abrupt switchingbetween languages, premature termination, or reasoning
loops. These are less frequent but critical, as they under-
mine answer validity.
•Type D: Conceptual Reference Shiftcaptures cultur-
ally or semantically biased generations, where back-
ground knowledge in the dominant language (e.g., En-
glish) activates unintended associations. Though rare,
these often produce fluent but incorrect content.
This taxonomy enables structured annotation and inter-
pretation of drifted outputs, and informs the design of tar-
geted mitigation strategies. As shown in Figures??–??,
the relative prevalence of these types is consistent across
datasets and models, with A and B accounting for the ma-
jority of drift cases.
Appendix D: ICL Improves Performance but
Undermines Consistency
Table 7 reports the full evaluation metrics across all tar-
get–context language pairs, with and without ICL, on the
HotpotQA,Musique, andDuReaderdatasets. These re-
sults correspond to the high-level trends shown in Figure 2
and provide detailed quantitative insights into how ICL in-
fluences both task performance and output language con-
sistency. Across datasets, we observe consistent trends that
echo our radar plot findings:
•ICL significantly improves task accuracyas measured
by ROUGE and BLEU. For example, in HotpotQA under
theZH{ENcondition, ROUGE improves from 0.124 to
0.182, and BLEU from 0.054 to 0.086.
•ICL often reduces language consistencyin cross-
lingual settings. Notably, in theZH{ENcase, LC drops
from 0.883 (w/o ICL) to 0.684 (w/ ICL), despite bet-
ter content metrics—confirming that improved reasoning
can come at the cost of linguistic fidelity.
•The LC drop is most severe for typologically distant
language pairs, particularly when English is the con-
text language for non-English targets. For instance, in
Musique underAR{EN, LC drops from 0.992 to 0.774.
•The EN target language is highly robustacross con-
figurations, maintaining near-perfect LC and strong
ROUGE/BLEU gains even under multilingual contexts
(e.g.,EN{ZH,EN{AR).

0.100.150.200.250.300.350.400.45ROUGE
ROUGE - Context: EN
PLI
SCD
0.150.200.250.300.350.40ROUGE
ROUGE - Context: AR
PLI
SCD
0.100.150.200.250.300.350.40ROUGE
ROUGE - Context: RU
PLI
SCD
<50 50-100 100-150 150-200 >200
Reasoning Length6065707580859095LC (%)
Language Consistency - Context: EN
PLI
SCD
<50 50-100 100-150 150-200 >200
Reasoning Length65707580859095100LC (%)
Language Consistency - Context: AR
PLI
SCD
<50 50-100 100-150 150-200 >200
Reasoning Length60708090100LC (%)
Language Consistency - Context: RU
PLI
SCDFigure 9: ROUGE and LC scores ofPrompted Language Instruction (PLI)andSoft-Constrained Decoding (SCD)across
different reasoning length intervals on the HotpotQA dataset (target language:ZH). Each column corresponds to a different
context language (EN,AR,RU). SCD consistently outperforms PLI across all bins, with the most significant gains observed in
the 50–150 token range. LC improvements are particularly dramatic, with SCD achieving over 95% consistency even under
strong cross-lingual interference.
These complete results support the conclusion that while
ICL enhances factual correctness and reasoning depth, it am-
plifies exposure to cross-lingual evidence, thereby increas-
ing the risk of output language drift.
Appendix E: Decoding Strategy Comparison on
Qwen2.5-7B.
Table 6 presents the full evaluation results of three decoding
strategies—Prompted Language Instruction, Translation-
Based Evaluation, and our proposed Soft-Constrained De-
coding (SCD)—onQwen2.5-7Bacross the HotpotQA,
Musique, and DuReader datasets. These results complement
Table 2.6 forLLaMA3-8B, and confirm that the benefits of
SCD generalize across different model architectures.
Consistent with our findings onLLaMA3-8B, we observe
the following trends:
•Prompted-only decodingleads to moderate task perfor-
mance but suffers from substantial language drift. For ex-
ample, on HotpotQA withZH{EN, LC drops to 78.0%,
despite achieving a ROUGE score of 0.331.
•Translation-based evaluationachieves perfect LC by
construction (100% for all settings), but generally un-
derperforms SCD in ROUGE and BLEU. This highlights
that post-hoc translation does not recover reasoning fi-
delity lost during drift. For instance, in MusiqueAR{EN,SCD yields higher ROUGE (0.272 vs. 0.221) and BLEU
(0.158 vs. 0.125), despite slightly lower LC (96.6% vs.
100%).
•SCD consistently outperforms Prompted decodingin
both content quality and LC. Across nearly all context-
target configurations, SCD raises LC by 5–15 percent-
age points compared to Prompted, while also producing
longer and more accurate reasoning chains. Notably, in
DuReader underZH{EN, SCD improves LC from 94.0%
to 97.4% and ROUGE from 0.447 to 0.462.
Together with LLaMA3-8B results, these findings demon-
strate thatSCD provides a robust and general decoding-
time interventionthat enhances both reasoning qual-
ity and language alignment across multilingual RAG set-
tings—without requiring additional training or model modi-
fication.
Appendix F: Dataset Construction and Examples.
To construct multilingual versions of retrieval-augmented
QA datasets (e.g., HotpotQA), we follow a semi-automatic
pipeline that translates relevant data components using a
high-quality GPT-based translation API (GPT-4o), followed
by structural reorganization. Specifically, for each original
English example, we translate the query, answer, support-
ing context into three target languages: Chinese (ZH), Rus-

sian (RU), and Arabic (AR). This results in a multilin-
gual version of each sample, where all input components
are aligned across languages. All translations are performed
with instruction-based prompting to preserve semantic accu-
racy, named entities, and task relevance. Finally, we reorga-
nize each translated instance into a structured multilingual
format, enabling controlled experiments where the language
of any input component (query, context, exemplars) can be
independently varied.
{
    "_id": "5a73bcf055429978a71e909c",
    "query": {
      "en": "Which novelist wrote under the pen name Walter Ericson: Laura 
Esquivel or Howard Fast?",
      "zh": "哪位小说家使用笔名Walter Ericson：Laura Esquivel还是Howard Fast？
",
      "ru": "Какой из романистов писал под псевдонимом Уолтер Эриксон: 
Лаура Эскивель или Ховард Фаст?",
      "ar": "يأ يئاور بتك تحت االا ااعتتملي ئااتو نيسكترإ: اريا نلكسكسإ يأ درايه ؟للتف"
    },
    "answer": {
      "en": "Howard Melvin Fast",
      "zh": "霍华德·梅尔文·法斯特",
      "ru": "Говард Мелвин Фаст",
      "ar": "درايه نسفكسم ؟للت"
    },
    "evidence": {
      "en": [
        "Laura Esquivel (born September 30, 1950) is a Mexican novelist, screenwriter 
and a politician who serves in the Chamber of Deputies (2012-2018) for the Morena 
Party. Her first novel \"Como agua para chocolate\" (\"Like Water for Chocolate\") 
became a bestseller in Mexico and the United States, and was later developed into 
an award-winning film.",
        "......"
      ],
      "zh": [
        "劳拉·埃斯基维尔（1950年9月30日出生）是墨西哥小说家、编剧兼政治
家，曾在墨西哥众议院（2012-2018年）代表莫雷纳党任职。她的第一部小说
《巧克力情缘》（\"Like Water for Chocolate\"）在墨西哥和美国成为畅销书，
并后来被改编成一部获奖电影。",
        "......"
      ],
      "ru": [
        "Лаура Эскивель (родилась 30 сентября 1950 года) — мексиканская 
писательница, сценаристка и политик, которая работала в Палате депутатов 
(2012-2018) от партии Morena. Ее первый роман \"Como agua para chocolate\" 
(\"Как вода для шоколада\") стал бестселлером в Мексике и Соединенных 
Штатах и позже был экранизирован в отмеченный наградами фильм.",
        "......"
      ],
      "ar": [
        "اريا نلكسكسإ (ئالو ؟ر 30 لبتعبو 1950) در يئاوسر ئبلتبر لسرليسر ئلسللسر نكتسكسر تمعإ ؟ر 
نلفم اارراا (2012-2018) اللال بزا نريسرل. يحبحت يئاستول اوئاا \"Como agua para 
chocolate\" (\"نثإ ااعلا افلربراتر\") نم اوبثو نبسمبل ؟ر ااعكتسا ئااراسلو ااعتحللا ئتا تهرسودل 
ابحبل ناا ؟سفا بلح ىفا ااملسل نم االراوز".,
        "....."
      ]
    },  }
Figure 10: Example of a multilingual sample from our con-
structed dataset. Each instance includes queries, answers,
and evidence passages in four languages: English (en), Chi-
nese (zh), Russian (ru), and Arabic (ar). This unified
JSON-style structure enables controlled evaluation and fine-
grained analysis across languages.
Appendix G: Extended Analysis: Impact of
Reasoning Length.
To further understand how language drift varies with the
depth of reasoning, we analyze model performance acrossdifferent reasoning length intervals. Specifically, we seg-
ment the generated Chain-of-Thought (CoT) outputs into
five bins based on token length:<50, 50–100, 100–150,
150–200, and>200 tokens. We then compare the perfor-
mance of SCD and PLI in terms of ROUGE and language
consistency (LC), under three different context languages.
As shown in Figure 9, SCD consistently surpasses PLI in
both ROUGE and LC across all length intervals. The per-
formance gap is most pronounced in the mid-length range
(50–150 tokens), where SCD reaches peak effectiveness. For
example, in the 100–150 bin under RU context, SCD boosts
LC to over 95%, while PLI lags significantly behind. No-
tably, even in shorter (<50 tokens) and longer (>200 to-
kens) reasoning segments, where performance is generally
more volatile, SCD maintains a clear advantage.
These results indicate that prompt-only language con-
trol (PLI) struggles with longer or more complex reason-
ing, likely due to insufficient guidance throughout the CoT
trajectory. In contrast, SCD preserves target language align-
ment throughout the decoding process by dynamically dis-
couraging non-target tokens, allowing the model to retain
both factual correctness and linguistic fidelity. This anal-
ysis reinforces the practical value of SCD, especially for
real-world multilingual applications involving long-form or
multi-step reasoning. All detailed experiments are shown in
Table 5.

<50 50-100 100-150 150-200 >200
Context Language PLI SCD PLI SCD PLI SCD PLI SCD PLI SCD
ROUGE
EN 0.219 0.166 0.169 0.451 0.204 0.334 0.168 0.228 0.081 0.180
AR 0.235 0.135 0.220 0.417 0.206 0.323 0.181 0.233 0.154 0.116
RU 0.196 0.146 0.213 0.419 0.197 0.356 0.164 0.228 0.070 0.147
LC
EN 68.5% 58.2% 67.7% 93.2% 68.5% 96.7% 73.2% 97.2% 73.3% 97.3%
AR 66.7% 68.3% 76.4% 93.8% 81.4% 97.2% 76.1% 97.8% 62.5% 98.9%
RU 56.4% 54.5% 75.5% 94.9% 74.5% 95.9% 63.0% 96.5% 60.0% 98.2%
Sample Proportion
EN 5.4% 12.2% 54.2% 22.0% 33.3% 33.1% 5.6% 17.8% 1.5% 14.9%
AR 3.6% 10.1% 48.3% 25.6% 37.7% 28.9% 8.8% 17.8% 1.6% 17.6%
RU 3.9% 8.8% 47.3% 25.7% 37.6% 29.0% 9.2% 20.0% 2.0% 16.5%
Table 5: Comparison of Prompted Language Instruction (PLI) and SCD across reasoning length bins on HotpotQA, using
LLaMA3-8B-InstructwithZHas the target language. We report ROUGE, LC, and sample proportion under three context
languages. SCD achieves consistent improvements across most length ranges, especially in the 50–150 token interval.
HotpotQA Musique DuReader
Targe
LanguageContext
LanguageROUGE BLEU LC ROUGE BLEU LC ROUGE BLEU LC
Prompted Language Instruction
ZHEN 0.331 0.196 78.0% 0.317 0.204 81.9% 0.447 0.270 94.0%
AR 0.342 0.204 86.8% 0.320 0.203 90.3% 0.448 0.268 97.2%
RU 0.337 0.202 82.3% 0.330 0.214 86.7% 0.445 0.266 96.1%
AREN 0.201 0.121 86.2% 0.126 0.073 89.0% 0.169 0.087 89.4%
ZH 0.187 0.106 89.8% 0.106 0.050 92.0% 0.160 0.076 93.2%
RU 0.206 0.122 90.0% 0.128 0.070 94.3% 0.174 0.085 93.5%
RUEN 0.240 0.162 83.2% 0.200 0.132 85.7% 0.300 0.190 87.1%
ZH 0.248 0.158 88.6% 0.163 0.094 92.7% 0.309 0.188 86.9%
AR 0.248 0.160 90.0% 0.177 0.107 93.0% 0.323 0.207 92.8%
Translation-Based Evaluation
ZHEN 0.352 0.212 100.0% 0.357 0.229 100.0% 0.455 0.272 100.0%
AR 0.366 0.221 100.0% 0.337 0.213 100.0% 0.448 0.267 100.0%
RU 0.359 0.215 100.0% 0.351 0.225 100.0% 0.448 0.266 100.0%
AREN 0.221 0.125 100.0% 0.144 0.084 100.0% 0.181 0.093 100.0%
ZH 0.202 0.111 100.0% 0.118 0.057 100.0% 0.164 0.079 100.0%
RU 0.220 0.122 100.0% 0.134 0.072 100.0% 0.181 0.090 100.0%
RUEN 0.262 0.155 100.0% 0.222 0.149 100.0% 0.323 0.204 100.0%
ZH 0.257 0.145 100.0% 0.175 0.104 100.0% 0.326 0.201 100.0%
AR 0.252 0.144 100.0% 0.184 0.114 100.0% 0.329 0.210 100.0%
Soft-Constrained Decoding (Ours)
ZHEN 0.376 0.216 97.4% 0.349 0.221 98.2% 0.462 0.278 97.9%
AR 0.366 0.214 98.2% 0.339 0.211 98.5% 0.453 0.269 98.9%
RU 0.370 0.215 97.2% 0.349 0.221 98.9% 0.453 0.268 98.9%
AREN 0.272 0.158 96.6% 0.184 0.112 99.1% 0.198 0.100 97.4%
ZH 0.239 0.131 97.1% 0.137 0.068 98.7% 0.181 0.084 97.9%
RU 0.259 0.147 97.8% 0.173 0.100 99.1% 0.197 0.096 98.8%
RUEN 0.329 0.194 97.2% 0.221 0.141 98.6% 0.325 0.201 97.7%
ZH 0.348 0.185 97.3% 0.186 0.105 98.6% 0.335 0.201 97.3%
AR 0.320 0.181 97.7% 0.193 0.113 99.1% 0.340 0.214 97.5%
Table 6: Evaluation results of three decoding strategies—Prompted Language Instruction, Translation-Based Evaluation, and
SCD onQwen2.5-7Bacross HotpotQA, Musique, and DuReader. This table complements Table 2.6 (LLaMA3-8B results),
demonstrating that SCD achieves consistent improvements in both content quality and output language alignment across mod-
els.

Dataset HotpotQA Musique DuReader
Targe
LanguageContext
LanguageROUGE BLEU LC ROUGE BLEU LC ROUGE BLEU LC
w/o. ICL
ENEN 0.292 0.142 0.999 0.177 0.084 1 0.201 0.089 0.969
ZH 0.239 0.114 0.998 0.13 0.053 0.999 0.17 0.064 0.983
AR 0.237 0.106 0.999 0.123 0.049 0.999 0.171 0.065 0.983
RU 0.237 0.109 1 0.129 0.052 0.997 0.17 0.065 0.982
ZHZH 0.238 0.13 0.981 0.177 0.092 0.979 0.303 0.158 0.997
EN 0.124 0.054 0.883 0.091 0.037 0.87 0.186 0.073 0.96
AR 0.156 0.074 0.93 0.101 0.04 0.919 0.184 0.07 0.987
RU 0.127 0.059 0.903 0.082 0.032 0.894 0.184 0.07 0.983
ARAR 0.156 0.078 0.993 0.045 0.099 0.992 0.143 0.063 0.989
EN 0.119 0.055 0.976 0.065 0.023 0.98 0.113 0.043 0.994
ZH 0.113 0.049 0.98 0.059 0.018 0.979 0.11 0.04 0.977
RU 0.115 0.051 0.978 0.056 0.016 0.982 0.115 0.043 0.992
RURU 0.193 0.106 0.991 0.108 0.054 0.985 0.193 0.095 0.995
EN 0.142 0.072 0.969 0.079 0.032 0.966 0.167 0.072 0.968
ZH 0.154 0.076 0.977 0.088 0.036 0.985 0.166 0.07 0.991
AR 0.168 0.087 0.98 0.083 0.033 0.984 0.171 0.072 0.998
with. ICL
ENEN 0.601 0.317 0.999 0.504 0.334 0.999 0.385 0.206 0.985
ZH 0.552 0.277 0.997 0.372 0.199 0.999 0.352 0.166 0.967
AR 0.553 0.282 0.999 0.356 0.199 0.997 0.349 0.159 0.982
RU 0.545 0.279 0.996 0.382 0.217 0.994 0.343 0.164 0.963
ZHZH 0.341 0.212 0.92 0.341 0.23 0.942 0.608 0.4 0.973
EN 0.182 0.086 0.684 0.187 0.097 0.639 0.339 0.166 0.842
AR 0.211 0.106 0.777 0.181 0.089 0.765 0.358 0.175 0.901
RU 0.209 0.107 0.795 0.169 0.087 0.645 0.343 0.168 0.831
ARAR 0.366 0.221 0.954 0.218 0.138 0.966 0.284 0.153 0.97
EN 0.294 0.162 0.854 0.144 0.08 0.9 0.209 0.099 0.882
ZH 0.265 0.143 0.884 0.12 0.057 0.892 0.193 0.08 0.87
RU 0.28 0.151 0.886 0.121 0.061 0.898 0.186 0.077 0.895
RURU 0.373 0.215 0.895 0.29 0.185 0.938 0.374 0.223 0.936
EN 0.333 0.177 0.802 0.218 0.119 0.819 0.285 0.15 0.843
ZH 0.335 0.172 0.851 0.206 0.102 0.902 0.296 0.149 0.858
AR 0.339 0.179 0.868 0.214 0.109 0.925 0.288 0.143 0.909
Table 7: Detailed evaluation results for all target–context language pairs on HotpotQA, Musique, and DuReader using
LLaMA3-8B, with and without ICL. Each block reports ROUGE, BLEU, and language consistency (LC) under fixed target
language (rows) and varying context language (columns). These results complement the radar plot in Figure 2, revealing that
while ICL improves content metrics across the board, it often reduces language consistency in cross-lingual settings—especially
when English evidence is introduced.
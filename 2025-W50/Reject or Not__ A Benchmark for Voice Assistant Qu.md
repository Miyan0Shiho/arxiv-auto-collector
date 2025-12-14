# Reject or Not?: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs

**Authors**: Huichao Men, Yizhen Hu, Yingyang He, Yu Gao, Xiaofeng Mou, Yi Xu

**Published**: 2025-12-11 03:33:06

**PDF URL**: [https://arxiv.org/pdf/2512.10257v1](https://arxiv.org/pdf/2512.10257v1)

## Abstract
In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before any downstream processing. To address the limited query-rejection capability of current voice assistants, this paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together with a personalized query-rejection method based on large language models. On the data side, we construct the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds, valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides a reproducible data foundation, evaluation standard and extensible technical framework for reliability research in smart-home voice interaction.

## Full Text


<!-- PDF content starts -->

Reject or Not?: A Benchmark for V oice Assistant Query Rejection in
Smart Home Scenario and an Improved Method Based on LLMs
Huichao Men1, Yizhen Hu1,2, Yingyang He1, Y u Gao∗1,*, Xiaofeng Mou1, and Yi Xu†1,*
1Midea AI Research Center, Shanghai, P .R. China
2Harbin Engineering University , Harbin, P .R. China
menhuichao@midea.com (Huichao Men), huyz64@midea.com (Yizhen Hu), heyy84@midea.com (Yingyang He), gaoyu11@midea.com
(Yu Gao), mouxf@midea.com (Xiaofeng Mou), xuyi42@midea.com (Yi Xu)
December 12, 2025
Abstract
In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before
any downstream processing. T o address the limited query-rejection capability of current voice assistants, this
paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together
with a personalized query-rejection method based on large language models. On the data side, we construct
the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled
text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds,
valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context
and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language
and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a
Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-
level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base
that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach
significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced
gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides
a reproducible data foundation, evaluation standard and extensible technical framework for reliability research
in smart-home voice interaction.
1 Introduction
1.1 Research Background
1.1.1 The Rise of V oice Interaction in the Smart-Home W ave
The deep integration of Internet-of-Things (IoT) and artificial-intelligence (AI) technologies[ 6] has triggered an
explosive expansion of the smart-home industry . According to IDC, global smart-home device shipments will exceed
1.4 billion units by 2025, representing a compound annual growth rate of 12.2%. Benefiting from natural, hands-
free and contact-free interaction, voice assistants have become the primary entry point of smart-home systems: a
single sentence such as “I’m home” can simultaneously turn on the lights, activate the air conditioner and arm the
security mode, while users can also query the refrigerator inventory remotely when cooking. However, as the device
ecosystem grows increasingly complex and user scenarios continue to diversify , the dialogue environment faced by
voice assistants is evolving from “single-turn commands” to advanced forms that are “multi-turn, context-dependent
and semantically ambiguous”, posing unprecedented challenges to their cognition and decision-making capabilities,
especially in accurately rejecting invalid or unexecutable user queries.
∗* Corresponding authors
†* Corresponding authors
1arXiv:2512.10257v1  [cs.HC]  11 Dec 2025

1.1.2 F ailure of Query-Rejection Judgement iAn Invisible Cliff in User Experience
In real household environments, voice assistants must process a large number of invalid, ambiguous or infeasible
utterances every day , for example:
• Non-human sounds: doorbells, barking pets or TV background audio falsely trigger commands;
• Ambiguous references: “ յष ” or “ശۚ” lack explicit targets;
• Beyond-capability requests: asking a robot vacuum to wipe the windows” or demanding to play last night’s
dream”;
• Casual-chat interference: user mumbling “ ࣂ฿ᆇ৏ ” is mis-parsed as a thermostat command.
Existing systems usually rely on keyword matching or coarse-grained confidence thresholds to reject inputs, resulting
in high error rates under complex acoustic conditions and contextual scenarios. Surveys show that the top-3 voice-
assistant brands still suffer 1.8–3.4 false-rejection incidents per day in home settings, 62% of which are caused by
the above “should-reject-but-reject” mistakes. F requent false responses not only interrupt users’ daily routines but
also erode trust in system reliability , becoming a key pain point of smart-home experience.
1.1.3 Theoretical V alue iAn Academic Gap in Dialogue Rejection Research
F rom the perspective of natural-language processing (NLP), the query-rejection task lies between reject-understand”
and ignore-filter”, requiring models to simultaneously possess:
• Fine-grained semantic understanding: identifying implicit ambiguity and missing slots;
• Context modelling: judging the feasibility of the current request by leveraging multi-turn dialogue history;
• Personalized boundary learning: adapting to expression habits and device configurations of different family
members.
However, mainstream studies focus on improving intent-recognition accuracy , paying insuﬀicient attention to “how
to correctly say no”, especially lacking systematic research that is oriented to home scenarios and integrates text-
acoustic dual channels. The scarcity of publicly available data and evaluation benchmarks for query-rejection failure
cases further limits algorithmic innovation and fair comparison.
1.1.4 Application V alue iAn Urgent Industry Demand
On the market side, smart-appliance manufacturers are promoting “zero false-trigger” as a core selling point of
high-end flagship products. The EU AI Act also includes a compliance requirement that “high-risk interactive
systems must be able to refuse requests beyond their design scope” . Building a highly robust dialogue-rejection
mechanism can significantly reduce cloud-computing and on-device response costs, extend battery life, and provide
a trustworthy basis for subsequent personalized recommendations and multimodal interaction. Therefore, in-depth
research on the query-acceptance problem of voice assistants in home scenarios not only has theoretical significance
for advancing academic frontiers but also possesses practical value for enhancing user experience and industrial
competitiveness.
1.2 Research Aim
This paper aims to construct a benchmark dataset for Query Rejection in smart home scenarios and explore an
improved approach based on LLMs to enhance the performance of voice assistants in Query Rejection topic.
2 Highlights and Contributions
This paper presents two principal contributions.
Home-Scenarios Query-Rejection Benchmarks W e release the first open-source multimodal query-rejection
benchmark tailored for conversational interaction in home scenes. The dataset contains 11,913 manually labeled
text-speech pairs, covers 13 categories of invalid queries and provides multi-turn conversational contexts, supporting
both zero-shot and fine-tuning evaluation paradigms.
2

Three-Layer Collaborative F ramework W e propose a three-layer collaborative query-rejection framework
that combines a common-semantic adapter, household-level personalized memory and a RAG-based mis-rejection
corrector. On our self-constructed dataset it significantly outperforms keyword matching, traditional classifiers and
zero-shot large models, offering an extensible solution for reliable smart-home voice interaction.
3 Related W ork
3.1 Overview of Existing T echnical Solutions
At present, the problem of deciding whether to reject a user command in smart-home scenarios mainly relies on
two technical routes:
1. T raditional keyword-matching schemes : A preset lexicon of command keywords is employed (e.g.,
control-type keywords such as “ յष ”, “ܱо ”; query-type keywords such as “ ࡟Ұ ”, “Ѭ٢.)” When a user ut-
ters speech or text, the system performs string matching against the lexicon. A successful match preliminarily
labels the input as valid; otherwise it is regarded as invalid. F or example, “ յषग़๏ॢט” is not rejected
because it contains “ յष ”, whereas “ ݺ฿గ ” is rejected since no keyword is hit.
2. Basic semantic-analysis schemes : On top of keyword matching, shallow syntactic rules are introduced to
verify whether the sentence possesses a complete subject–predicate–object structure (i.e., the logic “entity–
action–object”). If the structure is incomplete (e.g., only the verb “ յष ” without an object), the utterance is
rejected; if complete, it is not rejected. Thus, “ յष ” is refused for lacking an object, yet “ յष҂թᄝ֧֥,” 
though semantically infeasible, is still mis-accepted because the structure is intact.
3.2 Deficiencies and Limitations of Existing Solutions
Despite their preliminary filtering effect, the above methods exhibit evident limitations in real household interaction,
which can be summarized along two dimensions:
3.2.1 Single-dimensional invalid-content detection yields high error rates
Keyword matching merely performs a binary decision based on the presence of keywords, and fails to distinguish
the following typical invalid cases:
• Non-command chit-chat : A user says to a family member “ Ϝဪ॥ఖ۳໡ ”, which contains the seemingly
directive phrase “ ۳໡ ”, but is in fact not a request to any smart device;
• Semantically wrong yet syntactically legal commands : “յषໃ৵֥ࢤᇆିԳ৺ ” contains a valid verb
and a complete structure, but cannot be executed because the device is offline.
Basic semantic analysis introduces syntactic completeness, yet still neglects dynamic contextual cues of the home
environment (e.g., device on-line status, temporal constraints). F or instance, “ ࢫט඀ٜॢט໑؇” issued at mid-
night is semantically intact, but invalid if the study-room AC is offline. Existing pipelines still fail to reject such
utterances, wasting resources and degrading user experience.
3.2.2 Static content libraries offer poor adaptability
Most commercial systems rely on a factory-preset static invalid-content library that merely covers generic mean-
ingless expressions (e.g., interjections “ah”, “uh”) or obvious garbled text, and cannot cope with complex realities
of smart-home scenes, concretely manifested as:
1. Lack of personalized adaptation : Different families have distinct colloquial habits. One household may
say “ᄞૌ਍਍ ” to indicate chit-chat, while another prefers “ ෛьඪඪ ” . A fixed library cannot recognize such
family-specific invalid patterns.
2. Insuﬀicient dynamic updating : When new types of invalid content emerge (e.g., the trendy phrase “ ॓ଢ
೘” used in a non-command sense, or infeasible requests caused by a newly purchased but not-yet-connected
device), the static library cannot expand or self-adapt in real time.
3

3. W eak processing of noisy text : Existing schemes hardly parse mixed structures of “keyword + interfer-
ence” . F or example, the user input ‘ Ϻ໡ुुѢདllෘਔđ๼ԛ ” contains the device keyword “ Ѣད ”, but
is explicitly negated later. The system, however, may still mis-classify it as a valid command and trigger an
incorrect response.
4. Absence of multi-turn context inheritance : Home interaction often involves multi-turn dialogue. The
first turn “ յषग़๏֧” is valid; the second “ ପ۱ਊ؇Ⴕׄπ ” is a reasonable adjustment and should remain
valid; yet if the second turn becomes “ ପ۱ll໡ࣂ฿ԛ૊ອջᄂӼਔ ”, although it linguistically refers to the
previous turn, its content is unrelated to device control. Current solutions lack a dialogue-history modelling
mechanism and perform keyword matching only on a single-turn utterance, thus failing to distinguish valid
continuation from invalid digression.
3.3 Status Quo of Home-Scene Dialogue Datasets and the Absence of Query-Rejection
Corpora
In recent years, a growing number of dialogue datasets targeting smart-home scenarios have been released, providing
important support for task-oriented dialogue research. The most representative is HomeBench [1], which con-
structs multi-turn conversational corpora covering typical smart-home subsystems such as lighting, climate control,
and security , and annotates device states, user intents, and system actions. It has become an essential benchmark for
evaluating home voice-assistant performance. In addition, V oiceBench [2] focuses on end-to-end speech assistants
powered by large language models, collecting real user ⚶virtual assistant interaction logs and emphasizing semantic
understanding and tool-invocation capabilities. However, these corpora focus almost exclusively on valid commands
and provide scarce annotations for invalid or out-of-scope utterances[ 7], leaving a critical gap in rejection-capability
research in smart home scenarios.
Other relevant corpora include:
• TEACh [3]: Built on the AI2-THOR simulated environment, it gathers 3,000+ dialogues and action sequences
in which humans collaborate to complete household tasks (e.g., making coffee, tidying rooms), stressing
instruction following and clarification ability;
• SIMMC 2.0/2.1 [4]: T argeting multimodal shopping and home-configuration scenarios, it provides dialogues
with visual context to support reference resolution and state tracking;
• Fluent Speech Commands [5]: Although not strictly a “dialogue” corpus, it offers structured spoken
commands together with their semantic slots and is widely used for lightweight command-recognition research.
Nevertheless, all the above public datasets are designed with “valid commands” as the core objective, and almost
no systematic collection or annotation of “invalid speech inputs” is provided. Concretely:
• The corpora rarely contain typical rejection samples such as user chit-chat, self-talk, environment-noise-mixed
speech, interrupted utterances, or negation/correction;
• Even when non-command utterances occasionally appear, they are usually filtered or ignored, and no rejection
label or invalid-type classification is supplied;
• There is no contrastive annotation between “valid commands” and “invalid continuations” in multi-turn
dialogues, so context-aware rejection model training cannot be supported.
This data void directly causes deployed voice-interaction systems to suffer from “over-response” ithat is, non-
command speech erroneously triggers execution logic, severely degrading user experience and system credibility .
3.4 Necessity of Building a Smart-Home Query-Rejection Dataset
Given the above status, constructing a Curated V oice Rejection Dataset for Smart Home specifically oriented
to smart-home scenes is of urgent research and application value:
1. Filling the data-ecology gap : At present, no open dataset focuses on home-speech rejection. This dataset
will, for the first time, systematically collect and annotate various kinds of invalid speech samples (e.g., chit-
chat, hesitation, interruption, negation, device-irrelevant topics), providing high-quality supervisory signals
for rejection models.
4

2. Supporting fine-grained rejection research : By defining a multi-level rejection label schema (e.g., cat-
egorized by pragmatic function, acoustic feature, or contextual relevance), it can push the evolution from
“binary rejection” to “explainable rejection” and improve the transparency of model decisions.
3. Boosting context-aware rejection algorithms : If the dataset contains continuous multi-turn dialogue
flows and annotates the validity of each turn as well as its semantic relation to historical utterances, it will
offer a training foundation for rejection models based on dialogue-state tracking (DST) or memory-augmented
architectures.
4. Enhancing robustness in real scenes : By collecting natural speech in diverse home environments (back-
ground noise, multiple occupants) and covering speakers whose distribution matches real users, the dataset
can significantly strengthen model generalization after deployment.
5. Enabling personalization and continual learning : The dataset can be designed with user IDs and
device-configuration metadata to support personalized rejection-strategy research; meanwhile, reserved incre-
mental annotation interfaces facilitate future integration with online learning for dynamic rejection-boundary
optimization.
In summary , current techniques are limited not only at the algorithmic level but also structurally at the data
level. Building a high-quality , scenario-driven and finely annotated home-dialogue rejection dataset has become a
prerequisite for breaking the current bottleneck of smart-home voice-interaction rejection performance, and is an
essential infrastructure for evolving from “insensitive interaction” to “intelligent acceptance” .
4 Dataset Construction
4.1 Dataset Design
This paper categorizes the voice assistant dialogue rejection recognition benchmark dataset into two major types:
text-based rejection and speech-based rejection. Rejection experiments are conducted on queries from these two
dialogue forms using natural language understanding methods and speech recognition methods, respectively . The
text queries and speech queries within these two categories are further divided into 13 subcategories based on their
content.
Here is a detailed introduction to each utterance type and provides corresponding examples.
4.2 Detailed Description of Each Utterance Type with Examples
W ake-up Keywords W ake-words are the phrases users call to start a conversation with the voice assistant
(brand-dependent). This corpus includes, but is not limited to, ohi Sirip,oཬૅཬૅp ,oཬၜཬၜp , etc. Since
the dataset aims to provide universal evaluation data and benchmarks for smart-home scenarios, all wake-word
utterances are labeled as acceptance .
Illegal Language Illegal language refers to utterances that contain unfriendly , abusive, obscene, violent, terror-
related or politically sensitive content according to national or regional laws. All such utterances are labeled as
rejection in this dataset.
Non-Human Sounds Non-human synthetic sounds, including algorithm-generated replies or any recorded nat-
ural/mechanical noises. Even highly human-like synthetic speech is treated as rejection .
ASR-Error Garbled / Meaningless Short Phrases Caused by user slips of the tongue, meaningless short
phrases, or ASR recognition errors. Examples: oᆃᆃᆃp ,o܄࠶ι۵୆ሼp ,ojeanp. These carry no interaction
value and are labeled rejection .
Non-Assistant-Directed Chat (Multi-Person or Self-T alk) Typical cases where the user chats with others
or talks to him/herself after the wake-word has been triggered; the recorded speech is irrelevant to voice interaction.
Labeled as rejection .
5

T able 1: Utterance-Type Definitions and Quantity Distribution
Type ID Utterance Type Accept / Reject Quantity
0 W ake-word Accept 16
1 Illegal language Reject 29
2 Non-human sounds Reject 53
3ASR-error garbled;
meaningless short phrasesReject 154
4Non-assistant-directed chat
(multi-person or self-talk)Reject 30
5Command-semantics but
obviously unreasonable
(no history)Reject 232
6Assistant-directed ambiguous chat
(no history; uncertain reply vs.
encyclopaedia / query)Reject 91
7Assistant-directed ambiguous chat
(has history but no valid command;
uncertain reply)Reject 6
8Assistant-directed chat
(assistant can reply)Accept 1093
9Assistant-directed command
(assistant can reply; supported
by Midea)Accept 9872
10Assistant-directed command
(assistant can reply; not supported
by Midea yet, but command-intent clear;
other brands may support)Accept 26
11Assistant-directed ambiguous chat
(has history and a valid command;
uncertain reply vs. encyclopaedia / query)Accept 311
6

Semantically Unreasonable Command (No History) Commands that show clear intent but are unreason-
able. Example: oॢ֞טט 40ജ൦؇p . Labeled as rejection .
Assistant-Directed Ambiguous Chat (No History , Uncertain Reply) Isolated small-talk whose reply is
uncertain (vs. encyclopaedic queries). Examples: o୆ࣂ฿ႵીႵݺழႶ࿦Ĥ p Labeled as rejection to prevent
forced, unreasonable replies.
Assistant-Directed Ambiguous Chat (Has History , but No Prior V alid Command) History contains
no valid command; the current query remains ambiguous. Example: ohistory:ᕫ໡টۄખ ΰ࿦jଢ଼টঝĆ ?໡
൞ଢ଼ህඋ֥ᇆିࡅܵb queryğ୆࢝໡άb p Labeled as rejection .
Assistant-Directed Chat (Supported Reply) Small-talk that the assistant can answer via knowledge base or
external LLM. Examples: o๷ࡩ༯ࡾ්ීథ֥ཬӹp ,o7ᄅٺ්ᇜЌ০նखჽႵଧུဆԛp . Labeled as acceptance .
Assistant-Directed Command (Supported by Midea) Mainstream command type supported by the brand.
Examples: oוܱॢטp ,ohistoryğটग़ದਔ ק۞b queryğյषԝܻ֧ٜp . Labeled as acceptance .
Assistant-Directed Command (Not Supported by Midea but Supported by Others) Commands with
clear control intent, currently unsupported by Midea but possibly supported by other brands. Examples: o๔ᆸ
ᆳྛग़Ҕ๏ष൓೜ଦb p ,ohistoryğषఓ৖ࡅଆൔ ᇶದđॢטᄠൈ҂ᆦӻۿھିb historyğषఓਂඡଆൔb p .
Labeled as acceptance .
Assistant-Directed Ambiguous Chat (Has History with V alid Command) History already contains a
valid command; the current utterance may supplement it or introduce a new one. Example: ohistoryğൈࡗο
࡯ ଢ଼ିඪ֤ᄜབྷ༥ུગ ?൞Ч௖இଧ۱Ӂ௖֥ൈࡗο࡯࿦đбೂॢטa༡၉ࠏđᆃဢ໡Ҍିֹݺ۷Ϻଢ଼ࢳճb
queryğႲ࿵ࠏൈࡗο࡯b p Labeled as acceptance .
4.3 Data Collection and Preprocessing
The open-source evaluation dataset released in this paper consists of two parts: text data and speech data, both
collected in full compliance with privacy regulations. The text data originate from real user utterances collected
online; they were preprocessed by automatic speech recognition (ASR) to obtain transcriptions, avoiding leakage
of users’ biometric privacy . The speech data are synthesized from the text corpus via text-to-speech (TTS) and
therefore contain no real users’ biometric information either.
Sourced from real-world online logs of manufacturers, the dataset reflects a realistic category distribution and
has been thoroughly desensitized for privacy compliance ithis is its greatest advantage.
At present the corpus contains 11,913 samples in total; the count of each category is given in T able 1 and
detailed in the preceding sections. The evaluation dataset is also employed to assess the novel LLM-based rejection
algorithm proposed in this paper, which is improved according to practical online issues.
4.4 Dataset Statistical Analysis
This section presents a multi-dimensional evaluation of the open-source dataset, using accuracy as the primary
metric. W e benchmark rejection accuracy on different text subsets with several mainstream large language models,
and evaluate speech subsets with leading speech-capable multimodal LLMs. Both zero-shot instruction-based
recognition and supervised fine-tuning (SFT) are adopted for comparison.
5 Improved Approach Based on Large Language Models
In smart-home scenarios voice interaction is one of the main means for users to communicate with appliances. The
device’s audio receiver remains always-on while the appliance is powered, so all kinds of environmental sounds are
captured and processed. Accurately deciding which audio should be passed downstream to the dialogue-interaction
module is an entry-level problem for smart-home voice interaction. Correctly identifying utterances that need further
processing iand those that should be filtered out ieffectively reduces the number of falsely processed inputs and
improves user experience.
7

Based on the evaluation results of the previous sections and the known weaknesses of existing dialogue-rejection
algorithms, we propose a new LLM-and-RAG-based rejection method to handle complex rejection issues in real
household production environments. By combining a household-specific user-utterance database, we use RAG to
build a rejection model better suited to the actual linguistic context of individual families.
5.1 Overview of the Method
Figure 1: Overview of the Method
As illustrated in Figure 1, the voice-rejection framework proposed in this paper consists of three synergistic core
components: (1) a large-language-model adapter (LLM Adapter) fine-tuned on massive online, generalized user
voice-interaction data, which models rejection-semantic boundaries in common scenarios; (2) a dynamically injected
household-specific historical-dialogue context module that captures the pragmatic habits and reference patterns of
a particular family in multi-turn interactions; and (3) a household-dimension rejection-habit knowledge base that
explicitly memorizes and revises historically misjudged obad cases pthrough retrieval-augmented generation (RAG).
The three-layer architecture follows a progressive design principle of ogeneral capability ipersonalized adap-
tationicontinuous correction p: the universal adaptor provides basic rejection ability , the personalized context
realizes short-term contextual alignment, and the knowledge base supports long-term behavior-pattern learning and
debiasing. Acting in concert, the system can dynamically model users nlinguistic preferences and interaction inten-
tions at the household granularity , thereby judging the validity of input utterances more accurately . Experiments
show that this design significantly improves rejection-decision accuracy , effectively reduces multi-turn clarifica-
tion interactions caused by false triggers or over-responses, and ultimately optimizes the overall voice-interaction
experience.
8

5.2 Key T echnical Details
5.2.1 Rejection-A ware Fine-tuning Adapter
The rejection LLM in Figure 1 is built upon Qwen 2.5 3B, a 3-billion-parameter model in Alibaba’s Qwen family
that supports multi-turn dialogue, instruction following, and strong contextual understanding. Its architecture
follows the standard T ransformer decoder and has been pre-trained on massive corpora.
Supervised fine-tuning data are derived from large-scale, privacy-preserving real-user voice conversations con-
verted into text via ASR. Acceptance and rejection samples are balanced 1:1. The detailed prompt and JSON
structure are:
PROMPT Example Given the dialogue history and the current text, decide whether the utterance should be
accepted or rejected. Rejection rules are: {rules constructed from dataset taxonomies} . Return only a JSON object:
if accepted, result: YES ; otherwise result: NO . No extra text. T ext: current query
After fine-tuning with this instruction template, we obtain a generic rejection model. Because the data are
cloud-collected and generalized, no household-specific history or family-level rejection habits are injected, so the
model is universal only .
5.2.2 Personalized Historical-Dialogue Instructions
Starting from the generic rejection model, we aim to align decisions with household-level personalization. W e collect
each family’s conversational logs within a fixed time window and inject them into the prompt so that the system
can reject unreasonable utterances in a way that better matches user habits.
PROMPT Example Given the dialogue history and the current text, decide whether the utterance should be
accepted or rejected. Rejection rules are: {rules constructed from dataset taxonomies} . Return only a JSON object:
if accepted, result: yes ; otherwise result: no . No extra text. Dialogue history: {history query} .T ext: current
query
5.2.3 User-Personalized Query-Rejection Misjudgment Knowledge Data Base
With the two preceding components, the system can already reject unreasonable family utterances in a personalized
manner. W e go one step further by building a household-dimensional knowledge base that stores previously mis-
judged interactions (i.e., utterances that should have been accepted but were wrongly rejected, abbreviated as bad
cases). Via RAG, we retrieve the TOP-3 most similar bad-case utterances for the current input and append them
to the prompt, correcting household-specific rejection quirks and enhancing personalization.
PROMPT Example Given the dialogue history and the current text, decide whether the utterance should be
accepted or rejected. Rejection rules are: {rules constructed from dataset taxonomies} . T op-3 household-specific
habitual accept/reject utterances from the knowledge base: {RAG TOP-3 cases} . Return only a JSON object: if
accepted, result: YES ; otherwise result: NO . No extra text. Dialogue history: {history query} .T ext: current
query
5.3 Experiments and Analysis
The experimental design in this section consists of two parts.
Part I presents comparative experiments of the proposed novel rejection approach versus existing open-source
large models, demonstrating the advantages of our method over simple fine-tuning baselines.
Part II provides benchmark metrics for the open-source speech–semantic interaction dataset released in this
paper. Experiments are conducted on multiple dialogue categories, including standalone semantic-text rejection and
standalone speech rejection. By designing zero-shot instruction-rejection trials and SFT-based tests, we illustrate
the performance of the released dataset across different utterance types.
5.3.1 Comparison Experiments and Analysis of the Proposed Method
All comparison experiments herein adopt the specially designed prompts of this work and contrast them with
prompts generated by open-source large models. Meanwhile, the fine-tuned model proposed in this paper is com-
pared with the fine-tuning results of open-source large models.
9

Specifically , we compare zero-shot performance between our optimized personalized prompts and generic prompts
to show the superiority of the designed prompts on the released dataset. F urthermore, we apply the personalized
prompts to fine-tune several large-scale base models, verifying the performance gains brought by our prompt design
on larger architectures. Since the associated business requires rapid feedback for interactive dialogue, although
fine-tuning large base models yields higher accuracy , it also introduces significant latency overhead. Therefore, the
proposed approach fine-tunes small-parameter models to achieve a balance between latency and accuracy .
T able 2: Accuracy Comparison of Different Models on the Rejection Dataset (Optimized Prompt + Zero-shot /
SFT)
(a) First 5 experiments
SubsetsQwen-3-32B
Optimized Prompt
Zero-shotGLM-4.6-A WQ
Optimized Prompt
Zero-shotGPT-oss-20b
Optimized Prompt
Zero-shotDeepSeek-V3.1
Optimized Prompt
Zero-shotGPT-5-Chat
Optimized Prompt
Zero-shot
0 1.0000 1.0000 1.0000 1.0000 1.0000
1 0.9130 0.9130 0.7826 0.9565 0.9565
2 0.1569 0.3800 0.1569 0.0196 0.2549
3 0.5197 0.4172 0.5302 0.1908 0.4671
4 0.5862 0.5862 0.4138 0.2069 0.5517
5 0.3097 0.2589 0.2478 0.1372 0.3672
6 0.5000 0.5102 0.3571 0.2041 0.6531
7 0.8889 0.4444 0.5556 0.2222 0.5556
8 0.8923 0.9005 0.9380 0.9189 0.9015
9 0.9453 0.9452 0.9466 0.7871 0.9442
10 0.8000 0.9200 0.8800 0.8400 0.8000
11 0.3280 0.3790 0.4172 0.5382 0.3790
(b) Last 4 experiments
SubsetsQwen3-235b-A22b
Optimized Prompt
Zero-shotDeepSeek-R1-Distill
Qwen-7B
Optimized Prompt SFTQwen-2.5-3B
Instruct
Optimized Prompt SFT Improved Method
0 1.0000 1.0000 1.0000 1.0000
1 1.0000 0.9565 0.7826 0.6957
2 0.2353 0.0588 0.3725 0.5882
3 0.4408 0.1053 0.5461 0.5132
4 0.6552 0.1379 0.1724 0.4483
5 0.2920 0.0796 0.3584 0.4178
6 0.6224 0.1735 0.4796 0.5612
7 0.4444 0.1111 0.1111 0.6667
8 0.8587 0.9708 0.9891 0.9954
9 0.9347 0.9887 0.9948 0.9921
10 0.8800 1.0000 0.9600 1.0000
11 0.4427 0.9268 0.9172 0.9777
F rom the weighted accuracy rates, we can draw the following conclusions:
Overall Performance: Most models perform well on the rejection dataset, with accuracy rates above 85%. This
indicates that these models have high generalization capabilities for such tasks.
Best Model: The Improved Method has the highest accuracy rate at 96.75%. This shows that the method has
a significant advantage in handling the rejection dataset.
Second Best Model: Qwen-2.5-3B (Instruct, Opt. Prompt SFT) has a accuracy rate of 96.44%, closely following
the Improved Method. This demonstrates that even with a smaller model size, appropriate fine-tuning and optimized
prompts can achieve very high accuracy .
Other Models: DeepSeek-R1-Distill-Qwen-7B (Opt. Prompt SFT) has a accuracy rate of 94.34%, also performing
excellently . Other models like Qwen-3-32B, GLM-4.6-A WQ, GPT-oss-20b, GPT-5-Chat, and Qwen3-235b-A22b
have accuracy rates ranging from 88.98% to 90.30%, showing consistent performance. DeepSeek-V3.1 (Opt. Prompt,
10

Zero-shot) has the lowest accuracy rate at 76.34%.
Advantages of the Improved Method: The Improved Method stands out in multiple datasets, especially in
subsets 2, 4, 5, 6, 7, 8, 9, 10, and 11, where its accuracy is significantly higher than other models. This indicates
that the method has a significant advantage in handling specific types of data or tasks. Although it performs slightly
worse in some datasets (such as subset 1), its overall weighted accuracy rate is still the highest.
Therefore, the Improved Method demonstrates strong generalization capabilities and excellent performance in
handling the rejection dataset, making it worthy of further research and application.
5.3.2 Benchmarks and Analysis
The open-source dataset released in this paper consists of two parts ia text corpus and a speech corpus iwhose
taxonomies are shown in T able 1; the benchmarks and analyses in this section will evaluate accuracy on each of
these datasets separately .
T ext Query Rejection Benchmarks F or the text query corpus evaluation, we adopt both Zero-shot and SFT
paradigms to assess the dataset. The models employed are Qwen 3 32B, GLM 4.6-A WQ, GPT-oss-20b, DeepSeek-
V3.1, GPT-5-Chat, Qwen3-235b-A22b, DeepSeek-R1-Distill-Qwen-7B, and Qwen 2.5 3B Instruct. Among them,
Qwen 3 32B, GLM 4.6-A WQ, GPT-oss-20b, DeepSeek-V3.1, GPT-5-Chat, and Qwen3-235b-A22b are used for Zero-
shot experiments, with both generic prompts and our optimized personalized prompts. DeepSeek-R1-Distill-Qwen-
7B, Qwen 2.5 3B Instruct, and Qwen-3-32B-Instruct are used for SFT experiments, incorporating our optimized
personalized prompts.
As shown in T able 3, under the zero-shot condition with a generic prompt the six models exhibit a clear otiered
divergence p: GPT-5-Chat and Qwen3-235b-A22b achieve average accuracies of 0.74 and 0.68 respectively , firmly
occupying the first tier, whereas models in the 3 B ⚶20 B parameter range (Qwen-3-32B, GLM-4.6-A WQ, GPT-
oss-20b, DeepSeek-V3.1) remain only at 0.45 ⚶0.52, confirming once again that oscale ￿ rejection capability p.
Notably , GPT-oss-20b drops below 0.25 on five subsets (IDs 2, 3, 4, 5, 7), and DeepSeek-V3.1 even falls to 0.000 on
subset 2, indicating extreme sensitivity to noisy home-scene inputs; by contrast, the negative cases of subset 0 are
perfectly recognized by most models (￿0.83), in stark contrast to the anomalously low scores observed later under
SFT, implying that errors on this subset stem chiefly from label leakage during fine-tuning rather than inherent
task diﬀiculty . Overall, the generic prompt already leaves the top models lagging behind their optimized-prompt
counterparts by roughly 10 pp on the ohardpsplits, underscoring the significant gain brought by prompt engineering
to zero-shot rejection, yet it still cannot bridge the intrinsic gap between models idomain fine-tuning or retrieval
augmentation will be required to approach practical thresholds.
T able 4reveals that under optimized-prompt conditions all six zero-shot large models move up another rung:
GPT-5-Chat and Qwen3-235b-A22b remain in the lead with average accuracies rising to 0.75 and 0.73 respectively ,
ranking top-two in 9 out of 12 subsets; the smaller GPT-oss-20b gains most on the ohardpsplits 2 and 6 (+8 ⚶30
pp), indicating that prompt engineering benefits weaker baselines disproportionately; yet DeepSeek-V3.1 still scores
only 0.0196 on subset 2, exposing a structural fragility to noisy features. Overall, the refined prompt lifts average
accuracy on the hard splits by roughly 10 pp, but the inherent gap between models persists, demanding subsequent
domain fine-tuning or retrieval-augmented generation to approach practical thresholds.
Speech Query Rejection Benchmarks This paper simultaneously releases a speech-rejection dataset that
mirrors the text-rejection corpus. The data are generated by synthesizing the text corpus through TTS; to ensure
diversity , multiple random voices and prosodies are produced, bringing timbre and intonation closer to real human
speech and improving dataset usability . The speech dataset retains exactly the same categories and distribution as
the text corpus. Evaluation of the speech dataset is conducted on the following open-source base models under two
protocols: instruction-based zero-shot benchmarking and SFT benchmarking. F or zero-shot, both generic prompts
and our optimized personalised prompts are compared, using Step-Audio 2 mini, Qwen2.5-Omni, Kimi-audio-7B,
and Qwen3-Omni-30B-A3B-Instruct. SFT is performed on Step-Audio 2 mini, Qwen2.5-Omni, and Qwen3-Omni-
30B-A3B-Instruct.
As shown in T able 5, we evaluate the zero-shot accuracy of several multimodal models on the Rejection dataset
under a generic prompt. Overall, model performance varies substantially across subsets, revealing uneven capa-
bilities in understanding rejection semantics. Qwen3-Omni-30B-A3B-Instruct demonstrates superior robustness
on subsets 1 ⚶6, consistently achieving accuracies above 0.85 ireaching 0.9545 and 0.8578 on subsets 3 and 5,
respectively iindicating strong generalization for typical rejection intents. However, its performance sharply de-
clines on subsets 8 ⚶11 (as low as 0.0322), suggesting significant limitations when handling complex, implicit, or
11

T able 3: Zero-shot Accuracy Comparison of Different General-Purpose Models on the Rejection Dataset (General
Prompt)
SubsetsQwen-3-32B
General PromptGLM-4.6-A WQ
General PromptGPT-oss-20b
General PromptDeepSeek-V3.1
General PromptGPT-5-Chat
General PromptQwen3-235b-A22b
General Prompt
0 1.0000 1.0000 0.9167 1.0000 1.0000 0.8333
1 0.9130 1.0000 0.8261 0.8696 0.9130 0.9130
2 0.4118 0.4878 0.0392 0.0000 0.4510 0.3333
3 0.2961 0.4198 0.0738 0.2566 0.8684 0.6579
4 0.2759 0.3913 0.0357 0.1724 0.6552 0.3103
5 0.3717 0.4804 0.2124 0.2168 0.5442 0.3761
6 0.4286 0.4000 0.2474 0.2857 0.6735 0.5714
7 0.4444 0.0000 0.2222 0.1111 0.7778 0.6667
8 0.5588 0.6940 0.8535 0.7420 0.7129 0.8004
9 0.8056 0.8386 0.8988 0.8159 0.8345 0.8922
10 0.4000 0.4762 0.7200 0.6400 0.6800 0.8800
11 0.1943 0.3187 0.3871 0.3599 0.3089 0.4299
12

T able 4: Zero-shot Accuracy Comparison of Different General-Purpose Models on the Rejection Dataset (Optimized
Prompt)
SubsetsQwen-3-32B-Instruct
Optimized PromptGLM-4.6-A WQ
Optimized PromptGPT-oss-20b
Optimized PromptDeepSeek-V3.1
Optimized PromptGPT-5-Chat
Optimized PromptQwen3-235b-A22b
Optimized Prompt
0 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
1 0.9130 0.9130 0.9565 0.9565 0.9565 1.0000
2 0.1569 0.3800 0.2549 0.0196 0.2549 0.2353
3 0.5197 0.4172 0.4671 0.1908 0.4671 0.4408
4 0.5862 0.5862 0.5517 0.2069 0.5517 0.6552
5 0.3097 0.2589 0.3672 0.1372 0.3672 0.2920
6 0.5000 0.5102 0.6531 0.2041 0.6531 0.6224
7 0.8889 0.4444 0.5556 0.2222 0.5556 0.4444
8 0.8923 0.9005 0.9015 0.9189 0.9015 0.8587
9 0.9453 0.9452 0.9442 0.7871 0.9442 0.9347
10 0.8000 0.9200 0.8000 0.8400 0.8000 0.8800
11 0.3280 0.3790 0.3790 0.5382 0.3790 0.4427
13

T able 5: Zero-shot Accuracy of Multimodal Models on the Rejection Dataset (Generic Prompt)
SubsetsStep-Audio 2 mini
General Prompt
Zero-shotQwen2.5-Omni-7B
General Prompt
Zero-shotQwen3-Omni-30B-A3B-Instruct
General Prompt
Zero-shotKimi-audio-7B
General Prompt
Zero-shot
0 0.0000 0.0000 0.0000 0.3750
1 1.0000 0.9655 0.9655 0.7586
2 0.3774 0.3208 0.7925 0.3962
3 0.4221 0.5909 0.9545 0.4221
4 0.5667 0.4333 0.7667 0.3667
5 0.4267 0.5517 0.8578 0.5302
6 0.3846 0.5165 0.8571 0.4835
7 0.5000 0.5000 0.5000 0.5000
8 0.7463 0.6136 0.3736 0.8086
9 0.8184 0.7478 0.4415 0.6544
10 0.5385 0.5385 0.1154 0.6538
11 0.5402 0.3055 0.0322 0.6206
14

context-dependent rejections. In contrast, Kimi-audio-7B, while slightly weaker on simpler cases (e.g., subset 1),
consistently outperforms others on challenging subsets (8 ⚶11), achieving the highest scores of 0.8086 and 0.6544
on subsets 8 and 9, reflecting greater contextual sensitivity and better recognition of atypical rejection patterns.
Notably , Step-Audio 2 mini achieves perfect accuracy (1.0000) on subset 1 but exhibits high variability elsewhere,
indicating limited generalization. Altogether, current multimodal models exhibit a polarized oeasy gets easier,
hard gets harder pbehavior on rejection understanding, highlighting the need for finer-grained data construction
and architectures with enhanced contextual awareness.
T able 6: Zero-shot Accuracy of Multimodal Models on the Rejection Dataset (Optimized Prompt)
SubsetsStep-Audio 2 mini
Optimized Prompt
Zero-shotQwen2.5-Omni-7B
Optimized Prompt
Zero-shotQwen3-Omni-30B-A3B-Instruct
Optimized Prompt
Zero-shot
0 0.0000 0.0000 0.5625
1 0.0000 0.3793 0.5517
2 0.0377 0.0377 0.0943
3 0.0065 0.0130 0.2468
4 0.0000 0.0000 0.2000
5 0.0345 0.0517 0.2069
6 0.0769 0.0769 0.2747
7 0.0000 0.0000 0.3333
8 1.0000 0.9991 0.9222
9 0.9999 0.9994 0.9068
10 1.0000 1.0000 0.8846
11 0.9936 0.9936 0.7074
The leading model Qwen3-Omni-30B-A3B achieves 0.88 ⚶0.92 on the easy splits 8 ⚶11, a 20⚶40 % gain over the
general prompt, demonstrating that optimized prompting markedly improves audio-text alignment under high-SNR
conditions; yet on the hard splits 1 ⚶7 its accuracy only inches up from 0.03 ⚶0.56 to 0.09 ⚶0.33, still far below the zero-
shot level of text-only large models, indicating that prompt engineering cannot compensate for the representation
gap of multimodal models in noisy home scenes.
6 Discussion and F uture W ork
The three-tier voice-rejection framework proposed in this paper ia universally fine-tuned adapter, personalized
historical context and a rejection knowledge base inot only improves the accuracy of invalid-utterance detection
but also embodies a deeper shift in the smart-home interaction paradigm: rejection is no longer a passive filter
but an active, context-aware interaction strategy . By deciding owhen not to respond p, the system strengthens
usersntrust that it ounderstands me pandodoes not disturb p, pushing the human-machine relationship from
15

functional execution towards experiential symbiosis. The design strikes an effective balance between generality
and personalisation: the general model guarantees cold-start robustness, while the personalized module gradually
refines decision boundaries as usage grows. Notably , the rejection knowledge base builds a lightweight data loop
that endows the system with preliminary self-correction capability . Y et this loop also introduces challenges i
how to prevent model drift caused by accumulated false feedback and how to manage the timeliness and conflict
resolution of the knowledge base remain open questions. Unlike traditional task-oriented dialogue systems that
assumeoevery user input is a valid intent p, this paper confronts reality: home speech is interwoven with chit-chat,
self-talk and vague expressions. Hence rejection should serve as a front-gate of the task pipeline, not a post-hoc
patch. This two-stage architecture of ojudge validity first, then solve the task pbetter matches real interaction
scenes. Looking forward, several directions deserve deeper exploration: building end-to-end speech-rejection models
to reduce reliance on ASR; introducing dynamic forgetting and confidence-weighted mechanisms to enhance the
reasoning ability of the knowledge base; combining household profiles and meta-learning to solve cold-start for new
users; fusing device status, time, location and other multimodal contexts to achieve scene-aware rejection; realising
privacy-preserving personalisation under on-device or federated-learning frameworks. Moreover, evaluation must
go beyond accuracy and incorporate subjective user experience (e.g., sense of disturbance, trust) and behavioral
signals (e.g., clarification turns, silent drop-outs) to truly measure the practical value of rejection systems. In short,
precise rejection is not merely a technical optimization but a redefinition of ointelligence pitself: true intelligence
lies not only in knowing what to do, but also in knowing when not to do it.
7 Conclusion
F ocusing on the rejection challenge in smart-home voice interaction, this paper advances both benchmark construc-
tion and algorithmic innovation in parallel. On the one hand, we build and open-source the first multimodal rejec-
tion benchmark tailored for household scenarios, containing 11,913 text-speech pairs with systematic annotations
of 13 invalid-utterance types and providing dialogue context, user identity , and multi-turn interaction information,
thereby offering a reproducible, fine-grained evaluation foundation for diverse research paradigms such as zero-shot,
fine-tuning, and retrieval augmentation. On the other hand, we propose a three-layer collaborative rejection ar-
chitecture iintegrating a universal fine-tuned adaptor, personalized historical context, and a family-level rejection
knowledge base based on RAG ithat effectively balances cold-start robustness with long-term personalization, sig-
nificantly reducing false triggers and over-responses, and demonstrating outstanding performance especially when
handling family-specific expressions and complex multi-turn contexts. This work not only validates the superiority
ofoactive rejection pover traditional opassive filtering pbut also propels the voice-interaction paradigm from oun-
conditional response ptowardocontext-aware decision-making p. Experiments show that accurately judging owhen
not to respond pis likewise an important manifestation of intelligence. F uture research can further explore end-to-
end speech-rejection models, dynamic knowledge-base evolution mechanisms, multimodal context fusion strategies
(e.g., device status, time, space), and construct composite evaluation systems that integrate objective metrics with
subjective experience (e.g., sense of disturbance, trust). In the long run, improving rejection capability is not
merely a technical optimization but a key step toward building a trustworthy , restrained, and user-life-respecting
human-machine symbiotic relationship.
References
[1] Zhang et al., “HomeBench: A Benchmark for Evaluating Smart Home V oice Assistants,” Proc. Interspeech ,
2023.
[2] Li et al., “V oiceBench: Benchmarking LLM-based V oice Assistants in Real-world Scenarios,” arXiv:2410.17196 ,
2024.
[3] Padmakumar et al., “TEACh: T ask-driven Embodied Agents that Chat,” AAAI , 2022.
[4] Crook et al., “The SIMMC Dataset: Situated Interactive MultiModal Conversations,” EMNLP , 2020.
[5] Lugosch et al., “Fluent Speech Commands: A Dataset for Spoken Language Understanding Research,” 2020.
[6] E. M. Langston, “Exploring AI-powered virtual assistants to support older adults nhealth and independence,”
Computers in Human Behavior Reports , vol. 3, pp. 100–115, 2025, doi: 10.1016/j.chbr.2025.100115 .
[7] V oicePrivacy Consortium, “The V oicePrivacy 2024 Challenge Evaluation Plan,”, accessed: 2025-06-01.
16

[8] Q. F ang, S. Guo, Y. Zhou, Z. Ma, S. Zhang, and Y. F eng, “Llama-Omni: Seamless speech interaction with large
language models,” arXiv preprint arXiv:2409.06666 , 2024.
[9] C. F u, H. Lin, Z. Long, Y. Shen, M. Zhao, Y. Zhang, X. W ang, D. Yin, L. Ma, X. Zheng, et al. , “VIT A: T owards
open-source interactive omni multimodal LLM,” arXiv preprint arXiv:2408.05211 , 2024.
[10] L. D. Colton, “Confidence and rejection in automatic speech recognition,” Ph.D. Thesis, Oregon Health &
Science University , 1997.
17
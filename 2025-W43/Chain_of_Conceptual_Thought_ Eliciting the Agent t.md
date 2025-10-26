# Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response

**Authors**: Qingqing Gu, Dan Wang, Yue Zhao, Xiaoyu Wang, Zhonglin Jiang, Yong Chen, Hongyan Li, Luo Ji

**Published**: 2025-10-21 09:08:21

**PDF URL**: [http://arxiv.org/pdf/2510.18434v1](http://arxiv.org/pdf/2510.18434v1)

## Abstract
Chain-of-Thought (CoT) is widely applied to improve the LLM capability in
math, coding and reasoning tasks. However, its performance is limited for
open-domain tasks since there are no clearly defined reasoning steps or logical
transitions. To mitigate such challenges, we propose another prompt-based
paradigm called Chain of Conceptual Thought (CoCT), where the LLM first tags a
concept, then generates the detailed content. The chain of concepts is allowed
within the utterance, encouraging the LLM's deep and strategic thinking. We
experiment with this paradigm in daily and emotional support conversations
where the concept is comprised of emotions, strategies and topics. Automatic,
human and model evaluations suggest that CoCT surpasses baselines such as
Self-Refine, ECoT, ToT, SoT and RAG, suggesting a potential effective
prompt-based paradigm of LLM for a wider scope of tasks.

## Full Text


<!-- PDF content starts -->

Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within
the Response
Qingqing Gu1, Dan Wang1, Yue Zhao1, Xiaoyu Wang1,2*,
Zhonglin Jiang1,Yong Chen1,Hongyan Li1,Luo Ji1†
1Geely AI Lab, Beijing, China2Beijing Institute of Technology, Beijing, China
Correspondence:Luo.Ji1@geely.com
Abstract
Chain-of-Thought (CoT) is widely applied to
improve the LLM capability in math, coding
and reasoning tasks. However, its performance
is limited for open-domain tasks since there are
no clearly defined reasoning steps or logical
transitions. To mitigate such challenges, we
propose another prompt-based paradigm called
Chain of Conceptual Thought (CoCT), where
the LLM first tags a concept, then generates
the detailed content. The chain of concepts
is allowed within the utterance, encouraging
the LLM’s deep and strategic thinking. We ex-
periment with this paradigm in daily and emo-
tional support conversations where the concept
is comprised of emotions, strategies and topics.
Automatic, human and model evaluations sug-
gest that CoCT surpasses baselines such as Self-
Refine, ECoT, ToT, SoT and RAG, suggesting
a potential effective prompt-based paradigm of
LLM for a wider scope of tasks.
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities in tasks of natural
language understanding, question-answering, ma-
chine translation, and multi-turn dialogue (Ouyang
et al., 2022; AI@Meta, 2024). To further elicit
their reasoning abilities, a Chain-of-Thought (CoT)
paradigm (Wei et al., 2022) is usually adopted,
which prompts the LLM to think ‘step by step’ be-
fore producing the final response. Several effective
paradigms are then proposed by incorporating CoT
with specific reasoning hierarchies such as Tree of
Thoughts (Yao et al., 2023a), Graph of Thoughts
(Besta et al., 2024), Structured Chain-of-Thought
(Li et al., 2025), and test-time scaling (Yao et al.,
2023b; Shinn et al., 2023; Madaan et al., 2023;
Ning et al., 2024).
*Work was done during the internship at Geely.
†Corresponding Author.
Llama3-8B Llama3-70B GPT-401234Human Scoreoriginal datasetDirect Self-Refine CoT CoCTFigure 1: Comparison between CoCT and other prompt-
based baselines on ESconv.
Although effective, such CoT-related methods
are often focused on math, coding, or other rea-
soning scenarios that have clear reasoning steps.
When dealing with open-domain tasks such as con-
versation, CoT-like methods (Wei et al., 2022; Yao
et al., 2023a; Besta et al., 2024; Li et al., 2025)
may encounter difficulties since conversations gen-
erally lack clear definitions of reasoning steps, step
transitions, and stepwise standardized answers. To
alleviate such issues, ECoT (Li et al., 2024; Kang
et al., 2024) is proposed which applies sequential
emotional CoT prompts on emotional support con-
versations (ESC). Cskills (Zhou et al., 2024), on
the other hand, try to improve ESC by allowing
the LLM to self-play the future dialogue. However,
such methods still do not provide solutions to open-
ended conversations and also introduce excessive
computational costs from iterative inference.
To bridge this gap, we aim to propose another
generalized prompt-based paradigm that enhances
LLM in broader, open-ended scopes of conversa-
tions. Our idea is motivated by the hierarchical
1arXiv:2510.18434v1  [cs.CL]  21 Oct 2025

Table 1: Lists of annotated emotions, strategies and topics from ESConv (Liu et al., 2021), DailyDialogue (Li
et al., 2017), EmpatheticDialogues (Rashkin et al., 2019) and Cskills (Zhou et al., 2024), respectively. Strategies
of ESconv are abbreviated (for full names, refer to (Liu et al., 2021)). Only the most frequent 10 emotions of
EmpatheticDialogues and the most frequent 8 topics of Cskills are listed.
Dataset Concepts Concept Categories
ESconvEmotion anger, anxiety, depression, disgust, fear, nervousness, sadness, shame
Strategy Que., Paraphrasing &Res., Ref., Self-Dis., Aff.& Rea., Pro., Inf., Others
DailyDialgoueEmotion anger, disgust, fear, happiness, sadness, surprise, no emotion
Strategy inform, question, directive, and commissive
EmpatheticDialogues Emotion surprised, grateful, proud, sentimental, annoyed, excited, sad, disgusted,. . .
Cskills Topic sports, travel, art, music, technology, food&drink, hobbies, entertainment,. . .
Supporter<Anger> It sounds like the argument with your mother really got to you. <Reﬂection of feelings> I’m really sorry to hear that you’re feeling this way. Arguments with those we love, like our parents, can indeed ﬁll us with anger. <crafts> Have you tried in any hobbies or crafts to distract yourself and calm your mind? (b) ECoT{available emotions}{available strategies}{available topics}At the beginning of each generated sentence, use tags such as <XX> to denote different concepts, <emotions> to denote emotional transitions, and use <strategies> for strategy determination.<topics> for topic changes.{available emotions}{available strategies}{available topics}Please ﬁrst identify the seeker's emotion, then determine the appropriate strategy, select a shifted topic, and ﬁnally provide the response. SupporterThe supporter's reply should acknowledge and reﬂect the seeker's emotions to show empathy and understanding.Here's the response: I can see that you’re upset. It must be tough when disagreements arise with your mom. Would you like to share about what happened?(c) CoCTI am just a little angry. My mother and I had another blow-up.Seeker
SupporterIs your bit of anger sparked from the interaction with your mother?(a) Direct PromptPrompt
Figure 2: The paradigm of CoCT. The red bold text represents emotions, the blue bold text represents strategies,
and the green bold text represents topics of conversation transitions.
nature of human thinking, where a person gener-
ally first formulates a list (or chain) of strategic
concepts, each of which may then guide detailed
thoughts (team et al., 2024; Ning et al., 2024).
This chain of concepts and corresponding concept-
guided contents then formulates the ultimate hu-
man thought. For example, a customer may delib-
erately first exaggerate the product’s disadvantages,
then make a solid decision during a bargain. A
psychological counselor may first express an em-
pathetic attitude, then provide a detailed, practical
solution. In daily life, we summarize these con-
cepts into the primary types, including
•Emotion: the agent’s emotional or sentiment
states, such as joy, sadness, or anger.
•Strategy: the agent’s conversational strategy,
such as asking for more details or summariz-
ing the contents.
•Topic: the agent might choose to deepen the
topic or shift the user’s intention to a new topicfor better effects.
Assisted by previous annotated textual datasets
(Liu et al., 2021; Li et al., 2017; Rashkin et al.,
2019; Zhou et al., 2024) exemplified in Table 1,
we argue that LLM can also benefit from this way
of thinking. On the other hand, although there
are also studies to facilitate LLM to complete the
response based on pre-generated skeletons (Ning
et al., 2024), our paradigm allows the LLM to
deeply thinkwithina single utterance, instead of
acrossdifferent utterances.
In this paper, we propose a straightforward but
effective paradigm calledChainofConceptual
Thought (CoCT), which introduces aconceptual
chainedvariant of the classical CoT, as the solution
to open-domain or Open-ended conversations. For
each response utterance, we prompt the LLM to
first propose a concept (either emotion, strategy,
or topic, surrounded by special tokens)1, then gen-
1either from a pre-defined list, or dynamically generated
2

erate the remained content grounded by that con-
cept. Multiple conceptual transitions are allowed
within one response. To better validate this idea, we
then test CoCT on concept-annotated datasets, and
compare the quantitative metrics to varied prompt-
based baselines. Result validates the superiority of
CoCT, with a quick snapshot by Figure 1. Figure 2
shows the comparison between CoCT and previous
paradigms. Our key contributions include:
•We propose CoCT, a prompt-based paradigm
which allows the LLM to think conceptually
and strategically in open-ended conversation.
•We test CoCT on tasks of open-ended conver-
sation and emotional support conversations,
surpassing other prompt-based baselines.
•CoCT is robust to out-of-domain concepts and
queries, and also generalizes to finetuning,
stronger bases, or with concepts on-the-fly.
2 Method
This section introduces our method details, includ-
ing the prompt paradigm, the possible concepts,
and resulting inner-utterance concept transitions.
2.1 Chain-of-Conceptual-Thought Prompt
According to the studied scenario, we assume a
pre-defined list of concepts {c}is available based
on prior knowledge. These concepts are then be
applied to steer the LLM generation
output←LLM({c 1, c1,···, c n},I, input)(1)
in which Iis the instruction template of CoCT, c
is a concept andnis the total number of concepts.
Below is the template of our CoCT prompt:
CoCT Prompt Format
‘{possible concepts}’,
At the beginning of each generated re-
sponse, use tags such as <XX> to denote
different concepts, then follow the sentence
content.
Use chain of concepts to denote the con-
cept transitions.
To summarize, the CoCT prompt Istarts from
a detailed definitions of the conceptual list {c},
which helps the LLM extract usable concepts from
potentially unlimited space. After that, for each
Query 1 “I want to learn a new skill.”
!
Emotion 1 :Strategy 1 :“Which places are suitable?”
"
#
$
Topic 1 :Response 1:Query 2EmotionA StrategyA TopicA
!
$
EmotionB StrategyB TopicBEmotionC StrategyC TopicC
#
"
%
&
Emotion 2 :Strategy 2 :Topic 2 :Response 2:Query 1 “I want to learn a new skill.”Response 1:
(b) CoCT(a) Nominal Figure 3: Comparative visualizations between outer-
utterance transitions (Left) and inner-utterance transi-
tions (Right). The dash arrows indicate the conceptual
transitions between emotions, strategies or topics.
generated sentence, we ask the LLM to first de-
code the special token of concept, then decode the
detailed sentence content.
2.2 Formulation of Concepts
As discussed in Section 1, in this paper we study
three main concept types: emotion, strategy, and
topic. While we find CoCT can also work with
concepts generated on-the-fly, better performance
can be obtained with predefined concept lists.
As indicated by Table 1, we get the emotion and
strategy lists directly from the annotations of ES-
conv and DailyDialogue. For topics, we refer to
the topic guidance study in Cskills, which provides
about 8 topics after frequency filtering:sports,
travel, art, music, technology, food and drink, hob-
bies and crafts, entertainmentandanimal.
2.3 CoCT Elicits the Inner-utterance
Transitions
Human talk sometimes implies the transition of
concepts within one single utterance; however,
nominal LLM-based conversation usually fails to
do so. Instead, one can observe that each of their
responses contains a single emotion or topic, and
such concepts’ transitions happen on the outer-
utterance (Figure 3 (Left)). On the other hand,
CoCT naturally encourages the inner-utterance con-
cept transitions (Figure 3 (Right)), which is part of
our main purpose:
ci→cj, i, j∈ {1,···, n}(2)
3

3 Experiment
In this section, we first briefly introduce the ex-
perimental setting, including the implementation,
baseline and metric details. Then we exhibit the
effect of CoCT on Open-ended conversations. To
provide more quantitative results, we then conduct
more experiments on emotional support conversa-
tions. Finally, we provide some insightful analysis.
3.1 Setting
Implementation.The experiment is conducted
on vLLM, with 8 80GB PCIe GPUs. The context
window is restricted to 4096. To validate the ef-
fectiveness of CoCT on different model scales, we
experiment with the performances on Llama3-8B-
instruct (AI@Meta, 2024), Llama3-70B-instruct
(AI@Meta, 2024) and GPT4o, respectively.
Datasets and tasks.In this paper, we conduct
CoCT on both in-domain (ID) and out-of-domain
(OOD) tasks. In ID tasks, the datasets have an-
notations of emotions, strategies, or topics, which
are used as the possible concepts {c}during the
test. For OOD tasks, the datasets do not have such
concept annotations. Therefore, we use {c}de-
fined in ID tasks, and test CoCT’s generalization
capability of these concepts to this new dataset.
Since it is more common to assume there are no
conceptual annotations, the OOD tasks can reflect
the capability of CoCT in most of scenarios. The
conversational datasets we use can be classified
into two main categories:
(1) Open-ended conversations: we use the anno-
tated lists of emotions and strategies fromDaily-
Dialogue(Li et al., 2017), and the list of topics
fromCskills(Zhou et al., 2024). Together, we have
7 emotions, 4 strategies around 20 topics. Corre-
spondingly, we conduct CoCT on the test sets of
DailyDialogueandCskillsas the ID test. We then
apply CoCT on another open-topic datasetMulti-
WoZ(Budzianowski et al., 2018) as the OOD test,
which does not have concept annotations.
(2) Emotional support conversations: we then focus
on this specific scenario for more in-depth analy-
sis. We employ 11 emotions and 8 strategies from
ESConv(Liu et al., 2021), experiment CoCT on
ESConvfor ID test and onEmpatheticDialogues
(Rashkin et al., 2019) for OOD test.
Appendix A provides more dataset details.
Metrics.We evaluate the results in different man-
ners, such as automatic metrics, LLM simulations,LLM-as-a-Judge, and human annotations:
•Automatic Metrics: Include similarity-based
metrics such as BLEU-2 (B-2) (Papineni et al.,
2002), Rouge-L (R-L) (Lin, 2004) and CIDEr
(CDr) (Vedantam et al., 2015); and diversity-
based metric Distinct-2 (D-2).
•LLM-Based Simulations: The Cskills bench-
mark (Zhou et al., 2024) employs GPT4o to
simulate the conversation, until there is no
significant intention to communicate. Aver-
aged utterance length and dialogue turns are
calculated and played as qualitative metrics.
•LLM-as-a-Judge: We used GPT-4o to com-
pare different responses and provide the win-
tie-lose suggestion. The evaluation prompt is
from (Madani et al., 2024).
•Human Scoring: Similar to (Kang et al.,
2024), the scoring dimensions includeAccep-
tance,Effectiveness,Sensitivity,Fluency, and
Emotion, and theuser’s Satisfaction.
We provide more evaluation details in Appendix
B, such as computation formula, annotation princi-
ple and evaluation prompts.
Baselines.As fair comparisons, we implement
several prompt-based baselines, with the same base
model and context. Such baselines can be classi-
fied into several categories: (1)Directinference
the LLM,Direct-Refinethe result within the same
response, and Self-Refine (Madaan et al., 2023) the
result by inference the second time. (2) Methods
based on chained or other structured data format,
includingECoT(Li et al., 2024),SoT(Ning et al.,
2024),ToT(Yao et al., 2023a), andPlan-and-Solve
(Wang et al., 2023). (3) Retrieval augmented gen-
eration (RAG) (Fan et al., 2024) which retrieves
the suitable concepts and descriptions and prompts
the LLM to respond. In this work, we employ E5-
Large (Wang et al., 2024) as the retriever. (4)CSIM
(Zhou et al., 2024): a self-chat method which pro-
duces the response based on its implicit simulation
of future dialogue.
More details can be found in Appendix C.
3.2 Result of Open-ended Conversation
We first test CoCT on open-ended conversations,
including DailyDialogue and MultiWoZ.
4

Table 2: Results on automatic metrics on open-ended conversations (DailyDialogue and MultiWoZ) and ESC
(ESConv and EmpatheticDialogues). The best results on each LLM areboldedand the second best are underlined .
Method↓DailyDialogue (ID) MultiWoZ (OOD) ESConv (ID) EmpatheticDialogues (OOD)
B-2 R-L CDr B-2 R-L D-2 CDr B-2 R-L CDr B-2 R-L D-2 CDr
LLaMA3-8B-Instruct
+ Direct 3.35 10.33 1.99 4.94 13.18 16.09 4.12 3.47 10.64 1.14 3.09 9.91 25.231.60
+ Direct-Refine 2.56 8.70 1.75 4.02 11.90 17.92 1.57 3.10 6.13 1.30 2.56 9.12 22.32 0.42
+ Self-Refine 2.40 7.75 1.27 4.47 12.61 19.13 3.07 3.34 9.71 1.79 3.08 9.91 25.20 1.56
+ ECoT 1.78 6.00 2.43 4.22 11.8926.282.36 3.16 10.50 1.50 2.91 9.79 32.65 1.37
+ Plan-and-Solve 2.60 7.76 1.92 4.32 10.74 18.99 4.87 2.81 8.27 1.59 2.69 6.93 24.02 0.99
+ ToT 2.52 8.84 0.61 3.06 10.05 15.98 0.23 2.65 9.81 0.60 2.31 9.05 29.09 0.12
+ SoT 2.53 7.97 1.76 3.58 9.77 17.09 3.55 3.07 8.76 1.70 1.79 5.6648.590.19
+ CoCT (ours) 3.62 11.64 3.84 6.94 17.9519.79 9.73 4.75 13.31 4.87 3.29 10.8426.27 1.08
LLaMA3-70B-Instruct
+ Direct 3.50 11.20 2.46 5.19 14.38 14.38 1.37 3.06 10.16 0.8 2.60 8.90 26.71 0.44
+ Direct-Refine 4.16 12.84 6.04 4.63 13.32 15.65 1.40 2.54 9.40 0.27 2.53 9.07 25.28 0.30
+ Self-Refine 3.40 10.83 2.97 4.82 13.47 17.09 1.51 2.97 10.12 1.07 2.81 9.74 27.06 0.48
+ ECoT 2.09 7.62 1.23 4.82 13.47 17.09 1.51 1.85 9.67 1.03 2.60 9.0634.021.14
+ Plan-and-Solve 2.13 6.32 0.80 3.96 10.41 13.49 3.09 2.01 8.68 1.36 2.14 6.25 37.16 0.50
+ ToT 2.68 9.43 0.69 4.08 13.91 16.62 0.40 2.39 8.96 0.16 2.16 8.24 27.14 0.13
+ SoT 2.72 8.55 1.72 3.92 11.0120.803.96 3.13 9.49 1.16 2.12 6.83 45.98 0.35
+ RAG 2.78 9.67 1.67 - - - - 3.44 10.95 1.65 2.10 8.18 28.92 0.13
+ CoCT (ours) 3.87 11.94 3.81 6.45 17.0017.34 4.78 4.30 12.71 2.07 3.48 11.2029.34 1.45
GPT4o
+ Direct 4.25 12.80 6.09 - - - - 3.48 13.21 3.83 - - - -
+ Direct-Refine 4.03 12.22 4.49 - - - - 4.19 12.61 2.65 - - - -
+ Self-Refine 3.19 9.84 3.19 - - - - 5.4012.70 2.82 - - - -
+ ECoT 3.48 11.10 3.46 - - - - 3.64 11.64 1.95 - - - -
+ RAG 4.15 9.75 7.14 - - - - 3.68 11.05 3.67 - - - -
+ CoCT (ours) 4.60 13.25 8.73 - - - - 4.62 13.77 4.79 - - - -
Performance on in-domain concepts.Table 2
shows the automatic metrics of DailyDialogue,
where CoCT performs the best with the basis of
LLaMA3-8B-Instruct and GPT4o, indicating it can
generate responses in a similar style to the original
annotated response. CoCT performs the second-
best on LLaMA3-70B-Instruct (worse than Direct-
Refine); which we suppose this may be due to in-
sufficient exploration when we apply the prompt
tested on 8B directly to 70B.
Generalize with out-of-domain concepts or
queries.Table 2 also exhibits the result of Multi-
WOZ on LLaMA3-8B-Instuct and LLaMA3-70B-
Instuct. CoCT still performs the best on B-2, R-L
and CDr, indicating its strong generalization capa-
bilities on different topics and domains. On the
other hand, CoCT is the second-best on D-2, a
diversity-based metric, which is reasonable since
the pre-defined concepts may constrain the method-
ology’s scope2.
2We will discuss the improvement method in Section 3.4.Table 3: Results on the Cskills. We use the ‘concept
guidance’ subset of the original benchmark.
Method→ Direct CSIM RAGCoCT(ours)
AvgLen 46.41 47.73 49.1367.04
Rounds 3.53 4.26 4.364.62
Engagement in Open-ended conversations.Fi-
nally, Table 3 shows the Cskill results. Our CoCT
surpasses Direct, CSIM, and RAG on both aver-
aged lengths and rounds of conversations. This
result indicates that CoCT possesses comprehen-
sive communication skills, which could benefit the
Open-ended conversation engagement.
Cases on more situationsTo test the generaliza-
tion capability of CoCT, we also attempt arbitrary
user queries; we then add two more arbitrary tags,
<Humor> and <Interest>, expecting they can be
adopted and trigger more interesting responses; we
also test different combinations of concept types:
only with strategy, with emotion, or with both emo-
tion and strategy.
Table 4 showcases CoCT’s performance on these
5

Table 4: Open cases generated by CoCT.w/ strategyuser I’ve started learning how to cook.
CoCT<Affirmation and Reassurance>That’s fantastic! Learning to cook is
such a valuable skill and can be really rewarding.
<Question>What type of dishes are you interested in trying out?
user I tried making pizza last week. It was a bit of a fail the first time, but it still tasted okay!
CoCT<Reflection of feelings>It sounds like you were a bit disappointed with the first
attempt, but also glad that it still turned out tasty.
<Humor>Hey, even a "fail" pizza is still pizza, right? That’s a win in my book!
<Question>What part of making the pizza did you find the most challenging?w/ emotionuser I’ve started learning how to cook.
CoCT<Excitement>That’s fantastic! Learning to cook is such a rewarding skill.
What dish are you most excited to try making?
user I tried making pizza last week. It was a bit of a fail the first time, but it still tasted okay!
CoCT<Nervousness>First attempts can be tricky, but it’s great that it still tasted okay!
<Encouragement>Practice makes perfect, and every time you cook, you’ll get better.
What part of making the pizza did you find most challenging?w/ emotion+strategyuser I’ve started learning how to cook.
CoCT<Interest> That’s fantastic! Cooking can be a rewarding and enjoyable skill to develop.
<Questioning>What kinds of dishes are you learning to make?
user I tried making pizza last week. It was a bit of a fail the first time, but it still tasted okay!
CoCT<Affirmation>It’s great that you tried making pizza!
<Reassurance>Everyone has a few mishaps when they’re learning something new.
<Humor>Besides, an "okay" tasting pizza is still pizza, right?
<Questioning>What part of the process do you think was the most challenging?
arbitrary scenarios. It can be observed that CoCT
still performs well on these out-of-domain queries
and concepts, generating reasonable concept transi-
tions. Especially, <Humor> is utilized in the first
case and <Interest> is utilized in the third case,
generating more funny responses and steering the
conversation to a higher communication level.
3.3 Result of Emotional Support Conversation
Here we provide further results on a subset of sce-
narios, the emotional support conversation (ESC).
Automatic Metrics.The last two columns of Ta-
ble 2 illustrate the ID results on ESconv and the
OOD results on EmpatheticDialogues. Similar to
the open-ended case, CoCT is still the best on B-2,
R-L and CDr, and the second best on D-2. This
observation indicates CoCT holds stable and robust
performance for a specific domain.
Human Scores.Table 5 shows the human an-
notation results on ESC. CoCT has higher scores
than baselines on different model bases. Figure 1
in Section 1 also provides a snapshot of the Sat-
isfaction scores. It is also worth mentioning that
although the original dataset is originally annotated
as expert demonstrations, CoCT based on Llama3-
70B-Instruct starts to surpass the original dataset,while other baselines can not. For results based on
GPT4o, the human scores are all higher than the
original dataset, due to the stronger model basis.
Our CoCT still remains the best.
Ablation Study.We conduct several ablations,
includingwo/ strategyandwo/ emotionin the con-
cept list. We compare their performances to the
formal CoCT on human and automatic evaluations,
as well as the pairwise evaluation (versus origi-
nal dataset) by GPT4o. Table 6 shows that CoCT
still outperforms these ablations, indicating both
strategy and emotion are helpful concept types.
3.4 Discussion
Conceptual transition within utterance.Com-
pared to previous approaches, CoCT allows for the
transitions between different concepts (e.g., recog-
nition of emotions and strategies) within a single
utterance, which is aligned with the human-like
thinking behavior, and closer to the original anno-
tated response of ESconv. To further illustrate this
behavior, we exhibit the distribution of the inner-
utterance conceptual transitions in Figure 4. For a
specific grid (i, j) , where idenotes the row index
andjdenotes the column index, its color level indi-
cates the relative occurrence of transitionc i→cj.
6

Table 5: Human evaluations on ESC. The best results of each LLM areboldedand the second best are underlined .
Size MethodHuman Annotation
Fluency Emotion Acceptance Effectiveness Sensitivity Alignment Satisfaction
- orignal dataset 3.51 3.61 3.40 3.10 3.50 3.20 3.30
8BLLaMA3 (8B) 2.95 3.00 2.60 2.40 2.70 2.70 2.60
+ Direct-Refine 3.09 3.09 2.73 2.91 2.91 2.82 2.84
+ Self-Refine 3.103.15 2.80 2.70 2.90 2.80 2.80
+ ECoT 3.08 3.08 2.83 2.67 3.00 2.83 2.83
+CoCT(ours) 3.10 3.25 3.00 3.20 3.10 3.00 3.08
70BLLaMA3 (70B) 3.05 3.30 2.70 3.00 3.20 3.10 3.00
+ Direct-Refine 3.403.50 3.00 3.10 3.10 2.90 3.03
+ Self-Refine 3.403.35 2.90 3.10 3.003.203.05
+ ECoT 3.25 3.30 2.80 3.00 3.103.203.03
+CoCT(ours) 3.32 3.77 3.36 3.55 3.553.18 3.41
largerGPT-4o 3.56 3.673.673.56 3.56 3.22 3.50
+ Direct-Refine 3.80 3.70 3.60 3.503.703.60 3.60
+ Self-Refine 3.80 3.75 3.50 3.70 3.60 3.60 3.60
+ ECoT 3.75 3.70 3.50 3.50 3.60 3.50 3.53
+CoCT(ours) 3.85 4.003.60 4.10 3.70 3.70 3.82
Table 6: Ablation study of CoCT on Llama3-8B-
Instruct, with human, automatic and win-tie-lose rates
(%) of GPT-4o evaluation results. Experiments are con-
ducted on ESconv. ‘Satis.’ abbreviates Satisfaction.
Method↓human automatic VS dataset
Satis. B-2 R-L win↑lose↓
wo/ strategy 2.80 4.25 12.67 51.90 37.40
wo/ emotion 3.00 4.03 12.60 60.90 33.70
free concept 2.90 3.21 11.05 66.86 16.75
CoCT 3.08 4.75 13.31 71.71 26.80
Figure 4 (left) exhibits the strategy transitions
generated by CoCT on ESConv. According to (Liu
et al., 2021), the strategies can be related to three
supporting stages (Exploration, Comforting and
Action), therefore, we mark each strategy with the
corresponding stage I, II and III and order the strate-
gies accordingly. As a result, a more reasonable
transition may happen from a left strategy to a right
one,i.e., the upper-triangle part of the matrix. Such
a pattern can be observed from Figure 4 (left), in-
dicating that the strategy determination of CoCT
is aligned with the ‘Exploration, Comforting and
Action’ paradigm.
Figure 4 (right) shows the emotion transitions
generated by CoCT on DailyDialogue. From
the distribution, we can observe that more fre-
quent emotion transitions of CoCT happen on Dis-
gust&Anger →Happiness, Anger →Disgust, and
Fear&Sadness →Surprise. On the contrary, too ex-Table 7:The finetuning result of CoCT on LLaMA3-
70B-Instruct, from responses of GPT4o.
Method↓ESConv
B-2 R-L D-2 CDr
CoCT 4.30 12.71 18.16 2.07
CoCTw/ finetuning 4.75 13.49 42.54 8.68
treme emotional transition (Happiness →Sadness)
has a low frequency. CoCT mimics the human emo-
tion curve when expressing opinions, providing a
more fine-grained emotional interaction.
CoCT without predefined concepts.Previous
results (especially OOD) suggest that the prede-
fined concepts may hinder CoCT from exploring
broader scopes. In Table 6, we also try to allow
the LLM to generate concepts on-the-fly (free con-
cept). Surprisingly, althoughfree conceptendures
some performance degradation, it still performs
well and surpasses the other baselines. This obser-
vation suggests the potential of CoCT as a more
generalized methodology,e.g., remove the concept
list in the prompt, but pretrain the LLM with some
conceptual knowledge.
CoCT with finetuning.Although CoCT is pri-
marily proposed as a prompt-based paradigm, it
is also possible to collect the CoCT result from a
stronger LLM, then finetune it on a smaller LLM.
This methodology is also known as distillation from
7

Que.(I)
Res. & Par.(I)
Ref.(II)
Self-Dis.(II)
Aff. & Rea.(II)
Pro.(III)
Inf.(III)Que.(I)
Res. & Par.(I)
Ref.(II)
Self-Dis.(II)
Aff. & Rea.(II)
Pro.(III)
Inf.(III)
0.00.20.40.60.81.0
Happiness
Surprise
Disgust
Anger
Fear
SadnessHappiness
Surprise
Disgust
Anger
Fear
Sadness
0.00.10.20.30.40.50.6
Figure 4: Distribution of conceptual transitions within the utterance. Left: the strategy transition on ESConv. Right:
the emotion transition on DailyDialogue.
Table 8:Performance comparison of different special
tokens on ESConv.
Method↓ESConv
B-2 R-L D-2 CDr
LLaMA3 (8B)
w/ special token: <> 4.75 13.31 18.54 4.87
w/ special token: ˆ 4.38 13.16 18.34 3.79
w/ special token: # 4.8614.12 18.65 4.99
w/ special token: @ 4.54 13.54 18.43 4.07
w/ special token: [] 4.82 14.1318.78 5.09
w/ special token: & 4.72 13.7119.044.62
LLaMA3 (70B)
w/ special token: <> 4.30 12.71 18.16 2.07
w/ special token: ˆ 3.96 12.14 16.60 1.32
w/ special token: # 4.17 12.68 17.04 1.88
w/ special token: @ 4.11 12.54 16.68 1.74
w/ special token: [] 4.01 12.45 16.18 1.48
w/ special token: & 3.92 12.06 15.94 1.25
the teacher LLM. In Table 7, we compare the 70B
result of CoCT with finetuning to the original re-
sult, indicating that CoCT is friendly to finetuning,
and its performance can be further enhanced by
distillation from a teacher,i.e., GPT4o.
Choice of special tokens.The prompt of CoCT
also requires an arbitrary choice of special token.
In Section , we take ‘<>’ as the example, while here
we compare its results to other special tokens (e.g.,
#, @,ˆ, [], &). Table 8 exhibits this comparison
result on 8B and 79B, showing that the current
choice is optimal on 70B, while different special
tokens may perform similarly on 8B. In practice,
one may conduct similar tests to determine the
special token before the formal experiment.4 Related Work
Large language models can generalize to differ-
ent tasks given an informative and instructional
prompt. Chain-of-thought (CoT) (Wei et al., 2022)
is a famous study that prompts the LLM by the
paraphrase ‘Let’s think step by step’ and obtains
substantial performance improvement on metathet-
ical and reasoning tasks. Further studies are pro-
posed to enhance the performance by using dif-
ferent prompt paradigms. For example, Chain-of-
Hindsight (CoH) (Liu et al., 2024) allows the LLM
to reflect from human feedback and have a bet-
ter alignment with human preference. Self-Refine
(Madaan et al., 2023) let the LLM generate feed-
back from its initial response, then refine the re-
sponse again based on this feedback. SoT (Ning
et al., 2024) first guides LLMs to generate the skele-
ton of the answer, and then conducts parallel API
calls or batched decoding to complete the contents.
In contrast, here we mainly focus on the open-
ended conversation scenario and argue that LLM
can also benefit from a chain of concepts, instead of
reasoning steps. Such concepts may include emo-
tions, strategies and topics, and can transit within
the utterance. LLM is prompted to first generate
the sketchpad of concepts, followed by the formal
response.
5 Conclusion
In this work, we propose a novel prompt paradigm
called CoCT which can be applied on open-domain
tasks. We use the conceptual modeling theory and
let the LLM first generate the tag of the concept,
then the detailed content of the corresponding con-
8

cept. The chain of concept facilitates the LLM to
have a strategic way of thinking and highlights its
decision transitions. Experiments on open-ended
and daily conversations show that CoCT produces
better responses than the baselines, and the case
studies further verify its generalization capability.
9

6 Limitation
CoCT is orthogonal to few-shot demonstrations,
finetuning, self-reflection, and usage of structured
or external knowledge. Combinations of these
methods might further benefit.
Currently, CoCT also needs a manually provided
list of concept candidates, which might constrain
its application. Future work may consider invoking
the LLM’s hierarchical thinking mechanism, which
automatically first formulates the next key concept,
then deduces the detailed response.
Although typical cases and quantitative metrics
primarily indicate the effectiveness of CoCT, due
to the capability limitation of the foundation model,
CoCT still sometimes fails to understand the con-
cept list, or conduct the semantically wrong concep-
tual transition. We also include a bad case analysis
in the Appendix.
7 Ethical Considerations
CoCT can exhibit inconvenient, aggressive behav-
iors given specific concepts. It is suggested that the
concept list needs to be carefully checked before
the formal deployment. If concepts are generated
on-the-fly, the finetuning may be needed to con-
strain the behavior within safety.
References
AI@Meta. 2024. Llama 3 model card.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Ger-
stenberger, Michał Podstawski, Lukas Gianinazzi,
Joanna Gajda, Tomasz Lehmann, Hubert Niewiadom-
ski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph
of thoughts: solving elaborate problems with large
language models. InProceedings of the Thirty-
Eighth AAAI Conference on Artificial Intelligence
and Thirty-Sixth Conference on Innovative Applica-
tions of Artificial Intelligence and Fourteenth Sym-
posium on Educational Advances in Artificial Intelli-
gence, AAAI’24/IAAI’24/EAAI’24. AAAI Press.
Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang
Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ra-
madan, and Milica Gaši ´c. 2018. MultiWOZ - a large-
scale multi-domain Wizard-of-Oz dataset for task-
oriented dialogue modelling. InProceedings of the
2018 Conference on Empirical Methods in Natural
Language Processing, pages 5016–5026, Brussels,
Belgium. Association for Computational Linguistics.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD Conference onKnowledge Discovery and Data Mining, KDD ’24,
page 6491–6501, New York, NY , USA. Association
for Computing Machinery.
Dongjin Kang, Sunghwan Kim, Taeyoon Kwon, Se-
ungjun Moon, Hyunsouk Cho, Youngjae Yu, Dongha
Lee, and Jinyoung Yeo. 2024. Can large language
models be good emotional supporter? mitigating
preference bias on emotional support conversation.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 15232–15261, Bangkok, Thai-
land. Association for Computational Linguistics.
Jia Li, Ge Li, Yongmin Li, and Zhi Jin. 2025. Struc-
tured chain-of-thought prompting for code genera-
tion.ACM Trans. Softw. Eng. Methodol., 34(2).
Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang
Cao, and Shuzi Niu. 2017. DailyDialog: A manually
labelled multi-turn dialogue dataset. InProceedings
of the Eighth International Joint Conference on Nat-
ural Language Processing (Volume 1: Long Papers),
pages 986–995, Taipei, Taiwan. Asian Federation of
Natural Language Processing.
Zaijing Li, Gongwei Chen, Rui Shao, Yuquan Xie,
Dongmei Jiang, and Liqiang Nie. 2024. Enhanc-
ing emotional generation capability of large language
models via emotional chain-of-thought.Preprint,
arXiv:2401.06836.
Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81.
Hao Liu, Carmelo Sferrazza, and Pieter Abbeel. 2024.
Chain of hindsight aligns language models with feed-
back. InThe Twelfth International Conference on
Learning Representations.
Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand
Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie
Huang. 2021. Towards emotional support dialog
systems. InProceedings of the 59th Annual Meet-
ing of the Association for Computational Linguistics
and the 11th International Joint Conference on Natu-
ral Language Processing (Volume 1: Long Papers),
pages 3469–3483.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Sean Welleck, Bodhisattwa Prasad Majumder,
Shashank Gupta, Amir Yazdanbakhsh, and Peter
Clark. 2023. Self-refine: Iterative refinement with
self-feedback.ArXiv, abs/2303.17651.
Navid Madani, Sougata Saha, and Rohini Srihari. 2024.
Steering conversational large language models for
long emotional support conversations.Preprint,
arXiv:2402.10453.
Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang,
Huazhong Yang, and Yu Wang. 2024. Skeleton-
of-thought: Prompting LLMs for efficient parallel
10

generation. InThe Twelfth International Conference
on Learning Representations.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Car-
roll L. Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback.Preprint, arXiv:2203.02155.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. InProceedings of the
40th annual meeting of the Association for Computa-
tional Linguistics, pages 311–318.
Hannah Rashkin, Eric Michael Smith, Margaret Li, and
Y-Lan Boureau. 2019. Towards empathetic open-
domain conversation models: A new benchmark and
dataset. InProceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics,
pages 5370–5381, Florence, Italy. Association for
Computational Linguistics.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: language agents with verbal reinforcement
learning. InProceedings of the 37th International
Conference on Neural Information Processing Sys-
tems, NIPS ’23, Red Hook, NY , USA. Curran Asso-
ciates Inc.
LCM team, Loïc Barrault, Paul-Ambroise Duquenne,
Maha Elbayad, Artyom Kozhevnikov, Belen Alas-
truey, Pierre Andrews, Mariano Coria, Guillaume
Couairon, Marta R. Costa-jussà, David Dale, Hady
Elsahar, Kevin Heffernan, João Maria Janeiro,
Tuan Tran, Christophe Ropers, Eduardo Sánchez,
Robin San Roman, Alexandre Mourachko, Safiyyah
Saleem, and Holger Schwenk. 2024. Large concept
models: Language modeling in a sentence represen-
tation space.Preprint, arXiv:2412.08821.
Ramakrishna Vedantam, C Lawrence Zitnick, and Devi
Parikh. 2015. Cider: Consensus-based image de-
scription evaluation. InProceedings of the IEEE
conference on computer vision and pattern recogni-
tion, pages 4566–4575.
Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu,
Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim.
2023. Plan-and-solve prompting: Improving zero-
shot chain-of-thought reasoning by large language
models.Preprint, arXiv:2305.04091.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Improv-
ing text embeddings with large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models.Advances in Neural
Information Processing Systems, 35:24824–24837.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Thomas L. Griffiths, Yuan Cao, and Karthik R
Narasimhan. 2023a. Tree of thoughts: Deliberate
problem solving with large language models. In
Thirty-seventh Conference on Neural Information
Processing Systems.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023b.
React: Synergizing reasoning and acting in language
models. Publisher Copyright: © 2023 11th Inter-
national Conference on Learning Representations,
ICLR 2023. All rights reserved.; 11th International
Conference on Learning Representations, ICLR 2023
; Conference date: 01-05-2023 Through 05-05-2023.
Junkai Zhou, Liang Pang, Huawei Shen, and Xueqi
Cheng. 2024. Think before you speak: Cultivating
communication skills of large language models via
inner monologue. InFindings of the Association
for Computational Linguistics: NAACL 2024, pages
3925–3951.
A More Dataset Details
Detailed introductions.We introduce our exper-
imental datasets in more detail;
•TheDailyDialoguedataset (Li et al., 2017)
is a widely used benchmark for Open-ended
conversation. It contains 13,118 multi-turn di-
alogues across diverse daily topics (e.g., fam-
ily, work, hobbies) with manually annotated
emotions and dialogue strategies (named ‘act’
in the original dataset). It is annotated with 7
emotions and 4 strategies.
•TheMultiWoZdataset (Budzianowski et al.,
2018) is a task-oriented dialogue corpus fo-
cused on service-oriented interactions (e.g.,
hotel booking, restaurant reservations, trans-
portation inquiries), with no concept annota-
tions.
•TheESConvdataset (Liu et al., 2021) is a
specialized corpus for emotional support re-
search. It contains 1,000+ multi-turn dia-
logues where users articulate personal strug-
gles (e.g., workplace stress, interpersonal
conflicts, self-esteem issues), and support
providers respond with empathy, validation,
and actionable guidance. It is annotated with
11 emotions and 8 strategies.
•TheEmpatheticDialoguesdataset (Rashkin
et al., 2019) contains conversations centered
11

on recognizing and responding to emotions
(e.g., joy, sadness, frustration). It has annota-
tions of user emotions but no strategies.
Dataset statistics.Table 9 summarizes the statis-
tics of ESConv, DailyDialogue, and EmpatheticDi-
alogues. Table 10 provides definitions of support
strategies in ESConv and DailyDialogue.
B More Evaluation Details
Automatic Metrics. Bleu-2(B-2)(Papineni et al.,
2002) first compute the geometric average of the
modified n-gram precisions, pn, using n-grams up
to length Nand positive weights wnsumming to
one. Next, let cbe the length of the prediction and
rbe the reference length. The BP and Bleu-2 are
computed as follows.
BP =1ifc > r
e(1−r/c)ifc≤r.(3)
Bleu = BP·exp NX
n=1wnlogp n!
.(4)
Rouge-L (R-L)(Lin, 2004) propose using LCS-
based F-measure to estimate the similarity between
two summaries Xof length mandYof length n,
assuming Xis a reference summary sentence and
Yis a candidate summary sentence, as follows:
Rlcs=LCS(X, Y)
m
Plcs=LCS(X, Y)
n
Flcs= 
1 +β2
RlcsPlcs
Rlcs+β2Plcs(5)
Where LCS(X, Y) is the length of a longest
common subsequence of XandY, and β=
Plcs/Rlcswhen ∂Flcs/∂R lcs=∂F lcs/∂Plcs. In
DUC, βis set to a very big number (→ ∞) . There-
fore, the LCS-based F-measure, i.e. Equation 5, is
Rouge-L.
CIDEr(Vedantam et al., 2015) score for n-
grams of length nis computed using the average
cosine similarity between the candidate sentence
and the reference sentences, which accounts for
both precision and recall:
CIDEr n(ci, Si) =1
mX
jgn(ci)·gn(sij)
∥gn(ci)∥∥gn(sij)∥,
(6)where gn(ci)is a vector formed by gk(ci)corre-
sponding to all n-grams of length nand∥gn(ci)∥
is the magnitude of the vector gn(ci). Similarly for
gn(sij).
Higher order (longer) n-grams are used to cap-
ture grammatical properties as well as richer seman-
tics. (Vedantam et al., 2015) combine the scores
fromn-grams of varying lengths as follows:
CIDEr(c i, Si) =NX
n=1wnCIDEr n(ci, Si),(7)
Empirically, Vedantam et al.(Vedantam et al., 2015)
found that uniform weights wn= 1/N work the
best. So, We also useN= 4.
LLM-as-a-Judge.Table 11 provides the detailed
prompt we feed to GPT4o for comparison evalua-
tion.
Human evaluation criteria.We enlisted four in-
terns as human evaluators to rate the models across
multiple dimensions, specifically Fluency, Emo-
tion, and Satisfaction. The Satisfaction dimension
encompasses four sub-aspects: Acceptance, Effec-
tiveness, Sensitivity, and Alignment. Throughout
this evaluation, we strictly adhered to international
regulations and ethical norms, ensuring that all pro-
cedures comply fully with the guidelines regarding
participant participation and data integrity. Each
evaluator was required to assess every sample in-
dependently and strictly in accordance with the
pre - pre-established criteria. By upholding these
principles, the evaluation process maintained ob-
jectivity, standardization, and consistency, thereby
improving the overall quality and credibility of the
evaluation results.
The detailed manual scoring criteria are as fol-
lows:
• Fluency:
1: The sentence is highly incoherent, making
it extremely difficult to understand and failing
to convey a meaningful idea.
2: The sentence has significant incoherence
issues, with only parts of it making sense and
struggling to form a complete thought.
3: The sentence contains some incoherence
and occasional errors, but can still convey the
general meaning to a certain extent.
12

Table 9: Statistics of ESConv, DailyDialogue, EmpatheticDialogues.
Category ESconv DailyDialogue EmpatheticDialogues
# Sessions 1.3K 13.1k 2.5K
# Utterances 38K 103.0k 11.0K
Average # Utterances 28.9 7.9 4.3
Average Utterance Length 18.8 13.6 16.7
Seeker/Speaker1# Utterances 20K 53.8k 5.7K
Avg # Utterances 15.4 4.1 2.2
Avg Uttr Len 16.8 13.2 20.8
# Strategies - 4 -
# Emotions 11 7 32
Supporter/Speaker2# Utterances 18K 49.2k 5.2K
Avg # Utterances 13.6 3.9 2.1
Avg Uttr Len 21.0 14.1 12.3
# Strategies 8 4 -
# Emotions - 7 32
Table 10: Strategy definitions and abbreviations of ESConv and DailyDialogue.
Dataset Strategies Abbreviation Definitions
ESconvQuestion Que.Inquiring about problem-related information to help the seeker clarify their issues,
using open-ended questions for best results and closed questions for specific details.
Restatement or Paraphrasing Res.& Par.A simple, more concise rephrasing of the help-seeker’s statements that could help
them see their situation more clearly.
Reflection of Feelings Ref. Articulate and describe the help-seeker’s feelings.
Self-disclosure Self-Dis.Divulge similar experiences that you have had or emotions that you share with the
help-seeker to express your empathy.
Affirmation and Reassurance Aff. & Rea.Affirm the help seeker’s strengths, motivation, and capabilities and provide reassur-
ance and encouragement.
Providing Suggestions Pro.Provide suggestions about how to change, but be careful to not overstep and tell
them what to do.
Information Inf.Provide useful information to the help-seeker, for example with data, facts, opinions,
resources, or by answering questions.
Others othersExchange pleasantries and use other support strategies that do not fall into the above
categories.
DailyDialogueInform -Provide factual or contextual information that the speaker believes the listener may
not know or is unaware of.
Question -.Seek specific information from the listener, assuming they possess the knowledge
being requested.
Directives -Express the speaker’s intention for the listener to take an action, including requests,
instructions, or suggestions.
Commissive -Indicate the speaker’s commitment to perform certain actions, such as accepting or
rejecting requests or offers.
4: The sentence is mostly fluent with only
minor errors or slight awkwardness in ex-
pression, and effectively communicates the
intended meaning.
5: Perfect. The sentence is completely fluent,
free of any errors in grammar, punctuation, or
expression, and clearly conveys the idea.
• Emotion:
1: The emotional expression is extremely in-
appropriate and chaotic, not in line with the
content, and may convey wrong emotions.
2: The emotional expression has obvious
flaws, either too weak or exaggerated, and
is disjointed from the content.
3: The emotional expression is average. It can
convey basic emotions but lacks depth and has
minor issues.4: The emotional expression is good. It can
effectively convey the intended emotion with
an appropriate intensity and is well integrated
with the content.
5: The emotional expression is excellent. It
is rich, nuanced, and perfectly matches the
content, capable of evoking a strong and ap-
propriate emotional response.
• Acceptance:
1: The response inescapably triggers emo-
tional resistance.
2: The response is highly likely to trigger
emotional resistance.
3: The response has a possibility of emotional
resistance occurring.
4: The response rarely provokes emotional
resistance.
13

Table 11: The Comparison Evaluation Prompt Content by GPT-4o from (Madani et al., 2024)
Prompt
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should
choose the assistant that follows the provided strategy more carefully and precisely to answer the user’s last utterance. You should
use your emotional support expertise and knowledge to judge the quality of the response considering how well the answer follows
the provided strategy. Your evaluation most importantly should consider strategy adherence and then the overall quality, naturalness,
consistency and coherence of the final utterance.
Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and
ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to
influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation,
output your final verdict by strictly following this format: "JUDGE: [[A]]" if assistant A is better, "JUDGE: [[B]]," if assistant B is better,
and "JUDGE: [[C]]" for a tie.
Conversation history:
{conversation_history}
<|The Start of Assistant A’s Response|>
{assistant_a_resp}
<|The End of Assistant A’s Response|>
<|The Start of Assistant B’s Response|>
{assistant_b_resp}
<|The End of Assistant B’s Response|>
5: The response has no occurrence of emo-
tional resistance.
• Effectiveness:
1: The response actually worsens the seeker’s
emotional distress.
2: The response carries the risk of increasing
stress levels, and this outcome varies depend-
ing on the individual user.
3: The response fails to alter the seeker’s cur-
rent emotional intensity and keeps it at the
same level.
4: The response shows promise in calming
the emotional intensity; however, it is overly
complicated or ambiguous for the user to fully
comprehend and utilize effectively.
5: The response appears to be highly effective
in soothing the seeker’s emotions and offers
valuable and practical emotional support.
• Sensitivity:
1: The response renders inaccurate evaluations
regarding the seeker’s state.
2: The response is characterized by rash judg-
ments, as it lacks adequate assessment and
in-depth exploration of the seeker’s state.
3: The response is formulated with a one-
sided judgment and a limited exploration ofthe seeker’s state.
4: The response demonstrates an understand-
ing that only covers a part of the seeker’s state.
5: The response precisely grasps the seeker’s
state and is appropriately tailored according
to the seeker’s actual situation.
• Alignment:
1: The response is in total contradiction to the
predicted strategy.
2: The response has a minor deviation from
the predicted strategy.
3: There is some ambiguity between the re-
sponse and the predicted strategy.
4: The response largely matches the predicted
strategy, yet it contains some ambiguous ele-
ments.
5: The response effectively makes itself con-
sistent with the predicted strategy.
• Satisfaction:
1: The response is extremely disappointing. It
doesn’t answer the question at all and is of no
help.
2: The response is poor. It only gives a partial
answer and leaves many doubts unresolved.
14

3: The response is average. It meets the basic
requirements but isn’t particularly outstand-
ing.
4: The response is good. It answers the ques-
tion clearly and provides some useful details.
5: The response is excellent. It not only an-
swers the question perfectly but also offers
valuable additional insights.
At the end, the cross-validation is conducted
with the averaged results collected.
C Detailed Introductions of Baselines
We introduce the baselines with more details:
• Direct: directly inference the LLM.
•Direct-Refine: a straightforward refinement
method in which the model revises its ini-
tial response to incorporate emotional support
considerations.
•Self-Refine (Madaan et al., 2023): generates
feedback about the emotional support based
upon its initial response, then refines its re-
sponse based on this feedback.
•ECoT (Li et al., 2024): an analogy of Chain-
To-Thought prompt (Wei et al., 2022) on the
emotional conversation, which first generates
the seeker’semotion, which then guides the
generation of strategy and response.
•SoT (Ning et al., 2024): first guides LLMs to
generate the skeleton of the answer, and then
in parallel decodes to complete the contents.
•ToT (Yao et al., 2023a): decomposes complex
problems into intermediate thought steps (like
a tree structure), where LLMs evaluate the
validity of each step and iteratively explore
multiple reasoning paths to select the optimal
solution, enhancing the ability to handle multi-
step logical tasks.
•Plan-and-Solve (Wang et al., 2023): first
prompts LLMs to generate a detailed plan out-
lining sub-goals and reasoning strategies, then
executes the plan step-by-step to complete the
solution, integrating planning and execution
to improve the coherence and completeness
of responses, especially in mathematical rea-
soning and multi-turn decision scenarios.•RAG (Fan et al., 2024): the retrieval-
augmented generation, which chooses the ap-
propriate strategy by a semantic retriever, and
includes it in the prompt to improve the re-
sponse quality. In this work, we employ E5-
Large (Wang et al., 2024) as the retriever.
•CSIM (Zhou et al., 2024): a self-chat method
which produces the response based on its im-
plicit simulation of future dialogue.
D More Experimental Results
D.1 More Typical Cases
Table 12 provides typical cases in DailyDialogue.
From the contents generated, we can observe that
CoCT can recognize the concept list, generate a
chain of concepts, and then produce the detailed re-
sponse based on the concept skeleton. The concept
flow induces the LLM’s reasoning path. Table 13
shows cases of MultiWoz which show that CoCT
adapts well to MultiWoz’s topics.
D.2 Bad Case Analysis
Table 14 provides a typical bad case. The red part
indicates that CoCT attempts to transfer to the topic
of ‘Hobbies and crafts’, however, which is almost
the same as the original topic. Improvement of
foundation models may alleviate such issues.
15

Table 12: Cases of CoCT in DailyDialogue, in contrast to Direct Inference.Case 1user I’m sorry I’m so late . I had a really bad day.
Direct It’s ten after six. We’re late. But dinner is at six thirty.
CoCT<Commissive>No need to apologize; I’m just glad you’re here now.
<Question>Do you want to talk about your day? Sometimes sharing can
really help.Case 2user But Sunday is the same with Saturday. If you don’t mind, that’s OK.
DirectHum.Maybe it will be better. The most important thing is that I can have
a day to relax after a week’s work .
CoCT<Surprise>Oh, really? I never noticed that.
<Question>Are you sure Sunday is just as busy as Saturday?
Maybe we could go early in the morning when it’s less crowded?Case 3user Believe it or not, tea is the most popular beverage in the world after water.
Direct Well, people from Asia to Europe all enjoy tea.
CoCT<Inform>Yes, that’s true! Tea is indeed the most widely consumed
beverage in the world after water. Its popularity spans across many cultures
and countries, with different varieties like black, green, white, and oolong tea.
Table 13: Case of CoCT on MultiWOZ, in contrast to the original response.
user I will be needing a place to stay on the west side of town.
assistant Hobsons house is in the west part of town, would you like to stay there?
CoCT<Affirmation and Reassurance>You’ve got a good head start on
planning your trip to Cambridge.<travel>For accommodations on the
west side. I can suggest checking out hotels near the River Cam or universities
like Churchill College, what’s your budget like for the stay?
Table 14: Bad case example of CoCT in ESconv.
user I just need ways to find validation that are not based on my job. But validation doesn’t pay the bills.
CoCT<Reflection of feelings>It sounds like you’re feeling a bit lost and uncertain about your sense of
purpose and identity outside of your job.
<Questioning>Can you tell me more about what you mean by "validation", what does that look like
for you, and what are some things that have given you a sense of fulfillment and self-worth in the past?
<Hobbies and crafts>Perhaps we can explore some hobbies or activities that bring you joy and help
you feel more connected to yourself and others.
16
# BugRepro: Enhancing Android Bug Reproduction with Domain-Specific Knowledge Integration

**Authors**: Hongrong Yin, Tao Zhang

**Published**: 2025-05-20 15:48:34

**PDF URL**: [http://arxiv.org/pdf/2505.14528v1](http://arxiv.org/pdf/2505.14528v1)

## Abstract
Mobile application development is a fast-paced process where maintaining
high-quality user experiences is crucial. Current bug reproduction methods
predominantly depend on precise feature descriptions in bug reports. However,
the growing complexity and dynamism of modern software systems pose significant
challenges to this crucial quality assurance process, as ambiguous or
incomplete steps-to-reproduce (S2Rs) in reports frequently impede effective
debugging and maintenance. To address these challenges, we propose BugRepro, a
novel technique that integrates domain-specific knowledge to enhance the
accuracy and efficiency of bug reproduction. BugRepro adopts a
Retrieval-Augmented Generation (RAG) approach. It retrieves similar bug reports
along with their corresponding S2R entities from an example-rich RAG document.
This document serves as a valuable reference for improving the accuracy of S2R
entity extraction. In addition, BugRepro incorporates app-specific knowledge.
It explores the app's graphical user interface (GUI) and extracts UI transition
graphs. These graphs are used to guide large language models (LLMs) in their
exploration process when they encounter bottlenecks. Our experiments
demonstrate the effectiveness of BugRepro. Our method significantly outperforms
two state-of-the-art methods. For S2R entity extraction accuracy, it achieves
improvements of 8.85% and 28.89%. For bug reproduction success rate, the
improvements reach 74.55% and 152.63%. In reproduction efficiency, the gains
are 0.72% and 76.68%.

## Full Text


<!-- PDF content starts -->

arXiv:2505.14528v1  [cs.SE]  20 May 2025BugRepro: Enhancing Android Bug Reproduction
with Domain-Speciﬁc Knowledge Integration
Hongrong Yin1and Tao Zhang1*
1Department of Computer Science and Engineering, Macau Univ ersity
of Science and Technology, Avenida Wai Long, Taipa, MACAU, 9 99078,
Taipa MACAU, China.
*Corresponding author(s). E-mail(s): tazhang@must.edu.mo ;
Contributing authors: hongrongyin@proton.me ;
Abstract
Mobile application development is a fast-paced process whe re maintaining
high-quality user experiences is crucial. Current bug repr oduction methods pre-
dominantly depend on precise feature descriptions in bug re ports. However, the
growing complexity and dynamism of modern software systems pose signiﬁcant
challenges to this crucial quality assurance process, as am biguous or incom-
plete steps-to-reproduce (S2Rs) in reports frequently imp ede eﬀective debugging
and maintenance. To address these challenges, we propose Bu gRepro, a novel
technique that integrates domain-speciﬁc knowledge to enh ance the accuracy
and eﬃciency of bug reproduction. BugRepro adopts a Retriev al-Augmented
Generation (RAG) approach.Itretrieves similar bugreport salongwith their cor-
responding S2R entities from an example-rich RAG document. This document
serves asavaluablereferenceforimprovingtheaccuracyof S2Rentityextraction.
In addition, BugRepro incorporates app-speciﬁc knowledge . It explores the app’s
graphical user interface (GUI) and extracts UI transition g raphs. These graphs
are used to guide large language models (LLMs) in their explo ration process
when they encounter bottlenecks. Our experiments demonstr ate the eﬀectiveness
of BugRepro. Our method signiﬁcantly outperforms two state -of-the-art meth-
ods. For S2R entity extraction accuracy, it achieves improv ements of 8.85% and
28.89%. For bug reproduction success rate, the improvement s reach 74.55% and
152.63%. In reproduction eﬃciency, the gains are 0.72% and 7 6.68%.
Keywords: Large Language Model, Retrieval-Augmented Generation, Andro id
Automation, Crash Reproduction
1

1 Introduction
Mobile applicationshavebecomean essentialpartofdailylife, with ove r4millionapps
available on Google Play and the Apple App Store as of 14 December 202 4 (42matters
2023). As the market for mobile apps continues to grow, ensuring a seam less user
experience has become a critical priorityfor developersstrivingto remain competitive.
A key aspect of maintaining app quality is addressing the issues repor ted by users,
typically through bug reports. These reports often include steps to reproduce (S2R),
enabling developers to replicate the reported problems, identify th eir root causes, and
implement ﬁxes.
However, these reproduction steps are frequently unclear, inco mplete, or inaccu-
rate. As a result, manually reproducing bugs can be time consuming a nd resource
intensive, further delaying the process of bug resolution ( Rahman et al. 2022 ;Aranda
and Venolia 2009 ;Just et al. 2018 ). To address this challenge, many automated solu-
tions have been proposed to assist in bug reproduction. These met hods typically
leverage Natural Language Processing (NLP) techniques to extr act structured S2R
entities from bug reports using predeﬁned rules or learned models ( Fazzini et al. 2018 ;
Zhao et al. 2019 ,2022;Huang et al. 2023 ). Then they leverage heuristic search algo-
rithms to explore the app’s graphical user interface (GUI) and rep lay the extracted
S2R entities. Despite these advancements, current automated m ethods still face signif-
icant diﬃculties due to the ambiguity and vagueness of S2Rs, which po ses a signiﬁcant
challenge to state-of-the-art NLP techniques ( Dav 2018 ;Nogueira and Cho 2019 ).
Moreover, the complexity of modern mobile apps further complicate s the task of
accurately replaying the extracted S2R entities ( Huang et al. 2023 ;Feng and Chen
2024). Recently, large language models (LLMs) have developed rapidly an d demon-
strated remarkable capabilities in understanding diverse linguistic ex pressions( Jansen
et al. 2023 ). Building on this, Sidong Feng et al. propose AdbGPT ( Feng and Chen
2024) , which harnesses the power of LLMs to improve bug reproduction . Speciﬁcally,
AdbGPT utilizes the in-context learning abilities of LLMs to extract S2 R entities by
learning from developer-provided few-shot examples. Then, AdbG PT iteratively asks
the LLM to match S2R entities with GUI events to reproduce the bug s.
Existing methods still heavily rely on the original content of bug repo rts. Specif-
ically, when the descriptions are ambiguous or diﬃcult to comprehend , current
approaches fail to extract accurate and useful information fro m these reports. More-
over, modern applications exhibit inherent complexity with multiple pag es and
diverse functionalities, making it challenging for existing methods to m ap the rele-
vant information in bug reports to the corresponding UI elements t hat need to be
manipulated.
Challenge 1:Lack of Domain Understanding in Bug Report Inte rpreta-
tion.Since S2Rs provided by non-expert users frequently exhibit a lack o f precision
and often omit essential procedural steps, thereby compromisin g their completeness
and clarity. Prior work demonstrates LLMs’ limited comprehension o f Android bug
reports, yielding hallucinated outputs in S2R extractions ( Lewis et al. 2020 ;Wu et al.
2024;Huang et al. 2025 ), making them ineﬀective at extracting structured S2R enti-
ties from unstructured bug reports. While AdbGPT ( Feng and Chen 2024 ) attempts
to address this issue by leveraging few-shot examples; however, it s performance falls
2

short due to the small number of examples (only 15) and the high cos t of manually
constructing them.
Challenge 2: Lack of Dynamic Behavior Modeling in App Reprod uc-
tion.Modern apps often involvemulti-page ﬂows and complex functionalitie s, making
it diﬃcult to replay S2Rs without app-speciﬁc knowledge of screen tr ansitions and
dynamic GUI states. For instance, reproducing a bug in a shopping a pp’s check-
out process requires navigating several pages, each dependent on user actions and
dynamic elements. Existing methods, including LLM-based ones, lack this contextual
understanding, often resulting in failed reproductions.
This motivates our core research question: How can we eﬀectively lever-
age LLMs to serve as the “brain” of developers and better faci litate the
reproduction of Android crashes?
To overcome these constraints, we propose BugRepro, a novel t echnique that inte-
grates bug-report-speciﬁc knowledge and app-speciﬁc knowledg e to enhance the bug
reproduction process. Speciﬁcally, BugRepro ﬁrst employs a Retr ieval-Augmented
Generation (RAG) ( Lewis et al. 2020 ) approach to retrieve similar bug reports and
their corresponding S2R entities from an example-rich RAG documen t. The retrieved
knowledge serves as a reference, enabling LLMs to extract S2R en tities more accu-
rately for the currentbug report. Next, BugReproincorporate sapp-speciﬁc knowledge
by exploring the app’s GUI to improve the LLM-guided bug reproduct ion process fur-
ther. When LLMs encounter challenges, such as diﬃculty in determin ing which UI
component to interact with or missing steps in the reproduction pro cess, BugRepro
extracts UI transition graphs(UTG) that model the changes and interactions of the
app state on diﬀerent screens. These graphs are then fed into th e LLM to guide its
decision-making,ensuring the reproduction follows the correct se quence of actions and
app state transitions.
We conduct extensive experiments on 151 real-world bug reports t o evaluate the
performance of BugRepro. First, we evaluate the accuracy of S2 R extraction and com-
pare BugRepro with two state-of-the-art techniques (i.e., RecDr oid and AdbGPT ).
The experimental results demonstrate that BugRepro outperfo rms these approaches,
achieving improvements of 7.57% and 28.89%. in terms of the extract ion accuracy,
respectively. Then, we evaluate the eﬀectiveness of bug replay. T he experimental
results show that RecDroid and AdbGPT successfully reproduced 3 8 and 55 bugs,
whereas BugRepro can reproduce 96 bugs, achieving an improveme nt of 152.63%
and 74.55%, respectively. Furthermore, BugRepro reduces the t ime required for bug
reproduction, with an average time of 124.7 seconds, compared to 534.9 seconds for
RecDroid and 125.6 seconds for AdbGPT. Finally, ablation studies con ﬁrm that each
component of BugRepro contributes to the overall performance .
The contributions of this paper are as follows:
•We proposeBugRepro,a noveltechnique that integratesLLMswit h domain-speciﬁc
knowledge to enhance Android bug reproduction.
•BugRepro eﬀectively addresses three key challenges in Android bug reproduc-
tion: the lack of domain-speciﬁc knowledge, the complexity of moder n applications
with multi-page structures and diverse functionalities, and the inst ability of
LLM-generated output formats in the context of code interactio n.
3

•We evaluate the eﬀectiveness of BugRepro through extensive exp eriments, demon-
strating signiﬁcant improvements over two state-of-the-art me thods in terms of
S2R entity extraction accuracy, bug reproduction success rate , and reproduction
eﬃciency.
The rest of the paper is organized as follows: Section 2provides an introduction
to the background and motivation for this work. Section 3details the analysis of our
method. Section 4formulates four key research questions, characterizes the exp er-
imental data and conﬁguration, reports empirical ﬁndings with the corresponding
evaluations. Section 5presents the results corresponding to the research questions a nd
analyzes them in detail. Section 6discusses the limitation of our method. Section 7
reviews existing research in related ﬁelds. Finally, Section 8concludes the paper.
2 Motivation and Background
2.1 Motivation
Despite progress in prior research and tool development, fundam ental gaps hinder
their application to Android crash reproduction.
Table1provides examples of real-world bug reports, and Table 2presents a
comparison of the corresponding S2R entity extraction results.
Table 1: An example of Bug Reports
Example Case Description
Example 1 1. Crash application when search.
Example 2 1. On Category screen (either opened from transaction form o r from
Settings), click on search icon and enter A as search term
2. Category A now appears without children
3. Tap and hold on category A
Example 3 ...Enter “test” in the “Secret” ﬁeld (and ﬁll other required ﬁelds)...
We observe that prior approaches tend to heavily rely on the prese nce of explicit
action verbs(e.g.,“Tap”,“Click”).Consequently,they often fail to extractS2R entities
when action verbs are implicit or absent in the sentence. Moreover, these methods
frequently fail to extract entities when sentences contain multiple components (e.g.,
“Secret ﬁeld” or “other required ﬁelds”) or actions (e.g., “enter” or “ﬁll”) of the same
type, resulting in incomplete or ambiguous extraction results.
During the replay stage, prior methods ( Zhao et al. 2019 ;Feng and Chen 2024 )
share the same shortcoming—neither nor deal with scenarios that involve multi-
page ﬂows and complex functionalities. According to the algorithms u sed in prior
approaches, during the exploration process, if a bug report skips the description of
a component on the current screen and directly mentions a compon ent on the next
screen, the system tends to perform a “back” operation to retu rn to the previous
screen.Forexample,abugreportsays“SelectAbout”, but onth e currentscreen,there
4

Table 2: S2R entity extraction results comparison
Method Extraction Results
Example 1 Results
ReCDroid /
AdbGPT /
Ours [Tap] [search]
Example 2 Results
ReCDroid 1. [Tap][screen]
2. [Tap][icon]
3. [input]
4. [click]
AdbGPT 1. [Tap][search icon]
2. [Input][search term]
Ours 1. [Tap] [search icon]
2. [Input] [search term] [A]
3. [Long Tap] [category A]
Example 3 Results
ReCDroid [Input][ﬁeld][test]
[Input][ﬁeld][test]
AdbGPT [Input][Secret][test]
Ours [Input] [Secret ﬁeld] [test]
[Input] [other required ﬁelds]
is no component labeled “About”. We need to select any component f rom this screen
ﬁrst, and then we can see the “About” component. When prior met hods encounter
this situation, they simply choose the “back” action, which never lea ds to the correct
screen. This often leads to incorrect looping behavior, ultimately ca using the failure
of bug reproduction.
2.2 Large Language Models
LLMs represent a signiﬁcant advancement in natural language und erstanding and
generation, leveraging the Transformer architecture to achieve remarkable perfor-
mance across a broad spectrum of tasks. These models, including e xamples such
as Claude3.5 ( Anthropic 2024 ), GPT-4 ( OpenAI 2023 ), LLaMA ( Llama 2024 ), and
DeepSeek ( DeepSeek 2024 ) are characterized by their billions of parameters and
training on massive text corpora. Their scale and design enable capa bilities like
mathematical reasoning, program synthesis, and multi-step reas oning that outper-
form traditional benchmark models tailored to speciﬁc tasks. The c ore functionality
of LLMs revolves around processing prompts—task-speciﬁc instr uctions in natural
language( Liu et al.2023 ;Brownet al. 2020 ).Promptsaretokenizedinto wordsorsub-
words and passed through layers of the Transformer model, which employs multi-head
self-attention,normalization,andfeed-forwardnetworkstoco mprehendinputandgen-
eratecontextuallyappropriateresponses.Throughthis mechan ism,LLMscanperform
5

tasks without requiring task-speciﬁc ﬁne-tuning, relying instead o n prompt engineer-
ing to elicit desired behaviors. In this work, we leverage LLMs to tack le the complex
task of bug reproduction. By carefully designing prompts, we instr uct the LLMs to
extract S2R entities, generate actionable operations, and decod e their responses to
replicate bugs eﬀectively.
However, applying LLMs to bug reproduction requires precise mapp ing from
natural language to UI actions—a task where hallucinated outputs may occur.
2.3 Retrieval-Augmented Generation
RAG(Lewisetal.2020 )isahybridapproachthatenhancesthecapabilitiesofLLMsby
integrating retrieval mechanisms. Unlike standalone LLMs, which re ly solely on their
pre-trained knowledge, RAG retrieves relevant information from e xternal knowledge
sources, such as databases or document repositories, to improv e response accuracy
and relevance. This process involves two key components:
•Retrieval phase: A retrieval model, often based on dense vector embeddings, iden-
tiﬁes the most relevant documents or examples from a pre-constr ucted knowledge
base.
•Generation phase: The retrieved information is incorporated into t he prompt,
enabling the LLM to generate more informed and contextually appro priateoutputs.
RAG is particularly eﬀective for tasks requiring domain-speciﬁc or up -to-date
knowledge, where LLMs alone may struggle due to limitations in their tr aining
data (Falke et al. 2019 ;Duˇ sek and Kasner 2020 ;Brown et al. 2020 ;Liu et al. 2023 ;
Ji et al. 2023 ;Gao et al. 2023 ). By dynamically integrating external knowledge, RAG
enhances both the reasoning and contextual understanding of L LMs. In this work, we
employ a RAG approach ( Gao et al. 2023 ;Ma et al. 2023 ) to retrieve examples of
annotated S2R entities from a vector database, using them to guid e LLMs in accu-
ratelyextractingand generatingactionableoperationsfor bug re production.We adapt
RAGtoretrieveannotatedS2Rexamples,addressingLLMs’lackof bugreport-speciﬁc
knowledge.
2.4 Bug Reports and App UI Elements
Bug reports are essential for documenting unexpected applicatio n behaviors. They
are typically user-or tester-generated and describe the problem , the conditions under
which it occurs, and the S2R. However, many bug reports lack explic it S2R details,
which are crucial for replicating and diagnosing the issues. Accurat ely extracting
S2R information from bug reports is challenging but vital for stream lining the bug
reproduction and resolution process.
Mobile applications consist of dynamic UI screens, which serve as the visual canvas
for implementing app features and are composed of UI components (widgets) such
as buttons, text ﬁelds, and checkboxes. These components allow users to interact
with the app and are organized hierarchically within containers (layou ts). Each UI
screen is represented as a screenshot paired with metadata deta iling the hierarchy and
6

attributes of its components. Attributes include the component t ype (e.g., TextView,
Button), label or text, ID, description, visibility, and size.
Interactions with UI components are represented as actions deﬁ ned by a tuple
(target element, action type, value). The target element refers to the UI component
(e.g., button or text box), the action type describes the interact ion (e.g., “click”,
“input”, or “swipe”), and the value is speciﬁc to the action (e.g., inpu t text).
In this work, we deﬁne S2R entities as including the following componen ts: inter-
actions with UI components, referred to as Action; the target UI elements involved
in those interactions, referred to as Element ; the speciﬁc content associated with
the action (e.g., input text), referred to as Value; and the gesture direction when
interacting with scrollable elements, referred to as Direction . Identifying and accu-
rately mapping interactions with these UI components to S2R entitie s is crucial for
automating bug reproduction.
This work focuses on bridging the gap between textual bug report s and actionable
UI interactions by extracting S2R entities and correlating them with app UI compo-
nents, enabling accurate and automated bug reproduction. This m ismatch between
unstructured reports and structured UI interactions motivate s our approach.
3 Approach
This paper introduces BugRepro, an eﬀective approach to overco ming the challenges
posed by ambiguous S2Rs and the intricate nature of modern Andro id applications.
Fig.1presents an overview of BugRepro. In the following sections, we de tail the
two main components of BugRepro: RAG-enhanced S2R entity extr action ( §3.1) and
exploration-based replay ( §3.2).
65 HQWLW\ H[WUDFWLRQ
$XWRPDWHG
5HSURGXFWLRQ%XJ
UHSRUW
5$*
'RFXPHQW
/DEHOHG
([DPSOH
//0
&RPSRQHQW
$FWLRQ
 ,QSXW
'LUHFWLRQ65 (QWLWLHV
//0 *XLGHG 5HSOD\
$FWLRQ
7DUJHW
ZLGJHW
,QSXW'LUHFWLRQ&RPPDQG /LVW
$SSOLFDWLRQ
)XQFWLRQDOLW\DZDUH 8, H[SORUDWLRQ
8,
$XWRPDWRU
87*
 :LGJHWV
 )XQFWLRQDOLWLHV:LGJHWV
 )XQFWLRQDOLWLHV
6XFFHVV
)DLO
$FWWLLRQ ,QQSXW
''LLUHFWLLRQ
//
0
,,QSSXXW
'LUHHFFWLRRQ
6XFFHVV
)DLO
6XXFFFFHHHHHHVVVVVV
Fig. 1: The overview of our method.
7

Our method is structured into two distinct phases and comprises th ree core com-
ponents.The ﬁrst phase is RAG-enhanced S2R entity extraction, we employ
RAG to enhance the accuracy of S2R entity extraction 3.1.2. When a new bug report
is submitted, it is ﬁrst segmented into individual sentences. For eac h sentence, the
system automatically retrieves semantically similar sentences and th eir correspond-
ing labels (which include all associated S2R entities) from the RAG data base—an
example is shown in Table 3. Then, the original bug report along with the retrieved
labeled examples is fed into LLM, which is guided by the examples to analy ze and
extract the S2R entities corresponding to the new bug report. Af ter extracting the
S2R entities, we leverage LLMs to interact with UI components base d on these enti-
ties and corresponding bug reports to replay the bug. At the same time, based on
existing work and our observations, we found that even after app lying RAG, chal-
lenges remain in extracting components, which may lead to diﬃculties in correctly
matching and operating the corresponding components during the replay process.
Therefore, we need to provide the LLM with more UI-related knowle dge.the second
phase is Exploration-based Replay, which aims to provide app-related knowl-
edge to LLMs, enabling them to gain insights into the app, understan d its usage, and
make informed decisions. It consists of two primary components: UI explo-
ration based on functionality and LLM guided replay, which c ontinuously
interact with each other throughout the process. The functi onality-aware
UI exploration component based on functionality is responsible for exploring
the UI, and all information will be transmitted to LLM. Note that sinc e UI screens
are visual and can not be processed by LLMs directly, we encode th e UI screens
into text format for LLMs processing following existing work ( Feng and Chen 2024 ).
The LLM guided replay component receives the encoded UI information from
the functionality-aware UI exploration module, synthesizes it with in puts from the
RAG-enhanced S2R entity extraction module and the original bug re port, and then
generates guidance for the next interaction step based on this co mprehensive context.
After executing each step, Bugrepro checks whether the bug r eproduc-
tion has succeeded and provides feedback to the LLM. If repro duction
fails, it triggers the functionality-aware UI exploration process, allowing
the acquisition of additional domain-speciﬁc knowledge to enhance the
LLM’s decision-making. During the LLM’s decision-making p rocess, when
the LLM faces diﬃculties determining the next step ( §3.2.1),BugRepro
augments its decision-making by integrating app-speciﬁc knowledge , which encom-
passes both UI exploration and the synthesis of app-related infor mation. This process
involves analyzingthe app’s UI elements and interactions to constru ct the Synthesized
Functionality Table (Table 5), which consists of three components: <Synthesized
functionality, UI states, UI elements> . Once equipped with more comprehen-
sive domain-speciﬁc knowledge, the LLM is able to make more accurat e and eﬀective
decisions in response to the current situation.
3.1 RAG-enhanced S2R entity extraction
As discussed in Section 1, LLMs struggle with accurate S2R entity extraction due
to their lack of bug-report-speciﬁc knowledge. Without the backg round context or
8

established best practices for extracting these entities, LLMs of ten produce inaccurate
results. To overcome this, BugRepro enhances S2R entity extrac tion by integrating
LLMs with relevant, bug-report-speciﬁcknowledge using a RAG app roach. In particu-
lar, BugRepro incorporates a lightweight RAG component that retr ieves task-relevant
information (e.g., similar bug reports and their corresponding labeled S2R entities).
This retrieved information is then incorporated into the LLM’s promp t, equipping the
model with both bug-report-speciﬁc and task-oriented knowledg e, which signiﬁcantly
improves its contextual understanding and ensures more accura te extraction of S2R
entities. In the following, we ﬁrst describe how we construct the RA G database and
then explain how it enhances the S2R entity extraction process.
3.1.1 Constructing the RAG database
The Basic RAG approach stores all documents in a single database, r eﬂecting the
most common cases of RAG-based LLMs ( Lewis et al. 2020 ;Ma et al. 2023 ;Gao et al.
2023). Therefore, we ﬁrst construct such a database for the RAG pr ocess.
Data Collection . We begin by crawling a large collection of bug reports from
various web resources referenced in prior research ( Zhao et al. 2019 ;Su et al. 2021 ;
Cooper et al. 2021a ;Huang et al. 2023 ).These reportsare then embedded to construct
a vector database for the RAG process. Speciﬁcally, we use the all- MiniLM-L12-v2
model (Reimers and Gurevych 2024 ;Galli et al. 2024 ;Aperdannier et al. 2024 ;Park
and Shin 2024 ), which is known for its lightweight architecture, computational eﬃ -
ciency, and high accuracy. It is worth noting that the choice of emb edding model is
orthogonalto our approach,and it can easily be swapped for othe r embedding models.
Labeling S2R Entities . Although we obtain a set of bug reports in the previous
step, they are not annotated with the corresponding S2R entities . Following existing
work (Feng and Chen 2024 ;Wang et al. 2024 ), we employ four developers with 6–10
years of experience in both software testing and development to la bel the bug reports.
Before starting the labeling process, we provide the developers wit h a detailed
project overview. Through a lottery process, three developers are selected to label
the reports, while the fourth developer is tasked with consolidating their results. To
label those bug reports, the developers ﬁrst carefully break dow n each bug report into
individual sentences. The developers then carefully analysis each s entence to extract
its corresponding S2R entities, which consist of four distinct compo nents, as deﬁned
by our predeﬁned standards: action type, target component, in put values, and scroll
direction.
Cross-Validation and Consistency Assurance . Once the initial labeling is
complete, following existing work ( Feng and Chen 2024 ;Wang et al. 2024 ), the fourth
developerassumesacrucialroleinensuringtheconsistencyandac curacyofthelabeled
data. We follow the dataset construction approach used in AdbGPT , which involves
hiring two data annotators.When their annotations diﬀer, a third p erson is brought in
to make the ﬁnal decision. Speciﬁcally, the fourth developer syste matically compares
the results produced by the ﬁrst three developers, focusing on r esolving discrepancies
in how identical sentences are interpreted. Leveraging extensive expertise, the fourth
developeruniﬁesthevaryingopinions,ensuringtheﬁnallabeleddat asetisscientiﬁcally
consistent and reliable.
9

3.1.2 RAG-enhanced S2R entity extraction
Given a bug report Bconsisting of Nsentences, denoted as s1,s2,...,s n, BugRe-
pro ﬁrst divides the report into individual sentences. Next, BugRe pro uses the same
embedding model employed during the database construction (i.e., a ll-MiniLM- L12-
v2 model) to embed each sentence. For each sentence si(i= 1,2,...,n), BugRepro
calculates the cosine similarity between the embedding of siand each sentence in the
RAG database.The mostsimilarsentence, alongwith its annotatedS 2R entity, isthen
retrieved and used as a reference for extracting the correspon ding S2R entity for si.
Table 3: An example of retrieved results.
Sentence in current Bug Report Retrieved sentence with S2R entity
Long hold on any video and press add to
playlist. (Alternatively, select the add to
playlist option under any video when watch-
ing).Sentence: Click on the add to option and
select playlist.
The entities extracted from the sentence are:
1. [Tap] [add to option]
2. [Tap] [playlist]
Note: This table shows a example of a bug report sentence and a mat ched sentence retrieved from
the RAG database with extracted S2R entities.
Table3provides an example of a retrieval result. Given a sentence from a b ug
report (e.g., “Long hold on any video and press add to playlist. Altern atively, select
the add to playlist option under any video when watching.”), BugRepr o retrieves
the most similar sentence from the database (e.g., “Click on the add t o option and
select playlist.”)and then, the correspondingS2R entities. Incorp oratingthis retrieved
example into the prompt assists the LLM in accurately extracting th e S2R entity for
the given sentence.
Prompt Construction . After retrieving relevant examples for each sentence in
the bug report, BugRepro constructs a prompt to instruct the L LM to extract S2R
entities. Following prior work ( Feng and Chen 2024 ), we begin by identifying all
available actions and action primitives within the S2R entities to guide th e LLM.
Speciﬁcally, we deﬁne seven standard actions: [tap, input, scroll, rotate, delete,
double tap, long tap] , which we refer to as available actions . The context of
each action may vary, and thus each action requires a set of primitiv es to accurately
represent its entities. For instance, the Tapaction requires a target component to
interact with, such as a button in the GUI. This can be represented as[Tap] [Com-
ponent] . Similarly, for the Double-tap andLong-tap actions, we use analogous
linguistic primitives. The Scrollaction involves specifying the direction of the scroll,
such as upward or downward, which we formulate as [Scroll] [Direction] . Likewise,
theRotateaction is expressed using a similar structure, i.e., [Rotate] [Direction] .
TheInputaction corresponds to entering a speciﬁc value into a text ﬁeld and is thus
formulated as [Input] [Component] [Value] . Similarly, the Deleteaction is repre-
sented using the primitive [Delete] [Component] , and other similar actions follow
the same pattern.
10

After deﬁning the available actions and their corresponding primitive s, BugRe-
pro incorporates the retrieved examples into the prompt to furth er assist the LLM.
Speciﬁcally, BugRepro includes each retrieved sentence along with it s corresponding
annotated S2R entity, formatted as follows: Here are some examp les for S2R entity
extraction. The sentence is “Click on the add to option and select pla ylist.”, the
extracted S2R entities are: 1. [Tap] [add to option], 2. [Tap] [playlist].
Table4shows an example of the constructed prompt. The aforementione d infor-
mation is aggregated with the content of bto form the complete prompt. The
ﬁnal structure of the prompt sent to the LLM includes: 〈Available Actions 〉+
〈Action Primitives 〉+〈Retrieved Example Sentences with S2R Entities 〉
+〈Current Bug Report 〉.
Table 4: An example of designing prompts for S2R entity extraction.
Prompt Framework Implementation
Available Actions [tap(click), input(set text), scroll, swipe, rotate, delete, double
tap(click), long tap(click), restart, back]. Generate inp ut when none is
given.
Action Primitive [Tap] [Component], [Scroll] [Direction], [Input] [Componen t] [Value],
[Rotate] [Component], [Delete] [Component] [Value], [Doubl e-tap]
[Component], [Long-tap] [Component]. The actions you identi fy should
be in the available actions.
Retrieval Prompt Here are some examples for S2R entity extraction. The sentence i s
“Click on the add to option and select playlist.”, the extrac ted S2R
entities are: 1. [Tap] [add to option] 2. [Tap] [playlist]. T he sentence
is “Rotate your phone. (Enable auto-rotate ﬁrst).”, the ext racted S2R
entity is: 1. [Rotate].
Current BugReport Here are the sentences in current bug report: 1. Long hold on an y
video and press add to playlist. (Alternatively, select the a dd to playlist
option under any video when watching). 2. Rotate the screen w hile
auto-rotate feature is on.
3.2 Exploration-based Replay
3.2.1 UTG-Driven Functional Exploration
UI exploration is triggered when the LLM encounters uncertainty r egarding which UI
component to interact with. Therefore, BugRepro begin the explo ration by analyzing
the UTG starting from the page where the diﬃculty arises, denoted asp. The UTG,
created by the UI Automator, provides essential information abo ut the app, including
the relationships between UI states and the presence of various U I elements on each
screen. By examining the functionalities of these elements, we can u ncover the tasks
that can be performed within the app and identify the UI component s necessary to
execute them. Therefore, BugRepro parses the UI states and e lements in the UTG,
querying LLMs to extract their respective functions.
11

In this context, the UTG is modeled as a directed graph, where node s represent UI
states and edges represent actions, both recorded by the rand om Explorer. For each
UI state Ui, BugRepro queries the LLM to summarize the functionalities of all UI
elements eiwithin that state. Notably, BugRepro only extracts the functiona lity of an
element from the UI state closest to the initial UI if it appears in multip le UI states.
Table5showstheconstructionofthesynthesizedfunctionalitytable.Up ontravers-
ing the entire UTG, BugRepro compiles a synthesized functionality ta ble in the
app-related knowledge, which contains nentries, where nis the total number of UI
elements in the page p. Each entry corresponds to a UI element eiand includes three
components: <Synthesized functionality, UI states, UI elements> .
Table 5: A portion of the Synthesized Functionality Table
Problem Page Other UI States
Current
ActivityElementsCurrent
ActivityElements Synthesized Functionality
MainFlash
ActivityINFOWelcome
ActivityTextView/Static Text,
Button,
Status Bar,
Navigation BarThis is the “About LibreNews”
screen in the LibreNews app. It pro-
vides information about the app’s
features and purpose, including
that it oﬀers ad-free, decentralized,
and secure breaking news notiﬁ-
cations. The screen explains that
LibreNews is open source, doesn’t
track users, and primarily inter-
acts with users through notiﬁca-
tions (about 3 per day with default
settings). The main actionable ele-
ment is a “GO TO LIBRENEWS”
button at the bottom of the screen.
MainFlash
ActivityREFRESHWelcome
ActivityTextView/Static Text,
Button,
Status Bar,
Navigation BarThis screen displays the “About
LibreNews” page within the Libre-
News application. It provides infor-
mation about the app’s features
including being ad-free, decentral-
ized, secure breaking news notiﬁca-
tions, and its open-source nature.
The page contains several text
sections explaining the purpose and
beneﬁts of LibreNews, and a promi-
nent “GO TO LIBRENEWS” but-
ton at the bottom that is click-
able. The system UI elements vis-
ible include the status bar (with
time, wiﬁ, mobile signal, and bat-
tery indicators) and the navigation
bar (with back, home, and recent
apps buttons).
Note: This table presents a subset of the Synthesized Functi onality Table. Here, the Problem
Pagedenotes the UI page where the issue was initially observed. T he corresponding Current
Activity speciﬁes the Android Activity associated with that page. Elements refer to individual
UI components present on the page. Through Functionality-a ware UI exploration, we identify the
subsequent UI states triggered by interacting with these elements from the Probl em Page, and
extract their corresponding New Activity, Elements, and the Synthesized Functionality as
inferred by the LLM.
The “Synthesized functionality” represents the task associated withei, as summa-
rized by the LLM, which can be accomplished by interacting with this ele ment. “UI
12

elements” refers to the elements that are clicked to navigate from the initial UI state
toUi, and “UI states” represent the sequence of UI states travers ed during this pro-
cess. This table provides the LLM with the necessary information to determine the
actions, facilitating more eﬃcient planning.
In addition to the synthesized functionality table, the app-related knowledge also
includes a UI function table, which summarizes the functionality of ea ch UI state in
the UTG. This information is derived by querying the LLM to describe t he function
of each UI state in the graph.
3.2.2 Exploration-enhanced Prompting
Typically, to match the extracted S2R entities to a sequence of GUI events for
bug reproduction, a common solution is to use lexical computation to match the
extracted components against the displayed text of the UI compo nents on the cur-
rent screen ( Fazzini et al. 2018 ;Zhao et al. 2019 ;Feng and Chen 2024 ). However,
this approach can be inaccurate due to the absence of textual de scriptions or vague
bug reports ( White et al. 2019 ;Cooper et al. 2021a ). To overcome this limitation, we
utilize LLMs to generate dynamic guidance on the GUI screen, enablin g automatic
reproduction of the steps.
Speciﬁcally,webeginbyprovidingLLMswithinstructionsaboutthebu greproduc-
tion task. These instructions serve as a foundational guide for th e LLM, outlining the
objective, workﬂow, and detailed steps for reproducing Android b ugs. Next, we sup-
ply the LLM with the current bug report, the required S2R entities, and the encoded
text of the current UI screen, allowing the LLM to determine which U I component
to interact with and what action to perform. Speciﬁcally, our promp t consists of the
following components: (1) the bug report text, (2) the extracte d S2R entity, and (3)
the encoded text of the current UI screen.
However, when the LLM struggles to make the correct decision—su ch as when it
cannot determine which UI component to interact with, or when exp loration based on
the LLM’s guidance fails to reproduce the bug and the LLM does not u pdate its deci-
sion after revisiting the page—we provide additional app-related kn owledge obtained
from the previous exploration to assist the LLM. Speciﬁcally, given a problematic
page, we supply the LLM with the functionality of all UI components o n that page
and the functionalities of the new UI screens they can navigate to. This additional
information enables the LLM to decide which UI component to interac t with and what
action to take. At this stage, the prompt includes the following comp onents: (1) the
bug report text, (2) the extracted S2R entity, (3) the encoded text of the current UI
screen, and (4) the app-related knowledge, which includes the syn thesized function-
alities ( §3.2.1) of each UI component within current UI page, and the UI function
table, which includes a mapping of functionalities associated with UI pa ges accessible
through speciﬁc components on the current page.
3.2.3 Interpreting the response
Due to the inconsistent output formats of LLMs in prior work, it has been challenging
to bridge the gap between LLM-generated responses and struct ured executable code.
13

To address this issue, we adopt the LangChain framework to enfor ce a standardized
output structure from the LLM.
Table6illustrates the form a response may take. During each iteration of t he
bug reproduction process, the system relies on the LLM’s respons e to execute actions
on the current page. The response is consistently represented a s a JSON-formatted
array, regardless of whether it contains a single action primitive or a sequence of
multiple actions.Eachaction primitive includes essentialﬁelds such as the action type,
the target UI component (represented as feature), and, when applicable, additional
parameters like input text, direction, or duration.
Table 6: Example of a JSON-formatted action sequence gener-
ated by LLM
Single Action Primitive[
{
“action”: “click”,
“feature”: “REFRESH”
}
]
Sequences of Action Primitives[
{
“action”: “set text”,
“feature”: “https://librenews.io/api”,
“inputtext”: “xxyyzz”
},
{
“action”: “click”,
“feature”: “OK”
}
]
These generated actions are parsed using regular expressions ba sed on predeﬁned
patterns. After interpreting the generated actions, the syste m executes them and
provides immediate feedback to the LLM regarding the execution st atus—indicating
whether the actions were successfully carried out. This feedback enables the LLM to
assess the current UI state and decide whether to proceed to th e next step or revise its
response in case ofa failed execution. The reproductionloop contin ues iterativelyuntil
either the bug is successfully reproduced or the process terminat es due to exceeding
the allotted time budget.
After interpreting the generated actions, BugRepro executes t hem and provides
feedback to the LLM regarding the execution status, indicating wh ether the actions
were successfully carried out. This feedback allows the LLM to asse ss the current
status and decide whether it is appropriate to proceed to the next step or if the
response needs to be reformulated due to a failed execution. The r eproduction process
continues iteratively until either a successful reproduction is ach ieved or the bug fails
to be reproduced within the allocated time budget.
14

4 Experimental Setup
To evaluate BugRepro, we consider four key research questions:
•RQ1: How accurate is BugRepro in extracting S2R entities? This inves-
tigates the performance of our S2R entity extraction approach, emphasizing the
impact of RAG and few-shot examples. Comparisons are made with st ate-of-the-
art baselines to highlight BugRepro’s advantages. For more details, please refer to
Section5.1.
•RQ2: How eﬀective and eﬃcient is BugRepro in reproducing bug
reports? This examines the ability of our approach to reproduce crashes fro m bug
reports within a predeﬁned time limit, emphasizing both success rate and eﬃciency.
For more details, please refer to Section 5.2.
•RQ3: How do individual components impact performance when
removed? Through ablation studies, this explores the contribution of each inn o-
vation in our approach to S2R entity extraction and bug reproduct ion. For more
details, please refer to Section 5.3.
•RQ4: How does the robustness of each method vary across diﬀere nt
language models? This investigates the eﬀect of using diﬀerent LLMs (e.g., GPT-
4, DeepSeek) on the performance of our approach. For more det ails, please refer to
Section5.4.
4.1 Datasets
To construct our dataset, we follow established practices for colle cting real-world
bug reports for reproduction studies ( Zhao et al. 2019 ;Su et al. 2021 ;Huang et al.
2023;Cooper et al. 2021a ). Speciﬁcally, to minimize potential bias, we source bug
reports from four well-known open-source datasets: (i) the eva luation dataset of ReC-
Droid (Zhao et al. 2019 ); (ii) the evaluation dataset of ScopeDroid ( Huang et al.
2023); (iii) the empirical study dataset from AndroR2 ( Wendland et al. 2021 ); and
(iv) another empirical study on Android bug report reproduction ( Su et al. 2021 ).
Given the overlap across these datasets, we ﬁrst eliminate duplicat e entries, ensur-
ing that no bug reports from the same issue repository were count ed multiple times.
While users often include visual elements such as screenshots or vid eos in their bug
reports ( Bernal-C´ ardenas et al. 2020 ;Cooper et al. 2021b ,a;Fang et al. 2021 ;Feng
and Chen 2022 ), our focus in this work is on textual information, speciﬁcally natur al
language descriptions of S2Rs. Consequently, we manually review th e bug reports and
retain only those containing textual S2Rs. After these ﬁltering st eps, our ﬁnal experi-
mental dataset comprises 151 bug reports. This curated collectio n serves as a reliable
foundation for evaluating our approach.
4.2 Implementation and Environment
We implement BugRepro in Python. We leverage all-MiniLM- L12-v2 mode l as the
retrieval model used in the RAG component, which is based on the De nse Passage
Retrieval (DPR) ( Karpukhin et al. 2020 ) architecture and uses a bi-encoder to encode
the queries and documents into dense vectors. We utilize the DeepS eek-V3 model
15

as the underlying LLM in BugRepro, accessed via its API service ( DeepSeek 2024 ).
To address potential verbosity in the LLM’s output—such as repea ted questions or
chain-of-thought reasoning—we adopt a ﬁltering mechanism that r emoves non-JSON
content, retaining only valid JSON segments enclosed in square brac kets. For inter-
acting with UI widgets on the device, we leverage UI Automator2 ( Netease and
Contributors 2025 ) as the execution engine. We set the UTG exploration depth to 1
and the time budget to 5 minutes to balance eﬀectiveness and eﬃcien cy. All experi-
ments are performed on a workstation running Windows 11 Pro 64-b it, equipped with
a 13th Gen Intel(R) Core(TM) i7-13700KF CPU (24 cores, base fre quency 3.4GHz)
and an NVIDIA GeForce RTX 4090 GPU.
4.3 Baselines
We selecttwostate-of-the-artmethods widelyrecognizedforan droidbug reproduction
to compare against BugRepro: 1) ReCDroid (Zhao et al. 2019 ): This is a traditional
state-of-the-art method that analyzes dependencies among wo rds and phrases from
hundreds of bug reports and uses 22 predeﬁned grammar patter ns to extract S2R
entities. For example, a noun phrase (NP) followed by a “click” action is interpreted
as the target component. We adopt their released repository for evaluation. Although
ReCDroid+( Zhaoet al.2019 )is anextended versionofReCDroid,it primarilyfocuses
on scraping bug reports from issue tracking websites, which falls ou tside the scope of
this study. We use ReCDroid in this work for simplicity. 2) AdbGPT (Feng and Chen
2024): This is a state-of-the-art LLM-based method leveraging few-s hot learning and
chain-of-thought reasoning to extract S2R entities from bug rep orts eﬀectively.
TounderstandthecontributionsofindividualcomponentsinBugRe pro,wedevelop
two ablation variants of BugRepro: 1) BugRepro nor: This variant omits the retrieval
model and relies solely on the LLM-based generation model to produ ce actionable
operations. 2) BugRepro nou: This variant excludes the UI exploration process and
retrieves S2R entities directly from the database using the retriev al model alone.
To evaluate the generalization of BugRepro, we replace the default LLM
(DeepSeek-V3) with GPT-4, naming this variant BugRepro gpt. This allows us to
assess how diﬀerent LLMs aﬀect the robustness and performanc e of BugRepro.
4.4 Metrics
Following existing work ( Feng and Chen 2024 ), we measure the eﬀectiveness of S2R
entity extraction using the accuracy metric. An extracted S2R is deemed accurate if
all of the following match the ground truth, including steps (i.e., step ordering, sub-
step separation) and entities (i.e., the action types of each step, t he possible target
components if existed, the possible input values, and the possible sc roll directions).
A higher accuracy score reﬂects an approach’s superior ability to c orrectly interpret
and extract the steps-to-reproduce (S2R) entity from bug rep orts. To evaluate the
eﬀectiveness of reproducing bugs, we use the number of successful reproductions
(NSR). A higher NSR indicates better performance in replicating the S2Rs o n GUIs
to successfully trigger the reported bugs. To assess the eﬃcienc y of each technique,
16

Table 7: Accuracy Comparison on S2R Entity Extraction
Method Step Action Component Input Direction
Recdroid 63.02% 45.50% 4.98% 52.00% 0.00%
AdbGPT 76.16% 61.92% 28.77% 62.08% 73.12%
Ours 87.85% 69.49% 43.24% 70.93% 82.91%
Note: This table shows the accuracy comparison of diﬀerent meth ods for S2R entity extraction across
various categories.
we measure the average time taken for successful reproductions. Less time indi-
cates greater eﬃciency in reproducing the bugs, highlighting the me thod’s practical
applicability in real-world scenarios.
5 Results
5.1 RQ1: How accurate is BugRepro in extracting S2R
entities?
To answer RQ1, we evaluate the accuracy of BugRepro in extractin g S2R entities
compared to the state-of-the-art baselines. Table 7presents the comparison results of
BugRepro, AdbGPT, and ReCDroid across ﬁve dimensions: step ext raction, action
types, target components, input values, and action direction.
From the table, BugRepro outperforms both ReCDroid and AdbGPT across all
dimensions, with accuracy improvements ranging from 8.85% to 14.47 % compared to
AdbGPT. For instance, BugReprorecords69.49%accuracyin extr acting action types,
compared to 61.92% by AdbGPT and 45.50% by ReCDroid. Similarly, in iden tify-
ing target components, BugRepro achieves 43.24%, outperformin g AdbGPT (28.77%)
and ReCDroid (4.98%). This trend persists across input values and s croll directions,
underscoring BugRepro’s consistent advantage.
Through our analysis, there are three main reasons that BugRepr o outperforms
AdbGPT and ReCDroid:
•Enhanced Contextual Understanding: These improvements demo nstrate BugRe-
pro’sabilityto overcomelimitationsinherent in traditionalgrammar-b asedmethods
(ReCDroid) and static few-shot learning approaches (AdbGPT). T he RAG compo-
nent is pivotal in achieving this performance, enabling the retrieval of task-relevant
examples to provide richer context for the LLM. The use of RAG allow s BugRepro
to bridge the gap between generic LLM capabilities and the domain-sp eciﬁc needs
of S2R extraction.
•Precision in Multi-Step Scenarios: For bug reports requiring multi-st ep actions,
BugReproexcelsin maintainingthe correctsequence andsub-step separation,which
are critical for accurate S2R entity extraction. In contrast, Ad bGPT often struggles
with multi-step reasoning due to its reliance on few-shot learning with out dynamic
retrieval.
17

•RobustnesstoLinguisticVariations:BugReprodemonstratesahig hdegreeofadapt-
ability to diﬀerent writing styles, including reports with incomplete or c olloquial
descriptions. This ﬂexibility is attributed to the diverse examples pro vided by RAG,
which act as a bridge to align the model’s predictions with the intended s emantics.
RQ1 summary: Integrating RAG into our approach clearly demonstrates that
BugRepro outperforms baseline methods in S2R extraction, achiev ing accuracy
improvementsof8.85%to14.47%overAdbGPT(state-of-the-art baselines)across
all dimensions. This fully conﬁrms that RAG not only provides context ual-
ized examples, enabling BugRepro to eﬀectively handle ambiguous or c omplex
bug reports, but also enhances contextual understanding, pre cision in multi-step
scenarios, and robustness to linguistic variations.
5.2 RQ2: How eﬀective and eﬃcient is BugRepro in
reproducing bug reports?
To answer RQ2, we evaluate the eﬀective: NSRand eﬃciency of BugRepro in repro-
ducing bugs compared to baselines. Table 8provides the comparison results. From
the table, BugRepro achieves an NSR of 96/151, signiﬁcantly higher than AdbGPT
55/151 and ReCDroid 38/151.
Table 8: NSR and Extract Time in Reproducing Crashes
Method NSR Average Time
ReCDroid 38 534.9s
AdbGPT 55 125.6s
Ours 96 124.7s
BugRepro’s advantage is particularly evident in scenarios involving co mplex app
workﬂows or ambiguous S2Rs. One common limitation in existing method s is their
reliance solely on the content of bug reports for S2R entity extrac tion, often neglect-
ing the domain-speciﬁc context required for precise interpretatio n. This can result in
inaccuracies during the extraction phase. By incorporating RAG, B ugRepro enriches
the LLM’s understanding with task-relevant knowledge, eﬀectively addressing these
challenges and improving accuracy.
Moreover, existing approaches, such as AdbGPT and ReCDroid, fo llow strictly
sequential steps within the current screen, focusing on locating s peciﬁc UI widgets.
These methods rarely consider transitioning to a diﬀerent screen u nless explicitly
guided bypredeﬁnedheuristics,leadingtolimited explorationcapabilit ies.In contrast,
BugRepro integrates UTG to provide a broader understanding of t he app’s struc-
ture. This enables it to navigate across multiple screens dynamically a nd equips the
LLM with synthesized app-speciﬁc knowledge from this process, sig niﬁcantly reducing
redundant exploration and improving precision.
18

For instance, in a case requiring navigation through three app scre ens before
encountering a bug, BugRepro eﬀectively utilizes its synthesized ap p knowledge to
streamline the reproduction process. On the other hand, AdbGPT and ReCDroid
frequently revisit previously explored states or fail to identify the correct sequence,
leading to lower success rates.
The integration of domain-speciﬁc knowledge with comprehensive UI contextual
analysis enables BugRepro to exhibit superior performance in crash reproduction sce-
narios. The empirical evaluation demonstrates that BugRepro suc cessfully reproduces
96 crashcases, representinga 152.63%improvementoverReCDro id (38) and a 74.55%
improvement over AdbGPT (55) in reproduction capability. In terms of temporal
eﬃciency, BugRepro completes reproduction in 124.7 seconds on av erage, which is
4.29 times faster than ReCDroid (534.9s) and marginally 0.7% faster t han AdbGPT
(125.6s).ThisperformancemakesBugReproanoptimalsolutionfo rsoftwareengineers
engaged in debugging and quality assurance processes, where bot h comprehensive
crash reproduction and time eﬃciency are critical factors.
RQ2 summary: BugRepro signiﬁcantly outperforms baselines in bug repro-
duction success rate and eﬃciency, achieving the NSR of 96 and an a verage
reproductiontimeof124.7secondsperbug.Theseresultshighlight theimportance
of integrating RAG and UTGs, which enable BugRepro to resolve ambig uities,
navigate complex app workﬂows, and minimize redundant exploration . BugRepro
oﬀers a practical and eﬃcient solution for real-world debugging tas ks.
5.3 RQ3: How do individual components impact performance
when removed?
To answer RQ3, we conduct ablation studies by removing the RAG com ponent and
the UI exploration mechanism independently to assess the contribu tion of BugRepro’s
core components. This analysis quantiﬁes the individual impact of th ese components
on BugRepro’s overall performance in bug reproduction tasks. Ta ble9presents the
results, showinga signiﬁcantdecline in number ofsuccessful repro duction (NSR) when
either component is excluded. The ﬁndings highlight the complementa ry roles of RAG
and UI exploration in enabling BugRepro to extract accurate S2Rs a nd eﬀectively
reproduce bugs.
Table 9: Comparison between BugRepro and its variants
NSR Average Time Average Response Time1
BugRepro nor 80 75.2s 5.99s
BugRepro nou 76 62.7s 5.64s
Ours 96 124.7s 9.37s
1Average Response Time represents the total time spent by the LLM to generate responses during
the crash reproduction process.
19

Without RAG, BugRepro nor’s NSR drops to 80/151, compared to the BugRe-
pro’s 96/151. While the average reproduction time decreases signiﬁ cantly to 75.2
seconds from 124.7seconds, this eﬃciency comes at the cost of ac curacy, with 15 fewer
successful reproductions.
TheabsenceofRAGremovestheretrievaloftask-relevantexam ples,whicharecrit-
ical for enhancing the LLM’s contextual understanding during S2R entity extraction.
This deﬁciency becomes particularly evident in linguistically complex or a mbiguous
instructions. For example, phrases such as “swipe left in the settin gs” are frequently
misinterpreted as [Swipe] [Left].
However, with the support of retrieved examples, such as “press the settings icon
and rotate the phone,” where the extracted S2R entities are [Tap] [Settings] and
[Rotate], the model gains the contextual understanding needed t o correctly interpret
the instruction. Consequently, it is more likely to extract the intend ed S2R sequence
as [Tap] [Settings] followed by [Swipe] [Left].
Similarly, instructions involving conditional steps or implicit transitions are less
likely to be correctly parsed, resulting in incomplete or incorrect S2R s and subsequent
reproductionfailures.ThefasterLLMresponsetime of5.99scomp aredto9.37sfurther
conﬁrms that without RAG, the model performs less complex reaso ning but produces
less accurate results.
The removal of UI exploration also signiﬁcantly impacts performanc e, with
BugRepro noureducing the NSR to 76/151 and decreasing the average reproduc tion
time to 62.7 seconds. This highlights the critical role of app-speciﬁc k nowledge in
achieving reproduction accuracy, even though the process may e xecute more quickly
without UI exploration overhead.
UI exploration enables BugRepro to construct and leverage UTGs, which map the
dynamic states and interactions within an app. Without this mechanis m, BugRepro is
limited to interpreting S2Rs in isolation, often failing to navigate multi-s creen work-
ﬂows. For instance, reproducing a bug that requires traversing t hrough nested menus
before triggering the issue becomes signiﬁcantly more challenging wit hout a transition
graph to guide the process. This limitation leads to a higher likelihood of failure as
the system lacks the contextual understanding of app structur e.
The absence of UI exploration also impacts the LLM’s ability to resolve ambigui-
ties in S2R instructions. Without app-speciﬁc knowledge to context ualize the replay
process,BugReprostruggleswithtasksrequiringnuancedunder standingofappbehav-
ior, such as determining the correct sequence of actions across m ultiple screens. The
slightlyreducedLLMresponsetimeof5.64s(comparedto9.37s)indic ateslesscomplex
processing but at the expense of reproduction success.
The results conﬁrm that RAG and UI exploration are not only individua lly essen-
tial but also synergistic in enabling BugRepro to achieve optimal perf ormance. RAG
enhances the LLM’s understanding during S2R extraction by provid ing contextu-
ally relevant examples, while UI exploration complements this by guiding the replay
process with app-speciﬁc knowledge. Together, these componen ts address the diverse
challenges inherent in bug reproduction, including ambiguous instruc tions, dynamic
app states, and complex workﬂows, even though they increase th e overall processing
time to 124.7 seconds and LLM response time to 9.37 seconds.
20

Table 10 : Results Across Diﬀerent LLMs
S2R Entity Extraction
Method Step Action Component Input Direction
Ours (gpt-4) 87.61% 69.60% 33.76% 72.16% 83.04%
Ours (deepseek) 87.85% 72.32% 37.60% 74.24% 82.40%
Replay Performance
Method NSR Average Time Average Response Time
Ours (gpt-4) 86 83.0s 8.5s
Ours (deepseek) 96 124.7s 9.3s
RQ3 summary: The ablation study demonstrates the critical contributions
of RAG and UI exploration to BugRepro’s performance. RAG signiﬁca ntly
enhances S2R extraction accuracy by providing task-relevant ex amples, while UI
explorationenablespreciseandeﬃcientnavigationofcomplexappst ates.Thesyn-
ergisticrelationshipbetweenthesecomponentsallowsBugReproto addressdiverse
challenges in bug reproduction tasks, establishing their necessity f or achieving
state-of-the-art results.
5.4 RQ4: How does the robustness of each method vary across
diﬀerent language models?
To answer RQ4, we evaluate BugRepro’s performance using both GP T-4 and our
default DeepSeek model to assess its generalization capability. Tab le10presents
the results, demonstrating consistent S2R entity extraction per formance across both
models, with DeepSeek showing slight advantages in Action (72.32% vs . 69.60%)
and Component (37.60% vs. 33.76%) extraction. For replay perfor mance, DeepSeek
achieves a higher NSR of 96 compared to GPT-4’s 86, though with a lon ger average
reproduction time (124.7s vs. 83.0s).
These ﬁndings highlight BugRepro’s model-agnostic architecture, w hich integrates
RAG and UI exploration to enhance LLM capabilities. By transforming generic LLMs
into specialized bug reproduction tools, BugRepro reduces depend encies on speciﬁc
model features, enabling robust performance across diﬀerent L LMs.
Through detailed analysis, we observe that while both models show co mparable
S2R extraction accuracy, their performance diﬀers in key areas. DeepSeek demon-
strates superior bug reproduction capability with higher NSR, thou gh it requires more
processing time, which is reﬂected in its slightly higher average respo nse time (9.3s
vs. 8.5s). This also indicates that GPT-4 is eﬃcient in rapid debugging s cenarios,
but its lower NSR indicates that it may encounter diﬃculties when hand ling certain
complex application interactions. In contrast, DeepSeek’s higher s uccess rate in bug
reproduction indicates better performance in translating extrac ted S2R entities into
eﬀective app interactions, especially for complex scenarios involving multiple steps or
conditional logic.
21

RQ4 summary: Results demonstrate BugRepro’s ability to generalize across
LLMs,with DeepSeekachievinghigherreproductionsuccess(96vs .86)atthe cost
of longer execution time (124.7s vs. 83.0s). This ﬂexibility highlights Bu gRepro’s
adaptability to diverse LLMs and suggests opportunities for hybrid approaches
that leverage the strengths of diﬀerent models.
6 Discussion
The primary construct threats stem from the selection of evaluat ion metrics and
the inherent randomness in LLM outputs. LLMs frequently produc e unstructured
responses, which impedes their integration with structured code a nd automation
pipelines. To address these challenges, we initially adopt approaches from prior
work (Feng and Contributors 2025 ), which attempt to enforce output formatting
through prompt examples (e.g., “1. [Tap] [‘bookmark’]”). However, s uch eﬀorts have
not fully standardized the responses. As illustrated in Table 11, LLM outputs often
display inconsistencies in format, structure, and verbosity, ther eby complicating
automation and diminishing accuracy.
Our approach employs LangChain to enforce output constraints a nd ensure stan-
dardized formatting. The standardized outputs are shown in Table 6. The reﬁnement
of the output format further boosts the performance of BugRe pro, enforces structured
action output an ensure consistency and automation readiness.
Table 11 : Examples of unstructured LLM outputs from prior work
Example output 1: ### Suggestion:
1.Tap on the “share dialog” widget .
2.Switch apps .
Example output 2: Suggestion:
1. Install the app “org.mozilla.rocket.debug.ting” if not a lready installed.
2. Launch the app after installation.
Example output 3: Suggestion:
1.Action: Click the “Skip” >>button to proceed.
-Feature: “Skip”>>’
-Action: “click”
Example output 4: Suggestion:
1.Explore the “Licenses” section : Check for relevant options.
[{“action”: “click”, “feature”: “Licenses” }]
2.Scroll down if no options are found.
[{“action”: “scroll”, “target direction”: “down” }]
3.Back to main screen if needed.
[{“action”: “back” }]
Example output 5: ### Suggestion:
1.Click on the “LOG IN” button .
- Action: click
- Feature: LOG IN
The internal threats mainly lie in potential bugs in our implementations . To mit-
igate this, we leverage the provided artifacts for ReCDroid and Adb GPT to ensure
22

correctness. For BugRepro, two authors independently review a nd rigorously test the
code.
The external threats mainly lie in the generalizability of the dataset.I n our exper-
iments, we collect all released bug reports from existing works and a ll of them are
real-world bugs, which can be representative to some degree. How ever, the eﬀective-
ness of BugRepro on a wider range remains to be evaluated in the fut ure. Moreover,
the conclusions derived from using a speciﬁc LLM (e.g., DeepSeek) ma y not general-
ize to other LLMs. To address this, we conduct additional experime nts with GPT-4,
an advanced LLM, and observe promising and consistent results.
7 Related Work
7.1 Traditional Android Bug Reproduction.
Existing traditional Android bug reproduction technologies such as Yakusu ( Fazzini
et al. 2018 ) and ReCDroid ( Zhao et al. 2019 ), Yakusu, the ﬁrst method to propose
integrating bug reports with bug reproduction, consists of three modules: ontology
extraction, bug report analysis, and executable action search. T he ontology extrac-
tion module analyzes application UI-related ﬁles to build an ontology th at supports
mapping. The bug report analysis module uses natural language pro cessing tech-
niques (Mikolov et al. 2013 ) to extract abstract steps and address logical gaps. The
UI action search module takes abstract steps as input, dynamically searches and gen-
erates test cases, employing a depth-ﬁrst search strategy ( Choi et al. 2013 ) to handle
diﬀerent abstract steps. Each module leverages diﬀerent technic al tools to achieve its
functionality and generate speciﬁc format outputs. ReCDroid, co nsists of two main
phases: crash report analysis and dynamic exploration. The forme r uses 22 predeﬁned
grammarpatterns and deep-learning NLP techniques ( Honnibal and Montani 2017 ) to
map sentences into semantic representations. The latter phase is based on DOET and
DFS, combined with Q-learning ( Zhao et al. 2019 ) to optimize matching and complete
steps, generating the shortest event sequence. Existing appro aches for integrating bug
reports with reproduction processes suﬀer from several limitatio ns. Many rely heav-
ily on UI elements and general-purpose word embedding models, which restrict their
ability to capture application-speciﬁc semantics and contextual nu ances. Additionally,
methods based on predeﬁned grammar patterns often struggle t o accurately inter-
pret diverse and dynamic GUI components, leading to poor generaliz ation and limited
adaptability across diﬀerent applications.
7.2 Multimodal Bug Reproduction.
To bridge the gap between language parsing and practical applicatio n components, a
new method called ScopeDroid, based on multimodal processing ( Huang et al. 2023 ;
Fang et al. 2021 ;Yang et al. 2017 ), is proposed. ScopeDroid consists of three modules:
STG (State Transition Graph) construction and information extra ction, fuzzy repro-
duction step matching ( Liu et al. 2016 ;Robertson et al. 1995 ), and path planning
with STG expansion. During construction, Droidbot ( Project and Contributors 2025 )
is used to build the STG and extract component information. In the m atching phase,
23

a multimodal network structure is employed to calculate matching sc ores. In the path
planning and expansion phase, the algorithm selects the optimal pat h (Huang et al.
2023) for execution or guided exploration. Additionally, since the STG may be incom-
plete, newly encountered components are incorporated, and the matrix is updated to
re-plan the exploration. ScopeDroid has the following limitations: it on ly considers
the context of the current UI interface, cannot perform joint c ontext matching for
the entire report, and assumes that a single interaction between r eproduction steps
and GUI components may not hold, making it incapable of understand ing steps that
require reasoning or speciﬁc prior knowledge.
Compared to these methods, our approach has unique advantage s. It leverages
the exceptional natural language understanding capabilities of LL Ms while closely
integrating the context of bug reports and detailed information fr om the application
interface, thereby providing a more eﬀective and comprehensive s olution to the related
issues.
7.3 LLM-based Android Bug Reproduction.
LLMs demonstrate excellent performance in natural language pro cessing, which leads
to the proposal of the method AdbGPT. AdbGPT entails the manual creation of
ﬁfteen BRs ( Feng and Contributors 2025 ) annotated with labels, which are utilized
as exemplars. During subsequent processing, for each BR, the mo st appropriate 1 to
3 exemplars are selected to enable few-shot learning ( Wang et al. 2020 ) in large-scale
LLMs, which helps improve accuracy.
ReBL adapts general-purpose LLMs for bug reproduction. It use s activity names
for context and groups UI widgets to enhance understanding. It handles single/multi-
pleactions,providesfeedback(e.g.,executionstatus,sequence detection), andemploys
summarization to manage token limits. Success is judged by linking err or symptoms
to UI info; exploration stops if reproduction fails.
However, during the experimental process, we ﬁnd that these tw o methods are
highly dependent on speciﬁc LLM models, and changing the model has a signiﬁcant
impact on performance. At the same time, while LLMs are currently a t the forefront
of text processing BugRepro, their limitations are widely documente d by numerous
scholars ( Wang et al. 2020 ;Wei et al. 2022 ;Martino et al. 2023 ;Ji et al. 2023 ).Among
these, the most critical challenge arises when applying LLMs to the S 2R and repro-
duction approach stages, particularly due to the “hallucination” ( Martino et al. 2023 ;
Bang et al. 2023 ;Chang et al. 2024 ) problem. As the existing LLMs are not speciﬁ-
cally trained to adapt to the analysis of bug reports, meanwhile, it is w ell-known that
when posing questions to LLMs, the likelihood ofgenerating”hallucina tion”responses
increases in the absence of suﬃcient prior knowledge in the corresp onding domain
(Sahoo et al. 2024 ;Gao et al. 2023 ;Just et al. 2018 ;Varshney et al. 2023 ;Wu et al.
2024;Zhang et al. 2022 ), leading to inaccurate outputs during the S2R stage, which
causes the subsequent reproduction approach to function incor rectly, and the hallu-
cination issue may further lead to erroneous exploration in the repr oduction process.
Addressing these challenges is essential to ensure reliable and accu rate outputs. The
RAG approachdemonstratessuperiorperformancein enhancingt he accuracyandreli-
ability of generated content through its eﬀective utilization of exte rnal knowledge. In
24

contrast to methods ( Zhao et al. 2019 ) that rely exclusively on pre-trained language
models, RAG operates by: (1) retrieving relevant text fragments from a knowledge
repository in response to user queries, and (2) incorporating the se fragments into a
language generation model (e.g., GPT) to produce the ﬁnal output . This combined
retrieval-generation mechanism provides the language model with m ore precise and
contextuallyrelevantinformation,thereby:(i) substantiallymitiga tingthe riskof“hal-
lucinations” (instances where models generate factually incorrect content), and (ii)
consistently improving the overall quality of generated outputs. B y integrating exist-
ing RAG technology into the method and striving to provide the LLM wit h more
context information closely related to the app under test, we succ essfully enhance the
robustness of the method when switching between diﬀerent LLMs.
In the process of automatically reproducing bugs based on bug rep ort descrip-
tions, previousmethods haveexhibited signiﬁcant limitations. Our ap proachaddresses
these shortcomings through targeted improvements to existing t echniques, resulting
in substantial performance gains.
8 Conclusion
In conclusion, this paper proposes a novel method, named BugRep ro, to address the
challenges of unclear or incomplete S2Rs in bug reports and the inher ent complex-
ity of modern apps. By integrating domain-speciﬁc knowledge and lev eraging a RAG
method alongsideapp-speciﬁc GUI exploration,BugReproenhance s the accuracy,suc-
cess rate, and eﬃciency of bug reproduction. Extensive experime ntal results validate
BugRepro’s superiority over state-of-the-art methods, achiev ing signiﬁcant improve-
ments across multiple metrics. This highlights BugRepro’s potential t o streamline
debugging processes and maintain high-quality user experiences in m obile application
development.
In the future, BugRepro could be extended along several promisin g research direc-
tions to further advance its capabilities. First, the system’s conte xtual understanding
could be improved by incorporating developer feedback patterns, enabling more pre-
cise interpretation of ambiguous S2Rs and enhancing reproduction accuracy. Second,
a self-learning mechanism could be developed to iteratively reﬁne rep roduction strate-
gies based on successful outcomes, thereby improving long-term performance. Third,
extending the system’s cross-platform compatibility would allow bug r eproduction
across diverse operating systems and device conﬁgurations, bro adening its practi-
cal utility. Fourth, integrating predictive bug analysis could enable p roactive issue
detection, potentially identifying defects before they are formally reported. Finally,
an enhanced visualization system for reproduction steps could fac ilitate clearer com-
munication between QA teams and developers, streamlining the debu gging process.
These extensions would collectively improve BugRepro’s robustness , generalizability,
and usability in real-world development environments.
Data availability :The whole datasets generated and analyzed during the current
study are available from the corresponding author on reasonable r equest. Here is a
website that provides key code implementations and original datase ts:(Github 2025 ).
25

References
42matters: App Store and Google Play Statistics. Accessed: 2023 -10-05 (2023).
https://42matters.com/stats#available-apps-count
Aperdannier, R., Koeppel, M., Unger, T., Schacht, S., Barkur, S.K.: S ystematic eval-
uation of diﬀerent approaches on embedding search. In: Future o f Information and
Communication Conference, pp. 526–536 (2024). Springer
Anthropic: Claude 3.5 Sonnet. https://www.anthropic.com/news/claude-3-5-sonnet .
Accessed: [Insert Date] (2024)
Aranda, J., Venolia, G.: The secret life of bugs: Going past the error s and omissions
in software repositories. In: 2009 IEEE 31st International Conf erence on Software
Engineering, pp. 298–308 (2009). IEEE
Bernal-C´ ardenas, C., Cooper, N., Moran, K., Chaparro, O., Marcu s, A., Poshyvanyk,
D.: Translating video recordings of mobile app usages into replayable s cenarios.
In: Proceedings of the ACM/IEEE 42nd International Conferenc e on Software
Engineering, pp. 309–321 (2020)
Bang, Y., Cahyawijaya, S., Lee, N., Dai, W., Su, D., Wilie, B., Lovenia, H., J i, Z., Yu,
T.,Chung,W.,etal.:Amultitask,multilingual,multimodalevaluationofch atgpton
reasoning, hallucination, and interactivity. arXiv preprint arXiv:230 2.04023 (2023)
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., N eelakan-
tan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot
learners. Advances in neural information processing systems 33, 1877–1901 (2020)
Cooper, N., Bernal-C´ ardenas, C., Chaparro, O., Moran, K., Poshy vanyk, D.: It takes
two to tango: Combining visual and textual information for detect ing duplicate
video-based bug reports. In: 2021 IEEE/ACM 43rd Internationa l Conference on
Software Engineering (ICSE), pp. 957–969 (2021). IEEE
Cooper, N., Bernal-C´ ardenas, C., Chaparro, O., Moran, K., Poshy vanyk, D.: It takes
two to tango: Combining visual and textual information for detect ing duplicate
video-based bug reports. In: 2021 IEEE/ACM 43rd Internationa l Conference on
Software Engineering (ICSE), pp. 957–969 (2021). IEEE
Choi, W., Necula, G., Sen, K.: Guided gui testing ofandroidapps with min imal restart
and approximate learning. Acm Sigplan Notices 48(10), 623–640 (2013)
Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang,
C., Wang, Y., et al.: A survey on evaluation of large language models. ACM
Transactions on Intelligent Systems and Technology 15(3), 1–45 (2024)
Dav: Word2vec. Accessed: 2023-10-05 (2018). https://github.com/dav/word2vec
26

DeepSeek: DeepSeek. https://www.deepseek.com/ . Accessed: [Insert Date] (2024)
Duˇ sek, O., Kasner, Z.: Evaluating semantic accuracy of data-to- text generation with
natural language inference. arXiv preprint arXiv:2011.10819 (202 0)
Feng, S., Chen, C.: Gifdroid: Automated replay of visual bug report s for android apps.
In: Proceedings of the 44th International Conference on Softw are Engineering, pp.
1045–1057 (2022)
Feng, S., Chen, C.: Prompting is all you need: Automated android bug replay
with large language models. In: Proceedings of the 46th IEEE/ACM I nternational
Conference on Software Engineering, pp. 1–13 (2024)
Feng, S., Contributors: AdbGPT: Automating Android Device Inter actions with GPT
Integration. https://github.com/sidongfeng/AdbGPT .Accessed:2025-01-06(2025)
Fazzini, M., Prammer, M., d’Amorim, M., Orso, A.: Automatically translat ing bug
reports into test cases for mobile apps. In: Proceedings of the 27 th ACM SIGSOFT
International Symposium on Software Testing and Analysis, pp. 14 1–152 (2018)
Falke, T., Ribeiro, L.F., Utama, P.A., Dagan, I., Gurevych, I.: Ranking g enerated
summaries by correctness: An interesting but challenging applicatio n for natural
language inference. In: Proceedings of the 57th Annual Meeting o f the Association
for Computational Linguistics, pp. 2214–2220 (2019)
Fang, S., Xie, H., Wang, Y., Mao, Z., Zhang, Y.: Read like humans: Auton omous, bidi-
rectional and iterative languagemodeling for scene text recognitio n.In: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Reco gnition, pp.
7098–7107 (2021)
Galli, C., Donos, N., Calciolari, E.: Performance of 4 pre-trained sente nce transformer
models in the semantic query of a systematic review dataset on peri- implantitis.
Information 15(2), 68 (2024)
Github, A.: BugRepro: Enhancing Android Bug Repro-
duction with Domain-Speciﬁc Knowledge Integration.
https://anonymous.4open.science/r/YYyMUSTSTUDY-B889/REA DME.md .
Accessed: 2025-04-29 (2025)
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H .:
Retrieval-augmented generation for large languagemodels: A surv ey. arXiv preprint
arXiv:2312.10997 (2023)
Honnibal, M., Montani, I.: spacy 2: Natural language understanding with bloom
embeddings,convolutionalneuralnetworksandincrementalpar sing.Toappear 7(1),
411–420 (2017)
27

Huang,Y.,Wang,J.,Liu,Z.,Wang,S.,Chen,C.,Li,M.,Wang,Q.:Contex t-awarebug
reproduction for mobile apps. In: 2023 IEEE/ACM 45th Internatio nal Conference
on Software Engineering (ICSE), pp. 2336–2348 (2023). IEEE
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng , W.,
Feng, X., Qin, B., et al.: A survey on hallucination in large language models: Princi-
ples, taxonomy, challenges, and open questions. ACM Transaction s on Information
Systems 43(2), 1–55 (2025)
Jansen, B.J., Jung, S.-g., Salminen, J.: Employing large language models in survey
research. Natural Language Processing Journal 4, 100020 (2023)
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Mad otto, A.,
Fung, P.: Survey of hallucination in natural language generation. AC M Computing
Surveys55(12), 1–38 (2023)
Just, R., Parnin, C., Drosos, I., Ernst, M.D.: Comparing developer-p rovided to user-
provided tests for fault localization and automated program repair . In: Proceedings
of the 27th ACM SIGSOFT International Symposium on Software Te sting and
Analysis, pp. 287–297 (2018)
Karpukhin, V., O˘ guz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D ., Yih,
W.-t.: Dense passage retrieval for open-domain question answerin g. arXiv preprint
arXiv:2004.04906 (2020)
Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C., Berg, A .: Ssd: Single
shot multibox detector. in european conference on computer visio n. springer (2016)
Llama: Llama. https://www.llama.com/ . Accessed: [Insert Date] (2024)
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K ¨ uttler, H.,
Lewis, M., Yih, W.-t., Rockt¨ aschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in Neural Information Pr ocessing Systems
33, 9459–9474 (2020)
Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., Neubig, G.: Pre-train, p rompt, and
predict: A systematic survey of prompting methods in natural lang uage processing.
ACM Computing Surveys 55(9), 1–35 (2023)
Ma, X., Gong, Y., He, P., Zhao, H., Duan, N.: Query rewritingfor retrie val-augmented
large language models. arXiv preprint arXiv:2305.14283 (2023)
Martino, A., Iannelli, M., Truong, C.: Knowledge injection to counter la rge language
model (llm) hallucination. In: European Semantic Web Conference, p p. 182–185
(2023). Springer
28

Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., Dean, J.: Distribu ted repre-
sentations of words and phrases and their compositionality. Advan ces in neural
information processing systems 26(2013)
Nogueira, R., Cho, K.: Passage re-ranking with bert. arXiv preprint arXiv:1901.04085
(2019)
Netease, Contributors: uiautomator2: Android UI automation fr amework for Python.
https://github.com/openatx/uiautomator2 . Accessed: 2025-01-06 (2025)
OpenAI: GPT-4. https://openai.com/index/gpt-4/ . Accessed: [Insert Date] (2023)
Project,H., Contributors:DroidBot:A LightweightTest Input Ge neratorfor Android.
https://github.com/honeynet/droidbot . Accessed: 2025-01-06 (2025)
Park, Y., Shin, Y.: Adaptive bi-encoder model selection and ensemble for text
classiﬁcation. Mathematics 12(19), 3090 (2024)
Reimers, N., Gurevych, I.: all-MiniLM-L6-v2. Hugging Face (2024).
https://huggingface.co/sentence-transformers/all-MiniLM-L6 -v2
Rahman, M.M., Khomh, F., Castelluccio, M.: Works for me! cannot repr oduce–a large
scale empirical study of non-reproducible bugs. Empirical Softwar e Engineering
27(5), 111 (2022)
Robertson, S.E., Walker, S., Jones, S., Hancock-Beaulieu, M.M., Gatf ord, M., et al.:
Okapi at trec-3. Nist Special Publication Sp 109, 109 (1995)
Sahoo,P.,Singh,A.K.,Saha,S.,Jain,V.,Mondal,S.,Chadha,A.:Asyst ematicsurvey
of prompt engineering in large language models: Techniques and applic ations. arXiv
preprint arXiv:2402.07927 (2024)
Su, T., Wang, J., Su, Z.: Benchmarking automated gui testing for an droid against
real-world bugs. In: Proceedings of the 29th ACM Joint Meeting on E uropean
Software Engineering Conference and Symposium on the Foundatio ns of Software
Engineering, pp. 119–130 (2021)
Varshney,N., Yao,W., Zhang,H., Chen,J.,Yu, D.: Astitch intime save snine: Detect-
ing and mitigating hallucinations of llms by validating low-conﬁdence gene ration.
arXiv preprint arXiv:2307.03987 (2023)
White, T.D., Fraser, G., Brown, G.J.: Improving random gui testing wit h image-
based widget detection. In: Proceedings of the 28th ACM SIGSOFT International
Symposium on Software Testing and Analysis, pp. 307–317 (2019)
Wendland, T., Sun, J., Mahmud, J., Mansur, S.H., Huang, S., Moran, K., Rubin, J.,
Fazzini, M.: Andror2: A dataset of manually-reproduced bug repor ts for android
apps. In: 2021 IEEE/ACM 18th International Conference on Minin g Software
29

Repositories (MSR), pp. 600–604 (2021). IEEE
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zh ou,
D.,et al.: Chain-of-thought prompting elicits reasoning in large language mod els.
Advances in neural information processing systems 35, 24824–24837 (2022)
Wu, K., Wu, E., Zou, J.: How faithful are rag models? quantifying the t ug-of-war
between rag and llms’ internal prior. arXiv e-prints, 2404 (2024)
Wang, Y., Yao, Q., Kwok, J.T., Ni, L.M.: Generalizing from a few examples: A survey
on few-shot learning. ACM computing surveys (csur) 53(3), 1–34 (2020)
Wang, D., Zhao, Y., Feng, S., Zhang, Z., Halfond, W.G., Chen, C., Sun, X ., Shi, J.,
Yu, T.: Feedback-drivenautomatedwholebug reportreproductio nforandroidapps.
In: Proceedings of the 33rd ACM SIGSOFT International Symposiu m on Software
Testing and Analysis, pp. 1048–1060 (2024)
Yang, P., Fang, H., Lin, J.: Anserini: Enabling the use of lucene for info rmation
retrieval research.In: Proceedingsofthe 40th International ACM SIGIR Conference
on Research and Development in Information Retrieval, pp. 1253–1 256 (2017)
Zhao, Y., Su, T., Liu, Y., Zheng, W., Wu, X., Kavuluru, R., Halfond, W.G., Y u, T.:
Recdroid+: Automated end-to-end crash reproduction from bug reports for android
apps. ACM Transactions on Software Engineering and Methodology (TOSEM)
31(3), 1–33 (2022)
Zhao, Y., Yu, T., Su, T., Liu, Y., Zheng, W., Zhang, J., Halfond, W.G.: Rec droid:
automatically reproducing android application crashes from bug rep orts. In: 2019
IEEE/ACM 41st International Conference on Software Engineer ing (ICSE), pp.
128–139 (2019). IEEE
Zhang, Z., Zhang, A., Li, M., Smola, A.: Automatic chain of thought pro mpting in
large language models. arXiv preprint arXiv:2210.03493 (2022)
30
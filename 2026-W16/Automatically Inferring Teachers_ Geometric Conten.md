# Automatically Inferring Teachers' Geometric Content Knowledge: A Skills Based Approach

**Authors**: Ziv Fenigstein, Kobi Gal, Avi Segal, Osama Swidan, Inbal Israel, Hassan Ayoob

**Published**: 2026-04-15 09:34:46

**PDF URL**: [https://arxiv.org/pdf/2604.13666v1](https://arxiv.org/pdf/2604.13666v1)

## Abstract
Assessing teachers' geometric content knowledge is essential for geometry instructional quality and student learning, but difficult to scale. The Van Hiele model characterizes geometric reasoning through five hierarchical levels. Traditional Van Hiele assessment relies on manual expert analysis of open-ended responses. This process is time-consuming, costly, and prevents large-scale evaluation. This study develops an automated approach for diagnosing teachers' Van Hiele reasoning levels using large language models grounded in educational theory. Our central hypothesis is that integrating explicit skills information significantly improves Van Hiele classification. In collaboration with mathematics education researchers, we built a structured skills dictionary decomposing the Van Hiele levels into 33 fine-grained reasoning skills. Through a custom web platform, 31 pre-service teachers solved geometry problems, yielding 226 responses. Expert researchers then annotated each response with its Van Hiele level and demonstrated skills from the dictionary. Using this annotated dataset, we implemented two classification approaches: (1) retrieval-augmented generation (RAG) and (2) multi-task learning (MTL). Each approach compared a skills-aware variant incorporating the skills dictionary against a baseline without skills information. Results showed that for both methods, skills-aware variants significantly outperformed baselines across multiple evaluation metrics. This work provides the first automated approach for Van Hiele level classification from open-ended responses. It offers a scalable, theory-grounded method for assessing teachers' geometric reasoning that can enable large-scale evaluation and support adaptive, personalized teacher learning systems.

## Full Text


<!-- PDF content starts -->

Automatically Inferring Teachers’ Geometric
Content Knowledge: A Skills Based Approach
Ziv Fenigstein1[0009-0002-2237-1120], Kobi Gal1,2[0000-0001-7187-8572], Avi
Segal1[0000-0003-1422-2598], Osama Swidan1[0000-0002-2689-7173], Inbal
Israel1[0009-0005-3213-8922], and Hassan Ayoob1[0009-0005-7726-6602]
1Ben-Gurion University, Israel
zivfenig@post.bgu.ac.il, kobig@bgu.ac.il, avise@post. bgu.ac.il,
osamas@bgu.ac.il, inbalalmasy@gmail.com, Hassan.ayoob @gmail.com
2University of Edinburgh, U.K.
kgal@ed.ac.uk
Abstract. Assessing teachers’ geometric content knowledge is essent ial
for geometry instructional quality and student learning, b ut diﬃcult to
scale. The Van Hiele model characterizes geometric reasoni ng through
ﬁve hierarchical levels. Traditional Van Hiele assessment relies on manual
expert analysis of open-ended responses. This process is ti me-consuming,
costly, and prevents large-scale evaluation. This study de velops an au-
tomated approach for diagnosing teachers’ Van Hiele reason ing levels
using large language models grounded in educational theory . Our central
hypothesis is that integrating explicit skills informatio n signiﬁcantly im-
proves Van Hiele classiﬁcation. In collaboration with math ematics educa-
tion researchers, we built a structured skills dictionary d ecomposing the
Van Hiele levels into 33 ﬁne-grained reasoning skills. Thro ugh a custom
web platform, 31 pre-service teachers solved geometry prob lems, yield-
ing 226 responses. Expert researchers then annotated each r esponse with
its Van Hiele level and demonstrated skills from the diction ary. Using
this annotated dataset, we implemented two classiﬁcation a pproaches:
(1) retrieval-augmented generation (RAG) and (2) multi-ta sk learning
(MTL). Each approach compared a skills-aware variant incor porating
the skills dictionary against a baseline without skills inf ormation. Re-
sults showed that for both methods, skills-aware variants s igniﬁcantly
outperformed baselines across multiple evaluation metric s. This work
provides the ﬁrst automated approach for Van Hiele level cla ssiﬁcation
from open-ended responses. It oﬀers a scalable, theory-gro unded method
for assessing teachers’ geometric reasoning that can enabl e large-scale
evaluation and support adaptive, personalized teacher lea rning systems.
Keywords: AI in teachers training · Geometric Reasoning · LLMs in
EducationThe work is accepted for publication as a full paper (Main
Track) at the 27th International Conference on Artiﬁcial
Intelligence in Education (AIED 2026).arXiv:2604.13666v1  [cs.CY]  15 Apr 2026

2 Z. Fenigstein et al.
1 Introduction
Teachers’ content knowledge (CK) in mathematics has been sh own to directly
impact students’ learning outcomes [ 5,6,21]. Within the realm of Geometry, pre-
service teachers often demonstrate lower competency than e xpected for eﬀective
instruction [ 2,18,22]. Understanding teachers’ content knowledge in this domai n
is therefore critical for both research and professional de velopment.
The Van Hiele model of geometric thought characterizes geom etry content
knowledge through ﬁve hierarchical levels of reasoning ran ging from visualiza-
tion to formal rigor [ 13]. Building on this framework, researchers have argued
that Van Hiele assessment should focus on analyzing learner s’ responses to open-
ended questions rather than focus on the correctness of thei r answers [ 7,15]. Al-
though widely used in education research, existing approac hes for assessing Van
Hiele levels rely primarily on manual response analysis, wh ich is time-consuming,
costly, and hard to scale [ 18,21,22].
This paper addresses this gap by using Large Language Models (LLMs) to
classify teachers’ Van Hiele levels from their open-ended r esponses. The central
hypothesis of our work was that integrating explicit skills information would sig-
niﬁcantly improve Van Hiele classiﬁcation. To this end, we c onstructed a struc-
tured skills dictionary in collaboration with mathematics education researchers,
decomposing the ﬁve Van Hiele levels into a total of 33 ﬁne-gr ained reasoning
skills that characterize each level. We ﬁrst conducted a dat a collection study with
31 pre-service math teachers who provided open-ended respo nses to geometry
problems through a custom web-based platform, yielding 226 question-response
pairs. Expert researchers in mathematics education annota ted each pair with
its Van Hiele level and identiﬁed which of that level’s assoc iated skills from the
skills dictionary were demonstrated in the response. We the n developed several
classiﬁcation models that, given a question-response pair , diagnose the Van Hiele
level reﬂected in the response rather than just surface corr ectness.
We considered several approaches for classiﬁcation, (1) re trieval-augmented
generation (RAG), which incorporates skills by retrieving annotated examples
with skills labels and including the skills dictionary in th e model’s prompt, and
(2) multi-task learning, which incorporates skills throug h attention mechanisms
and an auxiliary skills prediction task alongside the prima ry Van Hiele level clas-
siﬁcation. For each classiﬁcation approach, we compared a s kills-aware variant
against a baseline without skills information, isolating t he contribution of explicit
skills modeling to classiﬁcation performance. For both cla ssiﬁcation approaches,
the skills-aware variants signiﬁcantly outperformed the b aseline variants on held
out test-sets. To better understand the models’ behavior, w e examined the mod-
els’ sensitivity to skill deﬁnitions in the RAG approach, th e contribution of in-
dividual skill components in the multi-task learning appro ach, and classiﬁcation
diﬃculty patterns across Van Hiele levels.
Our work makes several contributions. (1) it is the ﬁrst auto mated approach
for inferring Van Hiele levels from open-ended responses, a ddressing a scalability
challenge in geometry education research; (2) it provides t wo novel classiﬁcation
approaches demonstrating that explicitly modeling teache rs’ skills signiﬁcantly

Automatically Inferring Teachers’ Geometric Content Know ledge 3
improves classiﬁcation regardless of the modeling paradig m; and (3) it provides
new data sources for researchers in collaboration with math ematics education
researchers: a theoretically grounded skills dictionary a nd an annotated dataset
of 226 question-response pairs from pre-service teachers. Our code, models, data,
prompts, and skills dictionary are publicly available in Gi tHub3.
2 The Van Hiele Model of Geometric Reasoning
The Van Hiele model [ 13] describes ﬁve hierarchical levels of geometric reasoning
demonstrated by learners. Advancement through levels depe nds on mastery of
the preceding levels. At the most basic level (Level 1, Visua lization), learners
identify shapes based on their overall appearance. For exam ple, a student rec-
ognizes a rectangle as visually distinct from a trapezoid. A t Level 2 (Analysis),
learners recognize and describe properties of geometric sh apes. For example, a
student might state that a square has four right angles and al l sides of equal
length. Level 3 (Informal Deduction) involves reasoning ab out relationships be-
tween properties and between shape classes. For example, a s tudent might con-
clude that a rectangle is a type of parallelogram because it p ossesses all the
deﬁning properties of parallelograms. At Level 4 (Deductio n), learners construct
formal proofs based on deﬁnitions, axioms, and theorems. Fo r example, a student
can prove that the opposite angles of a parallelogram are con gruent. The highest
level (Level 5, Rigor) entails abstract reasoning in non-Eu clidean or axiomatic
systems. For example, a student analyzes how geometric prop erties, concepts,
or proofs change when axioms are modiﬁed or when comparing Eu clidean and
non-Euclidean geometries [ 17]. Although the Van Hiele model was originally de-
veloped to characterize geometric reasoning in school-age d students, its levels
reﬂect instructional experience rather than age or maturat ion [7]. This makes it
applicable to learners at any stage, including teachers. Ma yberry [ 23] conﬁrmed
this empirically, showing that the Van Hiele hierarchy hold s for undergraduate
preservice teachers. Since teachers’ geometric content kn owledge directly shapes
their students’ learning opportunities [ 18,4], understanding teachers’ Van Hiele
levels is essential for improving geometry instruction. Th e model has therefore
been applied to teachers for a range of purposes: (1) inferri ng their content
knowledge in geometry [ 18,22]; (2) measuring the relationship between their ge-
ometric reasoning and student achievement [ 21]; and (3) studying the eﬀects
of pedagogical interventions on their content knowledge [ 2,25], and examining
their ability to diagnose students’ geometric understandi ng [31]. All of these
studies have relied on either manual expert assessment of op en-ended responses
or multiple-choice questionnaires. Manual assessment is a time-consuming and
resource-intensive process that limits sample sizes and pr events large-scale longi-
tudinal tracking of teacher development. Automating this c lassiﬁcation enables
researchers to scale up their studies signiﬁcantly.
Usiskin [ 29] developed a multiple-choice test where learners at a given Van
Hiele level were expected to correctly answer items at that l evel and all preced-
3https://github.com/zivfenig/Van-Hiele-Level-Classiﬁ cation

4 Z. Fenigstein et al.
ing levels. However, researchers have critiqued this appro ach, arguing that Van
Hiele levels comprise multiple distinct reasoning process es [7,11]. Therefore, ac-
curate assessment requires evaluating the speciﬁc skills d emonstrated in learners’
explanations, not merely correctness on multiple-choice i tems. This skills-based
perspective directly motivates our modeling approach of de composing each Van
Hiele level into explicit, ﬁne-grained reasoning skills.
For example, consider the question: “Given a shape with four equal sides,
is it necessarily a rhombus?” One student might respond, “No, maybe it is a
square” . This reﬂects Level 2 reasoning (Analysis) demonstrating t he Level 2
skill of describing shapes in terms of their properties but n ot the Level 3 skill of
recognizing that squares are a special case of rhombus. Anot her student might
say,“Yes, because a square is also a type of rhombus” , demonstrating Level 3
type reasoning (Informal Deduction) by explicitly underst anding the inclusion
relationship, that one shape class can be a subset of another .
Building on this paradigm, our work develops a structured sk ills dictionary
decomposing each Van Hiele level into explicit reasoning sk ills in collabora-
tion with mathematics education researchers. This skill de composition forms the
foundation of our automated classiﬁcation approach for ans wers to open-ended
questions (the full skills dictionary available in our repo sitory4).
3 Related Work
Recent advances have made LLMs increasingly eﬀective for ed ucational assess-
ment. Henkel et al. [ 12] demonstrated that LLMs can achieve near-parity with
human raters when grading open-ended reading comprehensio n responses, dis-
tinguishing between fully correct, partially correct, and incorrect responses. Be-
yond correctness, Lee et al. [ 19] utilized LLMs to classify middle-school writ-
ten responses to science assessments into proﬁciency level s (Proﬁcient, Devel-
oping, and Beginning), ﬁnding that Chain-of-Thought promp ting with explicit
rubrics improved classiﬁcation accuracy across proﬁcienc y levels. However, LLMs
can struggle with nuanced reasoning assessment. Rachmatul lah et al. [ 24] used
LLMs to assess teachers’ Pedagogical Content Knowledge (PC K). They found
that while LLMs achieved high reliability when scoring less on plans, they strug-
gled to assess teachers’ analyses of student misconception s, relying heavily on
keyword matching and therefore failed to accurately identi fy more nuanced re-
sponses from teachers which did not include those keywords. This highlights the
challenge of capturing nuanced reasoning. Our skills-awar e approach addresses
this by explicitly deﬁning the reasoning patterns that dist inguish between Van
Hiele levels. While these studies demonstrate LLMs’ potent ial for educational as-
sessment, to our knowledge no prior work has applied LLMs or m achine learning
to assess geometric reasoning under the Van Hiele framework .
RAG has proven eﬀective for classifying open-ended student responses. Fa-
teen et al. [ 9] employed RAG to retrieve similar student answers as few-sh ot
4https://github.com/zivfenig/Van-Hiele-Level-Classiﬁ cation

Automatically Inferring Teachers’ Geometric Content Know ledge 5
examples for automatic short-answer scoring, improving pe rformance over base-
line methods. Additionally, Jauhiainen and Guerra [ 16] utilized RAG to retrieve
relevant reference material from scholarly articles for gr ading university exami-
nation answers. We build on these approaches by augmenting r etrieved examples
with explicit skills annotations, providing pedagogicall y grounded context that
guides classiﬁcation beyond surface similarity.
Multi-task learning is a well-established and eﬀective par adigm across AI
domains. Xu et al. [ 30] demonstrated that attention mechanisms informed by
auxiliary task outputs improve performance by incorporati ng structured do-
main knowledge in image classiﬁcation and segmentation tas ks. This approach
has been successfully applied to education research. Huang et al. [ 14] showed
that jointly learning multiple teacher-question classiﬁc ation tasks improves per-
formance through shared semantic representations. An et al . [1] and Geden et
al. [10] demonstrated that auxiliary objectives - such as option tr acing or per-
item prediction, lead to more accurate and stable student as sessment. Inspired
by these works, we apply multi-task learning where skills se rve as an auxiliary
prediction task alongside Van Hiele level classiﬁcation.
4 Methodology
In this section, we present our methodology to automaticall y classify teachers’
Van Hiele levels using skills-aware modeling approaches (s ee Figure 1). We ﬁrst
developed a comprehensive geometry question bank and struc tured skills dictio-
nary grounded in Van Hiele theory. We then collected and anno tated a dataset of
teacher responses with expert-assigned Van Hiele levels an d demonstrated skills.
Using this annotated dataset we developed the two classiﬁca tion methods - one
using retrieval-augmented generation, the other using mul ti-task learning - each
compared against a baseline without skills information to i solate the contribu-
tion of explicit skills modeling. We studied whether the ski lls-aware variant of
both methods outperformed the baselines.
Step 1: Design
Resources
Question Bank
Development
Skills Dictionary
(Grounded in Van Hiele
model)Step 2: Studies With
Pre-Service TeachersStep 3: Data Collection
&& Annotation
Question
 Answer
Demonstrated
Skills
Van Hiele LevelStep 4: Van Hiele
Classification Models
Method I:
Retrieval-Augmented
Classification
Method II:
Supervised Fine-
Tuning
Fig.1. Overview of the proposed skills-based Van Hiele classiﬁcat ion framework.

6 Z. Fenigstein et al.
4.1 Question Bank and Skills Dictionary Development
We developed a question bank building on problems from Usisk in’s Van Hiele
geometry test, which consists entirely of closed-ended mul tiple-choice items [ 29].
We adapted items into open-ended formats to elicit richer ex planations from
teachers, promoting reasoning beyond recognition of ﬁxed o ptions.
In total we constructed 59 diﬀerent open-ended geometry pro blems. The
questions addressed widely taught geometry topics such as q uadrilaterals, angle
relationships, triangle congruence, and similarity. Item s varied in diﬃculty to
elicit reasoning across the Van Hiele spectrum. Example of q uestion included
“List at least two properties that are shared by all squares bu t not shared by all
rhombus, justify your choices” .
We built a structured skills dictionary in collaboration wi th three mathemat-
ics education researchers with expertise in geometry educa tion and experience
teaching and studying pre-service mathematics teachers. T he skills dictionary
was grounded in Crowley’s theoretical decomposition of Van Hiele levels [ 7],
which establishes that each level is characterized by speci ﬁc observable reason-
ing behaviors and distinct linguistic markers that can only be reliably identiﬁed
from open-ended explanations. For example, Crowley explic itly identiﬁes level-
speciﬁc vocabulary and behaviors that directly correspond to skills in our dictio-
nary, such as the use of logical connectives (“if. . . then”, “ it follows that”) at the
Informal Deduction level, and the ability to identify what i s given versus what
must be proved at the Deduction level. Drawing on this theore tical foundation
and their pedagogical experience with pre-service teacher s, the experts decom-
posed each Van Hiele level into ﬁne-grained reasoning skill s that characterize
that level, resulting in 33 distinct skills across the ﬁve le vels.
To collect teachers’ responses, we developed a custom web-b ased platform.
The interface presents a sequence of problems from our quest ion bank and teach-
ers provide responses in free text, allowing them to articul ate their full reason-
ing process. For problems requiring geometric constructio n or manipulation, the
platform embeds an interactive GeoGebra [ 26] applet alongside the question.
The applet served only as a cognitive scaﬀold and was not used in our model.
4.2 Data Collection and Annotation
A total of 31 pre-service mathematics teachers from three re gional teacher train-
ing institutions participated in our study. All participan ts were enrolled in a
geometry course. The study protocol was reviewed and approv ed by the Institu-
tional Review Board (IRB). All participants provided infor med consent before
taking part in the study. The web-based platform described i n Section 4.1served
both for data collection and geometry practice, incorporat ing multiple-choice
(MCQ) and open-ended problems. Each participant was assign ed 20 problems
randomly selected (at least 10 open-ended) from our questio n bank, including
problems of varying diﬃculty designed to elicit responses a cross all Van Hiele
levels. This ensured coverage of the full spectrum of geomet ric reasoning, from

Automatically Inferring Teachers’ Geometric Content Know ledge 7
basic visualization (Level 1) to Rigor (Level 5). The platfo rm supported ﬂexi-
ble participation with no strict time limits and the ability to save and resume
work. For this study, we retained only open-ended responses ; MCQ responses and
empty submissions were excluded, yielding 226 valid questi on-response pairs.
We employed a double-blind annotation protocol to ensure ob jective ground-
truth labeling of Van Hiele levels. Two experts in mathemati cs education in-
dependently reviewed every response and labeled it with the Van Hiele level
without seeing the other’s labels. The inter-rater reliabi lity was high (Cohen’s
κ= 0.84) indicating substantial agreement between annotators’ in itial labels.
After assigning the Van Hiele level, the two experts collabo ratively identiﬁed
which skills from the skills dictionary were demonstrated i n each response.
This annotation process produced a dataset of 226 question- response pairs,
each annotated with a Van Hiele level and demonstrated skill s. The dataset
contains responses across all ﬁve levels, with Levels 2 and 3 most common and
Levels 4 and 5 less represented, aligning with past studies [ 2,22].
4.3 Van Hiele Level Classiﬁcation Models
We developed two methods to classify Van Hiele levels from qu estion–response
pairs. For each, we implemented a baseline without skills in formation and a
skills-aware variant incorporating our structured skills dictionary. All other com-
ponents are identical across variants, isolating the eﬀect of skills information.
Method I: Retrieval-Augmented Classiﬁcation. The ﬁrst method uses a
Retrieval-Augmented Generation (RAG) pipeline that recei ves a new question-
response pair as input, retrieves similar pairs from our ann otated dataset, and in-
tegrates them into a prompt for LLM classiﬁcation. Figure 2provides an overview
of this retrieval-augmented classiﬁcation architecture. Each question-response
pair in the dataset is encoded separately using multilingual-e5-base5embed-
ding model. All embeddings are L2-normalized to optimize retrieval within the
RAG pipeline, enabling eﬃcient and stable cosine similarit y computation via
normalized dot products [ 8].
For a new question-response pair, we construct a weighted qu ery embedding
assigning 80% weight to the response and 20% to the question. This weighting
reﬂects the Van Hiele framework’s emphasis on reasoning exp ressed in responses
rather than questions themselves (see discussion in Sectio n2). Cosine similarity
is computed between the new question-response embeddings a nd the stored vec-
tors, and the top-K (see Section 4.4for K) most similar examples are retrieved.
In the baseline variant (RAG without skills), each retrieved example includes
the geometry question, the teacher’s response, and the corr esponding Van Hiele
level. The LLM is also supplied with a system prompt containi ng only the stan-
dard Van Hiele level deﬁnitions. No skill annotations or ski ll deﬁnitions are
provided to the model. In the skills-aware variant , retrieved examples addition-
ally contain the annotated skills associated with each teac her’s response, and
5https://huggingface.co/intﬂoat/multilingual-e5-base

8 Z. Fenigstein et al.
Question
ResponseEmbedding
Q+ASimilarity
SearchVan Hiele
Definitions
Skills
Definitions
Top K
Retrieved
ExamplesLLM
ClassifierInputPrompt
Builder
Labeled
DatasetPredicted
Van Hiele
LevelOutput
Question-
Response
Fig.2. Retrieval-augmented generation architecture for Method I .
the system prompt includes both Van Hiele level deﬁnitions a nd the full skills
dictionary.
All other components of the pipeline - including the input en coding, retrieval
procedure, number of retrieved examples, language model, a nd inference settings
- are the same for both variants. As a result, any observed per formance diﬀer-
ences between the baseline and skills-aware versions can be attributed speciﬁcally
to the presence of explicit skills information in the prompt .
Method II: Multi-Task Learning. The second classiﬁcation method (see
Figure 3) ﬁne-tunes an open-source LLM for Van Hiele classiﬁcation. Similarly
to Method I, the model receives a question-response pair as i nput and outputs
the Van Hiele level. We implemented two variants: a baseline that performs clas-
siﬁcation without skills information during training, and a skills-aware variant
that incorporates an auxiliary skills prediction task with Van Hiele classiﬁcation.
Both variants used the same base open-source LLM.
Question
ResponseInput
Base
Language
Model
(Gemma-3-
4B-it)LoRA
AdaptersEncoding
Input 
Skills
Embeddings
×
Encoded Input
RepresentationSkills
Attention
Mechanism 
Input
Representation
+
Attention
WeightsAuxiliary Task
Head
(Multi-Label
Skills Head)
Main Task Head
(Classification
Head)Combined
LossPredicted
Van Hiele
levelOutput
Fig.3. Multi-task learning architecture for Method II.
In the baseline variant , the question-response pair is encoded by the LLM
and passed directly to a linear classiﬁcation head that pred icted the Van Hiele
level. This variant does not use skills information during t raining.
In the skills-aware variant , we augmented the baseline with a skills attention
mechanism and an auxiliary skills prediction task. Like the baseline, the question-
response pair is ﬁrst encoded by the LLM to produce an input re presentation.

Automatically Inferring Teachers’ Geometric Content Know ledge 9
Each Van Hiele skill from the skills dictionary is represent ed by a trainable em-
bedding vector, initialized from text encodings of the skil l descriptions using the
multilingual-e5-base model. Attention weights are computed via dot prod-
ucts between the LLM’s encoded input representation and the skill embeddings
followed by a softmax normalization to produce a probabilit y distribution over
skills, quantifying how strongly the question-response pa ir aligns with each skill.
This attention weights vector is then concatenated with the input representation
(encoded question-response pair). The enriched represent ation feeds two predic-
tion heads in parallel: a primary Van Hiele level classiﬁcat ion head (identical to
the baseline) and an auxiliary skills prediction head that p redicts which skills
are demonstrated. The model is trained by optimizing a combi ned loss balancing
the Van Hiele classiﬁcation and auxiliary skills predictio n tasks, weighted by λ:
Ltotal=Llevel+λ·Lskills (1)
whereLlevelis cross-entropy loss for Van Hiele level prediction, penal izing incor-
rect classiﬁcation across the ﬁve distinct levels, and Lskillsis binary cross-entropy
loss for skills prediction, penalizing incorrect presence /absence predictions for
each skill independently. See Section 4.4for implementation details.
These components create complementary learning signals du ring training.
The attention mechanism identiﬁes which skills matter for e ach input, while the
auxiliary prediction task ensures the model learns represe ntations that encode
these skills. The dual learning signals from the classiﬁcat ion loss and the skills
prediction loss shape both the LLM’s representations and th e skills embeddings
through backpropagation to better capture Van Hiele-speci ﬁc reasoning patterns.
At inference, both variants take the question-response pai r as input and out-
put the predicted Van Hiele level. The skills-aware variant also computes skills
predictions and attention weights internally, which can be accessed for inter-
pretability analysis if needed. As in Method I, all other com ponents and hyper-
parameters were the same for both variants.
4.4 Implementation Settings
For Method I, all retrieval-augmented variants use Gemini-2.0-Flash [27], cho-
sen for long-context processing. Retrieval uses K= 5examples, as prior work
shows this optimizes citation recall [ 20]. Generation uses temperature 0.0 for
deterministic predictions [ 3]. Variants diﬀer only in prompt content: the base-
line includes Van Hiele level deﬁnitions and retrieved exam ples annotated with
levels, while the skills-aware variant additionally inclu des the skills dictionary
and skill annotations. Gemini 2.0 Flash was accessed via Ver tex AI on Google
Cloud Platform.
For Method II, both ﬁne-tuning variants use Gemma-3-4B-IT [28], a mid-
sized instruction-tuned model balancing capacity and eﬃci ency for limited-data
supervision. We apply LoRA (rank 16), which performed best e mpirically, keep-
ing base weights frozen.
To mitigate overﬁtting risk given the limited dataset size, we apply dropout
in LoRA adapters (0.05) and in the Van Hiele classiﬁcation he ad (0.25), with

10 Z. Fenigstein et al.
learning rate 2×10−4and weight decay 0.05. Training runs up to 30 epochs
with early stopping on macro F1 (patience=4). The skills-aw are variant uses
auxiliary loss weight λ= 0.5, selected empirically. Both variants were trained on
an NVIDIA RTX 6000 GPU. Additional technical conﬁgurations can be found
in our GitHub repository6.
5 Experiments And Results
We evaluated both methods by comparing their baseline and sk ills-aware vari-
ants on classifying Van Hiele levels in our collected datase t. We used ﬁve-fold
cross-validation with a ﬁxed random seed (42) for reproduci bility. The dataset
was partitioned into ﬁve independent folds while preservin g the Van Hiele level
distribution. Each fold served as a test set in turn, with the remaining four folds
used for training and retrieval (for the RAG system). Within each training set,
15% was held out as a validation set for parameter tuning and e arly stopping.
We report macro-averaged and weighted-averaged F1 scores ( F1-macro, F1-
weighted) to assess standard classiﬁcation performance. F 1-macro measures how
consistently the model performs across Van Hiele levels by a ssigning equal weight
to each class, while F1-weighted reﬂects overall performan ce by accounting for
the empirical label distribution. To capture the ordinal st ructure of the Van Hiele
hierarchy, we additionally report Mean Absolute Error (MAE ), penalizing larger
ordinal misclassiﬁcations, and Quadratic Weighted Kappa ( QWK), which ac-
counts for chance agreement; QWK ranges from −1to1, where values close to
1indicate stronger agreement.
Figure 4shows the average results for RAG classiﬁcation (Method I) a nd
Multi-task learning methods (Method II), comparing baseli ne (gray bars) against
skills-aware (blue bars) classiﬁcation across ﬁve-fold te st sets. For both methods,
the skills-aware models signiﬁcantly outperformed the rel evant baseline in all
measures. We also note that the standard deviations of the sk ills-aware variants
in both methods were consistently lower compared to their re spective baselines,
suggesting the models are more stable than the baseline vari ants.
A series of paired t-tests conﬁrmed that the improvements of Method I
are statistically signiﬁcant across all metrics: F1-macro (t(4)=2.909, p=0.0437),
F1-weighted (t(4)=3.008, p=0.0396), MAE (t(4)=-5.2, p=0. 0065), and QWK
(t(4)=3.465, p=0.0257). With respect to Method II, statist ical signiﬁcance was
obtained for F1-macro (t(4)=2.908, p=0.043), MAE (t(4)=-3 .836, p=0.0185)
and QWK (t(4)=3.874, p=0.018), but not for F1-weighted meas ure. A possible
reason for this is the small data set (226 question-response pairs).
5.1 Sensitivity Analysis
We performed several sensitivity analyses to study how vari ations in skill-related
information inﬂuence model behavior. With respect to Metho d I, we wanted to
6https://github.com/zivfenig/Van-Hiele-Level-Classiﬁ cation

Automatically Inferring Teachers’ Geometric Content Know ledge 11
Fig.4. Average results across 5-fold cross-validation comparing baseline and skills-
aware variants for RAG (Method I) and MTL (Method II). Number s above bars show
mean scores; lines within the bars indicate standard deviat ion. For MAE measure,
lower is better.
validate the performance gain of the skills-aware variant t o selecting the right
skills for the right question-response pair, as opposed to simply providing the
model with additional context. To this end, we compared the b aseline and the
skills-aware variants to a “noisy skills” variant. In this s etup, we randomly shuf-
ﬂed the skills deﬁnitions, so that each skill label was assig ned a deﬁnition of
another skill. All other components of the skill-aware meth od and the empirical
methodology remained the same. Table 1shows that the noisy skills variant ex-
Table 1. Skills sensitivity analysis for Method I. Values are report ed as mean ±stan-
dard deviation across ﬁve cross-validation folds.
Variant F1-macro F1-weighted QWK MAE
Baseline (No Skills) 0.624±0.0930.678±0.0950.625±0.0350.47±0.09
Skills-Aware 0.695±0.090.736±0.080.721±0.0540.376±0.06
Noisy Skills 0.61±0.1220.635±0.1170.673±0.0930.495±0.156
hibited signiﬁcant performance degradation across all met rics compared to the
skills-aware, and even performed worse than baseline. This demonstrates that
performance gains depend on the model’s ability to utilize c orrectly aligned and
pedagogically meaningful skills information, rather than simply beneﬁting from
additional contextual input or increased prompt length.
For Method II, we quantiﬁed the individual contributions of the skills atten-
tion mechanism and auxiliary skills prediction head by isol ating each compo-
nent. We compared two partial variants: (1) Attention-Guided variant, trained
with the skills-based attention mechanism and the Van Hiele classiﬁcation head,
without the auxiliary skills prediction task; (2) Skills-Supervised variant, trained
with the auxiliary skills prediction head and the Van Hiele c lassiﬁcation head,
but without the skills-based attention mechanism. Table 2compares the full ap-

12 Z. Fenigstein et al.
proach Full Model (Method II) (skills-aware variant) to the Attention-Guided ,
theSkills-Supervised variants and the Baseline (No Skills) variant.
Table 2. Skills component analysis for Method II. Values are reporte d as mean ±
standard deviation across ﬁve cross-validation folds.
Variant F1-macro F1-weighted QWK MAE
Baseline 0.646±0.0820.657±0.0920.586±0.1570.523±0.119
Attention-Guided 0.676±0.0580.679±0.0660.632±0.1290.487±0.086
Skills-Supervised 0.652±0.0630.653±0.0880.581±0.0890.558±0.119
Full Model (Method II) 0.725±0.0430.725±0.0530.717±0.1030.403±0.102
Results show the attention-guided variant clearly outperf orms the baseline,
while skills supervision alone yields limited gains. The fu ll model performs best,
indicating the components are complementary: the auxiliar y task encourages
the encoder to capture patterns aligned with pedagogical sk ills and Van Hiele
levels, and the attention mechanism leverages these signal s to focus on skills-
relevant aspects of responses, producing more informative representations for
level classiﬁcation.
5.2 Error Analysis
To identify which Van Hiele levels are most diﬃcult to classi fy, we analyzed per-
level performance by aggregating predictions across all cr oss-validation folds.
Both methods achieve high accuracy on Level 4. A possible rea son is that Level 4
(Deduction) responses involve constructing formal proofs , which are character-
ized by structured patterns that are easier for models to lea rn and identify.
The skills dictionary likely contributes to this: Level 4 sk ills such as identifying
given information versus what must be proved, and construct ing formal logical
arguments, map directly onto textual patterns that models c an detect reliably.
In addition, for both methods, Level 2 accuracy is around 60% , with most
misclassiﬁcations occurring as Level 3. This demonstrates the diﬃculty of dis-
tinguishing between these adjacent levels, where diﬀerenc es in responses may be
subtle and evident only in speciﬁc linguistic details (see e xample in Section 2).
Level 5 is challenging for both methods but shows markedly di ﬀerent per-
formance: Method I achieves only 31% accuracy rate while Met hod II achieves
69%. In both cases, predictions are distributed broadly acr oss multiple levels
rather than concentrated near the true level, indicating pe rsistent classiﬁcation
diﬃculty. This likely stems from limited Level 5 representa tion in our dataset.
The performance gap between the methods at Level 5 can be attr ibuted to their
fundamental diﬀerence in approach: Method I relies on retri eving similar ex-
amples from the dataset, but with only 7% of responses at Leve l 5, similarity
search rarely surfaces relevant examples, leaving the mode l without meaningful
context for classiﬁcation. Method II, by contrast, learns r epresentations through
training, enabling it to capture Level 5 patterns from spars e supervision.

Automatically Inferring Teachers’ Geometric Content Know ledge 13
6 Discussion and Conclusion
This work demonstrates that Large Language Models can eﬀect ively infer teach-
ers’ Van Hiele reasoning levels when guided by structured pe dagogical infor-
mation. Our central hypothesis was that integrating explic it skills information
improves Van Hiele classiﬁcation. We tested this with two di stinct approaches:
Retrieval-Augmented Generation and Multi-Task Learning. In both cases, skills-
aware variants signiﬁcantly outperformed baselines witho ut skills information
across multiple evaluation metrics. We hypothesize that th e skills dictionary
helps models identify diagnostic patterns that distinguis h Van Hiele levels. These
include the use of logical language like "if...then" or "the refore," or linking prop-
erties across shape families. By deﬁning reasoning pattern s characteristic of each
level, the skills help models attend to features that diﬀere ntiate between levels.
Our research has practical implications for mathematics ed ucation research
and professional development. By automating Van Hiele asse ssment - tradition-
ally constrained by manual expert evaluation - our approach enables researchers
to study geometric reasoning development at scale across la rge teacher cohorts.
Automated assessment could also enable adaptive professio nal development sys-
tems that dynamically adjust content based on teachers’ cur rent Van Hiele lev-
els and skill proﬁles. Additionally, extending the propose d models to output
demonstrated skills alongside Van Hiele levels would creat e ﬁne-grained diag-
nostic proﬁles for detecting strengths and gaps, enabling t argeted interventions
that address speciﬁc reasoning weaknesses.
Several limitations should be noted. First, the dataset of 2 26 responses is
relatively small, particularly for higher Van Hiele levels , which may limit model
generalizability. Second, our research was conducted with a ﬁxed set of 59 ques-
tions, and models may not generalize to new problems; extend ing this approach
to arbitrary geometry questions remains an important chall enge for future work.
Third, in contrast to Van Hiele levels, skills were identiﬁe d through expert con-
sensus rather than independent annotation; while our exper iments validate their
utility for Van Hiele classiﬁcation, independent annotati on with inter-rater reli-
ability would strengthen conﬁdence in skill labels.
Beyond geometric reasoning, this work demonstrates a poten tially gener-
alizable approach: decomposing hierarchical learning fra meworks into explicit,
ﬁne-grained skills that guide automated assessment. While we validated this
methodology for Van Hiele levels, the principle may extend t o other structured
frameworks such as Bloom’s Taxonomy or subject-speciﬁc rea soning models.
Such approaches bridge AI capabilities with pedagogical th eory by ensuring
models assess learning using the same constructs educators use. Validating this
skills-based methodology remains important future work.
Acknowledgments. This study was funded in part by Israeli Ministry of Science a nd
Technology grant number 7774.
Disclosure of Interests. The authors have no competing interests to declare that
are relevant to the content of this article.

14 Z. Fenigstein et al.
References
1. An, S., Kim, J., Kim, M., Park, J.: No task left behind: Mult i-task learning of
knowledge tracing and option tracing for better student ass essment. Proceedings
of the AAAI Conference on Artiﬁcial Intelligence 36(4), 4424–4431 (Jun 2022).
https://doi.org/10.1609/aaai.v36i4.20364
2. Armah, R.B., Coﬁe, P.O., Okpoti, C.A.: Investigating the eﬀect of Van Hiele
phase-based instruction on pre-service teachers’ geometr ic thinking. Interna-
tional Journal of Research in Education and Science 4(1), 314–330 (2018),
https://eric.ed.gov/?id=EJ1169856
3. Bajan, C., Lambard, G.: Exploring the expertise of large l anguage models in ma-
terials science and metallurgical engineering. Digital Di scovery 4, 500–512 (2025).
https://doi.org/10.1039/D4DD00319E
4. Beswick, K., Goos, M.: Measuring pre-service primary tea chers’ knowledge for
teaching mathematics. Mathematics Teacher Education and D evelopment 14(2),
70–90 (2012)
5. Campbell, P.F., Malkus, N.N.: The impact of elementary ma thematics coaches
on student achievement. The elementary school journal 111(3), 430–454 (2011).
https://doi.org/10.1086/657654
6. Copur-Gencturk, Y., Li, J., Cohen, A.S., Orrill, C.H.: Th e impact of an interactive,
personalized computer-based teacher professional develo pment program on student
performance: A randomized controlled trial. Computers & ed ucation 210(2024).
https://doi.org/10.1016/j.compedu.2023.104963
7. Crowley, M.L.: The van hiele model of the development of ge ometric thought.
Learning and teaching geometry, K-12 1, 1–16 (1987)
8. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G. , Mazaré, P.E., Lomeli,
M., Hosseini, L., Jégou, H.: The faiss library. IEEE Transac tions on Big Data pp.
1–17 (2025). https://doi.org/10.1109/TBDATA.2025.3618474
9. Fateen, M., Wang, B., Mine, T.: Beyond scores: A modular ra g-based sys-
tem for automatic short answer scoring with feedback. IEEE A ccess12(2024).
https://doi.org/10.1109/ACCESS.2024.3508747
10. Geden, M., Emerson, A., Rowe, J., Azevedo, R., Lester, J. : Predictive stu-
dent modeling in educational games with multi-task learnin g. Proceedings of
the AAAI Conference on Artiﬁcial Intelligence 34(01), 654–661 (Apr 2020).
https://doi.org/10.1609/aaai.v34i01.5406
11. Gutiérrez, A., Jaime, A., Fortuny, J.M.: An alternative paradigm to evaluate the
acquisition of the van hiele levels. Journal for Research in Mathematics Education
22(3), 237–251 (1991)
12. Henkel, O., Hills, L., Roberts, B., McGrane, J.: Can llms grade open response
reading comprehension questions? an empirical study using the roars dataset. In-
ternational journal of artiﬁcial intelligence in educatio n35(2), 651–676 (2025).
https://doi.org/10.1007/s40593-024-00431-z
13. van Hiele, P.: The child’s thought and geometry. ERIC Arc hive (translated from
original Dutch) (1959), originally presented at the OEEC co nference, Sèvres, 1957
14. Huang, G.Y., Chen, J., Liu, H., Fu, W., Ding, W., Tang, J., Yang, S., Li, G., Liu,
Z.: Neural multi-task learning for teacher question detect ion in online classrooms.
In: International Conference on Artiﬁcial Intelligence in Education. pp. 269–281.
Springer (2020). https://doi.org/10.1007/978-3-030-52237-7_22
15. Jaime, A., Gutiérrez, A.: A model of test design to assess the van hiele levels. In:
Proceedings of the 18th PME Conference. vol. 3, pp. 41–48. Pm e (1994)

Automatically Inferring Teachers’ Geometric Content Know ledge 15
16. Jauhiainen, J.S., Guerra, A.G.: Evaluating students’ o pen-ended written responses
with llms: Using the rag framework for gpt-3.5, gpt-4, claud e-3, and mistral-large
(2024), https://arxiv.org/abs/2405.05444
17. Jupri, A.: Using the van hiele theory to analyze primary s chool teachers’ written
work on geometrical proof problems. Journal of Physics: Con ference Series 1013(1),
012117 (may 2018). https://doi.org/10.1088/1742-6596/1013/1/012117
18. Kurt-Birel, G., Deniz, S., Önel, F.: Analysis of primary school teachers’ knowledge
of geometry. International Electronic Journal of Elementa ry Education 12(4), 303–
309 (2020). https://doi.org/10.26822/iejee.2020459459
19. Lee, G.G., Latif, E., Wu, X., Liu, N., Zhai, X.: Applying l arge
language models and chain-of-thought for automatic scorin g. Com-
puters and Education: Artiﬁcial Intelligence 6, 100213 (2024).
https://doi.org/https://doi.org/10.1016/j.caeai.202 4.100213
20. Leto, A., Aguerrebere, C., Bhati, I., Willke, T., Tepper , M., Vo, V.A.: Toward
optimal search and retrieval for rag (2024), https://arxiv.org/abs/2411.07396
21. Lumbre, A.P., Beltran-Joaquin, M.N., Monterola, S.L.C .: Relationship between
mathematics teachers’ van hiele levels and students’ achie vement in geometry. In-
ternational Journal of Studies in Education and Science (IJ SES)4(2), 113–123
(2023)
22. Manero, V., Arnal-Bailera, A.: Understanding proof pra ctices of pre-service math-
ematics teachers in geometry. Mathematics Teaching-Resea rch Journal 13(3), 99–
130 (2021)
23. Mayberry, J.: The van hiele levels of geometric thought i n undergraduate preser-
vice teachers. Journal for research in mathematics educati on14(1), 58–69 (1983).
https://doi.org/10.5951/jresematheduc.14.1.0058
24. Rachmatullah, A., Tayde, S., Alozie, N., et al.: Explori ng large language model’s ca-
pabilities in identifying science teacher pck using lesson plans and open-ended ques-
tions. Disciplinary and Interdisciplinary Science Educat ion Research 8(3) (2026).
https://doi.org/10.1186/s43031-025-00151-x
25. Swaﬀord, J.O., Jones, G.A., Thornton, C.A.: Increased k nowledge in geometry
and instructional practice. Journal for Research in Mathem atics Education 28(4),
467–483 (1997). https://doi.org/10.5951/jresematheduc.28.4.0467
26. Tamam, B., Dasari, D.: The use of geogebra software in tea ching mathe-
matics. Journal of Physics: Conference Series 1882(1), 012042 (may 2021).
https://doi.org/10.1088/1742-6596/1882/1/012042
27. Team, G., et al.: Gemini: a family of highly capable multi modal models. arXiv
preprint arXiv:2312.11805 (2023). https://doi.org/10.48550/arXiv.2312.11805
28. Team, G., et al.: Gemma 3 technical report. arXiv preprin t arXiv:2503.19786
(2025). https://doi.org/10.48550/arXiv.2503.19786
29. Usiskin, Z.: Van hiele levels and achievement in seconda ry school geometry. cdassg
project. ERIC (1982)
30. Xu, M., Huang, K., Qi, X.: Multi-task learning with conte xt-oriented self-attention
for breast ultrasound image classiﬁcation and segmentatio n. In: 2022 IEEE
19th International Symposium on Biomedical Imaging (ISBI) . pp. 1–5 (2022).
https://doi.org/10.1109/ISBI52829.2022.9761685
31. Yi, M., Flores, R., Wang, J.: Examining the inﬂuence of va n hiele theory-based
instructional activities on elementary preservice teache rs’ geometry knowledge
for teaching 2-d shapes. Teaching and Teacher Education 91, 103038 (2020).
https://doi.org/https://doi.org/10.1016/j.tate.2020 .103038
# DesignPref: Capturing Personal Preferences in Visual Design Generation

**Authors**: Yi-Hao Peng, Jeffrey P. Bigham, Jason Wu

**Published**: 2025-11-25 17:19:10

**PDF URL**: [https://arxiv.org/pdf/2511.20513v1](https://arxiv.org/pdf/2511.20513v1)

## Abstract
Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste.

## Full Text


<!-- PDF content starts -->

DesignPref: Capturing Personal Preferences in Visual Design Generation
Yi-Hao Peng
Carnegie Mellon University
yihaop@cs.cmu.eduJeffrey P. Bigham
Carnegie Mellon University
jbigham@cs.cmu.eduJason Wu
Apple
jason wu8@apple.com
Abstract
Generative models, such as large language models and text-
to-image diffusion models, are increasingly used to create
visual designs like user interfaces (UIs) and presentation
slides. Finetuning and benchmarking these generative mod-
els have often relied on datasets of human-annotated de-
sign preferences. Yet, due to the subjective and highly per-
sonalized nature of visual design, preference varies widely
among individuals. In this paper, we study this problem by
introducing DesignPref, a dataset of 12k pairwise compar-
isons of UI design generation annotated by 20 professional
designers with multi-level preference ratings. We found that
among trained designers, substantial levels of disagreement
exist (Krippendorff’s alpha = 0.25 for binary preferences).
Natural language rationales provided by these designers in-
dicate that disagreements stem from differing perceptions
of various design aspect importance and individual prefer-
ences. With DesignPref, we demonstrate that traditional
majority-voting methods for training aggregated “judge”
models often do not accurately reflect individual prefer-
ences. To address this challenge, we investigate multiple
personalization strategies, particularly fine-tuning or incor-
porating designer-specific annotations into RAG pipelines.
Our results show that personalized models consistently out-
perform aggregated baseline models in predicting individ-
ual designers’ preferences, even when using 20 times fewer
examples. Our work provides the first dataset to study per-
sonalized visual design evaluation and support future re-
search into modeling individual design taste.
1. Introduction
“Beauty is no quality in things themselves: It exists
merely in the mind which contemplates them; and each
mind perceives a different beauty. ”— DAVIDHUME
Modern generative models can now support visual design
tasks across UI layout and presentation slides composi-
tion. Design quality matters at scale: a classic survey re-
ports that about half of an application’s code and a similar
share of development time focus on the user interface [33].With advances in large language, vision-language and mul-
timodal models (LLMs/VLMs/LMMs), recent work begins
to assess visual design quality automatically. Across de-
signs, methods pair compiler or runtime diagnostics with
text–image alignment to estimate intent match [61], train
scorers that judge design screenshots against written goals
and align with professional judgments [60, 66], and add
region-level critiques for design to turn scores into design
suggestion [10]. These works suggest that model-based
evaluation can potentially guide design workflows at scale.
Despite rapid progress, off-the-shelf evaluators do not
yet deliver accurate personalized assessment of visual de-
sign. Most preference data is collected for the purpose
of aggregation in benchmark [28, 67] or training settings
such as RLHF and DPO [35, 44]. Yet, previous work has
shown that visual preference can vary widely among in-
dividuals and be poorly approximated by globally aggre-
gated data [15, 23, 25]. This is especially true for UI de-
sign, which has been shown to vary based on aesthetic taste,
user needs [36, 58], cultural background [47], and design
trends [14, 27, 59]. Prior work has attempted to apply
rubric-guided ratings and rankings to collect designer feed-
back [10, 30] but found that subjectivity within the guide-
lines and differing preferences among designers led to a
noisy learning signal for model training [10, 60].
To better support visual design personalization, we in-
troduce DESIGNPREF, a dataset of 12,000 pairwise com-
parisons over generative UI designs, each with binary and
four-class preference ratings from one of 20 professional
graphic designers. Analysis reveals substantial disagree-
ment across trained designers (Krippendorff’s alpha = 0.25
for binary preferences). 28.5% of comparisons have 96%
or more pair-wise disagreement rate. We further collect the
rationale on top of pairs that raise highly divergent prefer-
ences. Our results show that designers can have different
preferences on color contrast (dark or bright), how dense
the screen should be (detailed vs simple), or how strongly
hierarchy is marked. To show the utility of individual de-
sign preference data in improving model performance, we
conduct several experiments. We show that finetuning an
existing design assessment model [60] on a designer’s indi-
1arXiv:2511.20513v1  [cs.CV]  25 Nov 2025

vidual labels leads to better preference prediction accuracy
and rating correlation than the same model trained on a 20
times larger aggregated dataset. We conduct a similar ex-
periment with an existing RAG-based UI assessment sys-
tem [10] and show similar improvements when retrieving
from a smaller, personalized pool of examples.
In summary, our work makes three contributions to-
wards personalizing design assessment. First, we intro-
duce DESIGNPREF, a benchmark of 12k identity-linked,
designer-authored pairwise comparisons with rationales
that supports research on personalized evaluation of visual
UI design. Second, we conduct an analysis of this dataset
where we quantify the prevalence of inter-rater disagree-
ments and perform a coding-based analysis of designers’ ra-
tionales to better understand the reasons behind divergence.
Finally, we conduct a series of modeling experiments that
demonstrate a smaller amount of personalized data leads to
better preference prediction than a 20 times larger aggre-
gated dataset in a finetuning and RAG setting.
2. Related Work
To contextualize our work, we review previous research on
automated UI design assessment, visual model-as-a-judge
systems, and personalized image assessment.
Automated Assessment of UI Design.Recent advances
in LLMs/LMMs make automatic UI design assessment fea-
sible. Early work prompts GPT-4 with written heuris-
tics; expert designers report helpful suggestions yet low
stability across revisions [11]. Followed with that, re-
searchers released UICrit, which is a dataset contains ex-
pert critiques with region boxes and quality ratings and
showed that few-shot and visual prompts raise LLM feed-
back quality [10]. Recent benchmark treated LMM as uni-
fied UI judges and found only moderate alignment with
human preferences across absolute and pairwise evalua-
tions [29, 60]. To make design quality modeling effective,
recent UI design autorater adopts UI- and quality-aware sig-
nals instead of generic vision-language objectives. UICLIP
introduced a design quality-oriented encoder that scores
prompt relevance and design quality and matched profes-
sional designer preferences better than off-the-shelf LMM-
as-a-judge while also producing design suggestions [60].
All these prior efforts focus on optimizing aggregated, uni-
versal visual design judges. Our work instead focuses on
modeling personal judgment, where we retain rater identity
and a four-level intensity scale to make models better pre-
dict each designer’s choices.
Model-as-a-Judge for Generative Visuals.Work on vi-
sual generation often relies on model judges that capture
populationpreferences and then steer and improve gener-ators with those rewards. Two lines of progress set the
stage. First, automatic evaluators replace generic prox-
ies with task-aware checks: TIFA asks a VQA system
questions about the prompt to measure faithfulness and
shows stronger agreement with humans than CLIP-style
metrics [18]. Second, preference-trained reward models
turn large-scale human choices into scalar scores: ImageRe-
ward, Pick-a-Pic, and HPSv2/v3 are examples of models
trained using expert pairwise comparisons and votes [23,
31, 62]. These types of automated judges can enable pref-
erence optimization for generative models [20, 52] without
hand-crafted rewards. Most of these works aim to learn a
single global utility and collapse diverse tastes into an aver-
age score. DESIGNPREFcomplements this setting by keep-
ing rater identity and a four-level confidence label on each
comparison, so a judge can learn whose preference to pre-
dict and at what strength. This personalized view matters
for design-centric visuals where trained professionals still
disagree at a high rate, and where downstream tuning can
potentially benefit from identity-aware targets.
Personalized Image Assessment.Personalized visual
preference research models user-specific taste instead of a
population mean. The field first framed personalized aes-
thetics as a supervised problem with rater identity: Ren et
al. introduced a task with AMT rater IDs and owner-rated
albums and showed that content and attribute cues predict
user-level deviations from a generic scorer [48]. PARA
broadened scale and subject coverage and added rich
user attributes, which enables identity-conditioned predic-
tion [64]. Personalized image quality assessment (IQA)
work has also trained models for specific age groups and
individuals [5, 57] to improve traditional metrics such as
NIMA, MUSIQ, and CLIP-IQA [21, 50, 55] In our pa-
per, we seek to extend this research to UI design assess-
ment, where individual preferences could be influenced
by a range of factors such as aesthetic taste, accessibility
needs [37, 42, 58], cultural background [47], and evolving
design trends [14, 27]. To this end, we contribute a dataset
of identify-linked designer preferences and reasoning which
we show can personalize VLMs through fine-tuning and
retrieval-augmented generation (RAG) approaches.
3. DesignPref Dataset
Our work aims to understand and model how individual de-
sign preferences arise for UIs. We introduce DesignPref, a
dataset that captures personalized preferences for generative
visual design. We focus on the domain of mobile UI, where
A/B testing and comparisons are standard practice [7, 22].
3.1. Dataset Construction
Existing UI datasets do not include annotations for prefer-
ence personalization, yet they provide screenshots, text de-
2

Figure 1. Our annotation interface includes a prompt that defines
the design task. Two variants generated from the prompt appear
side by side. The rater selects one of four preference options.
scriptions, and other metadata that we reuse to build con-
trolled pairs and collect judgments from individual design-
ers [2, 10, 53, 60]. The resulting dataset, methods, and
findings can be potentially extended to other graphic de-
sign domains such as presentation slides [12, 39, 40, 66]
and posters [16, 17]. We choose GPT-5 and Gemini-2.5-
Pro, two strong-performing language models from WebDev
Arena [28], as base generators. Each model produces ex-
ecutable UI code (HTML, CSS, and JavaScript) to render
mobile screens. Our study measures how people judge vi-
sual design quality and how those judgments vary across in-
dividuals, capturing shared criteria as well as personal taste.
3.1.1. Design Generation
We sampled 100 human-written screen descriptions from
existing dataset [53] and extended them to a more complete
screen descriptions using established method [41] as seed
prompts. To ensure coverage across UI types, we stratified
samples in different categories from prior defined taxonomy
[26]. For each description, each model produced four vari-
ants with a high temperature to increase diversity. We used
similar standard Design Arena [8] prompts to elicit high-
quality, renderable UI code and included explicit quality
checks in the instructions. In total, our dataset contains 400
generative designs. Each prompt yields 6 within-prompt
pairs, which gives 600 unique comparison pairs across all
generation.
3.1.2. Data Annotation Protocol
We recruited 20 designers via a university mailing list and
online platforms. Each designer who participated in our
study had at least one year of professional UI design experi-
ence. We designed our data annotation task to take approxi-
mately 90 minutes to complete. Designers who participated
Figure 2. Cohen’s kappa agreement for pairwise binary prefer-
ences across designers.
in our study were compensated $15.55 an hour, and our data
collection protocol was approved by our institution’s IRB.
Each designer viewed a pair of UI screens(A, B)in each
trial. The rater chose both direction and strength of prefer-
ence on a four-point Likert-scale (Fig. 1): “A much better
than B” (A≫B), “A better than B” (A>B), “B better than
A” (B>A) or “B much better than A” (B≫A). We chose
an even-numbered scale (i.e., no neutral option) to encour-
age raters to express even subtle preferences. The protocol
follows recent works that use fine-grained preference op-
tions for model alignment and evaluation [49, 51, 54]. We
linked every label to a persistent rater identifier to support
per-rater models and personalization analyses, consistent
with work that estimates judge-specific reliability in crowd-
sourcing settings [3, 4, 6, 45].
3.2. Dataset Analysis
Each designer rated 600 UI pairs under two label granu-
larities: a binary direction label that records which vari-
ant wins, and a four-way label that encodes preference
strength. Group-level reliability remains modest in both
schemes. For binary preferences, the mean pairwise agree-
ment is0.624, and both mean Cohen’sκand multi-rater
Krippendorff’sαequal0.248. For the four-way labels, the
corresponding values drop to0.386for agreement,0.114
forκ, and0.104forα. These values confirm substantial
disagreement among trained designers and align with prior
designer studies [10, 60]. Direction and four-wayκcorre-
late atr= 0.60, so designer pairs who agree on winners
also tend to agree on strength, although strength judgments
add additional noise.
Label-use patterns help explain the reliability gap be-
3

tween direction and four-way evaluation. Class frequen-
cies for the four-way choices are A>B36.97%, B>A
38.57%, A≫B11.82%, and B≫A12.64%. When aggre-
gated by strength,75.54%of judgments use the middle op-
tions (A>B or B>A), and24.46%use the extreme options
(A≫B or B≫A). Usage varies markedly across designers:
the middle-option share has mean75.38%with values from
6.00%to99.50%, and the extreme-option share has mean
24.41%with values from0.17%to93.67%. The distri-
bution suggests frequent reliance on slight preferences and
highly heterogeneous thresholds for strong claims.
Effect of Screen Type.We hypothesized that the type of
UI screen may have contributed to designers’ preference
distribution, as some screens (e.g., social media feeds) may
surface more variations than others (e.g., login screens). To
analyze the agreement level on different UI screen types, we
group items by screen types defined by prior taxonomy [26]
and measure direction consensus with the per-item mean
pairwise agreement and the entropy of binary votes, then
average within each type.Formscreens show the strongest
consensus, with mean pairwise agreement of 72.11% and
the highest share of extreme choices 32.71%.List,Di-
aler, andSearchscreens follow with agreement around
66%–68%. Designers disagree most onTermspages,Cam-
erainterfaces,Media Playerscreens, andLoginflows, all of
which stay near 54%–57% agreement. Across the 20 types,
mean agreement and the share of extreme judgments corre-
late strongly (r= 0.774), so screen types that yield clearer
winners also elicit stronger preference consensus. Overall,
structured layouts with clear primary actions (e.g., forms,
search pages) support more unified and decisive prefer-
ences, whereas visually complex or multifunctional screens
lead to more diffuse judgments.
3.3. Understanding Rationale Behind Preferences
We explore how designers explain their preferences with a
rationale annotation study. We invited the same 20 design-
ers from the preference study; 13 participated. The inter-
face resembled the original one but added a text box under
each selected pair. For each chosen comparison, designers
saw their earlier choice and wrote a brief rationale. To un-
derstand disagreements, we sampled highly contested pairs
with majority margin|choose A%−choose B%| ≤10%,
which corresponds to splits of 55% versus 45% or closer.
This process yielded 106 design pairs. Across these pairs,
designers wrote 1,378 rationales, with about 3 hours of
work per participant (we paid $15.55 per hour).
3.3.1. Common Themes in Preference Rationale
We characterize common patterns in the rationales with
a two-stage, embedding-based clustering pipeline that fol-
lows recent work on automatic and human-centered the-
Figure 3. Embedding clusters for preference rationale themes.
matic analysis [1, 38, 43, 56]. We split each rationale into
sentences and treat each sentence as a basic reasoning unit.
A transformer-based sentence encoder maps each unit into
a high-dimensional vector space [46]. We cluster these vec-
tors into fine-grained groups and ask GPT-5 to assign each
group a neutral label, short summary, and key concerns.
A second GPT-5 pass merges related groups into broader
themes. Our analysis of 1,378 written rationales surfaces
six recurrent themes (Figure 3), listed in order of how of-
ten designers mention them:clean layout and readabil-
ity,primary actions and controls,perceived modernity and
polish,color, contrast, and theme,information architec-
ture (IA) and scannability, andicon clarity and affordance.
Clean layout and readabilitycovers comments about gener-
ous spacing, clear alignment, and larger type.Primary ac-
tions and controlsreflects a preference for obvious, reach-
able call-to-actions that stand apart from secondary options.
When screens already satisfy basic readability,perceived
modernity and polishoften decides between functionally
similar variants. Designers often favored crisper typogra-
phy, smoother spacing, and cohesive accent colors and de-
scribe the preferred screen as more “modern” or “profes-
sional”.Color, contrast, and themefocuses on background
tone and palette cohesion, where contrast helps key con-
trols stand out.Information architecture and scannability
emphasizes clear sections and hierarchy, such as plan com-
parison screens with card-based layouts.Icon clarity and
affordancecaptures choices on dialer and camera screens,
where fewer, larger icons with higher contrast and short la-
bels make each control easier to interpret.
3.3.2. Divergence between Preference Rationale
While the previous analysis surfaces what designers gener-
ally agree on, contested pairs reveal structured disagreement
in how designers weigh the same concerns. We focus on
the selected contested pairs and treat each pair as a unit of
analysis. For each pair, we give GPT-5 the full set of ratio-
nales from designers on each side. The model describes the
main design trade-off in that pair and the main reasons re-
spondents favor each version. A second GPT-5 pass groups
related trade-offs into broader divergent themes. The anal-
ysis highlights four recurrent themes. The first theme con-
4

(a) In the upgrade screen, designers diverge mainly on preference for light
versus dark themes and contrast.
(b) In the Bible app screen, designers diverge between decorative imagery
and a more utility-focused layout.
Figure 4. Rationales behind divergent preferences. Each pair shows an example where half of the designers chose A and half chose B.
cernsinformation density. Some designers prefer calm lay-
outs with a short list of options or plans, while others favor
denser pricing or support screens that surface more chan-
nels or help paths in one view. The second theme concerns
visual style and tone. Across login, consent, and fintech
screens, some rationales emphasize light, neutral themes
that feel legible and trustworthy. Others praise darker or
more saturated palettes as more premium, expressive, or
more institutional language that signals seriousness (Fig-
ure 4a). The third theme concernsdecorative imagery ver-
sus focused utility. On some screens (e.g., media page), one
side values large photos, gradients, or illustrations that add
warmth and brand expression, while the other side prefers
restrained, text-led layouts where icons and controls keep
attention on the task or main goal of the app (Figure 4b).
The fourth theme concerns how strongly the interface fore-
groundsactions and task scope. Task-first rationales prefer
a single high-contrast primary button and minimal chrome,
such as camera screens with one clear capture control and
little status text. Feature-first rationales endorse richer lay-
outs that expose more modes, filters, alerts, or secondary
actions and describe very sparse screens as underpowered.
4. Modeling Approaches
In this section, we describe approaches for incorporating
DesignPref into machine learning models and pipelines to
improve their performance in predicting personalized de-
sign preferences. We focused on two example setups that
have been used by previous work to assess UI design qual-
ity: i) finetuning a CLIP-style dual-encoder VLM [60],
and ii) improving a decoder VLM/LMM with a multimodal
RAG pipeline [10].
Data Processing.We took steps to prepare the Design-
Pref for our modeling experiments. First, we processed our
collected data into a consistent format that could be used forfine-tuning and RAG pipelines.
Dpref={(t, IA, IB, c) :c∈ {−2,−1,1,2}}(1)
In the above equation, we construct tuples for every labeled
example consisting of the textt, image choicesIAandIB,
and the recorded choicecmapped to an integer value.
In addition, we created stratified data splits for our mod-
eling experiments. We chose to use60%train,20%val-
idation, and20%test split proportions to provide a large
enough sample size for measuring performance metrics.
Our splits were constructed so that all pairs containing the
a given screen ID belonged to the same split, preventing
data leakage. Across the entire dataset, this resulted in 7200
samples for the training set and 2400 samples for both the
validation and test splits. For each designer, this resulted in
360 samples for their training split and 120 samples for the
validation and test splits.
4.1. CLIP Finetuning
We show the utility of our preference data in improving a
CLIP-style dual-encoder VLM, which has previously been
used to assess UI design quality and relevance [60, 61]. The
CLIP architecture contains two transformer-based encoders
and previous work [60, 61] has used it to score UI by com-
puting the cosine similarity between its embedded screen-
shot and a natural language description of the screen, e.g.,
“ui screenshot. well-designed. settings page of an e-reader.”
Previous work [60] found that the original CLIP model
does not perform well on UI design assessment, and the
authors finetuned a derivative model, called UIClip, on a
large dataset of webpages screenshots with varying lev-
els of synthetically-induced design flaws to better calibrate
the predicted score [60]. In our experiments, we used
UIClip’s released model and inference code as a starting
point. Specifically, because we wanted to isolate the ef-
fect of our collected dataset, we chose to use UIClip’s pre-
5

training checkpoint1, which was not fine-tuned on the au-
thors’ previously collected human preferences. Our training
approach largely followed the pairwise contrastive learn-
ing approach used by UIClip to finetune on preference
pairs [60]. However, we incorporate additional granularity
from our data and better support model personalization.
4.1.1. Strength-aware Margin
First, we altered the originally proposed pairwise con-
trastive loss to a formulation with a cost-sensitive mar-
gin [19], which allowed us to incorporate the strength of
designers’ preferences. We used the following loss func-
tion for finetuning UIClip on preference examples from our
dataset.
L(sA, sB; ˆm) = max
0,ˆm−(sA−sB)	
,
ˆm=(
m1,if A>B,
m2,if A≫B.(2)
In the above equation,ˆmis the margin used for loss com-
putation, which depends on the ground truth label.sAand
sBrepresent the model’s predicted scores for a preferred
sampleAand rejected sampleB, respectively.
During model inference, the difference between pre-
dicted screens’ scores is used to make a four-way classi-
fication prediction.
ˆy(sA, sB; ˆm) =(
A>B,0≤sA−sB< m 2,
A≫B, sA−sB≥m 2.(3)
Negating these values leads to the thresholds for the op-
posite preference (e.g., B is better than A).
4.1.2. Model Personalization
In addition to using a strength-aware margin, we trained
separate models on their preference pairs to better reflect
their individual preferences. Due to the limited amount
of training data from each participant (360 examples), we
found that only unfreezing the last layer of UIClip dur-
ing training reduced the risk of overfitting and led to bet-
ter performance in our early experiments. We also intro-
duced other regularization measures, such as weight decay
to further combat overfitting. All designers’ models were
trained using the same set of hyperparameters and early
stopping schedule. This set of hyperparameters was ini-
tialized from previously published values [60] and further
refined through manual experimentation. Hyperparameter
values can be found in the supplemental material.
1https : / / huggingface . co / biglab / uiclip _
jitteredwebsites-2-224-paraphrased4.2. Retrieval-Augmented Generation
Prior work reports that decoder-style LMMs perform poorly
on zero-shot UI design assessment tasks [9–11, 60]. We hy-
pothesized that retrieval-augmented generation (RAG) with
our dataset could improve performance. Rather than rely-
ing only on knowledge stored in model weights, our RAG
setup retrieves a small set of labeled designer preference
examples as few-shot context for each query. We build this
retrieval pipeline on the design critique generation frame-
work from UICrit [10], and we adapt the prompt to pre-
dict pairwise design preferences instead of natural language
comments.
4.2.1. Sample Indexing
We largely followed UICrit’s approach to generating index
vectors using textual (screen description) and visual infor-
mation (UI screenshot). Following UICrit, we used an off-
the-shelf text-embedding model, Sentence BERT [46], to
generate a fixed-length embedding from the UI’s natural
language description. We chose to use UIClip’s visual en-
coder to generate embeddings for the screenshots, as op-
posed to the originally used CLIP model [10], since previ-
ous work suggested it had stronger UI screenshot retrieval
performance [60]. We used these three outputs to generate
two index vectors for each preference pair in the training
set, by concatenating the text embedding with both poten-
tial orderings of the screenshot embeddings. We used two
index vectors for each training example because we wanted
our retrieval metric to be invariant to the pair order. The
retrieval score for a training example was computed as the
maximum of the cosine similarity between the query vector
and two index vectors.
4.2.2. Personalized and Pooled Inference
During inference, a retrieval score was computed for each
example in training set and the results from the topkwere
injected into the input prompt. The prompt format can be
found in the supplemental materials of this paper. When
running RAG with a single designer’s labeled data, this pro-
cess resulted in exactlykretrieved samples, since each de-
signer was only asked to label a pair once. However, when
running RAG on the entire dataset ofndesigners’ data,k·n
samples are returned, since each designer labeled the exact
same pairs, which result in the same retrieval scores. In this
case, we aggregate designers’ labels by averaging all their
assigned scores into a single number then rounding to the
nearest integer. If the final score rounded to 0, then we ran-
domly chose between A<B and A>B for the few-shot
example.
5. Model Evaluations
To demonstrate the utility of our dataset in improving ML-
based UI design assessment, we conducted experiments that
6

Model Group Setup Preference Acc. (%) Four-way Acc. (%) SRCC
Personalized models
CLIP (UIClip)Personalized, strength-aware margin60.1634.370.217
Personalized, binary margin 57.24 28.89 0.180
LMM judges, RAGGPT-5 (8-shot) 58.89 38.53 0.211
Gemini-2.5-Pro (8-shot) 56.53 22.35 0.170
Qwen3-VL-235B-Thinking (8-shot) 56.8342.670.132
Qwen3-VL-30B-Thinking (8-shot) 56.20 40.41 0.128
Non-personalized models
CLIP (UIClip), pooledPooled, strength-aware (personal training size = N) 56.73 32.77 0.150
Pooled, strength-aware (full training set = 20N) 57.45 41.23 0.196
LMM judges, pooled RAGGPT-5 (8-shot) 57.62 34.36 0.203
Gemini-2.5-Pro (8-shot) 56.15 22.56 0.169
Qwen3-VL-235B-Thinking (8-shot) 55.78 41.59 0.122
Qwen3-VL-30B-Thinking (8-shot) 54.78 38.79 0.110
BaselinesUIClip (pretrained) 55.07 23.97 0.126
OpenAI CLIP B/32 46.68 23.26 -0.009
GPT-5 (zero-shot) 57.70 31.35 0.216
Gemini-2.5-Pro (zero-shot) 53.65 17.76 0.135
Qwen3-VL-235B-Thinking (zero-shot) 55.36 22.09 0.131
Qwen3-VL-30B-Thinking (zero-shot) 54.82 16.17 0.117
Table 1. Comparison of personalized and non-personalized models on each designer’s test split. We report binary accuracy, four-way
accuracy, and Spearman’s rank correlation coefficient (SRCC) between predictions and ground-truth preference labels.
compared data and model configurations with access to per-
sonalized data.
5.1. Evaluation Procedure
Tested Models.We tested several configurations and
baselines to measure the effect of our data on CLIP fine-
tuning and RAG. For our CLIP finetuning experiments, we
compared the following conditions:
1.Personalized, strength-aware margin.We fine-tuned
a separate UIClip model for each designer using our
strength-aware margin loss.
2.Personalized, normal margin.We fine-tuned a sepa-
rate UIClip model for each designer using a standard bi-
nary contrastive margin loss.
3.Pooled, full training set.We fine-tuned a single UIClip
model using the combined training data from all design-
ers.
4.Pooled, matched training size.We first combined the
training data from all designers then randomly selected
1/20 of the samples, to match the training size of a single
designer’s data.
We tested our RAG pipeline with four LMMs: GPT-5 [34],
Gemini-2.5-Pro [13], Qwen3-VL-235B-Thinking [63], and
Qwen3-VL-30B-Thinking. At the time of our experiments,
GPT-5 and Gemini-2.5-Pro were the top-performing pro-
prietary VLMs, and Qwen3-VL-235B and Qwen3-VL-30Bwere the top-performing open source VLMs that could fit
on a GPU server and consumer GPU. For each RAG model,
we varied the pool of data that the RAG system had access
to: i) each designer’s own training set (personalized) and
ii) all designers’ labels used our pooled inference strategy
(non-personalized). We set our RAG pipeline to retrieve 8
of the most relevant examples, following the best perform-
ing configuration found by previous work [10].
Performance Metrics.To measure model performance,
we chose three metrics based on those that have previously
been used to measure the alignment of automated scoring
metrics with human ratings [50]: i) binary accuracy, ii) four-
way accuracy, and iii) Spearman rank correlation coefficient
(SRCC) on four-way ratings. Metrics were computed indi-
vidually on each designers’ test split then averaged among
20 designers, i.e., macro-averaging.
5.2. Results
Table 1 shows the overall results. Our analysis focuses on
two questions: how designer-specific supervision compares
to pretrained or zero-shot judges, and whether personal-
ization helps when compared to global models pool labels
across designers.
7

5.2.1. Personalized vs. pretrained and zero-shot judges
Across backbones, designer-specific supervision improves
alignment with individual preferences. The per-designer
UIClip model with a strength-aware margin reaches 60.16%
binary accuracy and 0.217 SRCC, compared to 55.07%
and 0.126 for the frozen UIClip encoder and 46.68% and
−0.009for CLIP B/32. A personalized model that uses a
strength-aware loss performs better than the binary vari-
ant on all three metrics, suggesting that four-level labels
provide useful signal when the loss exposes preference
strength.
Personalized RAG with LMM on each designer’s prefer-
ence also shows various levels of improvements over zero-
shot LMM judges. For GPT-5, personalization raises binary
accuracy from 57.70% to 58.89% and four-way accuracy
from 31.35% to 38.53% with similar SRCC. Personaliza-
tion also helps both Qwen models gain about 1–2 points in
binary accuracy and more than 20 points in four-way accu-
racy when retrieval conditions on each designer’s own la-
bels, while SRCC stays roughly unchanged. The two se-
lected metrics show different aspects of the performance:
four-way accuracy measures exact matches to discrete pref-
erence strengths, reflecting class imbalance and design-
ers’ differing thresholds between slight and strong choices.
SRCC instead assesses how consistently model scores pre-
serve overall rankings across the four levels. Given uneven
distributions with most labels in the middle, and varied use
of extreme ratings among designers, strength predictions re-
main noisier than binary winner predictions.
5.2.2. Personalized vs. pooled models
Per-designer models consistently bring better personalized
prediction than global pooling. In UIClip, the per-designer
strength-aware model achieves higher binary accuracy and
SRCC than pooled four-class models, even when the pooled
judge sees roughly twenty times more labels per designer.
With a matched training set size, the personalized mod-
els improve all three metrics. RAG experiments with
LMM show similar pattern but with more marginal im-
provements. For GPT-5, Qwen-235B-Thinking, and Qwen-
30B-Thinking, retrieval over each designer’s own prefer-
ence labels yields higher binary and four-way accuracy and
equal or better SRCC than pooled RAG that averages labels
across designers. Gemini-2.5-Pro behaves close to neutral
under personalization, which suggests that the base model
already encodes a strong consensus prior. Overall, identity-
aware personalization helps both finetuned CLIP and few-
shot LLM judges align with individual design preferences
more closely than global pooling.
6. Limitations
Although our work suggests that personalization is impor-
tant for design, we note several limitations and provide av-enues for future improvement. First, our dataset is limited to
feedback from 20 designers. In contrast to existing crowd-
sourced design benchmarks [8, 28], we intentionally chose
to focus on skilled (and compensated) designers, who we
felt were more likely to understand and accurately apply
design guidelines. Yet, a larger sample size might facili-
tate new types of analyses, such as the emergence of “clus-
ters” of designers who might have similar design tastes, e.g.,
minimalism or Bauhaus school. Another limitation of our
data is that it focused primarily on labeling noise result-
ing from inter-rater disagreement; although personal uncer-
tainty might be another strong contributing factor. Anec-
dotally, we observed that designers in our study sometimes
changed their own answer during decision process, which
further suggests the difficulty of the task for ML models.
Our modeling experiments explored only a subset of pos-
sibilities enabled by DesignPref, and we specifically fo-
cused on replicating and improving two existing published
examples [10, 60]. Applying personalization approaches
led to the best-performing model (60.16% binary accuracy),
which is offers a similar level of performance as reward
models for challenging tasks such as math and instruction
following [24, 32]. Other modeling experiments, e.g., us-
ing preference data to finetune a decoder VLM, or using
designers’ critiques as reasoning traces are promising next
steps for further improving performance, and we expect our
dataset to enable additional future exploration.
Finally, our results show that, in our tested configura-
tions, a smaller amount of personalized data could lead to
better preference prediction than a 20 times larger aggre-
gated dataset. However, we did not attempt to study the
amount of individually-collected data needed for effective
model personalization. Each designer in our study spent
approximately 90 minutes to construct their dataset of 600
labels, a lengthy process for most practical applications. A
promising future direction is to study the scaling trend of
personal preference data or apply related techniques to im-
prove sample efficiency [21, 50, 55, 65].
7. Conclusion
DesignPref establishes a benchmark for personalized vi-
sual design evaluation with identity-linked judgments. The
dataset logs 12,000 pairwise UI comparisons from 20 pro-
fessional designers with four-class strength labels, which
enables per-designer modeling. Analyses show low cross-
designer consensus and reveal the agreement varies among
different designers and evaluated UI types. Per-designer
UIClip models beat the best pooled judges on held-out
pairs, showing the sample efficiency of personalized data.
DesignPref shifts evaluation and alignment from a single
global objective toward models that reflect individual taste
and lays foundations for personalized design generation.
8

References
[1] Doug Beeferman and Nabeel Gillani. Feedbackmap: A tool
for making sense of open-ended survey responses. InCom-
panion Publication of the 2023 Conference on Computer
Supported Cooperative Work and Social Computing, pages
395–397, 2023. 4
[2] Sara Bunian, Kai Li, Chaima Jemmali, Casper Harteveld,
Yun Fu, and Magy Seif Seif El-Nasr. Vins: Visual search
for mobile user interface design. InProceedings of the 2021
CHI Conference on Human Factors in Computing Systems,
pages 1–14, 2021. 3
[3] Franc ¸ois Caron and Arnaud Doucet. Efficient bayesian infer-
ence for generalized Bradley–Terry models.Journal of Com-
putational and Graphical Statistics, 21(1):174–196, 2012. 3
[4] Xi Chen, Paul N Bennett, Kevyn Collins-Thompson, and
Eric Horvitz. Pairwise ranking aggregation in a crowd-
sourced setting. InProceedings of the sixth ACM interna-
tional conference on Web search and data mining, pages
193–202, 2013. 3
[5] Olga Cherepkova, Seyed Ali Amirshahi, and Marius Peder-
sen. Individual contrast preferences in natural images.Jour-
nal of Imaging, 10(1):25, 2024. 2
[6] A. P. Dawid and A. M. Skene. Maximum likelihood estima-
tion of observer error-rates using the em algorithm.Journal
of the Royal Statistical Society: Series C (Applied Statistics),
28(1):20–28, 1979. 3
[7] Marcio Eduardo Delamaro, Jose Carlos Maldonado, and
Aditya P Mathur. Integration testing using interface muta-
tion. InProceedings of ISSRE’96: 7th International Sym-
posium on Software Reliability Engineering, pages 112–121.
IEEE, 1996. 2
[8] Design Arena. Design arena.https : / / www .
designarena.ai/, 2025. Accessed: 2025-11-05. 3, 8
[9] Peitong Duan, Chin-Yi Cheng, Bjoern Hartmann, and Yang
Li. Visual prompting with iterative refinement for design
critique generation.arXiv preprint arXiv:2412.16829, 2024.
6
[10] Peitong Duan, Chin-Yi Cheng, Gang Li, Bjoern Hartmann,
and Yang Li. Uicrit: Enhancing automated design evaluation
with a ui critique dataset. InProceedings of the 37th Annual
ACM Symposium on User Interface Software and Technol-
ogy, pages 1–17, 2024. 1, 2, 3, 5, 6, 7, 8
[11] Peitong Duan, Jeremy Warner, Yang Li, and Bjoern Hart-
mann. Generating automatic feedback on ui mockups with
large language models. InProceedings of the 2024 CHI Con-
ference on Human Factors in Computing Systems, pages 1–
20, 2024. 2, 6
[12] Jiaxin Ge, Zora Zhiruo Wang, Xuhui Zhou, Yi-Hao Peng,
Sanjay Subramanian, Qinyue Tan, Maarten Sap, Alane Suhr,
Daniel Fried, Graham Neubig, et al. Autopresent: Design-
ing structured visuals from scratch. InProceedings of the
Computer Vision and Pattern Recognition Conference, pages
2902–2911, 2025. 3
[13] Google DeepMind. Gemini 2.5 pro model card. Technical
report, Google DeepMind, 2025. Model card updated June
27, 2025. 7[14] Samuel Goree, Bardia Doosti, David Crandall, and Nor-
man Makoto Su. Investigating the homogenization of web
design: A mixed-methods approach. InProceedings of the
2021 CHI Conference on Human Factors in Computing Sys-
tems, pages 1–14, 2021. 1, 2
[15] Samuel Goree, Weslie Khoo, and David J Crandall. Cor-
rect for whom? subjectivity and the evaluation of personal-
ized image aesthetics assessment models. InProceedings of
the AAAI Conference on Artificial Intelligence, pages 11818–
11827, 2023. 1
[16] HsiaoYuan Hsu and Yuxin Peng. Postero: Structuring lay-
out trees to enable language models in generalized content-
aware layout generation. InProceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 8117–8127,
2025. 3
[17] Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and
Qing Zhang. Posterlayout: A new benchmark and approach
for content-aware visual-textual presentation layout. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 6018–6026, 2023. 3
[18] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari
Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accu-
rate and interpretable text-to-image faithfulness evaluation
with question answering. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pages 20406–
20417, 2023. 2
[19] Arya Iranmehr, Hamed Masnadi-Shirazi, and Nuno Vascon-
celos. Cost-sensitive support vector machines.Neurocom-
puting, 343:50–64, 2019. 6
[20] Shyamgopal Karthik, Huseyin Coskun, Zeynep Akata,
Sergey Tulyakov, Jian Ren, and Anil Kag. Scalable ranked
preference optimization for text-to-image generation. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 18399–18410, 2025. 2
[21] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and
Feng Yang. Musiq: Multi-scale image quality transformer.
InProceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pages 5128–5137, 2021. 2, 8
[22] Rochelle King, Elizabeth F Churchill, and Caitlin Tan.De-
signing with data: Improving the user experience with A/B
testing. ” O’Reilly Media, Inc.”, 2017. 2
[23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Ma-
tiana, Joe Penna, and Omer Levy. Pick-a-pic: An open
dataset of user preferences for text-to-image generation.Ad-
vances in neural information processing systems, 36:36652–
36663, 2023. 1, 2
[24] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, Lester
James Validad Miranda, Bill Yuchen Lin, Khyathi Chandu,
Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al.
Rewardbench: Evaluating reward models for language mod-
eling. InFindings of the Association for Computational Lin-
guistics: NAACL 2025, pages 1755–1797, 2025. 8
[25] Jun-Tae Lee and Chang-Su Kim. Image aesthetic assess-
ment based on pairwise comparison a unified approach to
score regression, binary classification, and personalization.
InProceedings of the IEEE/CVF International Conference
on Computer Vision, pages 1191–1200, 2019. 1
9

[26] Luis A Leiva, Asutosh Hota, and Antti Oulasvirta. Enrico:
A dataset for topic modeling of mobile ui designs. In22nd
International Conference on Human-Computer Interaction
with Mobile Devices and Services, pages 1–4, 2020. 3, 4
[27] Amanda Li, Yi-Hao Peng, Jeff Nichols, Jeff Bigham, and
Jason Wu. Waybackui: A dataset to support the longitudinal
analysis of web user interfaces, 2025. arXiv, 2025.12. 1, 2
[28] LMSYS Org. Webdev arena leaderboard.https://web.
lmarena.ai/leaderboard, 2025. 1, 3, 8
[29] Reuben A Luera, Ryan Rossi, Franck Dernoncourt,
Samyadeep Basu, Sungchul Kim, Subhojyoti Mukherjee,
Puneet Mathur, Ruiyi Zhang, Jihyung Kil, Nedim Lipka,
et al. Mllm as a ui judge: Benchmarking multimodal llms
for predicting human perception of user interfaces.arXiv
preprint arXiv:2510.08783, 2025. 2
[30] Kurt Luther, Amy Pavel, Wei Wu, Jari-Lee Tolentino, Ma-
neesh Agrawala, Bj ¨orn Hartmann, and Steven P. Dow.
Crowdcrit: crowdsourcing and aggregating visual design cri-
tique. InComputer Supported Cooperative Work, CSCW
’14, Baltimore, MD, USA, February 15–19, 2014, Compan-
ion Volume, pages 21–24. ACM, 2014. 1
[31] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li.
Hpsv3: Towards wide-spectrum human preference score. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 15086–15095, 2025. 2
[32] Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Mor-
rison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lam-
bert. Rewardbench 2: Advancing reward model evaluation.
arXiv preprint arXiv:2506.01937, 2025. 8
[33] Brad A Myers and Mary Beth Rosson. Survey on user in-
terface programming. InProceedings of the SIGCHI con-
ference on Human factors in computing systems, pages 195–
202, 1992. 1
[34] OpenAI. Gpt-5 system card. Technical report, OpenAI,
2025. 7
[35] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al. Training language
models to follow instructions with human feedback.Ad-
vances in neural information processing systems, 35:27730–
27744, 2022. 1
[36] Yi-Hao Peng, Muh-Tarng Lin, Yi Chen, TzuChuan Chen,
Pin Sung Ku, Paul Taele, Chin Guan Lim, and Mike Y Chen.
Personaltouch: Improving touchscreen usability by per-
sonalizing accessibility settings based on individual user’s
touchscreen interaction. InProceedings of the 2019 CHI
Conference on Human Factors in Computing Systems, pages
1–11, 2019. 1
[37] Yi-Hao Peng, Jeffrey P Bigham, and Amy Pavel. Slide-
cho: Flexible non-visual exploration of presentation videos.
InProceedings of the 23rd International ACM SIGACCESS
Conference on Computers and Accessibility, pages 1–12,
2021. 2
[38] Yi-Hao Peng, JiWoong Jang, Jeffrey P Bigham, and Amy
Pavel. Say it all: Feedback for improving non-visual presen-
tation accessibility. InProceedings of the 2021 CHI Confer-
ence on Human Factors in Computing Systems, pages 1–12,
2021. 4[39] Yi-Hao Peng, Jason Wu, Jeffrey Bigham, and Amy Pavel.
Diffscriber: Describing visual design changes to support
mixed-ability collaborative presentation authoring. InPro-
ceedings of the 35th Annual ACM Symposium on User Inter-
face Software and Technology, pages 1–13, 2022. 3
[40] Yi-Hao Peng, Peggy Chi, Anjuli Kannan, Meredith Ringel
Morris, and Irfan Essa. Slide gestalt: Automatic structure
extraction in slide decks for non-visual access. InProceed-
ings of the 2023 CHI Conference on Human Factors in Com-
puting Systems, pages 1–14, 2023. 3
[41] Yi-Hao Peng, Faria Huq, Yue Jiang, Jason Wu, Xin Yue Li,
Jeffrey P Bigham, and Amy Pavel. Dreamstruct: Under-
standing slides and user interfaces via synthetic data gener-
ation. InEuropean Conference on Computer Vision, pages
466–485. Springer, 2024. 3
[42] Yi-Hao Peng, Dingzeyu Li, Jeffrey P Bigham, and Amy
Pavel. Morae: Proactively pausing ui agents for user choices.
InProceedings of the 38th Annual ACM Symposium on User
Interface Software and Technology, pages 1–14, 2025. 2
[43] Tingrui Qiao, Caroline Walker, Chris Cunningham, and
Yun Sing Koh. Thematic-lm: a llm-based multi-agent sys-
tem for large-scale thematic analysis. InProceedings of the
ACM on Web Conference 2025, pages 649–658, 2025. 4
[44] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a
reward model.Advances in neural information processing
systems, 36:53728–53741, 2023. 1
[45] Vikas C. Raykar, Shipeng Yu, Linda H. Zhao, Germ ´an H.
Valadez, Charles Florin, Luca Bogoni, and Linda Moy.
Learning from crowds.Journal of Machine Learning Re-
search, 11:1297–1322, 2010. 3
[46] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks.arXiv preprint
arXiv:1908.10084, 2019. 4, 6
[47] Katharina Reinecke and Krzysztof Z Gajos. Quantifying vi-
sual preferences around the world. InProceedings of the
SIGCHI conference on human factors in computing systems,
pages 11–20, 2014. 1, 2
[48] Jian Ren, Xiaohui Shen, Zhe Lin, Radomir Mech, and
David J. Foran. Personalized image aesthetics. InProceed-
ings of the IEEE International Conference on Computer Vi-
sion (ICCV), pages 638–647, 2017. 2
[49] Yixiao Song, Yekyung Kim, and Mohit Iyer. Veriscore: Eval-
uating the factuality of verifiable claims in long-form text
generation. InFindings of the Association for Computational
Linguistics: EMNLP 2024, pages 9447–9474, 2024. 3
[50] Hossein Talebi and Peyman Milanfar. Nima: Neural image
assessment.IEEE Transactions on Image Processing, 27(8):
3998–4011, 2018. 2, 7, 8
[51] Hugo Touvron, Louis Martin, Kevin Stone, et al.
Llama 2: Open foundation and fine-tuned chat models.
arXiv:2307.09288, 2023. 3
[52] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou,
Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming
Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model align-
ment using direct preference optimization. InProceedings of
10

the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8228–8238, 2024. 2
[53] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi
Grossman, and Yang Li. Screen2words: Automatic mobile
ui summarization with multimodal learning. InThe 34th An-
nual ACM Symposium on User Interface Software and Tech-
nology, pages 498–510, 2021. 3
[54] Chenglong Wang, Yang Gan, Yifu Huo, Yongyu Mu,
Qiaozhi He, Murun Yang, Tong Xiao, Chunliang Zhang,
Tongran Liu, and Jingbo Zhu. Lrhp: Learning rep-
resentations for human preferences via preference pairs.
arXiv:2410.04503, 2024. 3
[55] Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy.
Exploring CLIP for assessing the look and feel of images.
InProceedings of the AAAI Conference on Artificial Intelli-
gence, pages 2555–2563, 2023. 2, 8
[56] Qile Wang, Moath Erqsous, Kenneth E Barner, and
Matthew Louis Mauriello. Lata: A pilot study on llm-
assisted thematic analysis of online social network data gen-
eration experiences.Proceedings of the ACM on Human-
Computer Interaction, 9(2):1–28, 2025. 4
[57] Yinan Wang, Andrei Chubarau, Hyunjin Yoo, Tara Akhavan,
and James Clark. Age-specific perceptual image quality as-
sessment. InIS&T International Symposium on Electronic
Imaging 2023, Image Quality and System Performance XX,
2023. 2
[58] Jacob O Wobbrock, Shaun K Kane, Krzysztof Z Gajos,
Susumu Harada, and Jon Froehlich. Ability-based design:
Concept, principles and examples.ACM Transactions on Ac-
cessible Computing (TACCESS), 3(3):1–27, 2011. 1, 2
[59] Jason Wu, Siyan Wang, Siman Shen, Yi-Hao Peng, Jeffrey
Nichols, and Jeffrey P Bigham. Webui: A dataset for en-
hancing visual ui understanding with web semantics. InPro-
ceedings of the 2023 CHI Conference on Human Factors in
Computing Systems, pages 1–14, 2023. 1
[60] Jason Wu, Yi-Hao Peng, Xin Yue Amanda Li, Amanda
Swearngin, Jeffrey P Bigham, and Jeffrey Nichols. Uiclip:
a data-driven model for assessing user interface design. In
Proceedings of the 37th Annual ACM Symposium on User
Interface Software and Technology, pages 1–16, 2024. 1, 2,
3, 5, 6, 8
[61] Jason Wu, Eldon Schoop, Alan Leung, Titus Barik, Jeffrey P
Bigham, and Jeffrey Nichols. Uicoder: Finetuning large lan-
guage models to generate user interface code through auto-
mated feedback.arXiv preprint arXiv:2406.07739, 2024. 1,
5
[62] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai
Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagere-
ward: Learning and evaluating human preferences for text-
to-image generation.Advances in Neural Information Pro-
cessing Systems, 36:15903–15935, 2023. 2
[63] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025. 7
[64] Yuzhe Yang, Liwu Xu, Leida Li, Nan Qie, Yaqian Li,
Peng Zhang, and Yandong Guo. Personalized image aes-
thetics assessment with rich attributes.arXiv preprint
arXiv:2203.16754, 2022. 2[65] Jooyeol Yun and Jaegul Choo. Scaling up personalized im-
age aesthetic assessment via task vector customization.arXiv
preprint arXiv:2407.07176, 2024. 8
[66] Jooyeol Yun, Heng Wang, Yotaro Shimose, Jaegul Choo,
and Shingo Takamatsu. Designlab: Designing slides
through iterative detection and correction.arXiv preprint
arXiv:2507.17202, 2025. 1, 3
[67] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan
Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with
mt-bench and chatbot arena.Advances in neural information
processing systems, 36:46595–46623, 2023. 1
11

DesignPref: Capturing Personal Preferences in Visual Design Generation
Supplementary Material
S1. Model Hyperparameters
Here are the hyperparameteres we used for training and prompting VLMs:
Table 2. Training hyperparameters for UICLIP.
ModelUICLIP
Learning rate5×10−4
ScheduleCosine (T max= 24,η min= 10−4)
Weight decay10−2
Batch size64
Gradient clipping1.0
Patience5
Margin multiplier1.1 for “much better” labels
All the LMM’s parameters for randomness or reasoning are set to default.
S2. Prompt Sets
Visual UI Design Generation
You are an expert mobile UI designer and developer.
Goal
• Output ONE standalone HTML file that renders ONE static mobile screen.
Canvas
• Wrap all content in
<div class="ui-canvas">...</div>.
• Size: 628x1118 px. Center the canvas on the page.
• No horizontal or vertical scroll. No overlap or cropping. Keep all content inside.ui-canvas.
• Include a small guard style to center the canvas and hide page overflow.
Tech
• Use only HTML, CSS, and JavaScript. No build tools.
• Use Tailwind via CDN in<head>:<script src=”https://cdn.tailwindcss.com”></script>.
• If you use icons, import the icon library via CDN before use (e.g., Heroicons) or inline SVG.
• If you add charts or 3D, prefer D3/Recharts/Three.js via CDN U.
• Use semantic tags. Bind labels to controls withfor/id.
• Provide meaningful alt text for images.
• Meet WCAG criteria for color contrast and content style. Show visible:focus-visibleoutlines.
• Group radios byname. Switches use native checkbox under the hood; reflect checked/disabled; label is clickable.
• Sliders use<input type="range">with a label and a live value.
• Tabs or segmented controls show selected state and support keyboard focus.
• If a modal exists, trap focus, provide ESC close, and a close button.
Images (Assets)
• If the user provides asset URLs, use ONLY those URLs in<img src="...">. Provide concise, specific alt text.
• Do not reference Unsplash/Picsum/placeholder services. Do not leave emptysrcattributes.
Return
• Return ONLY valid HTML from<!DOCTYPE html>to</html>. No commentary or Markdown.
1

UI Preference Judging Prompts
(A) Zero-shot UI Judge
Developer instructions.The model rates UIs for a generic population:
• Choose the screen most people would prefer.
• Use a rubric over clarity, readability, spacing and alignment, emphasis of primary actions, aesthetic balance, and fit to the screen
prompt.
• Evaluate only visible features, avoid position bias, do not allow ties, and keep reasons short and concrete.
• Print ONLY the fixed output template.
User message.The user provides one screen prompt (optional) and two images (Image A and Image B), then the output template:
•CHOICE 4WAY: <A >> B | A > B | B > A | B >> A>
•BINARY PREFERENCE: <A | B>
•CONFIDENCE: <0.00-1.00>
•REASONS:with 2–3 short bullet points.
(B) Few-shot UI Judge (Personalized or Pooled)
Developer instructions.The model adapts to one user’s taste:
• Infer this user’s preference pattern from labeled examples and apply it to new pairs.
• When the general rubric disagrees with the user’s past choices, follow the user’s taste.
• Use the same rubric only as a tie breaker, and obey the same rules on visibility, bias, ties, and concise reasons as above.
User message with examples.The prompt first lists several labeled examples for that user:
• For each example:Example i (user-labeled), optionalScreen prompt, images A and B, andUser’s label:
<A >> B | A > B | B > A | B >> A>(score in{−2,−1,1,2}).
Then the prompt introduces the target pair:
•Now predict this user’s preference for the TARGET pair.
•Screen prompt, images A and B, and the same fixed output template:CHOICE 4WAY,BINARY PREFERENCE,
CONFIDENCE, andREASONS.
S3. Guidelines for Preference Annotations
Preference Ratings Labeling (1 out of 4 options)
1. Based on your personal preferences, click the option or use 1,2,3,4 to select the preferences 1:A >> B,2 :A >
B,3 :A < B,4 :A << B
2. Use left and right to go forward and backward
3. Focus less on the quality of the placeholder content (e.g., generated text) and more on the overall design (layout, color
choices, style ..etc); Sometimes some images may not even be loaded properly. Instead of focusing on the unloaded
assets, focus on the overall design
4. If you feel both of them are of equal quality because they both contain flaws, think about which one you would rather
have as a starting point if it was your job to fix it.
Rationale Specification for Preference Choices
1. For each pair of UIs, you see your previous preference from Study 1. You may change the choice by clicking a button
or pressing keys 1–4.
2. Refer to the designs as “screen A” and “screen B” (or “image A” and “image B”). You may also write simply “A” and
“B” in your explanation.
3. Write a rationale of at least 2–3 sentences for every choice. Explain why you prefer one screen and why you chose
that strength of preference.
4. Give concrete visual and UI reasons, such as layout, color, hierarchy, typography, spacing, alignment, or missing
components from the prompt. Avoid short generic statements like “A looks better than B” without details.
5. If both screens have flaws, choose the one you would prefer as a starting point to fix and explain why. Very short or
vague rationales may be rejected and may not receive compensation.
2
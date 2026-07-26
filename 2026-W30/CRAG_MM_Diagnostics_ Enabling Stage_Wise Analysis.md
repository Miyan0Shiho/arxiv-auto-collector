# CRAG-MM-Diagnostics: Enabling Stage-Wise Analysis of Knowledge-Intensive VQA

**Authors**: Hanseok Oh, Parishad BehnamGhader, Benno Krojer, Hyunji Lee, Paul Liang, Siva Reddy, Verna Dankers

**Published**: 2026-07-23 10:37:01

**PDF URL**: [https://arxiv.org/pdf/2607.21155v1](https://arxiv.org/pdf/2607.21155v1)

## Abstract
Knowledge-Intensive Visual Question Answering (KI-VQA) benchmarks evaluate Vision-Language Models (VLMs) as multimodal knowledge assistants by requiring external information beyond a provided image to answer questions. KI-VQA involves multiple sub-problems -referring expression understanding, visual grounding, object recognition, knowledge retrieval, and reasoning-yet existing benchmarks typically report only end-task accuracy, obscuring where failures arise. To analyze the full KI-VQA pipeline, we introduce CRAG-MM-Diagnostics, a diagnostic benchmark with stage-wise data annotations that isolate 1) language-based visual grounding, 2) object identification, and 3) knowledge retrieval and reasoning. We evaluate fully parametric and retrieval-augmented VLMs, providing fine-grained analyses using newly collected metadata, such as target ROIs, entity names, and visual complexity scores. Our results point to knowledge retrieval and reasoning as the primary bottleneck, but also highlight issues in the other parts of the KI-VQA pipeline, such as the fact that VLMs struggle with target object identification or that image retrievers struggle to integrate textual cues. These findings expose fundamental limitations in current KI-VQA systems and motivate stage-aware evaluation. We, lastly, leverage these findings to propose a grounded bimodal RAG pipeline that integrates a visual grounding module to crop targets before image retrieval, boosting GPT-5 and Qwen's respective accuracies by 13.3 and 8.5 percentage points.

## Full Text


<!-- PDF content starts -->

CRAG-MM-Diagnostics: Enabling Stage-Wise
Analysis of Knowledge-Intensive VQA
Hanseok Oh1,3⋆, Parishad BehnamGhader2,3, Benno Krojer2,3,
Hyunji Lee4, Paul Liang5, Siva Reddy2,3,6, and Verna Dankers2,3
1New York University, New York NY, USA
2McGill University, Montreal, Canada
3Mila - Quebec AI Institute, Montreal, Canada
4UNC Chapel Hill, Chapel Hill NC, USA
5MIT, Cambridge MA, USA
6Canada CIFAR AI Chair
hanseok.oh@nyu.edu
McGill-NLP/crag-mm-diagnostics
 CRAG-MM-Diagnostics
Abstract.Knowledge-Intensive Visual Question Answering(KI-VQA)
benchmarks evaluateVision–Language Models(VLMs) as multimodal
knowledge assistants by requiring external information beyond a pro-
videdimagetoanswerquestions.KI-VQAinvolvesmultiplesub-problems
—referring expression understanding, visual grounding, object recogni-
tion, knowledge retrieval, and reasoning—yet existing benchmarks typ-
ically report only end-task accuracy, obscuring where failures arise. To
analyze thefullKI-VQA pipeline, we introduceCRAG-MM-Diagnostics,
a diagnostic benchmark with stage-wise data annotations that isolate
1language-based visual grounding, 2object identification, and 3
knowledge retrieval and reasoning. We evaluate fully parametric and
retrieval-augmented VLMs, providing fine-grained analyses using newly
collected metadata, such as target ROIs, entity names, and visual com-
plexity scores. Our results point to knowledge retrieval and reasoning as
the primary bottleneck, but also highlight issues in the other parts of
the KI-VQA pipeline, such as the fact that VLMs struggle with target
object identification or that image retrievers struggle to integrate textual
cues. These findings expose fundamental limitations in current KI-VQA
systems and motivate stage-aware evaluation. We, lastly, leverage these
findings to propose a grounded bimodal RAG pipeline that integrates a
visual grounding module to crop targets before image retrieval, boosting
GPT-5andQwen’s respective accuracies by 13.3 and 8.5 percentage points.
Keywords:Knowledge-intensive VQA·Retrieval Augmented Genera-
tion·Vision-Language Models
1 Introduction
Recent advances inVision-Language Models(VLMs) have accelerated interest
in their deployment in real-world applications [2,6,29,40]. A core requirement
⋆Work conducted during an internship at Mila.
arXiv:2607.21155v1  [cs.CV]  23 Jul 2026

2 H. Oh et al.
"Is this sold
in South
America?"
"No, the Lexus GX 470
is sold in North
American and Eurasian
markets."
Object identification
Do models know which target object this is?
We evaluate entity recognition accuracy, decoupling it
from stage 1 by providing ROIs explicitly.2
"Lexus GX 470"
Knowledge retrieval & reasoning
Can models answer knowledge-intensive questions?
We evaluate this decoupled from stages 1 and 2 via
verbose textual-only questions.3"Is the Lexus GX
470 sold in
South America?"KI-VQA benchmarks normally
only evaluate end-task
accuracy...Instead, we enable evaluation of the
entire KI-VQA pipeline, consisting of
the following stages:For stage-based analysis we release
CRAG-MM-Diagnostics, a benchmark with
a range of metadata, such as:
target ROIs
target entity
name/URL
text-only
ablationdisambiguated
linguistic
expressionsLanguage-based visual grounding
Do models know which part of the image this linguistic
expression denotes?
We enable stage-based evaluation through ROI accuray,
and analyses across linguistic and visual complexity.
We examine the effect of linguistic ambiguity by
disambiguating ambiguous questions.1
"Is the grey
car sold in
South America?"
Fig. 1:Illustrative summary of the stage-wise evaluation enabled byCRAG-MM-
Diagnostics, thanks to new types of metadata we add. The three stages evaluated
are (1) language-based visual grounding, (2) object identification, and (3) knowledge
extraction & reasoning, and are further detailed in the figure.
for these assistants is the ability to interpret visual scenes and provide informa-
tive responses, typically evaluated throughVisual Question Answering(VQA)
benchmarks. However, while standard VQA often focuses on describing visible
attributes, there is growing attention forKnowledge-IntensiveVQA (KI-VQA).
In this setting, the question cannot be answered solely from the image but also
requires external world knowledge [4,13,24,26,41]. KI-VQA is a vital step to-
wards a future in which VQA is seamlessly integrated in daily life, in which one
can ask complex questions on the go, when walking down the street using smart
glasses, or texting photos back and forth with a friend.
However, existing KI-VQA benchmarks often fail to capture real-world visual
complexity, instead relying on images with salient, centered objects and minimal
clutter [4,24,26]. In contrast, real-world scenes contain multiple, often visually
similarobjects,withlesssalientandoff-centertargets(seeFigure1).Challenging
benchmarks that do include visually complex images often focus only on end-
task evaluation [11,40], obscuring the root causes of failure (i.e., whether errors
stemfromflawedvisualgrounding,incorrectentityidentification,ordownstream
reasoning). For example, when a model fails to answer the question in Figure 1,
is the issue target identification or missing knowledge?
In this work, we take a step towards a more fine-grained analysis of KI-VQA,
aiming to understand how models perform in intermediate stages of the pipeline.
Unlike the majority of KI-VQA studies that only evaluate end-task accuracy, we
focus on three intermediate stages: 1language-based visual grounding,
used to localize the target region of interest in a complex image [31,43]; 2ob-
ject identification, used to map the localized region to a canonical entity (e.g.,
a Wikipedia page) [26,29]; and 3knowledge retrieval and reasoning, which
answersthequestionafterpossibletextualandvisualinformationretrieval[4,24].
To enable this analysis, we introduceCRAG-MM-Diagnostics, a benchmark built

CRAG-MM-Diagnostics3
on top ofCRAG-MM[40], as described in Section 3. WhileCRAG-MMprovides a
foundation for KI-VQA in egocentric, visually complex scenes, it serves pri-
marily as an end-to-end benchmark without the structural markers needed to
diagnose internal model failures.CRAG-MM-Diagnosticstransforms this resource
into a diagnostic framework by introducing stage-specific annotations. Figure 1
illustrates the three stages, and a small subset of the new metadata.
Using the extended benchmark, we conduct a comprehensive analysis for the
three stages in Sections 4, 5 and 6, using fully parametric VLMs, models spe-
cialized for each stage, and retrieval-augmented models. Our analysis uncovers
several fundamental limitations, among which:
–Visualgroundingoftargetobjectsismorechallengingforproprietarymodels;
–Linguistic ambiguity in QA pairs affects the entire KI-VQA pipeline;
–VLMs, in particular compared to specialized models, struggle to identify
target objects that are unpopular;
–Localizingthetargetobjectimprovesimageretrievalsubstantially,especially
in ambiguous queries;
–MostKI-VQAerrorsareduetoinadequateknowledgeretrievalandreasoning
rather than incorrect object identification (e.g., only 20.9% ofGPT-5’s errors
can be resolved by providing the ground-truth target object name).
By providing a stage-aware analysis, we offer actionable insights for the de-
sign of robust KI-VQA systems. In Section 6, we leverage our findings to study
a modeling pipeline combining components from all stages to achieve better
performance, boostingGPT-5andQwen’s respective accuracies by 13.3 and 8.5
percentage points. We conclude in Section 7.
2 Related Work
In this section, we discuss the evolution of KI-VQA and recent advances in
diagnostic evaluation for VLMs. We provide a more detailed comparison in Ap-
pendix A.
2.1 Knowledge-Intensive Visual Question Answering Benchmarks
Since the introduction of VQA [1], later benchmarks have increased difficulty by
requiring external knowledge, leading to the development ofknowledge-intensive
VQA (KI-VQA).OK-VQA[24] focuses on commonsense and simple factual
knowledge. Subsequent datasets impose more specialized knowledge demands:
EncyclopedicVQA[26] andSnapNTell[33] target long-tail entities (e.g., “Pinus
pinea”),whileViQuAE[16]andInfoSeek[4]emphasizenamedentitiesandrealis-
tic information-seeking scenarios. These benchmarks typically require retrieving
external knowledge.
Despite this progress, most KI-VQA benchmarks increase difficulty primar-
ily along thelanguage axis—e.g., by requiring complex reasoning [39] or in-
cluding temporally evolving knowledge [18]—while implicitly assuming object

4 H. Oh et al.
recognition is trivial or based on salient cues [4,26,33]. In contrast, fine-grained
visual benchmarks focus on challenges along thevisual axis. Datasets such as
V*[43],GigaGrounding[21], andMME-RealWorld[49] focus on visually chal-
lenging scenarios, including crowded, high-resolution, or remote-sensing images
where targets are small or non-salient, requiring precise localization. However,
these benchmarks are mainly studied outside information-seeking settings.
Suchvisualdifficultybecomesevenmorecriticalwhenretrievalisinvolved,as
external tools must correctly identify subtle targets under noisy conditions—an
aspect largely overlooked in KI-VQA. Recent multimodalRetrieval Augmented
Generation(RAG) benchmarks, such asMRAG-Bench[11] andCRAG-MM
[40], incorporate realistic visual challenges like viewpoint shifts, occlusion, and
egocentric perspectives. However, these benchmarks lack fine-grained diagnos-
tics to attribute failures to specific visual factors or to disentangle visual and
linguistic errors. We, therefore, presentCRAG-MM-Diagnostics, which provides
fine-grained stage-wise diagnostics forlanguage-based visual grounding,object
identification, andknowledge retrieval and reasoning.
2.2 Diagnostic evaluation of VLMs
Recent studies have proposed diagnostic evaluations of intermediate stages to
better understand VLMs’ capabilities in multimodal tasks.MMVet[48] anno-
tates examples from 16 multimodal tasks with six core vision–language skills,
which are not specifically curated for the KI-VQA task.HallusionBench[8] de-
composes hallucinations into language hallucination and visual illusion to ana-
lyze failure modes. Similarly,Prism[32] disentangles perception from reasoning
in general VQA. Despite these advances, diagnostic evaluation forknowledge-
intensiveinformation seeking tasks remains limited. Existing evaluations are
often confined to isolated visual recognition tasks [10] or rely solely on final
QA accuracy [4,24,26], overlooking realistic KI-VQA scenarios that require pre-
cise grounding and contextual target identification [22,46]. The recent KI-VQA
benchmarkVisualSimpleQA[42] partially addresses this gap by comparing mul-
timodal and text-only questions, but still focuses on end-task accuracy and is
limited to parametric VLMs.
3 The curation ofCRAG-MM-Diagnostics
To facilitate fine-grained KI-VQA analysis, we develop a diagnostic benchmark
(CRAG-MM-Diagnostics) by augmenting and refiningCRAG-MM[40].CRAG-MMcon-
sists of diverse (image, question, answer) triplets spanning 13 domains, including
nearly 1.9K single-turn egocentric images that mimic captures from wearable de-
vices. This egocentric setting enablesrealisticKI-VQA evaluation and provides
sufficient visual diversity and complexity to analyze performance across different
levels of visual saliency—e.g., by varying object sizes or scene crowdedness. How-
ever, it only permits end-to-end QA analysis, making it impossible to investigate
the ability of different KI-VQA components separately.

CRAG-MM-Diagnostics5
Human annotations.To ensure the benchmark’s quality, we apply an initial
filtering process prior to human annotation, removing samples that are not
knowledge-intensiveor concern dynamically changing knowledge, as detailed in
Appendix A.2. We then manually annotate textual metadata and target object
regions using bounding boxes.7Each sample in the dataset is annotated once by
an author and consecutively double-checked by another. The metadata added
through human annotation, illustrated using Figure 1, includes:
1.Entity name, Wikipedia URL: the target object’s name and correspond-
ing Wikipedia URL (e.g., “Lexus GX 470” in case of Figure 1).
2.Referring expression type: the queries’ labels that distinguishsalient
cases (e.g., where “this van” suffices as a referral to identify a unique object)
fromambiguousones(e.g.,thenon-specificexpression“this” inFigure1),and
ones where there is anin-image cue(e.g., “the red car”) from aknowledge-
intensive cue(e.g., recognizing the service associated with a logo to identify
that “this restaurant” refers to a specific building). See Table 2 for examples.
3.Disambiguated question: for examples with ambiguous referring expres-
sion, we add a disambiguated question. For instance, for Figure 1, this ques-
tion replaces “this car” with “the grey car on the right”.
4.Text-only question: for all examples, we add a question which allows mod-
els to rely on the textual modality only, by inserting the correct entity into
thequestion,e.g.,“IstheLexusGX470soldinSouthAmerica?” forFigure1.
5.Target region: in addition to the textual metadata mentioned above, the
human annotation also involves the selection of a bounding box in which the
target object is contained, as previously demonstrated in Figure 1.
After annotation, we removed 95 samples due to missing Wikipedia links
for the target entity, mismatched image-QA pairs, questions that do not require
visualinformation,low-qualityimagesthathinderlocalization(e.g.,duetoblur),
and time-dependent questions (e.g., “last year”) identified during the annotation.
Automated annotations.Additionally, we automatically augment the data with
metadata capturing an object’s popularity, and its visual complexity:
6.Target popularity: the Wikipedia page views in 2025, following [23,33].
7.Target object size: measured as the ratio of the target’s bounding box
area to the image area.
8.Distance to center: computed as the distance between the target centroid
and the image center.
9.Scenecrowdedness:approximatedbythenumberofobjectsdetectedusing
the open-set detector Grounding-DINO [20].
Using the three visual complexity dimensions (metadata 7–9), we define a sum-
mary metric,visual saliency, to quantify a target’s prominence in an image
(see Appendix A.4). After filtering and annotation,CRAG-MM-Diagnosticscon-
tains 1,149 samples. Dataset statistics are shown in Figure 8; example QA pairs,
further metadata details, the annotation protocol and the annotation tool are
described in Appendix A.
7Some metadata annotations were model-assisted via preliminary annotations, as
detailed in Appendix A.4.

6 H. Oh et al.
salient in-image cue KI cue ambiguous all020406080IoU  0.5 (%)
Llama 11B
Qwen 3BQwen 7B
Qwen 32BQwen 72B
GPT-5GPT-5-mini
G-DINOOwlVIT
Fig. 2:Grounding accuracy per referring expression type (horizontal lines: mean;
hatched: disambiguated query). It shows that: (1) Drops on non-salient and ambiguous
targets reveal high sensitivity to expression type. (2) Gains from disambiguation sug-
gest many failures arise from query ambiguity rather than visual processing deficits.
4 Stage 1Language-based Visual Grounding
We useCRAG-MM-Diagnosticsto systematically evaluate the KI-VQA stages (as
shown in Figure 1), beginning with the system’s ability to localize and iden-
tify the subject of a referring expression within a visual scene. We assess this
language-based visual groundingability by measuring how accurately mod-
els predict the bounding box of a target object across varying linguistic and
visual variations. We first describe the experimental setup.
4.1 Experimental Preliminaries
Models.Our evaluation spans both generalized VLMs (which are capable of
end-to-end KI-VQA) and specialized models designed specifically for this step
of the pipeline (i.e., localization). We examine the performance of both, to in-
form building KI-VQA solutions that integrate specialized models, which we will
perform later on, in Section 6 (when performing stage 3evaluation).
Among thegeneralized modelsthat jointly encode images and text, we in-
clude open-source architecturesLlama-3.2-11B[27] and theQwen2.5-VLfamily
(ranging from 3B to 72B parameters) [2], alongside proprietary state-of-the-
art modelsGPT-5andGPT-5-mini. Thespecialized modelsfocus explicitly
on alignment between language and visual regions:Grounding-DINO[20] and
OWL-ViT[28] perform open-vocabulary object detection by aligning text embed-
dings with image features for precise localization. Detailed prompts and details
for this task are included in Appendix C; the prompts provide models with the
question and image, and require outputting precise bounding box coordinates.
Evaluation metric.Following established related work on visual grounding eval-
uation [21,44], we measureIntersection over Union(IoU) between the predicted
and ground-truth bounding boxes of the target object. A prediction is classified
as correct if the IoU exceeds a threshold of 0.5.

CRAG-MM-Diagnostics7
salient in-image cue KI cue ambiguous all0.00.20.40.60.8IoU  0.5 (%)
Llama 11B
Qwen 3BQwen 7B
Qwen 32BQwen 72B
GPT-5GPT-5-mini
G-DINOOwlVIT
13
36
612
1220
20105
crowdedness204060IoU  0.5 (%)
00.1
0.10.1
0.10.2
0.20.3
0.30.8
target distance to center020406080IoU  0.5 (%)
00
00.2
0.20.4
0.40.7
0.71
target size0255075IoU  0.5 (%)
0.10.4
0.40.5
0.50.6
0.60.7
0.71
target saliency0255075IoU  0.5 (%)
Fig. 3:Visual grounding scores along four dimensions of visual complexity (black
squares indicate model mean). These plots show that low target saliency is the pri-
mary bottleneck, acutely compounding performance drops caused by scene density,
off-center placement, and small target size.
4.2 Analyses and Findings
Figure 2 details the IoU per model, per referring expression type, and across
all examples through the ‘all’ bar. TheQwen2.5-VLfamily demonstrates ro-
bust grounding across all sizes, peaking at 67.8% accuracy.Llama-3.2and the
GPT-5variants, however, have markedly lower performance. The fact that even
the smallestQwenis quite good at this task is likely because the model family
was explicitly trained to understand bounding boxes [2]; theGPT-5results, in
particular, underscore that spatial understanding of images does not naturally
emerge, even in SOTA models. Even ifGPT-5canperform KI-VQA in general,
the fact that it struggles with explicit grounding in this first stage limits the
explainability and interpretability of the model. When contrasting the gener-
alized and specialized models, it is noteworthy thatGrounding-DINOperforms
remarkably well; despite its modest 0.2B parameters, it rivals theQwenmodels in
accuracy, presenting a highly efficient alternative for modular KI-VQA pipelines
that include grounding.
Referring expressions highly influence performance.Overall performance ob-
scures the impact of thetypeof referring expression. The ‘referring expression’
metadata we include inCRAG-MM-Diagnosticsdistinguishes salient cases from
more complex ones, which constitute 36.6% of the dataset. Appendix A.3 pro-
vides concrete examples of the different types of expressions. Figure 2 shows
that while salient targets are the most straightforward to ground, ambiguous
expressions represent a severe challenge. This gap is especially pronounced for
smaller models (e.g.,Qwen3B and 7B), highlighting the role of model capac-
ity in handling linguistic nuance and multimodal integration. Moreover, results
demonstrate that althoughGrounding-DINOperforms similarly to theQwenfam-
ily overall, it falls short of the larger models in samples with complex referring
expressions (i.e., in the case of a ‘knowledge-intensive cue’ and ‘ambiguous’ ex-
amples).

8 H. Oh et al.
Ambiguity is a bottleneck for visual grounding.To isolate the effects of am-
biguity, we compared performance on samples with ambiguous queries against
disambiguated ones (e.g., changing “this” to “the grey car on the right” in Fig-
ure 1). Disambiguation resulted in a substantial performance surge, a mean
44.0% improvement (see hatched bars in Figure 2). This indicates that a lack of
linguistic specificity is a primary bottleneck. The gains are largest forQwenmod-
els, whereas disambiguated queries do not improve performance forLlamaand
GPT-5-mini, underscoring that their subpar performance is a more fundamental
issue with these models’ spatial reasoning abilities rather than merely the result
of ambiguity. The presence of ambiguous queries inCRAG-MMis likely due to the
data’s egocentric nature. If a QA pair does not capture the user’s gaze or intent,
evaluation using the original queries may underestimate models’ true KI-VQA
performance.
Grounding degrades as visual complexity grows.Language-based visual ground-
ing depends not only on the referring expression but also on image complex-
ity and target saliency [21,43]. Figure 3 demonstrates this for the four visual
complexity axes previously introduced in Section 3. Performance decreases with
increased crowdedness and distance—withpoint biserial(pb) correlations of
rpb=−.117and−.254, respectively, averaged over models—and improves with
larger size and higher saliency—r pb=.422and.391.
Intermediate takeaways based on visual grounding (stage 1) evaluation:
1. VLMs lack spatial awareness without being explicitly trained for this, as
LlamaandGPT-5(-mini)struggle a lot more thanQwenwhen predicting
bounding boxes;
2. Complex referring expressions and ambiguity negatively affect visual
grounding;
3. Inlinewithpriorwork[21,43]variousfactorsofvisualcomplexitydegrade
grounding performance;
4.Grounding-DINObest balances performance with efficiency.
5 Stage 2Object Identification
Next, we study the models’ ability to performtarget object identification;
a vital step towards KI-VQA since the vast majority ofCRAG-MM-Diagnostics
questions concern specific named entities. Here, we first elaborate on the experi-
mental setup prior to diving into our findings. We evaluate models by providing
the original image and question, alongside task-specific instructions (detailed in
Appendix D), and require the models to output the name of the target entity.
5.1 Experimental Preliminaries
Models.Consistent with our previous analysis, we evaluate two model classes.
Firstly, we considergeneralized models(see Section 4.1), which rely solely on

CRAG-MM-Diagnostics9
internal parametric knowledge. Evaluating them on object identification, there-
fore, isolates a step that these models normally perform implicitly during KI-
VQA. In contrast,specialized modelsretrieve images and metadata from ex-
ternal knowledge sources. They implicitly perform object identification when at
least one retrieved item corresponds to the target. Although this does not yield a
direct comparison with generalized models, it allows us to examine complemen-
tary strengths of parametric vs. retrieval-based identification. The specialized
models are the unimodal retrieverCLIP-ViT-Large-Patch14-336[34] and the
multimodal retrieversVLM2Vec-V2.0[25] andQwen-3-VL-Embedding-2B[17]. We
use theCRAG-MMimage knowledge graph [40] as the retrieval corpus and recom-
pute embeddings for each model.
Evaluation metric.For generalized models, we measure entity prediction accu-
racy using exact string match. For cases without an exact match, we utilize
LLM-as-a-judge (GPT-4o-mini) to determine semantic equivalence. For special-
ized retrieval models, we reportrecall@10. Specifically, we perform normalized
partial string matching between the ground-truth entity name and entity names
associated with the top-10 retrieved images following related work [7,37]. We do
not apply the LLM-as-a-judge approach here, as verifying semantic equivalence
for all top-kresults from retrieval sources is very computationally expensive.
5.2 Analyses and Findings
Before diving into fine-grained analyses, Figure 4a reveals the overall perfor-
mance and performance for examples with ambiguous referring expressions. Al-
though proprietary models struggled with visual grounding in Section 4, they
now achieve the strongest object identification performance. Grounding being a
prerequisite for object identification suggests that while these models lack the
specialized ‘language’ of coordinate-based grounding, they possess a strong im-
plicit grounding capability that facilitates identification. Yet, evenGPT-5strug-
gles to predict the correct entity name for approximately 40% of the examples,
which will have implications for the end-task KI-VQA accuracy down the line.
VLMs benefit from multimodal cues.Next, we analyze the contribution of each
modality to object identification for the generalized models (Figure 4b):
-∆cropping: Providing a crop of the target region instead of the full im-
age results in a slight net decrease in accuracy for most models, though it
predictably aids ambiguous stimuli. This suggests that models might ben-
efit from contextual cues beyond the target region for performing accurate
object identification.
-∆question: Replacing the original query with a simplified prompt (e.g.,
“What is this object?”) reduces accuracy by 7.9 percentage points. This con-
firms that models leverage both visual and textual cues to refine their visual
identification. An example of a useful textual cue is, for instance, if the ques-
tion refers to an “SUV”, greatly restricting the set of possible target objects.

10 H. Oh et al.
ambiguous all020406080accuracy (%)Llama 11B
Qwen 3B
Qwen 7B
Qwen 32BQwen 72B
GPT-5-mini
GPT-5
(a)OI performance
cropping
(all)question
(all)cropping
(ambig.)disambig.
15
10
5
051015accuracy increase (b)Effect of modifying the question/image
16.2k
6.2k 58k
58k202k
202k 450k
450k 20.7M
popularity of object20406080100accuracy (%)
Llama 11B
Qwen 3B
Qwen 7B
Qwen 32BQwen 72B
GPT-5
GPT-5-mini (c)Accuracy vs popularity
Fig. 4:Performance breakdown of generalized models on the stage 2object identifi-
cation task. (b) Accuracy changes under input manipulations:∆‘cropping’ marks the
accuracy change after cropping the target object;∆‘question’ marks the change after
simplifying the question;∆‘disambig.’ marks the change after using the disambiguated
questions for the ambiguous stimuli. (c) Accuracy across target object popularity bins.
-∆disambiguated:Usingdisambiguatedquestionsprovidesperformancegains
nearly on par with those of cropping, suggesting that ambiguous examples
cause an underestimation of a model’s latent KI-VQA capabilities.
VLMs are affected by objects’ popularity.Next, we utilize other types of meta-
data to further understand what does or does not affect object identification
performance when the generalized VLMs perform the task. Unlike the ground-
ing task in Stage 1, object identification is less sensitive to visual complexity; in
fact, saliency is (only weakly) negatively related to accuracy (r pb=−.168) (de-
scribedinAppendixD.1).Aclearertrend,however,existsfortheimpactofpopu-
larityasdepictedinFigure4c(withanaverageSpearman’sρof0.314):VLMsare
better at identifying entities with high Wikipedia page-view counts, consistent
with established literature on long-tail visual knowledge distribution [10,26,33].
Image retrievers benefit from visual isolation and struggle with linguistic com-
prehension.Specialized retrievers present a different performance profile. Fig-
ure 5a shows recall@10 for all examples versus ambiguous cases using image-only
queries.8These results confirm that object identification is a challenging task for
retrievers, whileCLIP’s performance stands out as relatively strong, when taking
into account the size gap to the other models (427M vs 2B). Notably, ambiguous
cases suffer from poor performance even without textual queries, likely because
these instances are more visually complex.9Figure 5b demonstrates performance
changes when varying the input to the specialized models:
-∆cropping: Unlike generalized models, specialized retrievers are consistently
and positively impacted by cropping, especially for ambiguous, visually com-
plex stimuli.
8Detailed performance of different input variants and performance trends across dif-
ferent top-kvalues are described in Appendix D.2.
9Mean visual saliency is 0.38 for ambiguous cases vs. 0.58 for non-ambiguous ones.

CRAG-MM-Diagnostics11
ambiguous all010203040recall@10 (%)CLIP
VLM2VEC2
Qwen-3-VL-2B
(a)OI performance
cropping
(all)text
(all)cropping
(ambig.)10
0102030recall@10 increase (b)Effect of modifying the question/image
0.10.4
0.40.5
0.50.6
0.60.7
0.71
target saliency020406080100recall@10(%)
CLIP
VLM2VEC2
Qwen-3-VL-2BCLIP crop
VLM2VEC2 crop
Qwen-3-VL-2B crop (c)Recall vs saliency
Fig. 5:Stage 2object identification results for specialized models. (b) Accuracy
change under input manipulations:∆‘cropping’ marks the change after cropping the
target object;∆‘text’ adds the question for the multimodal models. (c) Recall@10
across visual saliency bins.
-∆text: Surprisingly, providing bimodal retrievers (VLM2Vec2,Qwen-3-VL)
with the original question does not improve performance, unlike in general-
izedmodels.Thislikelyreflectslimitedlinguisticgroundingability,ascurrent
multimodal retrievers are not explicitly optimized to align textual cues with
precise region-level retrieval.
Cropping helps specialized models when targets are non-salient.Lastly, we revisit
the role of visual complexity and popularity, and analyze how they affect the
specialized models. Visual complexity and popularity only weakly affect the per-
formance of specialized models (Spearman’sρ= 0.05for popularity,r pb= 0.13
for visual saliency) (described in Appendix D.2), yet further inspection of the
relation between recall and visual saliency reveals a more subtle trend, as de-
picted in Figure 5c: the more non-salient a target object is, the more cropping
helps.
Intermediate takeaways based on object identification (stage 2):
1. All VLMs struggle with object identification, a prerequisite for successful
KI-VQA.CLIPappears relatively strong considering its small size;
2. Whereas VLMs do best when receiving rich multimodal input (full im-
age and question), specialized image retrievers perform best when only
receiving a cropped target object image;
3. Visual complexity plays less of a role than in stage 1, whereas target
popularity acts as a critical bottleneck—affecting VLMs far more acutely
than image retrievers.
6 Stage 3Knowledge Retrieval and Reasoning
The final KI-VQA stage requires synthesizing an answer byreasoning over
knowledge. We assess this stage through end-task performance, while providing

12 H. Oh et al.
inin
in
Query What were the first
three generations of the
car in front of the brown
car?Image
outGrounding Model
out
Image Retriever
(Wikimedia)in1.       
2. ...   
3. ...   
10. ...
Text Retriever
(Webcrawl, Wikipedia)
Queryinin1. Corolla is a
car made by ...
2. Toyota is a
company from ...
3. E10 is the
first
generation...
...
10. ...outVLM
in
Query + ImageAnswerThe first
three generations of
the Toyota Corolla
were ...
inmetadata
metadata
metadata
...entity name
entity name
entity name
...a
bcd
Fig. 6:Grounded bimodal RAG: In stage 3, we study a modeling pipeline that com-
bines VLMs with RAG and visual grounding. Adding the grounding module is meant
to make the image retriever more effective.
ananalysisthatisolatesitfrompreviousstagesusing‘text-onlyquery’metadata.
Below, we detail the experimental setup before elaborating on our findings.
6.1 Experimental Preliminaries
Models and RAG pipeline.We evaluate the same suite ofgeneralizedVLMs as
in Section 4.1. These models are first tested in a zero-shotbaseline configuration,
relying solely on their parametric knowledge to answer the KI-VQA questions.
We then equip these VLMs withRAG, utilizing text and image retriev-
ers as established in the originalCRAG-MMframework [40]. Given an input im-
age, the image retriever returns similar images and structured metadata from
a knowledge graph usingCLIP-ViT-Large-Patch14-336[34]. The text retriever
similarly returns webpages given an input text, using documents embedded by
BGE-large-en-v1.5[45]. We employ the originalCRAG-MMtext and image indices.
Standard RAG can be limited by imperfect image retrieval (as observed in
Section 5). Informed by Sections 4 and 5, we evaluate a grounded bimodal RAG
configuration(illustratedinFigure6)thatintegratesanupstreamvisualground-
ing module to improve retrieval similar to region-aware retrieval [12,14,19].10
This pipeline follows these steps:
aLanguage-based visual grounding:Grounding-DINOgenerates a precise crop
of the target region to eliminate visual noise.
bImage retrieval: The retriever picks images and metadata for cropped target.
cText retrieval: The text retriever fetches passages using the query and the
metadata from the visual retrieval step.
dMultimodal reasoning: The VLM generates the final answer by synthesizing
the query, the original image, and the retrieved context.
10Direct empirical comparison to related frameworks, such as [12,14,19], is precluded
by unavailable public codebases or incomplete implementation details; however, they
represent compelling directions for future benchmarking.

CRAG-MM-Diagnostics13
20 30 40 50 60
QA Accuracy (%)Llama 11B
Qwen 3B
Qwen 7B
Qwen 32B
Qwen 72B
GPT-5-mini
GPT-525.4 29.1
21.320.4
18.8 24.9
18.918.3
20.4 30.5
17.917.7
26.2 34.7
20.5 27.4
31.1 43.0
25.6 30.4
40.9 50.8
35.2 37.1
48.9 59.6
44.0 44.5txt+img
txt-onlyambiguous
disambiguated
(a)Generalized modelsModel All Subset
ground.
img ret.
txt ret.
txt+img
txt-only
ambig.
dis.Qwen 32B✗ ✗ ✗26.2 34.7 20.5 27.4
✗ ✓ ✗27.2-22.7 27.5
✗ ✗ ✓31.0 57.2 25.933.6
✗ ✓ ✓34.4-24.5 29.6
✓ ✓ ✓34.7-28.029.9
GT✓✓ 36.1- 32.8 33.6GPT-5✗ ✗ ✗48.9 59.6 44.0 44.5
✗ ✓ ✗61.5-50.9 53.9
✗ ✗ ✓56.7 73.3 53.157.1
✗ ✓ ✓61.5-46.4 55.2
✓ ✓ ✓62.2-53.956.5
GT✓✓ 62.2- 55.7 58.4
(b)Ablations on grounded RAG components
Fig. 7:Comparison of (a) generalized models and (b) RAG pipeline ablations on KI-
VQA accuracy.Allreports performance over all instances;Subsetevaluates ambiguous
(ambig.) vs. disambiguated (dis.) queries. Columnsground.,img ret., andtxt ret.de-
note the activation of grounding, image retrieval, and text retrieval modules.txt-only
evaluates questions with gold entity names replacing image, whiletxt+imguses the
standard VQA setup. Shaded rows indicate use of ground-truth (GT) target regions.
Evaluation metric.Following [40], we evaluate QA accuracy usinggpt-4o-mini
as a binary classifier given the question, ground-truth, and prediction (prompt
in Appendix F).11Reported scores are averaged over three runs of the judge.
6.2 Analyses and Findings
The end-task accuracies in Figure 7 underscore the difficulty of out-of-the-box
KI-VQA, even for proprietary models (see ‘txt+img’ in Figure 7a). Although
ambiguous expressions (‘ambiguous’ marker) degrade performance, the modest
1.8-point gain from disambiguation suggests ambiguity is not the primary bot-
tleneck.
Retrieval/reasoning is the primary KI-VQA bottleneck.Using the ‘text-only
questions’ fromCRAG-MM-Diagnostics—where target entity names are explicitly
mentioned—decouples retrieval and reasoning from stage 1and2localization
errors.Figure7a(‘txt-only’marker)showsthatwhilethisimprovesresultsby8.7
points on average, the majority of errors persist. Notably,GPT-5improves most,
with 20.9% of errors disappearing. This underscores that parametric knowledge
11[40] reports a 99.1% human agreement rate for this method.

14 H. Oh et al.
is insufficient for most KI-VQA tasks, particularly for smaller models with lim-
ited capacity.
Unimodal RAG yields modest gains.Figure 7b demonstrates the impact of cou-
pling VLMs with image and text retrievers. Image retrieval provides a consistent
6.8-point average boost forQwenandGPT-5. Additionally, text retrieval alone
adds 6.3 points; however, its effectiveness is limited by uninformative queries
(e.g., “Is this sold in South America?” in Figure 1). By using ‘text-only’ ques-
tions with explicit entity names (replacing “white vehicle” with “Tesla Model Y”
in “How does the range of the white vehicle compare to that of the hyundai
ioniq 5?”), performance jumps by 16.6 and 26.2 points forGPT-5andQwen, re-
spectively.Theseresultsemphasizethataccuratequeryreformulationandobject
identification are critical for maximizing retrieval gains.
Bimodal and grounded RAG yield top scores.Combining bimodal retrieval with
visual grounding achieves our highest performance (Figure 7b), confirming that
‘cleaning’ visual inputs affects the KI-VQA positively. Using ground-truth re-
gions (GT, highlighted in gray) provides an additional 1.4-point gain forQwen,
yet performance still trails the text-only retrieval baseline. Since image retrieval
is implicitly intended to serve object identification, this reinforces that it does
not succeed in that task, even when given a ground-truth target region. The
error analysis in Appendix G shows that 25% of failures stem from poor object
identificationevenwithGPT-5.Thishighlightstheneedformorerobustrecogniz-
ers handling fine-grained taxonomies and effective retrieval to bridge parametric
knowledge gaps.
Intermediate takeaways from the knowledge retrieval and reasoning stage:
1. Even proprietary models likeGPT-5see substantial gains when target
entity names are explicitly provided, indicating that internal knowledge
cannot substitute for effective retrieval in KI-VQA;
2. Integrating a visual grounding module to crop target objects before re-
trieval consistently yields the highest performance, demonstrating that
‘cleaning’ the visual input is vital for accurate multimodal RAG;
3. Even with ground-truth target regions, performance lags behind text-
only retrieval baselines, revealing that current retrieval pipelines still fail
at fine-grained object identification and reasoning.
7 Conclusion
We introducedCRAG-MM-Diagnostics, a diagnostic benchmark designed for the
stage-wise evaluation of KI-VQA under realistic visual conditions. By augment-
ingCRAG-MMwith fine-grained annotations—including bounding boxes, refer-
ring expression types, disambiguated or text-only queries, and visual saliency
metadata—we conducted a systematic analysis of the KI-VQA pipeline. Our

CRAG-MM-Diagnostics15
analysis spanned fully parametric VLMs, specialized grounding and retrieval
models, and modular retrieval-augmented pipelines.
Our findings surface many patterns that explain why KI-VQA is complex,
suchasthevisualcomplexityofthescene,thepopularityofthetargetobject,and
the complexity of the referring expression directly correlating with performance
in the various stages of the pipeline. We also uncovered failure modes (such
as the surprising find thatGPT-5struggles much more with explicit grounding
thanQwen) and determined that for some stages, a small, specialized model can
outperform larger, generalized VLMs. We end with the following three concrete
lessons for the design of future KI-VQA systems:
1.Linguistic ambiguity is a structural problem, not a minor nuisance.
Disambiguating referring expressions yields consistent gains across all stages
of the pipeline. Future benchmarks should audit and annotate query ambi-
guity, and models should explicitly address it as ignoring ambiguity will lead
to consistent underestimation of models’ reasoning capacity. We advocate
for models that explicitly detect and resolve ambiguity—potentially through
iterative query reformulation—prior to knowledge retrieval.
2.Object identification should inform text retrieval.Text retrieval with
original, image-dependent queries is often uninformative when the textual
question alone does not identify the target (e.g., “Is this car sold in South
America?”). Using the target entity name in the retrieval query—derived
from either model prediction or early-stage identification—yields the largest
performance gains observed in our study. This ‘identify first, then retrieve’
ordering is the most impactful design principle we uncover.
3.Knowledge retrieval and reasoning is the dominant bottleneck.
Critically, even when models are provided the ground-truth entity name, the
majority of errors persist, indicating that limited parametric knowledge and
imperfect reasoning—not visual grounding or object identification—account
for most KI-VQA failures. This suggests that the field should invest more in
improving retrieval quality and multi-hop reasoning over retrieved evidence.
By providing a foundation forstage-awareKI-VQA evaluation,CRAG-MM-
Diagnosticsencourages a shift from ‘black-box’ testing to principled, modular
assessment. We hope this benchmark serves as a catalyst for developing mul-
timodal knowledge assistants that can truly navigate the intersection of visual
perception and world knowledge.
Acknowledgements
We thank the reviewers for their valuable suggestions. We thank our colleagues
from McGillNLP and Mila, especially Marius Mosbach, Vaibhav Adlakha, Ra-
biul Awal, and Aishwarya Agrawal. PB is supported by the RBC Borealis AI
Global Fellowship Award and the ServiceNow-Mitacs Accelerate program. SR is
supported by the Canada CIFAR AI Chair, the NSERC Discovery Grant and the
Sloan Fellowship. The project is partly funded by the IVADO R3AI program.
VD acknowledges the support of the IVADO Postdoctoral Research Funding.

16 H. Oh et al.
References
1. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.:
VQA: Visual question answering. In: Proceedings of the 2015 IEEE International
Conference on Computer Vision (ICCV). p. 2425–2433. ICCV ’15, IEEE Computer
Society, USA (2015).https://doi.org/10.1109/ICCV.2015.279,https://doi.org/
10.1109/ICCV.2015.2793
2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang,
S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W.,
Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H.,
Lin, J.: Qwen2.5-vl technical report. arXiv preprint (2025),https://arxiv.org/
abs/2502.139231, 6, 7
3. Bhattacharya, N., Li, Q., Gurari, D.: Why does a visual question have different
answers? In: Proceedings of the IEEE/CVF International Conference on Com-
puter Vision (ICCV) (October 2019),https://openaccess.thecvf.com/content_
ICCV_2019/papers/Bhattacharya_Why_Does_a_Visual_Question_Have_Different_
Answers_ICCV_2019_paper.pdf22
4. Chen,Y.,Hu,H.,Luan,Y.,Sun,H.,Changpinyo,S.,Ritter,A.,Chang,M.W.:Can
pre-trained vision and language models answer visual information-seeking ques-
tions? In: Bouamor, H., Pino, J., Bali, K. (eds.) Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language Processing. pp. 14948–14968.
Association for Computational Linguistics, Singapore (Dec 2023).https://doi.
org/10.18653/v1/2023.emnlp-main.925,https://aclanthology.org/2023.emnlp-
main.925/2, 3, 4, 22, 28
5. Chroma: Chroma: The AI-native open-source embedding database.,https://www.
trychroma.com/31
6. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muen-
nighoff,N.,Lo,K.,Soldaini,L.,Lu,J.,Anderson,T.,Bransom,E.,Ehsani,K.,Ngo,
H., Chen, Y., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R.,
Bastani, F., VanderBilt, E., Lambert, N., Chou, Y., Chheda, A., Sparks, J., Skjons-
berg, S., Schmitz, M., Sarnat, A., Bischoff, B., Walsh, P., Newell, C., Wolters, P.,
Gupta, T., Zeng, K.H., Borchardt, J., Groeneveld, D., Nam, C., Lebrecht, S., Wit-
tlif, C., Schoenick, C., Michel, O., Krishna, R., Weihs, L., Smith, N.A., Hajishirzi,
H., Girshick, R., Farhadi, A., Kembhavi, A.: Molmo and PixMo: Open weights
and open data for state-of-the-art vision-language models. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
pp. 91–104 (June 2025),https://openaccess.thecvf.com/content/CVPR2025/
papers/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-
the-Art_CVPR_2025_paper.pdf1
7. Ding, Y., Ren, K., Huang, J., Luo, S., Han, S.C.: MMVQA: A comprehensive
dataset for investigating multipage multimodal information retrieval in pdf-based
visual question answering. In: Larson, K. (ed.) Proceedings of the Thirty-Third
International Joint Conference on Artificial Intelligence, IJCAI-24. pp. 6243–6251.
International Joint Conferences on Artificial Intelligence Organization (8 2024).
https://doi.org/10.24963/ijcai.2024/690,https://doi.org/10.24963/ijcai.
2024/690, main Track 9
8. Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang,
F., Yacoob, Y., Manocha, D., Zhou, T.: Hallusionbench: An advanced diagnos-
tic suite for entangled language hallucination and visual illusion in large vision-
language models. In: Proceedings of the IEEE/CVF Conference on Computer

CRAG-MM-Diagnostics17
Vision and Pattern Recognition (CVPR). pp. 14375–14385 (June 2024),https:
//openaccess.thecvf.com/content/CVPR2024/papers/Guan_HallusionBench_An_
Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_
2024_paper.pdf4, 22
9. Honnibal, M., Montani, I., Van Landeghem, S., Boyd, A.: spaCy: Industrial-
strength Natural Language Processing in Python (2020).https://doi.org/10.
5281/zenodo.121230323
10. Hu, H., Luan, Y., Chen, Y., Khandelwal, U., Joshi, M., Lee, K., Toutanova, K.,
Chang, M.W.: Open-domain visual entity recognition: Towards recognizing mil-
lions of wikipedia entities. In: Proceedings of the IEEE/CVF International Con-
ference on Computer Vision (ICCV). pp. 12065–12075 (October 2023),https:
//openaccess.thecvf.com/content/ICCV2023/papers/Hu_Open-domain_Visual_
Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_
ICCV_2023_paper.pdf4, 10, 22
11. Hu, W., Gu, J.C., Dou, Z.Y., Fayyaz, M., Lu, P., Chang, K.W., Peng,
N.V.: MRAG-Bench: Vision-centric evaluation for retrieval-augmented multi-
modal models. In: Yue, Y., Garg, A., Peng, N., Sha, F., Yu, R. (eds.) In-
ternational Conference on Learning Representations. vol. 2025, pp. 95558–
95581 (2025),https://proceedings.iclr.cc/paper_files/paper/2025/file/
ee46288ab2aaf5c6e53aebebe719712c-Paper-Conference.pdf2, 4, 22, 29
12. Jian, P., Yu, D., Zhang, J.: Large language models know what is key visual en-
tity: An LLM-assisted multimodal retrieval for VQA. In: Al-Onaizan, Y., Bansal,
M., Chen, Y.N. (eds.) Proceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing. pp. 10939–10956. Association for Computa-
tional Linguistics, Miami, Florida, USA (Nov 2024).https://doi.org/10.18653/
v1/2024.emnlp-main.613,https://aclanthology.org/2024.emnlp-main.613/12
13. Kabir, R., Haque, N., Islam, M.S., et al.: A comprehensive survey on visual ques-
tion answering datasets and algorithms. arXiv preprint (2024),https://api.
semanticscholar.org/CorpusID:2741318622
14. Kim,J.,Tao,R.,Sharma,S.,Wang,J.,Sun,K.,Lin,Z.,Moon,S.,Mathias,L.,Ku-
mar, A., Ji, H., et al.: Pixel-grounded retrieval for knowledgeable large multimodal
models. arXiv preprint (2026),https://arxiv.org/abs/2601.1906012
15. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J., Zhang,
H., Stoica, I.: Efficient memory management for large language model serving with
PagedAttention. In: Proceedings of the 29th Symposium on Operating Systems
Principles.p.611–626.SOSP’23,AssociationforComputingMachinery,NewYork,
NY, USA (2023).https://doi.org/10.1145/3600006.3613165,https://doi.org/
10.1145/3600006.361316530
16. Lerner, P., Ferret, O., Guinaudeau, C., Le Borgne, H., Besançon, R., Moreno,
J.G., Lovón Melgarejo, J.: ViQuAE, a dataset for knowledge-based visual ques-
tion answering about named entities. In: Proceedings of the 45th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
p. 3108–3120. SIGIR ’22, Association for Computing Machinery, New York, NY,
USA (2022).https://doi.org/10.1145/3477495.3531753,https://doi.org/10.
1145/3477495.35317533, 22
17. Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A.,
Liu, D., et al.: Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified frame-
work for state-of-the-art multimodal retrieval and ranking. arXiv preprint (2026),
https://arxiv.org/abs/2601.047209, 33

18 H. Oh et al.
18. Li, Y., Li, Y., Wang, X., Jiang, Y., Zhang, Z., Zheng, X., Wang, H., Zheng,
H.T., Huang, F., Zhou, J., Yu, P.S.: Benchmarking multimodal retrieval aug-
mented generation with dynamic VQA dataset and self-adaptive planning agent.
In: The Thirteenth International Conference on Learning Representations (2025),
https://openreview.net/forum?id=VvDEuyVXkG3, 22
19. Lin, L., Xie, Y., Chen, D., Xu, Y., Zhu, C., Yuan, L.: REVIVE: Regional vi-
sual representation matters in knowledge-based visual question answering. In: Oh,
A.H., Agarwal, A., Belgrave, D., Cho, K. (eds.) Advances in Neural Information
Processing Systems (2022),https://openreview.net/forum?id=wwyiEyK-G5D12
20. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang,
J., Su, H., Zhu, J., Zhang, L.: Grounding DINO: Marrying DINO with grounded
pre-training for open-set object detection. In: Computer Vision – ECCV 2024: 18th
European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings,
Part XLVII. p. 38–55. Springer-Verlag, Berlin, Heidelberg (2024).https://doi.
org/10.1007/978-3-031-72970-6_3,https://doi.org/10.1007/978-3-031-72970-
6_35, 6, 23, 27
21. Ma, T., Bai, B., Lin, H., Wang, H., Wang, Y., Luo, L., Fang, L.: When visual
grounding meets gigapixel-level large-scale scenes: Benchmark and approach. In:
ProceedingsoftheIEEE/CVFConferenceonComputerVisionandPatternRecog-
nition (CVPR). pp. 22119–22128 (June 2024),https://openaccess.thecvf.com/
content/CVPR2024/papers/Ma_When_Visual_Grounding_Meets_Gigapixel-level_
Large-scale_Scenes_Benchmark_and_Approach_CVPR_2024_paper.pdf4, 6, 8, 22
22. Ma, X., Ding, Z., Luo, Z., Chen, C., Guo, Z., Wong, D.F., Feng, X., Sun, M.: Deep-
Perception:Advancingr1-likecognitivevisualperceptioninMLLMsforknowledge-
intensive visual grounding. arXiv preprint (2025),https://arxiv.org/abs/2503.
127974
23. Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., Hajishirzi, H.: When
not to trust language models: Investigating effectiveness of parametric and non-
parametric memories. In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Pro-
ceedings of the 61st Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers). pp. 9802–9822. Association for Computational
Linguistics,Toronto,Canada(Jul2023).https://doi.org/10.18653/v1/2023.acl-
long.546,https://aclanthology.org/2023.acl-long.546/5
24. Marino, K., Rastegari, M., Farhadi, A., Mottaghi, R.: OK-VQA: A visual ques-
tion answering benchmark requiring external knowledge. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
(June 2019),https://openaccess.thecvf.com/content_CVPR_2019/papers/
Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_
Knowledge_CVPR_2019_paper.pdf2, 3, 4, 22
25. Meng, R., Jiang, Z., Liu, Y., Su, M., Yang, X., Fu, Y., Qin, C., Thirukovalluru, R.,
Zhang, X., Chen, Z., Xu, R., Xiong, C., Zhou, Y., Chen, W., Yavuz, S.: VLM2vec-
v2: Advancing multimodal embedding for videos, images, and visual documents.
Transactions on Machine Learning Research (2026),https://openreview.net/
forum?id=TpU38jbKIJ9, 33
26. Mensink, T., Uijlings, J., Castrejon, L., Goel, A., Cadar, F., Zhou, H., Sha, F.,
Araujo, A., Ferrari, V.: Encyclopedic VQA: Visual questions about detailed prop-
erties of fine-grained categories. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV). pp. 3113–3124 (October 2023),https:
//openaccess.thecvf.com/content/ICCV2023/papers/Mensink_Encyclopedic_VQA_

CRAG-MM-Diagnostics19
Visual_Questions_About_Detailed_Properties_of_Fine-Grained_Categories_
ICCV_2023_paper.pdf2, 3, 4, 10, 22, 28
27. Meta, A.: Llama 3.2: Revolutionizing edge AI and vision with open, customiz-
able models (2024),https://ai.meta.com/blog/llama-3-2-connect-2024-vision-
edge-mobile-devices/6, 23
28. Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovit-
skiy, A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z., Wang, X., Zhai, X.,
Kipf, T., Houlsby, N.: Simple open-vocabulary object detection. In: Computer Vi-
sion – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27,
2022, Proceedings, Part X. p. 728–755. Springer-Verlag, Berlin, Heidelberg (2022).
https://doi.org/10.1007/978-3-031-20080-9_42,https://doi.org/10.1007/
978-3-031-20080-9_426
29. Narayan, K., Xu, Y., Cao, T., Nerella, K., Patel, V.M., Shiee, N., Grasch, P.,
Jia, C., Yang, Y., Gan, Z.: DeepMMSearch-R1: Empowering multimodal llms in
multimodalwebsearch.arXivpreprint(2025),https://arxiv.org/abs/2510.12801
1, 2
30. Ni, M., Fan, Y., Zhang, L., Zuo, W.: Visual-o1: Understanding ambiguous in-
structions via multi-modal multi-turn chain-of-thoughts reasoning. In: The Thir-
teenth International Conference on Learning Representations (2025),https://
openreview.net/forum?id=v9CDpLpjiE22
31. Qiang,C.,Wei,Z.,Han,X.,Wang,Z.,Li,S.,Lan,X.,Jiao,J.,Han,Z.:VER-Bench:
Evaluating MLLMs on reasoning with fine-grained visual evidence. In: Proceedings
of the 33rd ACM International Conference on Multimedia. p. 12698–12705. MM
’25, Association for Computing Machinery, New York, NY, USA (2025).https:
//doi.org/10.1145/3746027.3758208,https://doi.org/10.1145/3746027.3758208
2
32. Qiao, Y., Duan, H., Fang, X., Yang, J., Chen, L., Zhang, S., Wang, J., Lin, D.,
Chen, K.: Prism: A framework for decoupling and assessing the capabilities of
VLMs.In:TheThirty-eighthAnnual Conference onNeuralInformationProcessing
Systems (2024),https://openreview.net/forum?id=qLnXPVvwLx4, 22
33. Qiu, J., Madotto, A., Lin, Z., Crook, P.A., Xu, Y.E., Damavandi, B., Dong, X.L.,
Faloutsos,C.,Li,L.,Moon,S.:SnapNTell:Enhancingentity-centricvisualquestion
answering with retrieval augmented multimodal LLM. In: Al-Onaizan, Y., Bansal,
M., Chen, Y.N. (eds.) Findings of the Association for Computational Linguistics:
EMNLP 2024. pp. 247–266. Association for Computational Linguistics, Miami,
Florida, USA (Nov 2024).https://doi.org/10.18653/v1/2024.findings-emnlp.
14,https://aclanthology.org/2024.findings-emnlp.14/3, 4, 5, 10, 22
34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable
visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.)
Proceedings of the 38th International Conference on Machine Learning, ICML
2021. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR
(2021),http://proceedings.mlr.press/v139/radford21a.html9, 12, 33
35. Redmon,J.,Divvala,S.,Girshick,R.,Farhadi,A.:Youonlylookonce:Unified,real-
time object detection. In: Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR) (June 2016),https://www.cv-foundation.org/
openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.
pdf27
36. Schwenk, D., Khandelwal, A., Clark, C., Marino, K., Mottaghi, R.: A-OKVQA: A
benchmark for visual question answering using world knowledge. In: Computer Vi-

20 H. Oh et al.
sion – ECCV 2022: 17th European Conference, Proceedings, Part VIII. p. 146–162.
Springer-Verlag, Berlin, Heidelberg (2022).https://doi.org/10.1007/978-3-031-
20074-8_9,https://doi.org/10.1007/978-3-031-20074-8_922
37. Singh, P., Dhawan, S., Agarwal, S., Thakur, N.: Implementation of an efficient
fuzzy logic based information retrieval system. arXiv preprint (2015),https://
arxiv.org/abs/1503.039579
38. Stengel-Eskin, E., Guallar-Blasco, J., Zhou, Y., Van Durme, B.: Why did the
chickencrosstheroad?RephrasingandanalyzingambiguousquestionsinVQA.In:
Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Proceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Pa-
pers). pp. 10220–10237 (2023).https://doi.org/10.18653/v1/2023.acl-long.569,
https://aclanthology.org/2023.acl-long.569/22
39. Tran, D.T., Tran, T.K., Hauswirth, M., Le Phuoc, D.: ReasonVQA: A multi-
hop reasoning benchmark with structural knowledge for visual question answer-
ing. In: Proceedings of the IEEE/CVF International Conference on Computer Vi-
sion (ICCV). pp. 18793–18803 (October 2025),https://openaccess.thecvf.com/
content/ICCV2025/papers/Tran_ReasonVQA_A_Multi-hop_Reasoning_Benchmark_
with_Structural_Knowledge_for_Visual_ICCV_2025_paper.pdf3, 22
40. Wang, J., Yang, X., Sun, K., Suresh, P., Sharma, S., Czyzewski, A., Andersen, D.,
Appini, S., Banerjee, A., Choudhary, S., et al.: CRAG-MM: Multi-modal multi-
turn comprehensive rag benchmark. arXiv preprint (2025),https://arxiv.org/
abs/2510.261601, 2, 3, 4, 9, 12, 13, 22, 28
41. Wang,P.,Wu,Q.,Shen,C.,Dick,A.,VanDenHenge,A.:Explicitknowledge-based
reasoning for visual question answering. In: Proceedings of the 26th International
Joint Conference on Artificial Intelligence. p. 1290–1296. IJCAI’17, AAAI Press
(2017),https://dl.acm.org/doi/10.5555/3171642.31718252
42. Wang, Y., Zhao, Y., Chen, X., Guo, S., Liu, L., Li, H., Xiao, Y., Zhang, J., Li, Q.,
Xu, K.: VisualSimpleQA: A benchmark for decoupled evaluation of large vision-
language models in fact-seeking question answering. arXiv preprint (2025),https:
//arxiv.org/abs/2503.064924, 22, 27, 29
43. Wu, P., Xie, S.: V?: Guided visual search as a core mechanism in multimodal
LLMs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR). pp. 13084–13094 (June 2024),https://openaccess.
thecvf.com/content/CVPR2024/papers/Wu_V_Guided_Visual_Search_as_a_Core_
Mechanism_in_Multimodal_CVPR_2024_paper.pdf2, 4, 8, 22
44. Xiao, L., Yang, X., Lan, X., Wang, Y., Xu, C.: Toward visual grounding: A
survey. IEEE Transactions on Pattern Analysis & Machine Intelligence48(03),
2749–2771 (Mar 2026).https://doi.org/10.1109/TPAMI.2025.3630635,https:
//doi.ieeecomputersociety.org/10.1109/TPAMI.2025.36306356
45. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., Nie, J.Y.: C-pack: Packed
resourcesforgeneralChineseembeddings.In:Proceedingsofthe47thInternational
ACMSIGIRConferenceonResearchandDevelopmentinInformationRetrieval.p.
641–649. SIGIR ’24, Association for Computing Machinery, New York, NY, USA
(2024).https://doi.org/10.1145/3626772.3657878,https://doi.org/10.1145/
3626772.365787812
46. Xu, Y., Zhu, L., Yang, Y.: MC-Bench: A benchmark for multi-context visual
grounding in the era of MLLMs. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV). pp. 17675–17687 (October 2025),https:
//openaccess.thecvf.com/content/ICCV2025/papers/Xu_MC-Bench_A_Benchmark_
for_Multi-Context_Visual_Grounding_in_the_Era_ICCV_2025_paper.pdf4

CRAG-MM-Diagnostics21
47. Yan, Y., Xie, W.: Echosight: Advancing visual-language models with wiki knowl-
edge. In: Findings of the Association for Computational Linguistics: EMNLP 2024.
pp. 1538–1551 (2024),https://aclanthology.org/2024.findings-emnlp.83.pdf
29
48. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: MM-Vet:
evaluating large multimodal models for integrated capabilities. In: Proceedings
of the 41st International Conference on Machine Learning. ICML’24, JMLR.org
(2024),https://dl.acm.org/doi/10.5555/3692070.36944514, 22
49. Zhang, Y., Zhang, H., Tian, H., Fu, C., Zhang, S., Wu, J., Li, F., Wang, K.,
Wen, Q., Zhang, Z., Wang, L., Jin, R.: MME-realworld: Could your multimodal
LLM challenge high-resolution real-world scenarios that are difficult for humans?
In: The Thirteenth International Conference on Learning Representations (2025),
https://openreview.net/forum?id=k5VHHgsRbi4, 22

22 H. Oh et al.
Table 1:Comparison of relevant VQA benchmarks.Knowledgedenotes the type of
knowledge required (‘-’ denotes no explicit requirement);Non-salientmarks whether
a benchmark focuses on non-salient targets;Decoupledindicates whether stage-wise
evaluationisperformed(Grounding,ObjectIdentification,QuestionAnswering);Am-
biguityindicates whether or not ambiguity effects are evaluated.12
Dataset Knowledge Non-salient Decoupled Analysis Ambiguity
Knowledge-Intensive VQA
OK-VQA [24] Commonsense✗ ✗(Q only)✗
A-OKVQA [36] Reasoning✗ ✗(Q only)✗
EncyclopedicVQA [26] Factual✗ ✗(Q only)✗
SnapNTell [33] Factual✗ ✗(Q only)✗
ViQuAE [16] Factual✗ ✗(Q only)✗
ReasonVQA [39] Reasoning✗ ✗(Q only)✗
DynVQA [18] Dynamic✗ ✗(Q only)✗
InfoSeek [4] Factual✗ ✗(Q only)✗
OVEN [10] -✗ ✗(O only)✗
Realworld Perception
V* Benchmark [43] -✓ ✗(Q only)✗
GigaGrounding [21] -✓ ✗(G only)✗
MME-RealWorld [49] -✓ ✗(Q only)✗
MRAG-Bench [11] Visual✓ ✗(Q only)✗
CRAG-MM [40] Factual✓ ✗(Q only)✗
Diagnostic/Capability
VisualSimpleQA [42] Factual✓Visual and Linguistic module (Q only)✗
HallusionBench [8] -✗Visual, Language Hallucination✗
Prism [32] -✗Perception and reasoning✗
MMVet [48] -✗Capability Integration✗
CRAG-MM-Diagnostics(Ours)Factual✓Stage-wise evaluation forG, O, Q✓
A Dataset
A.1 Dataset Comparison
Table 1 compares relevant VQA benchmarks withCRAG-MM-Diagnostics.
A.2 Dataset Statistics
Figure 8 presents the statistics of theCRAG-MM-Diagnosticssamples with ad-
ditional annotations. Note that, from theCRAG-MMdataset, we remove exam-
ples that (a) are not actuallyknowledge-intensive, based onCRAG-MM’s ‘simple-
recognition’ tag, (b) concern dynamically changing knowledge, based on the
‘fast-changing’ and ‘real-time’ tags, and (c) concern domains which do not allow
us to tag a specific entity visually (this concerns the domains ‘text understand-
ing’, ‘math & science’, ‘book’, ‘shopping’ and ‘food’).
In the main paper, we broke down performance for different intervals for
visual complexity metrics and the popularity metric. We did that by performing
12Although prior work has studied ambiguity in general VQA settings [3,30,38], we fo-
cus on a underexplored ambiguity in KI-VQA—errors in precise entity identification
and visual grounding that occur before knowledge retrieval.

CRAG-MM-Diagnostics23
0 50 100
crowdedness0100200300400500Count
0.00 0.25 0.50 0.75
target's distance to center050100150200Count
0.0 0.5 1.0
target size0100200Count
referring expressionsalient63.4%in-image cue
13.9%KI cue
11.8%ambiguous
10.9%
Fig. 8:CRAG-MM-Diagnosticsstatistics based on scene crowdedness, the target object’s
distance to center, the target’s size, and the referring expression types.
interval grouping and using similar subgroup sizes for each metric across the
dataset, with each subgroup comprising approximately 230 instances (≈20% of
the total pool).
A.3 Dataset Examples
Table 2 presents representative examples from the dataset across different types
of referring expressions, along with the metadata collected during the human
annotation stage forCRAG-MM-Diagnostics.
A.4 Metadata Annotation
This section includes details about the additional metadata added toCRAG-MM-
Diagnostics.
Preliminary annotationsTo evaluate intermediate stages such as visual ground-
ing and object identification, we build an automatic pipeline to collect target
entity names, Wikipedia URLs, and bounding box annotations aligned with the
original (image, question, answer) triplets. Target entities are extracted from the
answer using the spaCy entity linker [9]. Bounding box annotations are obtained
via a two-stage approach: (1) we extract a referring expression for the target ob-
ject using Llama-3.2-11b-Vision-Instruct [27], given the image and query; (2)
we apply Grounding-DINO-Base [20] with a text and box threshold of 0.1 to
localize the corresponding image regions. Yet, note that this annotation is only

24 H. Oh et al.
preliminary and that the human annotator should verify and potentially modify
entity names, Wikipedia URLs, and bounding boxes.

CRAG-MM-Diagnostics25
Table 2:Examples fromCRAG-MM-Diagnosticsfor different types of referring expres-
sions, along with various types of metadata we collect.
Referring Expression Salient Ambiguous
Image
Questionwhich came out first,this model or the
ford focus?was the empire state building around when
thiswas first built?
Answerthe ford focus, introduced in 1998, came
out before the ford fusion, introduced in
2006.no, the empire state building was not
around when st. patrick’s cathedral was
first built, as the cathedral was completed
in 1878 and the empire state building was
built from 1930 to 1931.MetadataSize of Target Obj. 0.1411 0.2419
Target ROI Box [643,1610,2854,2388] [1136, 256, 2320, 2747]
Target Obj. Name Ford Fusion (Americas) St. Patrick’s Cathedral (New York City)
Target Obj. Wiki URL en.wikipedia.org/Ford Fusion (Americas) en.wikipedia.org/St.Patrick’s Cathedral
Textual Only Questionwhich came out first, the ford fusion or the
ford focus?was the empire state building around when
st. patrick’s cathedral was first built?
Popularity 296969 203644
Disambig. Query -was the empire state building around when
this cathedral was first built?
Referring Expression Knowledge-Intensive Cue In-Image Cue
Image
Questionhow isthis restaurant’s logo color dif-
ferent from wendy’s logo?how does the range of thewhite vehicle
compare to that of the hyundai ioniq 5?
Answermcdonald’s logo is primarily yellow and
red,whilewendy’slogoisprimarilyredand
white.the tesla model y typically offers a slightly
longerrangethanthehyundaiioniq5,with
the model y long range reaching up to 330
miles and the ioniq 5 limited up to 303
miles, though the ioniq 5 charges faster.MetadataSize of Target Obj. 0.0649 0.0245
Target ROI Box [2,1400,1345,1989] [1071, 2003, 1736, 2452]
Target Obj, Name McDonald’s Tesla Model Y
Target Obj. Wiki URL en.wikipedia.org/McDonald’s en.wikipedia.org/Tesla Model Y
Textual Only Questionhow is mcdonald’s logo color different from
wendy’s logo?how does the range of the tesla model y
compare to that of the hyundai ioniq 5?
Popularity 2524618 650057
Disambig. Query - -

26 H. Oh et al.
📝  Data Anno tation Tool
Select Annotator
Annotator 1
Instructions:
General Guideline: https://www.notion.so/Human-Annotator-Data-Annotation-Instruction-2ce03b634b7f8047835acc84f08fb36f?source=copy_link
1. Select an example from the dataset.
2. Annotate Information about target object [Target Object Name / Target Entity wikipedia URL] / Target Object Coordinates]
3. Annotate additional metadata [ambiguity_label/ textual_only_question].
🖼 Input Selection
Previous
 NextGo to specific index
0Example 0 / 247 (Range: 0-247)
{
:
:
:
:
:
:
[
]
:
:
:
:
{
:
:
:
:
}
:
}
"id""675653e6-4477-4865-b4e1-3f2db72f891d"
"image"
"<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=3024x4032 at 0x7C809B652840>"
"question"
"what model engine of this vehicle has the highest top speed?"
"answer"
"the 320 ge engine of the mercedes-benz g-series has the highest top speed at 170 kilometers per hour."
"entity_name"
"engine"
"wikipedia_url"
"https://en.wikipedia.org/wiki/engine"
"ambiguity_label":
0:
"knowledge-intensive cue"
1:
"Determining the model engine with the highest top speed requires external knowledge about vehicle specifications, which is not visible in the image."
"textual_only_question"
NULL
"number_of_objects"
6
"size_of_target_object"
0.5801
"normalized_distance_to_center"
0.04881597582856184
"metadata":
"domain"
"vehicle"
"dynamism"
"slow-changing"
"image_quality"
"normal"
"query_category"
"aggregation"
"gdino_number_of_objects"
6Detected Objects
Select a detected object or add a part annotation:
Automatic Found [5, 964, 3015, 3314]
Selected object: Mercedes-Benz G-Class
This example has already been annotated. You can update the information below.
Entity Name
Mercedes-Benz G-Class
Wiki URL
https://en.wikipedia.org/wiki/Mercedes-Benz_G-Class
Referring expression label
salient
Disambiguated Question
Textual Only Question
What model engine of the Mercedes-Benz G-Class has the highest top speed?
Note
If you need some discussion about the example, please leave note and save!
Update Annotation
1/28/26, 6:23 PM Data Annotation Tool · Streamlit
https://ki-visual-question-answering-human-annotation.streamlit.app 1/1
📝  Data Anno tation Tool
Select Annotator
Annotator 1
Instructions:
General Guideline: https://www.notion.so/Human-Annotator-Data-Annotation-Instruction-2ce03b634b7f8047835acc84f08fb36f?source=copy_link
1. Select an example from the dataset.
2. Annotate Information about target object [Target Object Name / Target Entity wikipedia URL] / Target Object Coordinates]
3. Annotate additional metadata [ambiguity_label/ textual_only_question].
🖼 Input Selection
Previous
 NextGo to specific index
0Example 0 / 247 (Range: 0-247)
{
:
:
:
:
:
:
[
]
:
:
:
:
{
:
:
:
:
}
:
}
"id""675653e6-4477-4865-b4e1-3f2db72f891d"
"image"
"<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=3024x4032 at 0x7C809B652840>"
"question"
"what model engine of this vehicle has the highest top speed?"
"answer"
"the 320 ge engine of the mercedes-benz g-series has the highest top speed at 170 kilometers per hour."
"entity_name"
"engine"
"wikipedia_url"
"https://en.wikipedia.org/wiki/engine"
"ambiguity_label":
0:
"knowledge-intensive cue"
1:
"Determining the model engine with the highest top speed requires external knowledge about vehicle specifications, which is not visible in the image."
"textual_only_question"
NULL
"number_of_objects"
6
"size_of_target_object"
0.5801
"normalized_distance_to_center"
0.04881597582856184
"metadata":
"domain"
"vehicle"
"dynamism"
"slow-changing"
"image_quality"
"normal"
"query_category"
"aggregation"
"gdino_number_of_objects"
6Detected Objects
Select a detected object or add a part annotation:
Automatic Found [5, 964, 3015, 3314]
Selected object: Mercedes-Benz G-Class
This example has already been annotated. You can update the information below.
Entity Name
Mercedes-Benz G-Class
Wiki URL
https://en.wikipedia.org/wiki/Mercedes-Benz_G-Class
Referring expression label
salient
Disambiguated Question
Textual Only Question
What model engine of the Mercedes-Benz G-Class has the highest top speed?
Note
If you need some discussion about the example, please leave note and save!
Update Annotation
1/28/26, 6:23 PM Data Annotation Tool · Streamlit
https://ki-visual-question-answering-human-annotation.streamlit.app 1/1
Fig. 9:Human annotation interface.
Human Annotation Procedure and ToolTo refine and verify the preliminary
annotations as well as to generate additional metadata, we employ an annotation
tool,asillustratedinFigure9.Themainverificationprocessinvolvesthreesteps:
1) annotating the target region of interest (ROI) for each multimodal input (i.e.,
query and image), 2) mapping the target object to its corresponding Wikipedia
entity, including the associated Wikipedia URL; and 3) checking whether the
question is ambiguous or lacks sufficient clues to uniquely identify the target
within the example.
The annotation procedure is conducted in two stages. In the first stage, five
annotators independently annotate the dataset using the provided annotation
tool (Figure 9). In the second stage, two annotators review the annotations to
identify potential errors and ensure quality. Importantly, reviewers are assigned
instances different from those they annotated in the first stage, ensuring that no
annotators review their own annotations.
Distance from the center of the image of the target object.The remaining types
of metadata for visual complexity can be computed based on the ROI bounding
boxes. We compute the distance between the image center and the center of a
ROI, normalized by the image size.
Let the image have widthWand heightH. The image center is
Cimg= W
2,H
2
. (1)

CRAG-MM-Diagnostics27
An ROI is defined by its bounding box(x min, ymin, xmax, ymax), and its center
is given by
Croi= xmin+xmax
2,ymin+ymax
2
. (2)
The Euclidean distance between the centers is
d=r
W
2−xmin+xmax
22
+
H
2−ymin+ymax
22
. (3)
We normalize this distance by half of the image diagonal length. The image
diagonal is
D=p
W2+H2, (4)
so the half-diagonal isD/2. The normalized distance is therefore
dnorm =d
D/2=2d
D. (5)
This normalized score lies in[0,1], whered norm = 0indicates perfect alignment
of the ROI center with the image center, andd norm = 1corresponds to maximum
displacement along the half-diagonal.
Number of Objects in an Image.Toquantifythevisualclutterofanimage,which
increases the visual complexity of model recognition for a given query, we utilize
the number of objects detected by the object detection model Grounding DINO
[20]. We empirically find that a standard closed-vocabulary object detector with
81 predefined categories, such as YOLO [35], cannot cover all target objects in
our dataset; thus, we use an open-vocabulary detection model here. With 81
predefined object categories set by YOLO, we additionally add more categories
tocoverobjectsinourdataset,suchas‘building’,‘tower’,‘plant’and‘gardening’.
Proportion of Target Size.Given the bounding box of the target entity
(xmin, ymin, xmax, ymax), we compute its area asA roi= (x max−xmin)×(y max−
ymin). The image area isA img=W×H, whereWandHdenote the image
width and height, respectively. The proportion of target size is then defined as
the ratios=Aroi
Aimg. This normalized measure lies in[0,1], where values close to
0indicate that the target occupies a very small region in the image, and values
closer to1indicate that the target covers most of the image.
Visual SaliencyWe define a saliency score inspired by [42] that quantifies how
visually prominent a target object is within an image by jointly considering its
relative size, spatial centrality (distance to the center), and scene clutter (crowd-
edness). Intuitively, an object is more salient if it is larger, closer to the image
center, and appears in less cluttered scenes. Lets,d, andndenote the target
object’s size, its distance from the image center, and the number of detected ob-
jects in the image, respectively. Each component is first normalized to the range
[0,1]using a normalization functionN(·)(min–max or robust normalization):
N(x) =x−min(x)
max(x)−min(x). (6)

28 H. Oh et al.
Size is treated as positively correlated with saliency, while distance and clutter
are negatively correlated. Accordingly, we compute:
˜s=N(s), ˜d= 1− N(d),˜c= 1− N(log(1 +n)). (7)
The logarithmic transformation onnreflects the sub-linear perceptual effect of
clutter (e.g., differences between 10 and 100 objects are less salient than raw
counts suggest). The final saliency scoreS∈[0,1]is obtained via a weighted
geometric mean, which encourages balanced contributions across factors and
penalizes cases where any single component is weak:
S= exp 
αlog(1 + ˜s) +βlog(1 + ˜d) +γlog(1 + ˜c)
α+β+γ!
−1, (8)
whereα,β, andγcontrol the relative importance of size, centrality, and clutter.
In our experiments, all weights are set to 1.0. This formulation yields a bounded,
interpretable saliency measure that is robust to scale differences across images
and naturally captures the interaction between visual prominence cues.
A.5 Subjectivity Analysis
Toevaluatethepotentialimpactofannotationsubjectivityandambiguitywithin
our data creation pipeline, we conduct an independent human validation study.
Werecruit12professionalannotatorsviaProlific13toevaluateastratifiedsample
of 100 image–question pairs (n= 100, balanced with 25 instances per referring
expression category). Annotators are tasked with identifying the appropriate
referring expression categories for each pair, with majority-voting applied across
three independent responses per instance.
Our analysis demonstrates robust human consensus: we achieve moderate
agreement for the multi-class classification task (Cohen’sκ= 0.453) and sub-
stantial agreement for the binary ambiguity classification task (κ= 0.605)
between the expert ground-truth annotations and the crowd-sourced majority
votes. These results confirm a high degree of objective human consensus, indi-
cating that the dataset’s definitions remain robust against individual annotation
or author bias.
B Generalizability of Findings
Many existing KI-VQA benchmarks present significantly simplified visual en-
vironments. For instance, datasets like InfoSeek [4] and EncyclopedicVQA (E-
VQA) [26] exhibit low object densities, typically ranging from 1.2 to 1.6 objects
per image. In contrast, CRAG-MM [40] averages 12.5 objects per image, thereby
serving as a more rigorous testbed for multimodal knowledge assistants in the
13https://www.prolific.com/

CRAG-MM-Diagnostics29
wild settings. Furthermore, our insights regarding the relationship between vi-
sual complexity and model performance generalize well beyond egocentric do-
mains, aligning closely with findings from non-egocentric benchmarks such as
MRAG [11] and VisualSimpleQA [42].
To further investigate whether the visual grounding module integrated into
the RAG pipeline in Section 6 consistently boosts performance across differ-
ent KI-VQA environments, we integrate it into the EchoSight [47] framework,
leveraging its prebuilt retrieval index for EVQA-based RAG. As detailed in Ta-
ble 3, while introducing the grounding module leads to an overall performance
degradation on EVQA, a granular category-level analysis reveals nuanced behav-
ior. Specifically, grounding degrades performance in “landmark” categories where
global visual context would be inherently preferred. Conversely, performance im-
provements are concentrated in highly object-centric subsets (e.g., iNaturalist).
These findings suggest that universal grounding is not a one-size-fits-all solution;
instead, we recommend that future work explore routing architectures capable
of dynamically invoking grounding modules based on query and image charac-
teristics.
Table 3:The impact of grounding on another KI-VQA benchmark (EVQA).
Dataset / Subset Model Recall@1/5/10 (%) QA (%)
EVQA (All)EchoSight 13.4 / 31.8 / 41.7 41.7
+ Grounding 11.8 / 29.3 / 37.6 39.5
- iNaturalist (Obj)EchoSight 9.0 / 25.4 / 34.6 39.0
+ Grounding 9.6 / 27.5 / 35.8 40.3
- LandmarksEchoSight 18.2 / 38.9 / 49.7 44.8
+ Grounding 14.3 / 31.2 / 39.5 38.5

30 H. Oh et al.
C Language-based Visual Grounding
Prompt used for Visual Grounding Inference
system_prompt = (
You are a visual grounding assistant. Given an image and a question, output the
bounding box of the image region that contains the visual information needed to
answer the question.
The image is always 960 pixels wide and 1280 pixels tall.
Output format: [x1, y1, x2, y2] in pixels — NO other text.
- (0, 0) is the top-left corner
- x increases rightward (0 to 960), y increases downward (0 to 1280)
- Make the box as tight as possible around the target
- If no specific region applies, output the full image: [0, 0, 960, 1280]
Examples:
Question:'What is the name of the store?'
Output: [95, 40, 530, 160]
Question:'What color is the car?'
Output: [260, 580, 720, 940]
Question:'How many windows does the building have?'
Output: [60, 180, 900, 1100]
)
Fig. 10:Prompt used for Visual Grounding Inference.
Experimental Setup.We implemented both generalized VLMs and specialized
zero-shot object detection models, including Grounding-DINO and OWL-ViT.
The infrastructure is built on Python 3.10 using PyTorch and the Hugging Face
Transformers ecosystem. All experiments were conducted on four L40s multi-
GPU cluster.
Generalized MLLMs (Llama/Qwen): To optimize throughput and memory
management, these models are deployed via thevLLM(v0.10.1) [15] inference
engine. We utilize a tensor-parallel configuration across available GPUs with a
max model length of 8,192 tokens in bfloat16 precision. The generation process
is governed byvllm.SamplingParamswith a 75 max tokens, temperature of 0.1
and a top-p of 0.9 to promote deterministic and focused outputs.
Specialized grounding models are deployed using dedicated processors from
the transformers library (AutoProcessorandOwlViTProcessor). Grounding-
DINO is specifically tuned with a box threshold of 0.4 and a text threshold
of 0.3 to filter low-confidence detections during the zero-shot grounding tasks.
OWL-ViT uses threshold 0.1 when post processing the grounding output.

CRAG-MM-Diagnostics31
D Object Identification
Prompt used for Object Identification Inference
SYSTEM_PROMPT = (
You are a helpful visual assistant that identifies the specific target object
in an image referred to by a question.
Your task is not to answer the question, but to determine which object in the
image the question is referring to and describe or name that object precisely.
What is the object’s name which will help me answer the query?
Focus only on visual and contextual cues from the image that indicate the
subject of the question.
Instructions:
- Ignore the question’s semantic intent (e.g., do not explain, justify, or
give an opinion-based answer).
- Identify the visual target most relevant to the question.
- Output only the exact entity object name (e.g.,'Subaru WRX','The Empire
State Building','euphorbia aphylla').
- Do not include any additional explanation, reasoning, or answer content.
Example:
Image: A photo of a blue Subaru WRX in a parking lot and the blue Subaru WRX is
highlighted with a green border box.
Question: “Is this a good car for transporting seven passengers at once?”
Correct Output: Subaru WRX
Incorrect Output: No, the Subaru WRX can only fit 5 passengers.
Now analyze the following image and question to output only the target object
name.
### Response format:
target_object: [entity name of target object]
)
Fig. 11:Prompt used for Object Identification Inference
Experimental Setup.We follow the same experimental setup as stage 1, visual
grounding task overall (see Appendix C). Aside of that, we further elaborate dif-
ferent specialized models in this stage. These specialized models utilized during
the experiments are the unimodal retrieverCLIP-ViT-Large-Patch14-336and
the multimodal retrieversVLM2Vec-V2.0andQwen-3-VL-Embedding-2B. We use
the image knowledge graph14fromCRAG-MMas the retrieval corpus and recom-
pute embeddings for each model. For all image retrievers, similary scores for
query (Q) and candidate embeddings (C) are measured with cosine similarity
(sim=Q·C⊤). All retrieval indexes are built with ChromaDB [5].
14huggingface.co/datasets/crag-mm-2025/image-search-index-public-test

32 H. Oh et al.
Unimodal Retriever: For unimodal retriever, encoders only get either image
or text as an input for the model.CLIPis under this category and we utilize
image to retrieve relevant image KG information using prebuilt image index
fromCRAG-MM.
MultimodalEmbedding:Forunifiedmultimodalrepresentationforimageand
text question, we leverageVLM2VEC2and theQwen3-VL-Embedding-2Bmodels
allowing for the simultaneous encoding of system instructions, text prompts,
and image data into a singular embedding space. We follow experimental setups
from official documents per each model. ForVLM2VECv2”Represent the given
image with the following question” is utilized as instruction with the regarding
text input, and ”Represent the user’s input.” for theQwen3-VL-Embedding
D.1 Generalized Models (VLM)
Object Identification and Target Saliency.Unlike the grounding task in
Stage1,objectidentificationislesssensitivetovisualcomplexity;infact,saliency
is(onlyweakly)negativelyrelatedtoaccuracy(r pb=−.168)asshowninFig.12.
This likely stems from dataset distribution: less popular entities are more fre-
quent in salient groups, impacting object identification more significantly. This
is supported by specialized models gaining less from cropping for less popular
groups compared to popular ones (Fig. 14).
0.10.4
0.40.5
0.50.6
0.60.7
0.71
target saliency0.00.20.40.60.81.0accuracy (%)
Llama 11B
Qwen 3B
Qwen 7B
Qwen 32BQwen 72B
GPT-5-mini
GPT-5
Fig. 12:Generalized model (VLM) object identification accuracy with different
saliency interval.
D.2 Specialized Models (Image Retriever)
In this section, we evaluate the performance of various image retrieval models
across different input modalities.

CRAG-MM-Diagnostics33
Retrieval Performance across Top-k.As illustrated in Figure 13, we com-
pare the retrieval scalability ofCLIP-ViT-Large-Patch14-336[34] and the mul-
timodal retrieversVLM2Vec-V2.0[25] andQwen-3-VL-Embedding-2B[17] by mea-
suring Recall@kfork∈ {1,5,10,20,30}. For this comparison, all models utilize
only the image input with ground-truth (GT) target region cropping to isolate
the retriever’s capability from potential localization errors. We observe a consis-
tentgrowthinrecallacrossallmodelsaskincreases,withQwen-3-VL-Embedding-2B
maintaining a significant performance lead across the entire range, followed by
CLIP-ViT-Large-Patch14-336andVLM2Vec-V2.0. We follow the official experi-
mental setups for retrieval instructions when evaluating the multimodal retriev-
ers.
Sensitivity to Input Variants.Table 4 provides a detailed breakdown of how
different input modalities and preprocessing steps affect retrieval performance.
Several key trends emerge:
–Impact of Region Cropping:For all models, providing a localized view of the
target object results in substantial gains. Using GT crops yields the best
results; for instance, boostingQwen-3-VL-Embedding-2BR@1 from 22.11%
to 31.33%. Notably, even automated cropping via G-DINO provides a per-
formance uplift over the original full image in the CLIP baseline, indicating
potential for the grounding module to further bridge the gap to GT-level
performance.
–Multimodal vs. Unimodal Performance:Interestingly, for the multimodal re-
trievers (VLM2Vec-V2.0andQwen-3-VL-Embedding-2B), the "Image Only"
input consistently outperforms the "Image + Text" combination. For ex-
ample,Qwen-3-VL-Embedding-2Bperformance drops from 22.11% to 12.62%
R@1 when the text query is added to the full image input. This suggests that
the text queries may introduce noise or that current multimodal retrievers’
training objectives are not yet optimized for precisely grounding the target
region with textual cues to retrieve relevant information. Consequently, the
visual features of the target object remain significantly more discriminative
than the joint multimodal embeddings for this specific task.
–Text-Only Baseline:The "Text Only" variants perform poorly across all
models, with R@1 values below 2.1%. This underscores that the dataset
requires fine-grained visual perception that cannot be resolved through lin-
guistic cues alone.
Effect of Popularity.Figure 14 illustrates specialized image retrieval perfor-
mance trends stratified by object popularity.

34 H. Oh et al.
Table 4:Retrieval performance (Recall@k) comparison across different retrievers and
input variants. "w/ G-DINO Crop" indicates that the image input is cropped based on
predictions from Grounding-DINO, whereas "w/ GT Crop" denotes the use of ground-
truth region of interest (ROI) annotations for cropping. Note that CLIP is a unimodal
retrieverthatcanprocesseithersolelyimageortextinputforretrieval,andVLM2VEC-
v2.0 and Qwen3-VL-Emb. are multimodal retrievers that can process text and image
together.
Model Input R@1 R@5 R@10 R@20 R@30
CLIPImage 14.97 26.98 31.85 38.47 41.25
w/ G-DINO Crop17.41 31.77 36.12 42.91 45.43
w/ GT Crop20.28 36.03 41.78 47.87 51.20
VLM2VEC-v2.0Image 12.36 22.72 26.72 32.03 35.25
w/ GT Crop15.75 29.24 34.38 40.03 43.60
Text 1.65 4.00 6.01 9.57 11.92
Image + Text 8.88 17.06 20.54 26.20 28.20
w/ Image (GT Crop) + Text9.40 19.50 24.11 29.24 32.29
Qwen3-VL-EmbImage 22.11 36.99 42.99 49.35 51.61
w/ GT Crop31.33 46.30 51.87 56.92 58.83
Text 2.09 5.48 7.75 10.97 13.40
Image + Text 12.62 23.24 29.77 36.99 40.21
w/ Image (GT Crop) + Text18.54 30.64 36.73 42.12 45.52
CLIP
CLIP w/ GT cropVLM2VEC2 Image only
VLM2VEC2 Crop. Image onlyQwen-3-vl-2B Image only
Qwen-3-vl-2B Crop. Image only
1 5 10 20 30
k (Number of retrieved images)2030405060Recall (%)
Retrieval Performance across T op-k
Fig. 13:Image retriever results with dif-
ferenttopk.Forcomparison,allretrievers
utilize image only input and ground truth
target region cropped version.
16.2k
6.2k 58k
58k202k
202k 450k
450k 20.7M
popularity of object0.20.40.60.81.0recall@10 (%)
CLIP
CLIP (cropped)
VLM2VEC2VLM2VEC2 (cropped)
Qwen-3-vl-2B
Qwen-3-vl-2B
(cropped)Fig. 14:Specialized image retriever ob-
ject identification results with different
entity popularity interval. For compari-
son, all retrievers utilize image only input
and ground truth target region cropped
version. Black rectangle means model
mean.

CRAG-MM-Diagnostics35
E Knowledge Retrieval and Reasoning
Prompt used for Knowledge Extraction Inference
SYSTEM_PROMPT = (
You are a helpful assistant that truthfully answers user questions. Keep your
response concise and to the point.
)
Fig. 15:Prompt used for Knowledge Extraction Inference
Experimental Setup.We maintain an experimental configuration consistent with
the stage 1visual grounding and stage 2object identification tasks (see Sec-
tions C and D). For the additional text retrieval component, we employBGE
-large-en-v1.5as the primary text encoder. We leverage the prebuilt Web
search index provided byCRAG-MM15, which serves as the candidate Web corpus
for our stage 3knowledge retrieval and reasoning experiments.
F Evaluation
Figure 16 shows the prompt used for LLM-as-a-judge evaluation.
15huggingface.co/datasets/crag-mm-2025/web-search-index-public-test

36 H. Oh et al.
Prompt used for LLM-as-a-judge evaluation
SYSTEM_PROMPT=(
You will be given a question, a ground truth answer, and a model prediction. Your task is to judge
if the prediction is correct or not based on the ground truth answer.
## Instructions
Read the question, ground truth answer, and model prediction carefully. Follow the step by step
guideline below to make a judgment.
1. If the prediction indicates uncertainty or refusal to answer, output json {'accuracy': False}
2. If the prediction exactly matches the ground truth, output json {'accuracy': True}
3. If the ground truth is a number
3.1 If the prediction gives a number that almost exactly matches the ground truth, output json
{'accuracy': True}
3.2 If the prediction gives a number that is not the same as the ground truth, output json
{'accuracy': False}
4. If the prediction is self-contradictory, output json {'accuracy': False}
5. If the prediction is not answering the question, output json {'accuracy': False}
6. If ground truth contains a set of objects,
6.1 if the prediction contains exactly same objects as the ground truth, output json {'accuracy':
True}
6.2 if the prediction contains different objects from the ground truth, output json {'accuracy':
False}
6.3 if the prediction is almost same as the ground truth, use your best judgement to give output.
7. If the prediction is grounded by the ground truth, output json {'accuracy': True}
8. If the prediction is unrelated or contradictory to the ground truth, output json {'accuracy':
False}
## Additional Guidelines
- Take it as granted that the ground truth is always correct.
- If the prediction gives extra information that is not in the ground truth, it is still correct as
long as it is grounded by the ground truth.
- Be careful about numbers. 1 mile is about 1.60934 km. 1 foot is about 0.3048 m. 1 inch is about 2.54
cm. 1 yard is about 0.9144 m. 1 pound is about 0.453592 kg. 1 gallon is about 3.78541 liters. 1 ounce
is about 28.3495 grams.
## Output Format
Your judgment should first provide a VERY-SHORT explanation on your rationale. When relevant, you
need to include the guidelines above to explain your judgment. Finally, your judgment should
clearly state "answer: True" or "answer: False".
Below are some examples:
EXAMPLES START
Question: who will win the game?
Ground Truth: Lakers is favored to win the game.
Prediction: Sorry, it is hard to predict the outcome of the game.
"""
{
'explanation':'The prediction indicates it is not sure about the answer. So the prediction is
incorrect according to the guideline 1.',
'accuracy': False
}
"""
. . .
EXAMPLES END
)
Fig. 16:Prompt used for LLM-as-a-judge

CRAG-MM-Diagnostics37
G Manual Error Analysis
ToinvestigatethesystemicfailuremodesofbimodalandgroundedRAGpipelines,
even when utilizing ground-truth (GT) region cropping prior to image retrieval,
we conduct a qualitative manual analysis across two distinct backbones:GPT-5
andQwen-2.5-VL-32B. We randomly sample 200 failure cases identified from
the test instances. This evaluation allows us to categorize recurring bottle-
necksacrossmulti-stagesubtasks,includinggrounding,objectidentification,and
knowledgeextraction.Whilebothmodelsheavilysufferfromknowledge-retrieval
bottlenecks, distinct architectural trends emerge:Qwen-2.5-VLis primarily con-
strained by fine-grained object identification, whereasGPT-5failures are signif-
icantly harder to isolate, where full statistical distributions are summarized in
Table 5.
The primary failure dimensions are detailed below, with representative qual-
itative examples provided in Table 6:
Difficult Attribution:These cases involve errors where the failure cannot be
clearly assigned to a single component. For example, when asked about the
number of generations for aNissan Armada, the model predicted just “Five”
despitethegroundtruthbeingthree.Thedifficultyindiagnosingwhetherthe
model failed during object identification or knowledge retrieval underscores
ourmotivationfordecouplingKI-VQAevaluation,asaggregatemetricsoften
obscure whether a failure stems from retrieval or internal model bias.
Knowledge Bottlenecks:These errors occur when the model correctly iden-
tifies the target object but fails to retrieve or utilize specific, fine-grained fac-
tual data. In Table 6, theToyotaexample illustrates this: while the model
identifies the vehicle correctly, it fails to accurately retrieve the founder’s
birth date, providing a hallucinated response instead. This highlights that
even state-of-the-art models struggle with the ‘long tail’ of domain-specific
knowledge without more precise retrieval-augmentation.
Object Identification Failures:Identification errors are characterized by an
inability to distinguish between similar categories or recognize non-salient
targets. Notably, despite the localized focus provided by GT-region crop-
ping, both specialized retrievers and the generalized VLM occasionally failed
to correctly identify the primary subject. This suggests a critical need for
more robust, universal object recognizers capable of handling fine-grained
taxonomies.
– Fine-grained confusion:As seen in thePanther chameleoncase, the
model misidentified the subject as aJackson’s chameleon.
– Saliency issues:For theFlag of Missouri, the model failed to recognize
thespecificflaginacomplexoutdoorscene,defaultingtoamorecommon
French flag.
Reasoning and Ambiguity:Unlike knowledge bottlenecks, these failures oc-
cur when the model possesses the correct information but fails to process it
logically, or when the query is fundamentally unclear.

38 H. Oh et al.
– Incomplete Reasoning:In theMasadaexample, the model identifies
the builder correctly as “Herod the Great” but fails to provide the nu-
anced explanation required by the context, even when the retriever pro-
vides broader details (e.g., citing both Alexander Jannaeus and Herod).
Notethatretrievalcorrectlyretrievesufficientinformation(Masda: "builder":
"[[Alexander Jannaeus]] [[Herod the Great]]",), but fails to augment
these information while answering.
– Inherent Ambiguity:ThePaper towelcase demonstrates a failure
where the model provides a technical material breakdown (wood pulp,
cardboard) but fails to synthesize a direct, concise answer to the user’s
intent caused from ambiguity of the nature of referring expression “this”
in this case.
Table 5:Error analysis of grounded bimodal RAG pipeline.
Failure Category GPT-5 (%) Qwen-2.5-VL-32B(%)
Difficult Attribution 41 1
Knowledge Bottlenecks 21 23
Object Identification Failures: Fine-grained confusion 20 45
Incomplete Reasoning 6 14
Object Identification Failures: Saliency issues 5 10
Inherent Ambiguity 3 6
Wrong Judge 4 1

CRAG-MM-Diagnostics39
Table 6:Qualitative analysis of failure cases in the Bimodal and grounded RAG
pipeline (GPT-5 backbone, GT region cropping). Categorization based on manual error
case analysis.
Category Image Target Object Question & Answer Prediction
Difficult Attribution
Nissan Armada
en.wikipedia.org/wiki/Nissan_ArmadaQ:How many different generations of
this car are there?
A:As of 2025, there have been three
generationsoftheNissanArmada,with
thefirstgenerationenteringproduction
in 2003, the second in 2016, and the
third being in production since 2024.Five.
Object ID Failure
(Fine-grained Confusion)
Panther chameleon
en.wikipedia.org/wiki/Panther_chameleonQ:what is a difference between this
and a veiled chameleon?
A:the panther chameleon and the
veiledchameleonhavedifferentdisposi-
tionswiththepantherchameleonbeing
more friendly.It looks like a Jackson’s chameleon.
Key differences from a veiled
chameleon: ...
Knowledge Bottleneck
Toyota
en.wikipedia.org/wiki/ToyotaQ:when was the founder of this com-
pany born?
A:sakichi toyoda was born march 19,
1867.That’s a Toyota. Toyota Motor Corpo-
ration’s founder, Kiichiro Toyoda, was
born on June 11, 1894. (Sakichi Toy-
oda, founder of the Toyota Group, was
born Feb 14, 1867.)
Object ID Failure
(Non-salient)
Flag of Missouri
en.wikipedia.org/wiki/Flag_of_MissouriQ:when was this flag created?
A:the flag of missouri was adopted in
1913.It looks like the French flag. The
modern tricolor design was officially
adopted on February 15, 1794.
Incomplete Reasoning
Masada
en.wikipedia.org/wiki/MasadaQ:who is credited with building this
ancient fortress?
A:masada was built by herod the
great, although some sources suggest it
may have been initially constructed by
alexander jannaeus.Herod the Great.
Ambiguity
Paper towel
en.wikipedia.org/wiki/Paper_towelQ:what is this made of?
A:paper towels are made from either
virgin or recycled paper pulp which
is extracted from wood or fiber crops.
they are sometimes bleached during
the production process to lighten col-
oration, and may also be decorated
with colored images on each square.- The roll is paper towels, made from
paper (wood pulp). - The inner tube
is cardboard. - The stand/holder ap-
pears to be metal wire (likely stainless
or chrome-plated steel).
# WikiVideo: Article Generation from Multiple Videos

**Authors**: Alexander Martin, Reno Kriz, William Gantt Walden, Kate Sanders, Hannah Recknor, Eugene Yang, Francis Ferraro, Benjamin Van Durme

**Published**: 2025-04-01 16:22:15

**PDF URL**: [http://arxiv.org/pdf/2504.00939v1](http://arxiv.org/pdf/2504.00939v1)

## Abstract
We present the challenging task of automatically creating a high-level
Wikipedia-style article that aggregates information from multiple diverse
videos about real-world events, such as natural disasters or political
elections. Videos are intuitive sources for retrieval-augmented generation
(RAG), but most contemporary RAG workflows focus heavily on text and existing
methods for video-based summarization focus on low-level scene understanding
rather than high-level event semantics. To close this gap, we introduce
WikiVideo, a benchmark consisting of expert-written articles and densely
annotated videos that provide evidence for articles' claims, facilitating the
integration of video into RAG pipelines and enabling the creation of in-depth
content that is grounded in multimodal sources. We further propose
Collaborative Article Generation (CAG), a novel interactive method for article
creation from multiple videos. CAG leverages an iterative interaction between
an r1-style reasoning model and a VideoLLM to draw higher level inferences
about the target event than is possible with VideoLLMs alone, which fixate on
low-level visual features. We benchmark state-of-the-art VideoLLMs and CAG in
both oracle retrieval and RAG settings and find that CAG consistently
outperforms alternative methods, while suggesting intriguing avenues for future
work.

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
WIKIVIDEO : Article Generation from Multiple Videos
Alexander Martin1Reno Kriz1,2*William Walden1,2*Kate Sanders1
Hannah Recknor1,2Eugene Yang1Francis Ferraro3Benjamin Van Durme1,2
1Johns Hopkins University2Human Language Technology Center of Excellence
3University of Maryland Baltimore County
{amart233, vandurme }@jhu.edu
Abstract
We present the challenging task of automatically creating a high-level
Wikipedia-style article that aggregates information from multiple diverse
videos about real-world events, such as natural disasters or political elec-
tions. Videos are intuitive sources for retrieval-augmented generation
(RAG), but most contemporary RAG workflows focus heavily on text while
existing methods for video-based summarization focus on low-level scene
understanding rather than high-level event semantics. To close this gap, we
introduce WIKIVIDEO , a benchmark consisting of expert-written articles
and densely annotated videos that provide evidence for articles’ claims,
facilitating the integration of video into RAG pipelines and enabling the
creation of in-depth content that is grounded in multimodal sources. We
further propose Collaborative Article Generation ( CAG ), a novel interac-
tive method for article creation from multiple videos. CAG leverages an
iterative interaction between an r1-style reasoning model and a VideoLLM
to draw higher-level inferences about the target event than is possible with
VideoLLMs alone, which fixate on low-level visual features. We benchmark
state-of-the-art VideoLLMs and CAG in both oracle retrieval and RAG
settings and find that CAG consistently outperforms alternative methods,
while suggesting intriguing avenues for future work.1
1 Introduction
Audiovisual media is becoming an increasingly dominant form of online information con-
sumption. From firsthand, “in the wild” video footage of natural disasters to professionally
edited news coverage of major political events, videos serve as rich sources of information
for producing factual, grounded articles. Especially for actively unfolding events, grounding
articles in video not only can potentially combat misinformation among readers, but can
also provide a useful tool for journalists and other writers to quickly synthesize information
about new developments. Figure 1 motivates this task, taking an information request and
producing a Wikipedia-style article with references to the supporting video content.
Current methods and resources for article generation overwhelmingly rely on textual
sources (Liu et al., 2018a; Barham et al., 2023; Lawrie et al., 2024; Shao et al., 2024, i.a.), while
video understanding benchmarks largely focus on low-level tasks such as entity-centric
question answering or captioning (Xu et al., 2016; Krishna et al., 2017; Yu et al., 2019; Li
et al., 2022; Lin et al., 2024). Kriz et al. (2024) show that models trained on such tasks often
fail in more realistic settings that require understanding of high-level semantics, (e.g.) as
conveyed in articles about actual events. The MultiVENT benchmarks (Sanders et al., 2023;
Kriz et al., 2024) are distinctive in focusing on major real-world events as depicted in multiple
videos—spanning firsthand footage, amateur-edited clips, and news broadcasts.
In this work, we introduce WIKIVIDEO , a benchmark that builds upon MultiVENT and
evaluates the ability to write event-centric articles in the style of a Wikipedia lead (overview)
1Data and code can be found here: https://github.com/alexmartin1722/wikivideo
1arXiv:2504.00939v1  [cs.CV]  1 Apr 2025

Preprint. Under review.
Figure 1: WIKIVIDEO introduces the task of Article Generation from Multiple Videos, which
requires writing a high-level article in the style of a Wikipedia lead, given a target event ( T),
a query about that event ( Q), and a collection of Q-relevant videos ( V). All claims in the
article are grounded in visual, audio, and/or OCR content of video(s) in V(indicated by
matching colors between text and frame borders above).
section based only on video content. Given a request about a real-world event, systems
must retrieve a set of relevant videos and then generate an article from the videos’ (visual,
audio, and OCR) content. WIKIVIDEO consists of 52 events and nearly 400 relevant videos
(from a corpus of 109K; avg. 8 relevant/event), with expert-written reference articles that
synthesize information about each event across allrelevant videos—forcing systems to
not only understand high-level information within a single video, but also to synthesize
information across multiple videos on the same topic. Systems able to achieve strong results
onWIKIVIDEO would be of significant practical use in grounded article generation from
multimodal sources, and would enable both the rapid seeding of new Wikipedia articles for
actively unfolding events and the enriching of existing articles with audiovisual content.
To support the WIKIVIDEO task, we propose CAG (Collaborative Article Generation), a
novel method capable of generating high-level articles. Inspired by relevance feedback
(Rocchio, 1971) and recent advances in test-time scaling (DeepSeek-AI et al., 2025), CAG
involves collaborative interaction between (1) a VideoLLM, (2) a text-based reasoning model,
and (3) a text-only LLM to extract information from videos and to aggregate the information
into an article. The VideoLLM extracts low-level information, such as on-screen text and
descriptions of visual entities, while the reasoning model provides relevance feedback on
the extracted information, optionally requesting new extractions from the VideoLLM before
feeding relevant ones to the text-only LLM. The LLM then aggregates the relevant extractions,
drawing higher-level inferences about the underlying event, in order to generate the final
article. We summarize our contributions as follows:
1.We introduce WIKIVIDEO , a new dataset and task for generating articles from mul-
tiple videos. WIKIVIDEO is the first benchmark for multi-video article generation,
requiring reasoning across audio and visual information, covering 52 events (topics)
as depicted in nearly 400 videos that are densely annotated with modality-specific
claim grounding annotations, with expert-written reference articles for each event.
2.We introduce CAG , a novel method for article generation from multiple videos that
is based on relevance feedback and test-time scaling.
3.We present a broad suite of experiments that evaluate both CAG and popular Vide-
oLLMs on WIKIVIDEO across a range of settings, demonstrating CAG ’s superiority
to other approaches while revealing W IKIVIDEO to be a challenging benchmark.
2 Related Work
Video Understanding and Summarization Video summarization has been studied on
small-scale video datasets, such as SumMe (25 videos; Gygli et al., 2014) and VideoSum
(50 videos; Song et al., 2015)—considerably smaller than WIKIVIDEO (∼400 videos). Work
oncross-modal summarization that leverages video largely focuses on producing low-level
2

Preprint. Under review.
scene descriptions as the summary, since the associated tasks are chiefly concerned with
aligning video scenes with caption-like text (He et al., 2023; Lin et al., 2024; Hua et al., 2024).
Other work treats video summaries as mere LLM syntheses of frame-level captions (Hua
et al., 2024; Zhang et al., 2024b). In contrast, WIKIVIDEO is focused on summaries that
provide high-level information supported by video content.
Ren et al. (2025) recently proposed the task of video retrieval augmented generation (vide-
oRAG). While we too explore retrieval in our experiments (section 5), our data and task
are considerably different: whereas Ren et al. exclusively use highly polished videos (e.g.
documentaries, lectures), much of WIKIVIDEO consists of raw and amateur-edited footage
of events in the wild and in real time . Further, whereas they are concerned with short-form
question answering, we are concerned with long-form article generation.
Similarly, there is much other video understanding work oriented toward tasks other than
summarization, such as retrieval of (Chen & Dolan, 2011; Xu et al., 2016; Anne Hendricks
et al., 2017; Wang et al., 2020) question answering about (Jang et al., 2017; Lei et al., 2018;
Yu et al., 2019), and recognition of (Zhou et al., 2019; Sanders et al., 2024) low-level video
features and concepts that span a few seconds or that exist purely at the frame level.
Article Generation has largely been studied in text-only settings and as a multi-document
summarization task. Early work in this area was performed as part of the DUC and TAC
conferences2, including the DUC 2003-2007 multi-document summarization tasks and the
TAC 2010 and 2011 Guided Summarization track. Whereas this early work—and many more
recent efforts (Hermann et al., 2015; Nallapati et al., 2016; Fabbri et al., 2019; Huang et al.,
2024, i.a.)—focused on summarizing news articles, generation of Wikipedia-style articles has
received increasing attention. To our knowledge, Sauper & Barzilay (2009) were the first
to have attempted this, focusing on generation of full Wikipedia articles by filling learned
article templates with sentences extracted from source articles. Similar to us, Liu et al. (2018b)
focus on Wikipedia lead sections, taking article titles and a collection of source documents
as input and performing abstractive summarization using a decoder-only Transformer and
introducing the WikiSum dataset as part of their work. Zhu et al. (2021) also focus on leads,
but take a topic modeling-inspired approach, assigning topics to source article paragraphs
and conditioning generation of each lead sentence on paragraphs associated with either
a single predicted topic or a mixture of topics. Like Sauper & Barzilay (2009), Shao et al.
(2024) tackle full Wikipedia article generation, using a complex pipeline that entails (1)
surveying related Wikipedia articles, (2) generating perspectival questions and answers via
simulated dialogues, (3) using these dialogues to construct an outline for the article, and (4)
populating the outline from section titles and headings of source articles retrieved during
(2). Concurrently, Yang et al. (2025) explore multimodal article generation, but aimed at
incorporating figures into articles rather than synthesizing information from video sources.
Finally, in focusing on Wikipedia articles about events , we extend a recent line of work on
explicitly event-centric summarization, in which generations must cover relevant information
about a single target event (Vallurupalli et al., 2022; Gantt et al., 2024; Walden et al., 2024).
3 W IKIVIDEO Dataset
WIKIVIDEO is built using videos from MultiVENT 1.0 (Sanders et al., 2023) and MultiVENT
2.0 (Kriz et al., 2024) that are linked to Wikipedia articles obtained from the January 2025
dump provided by the MegaWika2 dataset (Barham et al., 2025). Our data collection process
consists of five steps: (1) initial event and article selection, (2) article claim decomposition,
(3) claim correction, (4) claim grounding, and (5) article rewriting. Figure 2 illustrates the
core components of the annotation process: decomposition, grounding, and rewriting.
Initial Event and Article Selection We select an initial set of events from MultiVENT
subject to two constraints: the event must (1) have English-language videos associated with
2DUC:https://duc.nist.gov/ ; TAC:https://tac.nist.gov
3

Preprint. Under review.
Figure 2: The WIKIVIDEO curation process. (1) Sentences in Wikipedia lead sections are
decomposed into a set of subclaims. (2) Subclaims are grounded in audio, video, and/or
OCR evidence. (3) Leads are rewritten to cover all and only the grounded information.
it and (2) must have a link to an English Wikipedia article about the event.3In total, there
are 58 events in MultiVENT satisfying both criteria, supported by 503 associated videos.
For each, we extract the lead section from the linked Wikipedia article to use as the basis
for the remainder of our annotation. Lead sections are distinctly suited to our goal of
curating high-level articles, as they are explicitly intended to provide a summary of the most
important aspects of the entire page.4
Claim Decomposition and Correction Next, following recent work on claim decomposition
(Min et al., 2023; Wanner et al., 2024b; Gunjal & Durrett, 2024, i.a.), we decompose each
sentence of the Wikipedia lead section into a set of contextualized, atomic subclaims via
few-shot prompting of Qwen 2.5 32B (Qwen et al., 2025). Expert annotators versed in
the claim decomposition literature then manually correct these decompositions to ensure
atomicity and faithfulness to the original text.5
WIKIVIDEO
Video Length (s) 79.6
Article Length (toks) 118
Videos 7.65
Audio Subclaims 25.0
Video Subclaims 18.3
OCR Subclaims 28.5
A/V/O Subclaims 13.0
Total Subclaims 51.1
Table 1: Dataset (top) and per-event
(bottom) averages.Subclaim Grounding Given the corrected sub-
claims for the Wikipedia lead section associated with
each event, we next attempt to ground the subclaims
in relevant videos. This grounding task provides
modality-specific annotations for each subclaim and for
each video associated with the target event, as anno-
tators were asked to indicate whether a subclaim is
supported by the video’s (non-text) visual content, its
OCR content, its audio content, or none of the above.
In-domain expert annotators completed these annota-
tions for all 58 events and 503 videos. Our pilot with
these annotators obtained a high overall agreement
(α=.767); see Appendix A.
Article Rewriting Finally, given the grounded set of subclaims and their corresponding
videos, three of the authors rewrote the Wikipedia lead sections such that the resulting
articles contained all and only information supported by the video-grounded subclaims.
During this stage, six events were found to have too few grounded subclaims to support a
rewritten article of any substance, and were subsequently removed from the final dataset.
3As MultiVENT is multilingual , not all events it contains satisfy (1) and many also do not satisfy (2).
4https://en.wikipedia.org/wiki/Wikipedia:Manual ofStyle/Lead section
5Appendix B has details on the claim decomposition and subclaim correction.
4

Preprint. Under review.
Figure 3: CAG involves an iterative exchange between (1) a VideoLLM that generates per-
video summaries and (2) a reasoning model that evaluates them and produces more event-
targeted prompts that are then fed back to the VideoLLM to obtain more comprehensive
summaries. Finally, a text-only LLM (3) aggregates these summaries into an full article.
Boxes A and B show shortened reasoning chains from the reasoner.
Final Dataset The final WIKIVIDEO dataset consists of 52 events (topics) spanning 398
videos annotated with grounded subclaims, where each event is associated with a fully
grounded, expert-written article. Table 1 provides a summary, with more in Appendix A, B.
4 Article Generation from Multiple Videos
Task The WIKIVIDEO article generation task takes as input (1) a topic event T, (2) a query
about T, and (3) a set of videos V={v1,. . .,vn}deemed relevant to (i.e. depicting some
facet of) T. The output is then a natural language article Apgenerated conditional on T,Q,
and V. In this work, we consider two possible sources for V: the reference set of videos for
Tas annotated in MultiVENT 1.0 and 2.0 (the oracle setting) and a set of videos obtained
from a retrieval model (the RAG setting).
4.1 C ollaborative A rticle G eneration (CAG)
Overview Conditional text generation from multiple videos faces several challenges that
hinder the efficiency and effectiveness of current methods. First, open-source VideoLLMs
are generally trained to produce low-level scene descriptions, making extraction of high-
level concepts (necessary for complex event understanding) a challenge, even based on a
single video—let alone multiple. Second, running inference over multiple long videos is
memory-intensive. For instance, in preliminary experiments with several of the VideoLLMs
we consider in section 5, even 8 80GB A100s struggled to accommodate a single long video
(5+ minutes) at 1 fps, as well as two or more videos at 0.25 fps.
To help address these limitations, we introduce Collaborative Article Generation ( CAG ;
Figure 3), a method for article generation from multiple videos that draws on recent devel-
opments in test-time scaling (DeepSeek-AI et al., 2025; Huang et al., 2025; Weller et al., 2025;
Jurayj et al., 2025) in addition to the classic notion of relevance feedback from information
retrieval (Rocchio, 1971). CAG features three core components: a VideoLLM, a text-based
reasoning model, and a text-only LLM.
Collaborative Per-Video Summarization The first phase of CAG involves a collaborative,
iterative exchange between the VideoLLM and the reasoning model. The VideoLLM begins
by generating generic summaries of each video based on a simple prompt to “describe the
video in detail.” The resulting summaries provide salient low-level information, covering
scene descriptions and prominent on-screen text.
5

Preprint. Under review.
Next, the reasoning model assists the VideoLLM in producing a refined summary for each
video that covers higher-level information about the underlying event. Concretely, the
reasoning model is given both Q(here, the name of the target event T) and the initial
generic summary for a particular video as input, and is then asked to either: (1) return the
original summary if the reasoning model deems it to be adequate, or else (2) generate a new
prompt seeking additional information about Tnot attested in the input summary. This
new prompt is then used to elicit a refined summary from the VideoLLM—an action we
dub REPROMPTING . The reasoning model can thus be understood as providing a form
of relevance feedback on the VideoLLM-generated summary with respect to Q(and thus
toT). This process is iterative because the reasoning model may in principle REPROMPT
repeatedly—requesting new summaries from the VideoLLM that are more relevant to Q
until it is satisfied with the result. In practice, we enforce a maximum REPROMPTING
iteration budget —analogous to test-time compute budgets for recent reasoning models—and
show in section 5 that a higher budget tends to yield higher-quality articles.6
Article Synthesis Once the reasoning model determines that the current query is adequate
(or the REPROMPTING iteration budget is exhausted), the final article is synthesized using
a text-only LLM. This model takes as input (1) the original generic summary output by
the VideoLLM for each video; (2) all REPROMPTED VideoLLM queries and their resulting
(more event-targeted) summaries; and (3; optionally) an audio transcript of each video.
Given these inputs, the model is then instructed to generate the full article about the event T.
However, we do not provide the explicit topic to the LLM to prevent generating the article
based purely on parametric memory.7
We note that in virtue of REPROMPTING —enabling a reasoning model to iteratively craft
prompts for the VideoLLM in order to produce summaries more explicitly targeted to an
event of interest ( T)—CAG goes some way toward mitigating the problem of summaries
that are overly focused on low-level descriptions (our first concern). Further, in processing
one video at a time, CAG reduces the memory burden of simultaneous processing of
multiple long videos (our second concern).
5 Experiments
We conduct experiments on WIKIVIDEO that (1) compare different VideoLLMs; (2) bench-
mark CAG against baseline report generation approaches; (3) assess the impact of including
raw audio and audio transcripts as input to the text-only LLM; and (4) evaluate the effec-
tiveness of different retrievers in the RAG setting, where Q-relevant videos must first be
retrieved. (1)-(3) are conducted in the oracle setting, where Vconsists of all and only the
Q-relevant videos.
Models For VideoLLMs, we consider LLaVA-Video-72B (Zhang et al., 2024a), VAST (Chen
et al., 2024), InternVideo2.5-8B (Wang et al., 2025), and QwenVL2.5-72B (Bai et al., 2025). We
use DeepSeek-R1 distilled to Qwen-32B (DeepSeek-AI et al., 2025) as the reasoning model
and Qwen2.5 Qwen et al. (2025) as the text-only LLM that generates the final article.
Metrics We use a suite of different metrics to evaluate the generated articles. We first
present ROUGE- {1,2,LCS }F1(R1, R2, RL; Lin, 2004) and BERTScore F1(BS; Zhang et al.,
2019) as two widely used metrics for free-form text generation, using the human-written
summaries as references. As these metrics largely focus on lexical similarity, we additionally
present AlignScore (AS; Zha et al., 2023), a metric for factual consistency based on a learned
text-pair alignment function that outputs a scalar value in [0, 1]representing the degree of
information alignment between the two texts.
6Additionally, the reasoning model’s test-time compute budget —the length of its reasoning chains—
could in principle be expanded in order to obtain higher-quality prompts, although we leave an
exploration of this hyperparameter to future work. Appendix C has further details on VideoLLM and
REPROMPTING prompts.
7See for further details Appendix D
6

Preprint. Under review.
Method VideoLLM R1 R2 RL BS Arg AS
CONCAT GENLLaVA-Video 7.34 1.60 4.78 71.99 19.31 5.08
VAST 16.62 1.71 11.19 80.55 8.04 7.13
InternVideo2.5 11.85 2.32 7.90 80.78 18.33 9.53
QwenVL2.5 11.34 3.13 7.06 81.60 23.72 8.01
CONCAT REPROMPTLLaVA-Video 6.36 1.51 4.22 80.03 21.34 5.50
InternVideo2.5 6.93 1.68 4.83 79.62 22.48 6.19
QwenVL2.5 8.38 2.71 5.49 81.94 22.89 7.17
CAG-0LLaVA-Video 30.02 8.68 17.59 77.59 26.21 13.51
VAST 19.55 1.45 12.40 82.21 11.23 10.87
InternVideo2.5 32.54 8.98 19.47 85.82 25.65 17.58
QwenVL2.5 33.58 10.15 19.15 86.18 28.97 15.63
CAG-2LLaVA-Video 33.38 10.05 19.44 84.55 28.26 15.23
InternVideo2.5 33.91 9.58 20.07 86.13 27.01 14.23
QwenVL2.5 33.96 10.90 19.45 86.35 30.77 14.29
Table 2: WIKIVIDEO article generation results for CAG and baselines (see §5)without audio
inputs (i.e. vision only). CAG obtains the strongest results ( bolded ) across most metrics.
QwenVL QwenVL+CAG
The video depicts a catastrophic fire engulf-
ing a historic cathedral, likely Notre-Dame
in Paris, France. The flames are intense and
widespread, with thick smoke billowing into
the sky. The iconic spire of the cathedral is vis-
ibly on fire, with the structure glowing red
from the heat. As the fire rages, the spire
begins to weaken and eventually collapses,
sending a cascade of debris downward. The
surrounding buildings and trees are partially
obscured by the smoke, emphasizing the scale
and severity of the blaze. The scene is both
dramatic and tragic, capturing a moment of
significant destruction to a beloved landmark.On April 15, 2019, a catastrophic fire broke out at Notre-Dame Cathedral in Paris, France, causing
extensive damage to the historic landmark. The fire, which began in the attic of the cathedral,
rapidly spread, engulfing the roof and the iconic spire, which collapsed in a dramatic fall wit-
nessed by a shocked crowd gathered outside. The cathedral, a masterpiece of Gothic architecture
dating back to the 12th century , suffered significant structural damage, with the roof and spire
completely destroyed. Despite the intense blaze, the main stone structure, including the two bell
towers, remained standing. Firefighters worked tirelessly throughout the night to contain the fire,
using advanced equipment and techniques to prevent further damage .The event drew global
attention, with people around the world expressing sorrow and solidarity . In the aftermath,
French President Emmanuel Macron pledged to rebuild the cathedral, and a fundraising cam-
paign was launched to support the restoration efforts. The fire raised questions about the safety
and preservation of historic landmarks and highlighted the cultural and spiritual significance
of Notre-Dame to both France and the international community.
Table 3: Qualitative Outputs on the Notre-Dame Fire query. The left response comes from
video 117864206475218944. Bold text represents information not in the videos.
Lastly, we also evaluate the extent to which predicted articles recover specific pieces of
event-relevant information. We map each WIKIVIDEO event into the 7-type event ontology
defined in the MultiVENT-G dataset (Sanders et al., 2024), each of which is associated with
a set of role-like questions about events of that type.8We use an LLM (GPT-4o) to extract
answers to these questions from both the reference and predicted articles. Since a question
may have multiple answers, we compute a maximum bipartite matching between predicted
and reference answers, obtaining an alignment between them that optimizes normalized
edit distance between paired answer spans. We then report an answer span F1given this
alignment, using normalized edit distance in lieu of (overly stringent) exact match. Prior
work on event extraction has leveraged similar metrics to evaluate event argument F1(Du
et al., 2021; Chen et al., 2023a;b; Vashishtha et al., 2024), so we refer to this metric as “Arg.”
Baselines We consider several baseline article generation methods that ablate different
components of CAG . The first baseline ( CONCAT GEN) simply concatenates the generic per-
video summaries to produce the final article, ablating both the aggregator and reprompting.
The second (C ONCAT REPROMPT ) concatenates only the per-video R EPROMPTED summaries,
excluding the generic ones, while still ablating the aggregator. The third, ( CAG-0 ), uses
the aggregator but fixes CAG ’s iteration budget to 0—relying exclusively on the generic
per-video summaries. The comparison between CAG-0 and CAG (with an iteration budget
of 2) thus offers an illustration of test-time scaling of CAG via a larger iteration budget.
8The event types are Sporting Events, Natural Disasters, Elections, Social Events, Demonstrations,
Discoveries/Launches, and Political Developments. Event specific scores can be found in Appendix F
7

Preprint. Under review.
Method VideoLLM R1 R2 RL BS Arg AS
CONCAT GEN+AUDIOLLaVA-Video 5.21 1.30 3.54 79.49 20.25 5.95
VAST 16.88 1.59 11.78 80.44 8.20 8.20
InternVideo2.5 11.56 2.33 7.83 80.72 18.72 8.48
QwenVL2.5 11.05 3.20 6.89 81.63 22.11 7.50
CAG-2 +AUDIOLLaVA-Video 29.35 8.07 16.85 77.30 25.58 13.17
InternVideo2 32.79 8.82 19.18 85.81 24.77 13.13
QwenVL2.5 32.05 9.17 18.99 85.74 26.25 12.62
CAG-2 QwenVL2.5 33.96 10.90 19.45 86.35 30.80 14.29
Table 4: WIKIVIDEO article generation results for CAG and baselines with audio inputs.
Bottom row shows CAG without audio (copied from Table 2), which performs best.
5.1 Experiment 1: CAG and Baselines
Table 2 shows WIKIVIDEO article generation results comparing CAG and the baselines de-
scribed above. We find that simple concatenations of the per-video summaries—whether the
initial generic ones ( CONCAT GEN) or those obtained via reprompting ( CONCAT REPROMPT )—
yield articles of very poor quality. Although manual inspection reveals these per-video
summaries to contain mostly accurate descriptions of scenes and notable visual entities (e.g.
the Eiffel tower), we take this as compelling evidence that individual video summaries are
thoroughly inadequate for our task, absent higher-level synthesis.
Results with CAG-0 and CAG , both of which incorporate the text-only aggregator LLM,
offer further evidence for this interpretation, as we observe large gains across all metrics for
both of these methods relative to the CONCAT baselines. For most metrics, CAG also obtains
superior results to CAG-0 , suggesting that supplying the aggregator with the REPROMPTED
summaries (in addition to the generic ones) further enhances article quality.
Table 3 provides a qualitative example comparing a summary for a single video about the
2019 Notre Dame Cathedral fire generated by QwenVL2.5 (left) and the final article output
by QwenVL2.5-based CAG (right). While the single video summary correctly identifies
the location (Paris) and the most salient visual entity (Notre Dame), the descriptions are
alternately too granular (e.g. the surrounding buildings are trees are overly obscured by smoke ) and
too florid (e.g. the scene is both dramatic and tragic, capturing a moment of significant destruction to
a beloved landmark ) for outputs intended to be read by humans, like a Wikipedia-style article
on the event. By contrast, the CAG -generated article achieves a much more appropriate level
of granularity, although generation of ungrounded details (bolded) remains a challenge—
despite our prompt excluding the target event mention and explicitly prohibiting reliance
on non-video knowledge sources.
5.2 Experiment 2: Incorporating Audio
Articles in WIKIVIDEO have many claims grounded partly or only in videos’ audio sig-
nal (Table 1). Here, we consider the impact of adding audio information as additional
input to CAG and to CONCAT GEN. For all methods except for VAST–which takes raw audio
input–we transcribe the audio of each video using whisper-v3 large (Radford et al., 2022).
For the CONCAT GENbaseline, we provide the transcription as additional input alongside
the instruction and frames to the VideoLLM. For CAG , we provide the transcriptions for
each video together with the per-video summaries as input to the text-only aggregator LLM.
Table 4 presents the results. Similar to the previous experiment, we find that CONCAT GEN
continues to yield poor quality articles, even with audio information. Notably, however,
both CONCAT GENand CAG consistently obtain worse results with audio inputs than with-
out (Table 2)—despite the sizable fraction of audio-support claims in WIKIVIDEO . For
CONCAT GEN, this may partly be explained by the fact that the pretraining data for the Vide-
oLLMs we study does not include audio transcripts, and thus the prompts that incorporate
them are out-of-distribution. For CAG , we found that including audio transcripts tended
to result in substantially shorter final articles (avg. ∼164 tokens) than omitting them (avg.
8

Preprint. Under review.
Method Retriever VideoLLM R1 R2 RL BS Arg AS
CAG-2V-ColBERTIV 20.46 3.83 12.77 82.74 17.24 7.49
QVL 24.13 4.68 14.42 83.67 20.93 10.59
MMMORRFIV 20.52 3.10 12.63 76.61 16.71 8.84
QVL 23.84 4.91 14.32 77.85 20.65 9.01
Oracle QVL 33.96 10.90 19.45 86.35 30.80 14.29
Table 5: Results with CAG using different retrievers. The top 5 videos in a ranked list are
used for generation. V-ColBERT=Video-ColBERT; IV=InternVideo2.5; QVL=QwenVL2.5.
Oracle retrieval results are from Table 2.
∼206 tokens)—suggesting that the former may be less thorough in their coverage of the
event relative to the references, leading to lower metric scores (Appendix E has quantitative
examples). Identifying ways to more effectively incorporate audio into the WIKIVIDEO task
thus constitutes an intriguing direction for future work.
5.3 Experiment 3: Retrieval Augmented Generation
In contrast to the previous two experiments, which were run using only relevant videos for
each target event (the oracle setting), here we consider the RAG setting, in which relevant
videos must be retrieved. We use the full set of MultiVENT 2.0 (Kriz et al., 2024) videos
from the test set as our corpus (109K videos).
We perform retrieval under two settings: video-only, and audio-visual. In the video-only
setting, we create our index using VideoColBERT (Reddy et al., 2025), an efficient bi-encoder
method for token-wise video retrieval. We create 26 vectors to represent the spatial (12)
and spatio-temporal (14) features and train on the subset of videos in the MultiVENT 2.0
training split that have queries associated with them. In the audio-visual setting, we use all
features from the videos to create the index, which includes visual frame features, extracted
OCR, and audio transcripts. We use MMMORRF (Samuel et al., 2025), the state-of-the-art
retrieval method on MultiVENT 2.0, which combines these features to produce ranked lists.
We generate articles using the top 5 videos for each query.
Table 5 reports article generation results using MMMORRF (nDCG@5: 0.66) and Video-
ColBERT (nDCG@5: 0.22). We observe a significant decrease in CAG performance in
moving from oracle retrieval to the RAG setting. This failure falls on the aggregation
module of CAG : the text-only aggregator LLM struggles to include information from each
video summary, even for irrelevant videos. In such cases, we find that the aggregator usually
partitions the lead section into distinct topics instead of writing about the event covered by
the (relevant) majority of retrieved videos.
6 Conclusion
In this paper we introduce the difficult task of automatically generating Wikipedia-style
articles based on multiple videos about real-world events. To facilitate work on this task,
we collect and release WIKIVIDEO , a benchmark of high-quality, expert-written articles
grounded in diverse videos, ranging from amateur footage to professional news coverage,
which are densely annotated for multimodal support of the articles’ claims. Further, since
existing systems for video-based summarization tasks are memory-intensive and overly
focused on low-level video descriptions, we introduce Collaborative Article Generation
(CAG)—a strong baseline for our task that leverages elements of relevance feedback and
test-time scaling to iteratively construct high-level event-centric summaries. Our experiments
demonstrate the effectiveness of CAG compared to alternative baselines—both in oracle
retrieval and RAG settings. While CAG takes a significant step toward addressing the
above limitations, other challenges remain—such as the need for improved methods for
integrating audio signal into the article generation process and more effective training of
VideoLLMs to better support this task—which represent exciting directions for future work.
9

Preprint. Under review.
7 Acknowledgments
This material is based upon work supported by the National Science Foundation Graduate
Research Fellowship under Grant No. DGE2139757. Any opinion, findings, and conclusions
or recommendations expressed in this material are those of the author(s) and do not nec-
essarily reflect the views of the National Science Foundation. We thank Neha Verma and
Marc Marone for their feedback on writing and narrative.
References
Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan
Russell. Localizing moments in video with natural language. In Proceedings of the IEEE
International Conference on Computer Vision (ICCV) , Oct 2017.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng
Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai
Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang,
Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin.
Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923 .
Samuel Barham, Orion Weller, Michelle Yuan, Kenton Murray, Mahsa Yarmohammadi,
Zhengping Jiang, Siddharth Vashishtha, Alexander Martin, Anqi Liu, Aaron Steven White,
Jordan Boyd-Graber, and Benjamin Van Durme. Megawika: Millions of reports and their
sources across 50 diverse languages, 2023. URL https://arxiv.org/abs/2307.07049 .
Samuel Barham, Chandler May, and Benjamin Van Durme. MegaWika 2: A more compre-
hensive multilingual collection of articles and their sources, 2025.
David Chen and William Dolan. Collecting highly parallel data for paraphrase evaluation.
In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea (eds.), Proceedings of the 49th Annual
Meeting of the Association for Computational Linguistics: Human Language Technologies , pp.
190–200, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.
URLhttps://aclanthology.org/P11-1020/ .
Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing
Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset.
Advances in Neural Information Processing Systems , 36, 2024.
Yunmo Chen, William Gantt, Tongfei Chen, Aaron White, and Benjamin Van Durme. A
unified view of evaluation metrics for structured prediction. In Houda Bouamor, Juan
Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing , pp. 12868–12882, Singapore, December 2023a. Association
for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.795. URL https:
//aclanthology.org/2023.emnlp-main.795/ .
Yunmo Chen, William Gantt, Weiwei Gu, Tongfei Chen, Aaron White, and Benjamin
Van Durme. Iterative document-level information extraction via imitation learning.
In Andreas Vlachos and Isabelle Augenstein (eds.), Proceedings of the 17th Conference of the
European Chapter of the Association for Computational Linguistics , pp. 1858–1874, Dubrovnik,
Croatia, May 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.
eacl-main.136. URL https://aclanthology.org/2023.eacl-main.136/ .
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan
Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong
Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu,
Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong
Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L.
Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
10

Preprint. Under review.
Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang,
Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun
Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu
Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu
Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu,
Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao,
Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An,
Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie,
Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin,
Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou,
Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao
Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong,
Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo,
Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui
Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren,
Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao,
Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie,
Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
URLhttps://arxiv.org/abs/2501.12948 .
Xinya Du, Alexander Rush, and Claire Cardie. GRIT: Generative role-filler transformers
for document-level event entity extraction. In Paola Merlo, Jorg Tiedemann, and Reut
Tsarfaty (eds.), Proceedings of the 16th Conference of the European Chapter of the Association
for Computational Linguistics: Main Volume , pp. 634–644, Online, April 2021. Association
for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.52. URL https://
aclanthology.org/2021.eacl-main.52/ .
Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A
large-scale multi-document summarization dataset and abstractive hierarchical model. In
Anna Korhonen, David Traum, and Llu ´ıs M `arquez (eds.), Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics , pp. 1074–1084, Florence, Italy, July
2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1102. URL
https://aclanthology.org/P19-1102/ .
William Gantt, Alexander Martin, Pavlo Kuchmiichuk, and Aaron Steven White. Event-
keyed summarization. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen
(eds.), Findings of the Association for Computational Linguistics: EMNLP 2024 , pp. 7333–
7345, Miami, Florida, USA, November 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.findings-emnlp.431. URL https://aclanthology.org/2024.
findings-emnlp.431/ .
Anisha Gunjal and Greg Durrett. Molecular facts: Desiderata for decontextualization in LLM
fact verification. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings
of the Association for Computational Linguistics: EMNLP 2024 , pp. 3751–3768, Miami, Florida,
USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.
findings-emnlp.215. URL https://aclanthology.org/2024.findings-emnlp.215/ .
Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool. Creating sum-
maries from user videos. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars
(eds.), Computer Vision – ECCV 2014 , pp. 505–520, Cham, 2014. Springer International
Publishing. ISBN 978-3-319-10584-0.
Bo He, Jun Wang, Jielin Qiu, Trung Bui, Abhinav Shrivastava, and Zhaowen Wang. Align
and Attend: Multimodal Summarization with Dual Contrastive Losses . In 2023 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) , pp. 14867–14878, Los Alami-
tos, CA, USA, June 2023. IEEE Computer Society. doi: 10.1109/CVPR52729.2023.01428.
URLhttps://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.01428 .
11

Preprint. Under review.
Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay,
Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend.
Advances in neural information processing systems , 28, 2015.
Qisheng Hu, Quanyu Long, and Wenya Wang. Decomposition dilemmas: Does claim
decomposition boost or burden fact-checking performance?, 2025. URL https://arxiv.
org/abs/2411.02400 .
Hang Hua, Yunlong Tang, Chenliang Xu, and Jiebo Luo. V2xum-llm: Cross-modal video
summarization with temporal prompt instruction tuning, 2024. URL https://arxiv.org/
abs/2404.12353 .
Kung-Hsiang Huang, Philippe Laban, Alexander Fabbri, Prafulla Kumar Choubey, Shafiq
Joty, Caiming Xiong, and Chien-Sheng Wu. Embrace divergence for richer insights: A
multi-document summarization benchmark and a case study on summarizing diverse
information from news articles. In Kevin Duh, Helena Gomez, and Steven Bethard
(eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , pp. 570–
593, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.
18653/v1/2024.naacl-long.32. URL https://aclanthology.org/2024.naacl-long.32/ .
Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao
Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large
language models, 2025. URL https://arxiv.org/abs/2503.06749 .
Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward
spatio-temporal reasoning in visual question answering, 2017. URL https://arxiv.org/
abs/1704.04497 .
William Jurayj, Jeffrey Cheng, and Benjamin Van Durme. Is that your final answer? test-time
scaling improves selective question answering, 2025. URL https://arxiv.org/abs/2502.
13962 .
Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-
captioning events in videos. In Proceedings of the IEEE International Conference on Computer
Vision (ICCV) , Oct 2017.
Reno Kriz, Kate Sanders, David Etter, Kenton Murray, Cameron Carpenter, Kelly Van Ochten,
Hannah Recknor, Jimena Guallar-Blasco, Alexander Martin, Ronald Colaianni, Nolan
King, Eugene Yang, and Benjamin Van Durme. Multivent 2.0: A massive multilingual
benchmark for event-centric video retrieval, 2024. URL https://arxiv.org/abs/2410.
11619 .
Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W. Oard, and
Luca Soldaini ˚aand Eugene Yang. Overview of the trec 2023 neuclir track, 2024. URL
https://arxiv.org/abs/2404.08071 .
Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. TVQA: Localized, compositional
video question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi
Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing , pp. 1369–1379, Brussels, Belgium, October-November 2018. Association for
Computational Linguistics. doi: 10.18653/v1/D18-1167. URL https://aclanthology.
org/D18-1167/ .
Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning
to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) , pp. 19108–19118, June 2022.
Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summa-
rization Branches Out , pp. 74–81, Barcelona, Spain, July 2004. Association for Computa-
tional Linguistics. URL https://aclanthology.org/W04-1013/ .
12

Preprint. Under review.
Jingyang Lin, Hang Hua, Ming Chen, Yikang Li, Jenhao Hsiao, Chiuman Ho, and Jiebo Luo.
Videoxum: Cross-modal visual and textural summarization of videos. IEEE Transactions
on Multimedia , 26:5548–5560, 2024. doi: 10.1109/TMM.2023.3335875.
Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and
Noam Shazeer. Generating wikipedia by summarizing long sequences. In International
Conference on Learning Representations , 2018a. URL https://openreview.net/forum?id=
Hyg0vbWC- .
Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser,
and Noam M. Shazeer. Generating wikipedia by summarizing long sequences. ArXiv ,
abs/1801.10198, 2018b. URL https://api.semanticscholar.org/CorpusID:3608234 .
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit
Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic
evaluation of factual precision in long form text generation. In Houda Bouamor, Juan
Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing , pp. 12076–12100, Singapore, December 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.741. URL https:
//aclanthology.org/2023.emnlp-main.741/ .
Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, C ¸a˘glar Gu ˙lc ¸ehre, and Bing Xiang.
Abstractive text summarization using sequence-to-sequence RNNs and beyond. In
Stefan Riezler and Yoav Goldberg (eds.), Proceedings of the 20th SIGNLL Conference on
Computational Natural Language Learning , pp. 280–290, Berlin, Germany, August 2016.
Association for Computational Linguistics. doi: 10.18653/v1/K16-1028. URL https:
//aclanthology.org/K16-1028/ .
Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu,
Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming
Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,
Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang
Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan
Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115 .
Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya
Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL
https://arxiv.org/abs/2212.04356 .
Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray,
Reno Kriz, Celso M. de Melo, Benjamin Van Durme, and Rama Chellappa. Video-
colbert: Contextualized late interaction for text-to-video retrieval, 2025. URL https:
//arxiv.org/abs/2503.19009 .
Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag:
Retrieval-augmented generation with extreme long-context videos, 2025. URL https:
//arxiv.org/abs/2502.01549 .
J. J. Rocchio. Relevance feedback in information retrieval. 1971. URL https://api.
semanticscholar.org/CorpusID:61859400 .
Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eis-
ape, Arun Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Car-
penter, David Etter, Efsun Kayi, Matthew Wiesner, Kenton Murray, and Reno Kriz.
Mmmorrf: Multimodal multilingual modularized reciprocal rank fusion, 2025. URL
https://arxiv.org/abs/2503.20698 .
Kate Sanders, David Etter, Reno Kriz, and Benjamin Van Durme. MultiVENT: Multilingual
videos of events and aligned natural text. In Thirty-seventh Conference on Neural Information
Processing Systems Datasets and Benchmarks Track , 2023. URL https://openreview.net/
forum?id=2CJUQe6IoR .
13

Preprint. Under review.
Kate Sanders, Reno Kriz, David Etter, Hannah Recknor, Alexander Martin, Cameron Carpen-
ter, Jingyang Lin, and Benjamin Van Durme. Grounding partially-defined events in multi-
modal data. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the
Association for Computational Linguistics: EMNLP 2024 , pp. 15905–15927, Miami, Florida,
USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.
findings-emnlp.934. URL https://aclanthology.org/2024.findings-emnlp.934/ .
Christina Sauper and Regina Barzilay. Automatically generating Wikipedia articles: A
structure-aware approach. In Keh-Yih Su, Jian Su, Janyce Wiebe, and Haizhou Li
(eds.), Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the
4th International Joint Conference on Natural Language Processing of the AFNLP , pp. 208–
216, Suntec, Singapore, August 2009. Association for Computational Linguistics. URL
https://aclanthology.org/P09-1024/ .
Yijia Shao, Yucheng Jiang, Theodore A. Kanell, Peter Xu, Omar Khattab, and Monica S. Lam.
Assisting in writing wikipedia-like articles from scratch with large language models, 2024.
URLhttps://arxiv.org/abs/2402.14207 .
Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing
web videos using titles. In 2015 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) , pp. 5179–5187, 2015. doi: 10.1109/CVPR.2015.7299154.
Neha Srikanth and Rachel Rudinger. Nli under the microscope: What atomic hypothesis
decomposition reveals, 2025. URL https://arxiv.org/abs/2502.08080 .
Sai Vallurupalli, Sayontan Ghosh, Katrin Erk, Niranjan Balasubramanian, and Francis
Ferraro. POQue: Asking participant-specific outcome questions for a deeper under-
standing of complex events. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang
(eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Pro-
cessing , pp. 8674–8697, Abu Dhabi, United Arab Emirates, December 2022. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.594. URL
https://aclanthology.org/2022.emnlp-main.594/ .
Siddharth Vashishtha, Alexander Martin, William Gantt, Benjamin Van Durme, and Aaron
White. FAMuS: Frames across multiple sources. In Kevin Duh, Helena Gomez, and
Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers) , pp. 8250–8273, Mexico City, Mexico, June 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.naacl-long.457. URL https://aclanthology.org/
2024.naacl-long.457/ .
William Walden, Pavlo Kuchmiichuk, Alexander Martin, Chihsheng Jin, Angela Cao, Claire
Sun, Curisia Allen, and Aaron Steven White. Cross-document event-keyed summariza-
tion, 2024. URL https://arxiv.org/abs/2410.14795 .
Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang.
Vatex: A large-scale, high-quality multilingual dataset for video-and-language research,
2020. URL https://arxiv.org/abs/1904.03493 .
Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang,
Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao,
Yali Wang, and Limin Wang. Internvideo2.5: Empowering video mllms with long and
rich context modeling, 2025. URL https://arxiv.org/abs/2501.12386 .
Miriam Wanner, Benjamin Van Durme, and Mark Dredze. Dndscore: Decontextualization
and decomposition for factuality verification in long-form text generation, 2024a. URL
https://arxiv.org/abs/2412.13175 .
Miriam Wanner, Seth Ebner, Zhengping Jiang, Mark Dredze, and Benjamin Van Durme. A
closer look at claim decomposition. In Danushka Bollegala and Vered Shwartz (eds.),
Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024) ,
pp. 153–175, Mexico City, Mexico, June 2024b. Association for Computational Linguistics.
14

Preprint. Under review.
doi: 10.18653/v1/2024.starsem-1.13. URL https://aclanthology.org/2024.starsem-1.
13/.
Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and Benjamin Van
Durme. Rank1: Test-time compute for reranking in information retrieval, 2025. URL
https://arxiv.org/abs/2502.18418 .
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for
bridging video and language. In 2016 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) , pp. 5288–5296, 2016. doi: 10.1109/CVPR.2016.571.
Zhongyu Yang, Jun Chen, Dannong Xu, Junjie Fei, Xiaoqian Shen, Liangbing Zhao, Chun-
Mei Feng, and Mohamed Elhoseiny. Wikiautogen: Towards multi-modal wikipedia-style
article generation, 2025. URL https://arxiv.org/abs/2503.19065 .
Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao.
Activitynet-qa: a dataset for understanding complex web videos via question answering.
InProceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First
Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on
Educational Advances in Artificial Intelligence , AAAI’19/IAAI’19/EAAI’19. AAAI Press,
2019. ISBN 978-1-57735-809-1. doi: 10.1609/aaai.v33i01.33019127. URL https://doi.org/
10.1609/aaai.v33i01.33019127 .
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. AlignScore: Evaluating factual
consistency with a unified alignment function. In Anna Rogers, Jordan Boyd-Graber,
and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pp. 11328–11348, Toronto, Canada, July
2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.634.
URLhttps://aclanthology.org/2023.acl-long.634/ .
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert. ArXiv , abs/1904.09675, 2019. URL https://api.
semanticscholar.org/CorpusID:127986044 .
Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun MA, Ziwei Liu, and Chunyuan Li. Video
instruction tuning with synthetic data, 2024a. URL https://openreview.net/forum?id=
8Livf4oZxz .
Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video
instruction tuning with synthetic data, 2024b. URL https://arxiv.org/abs/2410.02713 .
Luowei Zhou, Yannis Kalantidis, Xinlei Chen, Jason J. Corso, and Marcus Rohrbach.
Grounded video description. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) , June 2019.
Fangwei Zhu, Shangqing Tu, Jiaxin Shi, Juanzi Li, Lei Hou, and Tong Cui. TWAG: A topic-
guided Wikipedia abstract generator. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto
Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume
1: Long Papers) , pp. 4623–4635, Online, August 2021. Association for Computational
Linguistics. doi: 10.18653/v1/2021.acl-long.356. URL https://aclanthology.org/2021.
acl-long.356/ .
15

Preprint. Under review.
A Dataset Statistics
Video-Grounded Claims 954
Audio-Grounded Claims 1299
A+V-Grounded Claims 674
Avg. Claims / Event 51.10
Avg. Audio Claims / Event 24.98
Avg. Video Claims / Event 18.35
Avg. OCR Claims / Event 28.53
Avg. Claims in All / Event 12.96
(a) Claim StatisticsMax Videos for a Topic 12
Avg. Videos / Event 7.65
Total Relevant Videos 398
RAG Video Data Lake 109K
Avg. Video Length (Relevant) 79.57s
Avg. Video Length (RAG) 145s
Max Relevant Video Length 586.26s
Min Relevant Video Length 4.55s
(b) Video Statistics
Table 6: W IKIVIDEO Statistics
Table 6a contains additional statistics about the claims in WIKIVIDEO and Table 6b has
additional statistics about the videos. Further examples of WIKIVIDEO articles can be found
in Table 18 and Table 19. We also report the IAA metrics in Table 7.
B Data Collection
This appendix discusses the annotation process for WIKIVIDEO in greater detail. Beyond
topic selection, recall that the annotation process includes claim decomposition (subsec-
tion B.1), subclaim rewriting (subsection B.2), subclaim grounding (subsection B.3), and
article rewriting (subsection B.4).
B.1 Claim Decomposition
Much recent work has studied the appropriate granularity of subclaims in a claim decompo-
sition and the applications of such decompositions to natural language inference (NLI) and
claim verification (Min et al., 2023; Gunjal & Durrett, 2024; Wanner et al., 2024b;a; Hu et al.,
2025; Srikanth & Rudinger, 2025). Our own claim decomposition method is most similar
to those of Gunjal & Durrett (2024) and Wanner et al. (2024a) in that we decontextualize
subclaims—insert elided or abstracted context (e.g. by substituting pronouns with named
entities)—up to the point that no further verification of extracted facts is required. (An ex-
ample of one of our claim decomposition prompts can be found in Figure 4). Although not a
rigourous form of decomposition, this notion is straightforward to apply during annotation.
Additionally for simplicity, we treat dates as named entities and do not decompose them
beyond their mention as given in the text. For example, some methods, like Wanner et al.
(2024b), may decompose the claim “The event occurred on 15 April 2019” as:
• The event occurred on the 15th
• The event occurred in April
• The event occurred in 2019
Modality α
Video 0.446
Audio 0.780
OCR 0.722
None 0.682
Overall 0.767
Table 7: Krippendorff’s αfor the claim grounding agreement of each modality. Each
judgment reflects a binary decision about whether a given claim is supported by a given
modality (or None of the modalities).
16

Preprint. Under review.
Instructions: - You are given a paragraph, and one sentence from the paragraph to decompose - You must decompose this into a set of
claims - You must decompose this into a JSON format ["claim": "...", "claim": "...", ...]
PARAGRAPH: On 15 April 2019, just before 18:20 CEST, a structural fire broke out in the roof space of Notre-Dame de Paris, a medieval
Catholic cathedral in Paris, France. By the time the fire was extinguished, the cathedral ¨s wooden spire (fl ˋeche) had collapsed, most
of the wooden roof had been destroyed, and the cathedral ¨s upper walls were severely damaged. Extensive damage to the interior was
prevented by the vaulted stone ceiling, which largely contained the burning roof as it collapsed. Many works of art and religious
relics were moved to safety, but others suffered smoke damage, and some of the exterior art was damaged or destroyed. The cathedral ¨s
altar, two pipe organs, and three 13th-century rose windows suffered little or no damage. Three emergency workers were injured. The
fire contaminated the site and nearby areas of Paris with toxic dust and lead. Notre-Dame did not hold a Christmas Mass in 2019, for
the first time since 1803. Investigators in 2020 believed the fire to have been "started by either a cigarette or a short circuit
in the electrical system".
SENTENCE: On 15 April 2019, just before 18:20 CEST, a structural fire broke out in the roof space of Notre-Dame de Paris, a medieval
Catholic cathedral in Paris, France.
DECOMPOSITION: [ "claim": "A structural fire broke out", "claim": "The fire broke out on 15 April 2019", "claim": "The fire broke
out just before 18:20 CEST", "claim": "The fire broke out in the roof space", "claim": "Notre-Dame de Paris is a medieval Catholic
cathedral", "claim": "Notre-Dame de Paris is located in Paris, France" ]
PARAGRAPH: The 2022 United States Senate election in Georgia was held on November 8, 2022, to elect a member of the U.S. Senate to
represent the state of Georgia. Incumbent Democratic senator Raphael Warnock won his first full term in office, defeating Republican
former football player Herschel Walker. Under Georgia"s two-round system, Warnock was re-elected in a runoff election on December
6 after neither candidate received over 50% of the vote on November 8. Warnock"s win was the only statewide victory for Democrats
in Georgia in 2022, as Republicans swept all other races.
SENTENCE: Under Georgia"s two-round system, Warnock was re-elected in a runoff election on December 6 after neither candidate
received over 50
DECOMPOSITION: [ "claim": "Georgia has a two-round election system", "claim": "The runoff election is part of the two-round election
system in Georgia", "claim": "A run off election occurs when no candidate receives 50% of the vote", "claim": "Neither candidate
received more than 50% of the vote in the first election", "claim": "The runoff election took place on December 6", "claim": "Warnock
won the runoff election", "claim": "Warnock was the incumbent candidate", "claim": "The first election took place on November 8" ]
PARAGRAPH: Hurricane Irma was an extremely powerful Cape Verde hurricane that caused widespread destruction across its path in early
September 2017. Irma was the first Category 5 hurricane to strike the Leeward Islands on record, followed by Maria two weeks later.
At the time, it was considered the most powerful hurricane on record in the open Atlantic region, outside of the Caribbean Sea and
Gulf of Mexico, until it was surpassed by Hurricane Dorian two years later. It was also the third-strongest Atlantic hurricane
at landfall ever recorded, just behind the 1935 Labor Day Hurricane and Dorian. The ninth named storm, fourth hurricane, second
major hurricane, and first Category 5 hurricane of the extremely active 2017 Atlantic hurricane season, Irma caused widespread and
catastrophic damage throughout its long lifetime, particularly in the northeastern Caribbean and the Florida Keys. It was also the
most intense hurricane to strike the continental United States since Katrina in 2005, the first major hurricane to make landfall
in Florida since Wilma in the same year, and the first Category 4 hurricane to strike the state since Charley in 2004. The word
Irmageddon was coined soon after the hurricane to describe the damage caused by the hurricane.
SENTENCE: The ninth named storm, fourth hurricane, second major hurricane, and first Category 5 hurricane of the extremely active
2017 Atlantic hurricane season, Irma caused widespread and catastrophic damage throughout its long lifetime, particularly in the
northeastern Caribbean and the Florida Keys.
DECOMPOSITION: [ "claim": "Irma was the ninth named storm of the 2017 Atlantic hurricane season", "claim": "Irma was the fourth
hurricane of the 2017 Atlantic hurricane season", "claim": "Irma was the second major hurricane of the 2017 Atlantic hurricane
season", "claim": "Irma was the first Category 5 hurricane of the 2017 Atlantic hurricane season", "claim": "The 2017 Atlantic
hurricane season was extremely active", "claim": "Irma caused widespread damage", "claim": "Irma caused catastrophic damage",
"claim": "Irma’s damage was particularly severe in the northeastern Caribbean", "claim": "Irma’s damage was particularly severe in
the Florida Keys", "claim": "Irma had a long lifetime", "claim": "Irma occurred during the 2017 Atlantic hurricane season" ]
PARAGRAPH: On November 30, 2018, at 8:29 a.m. AKST (17:29 UTC), a magnitude 7.1 earthquake hit Anchorage in South Central Alaska.
The earthquake’s epicenter was near Point Mackenzie, about north of Anchorage, and occurred at a depth of . It was followed six
minutes later by a magnitude 5.7 aftershock centered north-northwest of the municipality. The earthquake could be felt as far away
as Fairbanks.
SENTENCE: On November 30, 2018, at 8:29 a.m. AKST (17:29 UTC), a magnitude 7.1 earthquake hit Anchorage in South Central Alaska.
DECOMPOSITION: [ "claim": "The earthquake occurred", "claim": "The earthquake occurred on November 30, 2018", "claim": "The
earthquake occurred at 8:29 a.m. AKST", "claim": "The earthquake hit Anchorage", "claim": "The earthquake hit South Central Alaska",
"claim": "The earthquake had a magnitude of 7.1" ]
PARAGRAPH: Pok ´emon Go (stylized as Pok ´emon GO) is a 2016 augmented reality (AR) mobile game, part of the Pok ´emon franchise, developed
and published by Niantic in collaboration with Nintendo and The Pok ´emon Company for iOS and Android devices. It uses mobile devices
with GPS to locate, capture, train, and battle virtual Pok ´emon, which appear as if they are in the player’s real-world location. The
game is free-to-play; it uses a freemium business model combined with local advertising and supports in-app purchases for additional
in-game items. The game launched with around 150 species of Pok ´emon, which had increased to around 700 by 2021.
SENTENCE: Pok ´emon Go (stylized as Pok ´emon GO) is a 2016 augmented reality (AR) mobile game, part of the Pok ´emon franchise, developed
and published by Niantic in collaboration with Nintendo and The Pok ´emon Company for iOS and Android devices.
DECOMPOSITION: [ "claim": "Pok ´emon Go is a mobile game", "claim": "Pok ´emon Go is an augmented reality (AR) game", "claim": "Pok ´emon
Go was released in 2016", "claim": "Pok ´emon Go is part of the Pok ´emon franchise", "claim": "Pok ´emon Go was developed by Niantic",
"claim": "Pok ´emon Go was published by Niantic", "claim": "Niantic collaborated with Nintendo to develop Pok ´emon Go", "claim":
"Niantic collaborated with Nintendo to publish Pok ´emon Go", "claim": "Niantic collaborated with The Pok ´emon Company to develop
Pok´emon Go", "claim": "Niantic collaborated with The Pok ´emon Company to publish Pok ´emon Go", "claim": "Pok ´emon Go was developed for
iOS devices", "claim": "Pok ´emon Go was published for iOS devices", "claim": "Pok ´emon Go was developed for Android devices", "claim":
"Pok´emon Go was published for Android devices" ]
PARAGRAPH [paragraph] SENTENCE [sentence] DECOMPOSITION:
Figure 4: Prompt For Qwen 2.5 32B Claim Decomposition
However, this produces an additional burden on our downstream annotations requiring
the annotator to verify all 3 concepts. While it is possible to fail to recover finer-grained
information that may be attested in the video—e.g. the event occurred in 2019 —such cases
are rare and this decision substantially reduces the amount of labor required for subclaim
rewriting and grounding.
B.2 Subclaim Rewriting
We take the subclaims decomposed by Qwen2.5 32B and correct them manually, with three
of the authors serving as annotators. Figure 5 shows the interface that the annotators
17

Preprint. Under review.
Figure 5: The annotation interface for our subclaim grounding task. In this protocol, the
left hand side is both versions of the Wikipedia context. The top context is the paragraph
a sentence comes from and the bottom context is the lead section of the Wikipedia article.
On the right hand side is the sentence to be decomposed and its claims. The claims from
Qwen32B are prepopulated in the protocol and the rewriters edit them.
used to rewrite, add, and remove claims, and Figure 6 shows the instructions provided to
annotators.
B.3 Subclaim Grounding
Once the final set of claims is completed by the human annotator, the same annotators
ground the claims in the video content. We provide the instructions for this process in
Figure 8 and the protocol for this populated with an example sentence worth of claims and
video from the Notre-Dame fire in Figure 7.
Note that this grounding annotation was initially attempted via Amazon Mechanical Turk.
However, despite several iterations of annotation instructions, even Turkers with additional
Masters qualifications struggled to consistently ground claims in videos. We suspect this is
in part due to the amount of domain expertise and world knowledge required to properly
ground claims, and partially because raw real-time videos of events are inherently ambigu-
ous. Thus, the decision was made to leverage domain experts in order to ensure consistent
and high-quality annotations.
B.4 Article Rewriting
The last stage in the annotation process involves rewriting the original Wikipedia articles
based on the grounded claims. The instructions for this are provided in Figure 9.
C Relevance and RePrompt
In this section, we include the prompt used with the reasoning model (Figure 10) and
examples of real queries provided to the VideoLLM (Figure 11). We note that we had to add
force the model to produce the prefix ”Describe the video in detail and focus on ...” or else
sometimes the reasoner would not follow our instructions and produce a new query.
18

Preprint. Under review.
Claim Rewriting Annotation
In this task, you will be shown an excerpt from Wikipedia, a sentence, and a set of claims associated with that sentence. Given
this information, you will rewrite the set of claims into a new set of \cleaner" claims. To create the final set of claims, you will
deal with common issues like splitting a claim into two or more claims, adding a missing claim, or removing a duplicated claim.
Do not remove duplicates you remember from a previous sentence of the document. Treat each sentence as a unique instance without
recalling what you had annotated previously.
Annotation Protocol
The annotation protocol has 4 main sections. The R.H.S. is what is most relevant to annotators. It contains the current sentence
and the claims for the sentences as well as editable cards with each claim from the sentence in them. (Sometimes not all of them
are prepopulated, so if any are missing just hit the + button). The L.H.S. has the context that the sentences are taken from. The
top context is the paragraph where the current sentence comes from and below is the lead section of the wikipedia article that the
paragraph comes from.
Editing a claim
To edit a claim, you will click on the card that the claim is in. The claim will include above it the original claim and the textbox
will allow you to delete, edit, and rewrite claims.
Adding extra claims Sometimes claims are missed by the decomposer. To fix these press the + button( ) in the right side of the
interface. This will put a new claim into the interface. Note that these claims won’t include base claims in the interface because
the system did not predict the claim. This will NOT impact the annotation or future use of the claim.
Error Types Here are some examples of errors that you might encounter when doing the annotations.
Under Decomposition Under decomposition is when a claim includes multiple pieces of information that could be split into two or more
atomic facts.
Under Decomposition Example (loaded claim): Input: On 15 April 2019, just before 18:20 CEST, a structural fire broke out in the
roof space of Notre-Dame de Paris, a medieval Catholic cathedral in Paris, France. Incorrect Decomposition: A structural fire broke
out The fire occurred on 15 April 2019 The fire occurred just before 18:20 CEST The fire broke out in the roof space Notre-Dame de
Paris is a cathedral Notre-Dame de Paris is a medieval Catholic cathedral Notre-Dame de Paris is located in Paris, France Correct
Decomposition: A structural fire broke out The fire occurred on 15 April 2019 The fire occurred just before 18:20 CEST The fire
broke out in the roof space Notre-Dame de Paris is a cathedral Notre-Dame de Paris is a medieval Catholic cathedral Notre-Dame
de Paris is located in Paris Notre-Dame de Paris is located in France Reasoning: Original claim 7 had 2 pieces of information in
it: The location of Paris and The location of France. While this may seem intuitive that Paris is in France, from an evaluation
perspective, it is better to have two distinct claims to verify. See another brief example below:
Incorrect Decomposition: The event happened in Seoul, South Korea Correct Decomposition: The event happened in Seoul The event
happened in South Korea Reasoning: When considering the evaluation of where the event happened, it’s better to split the claim into
the two locations: Seoul and South Korea, so that you can evaluate against systems that say only South Korea (if it happened in
other locations in SK) or against systems that only state the city.
Hallucinated Decomposition This is the addition of a claim that isn’t supported by the sentence.
Input: The 2016 World Short Track Speed Skating Championships took place from 11 to 13 March 2016 in Seoul, South Korea. Incorrect
Decomposition: The event took place The event was from 11 to 13 March 2016 The event happened in Seoul The event happened in South
Korea The event was the 41st speed skating championship Correct Decomposition: The event took place The event was from 11 to 13
March 2016 The event happened in Seoul The event happened in South Korea Reasoning: 5 is factual, but it is not supported by the
sentence.
Ambiguity Only resolve ambiguity if the claims make it difficult to disambiguate between entities.
Input: The 2016 World Short Track Speed Skating Championships took place from 11 to 13 March 2016 in Seoul, South Korea. Correct
Decomposition: The event took place The event was from 11 to 13 March 2016 The event happened in Seoul The event happened in South
Korea Reasoning: Only one event in the sentence. No need to disambiguate.
Input: Due to Imran Khan’s criticism of Macron’s comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens
and deported 118 from the country. Incorrect Decomposition: They cancelled the visas of 183 Pakistani citizens. They deported 118
Pakistani citizens from the country. He criticized Macron’s comments on Islam Correct Decomposition: French authorities cancelled
the visas of 183 Pakistani citizens. French authorities deported 118 Pakistani citizens from the country. Imran Khan criticized
Macron’s comments on Islam Reasoning: In this scenario, it’s better to disambiguate the references to the named entities because
they could be Imran, Macron, or the French authorities.
Current notes / edge cases during test evaluation Go over 2022 Senate Election annotations.
Sentence: The National Tsunami Warning Center|itself located inside the quake zone, in Palmer, Alaska, northeast of Anchorage|issued
tsunami warnings for nearby coastal areas, including Cook Inlet and the Kenai Peninsula, but they were lifted shortly after. Claim
1: The National Tsunami Warning Center issued warnings Claim 2: The warnings were for nearby coastal areas Claim 3: The warnings
included Cook Inlet Claim 4: The warnings included the Kenai Peninsula Claim 5: The warnings were lifted shortly after issuance
Claim 6: Palmer is located in Alaska Claim 7: Palmer is northeast of Anchorage Claim 8: The National Tsunami Warning Center is
located in Palmer Claim 9: The National Tsunami Warning Center is inside the quake zone Claim 10: Cook Inlet is a coastal area Claim
11: The Kenai Peninsula is a coastal area
6,7,8 as claims related to the location of NTWC or Palmer. Perspective matters probably. I would may rewrite these to be all about
the NTWC location.
Sentence: Notre-Dame did not hold a Christmas Mass in 2019, for the first time since 1803. Claim 1: Notre-Dame did not hold a
Christmas Mass in 2019 Claim 2: Notre-Dame did not hold a Christmas Mass was in 1803 Claim 3: Notre-Dame held a Christmas Mass every
year between 1803 and 2019
Sentence: Investigators in 2020 believed the fire to have been "started by either a cigarette or a short circuit in the electrical
system". Claim 1: The investigators believed the fire was started Claim 2: The investigators identified two possible causes for the
fire Claim 3: One possible cause was a cigarette Claim 4: Another possible cause was a short circuit in the electrical system Claim
5: The investigation took place in 2020
New: There was an investigation into the cause of the fire The investigators identified two possible causes for the fire The fire
was possibly started by a cigarette The fire was possibly started by a short circuit in the electrical system
The investigation took place in 2020 (this might not be factual) How to incorporate the date? In 2020 investigators believed the
fire was started.
original: 5 [’The investigators believed the fire was started’, ’The investigators identified two possible causes for the fire’,
’One possible cause was a cigarette’, ’Another possible cause was a short circuit in the electrical system’, ’The investigation took
place in 2020’] 2958: 4 [’The investigators identified two possible causes for the fire’, ’A possible cause was a cigarette’, ’A
possible cause was a short circuit in the electrical system’, ’The investigation took place in 2020’] 2959: 6 [’The investigators
believed the fire was started’, ’The investigators identified two possible causes for the fire’, ’The fire was possibly started by a
cigarette’, ’The fire was possibly started by a short circuit.’, ’The investigation took place in 2020’, ’The possible short circuit
occurred in the electrical system’] 2960: 5 [’There was an investigation into the cause of the fire.’, ’Investigators identified
two possible causes for the fire’, ’One possible cause was a cigarette’, ’One possible cause was a short circuit in the electrical
system’, ’An investigation took place in 2020’]
Figure 6: Annotation Instructions for Claim Rewriting
19

Preprint. Under review.
Figure 7: The claim grounding protocol. This protocol has the video on the left hand side
and a sentence and its claims on the right. Each claim has 4 buttons which the annotator can
select for the modality (or none) that support the claim.
D Article Generation
In this section we present the prompt for article synthesis (Figure 12)
E Qualitative Differences Between Outputs
Qualitative Results for Model Variations In this section we present additional qualitative
examples from the different generation methods. We present these for the Notre-Dame fire
query against the reference article (Table 8, Table 9, Table 10).
Qualitative Results for Audio vs. Video We also present the qualitative differences
between video only and video+audio CAG results in Table 11, Table 12, and Table 13. In
these tables you can see that the articles produced using transcripts are shorter than the
articles based only on video content. Numerically, the average length of a video-only article
is 206.36 tokens and the average length of a article with transcript provided is 163.90 tokens.
F Additional Results
We report more statistics in Table 14 to show the varying performance across model sizes
and variations. We report LLaVA-Video-7B,72B and QwenVL2.5-3B,7B,72B as well as
Qwen2.5-32B,72B for article synthesis.
Per-event type results In Table 15, we breakdown the argument F1 scores for each Vide-
oLLM+ CAG combination. Here we see the highest F1 scores in the most commonly recog-
nizable events: elections and sports. These events are often professionally broadcast and the
entities that participate in these events are “high-resource” visual concepts. However, in
events like disasters and demonstrations, we see a decrease in F1, especially in exact match
as there are no longer high-resource entities to identify or heavily populated OCR content.
G Human Analysis
To provide an upper-bound on model performance, we recruit 3 fluent english speakers
to write 3 articles. These annotators receive the information request and the set of “oracle”
relevant videos and are instructed to write the article from this information. This human
20

Preprint. Under review.
This task involves watching a video and deciding whether the video supports particular claims. A video supports a claim if it can
be verified from the video’s visual, audio, or text content.
Visual text (denoted \Text" in the interface and instructions) support refers to on-screen text that supports the claim. This may
include (e.g.) text on a street sign, text scrolling across the screen in a news broadcast, text on someone’s clothing, subtitles
on the screen, etc.
Visual support refers to anything else on-screen besides text that supports the claim. This includes any action happening on-screen,
still images, or other graphics (e.g. a map shown in a weather report) or animations.
Audio support refers to any sounds (e.g. sirens, gunshots) or speech (e.g. from a newscaster or from the person filming) that
supports the claim.
Each claim may have only one of these types of support in a given clip, may have multiple types of support, or may not be supported
at all. It is your job to determine which type(s) of support there are for each claim.
Note: Sometimes the clips may contain audio (speech) or visual text in a language that you do not speak. For these instances, do
not try to annotate claims for the information in another language. Task Interface On the left side of the interface, you will see
a video clip.
You can adjust the playback speed by clicking on the three vertical dots at the bottom right of the clip:
You can adjust the video’s playback speed by clicking the three vertical dots on the bottom right:
Note: We do not suggest making the video full screen.
On the right side of the interface, you will see: The name of the event that the clip is about a sentence about that event a list
of claims derived from that sentence
Each claim comes with three buttons: Audio (for audio support), Visual (for visual support), Text (for text support), and Neither
(for none of the above):
For each claim, you should check all boxes corresponding to the types of support for the claim attested in the video:
If a claim has no support in the video, you should click only the Neither checkbox:
You are allowed to select any combination of the Audio, Visual, and Text checkboxes, but the interface will prevent you from
selecting the Neither checkbox in combination with any of the other three.
When all claims have at least one button checked, the HIT will become ready for submission: the SUBMIT button will change from gray
to blue and you then can submit the HIT:
Task Instructions Your job is to:
Watch the video clip For each claim, select all checkboxes corresponding to the type(s) of support it receives in the video (Audio,
Text, or Visual) or Neither if it has no support
Note that you are allowed|and encouraged!|to rewatch the video if necessary in order to assess the types of support it provides for
a particular claim. At least one checkbox must be clicked for each claim before you can submit the HIT. What does it mean for a
claim to be supported by visuals/audio/text? You can conceive of each claim as both posing a question and providing an answer to
that question.
For example, the claim Dominguez hit a home run can be understood to be: Implicitly asking the question: Did Dominguez hit a home
run? Answering this question affirmatively: Yes, he did.
If the audio (sounds, speech), visuals (action, images, graphics), and on-screen text provides the same answer to the question
as the claim|i.e. if the audio, visuals, or text provide evidence that the claim is true|then you should click the corresponding
checkbox.
Alternatively, the clip may not provide an answer to the claim’s question at all, or may even provide an answer that contradicts
the answer implicit in the claim. In these cases, you should select only the Neither checkbox.
Note: in many cases, it will be difficult to say with absolute certainty whether a given claim is supported or not by the audio,
visuals, or text. The standard for determining whether a claim has one of these types of support is not certainty, but rather high
confidence, given the clip’s contents. You will have to rely on your own judgment of what a normal person could confidently infer
about the truth of the claim, having watched the video yourself. (See World Knowledge Support)
Audio Support Examples Audio support may come from speech or other sounds in the video clip. Below are two examples of cases in
which a claim has audio support.
Example 1
Claim: Dominguez hit the first home run of his career. Video: https://www.youtube.com/watch?v=c6U4AnW4ohM Audio Evidence: The
broadcaster in the video announces the homerun that Dominguez hit.
Example 2
Claim: Emergency responders went to the scene Video: https://www.youtube.com/watch?v=rVKwa4ZqQAA Audio Evidence: You can hear the
sirens of the fire trucks in this video. Thus, audio support. Visual Support Examples Visual support can come from any type of
non-text visual content, including any action happening in the video clip, or still images or graphics that are shown. We include
two examples of visual support below.
Example 3
Claim: Dominguez hit a homerun Video: https://www.youtube.com/watch?v=c6U4AnW4ohM Explanation: In the video you can see Dominguez
hit the homerun because he swings, hits the ball, it goes into the stands.
Example 4
Claim: The event took place in France Video: (shows the image of the Eiffel Tower below)
Explanation: The Eiffel Tower is an iconic landmark in Paris, France. Even if the video doesn’t explicitly state that it is located
in France, you can very confidently infer that the location is France based on this knowledge that the Eiffel Tower is in France.
Example 5
Claim: The hurricane hit florida Video: (shows the graphic of the Hurricane’s trajectory below)
Explanation: Although the map doesn’t explicitly say \Florida" you can see from the map that the Hurricane’s path goes through
Miami (and thus hits Flordia) Text Support Examples Text support can come from any type of text visible on screen. This can be text
deployed on the screen in a news broadcast, street signs, posters in a protest, etc. Some examples of text support are shown below.
Example 6
Claim: The event took place on August 30th. Video: (includes the frame below) Explanation: The date is clearly displayed on the
bottom left of the screen (30 Aug)
Example 7
Claim: City of London police were deployed. Video: (includes the frame below) Explanation: It’s clear from the text on the officer’s
back in the frame below that the people shown are City of London police, who appear to be deployed.
Example 8
Claim: Mueller subpoenaed a former aide of Trump. Video: (includes the frame below) Explanation: The text shown in the \Breaking
News" banner at the bottom of the screen clearly supports the claim.
World Knowledge Support
Claim: Texas borders Mexico
Explanation: Here in this example, nothing on the map shows the names of Mexico or Texas, but using common knowledge, you can see
both the outline of Texas and Mexico. Thus,
Neither Support Examples Claim: Dominguez hit the last home run of his career Video: https://www.youtube.com/watch?v=c6U4AnW4ohM
Explanation: This is about his first ever home run. You cannot tell that his career has ended or that he hasn’t hit a homerun since
this game.
Figure 8: Instructions for the Claim Grounding Protocol.
21

Preprint. Under review.
LLaVA-Video VAST InternVideo2.5 QwenVL2.5 Reference
VideoLLM
The video begins with
a view of a historic
cathedral engulfed
in flames, with thick
smoke billowing into
the sky. The cathedral’s
spire is prominently
visible, and the
surrounding buildings
are also affected by
the fire. The scene is
dramatic, with the fire
raging intensely and the
smoke creating a dark
cloud over the area. The
video then transitions
to a closer view of the
cathedral’s spire, which
is now almost entirely
consumed by the fire.
The flames are bright
and fierce, illuminating
the intricate details of
the spire. The smoke
continues to rise,
obscuring parts of the
structure. The video
progresses to show the
spire collapsing under
the heat, with the flames
still burning fiercely.
The final frames focus
on the aftermath of
the collapse, with the
remains of the spire and
the surrounding area
still ablaze. The video
concludes with a wider
shot of the cathedral,
showing the extent of
the damage and the
ongoing fire.fire is burning on a
tall tower that has been
burningThe video captures
the catastrophic fire
engulfing the Notre
Dame Cathedral in
Paris. The iconic
structure is ablaze,
with flames and smoke
billowing from its spire
and roof. The fire’s
orange glow contrasts
with the grey sky, while
thick smoke rises into
the air. The surrounding
environment appears
calm, with no visible
people or firefighters
in the immediate
vicinity of the blaze.
The cathedral’s gothic
architecture is partially
visible, with its skeletal
framework exposed by
the fire. The scene is
somber, highlighting
the loss of a historical
landmark.The video depicts a
catastrophic fire engulf-
ing a historic cathedral,
likely Notre-Dame in
Paris, France. The
flames are intense and
widespread, with thick
smoke billowing into
the sky. The iconic spire
of the cathedral is visibly
on fire, with the struc-
ture glowing red from
the heat. As the fire
rages, the spire begins
to weaken and eventu-
ally collapses, sending a
cascade of debris down-
ward. The surround-
ing buildings and trees
are partially obscured
by the smoke, emphasiz-
ing the scale and sever-
ity of the blaze. The
scene is both dramatic
and tragic, capturing a
moment of significant
destruction to a beloved
landmark.A structural fire broke
out in the roof space
of the Notre-Dame
de Paris, a medieval
Catholic cathedral
in Paris, France. By
the time the fire was
extinguished, the
cathedral’s wooden
spire, called a fl ´eche,
collapsed, most of the
wooden roof had been
destroyed, and the
cathedral’s upper walls
were severely damaged.
Extensive damage to
the cathedral’s interior
was prevented by the
vaulted stone ceiling,
which largely contained
the burning roof as it
collapsed. However,
some works of art and
religious relics suffered
smoke damage and
some exterior art was
damaged or destroyed.
The cathedral’s altar
and three 13th-century
rose windows suffered
little or no damage.
There were emergency
workers present
during the fire. The
fire contaminated the
site and nearby areas
of Paris with toxic
dust. French president
Emmanuel Macron
set a deadline for
the restoration of the
cathedral.
Table 8: Qualitative Results for VideoLLMs on the Notre-Dame Fire query. The results of the
table are from video ID: 1117864206475218944. Bold text represents model hallucinations.
This is any information included in the article that is not in any of the videos.
22

Preprint. Under review.
LLaVA-Video VAST InternVideo2.5 QwenVL2.5 Reference
Article Generation
On April 15, 2019, a
devastating fire broke
out at Notre-Dame
Cathedral in Paris, one
of the world’s most
famous and historic
Gothic landmarks.
The blaze, which
began in the attic,
rapidly spread, causing
significant damage to
the cathedral’s wooden
frame and leading
to the collapse of its
spire. The fire raged
for several hours, with
thick plumes of smoke
and flames visible from
across the city. Despite
the extensive damage,
firefighters managed
to save the cathedral’s
main structure,
including the two bell
towers. The incident
drew widespread
international attention
and prompted pledges
of support for the
reconstruction efforts
from global leaders
and private donors.
President Emmanuel
Macron vowed that
the cathedral would
be restored, and initial
assessments indicated
that the restoration
process could take
years to complete. The
fire raised questions
about the safety
and maintenance of
historical sites and
the methods used to
protect them from such
disasters.A major fire engulfed a
prominent building in
a French city, drawing
a crowd of onlookers
who gathered to witness
the event. Flames and
thick smoke were visi-
ble from various parts of
the structure, including
its roof and upper levels.
The fire, which broke
out in the middle of the
building, also affected
a nearby church, caus-
ing significant damage.
Emergency services re-
sponded to the scene,
while local residents and
visitors stopped to ob-
serve and document the
incident, some taking
photographs and dis-
cussing the unfolding
situation. The cause of
the fire and the extent of
the damage remained
under investigation.A catastrophic fire en-
gulfed the Notre-Dame
Cathedral in Paris on
April 15, 2019 , causing
extensive damage to
the historic structure.
Flames and thick smoke
billowed from the
roof and spire, which
partially collapsed
during the blaze. The
fire, which began
in the attic, spread
rapidly, threatening the
integrity of the entire
building. Despite the
intense heat and the
structural damage, the
main facade and two
bell towers remained
standing. Emergency
responders, including
hundreds of firefighters,
worked tirelessly
throughout the night
to control the fire.
The event drew large
crowds of onlookers and
prompted a widespread
outpouring of grief and
solidarity, both locally
and internationally. The
cause of the fire was
initially suspected to
be linked to renovation
work being carried out
on the cathedral. The
aftermath revealed a
charred interior and
significant structural
damage, though
some elements, such
as the stained glass
windows, survived.
Restoration efforts were
immediately pledged by
the French government
and private donors.On April 15, 2019 , a
catastrophic fire broke
out at Notre-Dame
Cathedral in Paris,
France, causing
extensive damage to the
historic landmark. The
blaze, which erupted in
the afternoon, rapidly
spread through the
cathedral’s roof and
spire, leading to the
collapse of the iconic
spire and significant
portions of the roof.
Thick plumes of smoke
billowed into the sky,
visible from across
the city, as firefighters
worked tirelessly to
contain the inferno.
The fire drew a large
crowd of concerned
onlookers, including
Parisians and tourists,
who gathered to watch
the unfolding disaster,
many expressing their
shock and sorrow.
Despite the extensive
damage, the structural
integrity of the cathe-
dral’s main walls and
towers was preserved,
and key religious
artifacts, including
the cross and some
statues, remained intact.
President Emmanuel
Macron pledged to
rebuild the cathedral,
which is a UNESCO
World Heritage site and
a symbol of French
heritage and culture.
The cause of the fire
was initially suspected
to be linked to ongoing
renovation work, and
investigations were
launched to determine
the exact cause. The
event highlighted the
global significance of
Notre-Dame and the
profound impact of
its partial destruction
on both the local
community and
the international
community.A structural fire broke
out in the roof space
of the Notre-Dame
de Paris, a medieval
Catholic cathedral
in Paris, France. By
the time the fire was
extinguished, the
cathedral’s wooden
spire, called a fl ´eche,
collapsed, most of the
wooden roof had been
destroyed, and the
cathedral’s upper walls
were severely damaged.
Extensive damage to
the cathedral’s interior
was prevented by the
vaulted stone ceiling,
which largely contained
the burning roof as it
collapsed. However,
some works of art and
religious relics suffered
smoke damage and
some exterior art was
damaged or destroyed.
The cathedral’s altar
and three 13th-century
rose windows suffered
little or no damage.
There were emergency
workers present
during the fire. The
fire contaminated the
site and nearby areas
of Paris with toxic
dust. French president
Emmanuel Macron
set a deadline for
the restoration of the
cathedral.
Table 9: Qualitative Results for Article Generation on the Notre-Dame Fire query. Bold text
represents model hallucinations. This is any information included in the article that is not
in any of the videos.
23

Preprint. Under review.
LLaVA-Video VAST InternVideo2.5 QwenVL2.5 Reference
CAG
On April 15, 2019 , a
massive fire broke
out at Notre-Dame
Cathedral in Paris, one
of the most iconic and
historically significant
Gothic cathedrals in
the world . The blaze,
which began in the
attic of the cathedral,
rapidly spread, causing
the collapse of the
spire and significant
damage to the roof and
interior. Thick black
smoke billowed into
the sky, and the flames
were visible from across
the city. Despite the
intensity of the fire,
firefighters managed to
control and eventually
extinguish the blaze
by early morning. The
incident drew a large
crowd of onlookers,
many of whom gathered
along the Seine River
and near the cathedral,
some singing hymns in
response to the tragedy.
President Emmanuel
Macron pledged to
rebuild the cathedral,
and restoration
efforts were soon
underway, with the
goal of preserving and
restoring this cherished
landmark. The fire
caused extensive
damage, including the
loss of the spire and
a significant portion
of the roof, but the
main structure and
many valuable artifacts
were saved. The
event highlighted the
cathedral’s enduring
cultural and spiritual
significance to both the
French people and the
global community.CAG was not run on
VAST due to poor extrac-
tion capabilities.A catastrophic fire
engulfed the Notre
Dame Cathedral in
Paris on April 15, 2019 ,
causing significant
damage to the historic
structure. Flames and
thick smoke billowed
from the roof and
spire, which partially
collapsed during
the blaze. The fire,
which spread rapidly,
was visible from afar,
drawing large crowds
of onlookers who
watched in shock and
dismay. Emergency
responders, including
firefighters, were
quickly deployed to the
scene, working tirelessly
to control the inferno
and prevent further
damage. Despite their
efforts, the cathedral
suffered extensive
damage, with the roof
and spire being among
the most affected areas.
The incident occurred
during the evening,
and the surrounding
area was evacuated to
ensure public safety.
The aftermath revealed
a charred and debris-
filled interior, with the
iconic spire and much
of the roof destroyed.
The fire’s cause was
initially unknown,
but investigations
were launched to
determine the origin
and circumstances
of the disaster. The
event sparked a global
outpouring of support
and grief, with many
gathering to mourn the
loss of this cultural and
religious landmark.On April 15, 2019, a
catastrophic fire broke
out at Notre-Dame
Cathedral in Paris,
France, causing
extensive damage to
the historic landmark.
The fire, which began
in the attic of the
cathedral, rapidly
spread, engulfing the
roof and the iconic spire,
which collapsed in a
dramatic fall witnessed
by a shocked crowd
gathered outside. The
cathedral, a masterpiece
of Gothic architecture
dating back to the
12th century , suffered
significant structural
damage, with the roof
and spire completely
destroyed. Despite
the intense blaze, the
main stone structure,
including the two
bell towers, remained
standing. Firefighters
worked tirelessly
throughout the night to
contain the fire, using
advanced equipment
and techniques to
prevent further damage .
The event drew global
attention, with people
around the world
expressing sorrow
and solidarity . In
the aftermath, French
President Emmanuel
Macron pledged to
rebuild the cathedral,
and a fundraising cam-
paign was launched to
support the restoration
efforts. The fire raised
questions about the
safety and preservation
of historic landmarks
and highlighted the
cultural and spiritual
significance of Notre-
Dame to both France
and the international
community.A structural fire broke
out in the roof space
of the Notre-Dame
de Paris, a medieval
Catholic cathedral
in Paris, France. By
the time the fire was
extinguished, the
cathedral’s wooden
spire, called a fl ´eche,
collapsed, most of the
wooden roof had been
destroyed, and the
cathedral’s upper walls
were severely damaged.
Extensive damage to
the cathedral’s interior
was prevented by the
vaulted stone ceiling,
which largely contained
the burning roof as it
collapsed. However,
some works of art and
religious relics suffered
smoke damage and
some exterior art was
damaged or destroyed.
The cathedral’s altar
and three 13th-century
rose windows suffered
little or no damage.
There were emergency
workers present
during the fire. The
fire contaminated the
site and nearby areas
of Paris with toxic
dust. French president
Emmanuel Macron
set a deadline for
the restoration of the
cathedral.
Table 10: Qualitative Results for CAG on the Notre-Dame Fire query. Bold text represents
model hallucinations. This is any information included in the article that is not in any of
the videos.
24

Preprint. Under review.
Video Audio
On April 15, 2019, a catastrophic fire broke out at Notre-Dame Cathe-
dral in Paris, France, causing extensive damage to the historic land-
mark. The fire, which began in the attic of the cathedral, rapidly
spread, engulfing the roof and the iconic spire, which collapsed in a
dramatic fall witnessed by a shocked crowd gathered outside. The
cathedral, a masterpiece of Gothic architecture dating back to the
12th century, suffered significant structural damage, with the roof
and spire completely destroyed. Despite the intense blaze, the main
stone structure, including the two bell towers, remained standing.
Firefighters worked tirelessly throughout the night to contain the fire,
using advanced equipment and techniques to prevent further dam-
age. The event drew global attention, with people around the world
expressing sorrow and solidarity. In the aftermath, French President
Emmanuel Macron pledged to rebuild the cathedral, and a fundrais-
ing campaign was launched to support the restoration efforts. The
fire raised questions about the safety and preservation of historic
landmarks and highlighted the cultural and spiritual significance of
Notre-Dame to both France and the international community.On April 15, 2019, a catastrophic fire broke out at Notre-Dame Cathe-
dral in Paris, France, causing significant damage to the historic land-
mark. The blaze, which started in the attic, rapidly spread, engulfing
the roof and the iconic spire, which subsequently collapsed. Thick
plumes of smoke and intense flames were visible from afar, drawing
crowds of onlookers who watched in distress and solidarity. The
fire, which lasted for several hours, was eventually brought under
control by hundreds of firefighters. Despite the extensive damage,
the main structure of the cathedral, including the two bell towers,
survived. President Emmanuel Macron pledged to rebuild the cathe-
dral, and international support for the restoration efforts was swift
and substantial. The incident highlighted the cultural and historical
significance of Notre-Dame, one of the world’s most recognized and
cherished Gothic cathedrals.
Table 11: QwenVL+CAG with and without audio. Notre-Dame Fire
Video Audio
The 2022 Georgia Senate election was a closely contested race be-
tween incumbent Democrat Raphael Warnock and Republican chal-
lenger Herschel Walker. After the initial election on November 8,
2022, neither candidate secured the required 50% majority, leading to
a runoff election on December 6, 2022. In the initial vote, Warnock re-
ceived 49.2% (1,944,893 votes) and Walker received 48.7% (1,927,402
votes), with a difference of 17,491 votes. An independent candidate,
Chase Oliver, received 2.1% of the vote. The runoff election, which
determined the ultimate winner, saw a significant turnout, with over
2 million early votes cast, highlighting the high level of voter engage-
ment. The outcome of the runoff was crucial for the balance of power
in the U.S. Senate, as it would determine whether the Democrats
maintained their slim majority or if the Republicans gained control.
Ultimately, Raphael Warnock was declared the winner, securing his
re-election to the Senate.The 2022 Georgia Senate election was a closely contested race be-
tween incumbent Democratic Senator Raphael Warnock and Republi-
can challenger Herschel Walker. With neither candidate reaching the
50% threshold required to avoid a runoff, the election will proceed to
a runoff on December 6, 2022. Preliminary results showed Warnock
leading with 49.2% of the vote (1,944,893 votes) and Walker trailing
slightly with 48.7% (1,927,402 votes). Independent candidate Chase
Oliver received 2.1% of the vote (81,130 votes). The outcome of the
runoff is expected to have significant implications for control of the
U.S. Senate and for voting rights legislation, as well as for the broader
political landscape in Georgia and nationally.
Table 12: QwenVL+CAG with and without audio. Georgia Senate Election
Video Audio
Hurricane Irma was a powerful and destructive Category 5 hurricane
that caused catastrophic damage across the Caribbean and parts of
the southeastern United States in September 2017. Originating as
a tropical wave, Irma rapidly intensified into a Category 5 storm
with maximum sustained winds of 180 miles per hour, becoming the
most powerful hurricane ever recorded in the open Atlantic. The
storm made landfall in the Caribbean, particularly devastating the
islands of St. Martin, the British Virgin Islands, and Barbuda. It then
moved northwest, causing extensive damage in the Bahamas and
Cuba before making landfall in Southwest Florida on September 10.
The hurricane brought torrential rain, storm surges, and high winds,
resulting in widespread flooding, structural damage, and power out-
ages. In Florida, cities such as Miami Beach and Tampa experienced
severe flooding and wind damage. The storm’s impact was felt across
multiple states, leading to widespread evacuations and a significant
humanitarian response. Recovery efforts were complicated by the
extensive damage to infrastructure, including roads, bridges, and
power lines. The hurricane’s aftermath highlighted the resilience of
affected communities and the critical need for coordinated disaster
preparedness and response.Hurricane Irma was a powerful and destructive Category 5 hurricane
that caused widespread damage across the Caribbean and the south-
eastern United States in September 2017. Originating as a tropical
wave, Irma rapidly intensified into a Category 5 storm with maxi-
mum sustained winds reaching 180 miles per hour. The hurricane
made landfall in the Leeward Islands, causing catastrophic damage,
particularly on the islands of Barbuda and St. Martin. It then moved
through the Turks and Caicos, the Bahamas, and eventually struck
the Florida Keys and the southwestern coast of Florida. The storm
brought torrential rainfall, storm surges, and strong winds, leading
to extensive flooding and structural damage. In the aftermath, com-
munities faced significant challenges, including the loss of electricity,
water, and communication services. Recovery efforts were hampered
by the scale of the destruction, and many areas required substantial
aid and support to rebuild.
Table 13: QwenVL+CAG with and without audio. Hurricane Irma
25

Preprint. Under review.
Your task is to write a new Wikipedia article to exclude the claims not found in the video content. You will be given a set of
claims and the sentences they come from on the L.H.S. of the protocol and your job will be to rewrite the article / sentences such
that only the supported claims are presented in the article. You should try to diversify your writing from the Wikipedia if possible
without stepping too far away from the general \Wikipedia" style.
Figure 9: Instructions for Article Rewriting
I am trying to find information about EVENT QUERY. I will show you a video summary that might be related to the event. Based on the
current summary, can you think of a new query that might help me find more information about the event? Please write a new query
that you think will help me find more information about the event. DO NOT write anything except for the new query. If you think the
current summary is sufficient, you can say ’no new query.’ Otherwise, start your new query with ’Describe the video in detail and
focus on’ Here is the video summary:
Figure 10: Prompt For Qwen32B Distilled R1.
annotation is fundamentally different than our data collection process because instead of
grounding and ‘discriminating’ against an existing text, the annotators perform the same
task as CAG taking the videos and writing information from them. To create the human
generated articles, we provide annotators the relevant videos and instruct them to write
the lead of a Wikipedia article. An interesting note from this experiment is we notice
the annotators perform article writing similar to CAG , taking notes on each video before
aggregating them in an article.
In Table 16, we baseline human performance against the original Wikipedia article as the
predicted article and the best method ( CAG +QwenVL). We observe that the current metrics
for the task don’t accurately capture the quality of the human written article, which has no
hallucinations and is fully follows the constraint of only including video content. We show
these results qualitatively in Table 17, Table 18, and Table 19.
26

Preprint. Under review.
1. "Describe the video in detail and focus on the specific examples of CRISPR applications in medicine and agriculture
mentioned, as well as the ethical considerations discussed."
2. "Describe the video in detail and focus on the specific demands of the protesters and any notable incidents or interactions
during the convoy."
3. "Describe the video in detail and focus on the specific locations affected, the extent of damage caused, and any unique
geological features observed during the eruption."
4. "Describe the video in detail and focus on the eruption’s causes and effects in relation to the earthquake and tsunami."
5. "Describe the video in detail and focus on the specific mission details, such as the mission name, duration, objectives,
and any unique features of the spacecraft or crew."
Figure 11: RePrompts from R1 Provided to a VideoLLM
You are an experienced Wikipedia editor. You will be shown summaries of one or more videos related to the same event. Your task
is to write the lead section of a Wikipedia article based ONLY on the information provided in the video summary or summaries. The
lead section MUST match the quality, style, and tone of real Wikipedia articles. DO NOT write in the style of a news journalist.
DO NOT use any external sources or additional knowledge you have about the event. DO NOT output anything other than the Wikipedia
lead section. DO NOT refer to any of the videos explicitly in your output. DO NOT write anything except for the Wikipedia lead
section, even if the summaries are cut off. You MUST start your output with "<lead>". ONLY START YOUR REPORT WITH <lead>. DO NOT
WRITE ANYTHING EXCEPT FOR THE WIKIPEDIA ARTICLE.
Figure 12: Prompt For Article Synthesis.
Method VideoLLM R1 R2 RL BS Arg AS
CONCAT GENLLaVA-Video-7B 4.24 1.02 2.93 79.35 20.07 6.26
LLaVA-Video-72B 7.34 1.60 4.78 71.99 19.31 5.08
VAST 16.62 1.71 11.19 80.55 8.04 7.13
InternVideo2.5 11.85 2.32 7.90 80.78 18.33 9.53
QwenVL2.5-3B 9.60 2.36 6.27 80.80 20.22 7.73
QwenVL2.5-7B 9.82 2.62 6.25 81.27 21.28 9.04
QwenVL2.5-72B 11.34 3.13 7.06 81.60 23.72 8.01
CONCAT REPROMPTLLaVA-Video 6.36 1.51 4.22 80.03 21.34 5.50
InternVideo2.5 6.93 1.68 4.83 79.62 22.48 6.19
QwenVL2.5 8.38 2.71 5.49 81.94 22.89 7.17
CAG-0 +32BLLaVA-Video-7B 31.04 7.96 18.14 85.65 24.49 18.71
LLaVA-Video-72B 28.55 7.10 16.00 77.00 22.33 15.54
VAST 16.80 1.12 11.18 81.63 9.17 14.01
InternVideo2.5 28.11 6.06 16.94 84.80 22.21 16.59
QwenVL2.5-3B 30.78 7.87 18.19 85.39 24.32 16.37
QwenVL2.5-7B 31.64 7.88 18.13 85.59 24.35 16.31
QwenVL2.5-72B 32.59 8.86 18.89 85.81 26.60 15.71
CAG-0 +72BLLaVA-Video-7B 34.87 10.72 20.18 86.44 28.09 16.34
LLaVA-Video-72B 30.02 8.68 17.59 77.59 26.21 13.51
VAST 19.55 1.45 12.40 82.21 11.23 10.87
InternVideo2.5 32.54 8.98 19.47 85.82 25.65 17.58
QwenVL2.5-3B 32.92 10.00 19.37 85.95 27.44 16.27
QwenVL2.5-7B 34.01 10.05 19.47 86.24 26.97 16.72
QwenVL2.5-72B 33.58 10.15 19.15 86.18 28.97 15.63
CAGLLaVA-Video 33.38 10.05 19.44 84.55 28.26 15.23
InternVideo2.5 33.91 9.58 20.07 86.13 27.01 14.23
QwenVL 2.5 33.96 10.90 19.45 86.35 30.77 14.29
Table 14: Vision only results by method. AG: Article Generation. RePrompt results are only
the outputs from the follow-up questions and does not include the generic captions.
27

Preprint. Under review.
All Sport Disaster Election Social Demonst Discover Political
Model ED EX ED EX ED EX ED EX ED EX ED EX ED EX ED EX
LV 28 09 35 11 24 04 58 44 36 08 17 00 34 13 26 08
IV 27 06 40 17 24 03 49 25 42 00 13 00 30 09 23 04
QVL 31 11 50 32 29 06 62 50 38 08 18 00 31 10 25 05
Table 15: Argument F1 by MultiVENT-G Event Type for CAG Sport, Disaster, Election
Social: Social Event, Demonst: Demonstration, Discover: Discovery OR Launch, Political:
Political Development. ED: Edit Distance, EX: Exact Match
Method VideoLLM R1 BS Arg AS
CAG QwenVL 40.57 86.77 30.80 14.29
RAG+CAG QwenVL 23.84 77.85 20.65 9.01
Human Annotator 38.54 86.34 39.07 17.17
Wikipedia 64.53 90.47 62.82 23.53
Table 16: Comparison to human performance.
28

Preprint. Under review.
CAG Human Reference
The 2022 Georgia Senate election was a closely
contested race between incumbent Democrat
Raphael Warnock and Republican challenger
Herschel Walker. After the initial election on
November 8, 2022, neither candidate secured
the required 50% majority, leading to a runoff
election on December 6, 2022. In the initial
vote, Warnock received 49.2% (1,944,893 votes)
and Walker received 48.7% (1,927,402 votes),
with a difference of 17,491 votes. An indepen-
dent candidate, Chase Oliver, received 2.1%
of the vote. The runoff election, which deter-
mined the ultimate winner, saw a significant
turnout, with over 2 million early votes cast,
highlighting the high level of voter engage-
ment. The outcome of the runoff was crucial
for the balance of power in the U.S. Senate, as
it would determine whether the Democrats
maintained their slim majority or if the Re-
publicans gained control. Ultimately, Raphael
Warnock was declared the winner, securing
his re-election to the Senate.On December 6th, 2022, ABC news declared
Raphael Warnock the winner of the Georgia
Senate Race. Warnock, the incumbent Demo-
crat, defeated his opponent, republican Her-
shel Walker, in a runoff election by a little
more than 1% of the vote. Warnock had ini-
tially won election in 2020, when he, along
with President Joe Biden and Senator John
Ossoff, led a surprise sweep of the presiden-
tial and senate elections in the historically
conservative state. A month prior to the
runoff, Warnock also held a slight lead over
Walker on election day. At that point, with
99% of the votes counted, Warnock led by
35,429 votes, 49.4% to 48.5%. Despite this,
in Georgia a runoff is triggered if no can-
didate wins at least 50% of the vote in the
general election. With Libertarian candidate
Chase Oliver pulling roughly 2% of the vote,
MSNBC and CNN both reported that neither
Warnock nor Walker were able to reach this
threshold in November. This result was gen-
erally expected, as polling averages from the
weekend prior to the initial general election
showed an extremely tight race, from any-
where between a 0.8% margin from MSNBC
to just a 0.1% margin from FiveThirtyEight.
Many battleground senate races were decided
relatively early on Election day, with North
Carolina, Ohio, and Florida all being called
for Republicans, while Pennsylvania and New
Hampshire were called for Democrats. How-
ever, after these initial results, both Georgia
and Wisconsin, where Ron Johnson held a
slight lead over Mandela Barnes, were too
close to call, while Nevada and Arizona also
took longer for a victor to be declared. Inter-
estingly, in contrast to the Georgia senate race,
the Georgia gubernatorial election was de-
cided without a runoff, with Republican Brian
Kemp winning re-election in his rematch with
Democrat Stacey Abrams, indicating a large
number of split-ticket voters. This came de-
spite Abrams’ long-term investments into an
activist-driven campaign.
In the lead up to the runoff election, Walker
heavily emphasized Joe Biden’s low approval
ratings and the economy, while Warnock fo-
cused on Walker’s lack of knowledge about
issues and allegations of violence. The incum-
bent senator’s strategy focused on turning out
the democratic base in cities, while the chal-
lenger focused on more conservative rural ar-
eas; both campaigns targeted moderate edu-
cated voters in the suburbs around Atlanta. In
addition, Georgia Democrats won a key court
battle to allow for early voting in runoff elec-
tions. This decision challenged SB 202, a 2021
state law that limited early voting periods and
restricted weekend voting.The 2022 United States senate election in Geor-
gia was held on November 8, 2022 to elect a
member of the U.S. senate to represent the
state of Georgia. The election was part of the
2022 midterm elections and featured incum-
bent democratic senator Raphael Warnock
and Republican former football player Her-
schel Walker. Warnock, who had one a short-
ened seat in a special election held in 2020-21
defeated Walker in a runoff election and was
re-elected for his first full term in office.
Under Georgia’s two-round election system, a
runoff election occurs if no candidate receives
more than 50% of the vote. In the November
8 election, Warnock received 49.4% of the vote
and Walker received 48.5%, triggering the De-
cember 6th runoff election.
Warnock’s victory in the runoff was a state-
wide victory for Democrats. Warnock’s win
gave Democrats a net gain of one seat in the
2022 midterms. The election was also one
of many very close Senate elections during
the 2022 midterms, with both Nevada and
Wisconsin also having tight races.
Table 17: Georgia Senate Election
29

Preprint. Under review.
CAG Human Reference
A significant earthquake with a magnitude
of 7.0 struck Anchorage, Alaska, causing
widespread disruption and damage. The Na-
tional Weather Service confirmed the event
and reassured the public that there was no
tsunami threat. Footage from the aftermath
showed damaged roads, disrupted infrastruc-
ture, and scenes of chaos, including a super-
market where items had fallen off shelves.
The earthquake prompted school evacuations
and the temporary closure of the Anchorage
airport, as the air traffic control tower was
evacuated. While the full extent of the dam-
age remains unclear, ongoing assessments and
recovery efforts are underway to address the
impact of the quake.A magnitude 7.0 earthquake hit southcentral
Alaska on Friday, November 30, 2018. The
earthquake hit approximately 7 miles north of
Anchorage at 8:29 AM. Anchorage is Alaska’s
largest city, with half of the state’s population
living in the region. Following the earthquake,
the National Tsunami Warning Center issued
and then canceled a tsunami warning for the
coastal zones of southern Alaska. Governor
Bill Walker issued a disaster declaration. In
the three hours after the earthquake, the US
Geological Survey recorded at least 30 after-
shocks with magnitudes ranging from 2.7 to
5.7; hundreds of aftershocks were eventually
recorded. During the earthquake and its af-
tershocks, students and office workers shel-
tered in place under their desks, and some
buildings were evacuated, including the air
traffic control tower at the Anchorage airport.
The earthquake caused major infrastructure
damage across the city, according to the An-
chorage police department. Impacts include
damage to roadways and water mains, visible
cracks in buildings, stores and homes in disar-
ray, and one house fire caused by a damaged
gas pipe. No deaths have been reported.On November 30, 2018 at 8:29 a.m. AKST
(17:29 UTC), a magnitude 7.1 earthquake hit
Anchorage in South Central Alaska. The earth-
quake’s epicenter was 10 miles north of An-
chorage and occurred at a depth of 29 miles.
It was followed by a magnitude 5.7 aftershock.
The National Tsunami Warning Center issued
tsunami warnings for nearby costal areas, in-
cluding Cook Inlet. The warnings were lifted
shortly after being issued.
Table 18: Anchorage Earthquake
CAG Human Reference
Hurricane Irma was a powerful and de-
structive Category 5 hurricane that caused
widespread damage across the Caribbean and
the southeastern United States in September
2017. Originating as a tropical wave, Irma
rapidly intensified into a Category 5 storm
with maximum sustained winds reaching 180
miles per hour. The hurricane made landfall
in the Leeward Islands, causing catastrophic
damage, particularly on the islands of Bar-
buda and St. Martin. It then moved through
the Turks and Caicos, the Bahamas, and even-
tually struck the Florida Keys and the south-
western coast of Florida. The storm brought
torrential rainfall, storm surges, and strong
winds, leading to extensive flooding and struc-
tural damage. In the aftermath, communities
faced significant challenges, including the loss
of electricity, water, and communication ser-
vices. Recovery efforts were hampered by the
scale of the destruction, and many areas re-
quired substantial aid and support to rebuild.Hurricane Irma was a record-setting Hurri-
cane that struck the south Atlantic region
in early September, 2017, and that caused
widespread damage. At the time, Irma
was the most powerful Hurricane recorded
in the open Atlantic region and was the
second-strongest ever to hit Cuba. Begin-
ning as a tropic wave, Irma struck many
places—including the Lesser Antilles, Do-
minica, Guadeloupe, Barbuda, Antigua, St.
Martin, Puerto Rico, Haiti, the Dominican Re-
public, and Cuba—as a Category 5 storm—as
well as Florida as a Category 3 storm. At
its peak intensity, Irma reached more than
180 mph winds and caused severe flooding
and damage to buildings and infrastructure
throughout the region.Hurricane Irma was an extremely powerful
Cape Verde hurricane that occurred in early
September 2017. Irma was a Category 5 hur-
ricane and the most powerful hurricane on
record in the open Atlantic region outside of
the Caribbean Sea and Gulf of Mexico. Irma
was also the strongest tropical cyclone by
wind speed worldwide in 2017. Irma caused
widespread and catastrophic damage through-
out its path, and was particularly severe in the
northeastern Caribbean.
Irma developed from a tropical wave near the
Cape Verde Islands. Irma then became a Cate-
gory 3 hurricane on the Saffir-Simpson wind
scale before resuming intensifying on Septem-
ber 4 and becoming a Category 5 hurricane by
early September 5. Irma’s intensity peaked on
September 6 with 1-minute sustained winds at
180mph and a minimum pressure of 914 hPa.
Before making landfall in Cuba, Irma weak-
ened to a Category 4 hurricane, but regained
its Category 5 status before hitting Cuba.
Irma hit both Caribbean islands and the conti-
nental United States. Irma caused catastrophic
damage in Barbuda, Saint Barth ´elemy, Saint
Martin, Anguilla, and the Virgin Islands as a
Category 5 hurricane. Irma also made land-
fall in Anguilla, Barbados, Cuba, French West
Indies, Haiti, Puerto Rico, and the Dutch side
of Sint Maarten. After crossing the Straits of
Florida, Irma made landfall in Cudjoe Key on
September 10 making landfall in Florida be-
fore then making landfall at Marco Island.
Table 19: Hurricane Irma
30
# GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination

**Authors**: Xinxi Chen, Tianyang Chen, Lijia Hong

**Published**: 2025-09-30 02:09:07

**PDF URL**: [http://arxiv.org/pdf/2509.25669v1](http://arxiv.org/pdf/2509.25669v1)

## Abstract
We propose a method to improve Visual Question Answering (VQA) with
Retrieval-Augmented Generation (RAG) by introducing text-grounded object
localization. Rather than retrieving information based on the entire image, our
approach enables the model to generate a bounding box around the object most
relevant to the question, allowing for targeted image cropping and focused
retrieval. This reduces background noise, improves alignment between visual and
textual cues, and helps mitigate hallucinations. Our RAG method enhances
context-aware VQA responses increased the accuracy from 22.19% to 25.64%, with
an absolute increase of 3.45 percentage points, compared to the baseline
Llama-3.2-Vision-11B agent. We also proposed a de-hallucination method based on
question type which can effectively reduce the hallucination rate from 65.79%
to 13.88% and improves the truthfulness score.

## Full Text


<!-- PDF content starts -->

GroundSight: Augmenting Vision-Language Models with
Grounding Information and De-hallucination
Xinxi Chen∗
Independent Researcher
xc336@cornell.eduTianyang Chen∗
Independent Researcher
tiche@seas.upenn.eduLijia Hong
Independent Researcher
lhong@alumni.ust.hk
Abstract
We propose a method to improve Visual Question Answering (VQA)
with Retrieval-Augmented Generation (RAG) by introducing text-
grounded object localization. Rather than retrieving information
based on the entire image, our approach enables the model to gener-
ate a bounding box around the object most relevant to the question,
allowing for targeted image cropping and focused retrieval. This
reduces background noise, improves alignment between visual and
textual cues, and helps mitigate hallucinations. Our RAG method
enhances context-aware VQA responses increased the accuracy
from 22.19% to 25.64%, with an absolute increase of 3.45 percentage
points, compared to the baseline Llama-3.2-Vision-11B agent. We
also proposed a de-hallucination method based on question type
which can effectively reduce the hallucination rate from 65.79% to
13.88% and improves the truthfulness score.
CCS Concepts
•Computing methodologies →Knowledge representation
and reasoning;Artificial intelligence.
Keywords
Object Detection·Image Grounding·Multi-modal learning·Large
Vision-Language Models
ACM Reference Format:
Xinxi Chen, Tianyang Chen, and Lijia Hong. 2025. GroundSight: Aug-
menting Vision-Language Models with Grounding Information and De-
hallucination. InProceedings of In Proceedings of 2025 KDD Cup Workship for
Multimodal Retrieval Augmented Generation at the 31th ACM SGKDD Con-
ference on Knowledge Discovery and Data Mining (KDDCup’ 25) (KDDCup’
25).ACM, New York, NY, USA, 12 pages.
1 Introduction
Visual Question Answering (VQA) [ 1] sits at the intersection of
computer vision and natural language processing, requiring sys-
tems to reason over both images and text to produce meaningful
answers. Recent advances in Vision-Language Models (VLMs) [ 5]
have significantly enhanced the ability of machines to jointly un-
derstand visual and linguistic content, enabling more accurate and
∗This work builds on a prior project at Stanford University, with sincere thanks for
the support.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
KDDCup’ 25, Toronto, ON, Canada
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.context-aware interpretations of complex visual scenes. However,
these models are inherently limited by the knowledge encoded in
their training data. To address this, Retrieval-Augmented Genera-
tion (RAG) [ 9] introduces an external knowledge retrieval step that
grounds model outputs in up-to-date or domain-specific informa-
tion, bridging the gap between perception and world knowledge.
The combination of VLMs with RAG is particularly important for
VQA, as it allows systems not only to interpret what they see,
but also to reason with additional contextual or factual informa-
tion—ultimately leading to more robust, informed, and trustworthy
responses. In this paper we investigate methods that can improve
VQA performance.
A core challenge in combining VQA with RAG lies in identifying
and retrieving external knowledge that is simultaneously relevant
to both the textual query and the visual content. Unlike pure text-
based RAG, where the query alone guides document selection, mul-
timodal RAG must interpret features from an image—objects, scenes,
spatial relationships—and align them with the user’s question to
construct a precise retrieval request. For example, when give the
question "How much does this cost?" and an image of users holding
one object in hand, while the background contains multiple other
objects, the VLM must be able to understand which object is the
question referring to in order to perform effective informational re-
trieval. If the retrieval system focuses too narrowly on one modality
(e.g., only on keywords in the text), it risks ignoring critical visual
cues; conversely, overemphasizing visual attributes may surface
facts that have little bearing on the question’s intent. Moreover, the
retrieved facts must be filtered and integrated in a way that respects
the visual context—misaligned or tangential information can lead
to confident but incorrect answers. Balancing these two streams
of information to surface grounded, image-aware knowledge is
therefore a delicate orchestration that remains an open research
frontier in multimodal language understanding.
Furthermore, dehallucinating VLMs is critical to ensuring the
reliability and safety of their outputs, especially in applications
like VQA where users may rely on responses for decision-making.
VLMs, like their language-only counterparts, are prone to hallu-
cination—generating plausible but factually incorrect or visually
inconsistent answers—when they lack sufficient understanding or
context. This issue is exacerbated when VLMs must reason about
complex scenes or incorporate external knowledge, as they may
confidently assert false claims not supported by the image or re-
trieved content. For the example provided by the previous section, a
better answer is "I don’t know" rather than responding with wrong
price for the wrong object.arXiv:2509.25669v1  [cs.AI]  30 Sep 2025

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
2 Related Work
Improving VQA has been a dynamic area of research, with recent
efforts focusing on enhancing the gorunding of visual and textual
information. GLIP [ 12] (Grounded Language-Image Pre-training)
is a unified vision-language architecture designed to bridge object
detection and phrase grounding by reformulating object detection
as a vision-language matching task. The core idea is to align image
regions (bounding boxes) with phrases from a natural language
prompt, enabling the model to detect and ground objects based
on open-vocabulary textual queries rather than a fixed set of class
labels. GLIP handles both object detection and phrase grounding
with a single architecture allowing the model to localize objects of
interest that can best answer a given question prompt. Similarly,
Grounding DINO [ 15] tightly integrates language and vision to
enable detection and localization of arbitrary objects specified by
natural language prompts, rather than being limited to a fixed set
of classes. Its architecture features dual backbones for image and
text, a feature enhancer module that deeply fuses visual and linguis-
tic features via cross-attention, a language-guided query selection
mechanism that dynamically chooses relevant image regions based
on the prompt, and a cross-modality decoder that refines predic-
tions by alternating attention between image and text features.
Grounding DINO is designed to handle referring expressions in
text prompts—including pronouns like "it"—as part of its referring
expression comprehension (REC) capability. The model can local-
ize and identify specific objects or regions within an image based
on a given textual description, which may include coreferences
such as "it" if the context in the prompt is clear enough for the
model to resolve what "it" refers to, which is crucial for targeted re-
trieval augmentation for VQA. The "Chain-of-Spot" [ 6] introduces
a novel and efficient approach to enhancing the visual reasoning
capabilities of large vision-language models (LVLMs) through an
interactive reasoning process. What sets Chain-of-Spot apart is its
focus on dynamically identifying and attending to key regions of
interest (ROI) within an image that are most relevant to the posed
question or instruction, rather than processing the entire image at
a fixed (often low) resolution. This is achieved by prompting the
model to first localize the critical region in response to a query,
cropping or zooming in on that region, and then generating the
answer based on both the original and the focused image. This
interactive, multi-step procedure allows the model to access more
detailed and multi-granularity visual features without increasing
the computational cost associated with higher-resolution process-
ing.
Visual grounding aims to localize the image region referred to
by a given language expression. Methodologies in this field have
evolved from multi-modal fusion over a fixed set of detected objects
to direct bounding-box prediction with open-vocabulary capabili-
ties. Early approaches integrated object-level visual features into
textual representations to enhance generic VQA performance (e.g.,
via object-text fusion strategies [ 3]). Later work introduced more
structured two-stage pipelines: for instance, a "Locate Then Gen-
erate" framework first predicts the relevant scene-text region and
then generates the answer from the cropped area [ 22]. Recent ef-
forts extend this paradigm to the video domain, where groundingscene-text temporally across frames proves beneficial for text-based
video QA [21].

GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada
3 Proposed Method: GroundSight
We aim to improve Visual Question Answering (VQA) in real-world
scenarios, where ambiguous questions and cluttered backgrounds
make it difficult for vision-language models to focus on the correct
visual content. To address this, we propose GroundSight, a retrieval-
augmented system that localizes the object of interest and mitigates
hallucination.
To address the problem discussed previously, we explored ex-
tending the capabilities of the existing VLM by introducing two
key functionalities: (1) localizing the region of interest (ROI), and
(2) de-hallucinating uncertain answers. These enhancements are
aimed at improving the robustness and accuracy of our RAG agent
(Please refer to Figure 1a for the overall system design).
Our main system design follows a Retrieval-Augmented Genera-
tion (RAG) framework composed of the following components:
•Vision-Language Model (VLM):We experimented with
several models including BLIP, QWen, and LLaMA 3.2, ul-
timately selecting LLaMA 3.2 for its performance and com-
patibility.
•Region of Interest Proposer:A module responsible for
identifying the object or region most relevant to answering
the question.
•Image-Based Information Retriever:Performs web-based
or local database search using cropped image regions as
queries.
In addition, we fine-tune the VLM to reduce hallucination in un-
certain scenarios. As discussed in the Results section, hallucination
remains a significant bottleneck to overall performance.
While each module—the VLM, ROI proposer, and retriever—can
be individually improved, our current focus is on enhancing the
Region of Interest Proposer. Preliminary results show that models
often generate irrelevant answers based on distracting background
elements. Improving object localization is therefore a promising
direction. Future work may explore optimization of the VLM and
retriever components.
3.1 Localizing Region of Interest
We reformulate visual understanding as a text-guided object local-
ization problem [ 13]. Given a question, the model is tasked with
identifying the most relevant object by outputting a bounding box
around it. This localized ROI is then used to crop the image, re-
ducing background noise and enabling a more focused retrieval
process. The retrieved content is passed back to the model to gener-
ate the final answer (see Figure 2). We explore different approaches
to make our RAG agent region-aware:
3.1.1 Using a Pretrained Localizer.Although integrating a trained
localization component keeps the system self-contained, it incurs
training costs and complexity. As an alternative, we also experi-
mented with incorporating a pretrained localizer, such as Ground-
ing DINO [ 14], to identify regions of interest prior to retrieval. The
cropped region is then used to perform visual search, and the re-
sulting information is fed back into the VLM to answer the original
question.
3.1.2 Training Vision-Language Models for Grounding via IoU Opti-
mization.To validate the hypothesis that a large vision-languagemodel (LVLM) can be trained to perform spatial grounding—specifically,
by accurately predicting bounding boxes from natural language
prompts—we developed a four-stage fine-tuning pipeline based on
the BLIP-2 architecture [ 10]. This setup progressively unlocks more
of the model’s capacity for localization, culminating in a best train-
ing Intersection over Union (IoU) of0.4552and a final evaluation
IoU of0.4473.
IoU=|Prediction∩GroundTruth|
|Prediction∪GroundTruth|
Training Data.We use the RefCOCOg dataset [ 16], which con-
sists of images paired with natural language referring expressions
and annotated bounding boxes. For this experiment, we curated a
subset of 5000 samples for training and 500 for final evaluation. This
proof-of-concept study emphasizes clarity and reproducibility over
dataset scale. We believe that expanding the dataset and unfreezing
more vision layers will yield substantial performance gains.
Training Procedure.Our method consists of four training stages:
•Stage 1:Train only the BBox prediction head, keeping BLIP-
2 frozen.
•Stage 2:Unfreeze the Q-Former for co-training with the
BBox head.
•Stage 3:Unfreeze the final 6 layers of the vision encoder.
•Stage 4:Use advanced optimization with 8 vision layers
unfrozen and a larger dataset.
Each stage incrementally improved the model’s grounding ca-
pability, confirming the feasibility of IoU-guided fine-tuning for
vision-language models. (See Figure 3).
Grounding DINO vs. LVLMs.While models such as Grounding
DINO demonstrate excellent grounding performance, they fall un-
der the category ofopen-vocabulary object detection. These models
leverage pre-trained vision encoders and text embeddings to match
visual regions to language descriptions, often using contrastive
objectives. However, they arenot large-scale vision-language models
in the generative or instruction-following sense.
In contrast, our goal is to study how toenable grounding ca-
pabilities directly within LVLMs, such as BLIP-2, by explicitly
optimizing for spatial localization tasks (e.g., IoU prediction). This
approach integrates grounding into the model’s multimodal rea-
soning process, laying the foundation for fully unified, end-to-end
instruction-following systems with spatial understanding.
Comparison with Other LVLMs.DeepSeek-VL 2 [ 18] has shown
competitive results on grounding benchmarks. It introduces a <
|grounding|> token and grounding-aware mode that help align vi-
sual and textual inputs for localization. In contrast, general-purpose
LVLMs like LlaMA3.2 might not have been pretrained on bounding
box data, making them less suitable for grounding tasks without
sufficient training data. According to a recent survey, there has
been a notable trend toward developing Grounding Multimodal
Large Language Models (GMLLMs) since 2023 [20].
IoU Progression Across Training Stages
These results validate the feasibility of training LVLMs such as
BLIP-2 for spatial grounding. With additional scale in data, model
size, and fine-tuning depth, we expect future iterations to cross the
0.5 IoU threshold required for high-quality object localization. Due

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
(a) Overview of the System Design
 (b) Example of the System Architecture in Use
Figure 1: System design and an example architecture
Figure 2: Overview of the Retrieval-Augmented Generation (RAG) Pipeline.
to compute limitation and the fact that Llama 3.2 was not explic-
itly pre-trained on grounding tasks for bounding box generation,
we decided to use Grounding DINO for object localization, while
finetuning models for localization is left for future work.3.2 Finetuning to Reduce Hallucination
We also fine tuned the model to say "I don’t know" in some cases
to reduce hallucination. For example, for the following question in
Table 2, Llama-3.2-Vison-11B outputs an incorrect answer. We can
train the model to answer "I don’t know" for "who" type question,
since this typically requires external knowledge. We investigate the

GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada
Figure 3: Architecture of the Proof-of-Concept VLM for Grounding
Training Stage Best IoU
Stage 1: BBox Head Only 0.2585
Stage 2: + Q-Former 0.3103
Stage 3: + Vision Layers (6) 0.3718
Stage 4: Advanced Optimization (8 Vision Layers) 0.4552
Final Evaluation (500 samples) 0.4473
Table 1: IoU scores at different stages of training for BLIP-2
on RefCOCOg.
questions based on the type shown in Figure 5b, and then fine tune
the model to only answer the questions with high confidence.
4 Results
4.1 Dataset and Task Definition
We analyzed a VQA dataset collected from Meta RayBan smart
glasses [ 2], which contains both single-turn and multi-turn image-
question-answer pairs across 14 diverse domains, including shop-
ping, food, and science. We primarily evaluate on the v2 version of
this dataset, which contains 1938 single-turn questions. For compar-
ison, we also report results on the earlier v1 version, which includes
1548 questions, during our ablation studies.
Used for evaluation of our work, this dataset presents a signif-
icant challenge due to its varying image quality and ambiguous
questions, requiring models to extract the most relevant informa-
tion from noisy inputs. For instance, consider an image showing
several cars near buildings. Performing image retrieval directly on
the full image would likely yield results focused on street scenes or
buildings, since large background elements tend to dominate the
image. Therefore, if the question is “How many passengers can the
red car seat?”, a retrieval system unaware of the object of interest
will fail to provide accurate information (see Figure 4).The dataset features a variety of question types, such as color,
counting, location, object recognition, reasoning, and yes/no queries.
According to the distribution of question types in the v2 dataset (see
Figure 5a), object recognition questions constitute the largest por-
tion. This skew towards object recognition highlights the need for
models to possess strong, text-grounded visual understanding, and
sometimes be able to localize the object of interest among multiple
other objects in the background based on the input text.
This dataset presents a realistic and challenging benchmark for
vision-language systems, making it a strong candidate for evalu-
ating model performance. In the following subsection, we present
baseline results using several state-of-the-art VLMs.
(a) Input image and region of
interest
(b) Retrieved image doesn’t
match the object in question
Figure 4: Example image retrieval without text grounding
4.1.1 Image Search API.We utilize a prebuilt image search API
provided by Meta [ 8], which contains a database of 900K images

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
Table 2: Evaluation record for Interaction ID00a48
ID Query Agent Response Ground Truth Result
00a48 Who invented this
kind of tape?The tape measure was invented by the French tailor
Pierre-Frédéric Guillaume, who patented the first practi-
cal retractable tape measure in 1829 (prototype in 1821).James Chesterm...Incorrect
(a) Question-type count
(b) Accuracy by question type
Figure 5: Analysis by question type
and associated metadata. This API converts images into embed-
dings using the CLIP model [ 17], retrieves similar images based
on cosine similarity, and returns them with relevant information.
We observed that specifying the region of interest (ROI) is crucial
for retrieval-augmented generation (RAG), since the retrieved re-
sults differ significantly depending on the ROI. When unrelated
regions are used, the retrieved information does not aid the model
in answering the question and may even promote hallucinations.
For example, as shown in Listing 2, using the full image from the
previous section results in completely irrelevant retrieval when the
user’s actual question concerns the car parked on the street rather
than the street itself.
Listing 1: Example Cropped Region Search Result
'entities': [{
'entity name':'Toyota Prius v',
'entity attributes': {
'alternative names': ['Prius alpha','Prius+'],
'production start':'May 2011',
'production end':'March 2021',
'body style':'compact MPV'}
}]
Listing 2: Example Image Search Without Knowing Region
Of Interest
'entities': [{
'entity_name':'Ocean Grove, New Jersey',
'entity_attributes': {
'official_name':'Ocean Grove, New Jersey',
'settlement_type':'[Census-designated place]',
'image_skyline':'Ocean_Grove_Welcome_Sign.jpg',
'imagesize':'250x200px',
'image_caption':'Ocean Grove welcome sign'
}
}]
4.2 Baseline Model Performance
We have collected baseline performance for three popular open
source VLMs: BLIP (Bootstrapping Language-Image Pre-training)
[11] , Llama 3.2 (11B) [ 7] and Qwen 2.5 (3B) [ 4], on this dataset
without any information retrieval implemented.
4.3 End To End Evaluation Method
We use GPT-4o-mini as a judge, guided by three rules: (i) a predic-
tion is correct when it contains all key information in the ground
truth, (ii) paraphrasing is acceptable if the meaning is unchanged,
and (iii) a prediction is incorrect if it introduces errors or omits
essentials. For each question we assign a scalar score—Perfect
(1.0),Acceptable(0.5, minor non-harmful flaws),Missing(0.0,
refusal/“I don’t know”), orIncorrect(–1.0, wrong or irrelevant). A
system’sTruthfulness Scoreis the mean of these values across the
evaluation set, yielding a range of –1 (all wrong) to 1 (all perfect).
Listing 5 shows the full prompt for the judge.
4.4 Results on Vanilla Vision-Language Models
Based on the preliminary results of three unmodified models pre-
trained on 1,548 questions from Table 4, the blip-vqa-base model
performs poorly in the real-world data set with only accuracy of
3. 49%. The Qwen-VL-2.5-3B and Llama-3.2-Vision-11B perform
relatively better with 18.09% and 26.23% accuracy. The Hallucina-
tion rate is relatively higher in Llama-3.2-Vision-11B compared to
Qwen-VL-2.5-3B. As a result, Llama-3.2-Vision-11B has the highest
truthfulness score, and Qwen-VL-2.5-3B ranks the second. This will
be used as a baseline in future experiments.
4.5 Ablation Study
4.5.1 Baseline on V1 Dataset.The CRAG MM dataset has two
versions, where the first version v1 includes 1548 single turn ques-
tions, and the second version v2 includes 1938 single turn ques-
tions. We evaluated our baseline on v1 dataset with three different

GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada
pre-trained models: BLIP, Qwen-VL-2.5-3B and Llama-3.2-Vision-
11B. Since Llama-3.2-Vision-11B performs relatively great, we use
Llama-3.2-Vision-11B for our customized solutions and evaluate
only Llama-3.2-Vision-11B on v2 dataset as baseline for our cus-
tomized solutions.
Based on the results of three pretrained unmodified models on
v1 1548 questions from Table 4, the blip-vqa-base model performs
poorly on the real world data set with only 3.49% accuracy. The
Qwen-VL-2.5-3B and Llama-3.2-Vision-11B perform relatively bet-
ter with 18.09% and 26.23% accuracy. The Hallucination rate is rela-
tively higher in Llama-3.2-Vision-11B compared to Qwen-VL-2.5-3B.
As a result, Llama-3.2-Vision-11B has the highest truthfulness score,
and Qwen-VL-2.5-3B ranks the second.
4.5.2 Localization.We compare two approaches to incorporating
region-of-interest localization into our retrieval-augmented gener-
ation (RAG) system. The first uses Grounding DINO, a pretrained
object localizer, to detect and crop the most relevant image region
for answering the question. The second follows a Chain-of-Spot-
style prompting strategy, where the model is first asked to describe
the region of interest before retrieving and answering. While both
approaches aim to improve retrieval quality by focusing on relevant
image content, they differ in whether localization is model-guided
(via prompting) or external (via a pretrained detector).
We evaluated our RAG agent on the new version of the dataset
using two different prompting strategies. Both experiments use
Grounding DINO (grounding-dino-tiny, 172M) for bounding box
extraction instead of our internally trained bounding box heads
due to limited compute budget.
Table 6 compares the performance of the baseline model (without
RAG) against the two prompting approaches. In the first strategy,
we crop the input image using the detected bounding box, perform
a retrieval based on the cropped region, and use the retrieved infor-
mation to answer the question. In the second strategy, we follow a
Chain-of-Spot-style (CoS) approach [ 6]: the model first summarizes
the region of interest, then we crop the image using Grounding
DINO, and finally feed both the summary and search results back
into the model to answer the original question.
Surprisingly, using Grounding DINO alone in the first experi-
ment led to a nearly 5% drop in accuracy, with a hallucination rate
comparable to the baseline. In contrast, the Chain-of-Spot-style
prompting improved accuracy beyond the baseline, but introduced
a higher hallucination rate (5%), which ultimately lowered the over-
all score.
Upon analyzing outputs from different stages of our RAG agent,
we observed the following:
•When image-based information retrieval returns completely
irrelevant content, it can mislead the model into producing
incorrect answers (Table 8). This limits the effectiveness of
RAG and results in performance comparable to the baseline.
In contrast, Chain-of-Spot prompting mitigates this issue
by first asking the model to describe what it sees before in-
corporating retrieved information. We hypothesize that this
approach encourages the model to rely more confidently on
its own visual understanding than on potentially misleading
external sources, as we have seen the model output the exact
same answer as the summary, ignoring RAG results.•However, Chain-of-Spot prompting can also increase hal-
lucination. That is, once the model identifies the region of
interest, it may become overly confident in its predictions.
As shown in Table 9, this can sometimes override correct
prior knowledge, leading to confidently incorrect answers.
This behavior contributes to a higher hallucination rate and
lowers the overall accuracy. This is evident given the low
missing rate and the model is less likely to output "I don’t
know".
•Computation wise, the gounding DINO is extremely effi-
cient to generate bounding boxes, while the Chain-of-Spot
prompting requires the model to look at the image twice,
leading to almost double inference time. On A100 with 80
GB VRAM using a batch size of 36, one full evaluation on
v2 dataset takes 1 hour, and on average each batch takes
1.8 seconds. This is consistent with other test-time scaling
technique like Chain-of-Thought [19].
4.5.3 De-hallucination with Fine-tuning Model.While accuracy im-
proved with more advanced prompting strategies, hallucination
rates also increased. To address this, we fine-tuned the model to
condition its responses on the question type. This enabled it to
abstain from answering difficult reasoning questions when uncer-
tain, effectively generating "I don’t know" responses instead of
hallucinations.
As shown in Table 5, without using RAG, this approach reduced
the hallucination rate from 65.79% to 19.14%, and improved the
truthfulness score from −0.4360to−0.0738. This significant im-
provement demonstrates the impact of hallucination control on
downstream performance.
Since our primary focus is to study the effectiveness of the predic-
tion with grounding information, we did not overengineer tricks on
reducing the hallucination rate to purely improving the truthfulness
score. That’s been said, there is clear path to decrease hallucination
further with this methodology.
4.5.4 De-hallucination with Image Search Threshold.We also ap-
plied an additional image search threshold in the final GroundSight
solution. The final solution has a threshold of 0.75. Setting a CLIP
similarity threshold of 0.75 filters irrelevant data while preserving
semantically grounded matches.
4.5.5 End to end performance.After analyzing localization and de-
hallucination strategies independently, we integrated them into our
full GoundSight agent and compare against individual techniques
in Table 7. We observed that they act complementarily: localization
enhances answer relevance by grounding the model’s attention,
Chain-of-Spot encourages the model to be more confident in it’s
own knowledge to avoid being misled by irrelevant RAG results,
while de-hallucination ensures cautiousness when the model is
uncertain. This balance improves overall performance by reducing
confidently wrong answers and encouraging selective abstention
on harder questions. Our agent achieved the highest truthfulness
score of -0.049.

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
In summary, grounding and de-hallucination each contribute to
improved VQA performance, and their combination in Ground-
Sight offers a practical path toward more accurate and trustworthy
answers.
5 Conclusion
In summary, this work demonstrates that incorporating text-grounded
object localization into retrieval-augmented VQA systems enables
models to produce more accurate and context-aware answers by
focusing on the most relevant image regions for each question. By
leveraging Chain-of-Spot-style prompting, our RAG agent is able to
effectively combine retrieved web content with the model’s own vi-
sual understanding. Experiments on challenging real-world datasets
show that this localization-based strategy improves accuracy over
baseline methods, though at the cost of increased hallucination.
Interestingly, our de-hallucinated model, which more frequently
responds with “I don’t know,” achieves the highest overall score—
highlighting a valuable real-world insight: providing an incorrect
answer can be more detrimental than admitting uncertainty. This
underscores a practical challenge of using RAG agents in safety-
critical applications.
Future work could explore combining Chain-of-Spot prompting
with a fine-tuned, de-hallucinated VLM, enabling the agent to retain
low hallucination rates while still leveraging external information
to enhance accuracy.
A Additional Results

GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada
1SYSTEM PROMPT: "You are a precise and cautious assistant that truthfully answers user questions about the provided image
augmented with online search information. Only answer if you are confident and have the necessary knowledge. If you are
not absolutely certain about the answer, reply with exactly:'I don't know', without any further explanation. Do not use
any other phrases like'I don't have details','It depends', or'I don't have enough information'. Your response must be
concise and must not exceed 75 words."}
2USER PROMPT: "Context that may be relevant to the objects in question:\n"
3+ search_results + "Answer this question: " + query}
Listing 3: Agent prompt
1SYSTEM PROMPT: "You are a helpful assistant that can summarize a region of interest of the image based on user's question. The
summary should be concise and only contain a simple description that must not exceed 10 words. The summary must not
answer the question."
2USER PROMPT: "Provide a concise summary for object of interest that can answer the following question:'" + query + "'"}
Listing 4: Chain Of Spot prompt
1"You are an expert evaluator for question answering systems. "
2"Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
3"Rules:\n"
4"1. The prediction is correct if it captures all the key information from the ground truth.\n"
5"2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
6"3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
7"Output a JSON object with a single field'accuracy'whose value is true or false."
Listing 5: LLM judge prompt
Table 3: Overall evaluation metrics for Llama-3.2-Vision-11B on V1 Dataset (Baseline vs. De-hallucination)
Model Total conv. Total
turnsExact acc.
(%)Accuracy
(%)Missing
rate (%)Hallucination
rate (%)Truthfulness
score
Llama-3.2-Vision-11B (Baseline) 1548 1548 0.84 26.23 13.24 60.53−0.3430
Llama-3.2-Vision-11B (De-hallucination) 1548 1548 0.90 12.60 69.64 17.76−0.0517
Table 4: Overall evaluation metrics for three vision–language models without RAG
Model Total conv. Total
turnsExact acc.
(%)Accuracy
(%)Missing
rate (%)Hallucination
rate (%)Truthfulness
score
blip-vqa-base 1548 1548 0.00 3.49 0.00 96.51−0.9302
Qwen-VL-2.5-3B 1548 1548 0.78 18.09 33.72 48.19−0.3010
Llama-3.2-Vision-11B 1548 1548 0.84 26.23 13.24 60.53−0.3430
Table 5: Overall evaluation metrics for Llama-3.2-Vision-11B on V2 Dataset (Baseline vs. De-hallucination)
Model Total conv. Total
turnsExact acc.
(%)Accuracy
(%)Missing
rate (%)Hallucination
rate (%)Truthfulness
score
Llama-3.2-Vision-11B (Baseline) 1938 1938 0.46 22.19 12.02 65.79−0.4360
Llama-3.2-Vision-11B (De-hallucination) 1938 1938 0.67 11.76 69.09 19.14−0.0738

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
Table 6: Overall evaluation metrics for RAG implementation
Model Total conv. Total
turnsExact acc.
(%)Accuracy
(%)Missing
rate (%)Hallucination
rate (%)Truthfulness
score
Llama-3.2-Vision-11B (Baseline no RAG) 1938 1938 0.46 22.19 12.02 65.79−0.4360
GDINO 1938 1938 0.15 17.75 16.87 65.38−0.4763
GDINO + chain of spot 1938 1938 0.83 25.64 3.77 70.59−0.4494
Table 7: Overall evaluation metrics for RAG implementation
Model Total conv. Total
turnsExact acc.
(%)Accuracy
(%)Missing
rate (%)Hallucination
rate (%)Truthfulness
score
Llama-3.2-Vision-11B (Baseline no RAG) 1938 1938 0.46 22.19 12.02 65.79−0.4360
GroundSight
+GDINO 1938 1938 0.15 17.75 16.87 65.38−0.4763
GroundSight
+GDINO + CoS 1938 1938 0.83 25.64 3.77 70.59−0.4494
GroundSight
Llama-3.2-Vision-11B (De-hallucination) 1938 1938 0.67 11.76 69.09 19.14−0.0738
GroundSight
(De-hallucination + GDINO + CoS
w/ Image Search threshold = 0.75) 1938 1938 0.36 8.98 77.14 13.88−0.0490
Table 8: Model output when search result is not useful
Question Image Image Crop + Search Chain-of-Spot Baseline
What is the typical filling
of this Chinese steamed
bun?Original:
Cropped:
Search result:Blood soup.
Response (incorrect):The typi-
cal filling of this Chinese steamed
bun is not blood soup, as the image
shows a steamed bun with a brown
filling, not a soup.Search result:Blood soup.
ROI summary:The typical filling
of this Chinese steamed bun is pork.
Response (correct):The typical
filling of this Chinese steamed bun
is pork.Response (correct):The typical
filling of this Chinese steamed bun
is pork.
Groundtruth:The typical
filling is pork.

GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada
Table 9: Model output when Chain-of-Spot makes overconfident predictions
Question Image Image Crop + Search Chain-of-Spot Baseline
How old was this artist
when he started hosting
his own show on NBC?Original:
Cropped:
Search result:Levi Strauss & Co.
is an American clothing company.
Final answer:I don’t know.Search result:Levi Strauss & Co.
is an American clothing company.
ROI summary:The object of inter-
est is a mural of Nat King Cole, an
American singer and musician.
Response (incorrect):Nat King
Cole was 31 years old when he
started hosting his own show on
NBC, "The Nat King Cole Show," in
1956.Response (correct):Nat King Cole
was born on March 17, 1919, and
he started hosting his own show on
NBC in 1956. Therefore, he was 37
years old when he started hosting
his own show on NBC.
Groundtruth:Nat King
Cole was 37 years old.

KDDCup’ 25, August 03–07, 2025, Toronto, ON, Canada Chen et al.
Acknowledgments
We sincerely thank Sabri Eyuboglu for the valuable insights on
the topic of vision-language models. We thank Qi Tang for sharing
knowledge about infrastructure. We are grateful to the Department
of Computer Science at Stanford University for support, including
computing resources, and to the teaching staff—including Fei-Fei
Li, Ehsan Adeli, Ranjay Krishna and others—for clarifying several
technical points; this work builds on a prior project conducted
there.
References
[1]Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence
Zitnick, Dhruv Batra, and Devi Parikh. 2015. VQA: Visual Question Answering.
arXiv:1505.00468 [cs.CV] https://arxiv.org/abs/1505.00468 arXiv:1505.00468.
[2]AIcrowd. 2025. Meta CRAG-MM Challenge 2025. https://www.aicrowd.com/
challenges/meta-crag-mm-challenge-2025. Accessed: 2025-04-25.
[3]Chris Alberti, Jeffrey Ling, Michael Collins, and David Reitter. 2019. Fusion of
Detected Objects in Text for Visual Question Answering. InProceedings of the
2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for
Computational Linguistics, Hong Kong, China, 2131–2140. doi:10.18653/v1/D19-
1219
[4]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang,
Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang,
Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo
Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu,
and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923 [cs.CV]
https://arxiv.org/abs/2502.13923
[5]F. Bordes et al .2024. An Introduction to Vision-Language Modeling. (2024).
arXiv:2405.17247 [cs.CV] https://arxiv.org/abs/2405.17247 Provides a comprehen-
sive introduction to VLMs, their architectures, training methods, and evaluation
strategies.
[6]Yuheng Dong et al .2024. Chain-of-Spot: Interactive Reasoning Improves Large
Vision-Language Models.arXiv preprint arXiv:2403.12966(2024). https://arxiv.
org/abs/2403.12966
[7]Aaron Grattafiori, Abhimanyu Dubey, and Abhinav Jauhri et al. 2024. The Llama
3 Herd of Models. arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
[8]Jyotish. 2025.Search API — Meta Comprehensive RAG Benchmark Starter
Kit. https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-
benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit/-
/blob/main/docs/search_api.md Accessed 16 May 2025.
[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems (NeurIPS). arXiv:2005.11401 [cs.CL] https://arxiv.org/abs/2005.11401
[10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping
language-image pre-training with frozen image encoders and large language
models. InInternational conference on machine learning. PMLR, 19730–19742.
[11] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping
language-image pre-training for unified vision-language understanding and
generation. InInternational conference on machine learning. PMLR, 12888–12900.
[12] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan
Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-
Wei Chang, and Jianfeng Gao. 2022. Grounded Language-Image Pre-Training. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 10965–10975.
[13] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan
Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-
Wei Chang, and Jianfeng Gao. 2022. Grounded Language-Image Pre-training. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 11032–11042. https://arxiv.org/abs/2112.03857
[14] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing
Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al .2024. Grounding dino: Marry-
ing dino with grounded pre-training for open-set object detection. InEuropean
Conference on Computer Vision. Springer, 38–55.
[15] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing
Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. 2023.
Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set
Object Detection.arXiv preprint arXiv:2303.05499(2023). https://arxiv.org/abs/
2303.05499[16] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille,
and Kevin Murphy. 2016. Generation and comprehension of unambiguous object
descriptions. InProceedings of the IEEE conference on computer vision and pattern
recognition. 11–20.
[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning. PmLR, 8748–8763.
[18] DeepSeek-VL Team. 2024. DeepSeek-VL 2: Towards Strong Generalist Vision-
Language Models with Mixture of Experts.arXiv preprint arXiv:2412.10302(2024).
https://arxiv.org/abs/2412.10302
[19] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia,
Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits
Reasoning in Large Language Models.arXiv preprint arXiv:2201.11903(2022).
https://arxiv.org/abs/2201.11903
[20] Linhui Xiao, Xiaoshan Yang, Xiangyuan Lan, Yaowei Wang, and Changsheng
Xu. 2024. Towards visual grounding: A survey.arXiv preprint arXiv:2412.20206
(2024).
[21] Sheng Zhou, Junbin Xiao, Xun Yang, Peipei Song, Dan Guo, Angela Yao, Meng
Wang, and Tat-Seng Chua. 2025. Scene-Text Grounding for Text-Based Video
Question Answering. arXiv:2409.14319 [cs.CV] https://arxiv.org/abs/2409.14319
[22] Yongxin Zhu, Zhen Liu, Yukang Liang, Xin Li, Hao Liu, Changcun Bao, and Linli
Xu. 2023. Locate Then Generate: Bridging Vision and Language with Bounding
Box for Scene-Text VQA. arXiv:2304.01603 [cs.CV] https://arxiv.org/abs/2304.
01603
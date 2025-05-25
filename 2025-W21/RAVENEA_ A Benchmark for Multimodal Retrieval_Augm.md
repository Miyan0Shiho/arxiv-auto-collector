# RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding

**Authors**: Jiaang Li, Yifei Yuan, Wenyan Li, Mohammad Aliannejadi, Daniel Hershcovich, Anders Søgaard, Ivan Vulić, Wenxuan Zhang, Paul Pu Liang, Yang Deng, Serge Belongie

**Published**: 2025-05-20 14:57:16

**PDF URL**: [http://arxiv.org/pdf/2505.14462v1](http://arxiv.org/pdf/2505.14462v1)

## Abstract
As vision-language models (VLMs) become increasingly integrated into daily
life, the need for accurate visual culture understanding is becoming critical.
Yet, these models frequently fall short in interpreting cultural nuances
effectively. Prior work has demonstrated the effectiveness of
retrieval-augmented generation (RAG) in enhancing cultural understanding in
text-only settings, while its application in multimodal scenarios remains
underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented
Visual culturE uNdErstAnding), a new benchmark designed to advance visual
culture understanding through retrieval, focusing on two tasks: culture-focused
visual question answering (cVQA) and culture-informed image captioning (cIC).
RAVENEA extends existing datasets by integrating over 10,000 Wikipedia
documents curated and ranked by human annotators. With RAVENEA, we train and
evaluate seven multimodal retrievers for each image query, and measure the
downstream impact of retrieval-augmented inputs across fourteen
state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented
with culture-aware retrieval, outperform their non-augmented counterparts (by
at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the
value of retrieval-augmented methods and culturally inclusive benchmarks for
multimodal understanding.

## Full Text


<!-- PDF content starts -->

arXiv:2505.14462v1  [cs.CV]  20 May 2025RAVENEA : A Benchmark for Multimodal
Retrieval-Augmented Visual Culture Understanding
Jiaang Li1†∗Yifei Yuan1,2†Wenyan Li1Mohammad Aliannejadi3
Daniel Hershcovich1Anders Søgaard1Ivan Vuli ´c4
Wenxuan Zhang5Paul Pu Liang6Yang Deng7‡Serge Belongie1‡
1University of Copenhagen2ETH Zürich3University of Amsterdam
4University of Cambridge5Singapore University of Technology and Design
6Massachusetts Institute of Technology7Singapore Management University
Abstract
As vision-language models (VLMs) become increasingly integrated into daily
life, the need for accurate visual culture understanding is becoming critical. Yet,
these models frequently fall short in interpreting cultural nuances effectively. Prior
work has demonstrated the effectiveness of retrieval-augmented generation (RAG)
in enhancing cultural understanding in text-only settings, while its application
in multimodal scenarios remains underexplored. To bridge this gap, we intro-
duce RAVENEA (Retrieval- Augmented Visual cultur EuNdErstAnding), a new
benchmark designed to advance visual culture understanding through retrieval,
focusing on two tasks: culture-focused visual question answering (cVQA) and
culture-informed image captioning (cIC). RAVENEA extends existing datasets by
integrating over 10,000 Wikipedia documents curated and ranked by human anno-
tators. With RAVENEA , we train and evaluate seven multimodal retrievers for each
image query, and measure the downstream impact of retrieval-augmented inputs
across fourteen state-of-the-art VLMs. Our results show that lightweight VLMs,
when augmented with culture-aware retrieval, outperform their non-augmented
counterparts (by at least 3.2%absolute on cVQA and 6.2%absolute on cIC).
This highlights the value of retrieval-augmented methods and culturally inclusive
benchmarks for multimodal understanding.
Website https://jiaangli.github.io/RAVENEA/
Code https://github.com/yfyuan01/RAVENEA
Data https://huggingface.co/datasets/jaagli/ravenea
1 Introduction
Vision-language models (VLMs) are increasingly deployed in real-world applications, from education
to assistive technologies [ 1,2,3,4], where understanding not only visual content but also the
surrounding cultural context is crucial. Despite achieving impressive performance on general tasks [ 5,
6,7,8], VLMs often struggle to capture cultural nuances, such as traditions, symbols, and region-
specific practices that require external, culturally grounded knowledge [ 9,10,11,12]. For example,
as shown in Figure 1, a VLM may incorrectly identify the season of a festival scene as ‘Autumn’,
overlooking that the image depicts Kyoto’s Gion Festival, which occurs in July and corresponds to
∗Project Lead.
†Equal contribution.
‡Principal senior advisor.

‘Summer’. A promising approach to address this limitation is the integration of external knowledge
through retrieval-augmented generation (RAG) [ 13], which has shown success in improving cultural
awareness in language models [ 14,15]. However, prior work in the culture domain has predominantly
been confined to text-only settings. Meanwhile, existing culture-related multimodal datasets primarily
focus on evaluating VLM outputs on culturally oriented tasks, with limited emphasis on the integration
of external cultural knowledge. As a result, the potential of RAG to improve multimodal cultural
understanding remains underexplored.
: The Mid-Autumn Festival is a harvest festival celebrated in Chinese culture. Held in mid-autumn, …                   : The Gion Festival is one of the largest festival in Japan, taking place annually during the month of Julyin Kyoto…     Question: In which season is the event most possibly held?A.SpringB.SummerC.AutumnD.Winter
B. Summer C.Autumn
❌
✅
✅
❌
Answer W/ cultural RAGAnswer W/O cultural RAGRetrieved Wiki Set
Figure 1: Effectiveness of culture-aware
RAG. Given a culturally grounded visual
question, VLMs enhanced with culture-aware
RAG—retrieving relevant Wikipedia docu-
ments—generate more accurate answers than
their non-RAG counterparts. (see Section 5.2).To bridge the gap, we introduce RAVE-
NEA (Retrieval- Augmented Visual cultur E
uNdErstAnding), a manually curated dataset
designed to evaluate cultural understanding
in VLMs with retrieval support. We con-
struct RAVENEA based on two existing datasets:
CVQA [ 16], which includes culturally relevant
visual questions and corresponding answers, and
CCUB [ 17], offering culturally contextualized
captions to foster inclusivity in text-to-image
generation4. For each instance drawn from the
source datasets, we append a set of Wikipedia
documents that have been human-ranked based
on their cultural relevance to the associated im-
age. This curation effort, designed to ensure
broad cultural representation, contains data re-
lated to eight countries and spans eleven diverse
categories, comprising more than 1,800 images
and10,000 human-ranked documents. RAVE-
NEA thus provides a retrieval-augmented bench-
mark for evaluating cultural sensitivity in multimodal retrievers, and further allows for assessing
how well VLMs integrate and make use of retrieved cultural context. Specifically, we focus on
two culturally grounded tasks: (i) culture-focused visual question answering (cVQA) and (ii)
culture-informed image captioning (cIC) . We select these two tasks for their centrality in assessing
cultural understanding in VLMs—question answering tests context-aware reasoning, while captioning
evaluates generation sensitivity to cultural nuances.
With RAVENEA , we first train and evaluate seven multimodal retrievers that use both visual and textual
inputs to retrieve Wikipedia documents for a given query image based on their cultural relevance.
Then, we evaluate a diverse set of state-of-the-art (SOTA) VLMs, including GPT-4.1 [ 18], LLaV A-
OneVision-7B [ 19], Pixtral [ 20], Phi-4 Multimodal [ 21], the Gemma3 family [ 22], the Qwen2.5-VL
family [ 23], the InternVL3 family [ 24], and the Deepseek-VL2 family [ 25], each with and without
multimodal retrieval, to assess the impact of retrieval augmentation on cultural understanding. Our
dataset provides a testbed for assessing the cultural relevance capabilities of multimodal retrievers
and the effectiveness of VLMs in consuming and using such retrieved cultural context.
Our contributions and key findings include:
•RAVENEA benchmark : We introduce RAVENEA , the first benchmark aimed at evaluating VLMs
and multimodal retrieval in leveraging external knowledge for visual culture understanding. It
comprises a diverse, large-scale collection of human-curated, culturally related documents linked
to images from eight countries across eleven categories, enabling evaluation on two tasks: culture-
focused visual question answering and culture-informed image captioning. (Section 3)
•Cultural grounding annotations enhance multi-modal retrieval : We evaluate seven retrievers
that integrate visual and textual cues to retrieve culturally relevant documents. We find fine-tuning
retrievers on culture-targeted annotations leads to marked gains in retrieval accuracy, highlighting
the value of explicit cultural supervision. (Section 6.3)
•Benefits of culture-aware retrieval : Culture-aware retrieval boosts task performance across VLMs,
with lightweight models showing the greatest improvement. This suggests that such retrieval can
seamlessly integrate into downstream VLM tasks, enhancing their performance. (Section 5.2)
4We reuse the cultural captions from CCUB as ground-truth references for the inverse task, image-to-text
generation, specifically for culture-aware image captioning.
2

Gemma3-4BGemma3-27BLLaVA-Onevision-7BQwen2.5VL-3BQwen2.5VL-7BQwen2.5VL-72BInternVL3-2BInternVL3-8BInternVL3-38BDeepSeek-VL2-TinyDeepSeek-VL2Phi4-MultimodalPixtral-12BGPT-4.1505560657075808590
Gemma3-4BGemma3-27BLLaVA-Onevision-7BQwen2.5VL-3BQwen2.5VL-7BQwen2.5VL-72BInternVL3-2BInternVL3-8BInternVL3-38BDeepSeek-VL2-TinyDeepSeek-VL2Phi4-MultimodalPixtral-12BGPT-4.11020304050607080W/O RAGW/ RAG (CaCLIP)cVQA Accuracy
cIC Region Score
 cVQAcICIndonesiaIndiaMexicoSpainChinaNigeriaRussiaIndiaKoreaNigeriaMexicoChina
Cuisine
History
Architecture
Art
Daily Life
Companies
Transportation
Sports & Recreation
History
Cuisine
Architecture
Transportation
Art
Daily Life
Sports & Recreation
Companies
Architecture
History
Cuisine
Daily Life
Companies
Sports & Recreation
Transportation
Art
History
Architecture
Art
Sports & Recreation
Cuisine
Companies
Transportation
Daily Life
History
Architecture
Cuisine
Art
Sports & Recreation
Daily Life
Transportation
Companies
Cuisine
History
Architecture
Daily Life
Transportation
Art
Sports & Recreation
Companies
History
Architecture
Cuisine
Art
Companies
Transportation
Daily Life
Art
Architecture
Daily Life
Cuisine
Nature
Religion
Tools
Architecture
Art
Cuisine
Religion
Daily Life
Nature
Art
Architecture
Daily Life
Cuisine
Religion
Nature
Tools
Architecture
Art
Daily Life
Cuisine
Religion
Tools
Nature
Architecture
Cuisine
Art
Daily Life
Religion
Tools
Q: What is the meaning of those horses?A.Dignity  B. Courage C. Beauty   D. Strength
High RelevanceLowRelevance
…Caption: Palacio de BellasArtes cultural center building with many people walking around it in Mexico city.High RelevanceLowRelevance
…Figure 2: RAVENEA : A Multimodal Retrieval- Augmented Visual cultur EuNdErstAnding dataset.
Left: Examples of cVQA and cIC tasks. Middle : Geographic and categorical distribution of cultural
references. Right : Performance comparison of 14 VLMs, evaluated with and without integration of
our culture-aware retriever. Here, CaCLIP="culture-aware CLIP-L/14@224px ".
•Cross-cultural variation: Evaluation across eight countries reveals that VLMs exhibit distinct
cultural preferences, with each model favoring different regional contexts—suggesting model-
specific cultural biases. (Section 6.2)
2 Related Work
Retrieval augmentation for cultural understanding. Retrieval augmentation has demonstrated
significant efficacy for culture-related NLP tasks [ 26,27,28]. Prior work uses sources like the World
Values Survey for cultural question answering [ 14,29], or retrieves web and knowledge base content
to enhance cultural contextualization [ 15]. In multimodal settings, however, cultural retrieval remains
underexplored. While some approaches retrieve culture-relevant images to fine-tune VLMs [ 10], they
require additional training. In contrast, our method introduces a plug-and-play retrieval system that
enriches cultural grounding at inference time without modifying the base model.
Vision-language culture datasets. Several recent studies examine cultural understanding in VLMs
through multicultural VQA [ 30,9], cuisine recognition [ 31,32], and concept-based image re-
trieval [ 33]. Others benchmark cultural entity recognition using Wikipedia-based prompts [ 34]
or curate culture-specific image sets for value-based tasks [ 29]. While these efforts highlight current
VLM limitations, they often rely on static, manually curated datasets. A natural extension to address
these deficiencies is through culture-relevant multimodal retrieval—an area lacking dedicated datasets.
Our work specifically aims to fill this critical research gap.
3 R AVENEA Dataset
We construct our dataset by building culturally relevant document lists for each image in two
existing culture-grounded datasets: CVQA [ 16], a widely used dataset for culture-focused VQA, and
CCUB [ 17], a dataset designed to mitigate cultural bias in text-to-image generation with culturally
contextualized captions. To ensure broad geographic and cultural coverage, we curate a diverse subset
comprising images from seven countries in CVQA: China, Nigeria, Russia, Spain, Mexico, India, and
Indonesia, and all five countries in CCUB: China, Korea, India, Mexico, and Nigeria (see Figure 2).
After that, we separate the dataset construction process into three critical stages shown in Figure 3.
3.1 Dataset Construction
Data collection. The data collection process consists of two main steps: (i) culture-related caption-
ingand(ii) document retrieval . For culture-related captioning , we generate culturally grounded
3

Noise 
FilterConsensus 
ResolutionManual 
InspectionData Combination
 1.Culture -related Captioning
GPT-4oCaption: The image 
depicts the coat of 
arms of Nigeria ,...
2.Document Retrieval
BM25CaptionAnnotation Questions YES NO
Same Country?
Same Category?
Explicitly Mentioned?
Caption
Data Collection Relevance Annotation Quality Control
Postprocessing ToolsFigure 3: RAVENEA construction pipeline. Left : A two-stage retrieval process to match each
image with relevant documents. Middle : Decomposition of cultural relevance into three interpretable
dimensions to improve human annotation. Right : Postprocessing methods for quality control.
captions for each image to facilitate more effective attachment of relevant documents. Since the
CVQA lacks captions and the CCUB provides only brief descriptions, we employ GPT-4o to generate
richer, culturally informative captions (see the prompt example in Table 10). For document retrieval ,
we first conduct a coarse filtering using the generated cultural captions as queries for a BM25 [ 35]
retriever to extract the semantically relevant documents from a large-scale corpus comprising over six
million English Wikipedia documents5. To mitigate the impact of inaccurate captions and ensure
precise document relevance, we then perform human annotation on the retrieved documents.
Relevance annotation. Based on the initial BM25 retrieval results, we refine the cultural relevance
label of retrieved documents via human annotation. For each image-caption pair, annotators are
presented with the top 10 Wikipedia documents retrieved by BM25. They are asked to assess whether
each document provides meaningful background or contextual information that is relevant to the
culture described in the caption or the image (see Appendix K). Specifically, we decomposed cultural
relevance into three interpretable and independently verifiable dimensions: Country association :Is
the topic of the Wikipedia article associated with the same country as the image and its caption? Topic
alignment :Does the topic of the Wikipedia article align with the semantic category of the image and
its caption? Explicit visual representation :Is the topic of the Wikipedia article explicitly mentioned
or visually represented in the image and its caption? Each dimension is framed as a binary (True /
False) question to reduce ambiguity and improve annotation consistency. However, for the country
association dimension (the first listed), we introduce an additional label, " Cannot be determined ",
to handle cases where this association is unclear from the annotator’s perspective. Additionally,
annotators are also instructed to include the title and URL of any relevant Wikipedia article they
believe is missing from the top-10 retrieved results. These manually suggested articles are treated as
the most cultural references closely related to the given image (see details in Appendix G).
Quality control. To ensure the quality and consistency of our annotations, we implement several
quality control methods. First of all, prior to the annotation process, all annotators are required to
carefully review a detailed instruction file outlining the relevance criteria and annotation guidelines.
Table 1: Statistics of the RAVENEA dataset. The dataset
is constructed by curating existing sources and augmenting
them with over 10,000 wiki-derived documents to broaden
cultural knowledge coverage and enhance content diversity.
Dataset Images Documents Pairs Questions Captions
CVQA 1,213 8,319 12,130 2,331 -
CCUB 655 4,441 6,550 - 655
RAVENEA 1,868 11,580 18,680 2,331 655To ensure proper understanding of the
guidelines, annotators are required to
complete a mock annotation test and
correctly answer all questions before
proceeding with the actual annotation
tasks (see Figure 14). We also per-
form an additional quality check on a
subset of the dataset. Specifically, for
each selected countries, we employ an
additional local quality checker who
is tasked with manually reviewing the annotations to verify their accuracy and adherence to the
guidelines. The quality checker reviews a random sample of annotated items, focusing on both the
relevance labels and the justification behind any edge cases, such as borderline relevance or use of
the “Cannot be determined” label. If inconsistencies or deviations from the annotation guidelines
are identified, the affected samples are flagged for re-annotation. The overall acceptance rate from
the meta quality checkers is 98.2%. The inter-annotator agreement (IAA) Cohen’s Kappa ( κ) [36]
between the meta checker and annotator on the sampled annotations is 0.83.
5https://huggingface.co/datasets/wikimedia/wikipedia
4

Q: What is the name of the dish shown in the image?A. Day of the Dead CakeB. Pan de muertoC. Three Kings cakeD. Fruit cake
   … (ring of kings or three King's bread) is eaten on "El Dia de Los Reyes" ("The Day of the Kings"), … celebration of the Three Kings visiting the infant Jesus to give him gifts …W/O RAG: B ❌
With CaCLIP: C ✅
Q: Which revolution of India is this brand associated with?A. GoldenB. GreenC. WhiteD. Yellow
    Amul spurred India's White Revolution, which made the country the world's largest producer of milk and milk products, and has since ventured into overseas markets.W/O RAG: A ❌
With CaCLIP: C ✅
Q: Between which Mexican states is the train route shown in the image?A. Chihuahua and VeracruzB. Chihuahua and SinaloaC. Michoacán and SinaloaD. Chihuahua and Sonora
    The Ferrocarril Chihuahua al Pacífico …, also known as El Chepe … is a major rail line in northwest Mexico, linking the city of Chihuahua to Los Mochis and its port, Topolobampo.W/O RAG: D ❌
With CaCLIP: B ✅
cVQA
cIC
W/O RAG: A bustling cityscape at dusk with modern skyscrapers and busy streets.With CaCLIP: A view of Seoul's bustling downtown area during twilight, showcasing its modern architecture and busy streets.   Seoul, officially Seoul Special Metropolitan City,is the capital …
W/O RAG: A bustling plaza in front of an ornate building with a golden dome, surrounded by people and greenery.With CaCLIP: A view of the Palacio de Bellas Artes in Mexico City showcasing its grand architecture and bustling surroundings.   The Palacio de Bellas Artes … is a prominent … in Mexico City.
W/O RAG: Three individuals wearing vibrant traditional attire perform a dance outdoors.With CaCLIP: A group of people wearing colorful traditional Igbo attire perform the Egedege dance at an event.   Igbo culture are the … Igbo people of southeastern Nigeria.
Figure 4: Examples demonstrating the impact of CaCLIP Wikipedia retrieval integration on
cVQA and cIC tasks using DeepseekVL2-Tiny. When augmented with culture-aware retrieval, the
model exhibits enhanced sensitivity to cultural context.
3.2 Dataset Statistics
We present the statistics of our dataset in Table 1. The dataset comprises a total of 1,868 culturally
diverse images, with approximately 65% originating from CVQA and the remaining 35% from
CCUB. Each image is paired with a GPT-4o-generated cultural caption, as well as the top-10 ranked
Wikipedia documents retrieved via a cultural relevance scoring pipeline (Figure 3), yielding a total
of18,680 image-document pairs (see illustrated examples in Figure 4). The collection spans eight
countries and encompasses a broad spectrum of cultural domains, such as traditional attire, festivals,
architecture, cuisine, and social practices (see more statistics in Appendix E).
4 Culture-aware Multimodal Retriever
Leveraging the RAVENEA dataset, we train and evaluate seven multimodal retrievers to retrieve
culturally relevant Wikipedia documents using both visual and textual inputs. We fine-tune five
representative models—spanning both generative and discriminative paradigms—to optimize multi-
modal document retrieval. Performance is evaluated using standard retrieval metrics, including Mean
Reciprocal Rank (MRR) [ 37], Precision@k (P@k) [ 38], and Normalized Discounted Cumulative
Gain (nDCG@k) [ 39], where k∈ {1,3,5}. We integrate responses from three annotation questions
per data point into a continuous scale ranging from −3to3, where higher values indicate stronger
cultural relevance. We fine-tune a VisualBERT-based [ 40,41] reranker following standard BERT-style
setups [ 42], and adapt two multimodal generators—VL-T5 [ 43] and LLaV A-OneVision-7B [ 19]—for
end-to-end document retrieval [ 44,45,46]. To enhance cultural awareness in the contrastive retrieval,
we introduce Culture-Aware Contrastive (CAC) learning, a supervised learning framework compati-
ble with both CLIP and SigLIP architectures. We denote the culture-aware fine-tuned versions of
CLIP-L/14@224px and SigLIP2-SO/14@384px using CAC as CaCLIP and CaSigLIP2, respectively.
4.1 Culture-aware Contrastive Learning
Given an image Iiassociated with Ttextual descriptions {Di1, Di2, . . . , D iT}, each document
Ditis annotated with a binary label yit∈ {0,1}, where yit= 1 indicates cultural relevance and
5

Table 2: Performance with different retriever models. Fine-tuned contrastive models consis-
tently outperform their frozen counterparts across tasks. Here, "CaSigLIP"=Culture-aware SigLIP2-
SO/14@384px, and "CaCLIP"=Culture-aware CLIP-L/14@224px. Models in gray are frozen.
Method MRR↑P@1↑P@3↑P@5↑nDCG@1 ↑nDCG@3 ↑nDCG@5 ↑
SigLIP2-SO/14@384px 66.71 50.42 40.14 34.25 58.85 62.43 67.50
CLIP-L/14@224px 70.76 54.58 43.47 36.58 62.54 67.25 72.43
VisualBERT 59.66 42.50 35.97 32.75 51.29 55.49 62.29
VL-T5 55.53 35.42 34.03 31.42 45.21 53.18 59.62
LLaV A-OneVision-7B 54.15 36.20 31.83 28.69 45.85 50.82 56.74
CaSigLIP2 (ours) 69.35 54.17 43.47 36.50 61.98 66.75 71.98
CaCLIP (ours) 78.34 65.42 49.44 39.50 72.25 75.22 79.32
yit= 0indicates irrelevance. For each image–text pair (Ii, Dit), we employ a shared vision-language
encoder—such as CLIP—to obtain modality-specific representations: EIi=EV(Ii)for the visual
input and EDit=EL(Dit)for the textual input. We then compute the cosine similarity score sit
between EIiand each corresponding EDit, resulting in a similarity vector Si= [si1, si2, . . . , s iT].
Culture-awareness classification now amounts to:
LCulture Classify =−1
B·TBX
i=1TX
j=1[yijlogσ(sij) + (1−yij) log(1 −σ(sij))], (1)
where B is the number of images; σ(·)denotes the sigmoid function.
To prioritize culturally relevant descriptions in the ranking, we apply a margin ranking loss between
all pairs of descriptions with differing cultural relevance. For each image Ii, we compare all pairs
(Dij, Dik)such that yij= 1 andyik= 0, and encourage the model to assign a higher similarity
score to the relevant description. The ranking loss is defined as:
LRank=1
BBX
i=1TX
j,k=1
yij=1,yik=0max (0 , δ−(sij−sik)), (2)
To mitigate the risk of overly similar positive text embeddings for the same image, we introduce a
penalty that encourages intra-modal diversity among textual representations. We apply a diversity-
promoting loss that forces the similarity between different text embeddings to be reduced while
keeping each embedding highly similar to itself. Specifically, the penalty is formulated using an
exponential function to emphasize the dissimilarity between embeddings:
LDiversity (Si) =−TX
t=1log 
exp(sit)PT
j=1exp(sij)!
(3)
Then we can get the culture-aware contrastive loss:
LCAC=1
3(LCulture Classify +LRank+LDiversity ). (4)
4.2 Multimodal Retrieval Results
We perform a comprehensive evaluation of both frozen and fine-tuned retrievers, and present the
results in Table 2. We find that fine-tuned models, particularly those based on contrastive learning,
consistently outperform their frozen counterparts. For instance, CaCLIP achieves a substantial
improvement in P@1, rising from 54.58% to 65.42%, and sets a new SOTA across all evaluation
metrics. Although SigLIP2-SO/14@384px also benefits from fine-tuning, the performance gains are
comparatively modest. In contrast, models such as LLaV A-OneVision-7B, VL-T5, and VisualBERT
lag behind after fine-tuning, even underperforming relative to frozen baselines. This underperfor-
mance likely stems from the fact that models such as LLaV A-OneVision-7B and VisualBERT were
originally pretrained for generative tasks with different objectives, whereas CLIP-L/14@224px and
SigLIP2-SO/14@384px were explicitly trained for similarity-based alignment, providing them with a
structural advantage in retrieval settings.
6

5 Multimodal Retrieval-augmented Visual Culture Understanding
We then evaluate the effectiveness of these retrievers with 14 SOTA VLMs, spanning a diverse set of
architectures. We conduct experiments on two downstream tasks: cVQA andcIC, respectively.
5.1 Experimental Setup
Models. We benchmark open and closed-weight SOTA VLMs on RAVENEA , leveraging various
retrievers against non-RAG baselines, assessing retrieval effectiveness across models of different sizes.
The open-weight models include LLaV A-OneVision-7B [ 19], Pixtral-12B [ 20], Phi-4 Multimodal-
Instruct [ 21], Gemma3-4B-Instruct and 27B-Instruct [ 22], Qwen2.5-VL-Instruct (3B, 7B, 72B6) [23],
InternVL3 (2B, 8B, 38B) [ 24], and Deepseek-VL2 variants (Tiny and Base) [ 25]. For the closed
models, we adopt GPT-4.1 [18] (accessed on 2025/04/14)7.
Table 3: Kendall’s τrank correlation [47] between automatic met-
rics and human judgments for the CCUB task. Statistically signifi-
cant correlations ( p <0.05) are marked with ✓. Our proposed metrics
correlate stronger with human evaluation than the others.
Rouge-L [48] CIDER [49] BERTScore [50] CLIPScore [51] RegionScore (ours)
-0.172 ✗ -0.316 ✗ -0.011 ✗ 0.139✗ 0.442✓Evaluation metrics. For
the cVQA task, we use ac-
curacy as the primary eval-
uation metric, which mea-
sures the proportion of cor-
rectly predicted answers.
For the cIC task, we employ
several evaluation metrics
including ROUGE-L [ 48], CIDEr [ 49], BERTScore [ 50], and CLIPScore [ 51], to assess the alignment
between generated and reference captions across lexical, syntactic, and embedding-based levels. To
further evaluate the cultural relevance and human-perceived quality, we conduct a human evaluation
study. We employed four researchers to select the most accurate caption from 14 VLMs (see details in
Appendix I and L), and find a significant mismatch between automatic metric scores and human judg-
ments of cultural appropriateness (see Table 3). To bridge this gap, we further introduce RegionScore ,
a novel evaluation metric designed to quantify cultural grounding (see details in Appendix H and
Table 9). It measures how well captions identify the correct country names tied to cultural elements,
adding geographic and cultural specificity in image captioning.
5.2 Overall Performance
We present the main results in Table 4. The results demonstrate the efficacy of incorporating culture-
aware retrieval augmentation. Employing fine-tuned retrievers yields substantial performance gains
over both non-RAG and frozen retrievers baselines. Specifically, the CaCLIP achieves the highest
average performance across both tasks, improves the accuracy for cVQA from 67.7% to 71.5%, and
substantially improving the RegionScore for cIC from 40.2% to 58.1%. While CLIP-L/14@224px
also offers improvements, fine-tuning consistently unlocks further potential. Furthermore, in the
cVQA task , among all evaluated models, GPT-4.1 achieves the highest accuracy (86.8%) without
RAG. Within the category of open-weight models, Qwen2.5-VL-72B leads with an accuracy of 81.0%.
For lightweight models ( ≤8B parameters), Qwen2.5-VL-7B achieves the best performance without
RAG, reaching 67.7%. However, incorporating a CaCLIP significantly boosts performance, which
enables InternVL3-8B to achieve 74.2% and outperform Qwen2.5-VL-7B by 0.6% with identical
reranking. Notably, across multiple model families, augmenting the smallest variant with CaCLIP
consistently elevates its performance to match or even exceed that of the next larger model tier. In
thecIC task , with culture-aware contrastive learning, CaCLIP demonstrates substantial gains in
identifying the culture in country-level of visual content, especially when built on top of VLMs with
strong vision-language priors. It achieves the highest average RegionScore (58.1%) among the six
reranking methods evaluated, with peak performance reaching 76.3% on the Gemma3-4B backbone.
CaCLIP achieves leading the scores on 9 of 14 diverse suits of VLMs. This result underscores
CaCLIP’s robustness and adaptability, particularly in culture-aware image captioning and retrieval
tasks that demand fine-grained multimodal alignment.
6Due to computational constraints, we use the quantized version Qwen2.5-VL-Instruct-72B-AWQ.
7Knowledge cutoff: June 1, 2024; https://platform.openai.com/docs/models/gpt-4.1
7

Table 4: cVQA and cIC Performance w/ and w/o RAG. Models in gray are frozen retrievers.
Results are colored as Best . VLMs augmented with finetuned retriever generally perform better.
Open Weights Closed Weights
Retriever Average
DeepSeek-VL2-Tiny
DeepSeek-VL2
Qwen2.5-VL-3B
Qwen2.5-VL-7B
Qwen2.5-VL-72B
InternVL3-2B
InternVL3-8B
InternVL3-38B
Gemma3-4B
Gemma3-27B
Phi4-Multimodal
Pixtral-12B
LLaV A-OneVision-7B
GPT-4.1
cV QA Accuracy ↑
W/O RAG 67.7 57.1 65.5 64.2 67.7 81.0 58.4 66.5 71.9 63.6 78.7 55.5 66.1 64.8 86.8
SigLIP2-SO/14@384px 67.2 58.7 55.8 63.9 67.7 77.1 62.3 68.4 71.9 60.7 71.9 62.6 69.4 67.1 82.6
CLIP-L/14@224px 70.6 64.2 56.1 67.7 73.2 79.0 68.1 71.0 76.1 66.1 74.5 69.4 71.0 69.7 81.9
Finetuned Models
VisualBERT 67.1 58.7 59.4 66.5 68.4 77.1 61.3 66.1 71.9 60.7 73.6 63.2 66.5 63.9 82.3
VL-T5 65.8 56.5 61.3 62.9 69.4 77.1 60.3 66.8 70.3 55.8 74.2 61.0 61.0 61.9 82.6
LLaV A-OneVision-7B 67.8 60.0 61.3 64.8 71.3 80.7 61.3 64.8 73.9 62.3 73.6 62.6 67.4 63.9 81.3
CaSigLIP2 (ours) 69.8 62.6 58.7 67.1 72.3 77.7 65.8 72.6 75.8 64.2 75.5 64.2 70.7 69.4 82.6
CaCLIP (ours) 71.5 65.8 60.0 69.4 73.6 81.0 68.7 74.2 75.2 66.8 75.2 66.8 71.0 70.0 83.2
cICRegionScore ↑
W/O RAG 40.2 31.3 36.3 36.3 48.8 47.5 27.5 37.5 22.5 67.5 62.5 10.0 35.0 18.8 47.5
SigLIP2-SO/14@384px 52.5 50.0 56.3 36.3 57.5 53.8 58.8 56.3 61.3 70.0 66.3 30.0 53.8 41.3 53.8
CLIP-L/14@224px 55.4 56.3 63.8 41.3 66.3 60.0 61.3 62.5 67.5 70.0 70.0 31.3 55.0 47.5 58.8
Finetuned Models
VisualBERT 56.3 58.8 63.8 41.3 58.8 62.5 61.3 61.3 66.3 71.3 70.0 30.0 58.8 45.0 62.5
VL-T5 55.4 60.0 65.0 41.3 62.5 66.3 62.5 58.8 67.5 71.3 68.8 30.0 61.3 40.0 61.3
LLaV A-OneVision-7B 56.0 60.0 65.0 40.0 60.0 62.5 61.3 62.5 66.3 71.3 71.3 28.8 57.5 47.5 60.0
CaSigLIP2 (ours) 56.3 57.5 65.0 36.3 61.3 63.8 58.8 61.3 67.5 71.3 70.0 27.5 57.5 46.3 65.0
CaCLIP (ours) 58.1 60.0 67.5 42.5 65.0 65.0 62.5 60.0 70.0 76.3 75.0 27.5 62.5 52.5 55.0
6 Analysis and Further Discussion
Culture-aware retrieval augmentation substantially benefits VLMs across both cVQA and cIC tasks,
compared to their no-RAG counterparts. In this section, we explore the margin of the improvement,
cultural preference and the effectiveness of cultural annotation.
6.1 Scaling Models Yields Diminishing or Negative Returns across Retrievers
−10010
Gemma3 Qwen2.5VL InternVL3 DeepSeek-VL202040
Smallest LargestAcc. ΔR.S ΔcVQA
cIC
Figure 5: Performance improvements for smallest and largest mod-
els per family with multimodal retrievers. Scaling models yields
marginal gains with various retrievers, even negative effects in both
cVQA and cIC tasks. "ACC." denotes accuracy; "R.S." refers to the
RegionScore; " ∆" represents the change incorporated with RAG com-
pared to the non-RAG baseline.In the cVQA task , within
the same VLM family,
performance differences be-
tween RAG and non-RAG
approaches exhibit non-
monotonic trends as model
size scales. For all four
model families, larger mod-
els show marginal or even
negative returns from RAG
integration. What’s more,
sensitivity to RAG varies
across model families.
Notably, DeepSeek-VL2
demonstrates the most
pronounced performance
gap: the smallest model
benefits from RAG with an
average improvement of approximately +5%, whereas the largest model in the same family suffers
8

W/O RAG 
 3Qs
Q1
Q2
Q3
Q1+Q2
Q1+Q3
Q2+Q3
556065707580AccuracyW/O RAG 
 3Qs
Q1
Q2
Q3
Q1+Q2
Q1+Q3
Q2+Q3
203040506070Region ScoreW/O RAG 
 3Qs
Q1
Q2
Q3
Q1+Q2
Q1+Q3
Q2+Q3
556065707580With SigLIP2DeepSeek-VL2DeepSeek-VL2-TinyGemma3-4BGemma3-27BInternVL3-2BInternVL3-8BInternVL3-38BLLaVA-Onevision-7BPhi4-MultimodalPixtral-12BQwen2.5-3BQwen2.5-7BQwen2.5-72BAVG (CaSigLIP2)AccuracyW/O RAG 
 3Qs
Q1
Q2
Q3
Q1+Q2
Q1+Q3
Q2+Q3
556065707580With CaCLIPDeepSeek-VL2DeepSeek-VL2-TinyGemma3-4BGemma3-27BInternVL3-2BInternVL3-8BInternVL3-38BLLaVA-Onevision-7BPhi4-MultimodalPixtral-12BQwen2.5-3BQwen2.5-7BQwen2.5-72BAVG (CaCLIP)AccuracyFigure 7: Ablation for different annotation questions. Combining all three culture-relevant
annotation questions yields the best performance.
a degradation of around -6% on average. In the cIC task , the effectiveness of RAG exhibits a
consistent trend with respect to model scale within a given model family. Across all four model
families evaluated, larger models tend to benefit less—or at most comparably—from the integration
of RAG, suggesting diminishing returns at higher capacity. Among them, Gemma3 models show
the smallest relative improvement, achieving approximately a +7% performance gain on average,
whereas InternVL3 models yield the highest benefit, with performance gains reaching up to +30%.
6.2 Differences across Countries
We evaluate all models using CaCLIP across a range of countries for both tasks, as shown in Figure 6.
In the cVQA setting, most VLMs exhibit substantially diminished performance on culture-specific
questions regarding Nigeria and Indonesia, in contrast to their performance on questions under other
national contexts. Interestingly, questions related to Spanish culture reveal high inter-model variance,
40 50 60 70 80 90 100ChinaIndiaNigeriaMexicoIndonesiaSpainRussia
10 20 30 40 50 60 70 80NigeriaMexicoChinaKoreaIndia
cVQA Accuracy cIC Region Score
Figure 6: Performance of 14 VLMs with CaCLIP across dif-
ferent countries. Despite the integration of CaCLIP, disparities
in model performance persist across countries.with accuracy differentials reach-
ing up to 35%, underscoring sig-
nificant discrepancies in cultural
representation across models. In
the cIC task, VLMs consistently
underperform on images and doc-
uments associated with Indian
cultural contexts, while achiev-
ing the highest RegionScores
on Korean culture-related inputs.
Model performance on Indian
culture is particularly volatile, indicating inconsistent cultural grounding across architectures. By com-
parison, Korean and Chinese cultural inputs yield more stable performance across models, suggesting
entrenched model-specific preferences in cultural alignment (see more results in Appendix J).
6.3 Ablation Study on Annotation Questions
We further perform ablation studies across diverse combinations of annotation questions to assess
their impact on downstream performance. Specifically, we evaluate 13 open-weight VLMs equipped
with either CaSigLIP or CaCLIP, each trained on datasets constructed using varying subsets of
culture-relevant annotations. From Figure 7, we can observe that leveraging all three questions
(Q1 regarding country association; Q2 for topic alignment; Q3 for visual representation fidelity)
yields the strongest performance on both cVQA and cIC tasks. For the cVQA task, we find Q1
provides the most significant benefit to CaSigLIP, whereas CaCLIP gains more from Q2. Among all
pairwise combinations, the joint supervision from Q1 (country associations) and Q2 (topic alignment)
proves slightly more effective than other pairs. In the cIC task, both CaSigLIP and CaCLIP achieve
better performance improvements when trained with data derived from Q1, compared to other single-
question sets. For pairwise combinations, CaCLIP benefits most from the Q1+Q3 combination, while
CaSigLIP shows a clear preference for the Q2+Q3 setup.
9

7 Conclusion
We introduce RAVENEA , a novel benchmark dataset designed to comprehensively evaluate the cultural
sensitivity of diverse multimodal retrievers across 14 SOTA VLMs. The benchmark consists of
culturally contextualized queries and image captions from eight countries, curated and paired with
Wikipedia passages ranked by human annotators based on their cultural relevance. Our findings
highlight the potential of RAG for visual cultural understanding, particularly when lightweight
VLMs are enhanced with culturally-aware multimodal retrievers such as CaCLIP, which consistently
outperform their non-augmented counterparts – with the caveat that the largest models in each
VLM family often exhibit diminishing returns when integrated with RAG. Notably, the inclusion of
culture-sensitive questions during data annotation significantly improves effectiveness of multimodal
retrievers and enhances performance in downstream tasks.
8 Acknowledgment
We would like to thank all the annotators for their work. Special thanks to Nico Lang, Peter Ebert
Christensen, Zhaochong An, Stella Frank, Srishti Yadav, for providing helpful research advice.
Serge Belongie and Jiaang Li are supported by the Pioneer Centre for AI, DNRF grant number
P1. Ivan Vuli ´c is supported by a personal Royal Society University Research Fellowship ‘Inclusive
and Sustainable Language Technology for a Truly Multilingual World’ (no 221137). Wenyan Li is
supported by the Lundbeck Foundation (BrainDrugs grant: R279-2018-1145).
References
[1]Elisa Di Nuovo, Manuela Sanguinetti, PIER Balestrucci, Luca Anselma, Cristian Bernareggi,
Alessandro Mazzei, et al. Educational dialogue systems for visually impaired students: Introduc-
ing a task-oriented user-agent corpus. In Proceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) ,
pages 5507–5519. ELRA and ICCL, 2024.
[2]Maria De Marsico, Chiara Giacanelli, Clizia Giorgia Manganaro, Alessio Palma, and Davide
Santoro. Vqask: a multimodal android gpt-based application to help blind users visualize
pictures. In Proceedings of the 2024 International Conference on Advanced Visual Interfaces ,
pages 1–5, 2024.
[3]Luke Bates, Peter Ebert Christensen, Preslav Nakov, and Iryna Gurevych. A template is all
you meme. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025
Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages 10443–10475,
Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
[4]Ziyu Yao, Xuxin Cheng, Zhiqi Huang, and Lei Li. Countllm: Towards generalizable repetitive
action counting via large language model. arXiv preprint arXiv:2503.17690 , 2025.
[5]Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens,
Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal
understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition , pages 9556–9567, 2024.
[6]Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer
vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014,
proceedings, part v 13 , pages 740–755. Springer, 2014.
[7]Zhaochong An, Guolei Sun, Yun Liu, Runjia Li, Min Wu, Ming-Ming Cheng, Ender Konukoglu,
and Serge Belongie. Multimodality helps few-shot 3d point cloud semantic segmentation. In
The Thirteenth International Conference on Learning Representations , 2025.
[8]Zhaochong An, Guolei Sun, Yun Liu, Runjia Li, Junlin Han, Ender Konukoglu, and Serge
Belongie. Generalized few-shot 3d point cloud segmentation with vision-language model. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2025.
10

[9]Shravan Nayak, Kanishk Jain, Rabiul Awal, Siva Reddy, Sjoerd Van Steenkiste, Lisa Anne
Hendricks, Karolina Stanczak, and Aishwarya Agrawal. Benchmarking vision language models
for cultural understanding. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 5769–5790, Miami, Florida, USA, November 2024. Association for Computational
Linguistics.
[10] Shudong Liu, Yiqiao Jin, Cheng Li, Derek F Wong, Qingsong Wen, Lichao Sun, Haipeng Chen,
Xing Xie, and Jindong Wang. Culturevlm: Characterizing and improving cultural understanding
of vision-language models for over 100 countries. arXiv preprint arXiv:2501.01282 , 2025.
[11] Simran Khanuja, Sathyanarayanan Ramamoorthy, Yueqi Song, and Graham Neubig. An image
speaks a thousand words, but can everyone listen? on image transcreation for cultural relevance.
In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing , pages 10258–10279, Miami,
Florida, USA, November 2024. Association for Computational Linguistics.
[12] Yong Cao, Wenyan Li, Jiaang Li, Yifei Yuan, Antonia Karamolegkou, and Daniel Hershcovich.
Exploring visual culture awareness in gpt-4v: A comprehensive probing. arXiv preprint
arXiv:2402.06015 , 2024.
[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[14] Wonduk Seo, Zonghao Yuan, and Yi Bu. Valuesrag: Enhancing cultural alignment through
retrieval-augmented contextual learning. arXiv preprint arXiv:2501.01031 , 2025.
[15] Piyawat Lertvittayakumjorn, David Kinney, Vinodkumar Prabhakaran, Donald Martin, and
Sunipa Dev. Towards geo-culturally grounded llm generations. arXiv preprint arXiv:2502.13497 ,
2025.
[16] David Romero, Chenyang Lyu, Haryo Wibowo, Santiago Góngora, Aishik Mandal, Sukannya
Purkayastha, Jesus-German Ortiz-Barajas, Emilio Cueva, Jinheon Baek, Soyeong Jeong, et al.
Cvqa: Culturally-diverse multilingual visual question answering benchmark. Advances in
Neural Information Processing Systems , 37:11479–11505, 2025.
[17] Zhixuan Liu, Youeun Shin, Beverley-Claire Okogwu, Youngsik Yun, Lia Coleman, Peter
Schaldenbrand, Jihie Kim, and Jean Oh. Towards equitable representation in text-to-image
synthesis models with the cross-cultural understanding benchmark (ccub) dataset. arXiv preprint
arXiv:2301.12073 , 2023.
[18] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774 , 2023.
[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan
Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaV A-onevision: Easy visual task transfer.
Transactions on Machine Learning Research , 2025.
[20] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot,
Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet,
et al. Pixtral 12b. arXiv preprint arXiv:2410.07073 , 2024.
[21] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach,
Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini
technical report: Compact yet powerful multimodal language models via mixture-of-loras.
arXiv preprint arXiv:2503.01743 , 2025.
[22] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona
Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma
3 technical report. arXiv preprint arXiv:2503.19786 , 2025.
11

[23] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
[24] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan,
Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time
recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 , 2025.
[25] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao,
Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-
language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302 ,
2024.
[26] Simone Conia, Daniel Lee, Min Li, Umar Farooq Minhas, Saloni Potdar, and Yunyao Li. To-
wards cross-cultural machine translation with retrieval-augmented generation from multilingual
knowledge graphs. In Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing . Association for Computational Linguistics, 2024.
[27] Tianyi Hu, Maria Maistro, and Daniel Hershcovich. Bridging cultures in the kitchen: A
framework and benchmark for cross-cultural recipe retrieval. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language Processing . Association for Computational
Linguistics, November 2024.
[28] Yi Fung, Tuhin Chakrabarty, Hao Guo, Owen Rambow, Smaranda Muresan, and Heng Ji.
NORMSAGE: Multi-lingual multi-cultural norm discovery from conversations on-the-fly. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
Singapore, 2023. Association for Computational Linguistics.
[29] Srishti Yadav, Zhi Zhang, Daniel Hershcovich, and Ekaterina Shutova. Beyond words: Exploring
cultural value sensitivity in multimodal models. In Luis Chiruzzo, Alan Ritter, and Lu Wang,
editors, Findings of the Association for Computational Linguistics: NAACL 2025 , pages 7592–
7608, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
[30] Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and
Desmond Elliott. Visually grounded reasoning across languages and cultures. In Proceed-
ings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages
10467–10485, Online and Punta Cana, Dominican Republic, November 2021. Association for
Computational Linguistics.
[31] Genta Indra Winata, Frederikus Hudi, Patrick Amadeus Irawan, David Anugraha, Rifki Afina
Putri, Wang Yutong, Adam Nohejl, Ubaidillah Ariq Prathama, Nedjma Ousidhoum, Afifa Amri-
ani, Anar Rzayev, Anirban Das, Ashmari Pramodya, Aulia Adila, Bryan Wilie, Candy Olivia
Mawalim, Cheng Ching Lam, Daud Abolade, Emmanuele Chersoni, Enrico Santus, Fariz
Ikhwantri, Garry Kuwanto, Hanyang Zhao, Haryo Akbarianto Wibowo, Holy Lovenia, Jan Chris-
tian Blaise Cruz, Jan Wira Gotama Putra, Junho Myung, Lucky Susanto, Maria Angelica Riera
Machin, Marina Zhukova, Michael Anugraha, Muhammad Farid Adilazuarda, Natasha Christa-
belle Santosa, Peerat Limkonchotiwat, Raj Dabre, Rio Alexander Audino, Samuel Cahyawijaya,
Shi-Xiong Zhang, Stephanie Yulia Salim, Yi Zhou, Yinxuan Gui, David Ifeoluwa Adelani, En-
Shiun Annie Lee, Shogo Okada, Ayu Purwarianti, Alham Fikri Aji, Taro Watanabe, Derry Tanti
Wijaya, Alice Oh, and Chong-Wah Ngo. WorldCuisines: A massive-scale benchmark for
multilingual and multicultural visual question answering on global cuisines. In Luis Chiruzzo,
Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers) , pages 3242–3264, Albuquerque, New Mexico, April
2025. Association for Computational Linguistics.
[32] Wenyan Li, Crystina Zhang, Jiaang Li, Qiwei Peng, Raphael Tang, Li Zhou, Weijia Zhang,
Guimin Hu, Yifei Yuan, Anders Søgaard, Daniel Hershcovich, and Desmond Elliott. FoodieQA:
A multimodal dataset for fine-grained understanding of Chinese food culture. In Proceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing , 2024.
12

[33] Mehar Bhatia, Sahithya Ravi, Aditya Chinchure, EunJeong Hwang, and Vered Shwartz. From
local concepts to universals: Evaluating the multicultural understanding of vision-language
models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing . Association for Computational Linguistics, 2024.
[34] Malvina Nikandrou, Georgios Pantazopoulos, Nikolas Vitsakis, Ioannis Konstas, and Alessandro
Suglia. CROPE: Evaluating in-context adaptation of vision and language models to culture-
specific concepts. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the
2025 Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers) , pages 7917–7936,
Albuquerque, New Mexico, April 2025. Association for Computational Linguistics.
[35] Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and
beyond. Found. Trends Inf. Retr. , 3:333–389, 2009.
[36] Ron Artstein. Inter-annotator agreement. Handbook of linguistic annotation , pages 297–313,
2017.
[37] Yue Shi, Alexandros Karatzoglou, Linas Baltrunas, Martha Larson, Nuria Oliver, and Alan
Hanjalic. Climf: learning to maximize reciprocal rank with collaborative less-is-more filtering.
InProceedings of the sixth ACM conference on Recommender systems , pages 139–146, 2012.
[38] Kalervo Järvelin and Jaana Kekäläinen. Ir evaluation methods for retrieving highly relevant
documents. In ACM SIGIR Forum , volume 51, pages 243–250. ACM New York, NY , USA,
2017.
[39] Yining Wang, Liwei Wang, Yuanzhi Li, Di He, and Tie-Yan Liu. A theoretical analysis of ndcg
type ranking measures. In Conference on learning theory , pages 25–54. PMLR, 2013.
[40] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A
simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 ,
2019.
[41] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. What does BERT
with vision look at? In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors,
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages
5265–5275, Online, July 2020. Association for Computational Linguistics.
[42] Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert. arXiv preprint
arXiv:1901.04085 , 2019.
[43] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text
generation. In International Conference on Machine Learning , pages 1931–1942. PMLR, 2021.
[44] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yiqun Liu, Yixing Fan, and Xueqi Cheng. Cor-
pusbrain: Pre-train a generative retrieval model for knowledge-intensive language tasks. In
Proceedings of the 31st ACM International Conference on Information & Knowledge Manage-
ment , pages 191–200, 2022.
[45] Yifei Yuan, Clemencia Siro, Mohammad Aliannejadi, Maarten de Rijke, and Wai Lam. Asking
multimodal clarifying questions in mixed-initiative conversational search. In Proceedings of the
ACM Web Conference 2024 , pages 1474–1485, 2024.
[46] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin
Chen, Dawei Yin, Maarten Rijke, and Zhaochun Ren. Learning to tokenize for generative
retrieval. Advances in Neural Information Processing Systems , 36:46345–46361, 2023.
[47] Maurice G Kendall. A new measure of rank correlation. Biometrika , 30(1-2):81–93, 1938.
[48] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain, July 2004. Association for Computational
Linguistics.
13

[49] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image
description evaluation. In Proceedings of the IEEE conference on computer vision and pattern
recognition , pages 4566–4575, 2015.
[50] Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert. In International Conference on Learning Representations ,
2020.
[51] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A
reference-free evaluation metric for image captioning. In Marie-Francine Moens, Xuanjing
Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing , pages 7514–7528, Online and Punta Cana,
Dominican Republic, November 2021. Association for Computational Linguistics.
[52] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly
capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
[53] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card , 1:1,
2024.
[54] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based
on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) , June 2019.
[55] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems
Principles , pages 611–626, 2023.
[56] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár,
and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv
preprint arXiv:1504.00325 , 2015.
14

A Limitations
While RAVENEA establishes a solid foundation for advancing the study of visual culture understanding
with retrieval augmentation, it has three limitations that warrant future attention. First, due to
budgetary constraints, the dataset’s cultural scope is currently limited to eight countries and eleven
categories. Although this selection introduces meaningful diversity, it does not comprehensively
represent the global spectrum of cultural perspectives, particularly those of underrepresented or
marginalized communities. Second, our use of Wikipedia as the primary external knowledge source
introduces inherent biases and may lack the depth, plurality, and contextual richness necessary for
nuanced cultural interpretation. Finally, due to resource limitations, we were unable to include certain
proprietary VLMs that require paid APIs, such as Gemini 2.5 Pro [ 52] and Claude Opus 3.7 [ 53].
We hypothesize that their performance would be comparable to GPT-4.1, which was included in our
evaluation, but this remains an open empirical question.
B Future Directions
Our work opens several avenues for advancing visual culture understanding in multimodal models.
First, expanding RAVENEA to include more countries, cultural categories, and diverse knowledge
sources, beyond Wikipedia, would improve coverage and reduce institutional bias. Second, future
benchmarks could include richer tasks beyond cVQA and cIC, such as culture-grounded object recog-
nition, historical retrieval, and symbolic interpretation, to better capture cultural semantics. Third, our
results suggest a need for culturally-aware evaluation metrics, particularly for text generation. The
limited effectiveness of retrieval augmentation in larger models also warrants further study, especially
regarding how cultural knowledge is integrated and utilized. Together, these directions aim to support
the development of more culturally-sensitive and globally robust vision-language models.
C Ethics Statement
This work focuses on improving the cultural awareness of VLMs through retrieval-augmented
methods. All data used in the construction of the RAVENEA benchmark were sourced from publicly
available datasets and Wikipedia, a community-curated open-access knowledge base. To protect
individual privacy, we apply automated face detection and blur all identifiable faces in images prior to
release. To mitigate cultural bias and ensure broad representation, the benchmark includes images and
documents spanning eight countries and eleven cultural domains, curated and annotated by a diverse
group of annotators. We provide detailed documentation of the annotation process and guidelines to
support transparency and reproducibility. While enhancing cultural understanding is a central goal
of this work, we acknowledge that culture is inherently complex, dynamic, and context-dependent.
Consequently, the benchmark cannot capture the full richness of any cultural context. Finally, this
work does not involve any personally identifiable information (PII), biometric data, or sensitive
attributes. All human annotators were compensated fairly, and data annotation adhered to ethical
guidelines for responsible research.
D Experimental Details for Multi-modal Retrievers
Dataset. We integrate responses from three annotation questions per data point into a continuous
scale ranging from −3to3, where higher values indicate stronger cultural relevance. To ensure
fair evaluation across regions, we adopt the train-validation-test split strategy from class-imbalance
loss [ 54], yielding an approximate 85-5-10 data split. This setup guarantees that validation and
test sets contain a balanced set of images per country (Figure 8), thereby mitigating evaluation bias
from skewed geographic distributions. During training, we employ random cropping for images as a
sample-level augmentation to normalize the distribution of training instances across countries, further
mitigating region-specific sampling biases.
Hyperparameters. In this work, we adopt different sets of hyperparameters as VisualBERT, VL-T5,
LLaV A-OneVision-7B, CLIP-L/14@224px, SigLIP2-SO/14@384px. For VisualBERT, VL-T5, and
LLaV A-OneVision-7B we follow the setting in [ 44,45,46]. We show the training hyperparameters in
15

IndiaMexicoIndonesiaChinaNigeriaKorea Spain Russia050100150200250300350400Test Validation TrainNumber of ImagesFigure 8: Data distributions across eight countries. Restrict the dataset to the images paired with at
least one document annotated by human raters as culturally relevant.
Table 5: Hyperparameters for finetuning on five models.
Hyperparameters VisualBERT VL-T5 LLaV A-OneVision-7B CLIP-L/14@224px SigLIP2-SO/14@384px
batch size 32 128 4 64 64
lr 2e-4 1e-4 1e-4 1e-5 1e-5
lr warmup ratio - 0.1 - - -
weight decay - - - - -
Max Epoch 100 20 10 50 50
Patience 10 10 3 5 5
early stopping Yes
optimizer AdamW
Using LORA - - Yes - -
rerankering experiments for all models in Table 5. All experiments are conducted using a maximum
of 2 Nvidia H100 GPUs.
E Data Statistics
Table 6: Statistics of images in each country.
India Mexico Indonesia China Nigeria Korea Spain Russia
408 349 309 303 223 151 142 77
We then apply cryptographic hashing (SHA-256) to identify and remove duplicate images, resulting in
a cleaner and more distinct cultural image set. Consequently, RAVENEA comprises images collected
from eight countries across four continents, spanning eleven distinct categories. The distribution of
images by country and by category is detailed in Tables 6 and 7, respectively.
F Evaluation of VLMs
For closed-weight model, GPT-4.1, we directly call the corresponding API. For open-source models,
we use vllm [ 55]. During the evaluation, to ensure the stability of the results, we set the temperature
16

Table 7: Statistics of images in each category.
Architecture Cuisine History Art Daily Life Companies Sports & Recreation Transportation Religion Nature Tools
403 402 278 275 185 80 73 68 52 31 21
parameter to 0.0 and the maximum output length to 256. All open-weight models are listed in
Table 8. To avoid the impact of the length the retrieved content, we use the first 256 words in the
top-1 Wikipedia document.
Table 8: Model details: Hugging Face model names.
Model Hugging Face Model Name
LLaV A-OneVision-7B [19] llava-hf/llava-onevision-qwen2-7b-ov-hf
Phi-4-Multimodal [21] microsoft/Phi-4-multimodal-instruct
Pixtral [20] mistral-community/pixtral-12b
Qwen2.5VL family [23] Qwen/Qwen2.5-VL-3B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen2.5-VL-72B-Instruct-AWQ
DeepSeek-VL2 family [25] deepseek-ai/deepseek-vl2
deepseek-ai/deepseek-vl2-tiny
InternVL3 family [24] OpenGVLab/InternVL3-2B
OpenGVLab/InternVL3-8B
OpenGVLab/InternVL3-38B
Gemma3 family [22] google/gemma-3-4b-it
google/gemma-3-27b-it
G Annotation Details
Based on the initial BM25 retrieval results, we refine the cultural relevance label of retrieved
documents via human annotation. We found that directly asking annotators to rate overall cultural
relevance on a continuous scale (e.g., 0–10) led to unreliable and inconsistent labels. This difficulty
arises from several factors: (1) the semantic meaning of intermediate scores is ambiguous, (2)
annotators tended to overemphasize a few salient visual elements [ 56], and (3) small numerical
differences (e.g., between 5 and 6) often fail to reflect meaningful distinctions, especially given the
cognitive load of processing lengthy Wikipedia documents, resulting in intra-annotator variance even
on repeated examples. Instead, given an image–caption–document triplet, we decomposed cultural
relevance into three interpretable and independently verifiable dimensions: country association ,
topic alignment , and explicit visual representation .
Prior to the annotation process, all annotators are required to carefully review a detailed instruction
file outlining the relevance criteria and annotation guidelines. To ensure proper understanding of
the guidelines, annotators are required to complete a mock annotation test and correctly answer all
questions before proceeding with the actual annotation tasks. For each image-caption pair, annotators
are presented with the top 10 Wikipedia documents retrieved by BM25. They are asked to assess
whether each article provides meaningful background or contextual information that is directly
relevant to the cultural elements described in the caption or depicted in the image.
H A New Metric: RegionScore
To evaluate the extent to which captions reference specific geopolitical regions, we introduce a
RegionScore . This metric measures whether a caption contains explicit references to a country or
its common demonyms. Nbe the total number of samples. PRED idenote predicted captions for
thei-th sample. Cibe the country name associated with the i-th sample. Adj(Ci)denote the set of
adjectives or demonyms associated with Ci.Ti={Ci} ∪Adj(Ci)denote the set of region-related
17

terms for the i-th sample. We define binary indicators for each sample:
δpred
i=1if any t∈ Tiappears in PRED i
0otherwise(5)
The RegionScore for predicted captions are then computed as:
RegionScorePRED=1
NNX
i=1δpred
i (6)
These scores reflect the proportion of captions that include explicit regional identifiers. A higher
score indicates stronger region-awareness in the captioning. For the ground truth in CCUB dataset,
the RegionScoreGT= 99% .
I Details of Human Evaluation in the cIC Task
We randomly sample 10 images and generate captions using 14 vision-language models (VLMs)
under three configurations: CaCLIP, CLIP, and a no-retrieval baseline, yielding 420 captions in total.
Four expert annotators participated in the evaluation. For each image, they were presented with
caption triplets—one per retrieval configuration—and asked to assess approximately 35 such triplets
each. Annotators are instructed to select the caption that most accurately and appropriately reflects
the cultural context depicted. To evaluate annotation consistency, we randomly sample 30 triplets and
assign them to a fifth expert annotator. Inter-annotator agreement (IAA) between this annotator and
the original four, measured using Cohen’s κ, is 0.595.
J Additional Results
Phi4-Multimodal Pixtral-12B LlaVA-Onevision-7B Gemma3-4B Gemma3-27B DeepSeek-VL-2-Tiny DeepSeek-VL-2 InternVL3-2B InternVL3-8B InternVL3-38B Qwen2.5VL-3B Qwen2.5VL-7B Qwen2.5VL-72B GPT-4.1 020406080100Nigeria Indonesia Mexico Spain Russia China IndiaAccuracy
Figure 9: Performance of the 14 VLMs equipped with CaCLIP in cVQA task.
As illustrated in Figures 9 and 10, the models exhibit diverse cultural preferences. Notably, most
models achieve relatively stronger performance on Chinese and Indian cultural contexts in the
cVQA task, and on Chinese and Korean contexts in the cIC task. We report the CIDEr, ROUGE-L,
Phi4-Multimodal Pixtral-12B LlaVA-Onevision-7B Gemma3-4B Gemma3-27B DeepSeek-VL-2-Tiny DeepSeek-VL-2 InternVL3-2B InternVL3-8B InternVL3-38B Qwen2.5VL-3B Qwen2.5VL-7B Qwen2.5VL-72B 020406080100Nigeria Mexico China Korea IndiaRegionScore
Figure 10: Performance of the 14 VLMs equipped with CaCLIP in cIC task.
BERTScore, and CLIPScore metrics for 14 VLMs on the cIC task in Table 9.
As shown in Figures 12 and 11, VLMs demonstrate varying degrees of performance shifts across
cultural contexts in both the cIC and cVQA tasks. In the cIC setting, most fine-tuned retrievers yield
18

Table 9: Four metrics comparison of RAG and Non-RAG methods for CCUB task.
Open Weights Closed Weights
Method Average
DeepSeek-VL2-Tiny
DeepSeek-VL2
Qwen2.5-VL-3B
Qwen2.5-VL-7B
Qwen2.5-VL-72B
InternVL3-2B
InternVL3-8B
InternVL3-38B
Gemma3-4B
Gemma3-27B
Phi4-Multimodal
Pixtral-12B
LLaV A-OneVision-7B
GPT-4.1
CCUB Rouge −L
W/O RAG 18.3 18.7 18.2 21.8 18.2 16.5 17.8 19.6 19.8 16.7 15.9 20.6 15.4 21.4 19.6
Frozen Models
SigLIP2-SO/14@384px 18.0 20.0 19.1 23.8 17.2 14.9 19.2 17.4 17.5 15.1 14.9 23.1 14.0 23.6 17.0
CLIP-L/14@224px 18.1 20.9 18.5 23.0 16.8 14.5 18.3 17.2 17.5 15.1 15.8 22.4 14.4 24.7 16.1
Finetuned Models
VisualBERT 18.6 19.5 18.3 22.5 16.8 14.8 17.4 17.4 16.9 15.0 16.0 22.5 14.8 24.3 19.0
VL-T5 17.6 19.0 18.1 23.1 17.9 14.4 18.2 17.2 16.9 14.7 15.4 20.8 13.8 23.5 17.4
LLaV A-OneVision-7B 17.3 18.4 17.7 23.0 17.2 14.5 17.6 16.4 17.1 14.6 15.0 20.9 13.1 22.8 17.3
SigLIP2-SO/14@384px 18.4 19.3 19.2 24.5 18.5 14.8 18.4 18.2 17.2 15.2 16.4 22.2 14.3 23.5 18.9
CLIP-L/14@224px 18.1 19.6 18.6 23.9 17.1 14.7 18.0 17.3 17.7 16.0 16.3 21.5 13.7 24.2 17.0
CCUB CIDER
W/O RAG 28.1 30.0 27.2 49.2 22.9 16.9 10.5 18.6 35.8 11.2 10.1 49.1 16.7 47.4 34.4
Frozen Models
SigLIP2-SO/14@384px 25.2 39.2 20.5 58.5 16.9 7.6 12.4 7.1 6.3 4.7 6.1 62.0 10.1 51.4 17.2
CLIP-L/14@224px 23.8 40.9 18.8 59.6 13.0 6.2 9.6 7.8 4.8 4.7 6.6 56.3 11.1 49.7 14.6
Finetuned Models
VisualBERT 26.1 40.2 24.0 60.9 16.3 8.2 10.0 7.5 5.1 6.4 10.2 51.1 15.1 49.5 24.4
VL-T5 24.5 37.0 20.6 51.1 16.0 7.3 10.2 5.0 4.0 6.8 7.5 52.5 10.3 48.4 21.6
LLaV A-OneVision-7B 21.7 38.8 18.5 62.6 13.2 4.8 8.4 5.7 3.4 5.7 4.5 48.9 11.4 42.2 17.6
SigLIP2-SO/14@384px 28.2 40.2 22.8 64.7 17.8 8.4 9.7 5.8 4.4 4.7 7.7 62.3 13.2 54.8 26.5
CLIP-L/14@224px 24.6 35.4 22.1 61.5 15.7 8.9 9.9 6.5 5.8 5.8 7.8 53.1 11.2 51.7 18.3
CCUB BERTScore
W/O RAG 54.7 55.4 55.6 55.6 55.7 54.2 52.5 54.5 55.4 54.1 54.8 53.5 52.7 56.6 56.8
Frozen Models
SigLIP2-SO/14@384px 54.9 57.2 56.4 57.9 55.2 53.0 55.1 54.8 55.0 52.4 54.0 56.9 52.6 58.7 54.8
CLIP-L/14@224px 54.8 57.6 56.2 57.7 54.8 52.4 54.8 55.1 55.3 52.6 53.7 55.9 52.5 59.4 54.6
Finetuned Models
VisualBERT 54.8 57.1 56.5 57.6 54.9 52.8 54.6 54.9 54.4 52.1 53.8 55.7 52.8 58.5 56.2
VL-T5 54.2 56.7 56.0 58.1 54.9 52.6 53.9 54.0 54.4 52.0 53.4 54.4 52.1 57.8 55.7
LLaV A-OneVision-7B 54.2 56.6 55.5 58.2 54.5 52.1 53.9 54.2 54.6 51.6 53.3 55.0 51.8 58.3 54.9
SigLIP2-SO/14@384px 55.2 57.4 57.1 57.9 55.7 53.2 54.7 55.2 54.9 52.5 54.2 56.8 52.6 58.7 56.7
CLIP-L/14@224px 55.0 57.0 56.5 58.2 54.8 53.2 54.6 55.3 55.3 52.9 54.2 56.0 52.3 59.3 55.4
CCUB CLIPScore
W/O RAG 19.1 19.1 18.8 19.6 18.3 18.2 18.4 18.7 19.2 18.4 18.4 20.2 18.5 20.2 18.7
Frozen Models
SigLIP2-SO/14@384px 19.1 19.3 19.0 19.9 18.9 18.8 19.1 18.8 18.7 18.6 18.5 20.5 18.8 20.0 18.6
CLIP-L/14@224px 19.0 19.2 19.0 20.0 19.0 18.5 19.2 18.8 18.6 18.8 18.4 20.4 18.5 19.6 18.3
Finetuned Models
VisualBERT 19.1 19.1 18.9 20.2 18.9 18.9 19.1 18.9 18.8 18.5 18.6 20.3 18.7 19.9 18.5
VL-T5 19.1 19.4 19.1 19.9 19.0 18.8 19.0 19.1 18.7 18.5 18.7 20.4 18.7 19.9 18.4
LLaV A-OneVision-7B 19.1 19.3 18.9 20.1 18.9 18.7 19.3 18.9 18.6 18.5 18.6 20.4 18.8 19.7 18.6
SigLIP2-SO/14@384px 19.1 19.2 18.9 20.1 18.8 18.6 19.1 19.0 18.5 18.6 18.5 20.4 18.8 19.8 18.5
CLIP-L/14@224px 19.0 19.2 18.9 20.0 19.1 18.6 19.2 18.9 18.7 18.6 18.4 20.1 18.8 19.7 18.6
noticeable improvements over their original counterparts, indicating the effectiveness of retrieval
adaptation. Performance on the cVQA task reveals more nuanced outcomes. While fine-tuned
retrievers generally exhibit large performance improvement compared to non-RAG baselines, certain
countries, such as Spain and Indonesia, experience exhibit diminishing returns. These discrepancies
may stem from the limited presence of culturally representative visual content in the training data.
Although targeted image augmentation strategies were employed to alleviate this imbalance, the
results suggest that data distribution remains a significant bottleneck. Understanding and addressing
19

China India Nigeria Mexico Indonesia Spain Russia−5051015
SigLIP2-SO/14@384px CaSigLIP2 CLIP-L/14@224px CaCLIPAcc. ΔFigure 11: Average improvement across 14 VLMs in different countries with 4 retrievers for cVQA
task.
such cross-cultural performance disparities in multimodal tasks like cVQA will be an important
direction for future work.
China India Nigeria Mexico Korea−10−50510152025
SigLIP2-SO/14@384px CaSigLIP2 CLIP-L/14@224px CaCLIPR.S. Δ
Figure 12: Average improvement across 14 VLMs in different countries with 4 retrievers in cIC task.
K Human Annotation & Evaluation Interface
The annotation interface are shown in Figure 14 and 15. The interface for human evaluation in cIC
task is shown in Figure 13.
🖼  Im age Description Evaluation
Select an image and choose the description you find most accurate and culturally informative.
You have 33 images to annotate.
⬅ PreviousImage: China_102 (2 / 33)
Next ➡
Image
Image: China_102Descriptions
Select the most accurate and culturally informative description:
A traditional Chinese New Year's Eve feast is served, featuring a variety of dishes like braised pork, shrimp, fish, and dumplings, symbolizing prosperity and family
reunion.
A large, traditional reunion dinner is served, featuring a variety of dishes including dumplings, chicken, pork, fish, and other local specialties, symbolizing prosperity
and good fortune.
A table is filled with a variety of dishes, including seafood, meat, rice, and fried items, reflecting a traditional feast.
Submit Choice
Manage app
Figure 13: Human evaluation interface for cIC task.
20

🕹 Annotation Instructions:
Please read the following instruction carefully and finish the trial annotation within 1 mins. The main annotation only starts after you pass the trial.
This task involves providing a culturally relevant image along with a corresponding text caption (2-3 sentences). Some captions include the names of significant cultural entities depicted in the image.
Once start the main annotaion, the annotater will be oﬀered an unique user ID. Please keep t he user ID c arefully.
1. The annotator first needs to verify the correctness of the text caption.
2. In addition to the image and text description, a corresponding Wikipedia article (with its topic and URL) is provided. The annotator must evaluate the article and answer related questions based on three specific aspects (scoring criteria
detailed in the next section).
3. For each image and its corresponding text description, 10 Wikipedia articles are provided for scoring. If omissions are identified during annotation (e.g., missing important related articles), the annotator must provide the titles and links to the
relevant Wikipedia articles.
Scoring Criteria:
1. Does the caption accurately describe the content of the picture?
Yes / No
2. Is the topic of the Wi kipedia article from the same country as the image and its caption? (Cannot be de termined u sually applies to some more general Wikipedia topics)
Yes / No / Cannot be determined
3. Does the topic of the Wi kipedia article align with the category of the content in the image and its caption? ( We de fine categories based o n appearance, items within the same category should share similar physical characteristics, visual
features, or structural forms. For example, Spaghetti and Lasagna are the same category, but Pizza and Spaghetti are not.)
Yes / No
4. Is the topic of the Wi kipedia article mentioned or depicted in the image and its caption?
Yes / No
Task Example:
🖼 Welcome
Enter your user ID:
Like Japan_d5954304-bf98-4e89-bafc-d0
If it's your first time and you don't have a
user ID, please leave it blank. We will
generate one for you later.
Select your country:
Choose an option
Please select your country strictly
according to your nationality. Once you
start, you can't change the country again.
Manage appFigure 14: Human annotation instructions.
Image Des cription:
Sushi, a traditional Japanese delicacy, showcases a harmonious blend of fresh ingredients, meticulous preparation, and cultural artistry. It is a celebrated symbol of Japanese cuisine worldwide.
Image Des cription Verification
❓Does the description (not Wikipedia document here) accurately describe the content of the picture?
Yes
No
Document Title:
SushiLink:
https://en.wikipedia.org/wiki/Sushi
Questions
❓Is topic of the Wikipedia article from the same country as the image and its caption?
Yes
No
Cannot be determined
❓Does topic of the Wikipedia article align with the category of the content in the image and its caption?
Yes
No
❓Is the topic of the Wikipedia article mentioned or depicted in the image and its caption?
Yes
No
NextImage to be annotated:
Current image index: 0 | Pending: 1
Current doc index: 0 | Pending: 1
Manage app
(a)Annotation interface (1).
Image Des cription:
Sushi, a traditional Japanese delicacy, showcases a harmonious blend of fresh ingredients, meticulous preparation, and cultural artistry. It is a celebrated symbol of Japanese cuisine worldwide.
Missing Wiki Documents Form:
Are there any missing wiki documents? If so, please provide the details below. If not, just click the 'Submit' button.
Enter the URL(s). If there are several urls, use comma to split, like url1, url2
https://en.wikipedia.org/wiki/Big_Ben
Enter the Title(s), If there are several titles, use comma to split, like Big Ben1, Big Ben2
Big Ben
SubmitImage to be annotated:
Manage app
(b)Annotation interface (2).
Figure 15: Human annotation interfaces.
L Correlation between Automatic Metrics and Human Judgments
To assess the correlation between human preferences and automatic evaluation metrics, we compute
Kendall’s Tau rank correlation. Human annotations are segmented according to the output chunks
produced by corresponding 14 VLMs with each retriever variant (CaCLIP-based, CLIP-based, and
non-RAG). Within each segment, we calculate the selection winning ratio for each retrieval method,
yielding a human preference vector formed by concatenating these ratios across all evaluation
instances. For the automatic evaluation, we extract BERTScore, CIDEr, ROUGE-L, and CLIPScore
21

Table 10: Prompt example for generating captions used to retrieve the Wiki documents.
Prompt for data collection
GPT-4o
SYSTEM: Default
USER:
Generate a culture related caption given the image with around 2 sentences.
Please include the name of the thing shown in the picture within the caption if
it’s strongly related to the culture. Name the three most culturally relevant
entities in the image and attach the name at the end. Follow the format of:
caption. entity name1; entity name2; entity name3.
for each corresponding retrieval variant and VLM. These scores are similarly concatenated into a
metric-based vector. Finally, we compute Kendall’s Tau between the human and metric vectors to
quantify the consistency between automatic rankings and human judgments.
M Prompts
The prompts that we used to collect the documents and two downstream tasks were designed to
ensure consistency across different models. One prompt example for cultural caption generation is
shown in Table 10. The prompts for both with and without retrieval augmentation for VLMs are
shown in Table 12 and 11.
22

Table 11: Prompt examples without RAG. Multimodal prompt samples with interleaved image are
shown for both CVQA and CCUB tasks.
CVQA task CCUB task
SYSTEM:
You are a helpful assistant.
USER:
Answer the following multiple choice
question. The last line of your
response must be of the following
format: Ánswer: $LETTER ´(without
quotes) where LETTER is one of ABCD.
Question: Which city is famous for
such artwork?
A) Zhejiang Yiwu
B) Jingdezhen, Jiangxi
C) Datong, Shanxi
D) Zibo, ShandongSYSTEM:
You are a helpful assistant.
USER:
Write a concise, one-sentence caption
for the given image. The generated
caption must contain the visual
content and culturally relevant
elements of the image. Avoid
explicit references to the image
itself (e.g., "This image shows...",
"Pictured here is...", "In this
photograph..."). Do not generate
multiple options.
23

Table 12: Prompt examples with RAG. Multimodal prompt samples with interleaved image are
shown for both CVQA and CCUB tasks.
CVQA task CCUB task
SYSTEM:
You are a helpful assistant.
USER:
Answer the following multiple choice
question. The last line of your
response must be of the following
format: Ánswer: $LETTER ´(without
quotes) where LETTER is one of ABCD.
The scope of the question is strictly
limited to the given image. However,
please analyze and incorporate
information from both the image and
the following document to answer the
question.
Document:
Buildings and structures Buildings
about 800 – Borobudur temple in
Java completed. 802 Haeinsa of
Korea, is constructed. Palace of
Charlemagne in Aachen, Carolingian
Empire completed (begun about 790).
The Palatine Chapel still stands. At
Oviedo in the Kingdom of Asturias
Cámara Santa constructed. First
reconstruction of Oviedo Cathedral
begun by Tioda. 815 – Second Temple
of Somnath built in the Pratihara
Empire, India. 816 – Reims Cathedral
begun. 810s – Chapel of San Zeno in
Santa Prassede, Rome decorated. 818 –
Old Cologne Cathedral built....
Question: What is the name of the
square where the cathedral and the
statue of the image are located?
A) Riego Square
B) Trascorrales Square
C) The square of Alfonso II the
Chaste
D) The Fontan squareSYSTEM:
You are a helpful assistant.
USER:
Write a concise, one-sentence caption
for the given image. The generated
caption must contain the visual
content and culturally relevant
elements of the image. Avoid
explicit references to the image
itself (e.g., "This image shows...",
"Pictured here is...", "In this
photograph..."). Do not generate
multiple options. Please consider
the following context:
Architecture of Nigeria was
historically influenced by
environmental conditions as well
as social and cultural factors.
The coming of missionaries and
political changes brought about by
colonialism precipitated a change in
architectural style and utility of
buildings. A Gothic revival style
was adopted for early churches built
in the colony of Lagos. A one or
two storey timber house building
made with pre-fabricated material
components and designed with the
influence of classic antiquity styles
served as mission house for the
missionaries...
24
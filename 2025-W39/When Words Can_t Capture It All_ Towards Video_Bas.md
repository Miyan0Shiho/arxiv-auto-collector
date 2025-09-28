# When Words Can't Capture It All: Towards Video-Based User Complaint Text Generation with Multimodal Video Complaint Dataset

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Kirtan Jain, Vinayak Goyal, Sriparna Saha, Manish Gupta

**Published**: 2025-09-24 10:00:05

**PDF URL**: [http://arxiv.org/pdf/2509.19952v1](http://arxiv.org/pdf/2509.19952v1)

## Abstract
While there exists a lot of work on explainable complaint mining,
articulating user concerns through text or video remains a significant
challenge, often leaving issues unresolved. Users frequently struggle to
express their complaints clearly in text but can easily upload videos depicting
product defects (e.g., vague text such as `worst product' paired with a
5-second video depicting a broken headphone with the right earcup). This paper
formulates a new task in the field of complaint mining to aid the common users'
need to write an expressive complaint, which is Complaint Description from
Videos (CoD-V) (e.g., to help the above user articulate her complaint about the
defective right earcup). To this end, we introduce ComVID, a video complaint
dataset containing 1,175 complaint videos and the corresponding descriptions,
also annotated with the emotional state of the complainer. Additionally, we
present a new complaint retention (CR) evaluation metric that discriminates the
proposed (CoD-V) task against standard video summary generation and description
tasks. To strengthen this initiative, we introduce a multimodal
Retrieval-Augmented Generation (RAG) embedded VideoLLaMA2-7b model, designed to
generate complaints while accounting for the user's emotional state. We conduct
a comprehensive evaluation of several Video Language Models on several tasks
(pre-trained and fine-tuned versions) with a range of established evaluation
metrics, including METEOR, perplexity, and the Coleman-Liau readability score,
among others. Our study lays the foundation for a new research direction to
provide a platform for users to express complaints through video. Dataset and
resources are available at: https://github.com/sarmistha-D/CoD-V.

## Full Text


<!-- PDF content starts -->

When Words Can’t Capture It All: Towards Video-Based User
Complaint Text Generation with Multimodal Video Complaint
Dataset
Sarmistha Das
Indian Institute of Technology Patna
Patna, India
sarmistha1515@gmail.comR E Zera Marveen Lyngkhoi
Indian Institute of Technology Patna
Patna, India
zera_2311ai06@iitp.ac.inKirtan Jain
Indian Institute of Technology Patna
Patna, India
kirtanjain0504@gmail.com
Vinayak Goyal
Indian Institute of Technology Patna
Patna, India
vinayakgoyal2410@gmail.comSriparna Saha
Indian Institute of Technology Patna
Patna, India
sriparna@iitp.ac.inManish Gupta
Microsoft, India
Bengaluru, India
gmanish@microsoft.com
Abstract
While there exists a lot of work on explainable complaint mining,
articulating user concerns through text or video remains a signifi-
cant challenge, often leaving issues unresolved. Users frequently
struggle to express their complaints clearly in text but can easily
upload videos depicting product defects (e.g., vague text such as
‘worst product’paired with a 5-second video depicting a broken
headphone with the right earcup). This paper formulates a new
task in the field of complaint mining to aid the common users’ need
to write an expressive complaint, which is Complaint Description
from Videos (CoD-V) (e.g., to help the above user articulate her com-
plaint about the defective right earcup). To this end, we introduce
ComVID, a video complaint dataset containing 1,175 complaint
videos and the corresponding descriptions, also annotated with the
emotional state of the complainer. Additionally, we present a new
complaint retention (CR) evaluation metric that discriminates the
proposed (CoD-V) task against standard video summary generation
and description tasks. To strengthen this initiative, we introduce
a multimodal Retrieval-Augmented Generation (RAG) embedded
VideoLLaMA2-7b model, designed to generate complaints while
accounting for the user’s emotional state. We conduct a compre-
hensive evaluation of several Video Language Models on several
tasks (pre-trained and fine-tuned versions) with a range of estab-
lished evaluation metrics, including METEOR, perplexity, and the
Coleman-Liau readability score, among others. Our study lays the
foundation for a new research direction to provide a platform for
users to express complaints through video. Dataset and resources
are available at: https://github.com/sarmistha-D/CoD-V.
CCS Concepts
•Computing methodologies →Natural language generation;
Video summarization;Computer vision tasks;Neural networks.
This work is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License.
CIKM ’25, Seoul, Republic of Korea
©2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2040-6/2025/11
https://doi.org/10.1145/3746252.3761576Keywords
Video analysis, complaint generation, video to text,ComVID
ACM Reference Format:
Sarmistha Das, R E Zera Marveen Lyngkhoi, Kirtan Jain, Vinayak Goyal,
Sriparna Saha, and Manish Gupta. 2025. When Words Can’t Capture It All:
Towards Video-Based User Complaint Text Generation with Multimodal
Video Complaint Dataset. InProceedings of the 34th ACM International
Conference on Information and Knowledge Management (CIKM ’25), November
10–14, 2025, Seoul, Republic of Korea.ACM, New York, NY, USA, 9 pages.
https://doi.org/10.1145/3746252.3761576
1 Introduction
In recent years, e-commerce giants have made significant strides
in expanding their online marketing efforts, particularly targeting
rural or underserved areas, e.g., Amazon, Jio Mart, and Flipkart in
India1, Jumia in Nigeria2, Mercado Libre in Latin America3, and
Daraz in South Asia4. Comprehending customer mindset is essential
for engineering advanced interactive complaint analysis systems
that enable significant engagement, especially for individuals who
find it difficult to express their grievances in written form.
Effectively expressing complaints is crucial for building an in-
clusive marketplace, especially for individuals with expressive lan-
guage disorders or limited education—such as farmers, transport
laborers, and retail workers—who are increasingly shopping online.
For organizations, customer reviews and ratings are vital for ana-
lyzing user sentiment toward products [ 2,37]. However, users with
limited literacy often struggle to convey nuanced issues, resulting
in sparse or ambiguous complaints. To address this, prior work
explainable complaint-mining [ 7] used transformer-based mod-
els on Twitter data, combining sentence- and word-level encoders
with attention-guided explanations to classify complaints and infer
root causes. Building on this, a later study [ 5] introduced a multi-
modal vision–language transformer with image-segment encoding
to jointly process video features and transcripts, enabling precise
mapping of nonverbal cues and spoken narratives to complaint
causes.
Although many users can upload videos illustrating these issues,
the impact is often diluted by unclear communication. For example,
one user posted a video review showing broken headphones (see
1https://timesofindia/retail-giants-eye-rural-india-small-towns- to-push-growth
2jumia.com.ng3mercadolibre.com4daraz.comarXiv:2509.19952v1  [cs.CV]  24 Sep 2025

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Sarmistha Das et al.
Figure 1: Comparative Analysis of Proposed Complaint Gen-
eration Task, Traditional Summary Generation, and Video
Description Tasks on User-Uploaded Video Reviews
Fig. 1) alongside a brief text review stating,“I am just done with this
keyboard!!My fingers are now extremely sore”expecting anexchange
or repair. Customer service teams struggle with high complaint
volumes, while traditional video summarization models produce
generic descriptions that miss the core issue, such as“the person is
trying to take a screenshot”. Fine-grained analysis tools also fail to
detect implicit intent, delaying resolution.
Motivation: Previous studies have focused on fine-grained multi-
modal complaint mining, examining specific aspects [ 16,33], but
often fall short in capturing the full intent behind user complaints,
such as specific issues likenon-functioning keyboardsorpoor prod-
uct qualityhighlighted through video content. This limitation poses
significant challenges, especially for individuals with dysgraphia,
who face cognitive barriers in articulating their thoughts in writ-
ten form, or those with constrained linguistic repertoires, thereby
hindering their ability to effectively communicate their concerns
or make informed purchasing decisions. Addressing the pressing
need for an integrative framework that links visual media with
corresponding textual descriptions, we introduce a pioneering com-
plaint generation task,CoD-V, in the domain of complaint mining,
complemented by a Complaint Nature Retention (CR) metric to
substantiate our approach. This task seeks to generate detailed com-
plaint narratives from user-uploaded videos, enhancing customers’
ability to convey their concerns and enabling a more nuanced un-
derstanding of their discomforting experiences.
Research Objectives:1) Our primary objective is to establish a
structured platform that enables inarticulate or busy users to ar-
ticulate their grievances through video content. We also aim to
empower inarticulate consumers and enhance their engagement
with e-commerce platforms. 2) We intend to investigate whetherTable 1: Comparison of complaint datasets and their associ-
ated labels.
Datasets Labels Modality
Complaint Emotion Description Count Text/Image/Video
Complaints [26] ✓×× 3499 Text
Complaint_ESS [34] ✓ ✓× 1971 Text
CESAMARD [33] ✓ ✓× 3962 Text + Image
X-FinCORP [6] ✓ ✓× 6282 Text
VCD [9] ✓×× 450 Video
ComVID(Ours) ✓ ✓ ✓ 1,175 Video
the generated textual descriptions from complaint videos effec-
tively capture the factual elements and the emotional state accu-
rately reflecting the user’s true feelings and actual expectations. 3)
Furthermore, we aim to elucidate how the proposed video-based
complaint generation task fundamentally differs from conventional
video summarization and description tasks.
Contributions:We summarize the key contributions as follows:
1) We introduce a novel task, Complaint Description from Videos
(CoD-V), along with the Complaint Retention (CR) metric, designed
to generate and evaluate comprehensive complaint narratives di-
rectly from video content. This approach addresses a critical gap
in existing methods for managing consumer grievances. 2) In addi-
tion, we present the uniqueComVIDdataset, which serves as an
invaluable resource for researchers and practitioners to advance
studies in video-based complaint analysis and generation. 3) Fur-
thermore, we propose a fine-tuned Retrieval Augmented Generation
(RAG)-embedded VideoLLaMA2-7b model and conduct extensive
experiments on various video-language models. These experiments
explore the impact of incorporating the complainer’s emotional
state, using several quantitative metrics for evaluation.
2 Related Work
The expansion of e-commerce to less literate demographics ne-
cessitates robust complaint-handling frameworks [ 1,13,14]. Re-
cent advancements leverage transformers-based architectures for
complaint severity classification [ 8,18,32], while multimodal ap-
proaches incorporate vision-language integration for nuanced emo-
tion and sentiment analysis [ 25,29,33]. CMA-CLIP [ 24] employs a
multimodal classifier for predefined attribute prediction but strug-
gles with generalization to unseen data. Generative approaches [ 6,
28] reformulate classification as sequence generation task. Resources
such as the Video Dataset of Incidents (VIDI) [ 30], containing 4,534
clips across 43 incident types, enable extracting meaningful nar-
ratives for refined analysis. The ITS series datasets (its4s, its4s2,
its4s3) [ 11] offer subjective video quality evaluations, critical for
assessing the impact of video clarity on user perception. The VCD
dataset [ 9], which gathers e-commerce complaint videos, remains
limited due to a lack of detailed complaint descriptions, limiting its
utility for less literate users.
These existing studies have integrated modalities such as im-
ages, text, and audio to enhance complaint detection via fine-tuning
and advanced attention mechanisms. However, there is no existing
work and resources, as shown in Table 1, to help users write com-
plaints about uploaded videos. To shape the user’s perception, we
propose a novel task alongside an innovative model with complaint
assessment metrics and datasets to advance this field.

Video-Based User Complaint Text Generation CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
Figure 2: Dataset samples: Video along with corresponding
review text, rating, gold label-annotated description, and
emotion label.Note: Emotion, Aspects and Product type labels
are assigned after viewing the complete video and reviewing
the corresponding review text.
3ComVIDDataset Details
Dataset Collection:Initially, a dataset comprising 1200 Amazon
review videos was assembled, focusing on electronic products such
as keyboards, earbuds, mouse, trimmers, and headphones, as well as
non-electronic items, including bags, shoes, and household goods.
Amazon’s extensive reach, particularly in rural regions, provides
a valuable platform for analyzing consumer behavior, capturing
the essential needs of users such as farmers and laborers for prod-
ucts such as trimmers and shoes. To extract 1- and 2-star reviews,
we employed Beautiful Soup for web scraping. The dataset en-
compasses key attributes, including review ID, rating, review text,
aspects, domain name, product name and the associated m3u8 video
links. Next, we transcoded these m3u8 URLs into mp4 format for
enhanced accessibility. To ensure broad applicability, the dataset
includes a balanced distribution of complaint aspects across four
domains: Fashion, Electronics, Household, and Others. Key issues
span Quality, Functionality, Defects, Missing items, Refunds, and
Performance. Fashion items (e.g., shoes, bags) center on quality and
design flaws; electronics (e.g., keyboards, trimmers) exhibit a wider
range including delays; household products report functional de-
fects; while the Others category captures isolated malfunction cases
(e.g., tents). This domain-wise stratification enhances the dataset’s
diversity and supports fine-grained complaint analysis.
After an extensive selection process, 1,175 video samples were incor-
porated into the final dataset as depicted in Fig. 2, each accompanied
by annotated descriptions that capture a range of distinct emotional
nuances. This selection reflects the high volume and popularity
Table 2: Domain wise aspects and product names
Domain Aspects Product Name
Fashion Quality, Functionality, Defective,
Design, Missing, RefundShoes, Bag, Watch, T-
Shirt
Electronic
ProductsQuality, Functionality, Defective,
Missing, Performance, Refund, De-
layMouse, Keyboard,
Headphone, Trimmer
House hold Functionality, Performance, Defec-
tiveBottle, Plates, Plastic
Pots
others Defective Tent, Rain-Coatof Amazon reviews, particularly for electronics, where items such
as headphones, mouse, and keyboards garner significant attention
due to frequent technological advancements. Notably, six samples
featured non-English review texts, further enriching the dataset’s
linguistic diversity.
Dataset Statistics:The dataset comprises 655 reviews on elec-
tronic gadgets, 273 on household items, 202 on fashion items, and
45 in other categories, offering a comprehensive view of consumer
insights across both electronic and non-electronic domains. The fo-
cus on electronics, particularly keyboards, mouse, and headphones,
highlights their significance in the rapidly growing sector, with
evaluations across eight key aspects, including quality, functional-
ity, and defect. These segments represent key areas in an industry
undergoing significant growth and technological innovation. Ta-
ble 2 shows domain-wise aspects and product names. By targeting
these domains, we aim to assess evolving consumer sentiments and
preferences in high-interest areas.
Dataset Annotation: The substantial heterogeneity in complaint
structure, user’s emotional aspects and linguistic patterns, necessi-
tates a paradigm of dynamic adaptability in generating complaint
descriptions. Tailoring some unyielding approach, such as enforc-
ing a fixed word count [ 21], would fail to account for the inher-
ent linguistic diversity and structural nuances. A one-size-fits-all
strategy can compromise the quality and contextual relevance of
generated descriptions, often failing to capture critical details. The
collected datasets contain videos but lack adequate textual descrip-
tions. Therefore, our approach prioritizes flexibility in both the
length and structure of descriptions, allowing us to effectively con-
vey the unique characteristics, context, and information density
inherent to each video. This adaptability ensures that the gener-
ated descriptions remain meaningful and capture the essence of the
original content. Given these challenges, manual annotation was
required to provide accurate text descriptions for the videos.
Phase-1:A team of five expert linguists collaborated to generate
high-quality complaint descriptions and assign emotional labels.
The team included one doctoral-level annotator (Category A), two
annotators with master’s degrees (Category B), and two undergradu-
ate annotators (Category C), all proficient in Hindi and English. The
Category Aannotator initially provided 50 gold-standard samples
as ground truth for reference. Based on these, oneCategory Band
oneCategory Cannotator generated descriptions for half of the
dataset, while the remaining two annotators rigorously reviewed
all descriptions to ensure accuracy, coherence, and adherence to
established guidelines. Annotators cross-validated samples they
did not annotate, assessing factual accuracy (alignment with video
content) and coherence. Only samples with an acceptance tag were
included as gold-standard descriptions.
Phase-2:Following the description generation,Category Aand
Bannotators independently assigned emotional labels, resolving
discrepancies through a consensus process. The process achieved a
Fleiss’ kappa score of 0.64, indicating substantial inter-annotator
agreement and ensuring annotation reliability. Annotators vali-
dated instances against user-provided content, refined annotations
for clarity, and were compensated at $0.50 per sample, resulting in a
robust and unbiased dataset for complaint description and emotion
analysis.

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Sarmistha Das et al.
4 Methodology for Generating Complaint Text
Description
Problem Statement:The video complaint dataset comprises videos
𝑉∈R𝐹×3×𝑊×𝐻and corresponding textual reviews 𝑅={𝑟 1,𝑟2,...,𝑟𝑚},
where𝐹is the number of sampled frames, 𝑚is the number of words
and𝑊,𝐻 are the frame width and height, respectively. Each 𝑅is as-
sociated with emotion labels 𝑒∈{ dissatisfaction, blame, frustration,
disappointment}, reflecting the user’s emotional state. Reviews 𝑅
are often vague, hence we aim to develop a model Fthat simulta-
neously processes both the video 𝑉with textual prompt 𝑃and the
associated emotion labels to generate a coherent and contextually
accurate descriptive complaint 𝑌, formulated as 𝑌=F(𝑉,𝑃,𝑒) .
The output 𝑌should accurately encapsulate the user’s complaint
by encoding the user’s emotional state. To address theCoD-Vtask,
the proposed architecture as depicted in Figure 3 is designed in the
following two steps.
Step 1: Developing Multimodal Retrieval Augmented Gener-
ation (MR):.The MR framework is grounded on a large-scale set
of publicly accessible Amazon product reviews5, utilizing 75.26 mil-
lion reviews where each user and product has at least five reviews.
For complaint analysis, the dataset is further filtered based on three
criteria: reviews with a rating of 1, text exceeding 150 characters
to ensure detailed complaints, and at least one image per review
to capture multimodal cues. The filtering steps ensure the dataset
focuses on genuine complaints with adequate contextual and visual
detail, resulting in a refined collection of negative reviews, each
with comprehensive text and associated images
To extract multimodal embeddings, we use CLIP [ 27] for both text
and image components. The textual content 𝑇of a review, consisting
of the title𝑅𝑡𝑠and main text 𝑅𝑡, is processed by CLIP’s pretrained
text encoder, yielding the text embeddinge 𝑡=TextEncoder(𝑇)∈
R𝑑. For image embeddings, each review’s associated images are pro-
cessed by CLIP’s image encoder to generatee 𝑖𝑘=ImageEncoder(Img𝑘)
∈R𝑑, and for multiple images their embeddings are averaged:
¯e𝑖=1
𝑛Í𝑛
𝑘=1e𝑖𝑘. The final multimodal embedding is obtained by
combining the text and image embeddings:e mm=1
2(e𝑡+¯e𝑖). More-
over, for efficient similarity searche mmare further aggregated
into matrix 𝐸and indexed using FAISS [ 19] defined as Index=
FAISSIndexFlatL2(𝐸).
We encode the review videos by employing GMFlow [ 38] and
extract four keyframes, further process them with CLIP’s image en-
coder to obtain embeddingse 𝑓1,e𝑓2,e𝑓3,e𝑓4, followed by aggregating
them intoq video=1
4Í4
𝑖=1(e𝑓𝑖). Subsequently, we take correspond-
ing product aspects and pass through CLIP’s text encoder to produce
etextual , resulting into the final query embedding as:q =𝛼qvideo+
(1−𝛼) etextual . The top𝑘nearest neighbors {e(𝑖1)
mm,..., e(𝑖𝑘)
mm}are
retrieved from the FAISS index, providing retrieval-augmented con-
text for the query.
Step 2: Supervised Fine-Tuning and Generation:VideoLLaMA2-
7b is fine-tuned on supervised video-text pairs. During inference,
the query prompt is enhanced with the user’s emotional state (e.g.,
frustration) and the top- 𝑘retrieved complaint reviews from MR. The
model processes this enriched input to generate a refined output,
5https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2effectively integrating textual, visual, and domain-specific context.
Fig. 3 illustrates the overall framework.
5 Experimental Results and Analysis
We describe our experimental setup, comparative baselines, empiri-
cal results, and performance analysis of fine-tuned VideoLLaMA2-
7b+MR for the novelCoD-Vtask. We conduct qualitative evalua-
tions to scrutinize result quality and discuss observed performances.
Our research addresses the following questions:
RQ1:Can the fine-tuned VideoLLaMA2-7b+MR surpass conven-
tional baselines and state-of-the-art methods in technical perfor-
mance, serving as a superior method for theCoD-Vtask?
RQ2:Can providing emotion as extra input help? How do various
models perform based on human perception?
RQ3:How doesCoD-Vtask differ from traditional summary gen-
eration and video description?
RQ4:What are the societal benefits of the proposed work, and how
extensive is the model’s applicability across diverse NLP tasks?
We use the following multimodal backbone models: fine-tuned
VideoLLaVA2-7b [ 41], the fine-tuned BLIP-VQA-BASE [ 20], Qwen2-
VL-7b [ 36], Gemma3-12b [ 35] and the integration of LSTM networks
with VGG16 [31] and ResNet50 [12].
5.1 Evaluation Metrics
We propose a standard evaluation metric to assess the nature of the
complaint retained (CR) in the generated output, focusing on senti-
ment score, emotion detection, and aspect identification, compared
against the gold label aspects.
Sentiment score: To analyze the sentiment, we utilize Vader score [ 15].
We first calculate Vader score 𝑆Vader,𝑖 for a sentence in the gener-
ated inferences. This score ranges from -1 to 1, so we normalize
it to a range from 0 to 1 as: 𝑆N Vader,𝑖 =𝑆Vader,𝑖+1
2. Next, the average
normalized sentiment score for each 𝑘-th sample with 𝑁𝑘sentences
is calculated as 𝑆N Vader,𝑘 =1
𝑁𝑘Í𝑁𝑘
𝑖=1𝑆N Vader,𝑖 . Finally, we compute
the overall average Vader sentiment score 𝑆N Vader as the average
across all inference samples.
Emotional intensityis a critical component of complaints. We
analyze predicted text using Text2Emotion library, which extracts
scores for four key emotions: Happy (H), Angry (A), Surprise (S),
and Fear (F). Each emotion score lies in [0,1], and Emotion Score
(ES) for a sample 𝑘is computed as the average of all emotion scores:
𝐸𝑆𝑘=𝐻𝑘+𝐴𝑘+𝑆𝑘+𝐹𝑘
4. This provides an aggregate measure of emo-
tional intensity in the generated complaint.
Aspect Score (AS)measures how effectively a generated complaint
retains key aspects from the original complaint. Since complaints
often focus on specific issues (e.g., product quality, product defects),
it is crucial to ensure that these ground-truth aspects are preserved
in the predicted text. To evaluate this, we use GPT-4 as a classifier.
For each manually annotated ground-truth aspect, we provide the
following prompt: “Is the aspect ‘predicted_aspect’ present in the
given ‘text’? Answer with only ‘Yes’ or ‘No’.” GPT-4 responds with
either Yes or No, which is mapped to a binary score of 1 or 0
respectively. The final Aspect Score (AS) for the 𝑘-th sample is
computed as the fraction of ground-truth aspects that are retained
in the predicted text: 𝐴𝑆𝑘=Í𝑁𝑘
𝑖=11(Aspect𝑖∈𝑇𝑝)
𝑁𝑘where𝑁𝑘is the total

Video-Based User Complaint Text Generation CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
Figure 3: Architectural overview of the multimodal Retrieval-Augmented Generation (RAG) model integrating VideoLLaMA2-
7B for generating complaints from user-uploaded videos. The model leverages the Amazon product review dataset containing
review text, image pairs, which serves as the superset of product reviews included in theComVIDdataset.
number of ground-truth aspects, and1 (·)is an indicator function.
Aspect score provides a structured method for assessing how well
the generated complaint preserves essential details compared to
the original.
Complaint retention (CR) scoreis calculated as CR score=
𝑆N Vader+𝐸𝑆+𝐴𝑆
3. Thus, the complaint retention score combines the
three metrics (normalized sentiment, emotion, and aspect presence)
into a single metric, and gives a balanced evaluation of the nature
of the complaint retained.
Standard NLG metrics: Further, we evaluate model performance
using 8 standard metrics: BLEU [ 23], ROUGE [ 22], BERTScore [ 39],
MoverScore [ 40] METEOR [ 3], Perplexity [ 17], Flesch Reading Ease
[10], and Coleman-Liau Index [ 4]. BLEU and ROUGE measure n-
gram similarity, BERTScore (BS) and MoverScore (MoS) evaluate
semantic alignment via contextual embeddings, METEOR (MeS)
captures precision-recall with synonym flexibility, Perplexity (PS)
quantifies fluency, while Flesch Reading Ease (FRES) and Coleman-
Liau (CLIS) assess the readability of generated content.
Experimental Configuration:To ensure a fair comparison among
all models, both VideoLLAMA2-7b and its LoRA fine-tuned variant
were trained with these hyperparameters: LoRA rank set to 64,
LoRA alpha at 128, a batch size of 16, and a learning rate (LR) of
2×10−4with a cosine learning rate scheduler. We use LR of2 ×10−5
for multi-modal projector layer. We trained for 10 epochs.
5.2 Results and Discussions
Answer to RQ1 Main Results:Evidently, the fine-tuned VideoLLaMA2-
7b+MR model surpasses conventional baselines and state-of-the-art
methods on theCoD-Vtask. As shown in Table 3, it consistently out-
performs across both with emotion and without emotion settings,excelling in ROUGE, BLEU, BERTScore, and FRES, ensuring high
fluency, contextual relevance, and nuanced emotional expression
in complaint narratives. Compared to BLIP-VQA-BASE, QWEN2-
VL-7b, Gemma3-12b, and LSTM-based models, it retains complaint-
specific features more effectively, while lower PS, CLIS, and MoS
scores indicate superior coherence and reduced semantic drift.
These gains stem from the Multimodal Retrieval-Augmented
Generation (RAG) module, which dynamically incorporates rele-
vant visual-textual evidence, yielding a 3–4% improvement over
fine-tuned VLMs alone. Unlike static architectures, RAG retrieves
semantically aligned instances to improve factual grounding and
visual-linguistic alignment, enabling fine-grained disambiguation
(e.g., distinguishing typing errors from hardware faults). The MR
further injects contextual cues (e.g., non-functionality, quality degra-
dation) from the Amazon review dataset, enhancing the model’s
ability to generalise across diverse complaint scenarios and rein-
forcing its robustness inCoD-V.
Answer to RQ2 Performance Observations:Table 3 underscores the
critical role of emotion consideration across all models. Emotion in-
jection enhances contextual relevance and ensures a consistent tone,
significantly improving the quality of complaint descriptions gener-
ated by the models. Notably, models incorporating emotion-based
features outperformed the emotion-agnostic setup, highlighting the
strength of context-specific model selection. Furthermore, we eval-
uated the models’ performance through qualitative assessments,
error analysis, and human evaluations, ensuring a thorough and
unbiased validation process.
1. Qualitative Analysis:Table 5 highlights the strengths of MR
+ VideoLLaMA2-7b, particularly its ability to generate complaints
reflecting negative emotions. In theWithout Emotioncategory,

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Sarmistha Das et al.
Table 3: Main results: Proposed framework vs popular fine-tuned visual language models; (R1-RL): ROUGE variants; (B1-B3):
BLEU variants; BS: BERTScore; FRES: Flesch Reading Ease Score, CLIS: Coleman-Liau Index Score; PS: Perplexity Score, MoS:
Mover Score and MeS: Meteor Score. Best scores are highlighted in bold. Second best are underlined.
Setting Model Name R1↑ R2↑ RL↑ B1↑ B2↑ BL↑ BS↑ FRES↑ CLRS↓ PS↓ MeS↑ MoS↑
No EmotionLSTM+VGG16 0.41 0.3 0.38 0.62 0.53 0.48 0.72 66.47 5.42 86.36 0.42 -0.39
LSTM+ResNet50 0.42 0.32 0.39 0.64 0.55 0.48 0.72 65.84 5.57 87.95 0.42 -0.37
BLIP-VQA-BASE 0.5 0.34 0.45 0.6 0.53 0.47 0.91 81.87 5.95 80.28 0.39 0.07
QWEN2-VL-7b 0.45 0.32 0.42 0.61 0.52 0.48 0.90 70.48 8.91 70.12 0.30 0.05
Gemma3-12b 0.46 0.33 0.45 0.61 0.52 0.51 0.90 73.45 9.11 78.44 0.41 0.08
VideoLLaVA2-7b 0.5 0.38 0.49 0.65 0.54 0.53 0.91 76.9 7.55 120.4 0.44 0.115
VideoLLaMA2-7b 0.52 0.38 0.50 0.64 0.56 0.53 0.90 77.01 7.41 96.29 0.43 0.12
BLIP-VQA-BASE+ MR 0.52 0.36 0.47 0.63 0.55 0.49 0.91 82.11 7.98 82.76 0.41 0.09
QWEN2-VL-7b+MR 0.47 0.35 0.43 0.63 0.54 0.5 0.90 72.62 10.11 72.55 0.33 0.08
Gemma3-12b+MR 0.48 0.39 0.47 0.63 0.54 0.52 0.89 79.21 9.14 80.44 0.43 0.09
VideoLLaVA2-7b+MR 0.53 0.39 0.49 0.63 0.55 0.54 0.92 78.11 7.09 121.14 0.44 0.14
VideoLLaMA2-7b+MR
(Proposed)0.55 0.40 0.51 0.65 0.57 0.53 0.91 78.07 7.26 95.39 0.46 0.14
EmotionLSTM+VGG16 0.43 0.31 0.4 0.63 0.54 0.49 0.73 67.72 5.32 88.14 0.43 -0.37
LSTM+ResNet50 0.43 0.33 0.42 0.65 0.56 0.5 0.74 69.35 5.62 89.03 0.44 -0.36
BLIP-VQA-BASE 0.51 0.36 0.47 0.62 0.55 0.5 0.91 78.83 6.8 118.56 0.4 0.11
QWEN2-VL-7b 0.48 0.37 0.5 0.6 0.54 0.49 0.91 71.38 8.93 71.23 0.31 0.06
Gemma3-12b 0.49 0.4 0.48 0.64 0.55 0.53 0.91 76.22 9.15 80.45 0.44 0.1
VideoLLaVA2-7b 0.54 0.42 0.53 0.64 0.55 0.57 0.92 77.15 7.29 116.2 0.44 0.16
VideoLLaMA2-7b 0.56 0.42 0.49 0.64 0.56 0.57 0.92 79.15 7.01 111.1 0.47 0.19
BLIP-VQA-BASE+ MR 0.53 0.38 0.49 0.64 0.57 0.52 0.91 77.85 6.82 118.58 0.42 0.13
QWEN2-VL-7b+MR 0.50 0.39 0.52 0.62 0.56 0.51 0.91 71.4 8.95 71.25 0.33 0.08
Gemma3-12b+MR 0.51 0.42 0.50 0.66 0.57 0.55 0.92 79.24 9.17 80.47 0.46 0.12
VideoLLaVA2-7b+MR 0.57 0.46 0.54 0.68 0.61 0.58 0.92 78.14 7.35 114.51 0.51 0.21
VideoLLaMA2-7b+MR
(Proposed)0.59 0.47 0.56 0.69 0.63 0.59 0.93 79.58 7.27 97.16 0.51 0.24
Table 4: Comparison across tasks: ProposedCoD-Vvs Summary Generation (SG) and Video Description (VD) on popular visual
language models in zero-shot setting; VS: Vader Score, CR: Complaint Retention. Best scores are highlighted in bold.
Model Name Task R1↑ R2↑ RL↑ B1↑ B2↑ BL↑ BS↑ FRES↑ CLRS↓ PS↓ MeS↑ MoS↑VS↑ CR↑
VideoLLaVA2-7b SG 0.21 0.02 0.17 0.55 0.43 0.32 0.85 68.51 7.58 15.05 0.14 -0.1 0.06 0.41
VD 0.15 0.02 0.12 0.27 0.23 0.19 0.84 61.10 8.60 34.40 0.15 -0.13 0.16 0.48
CoD-V 0.31 0.01 0.24 0.50 0.42 0.39 0.89 57.80 10.80 34.30 0.16 0.00 -0.14 0.57
GPT4O SG 0.21 0.02 0.15 0.54 0.44 0.33 0.84 55.60 10.30 41.20 0.14 -0.10 0.30 0.45
VD 0.14 0.02 0.10 0.42 0.33 0.23 0.83 52.80 9.60 31.04 0.12 -0.13 0.22 0.42
CoD-V 0.28 0.06 0.19 0.59 0.49 0.39 0.86 51.50 10.80 38.04 0.16 -0.04 -0.26 0.56
GeminiFlash SG 0.22 0.03 0.17 0.49 0.40 0.31 0.84 54.60 10.50 39.11 0.16 -0.08 -0.01 0.47
VD 0.22 0.03 0.16 0.50 0.41 0.31 0.85 54.30 10.60 42.40 0.15 -0.08 -0.01 0.48
CoD-V 0.28 0.07 0.19 0.53 0.45 0.36 0.86 55.50 10.60 42.50 0.20 0.00 -0.24 0.55
Proposed SG 0.32 0.11 0.26 0.53 0.44 0.36 0.88 78.40 5.80 42.20 0.24 -0.01 0.05 0.59
VD 0.25 0.08 0.20 0.39 0.34 0.28 0.86 70.10 7.40 41.10 0.22 -0.04 0.14 0.58
CoD-V 0.39 0.27 0.36 0.49 0.43 0.39 0.89 79.91 5.50 41.60 0.34 -0.01 -0.28 0.62
Table 5: Qualitative Analysis for a sample. The section highlighted in blue indicates the observed performance improvements.
Ground truth: “The user wants to convey about the complaint of mouse. The user claims that the mouse is defective. He has
received a damaged product. The scrolling button is not working properly.”
Video BLIP-VQA-BASE VideoLLaVA2-7b VideoLLaMA2-7b+MR
Without Emotionthe user wants to convey about The user wants to convey about The user wants to convey about
the complaint of mouse. the the complaint of mouse is old. the complaint of mouse. The user
user is saying that the mouse The user has received an old product. shows the quality of the mouse
scroller is not working and the Which is not having good quality. which is not good by placing
scroll button is not working. The scrolling button is not working it in his hand. The sensor is
as well. not workingWith Emotionthe user wants to convey about The user wants to convey about The user wants to convey about
the complaint of mouse. the the complaint of mouse. The user the complaint of mouse.The user
user shows the quality of the has dissatisfaction regarding the claims that the mouse is defective
mouse which is not good.the product durability. and he has received a damaged
user has shown this by taking product. The scrolling button is
the mouse in his hand. not working properly.
MR + VideoLLaMA2-7b effectively captures the user’s compre-
hensive concerns by detailing both the poor quality and the non-
functionality of the mouse sensor, aligning closely with the ground
truth, which emphasizes the product’s defectiveness. In contrast,
BLIP-VQA-BASE provides a basic complaint identification without
addressing quality, while VideoLLaVA2-7b mentions the product’s
age but lacks specificity regarding functional issue aspects. In theWith Emotioncategory, VideoLLaMA2-7b+MR excels by articu-
lating theuser’s frustration over the defective mouse and its poor
quality, showcasing a nuanced understanding of emotional expres-
sion. BLIP-VQA-BASE falls short by offering a superficial quality
assessment, and VideoLLaVA2-7b acknowledges dissatisfaction re-
garding durability but lacks a direct emotional connection. Overall,

Video-Based User Complaint Text Generation CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
01 Gemini-Flash Gemma3-12b QWEN2-VL-7b VideoLLaMA2-7b
00.20.40.60.81Adequacy
Fluency
Emotion
EncodingRelevanceVisual Content
Alignment
00.20.40.60.81Adequacy
Fluency
Emotion
EncodingRelevanceVisual Content
Alignment
Model Performance with Multimodal RAG Model Performance without Multimodal RAGVideoLLaVA2-7b BLIP-VQA
-BASE
Figure 4: Illustration of Human Evaluation among popular VLM vs the proposed model.
Table 6: Error Analysis for a sample. Red text indicate ob-
served performance deficiencies.
Output from MR + VideoLLaMA2 -7b: The user wants to complain that the keyboard is not
functioning properly. Despite multiple attempts to press the keys, they remain unresponsive and
instead automatically type other. The user is disappointed with Amazon for selling defective items
and is unable to contact customer care for a refund.
Ground Truth: The user wants to convey the complaint that the keyboard is not working. The USB
jack is also not working. Wrong typing occurs on the screen. words and trigger the escape button.
VideoLLaMA2-7b+MR demonstrates superior capability in integrat-
ing detailed observations of product quality and user sentiment,
positioning it as a more robust model compared to its counterparts.
2. Error Example:Table 6 shows an example where the MR +
VideoLLaMA2-7b response introduces additional details not present
in the ground truth, leading to an inaccurate representation of the
user’s complaint. While the ground truth focuses on specific issues,
namely, a non-functional keyboard, malfunctioning USB jack, and
unintended typing on the screen, VideoLLaMA2-7b+MR includes
information that the user is “disappointed with Amazon” and “unable
to contact customer care for a refund.” These additions misinterpret
the original complaint and could convey a sense of frustration or
escalation not initially expressed by the user. These inaccuracies
result in a response that overstates the situation, detracting from
the precision required to represent the user’s original message.
3. Human Evaluation:For a comprehensive evaluation, we con-
ducted a human assessment withCategory Bannotators on 80
randomly sampled test instances, using Information Preservation
Ratings (IPR) across Adequacy, Fluency, Emotion Encoding, Rele-
vance, and Visual Content Alignment. Scores ranged from 1 to 5,
based on the extent of preservation. As shown in Figure. 4, BLIP-
VQA-BASE underperformed in Visual Content Alignment, while
VideoLLaVA2-7b and VideoLLaMA2-7b+MR excelled. The proposed
VideoLLaMA2-7b+MR outperformed both in emotion encoding and
capturing nuanced emotional contexts more effectively. Average
scores were cross-verified byCategory Aannotators for robustness.
Answer to RQ3 (Uniqueness ofCoD-Vtask):TheCoD-Vtask dif-
fers significantly from traditional summary generation and videodescription tasks, as highlighted in Table 4. We prompt multiple
models for 3 tasks: ProposedCoD-V, summary generation (SG) and
Video Description (VD). Unlike conventional tasks, which primarily
rely on traditional lexical analysis metrics, theCoD-Vtask intro-
duces a unique focus on complaint-specific characteristics, such
as the Vader Score, which is crucial for capturing the sentimen-
tal essence of complaints. This metric, along with the retention
characteristics defined for complaint nature (CR), is central to the
task. InCoD-Vtask, the proposed VideoLLaMA2-7b+MR model has
demonstrated better performance with more negative Vader Score,
while traditional tasks such as summary generation and video de-
scription tend to yield more positive sentiment scores. Models such
as GeminiFlash, GPT4O, and VideoLLaMA2-7b+MR show competi-
tive results, but there remains a significant performance gap when
evaluated on task-specific criteria. WithCoD-Vmodels consistently
achieving high Complaint Retention (CR) scores, the complaint
generation task clearly demonstrates its optimal alignment with
the complaint mining domain while justifying Figure 1.
Answer to RQ4 (Societal benefits and applicability):The proposed
work enhances video-to-text generation for complaint articulation,
aiding less literate users in expressing grievances on e-commerce
platforms such a Amazon. It processes unstructured video reviews
to extract key issues (e.g., product defects), facilitating efficient
resolutions. The VideoLLaMA2-7b+MR model, applied toCoD-V,
extends to diverse NLP tasks, including real-time patient issue re-
porting in healthcare, automated movie trailer summarization, and
toxicity detection in online content, demonstrating broad applica-
bility in structured text generation from multimodal inputs.
6 Conclusion
This paper presentsComplaint Description from Videos (CoD-V),
aimed at addressing communication challenges faced by less literate
consumers on e-commerce platforms. We introduce theComVID
dataset with 1,175 annotated complaint videos across 4 domains
and 8 complaint aspects. A fine-tuned multimodal RAG-configured
VideoLLaMA2-7b model is proposed for generating descriptive
complaints, outperforming existing methods in transparency. To
further validate the uniqueness of theCoD-Vtask, we introduce
the Complaint Retention (CR) score, coupled with the Vader score
(VS), distinguishing it from summarization and video description
tasks. Additionally, emotion and non-emotion attribute prompting
strategies are employed to enrich the diversity of complaint de-
scriptions, ultimately improving customer service responsiveness
for e-commerce.

CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea Sarmistha Das et al.
7 Acknowledgement
The authors sincerely acknowledge the invaluable contributions of
the dataset annotators, Tannu, Pavnee, and Jheel, for their consis-
tent dedication and meticulous effort in the annotation process.
8 GenAI Usage Disclosure
Generative AI tools (ChatGPT, Perplexity) were used in a limited ca-
pacity, solely for minor grammatical refinement and formatting. The
core research, including conceptualization, experimental design,
implementation, analysis, and validation, was conducted entirely by
the authors. All technical contributions, data curation, experiments,
and interpretations remain the exclusive work of the authors.
References
[1]Mustafa Seref Akin. 2024. Enhancing e-commerce competitiveness: A compre-
hensive analysis of customer experiences and strategies in the Turkish market.
Journal of Open Innovation: Technology, Market, and Complexity10, 1 (2024),
100222.
[2]Arwa SM AlQahtani. 2021. Product sentiment analysis for amazon reviews.
International Journal of Computer Science & Information Technology (IJCSIT) Vol
13 (2021).
[3]Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for
MT evaluation with improved correlation with human judgments. InProceedings
of the acl workshop on intrinsic and extrinsic evaluation measures for machine
translation and/or summarization. 65–72.
[4]Meri Coleman and Ta Lin Liau. 1975. A computer readability formula designed
for machine scoring.Journal of Applied Psychology60, 2 (1975), 283.
[5]Sarmistha Das, Basha Mujavarsheik, RE Zera Lyngkhoi, Sriparna Saha, and Alka
Maurya. 2025. Deciphering the complaint aspects: Towards an aspect-based
complaint identification model with video complaint dataset in finance. In2025
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE,
7195–7204.
[6]Sarmistha Das, Apoorva Singh, Raghav Jain, Sriparna Saha, and Alka Maurya.
2023. Let the Model Make Financial Senses: A Text2Text Generative Approach
for Financial Complaint Identification. InPacific-Asia Conference on Knowledge
Discovery and Data Mining. Springer, 58–69.
[7]Sarmistha Das, Apoorva Singh, Sriparna Saha, and Alka Maurya. 2024. Negative
review or complaint? Exploring interpretability in financial complaints.Ieee
Transactions on Computational Social Systems11, 3 (2024), 3606–3615.
[8]Sarmistha Das, Apoorva Singh, Sriparna Saha, and Alka Maurya. 2024. Negative
Review or Complaint? Exploring Interpretability in Financial Complaints.IEEE
Transactions on Computational Social Systems(2024), 1–10. doi:10.1109/TCSS.
2023.3338357
[9]Rishikesh Devanathan, Apoorva Singh, AS Poornash, and Sriparna Saha. 2024.
Seeing Beyond Words: Multimodal Aspect-Level Complaint Detection in Ecom-
merce Videos. InACM Multimedia 2024.
[10] James N Farr, James J Jenkins, and Donald G Paterson. 1951. Simplification of
Flesch reading ease formula.Journal of applied psychology35, 5 (1951), 333.
[11] Institute for Telecommunication Sciences. 2024. Video Quality Research Data.
https://its.ntia.gov/research/qoe/video-quality-research/data Accessed: 2024-11-
01.
[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual
learning for image recognition. InProceedings of the IEEE conference on computer
vision and pattern recognition. 770–778.
[13] YEHIA HELMY, MERNA ASHRAF, and LAILA ABDELHAMID. 2024. A DE-
CISION SUPPORT MODEL TO IMPROVE COMPLAINT HANDLING IN E-
COMMERCE TO ENHANCE CUSTOMER TRUST.Journal of Theoretical and
Applied Information Technology102, 11 (2024).
[14] Yanrong Huang, Zhiyi He, Han Lv, and Jian Min. 2024. Research on Mining
Negative Online Reviews on E-commerce Platforms Based on Social Network
Analysis and LDA Model. InIntelligent Management of Data and Information
in Decision Making: Proceedings of the 16th FLINS Conference on Computational
Intelligence in Decision and Control & the 19th ISKE Conference on Intelligence
Systems and Knowledge Engineering (FLINS-ISKE 2024). World Scientific, 177–185.
[15] Clayton Hutto and Eric Gilbert. 2014. Vader: A parsimonious rule-based model
for sentiment analysis of social media text. InProceedings of the international
AAAI conference on web and social media, Vol. 8. 216–225.
[16] Raghav Jain, Apoorva Singh, Vivek Gangwar, and Sriparna Saha. 2023. AbCoRD:
Exploiting multimodal generative approach for Aspect-based Complaint and
Rationale Detection. InProceedings of the 31st ACM International Conference on
Multimedia. 8571–8579.[17] Fred Jelinek, Robert L Mercer, Lalit R Bahl, and James K Baker. 1977. Perplexity—a
measure of the difficulty of speech recognition tasks.The Journal of the Acoustical
Society of America62, S1 (1977), S63–S63.
[18] Mali Jin and Nikolaos Aletras. 2021. Modeling the severity of complaints in social
media.arXiv preprint arXiv:2103.12428(2021).
[19] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs.IEEE Transactions on Big Data7, 3 (2019), 535–547.
[20] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Boot-
strapping Language-Image Pre-training with Frozen Image Encoders and Large
Language Models. InICML.
[21] Piji Li, Lidong Bing, and Wai Lam. 2017. Reader-Aware Multi-Document Sum-
marization: An Enhanced Model and The First Dataset. InProceedings of the
Workshop on New Frontiers in Summarization. 91–99.
[22] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries.
InText Summarization Branches Out. Association for Computational Linguistics,
Barcelona, Spain, 74–81. https://aclanthology.org/W04-1013
[23] Chin-Yew Lin and Franz Josef Och. 2004. ORANGE: a Method for Evaluating
Automatic Evaluation Metrics for Machine Translation. InCOLING 2004: Proceed-
ings of the 20th International Conference on Computational Linguistics. COLING,
Geneva, Switzerland, 501–507. https://www.aclweb.org/anthology/C04-1072
[24] Huidong Liu, Shaoyuan Xu, Jinmiao Fu, Yang Liu, Ning Xie, Chien-Chih Wang,
Bryan Wang, and Yi Sun. 2021. Cma-clip: Cross-modality attention clip for
image-text classification.arXiv preprint arXiv:2112.03562(2021).
[25] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik
Cambria, and Rada Mihalcea. 2018. Meld: A multimodal multi-party dataset for
emotion recognition in conversations.arXiv preprint arXiv:1810.02508(2018).
[26] Daniel Preotiuc-Pietro, Mihaela Gaman, and Nikolaos Aletras. 2019. Automati-
cally Identifying Complaints in Social Media. InProceedings of the 57th Conference
of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July
28- August 2, 2019, Volume 1: Long Papers, Anna Korhonen, David R. Traum,
and Lluís Màrquez (Eds.). Association for Computational Linguistics, 5008–5019.
doi:10.18653/v1/p19-1495
[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning. PMLR, 8748–8763.
[28] Kalyani Roy, Pawan Goyal, and Manish Pandey. 2021. Attribute value generation
from product title using language models. InProceedings of The 4th Workshop on
e-Commerce and NLP. 13–17.
[29] Tulika Saha, Apoorva Upadhyaya, Sriparna Saha, and Pushpak Bhattacharyya.
2021. Towards sentiment and emotion aided multi-modal speech act classification
in twitter. InProceedings of the 2021 conference of the North American chapter
of the association for computational linguistics: Human language technologies.
5727–5737.
[30] Duygu Sesver, Alp Eren Gençoğlu, Çağrı Emre Yıldız, Zehra Günindi, Faeze
Habibi, Ziya Ata Yazıcı, and Hazım Kemal Ekenel. 2022. VIDI: A Video Dataset of
Incidents. In2022 IEEE 14th Image, Video, and Multidimensional Signal Processing
Workshop (IVMSP). IEEE, 1–5.
[31] Karen Simonyan and Andrew Zisserman. 2014. Very deep convolutional networks
for large-scale image recognition.arXiv preprint arXiv:1409.1556(2014).
[32] Apoorva Singh, Soumyodeep Dey, Anamitra Singha, and Sriparna Saha. 2022. Sen-
timent and emotion-aware multi-modal complaint identification. InProceedings
of the AAAI Conference on Artificial Intelligence, Vol. 36. 12163–12171.
[33] Apoorva Singh, Vivek Gangwar, Shubham Sharma, and Sriparna Saha. 2023.
Knowing what and how: a multi-modal aspect-based framework for complaint
detection. InEuropean Conference on Information Retrieval. Springer, 125–140.
[34] Apoorva Singh, Arousha Nazir, and Sriparna Saha. 2022. Adversarial Multi-
task Model for Emotion, Sentiment, and Sarcasm Aided Complaint Detection. In
Advances in Information Retrieval: 44th European Conference on IR Research, ECIR
2022, Stavanger, Norway, April 10–14, 2022, Proceedings, Part I. Springer, 428–442.
[35] Gemma Team. 2025. Gemma 3. (2025). https://goo.gle/Gemma3Report
[36] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin
Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du,
Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang
Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the
World at Any Resolution.arXiv preprint arXiv:2409.12191(2024).
[37] Sobia Wassan, Xi Chen, Tian Shen, Muhammad Waqar, and NZ Jhanjhi. 2021.
Amazon product sentiment analysis using machine learning techniques.Revista
Argentina de Clínica Psicológica30, 1 (2021), 695.
[38] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. 2022.
Gmflow: Learning optical flow via global matching. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 8121–8130.
[39] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav
Artzi. 2019. Bertscore: Evaluating text generation with bert.arXiv preprint
arXiv:1904.09675(2019).
[40] Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen
Eger. 2019. MoverScore: Text Generation Evaluating with Contextualized Em-
beddings and Earth Mover Distance. InProceedings of the 2019 Conference on

Video-Based User Complaint Text Generation CIKM ’25, November 10–14, 2025, Seoul, Republic of Korea
Empirical Methods in Natural Language Processing. Association for Computational
Linguistics, Hong Kong, China.[41] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang,
Wenhao Jiang, Junwu Zhang, Zongwei Li, et al .2023. LanguageBind: Extend-
ing Video-Language Pretraining to N-modality by Language-based Semantic
Alignment.arXiv preprint arXiv:2310.01852(2023).
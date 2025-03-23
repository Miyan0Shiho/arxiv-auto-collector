# RAD: Retrieval-Augmented Decision-Making of Meta-Actions with Vision-Language Models in Autonomous Driving

**Authors**: Yujin Wang, Quanfeng Liu, Zhengxin Jiang, Tianyi Wang, Junfeng Jiao, Hongqing Chu, Bingzhao Gao, Hong Chen

**Published**: 2025-03-18 03:25:57

**PDF URL**: [http://arxiv.org/pdf/2503.13861v1](http://arxiv.org/pdf/2503.13861v1)

## Abstract
Accurately understanding and deciding high-level meta-actions is essential
for ensuring reliable and safe autonomous driving systems. While
vision-language models (VLMs) have shown significant potential in various
autonomous driving tasks, they often suffer from limitations such as inadequate
spatial perception and hallucination, reducing their effectiveness in complex
autonomous driving scenarios. To address these challenges, we propose a
retrieval-augmented decision-making (RAD) framework, a novel architecture
designed to enhance VLMs' capabilities to reliably generate meta-actions in
autonomous driving scenes. RAD leverages a retrieval-augmented generation (RAG)
pipeline to dynamically improve decision accuracy through a three-stage process
consisting of the embedding flow, retrieving flow, and generating flow.
Additionally, we fine-tune VLMs on a specifically curated dataset derived from
the NuScenes dataset to enhance their spatial perception and bird's-eye view
image comprehension capabilities. Extensive experimental evaluations on the
curated NuScenes-based dataset demonstrate that RAD outperforms baseline
methods across key evaluation metrics, including match accuracy, and F1 score,
and self-defined overall score, highlighting its effectiveness in improving
meta-action decision-making for autonomous driving tasks.

## Full Text


<!-- PDF content starts -->

RAD: Retrieval-Augmented Decision-Making of Meta-Actions with 
Vision-Language Models in Autonomous Driving 
Yujin Wang! Quanfeng Liu! 
Junfeng Jiao? 
'Tongji University Zhengxin Jiang! 
Hongging Chu!” 
Yale University Tianyi Wang” 
Bingzhao Gao! Hong Chen! 
3University of Texas at Austin 
chuhongqing@tongji.edu.cn 
Abstract 
Accurately understanding and deciding high-level meta- 
actions is essential for ensuring reliable and safe au- 
tonomous driving systems. While vision-language mod- 
els (VLMs) have shown significant potential in various au- 
tonomous driving tasks, they often suffer from limitations 
such as inadequate spatial perception and hallucination, 
reducing their effectiveness in complex autonomous driv- 
ing scenarios. To address these challenges, we propose 
a retrieval-augmented decision-making (RAD) framework, 
a novel architecture designed to enhance VLMs’ capabili- 
ties to reliably generate meta-actions in autonomous driv- 
ing scenes. RAD leverages a retrieval-augmented genera- 
tion (RAG) pipeline to dynamically improve decision accu- 
racy through a three-stage process consisting of the embed- 
ding flow, retrieving flow, and generating flow. Addition- 
ally, we fine-tune VLMs on a specifically curated dataset 
derived from the NuScenes dataset to enhance their spa- 
tial perception and bird’s-eye view image comprehension 
capabilities. Extensive experimental evaluations on the cu- 
rated NuScenes-based dataset demonstrate that RAD out- 
performs baseline methods across key evaluation metrics, 
including match accuracy, and F1 score, and self-defined 
overall score, highlighting its effectiveness in improving 
meta-action decision-making for autonomous driving tasks. 
1. Introduction 
In recent years, the race towards fully autonomous vehicles 
has spurred extensive research into robust decision-making 
approaches, a fundamental task in autonomous driving sys- 
tems [26, 41, 49]. Ensuring safe and efficient motion plan- 
ning requires continuous interpretation of dynamic environ- 
ments, real-time reasoning under uncertainty, and efficient 
integration of vast amounts of multimodal data [28]. 
Traditional autonomous driving systems adopt a modu- 
lar development strategy, in which perception, prediction, planning, and control are developed and optimized inde- 
pendently before being integrated into the vehicle system 
[15, 47]. However, as the information flow propagates 
across these modules, errors and delays can accumulate, po- 
tentially leading to suboptimal or even unreasonable driv- 
ing decisions. To further mitigate these errors and improve 
computational efficiency, end-to-end autonomous driving 
has emerged as a prominent research direction [7, 8]. 
End-to-end refers to a model that directly receives in- 
put from sensor data (e.g., cameras, LiDAR) and directly 
outputs vehicle planning decisions. In recent studies [1 1, 
18, 22], end-to-end autonomous driving algorithms have 
demonstrated their superiority in both simulation environ- 
ments and real-world road tests. Moreover, the emergence 
of foundation models provides a promising solution to en- 
hance motion planning performance, improve generaliza- 
tion across diverse scenarios, and increase interpretability 
in end-to-end autonomous driving [13, 16, 29, 38]. Trained 
on huge amounts of human knowledge, these models ex- 
hibit advanced comprehension and reasoning capabilities, 
highlighting the immense potential of artificial intelligence 
in complex decision-making tasks. Integrating such foun- 
dation models into autonomous driving systems could fa- 
cilitate the development of human-like driving behaviors, 
advancing the field toward safer and more adaptable au- 
tonomous vehicles. 
Autonomous driving tasks require models with robust vi- 
sual perception capabilities, making vision-language mod- 
els (VLMs) particularly well-suited for this domain. VLMs 
trained on large-scale data often demonstrate strong rea- 
soning capabilities, enabling them to infer the evolution of 
complex driving scenarios. Current research [19, 31, 33, 
34, 36] has focused on fine-tuning pre-trained VLMs us- 
ing visual question-answer (VQA) pairs composed of scene 
images and corresponding driving actions. This approach 
enables VLMs to generate feasible trajectories, enhancing 
their applicability in real-world autonomous driving tasks. 
However, fine-tuning or even full-scale fine-tuning of 
VLMs using large-scale datasets requires substantial com-

NuScenes Dataset 
Front-View 
Scene Images _ 
Concatenating RAD: Retrieval-Augmented Decision-Making of Meta-Actions 
with Vision-Language Models in Autonomous Driving Query Scene 
New Front-View 
Scene Image —_— A —, 
Retrieving with 
— 8 Cosine Similarity ; 2 —— 
BLIP-2 0G Vector = Embedding . By ip.2 = = => Model ——~ Database Model «= ", 
Indexing | i 1 T 
-— i f 1 _ Retrieved Scene , 
, Ground Truth = Aon niad _ Meta-Actions . we New Generated 
my BEV Image Generated a ae a 1 a BEV Images Corresponding = 
Meta-Action 
Front-View BEV FRESE 
I | 
LoRA | =e Embedding Flow 
Spatial Fine-Tuning é z = ——» Retrieving Flow . A enerate 
Perception —— Fine-Tuned ——  seta-actions ——  Fine-tuning Flow 
Enhancing VQAs VLMs —— Generating Flow 
Figure 1. The overview of our RAD method. The framework consists of four working flows, namely embedding flow, retrieving flow, 
fine-tuning flow and generating flow. The embedding flow encodes front-view images and BEV images into a vector database. Given 
a query scene, the retrieving flow retrieves the most similar scene from the database. The fine-tuning flow involves fine-tuning VLMs 
to enhance spatial perception and BEV image comprehension. The generating flow guides VLMs in generating contextually appropriate 
meta-actions according to the query scene, the retrieved scene, its ground truth meta-action, and proper prompts. 
putational resources. Additionally, deploying VLMs with 
an extremely large number of parameters on vehicle- 
end hardware poses significant constraints. To address 
these challenges, retrieval-augmented generation (RAG) 
has emerged as a promising approach to enhance the 
decision-making capabilities of VLMs by incorporating ex- 
ternal knowledge bases [14, 42]. The core idea of RAG is to 
augment generative models with a retrieval module that dy- 
namically retrieves relevant textual information during the 
generation process. In vision-language tasks, RAG can ef- 
fectively mitigate limitations caused by knowledge scarcity. 
By integrating external knowledge bases, models can not 
only extract information from images but also retrieve sup- 
plementary knowledge, thereby improving the robustness 
and accuracy of the generated outputs. Although the direct 
application of RAG to the decision-making process in au- 
tonomous driving remains limited, an increasing number of 
studies have explored its potential in specific tasks such as 
scene understanding and regulation retrieval [4, 20, 46]. 
In this work, we propose a retrieval-augmented decision- 
making (RAD) framework, introducing a novel approach 
to assist VLMs in generating meta-actions using RAG for 
the first time, as depicted in Figure |. The main research 
contributions of this work are outlined as follows: 
¢ Pre-Training VLMs for Spatial Perception Tasks: 
We construct obstacle perception tasks based on the 
NuScenes dataset [3], incorporating VQA pairs designed to capture obstacle categories, positions, and other spatial 
information. This pre-training process enables VLMs to 
explicitly learn key geometric features such as the loca- 
tions and sizes of obstacles, leading to improved perfor- 
mance in spatial perception tasks. 
Establishing an External Knowledge Base with 
NuScenes Ground Truth Data: We select a subset of 
scenes containing navigation information, historical tra- 
jectory data, and future meta-action ground truth. Fur- 
thermore, we generate bird’s-eye view (BEV) images cor- 
responding to the scene images. The surround-view im- 
ages from these scenes are then encoded into vector repre- 
sentations using BLIP-2 [25], alongside the BEV images, 
to form the knowledge base. 
Developing a Retrieval and Generation Pipeline 
for Meta-Action Decision-Making using Fine-Tuned 
VLMs and RAG: We employ cosine similarity to retrieve 
the most similar scene from the external knowledge base 
including the front-view image of the current scene. The 
corresponding six surround-view images, speed informa- 
tion, navigation data, and ground truth trajectory are then 
used as auxiliary inputs, guiding the VLM in generating 
a trustworthy planning trajectory for the current scene. 
The remainder of this paper is organized as follows: In 
Section 2, the detailed literature review is conducted. In 
Section 3, four working flows of the proposed RAD frame- 
work are introduced. In Section 4, comparative experiments

and ablation studies are designed. Section 5 summarizes the 
work and discusses future research directions. 
2. Related Works 
2.1. Multimodal Large Language Models in Au- 
tonomous Driving 
Utilizing multimodal large language models (MLLMs) in 
autonomous driving enhances decision-making by lever- 
aging their extensive knowledge and reasoning capabili- 
ties through multisource information such as vision, lan- 
guage, and rules, significantly improving scene understand- 
ing, strategy generation, and interpretability. Drive MLM 
[39] employed MLLMs to generate high-level behavioral 
decisions (e.g., lane-changing, deceleration, acceleration, 
etc.), which were then integrated with traditional motion 
planning modules, balancing flexibility and interpretability. 
“Drive as you speak” [10] enriched large language models 
(LLMs) with comprehensive environmental data from dif- 
ferent vehicle modules, leading to safer decisions. “Driving 
with LLMs” [6] introduced a LLM that generated 10,000 
driving scenarios for agent training. “Drive like a human” 
[13] demonstrated LLMs’ capabilities of understanding and 
interacting with environments in closed-loop systems, ef- 
fectively navigating long-tail autonomous driving scenar- 
ios. DriveVLM [37] adopted a multistage reasoning chain 
that combined scene description, dynamic analysis, and 
hierarchical planning. Additionally, DriveVLM-Dual in- 
corporated traditional 3D perception algorithms to ensure 
both cognitive depth and real-time control. Pix2Planning 
[30] formulated planning as an autoregressive sequence pre- 
diction problem, using a vision-language Transformer to 
generate trajectory points. VLP [31] incorporated linguis- 
tic descriptions into the training process and aligned them 
with visual features, significantly improving cross-city and 
cross-scenario generalization. To enhance interpretability, 
some studies [44, 45] introduced “future trajectory images”, 
which were processed by multimodal models to generate 
natural language explanations. Senna [23] further refined 
the decision-making process by separating high-level meta- 
actions from low-level trajectory predictions. In this frame- 
work, VLMs first produced directional or speed-level de- 
cisions before end-to-end models executed precise paths, 
thereby achieving a hierarchical strategy that was similar 
to human driving behaviors. 
However, these methods are prone to hallucination, a 
limitation arising from the reliance of MLLMs on learned 
associations between visual inputs and language-based rea- 
soning. As a result, they may misinterpret ambiguous or 
occluded objects, leading to incorrect high-level decision- 
making. This issue becomes particularly critical in long-tail 
scenarios, where the model encounters rare or underrepre- 
sented driving conditions not well-covered in the training data. Such misinterpretations can ultimately compromise 
the reliability and safety of the autonomous driving system. 
2.2. Retrieval-Augmented Generation in Vision- 
Language Models 
In vision-language tasks, RAG mitigates knowledge limi- 
tations by leveraging external knowledge bases, enabling 
models to extract insights from images while supplementing 
them with retrieved contextual data. This dual approach sig- 
nificantly helps mitigate model hallucination and improve 
planning accuracy. Jiang et al. [24] introduced a RAG- 
based framework for VLMs, demonstrating its effectiveness 
in complex tasks requiring extensive background knowl- 
edge. Their study underscored the limitations of conven- 
tional end-to-end VLMs when faced with knowledge defi- 
ciencies, whereas RAG facilitated richer contextual integra- 
tion, enhancing both reasoning and generation. Building 
on this, Shao et al. [35] further investigated RAG’s role in 
VQA tasks, showing that combining retrieval mechanisms 
with pre-trained VLMs significantly strengthened model 
performance in complex reasoning scenarios. Additionally, 
Ram et al. [32] examined RAG’s impact on pre-training 
and fine-tuning, illustrating that incorporating large-scale 
external data sources during pre-training improved down- 
stream performance by enhancing cross-modal reasoning, 
particularly in retrieval-based tasks. Meanwhile, Zheng et 
al. [50] emphasized RAG’s broader advantages, particu- 
larly in improving generative flexibility and adaptability in 
multimodal tasks. Their findings highlighted RAG’s ef- 
fectiveness in handling scenarios lacking sufficient anno- 
tations or domain-specific knowledge, reinforcing its po- 
tential in bridging knowledge gaps for more informed and 
context-aware model outputs. Hussien et al. [20] illus- 
trated how RAG-augmented VLMs enhanced cross-modal 
retrieval, particularly by strengthening associations between 
images and textual data. For performance optimization, 
Yuan et al. [46] introduced a dynamic knowledge retrieval 
mechanism, emphasizing real-time adjustments in retrieval 
and generation processes based on task-specific require- 
ments. This adaptive approach allowed RAG to selectively 
retrieve the most relevant background knowledge, improv- 
ing performance across various multimodal applications. 
Cai et al. [4] developed a traffic regulation retrieval agent 
based on RAG, enabling automatic retrieval of relevant traf- 
fic rules and guidelines based on the ego vehicle’s status. 
Moreover, Cui et al. [12] incorporated a RAG-based mem- 
ory module that continuously learned takeover preferences 
through human feedback to enhance motion planning. 
Despite its strong potential, research on directly utiliz- 
ing RAG to guide VLMs in meta-action decision-making 
remains limited. To address this gap, we propose the RAD 
framework, which, for the first time, integrates RAG with 
pre-training for spatial perception capabilities, enabling

more effective decision-making of meta-actions. 
3. Methodology 
As shown in Figure |, the proposed RAD framework com- 
prises four work flows: embedding flow, retrieving flow, 
fine-tuning flow and generating flow. Among these, the 
fine-tuning flow operates independently, as its primary ob- 
jective is to enhance the spatial perception capabilities of 
VLMs through separate fine-tuning. In the embedding flow, 
BEV images are generated to correspond with front-view 
scene images from the NuScenes dataset. These image 
pairs are encoded into a vector space using a frozen BLIP-2 
model and the separate embeddings are then concatenated 
and stored in a vector database. In the retrieving flow, a new 
front-view image and its corresponding BEV image serve 
as a query. These images are encoded into the vector space 
using the same frozen BLIP-2 model. Cosine similarity is 
then computed between the query images and those stored 
in the database, enabling the retrieval of the most similar 
scene from the database. Furthermore, based on the relative 
positional relationships between consecutive scenes in the 
NuScenes dataset, the ground truth meta-actions executed 
in each scene can be extracted. Finally, in the generating 
flow, the query scene, retrieved scene, its ground truth meta- 
action, and proper prompts serve as inputs to the VLMs. 
These inputs guide the model to make decisions and gener- 
ate meta-actions, ensuring more accurate and context-aware 
autonomous driving behaviors. 
All the extracted meta-actions are shown as follows: 
(1) Speed up (rapidly) (2) Slow down (rapidly) 
(3) Turn left/right (4) Drive along the curve 
(5) Turn around (6) Change lane to the left/right 
(7) Reverse (8) Shift slightly to the left/right 
(9) Stop (10) Go straight constantly/slowly 
3.1. The Fine-Tuning Flow 
Making precise meta-action decisions in autonomous driv- 
ing requires an accurate understanding of the environment. 
If a model lacks sufficient spatial perception capabilities, 
it may fail to construct a reliable environmental represen- 
tation, potentially leading to obstacle avoidance failures 
in meta-action decision-making. WLMs typically rely on 
monocular or surround-view camera inputs and estimate 
depth information from single-frame images. However, in 
long-trail scenarios, monocular vision exhibits significant 
depth estimation errors [5]. Experimental results on the 
NuScenes dataset indicate that existing VLMs generally 
lack robust spatial perception, which severely impacts the 
safety of decision-making and motion control [43]. 
To address the aforementioned challenges, VLMs should 
first undergo fine-tuning to enhance their spatial perception Image Filtering Question Generation 
2D Info Q: How many meters is the distance from 
<Object Class #1> to <Object Class #2> 
—> inthe image? 
A: The distance from <Object Class #1> 
to <Object Class #2> is [Distance]. Bounding Box 
Object Class 
Q: What is the size of <Object Class> 
a located within <Bounding Box> in the 
3D Caption image? 
Object Size A: The size of <Object Class> is <Object 
Object Size>. 
Coordinate oo 
Figure 2. The process of generating a dataset for spatial perception 
enhancement based on the NuScenes dataset 
capabilities. The structure of VLMs typically consists of 
a vision encoder and an LLM. In this work, we focus on 
fine-tuning only the LLM component to enhance its spa- 
tial perception. We utilize the NuScenes dataset to generate 
a specified dataset for spatial perception enhancement, fol- 
lowing the process illustrated in Figure 2. 
During the image filtering process, it is necessary to en- 
sure the uniqueness of the VQA pairs by cross-referencing 
the annotated data from the origin NuScenes dataset. The 
generated dataset for fine-tuning includes over 100,000 
training samples, covering key spatial perception tasks such 
as object class recognition, object distance estimation and 
object size estimation. 
For spatial perception enhancement fine-tuning, the loss 
function for a single sample is defined as follows: 
N n 
J= < S- bs. (- » Yeyi log (pe,i) ) 
3 
vaae(-$ Yr 8) Hal) 
() 
where, N is the batch size during fine-tuning; A;,; is the 
loss identifier for object class recognition in the i-th sam- 
ple (if there is a corresponding class, 1; will be set to 1; 
and otherwise, \;,; will be set to 0); Az,; is the loss identi- 
fier for object size estimation; A3,; is the loss identifier for 
object distance estimation; n is the total number of classes 
in the classification task; y-,; is the label for the 7-th sam- 
ple belonging to class c, represented by one-hot encoding; 
Pc,i is the probability of the i-th sample being classified as 
class c by the model; z;,; is the output size of the i-th sam- 
ple in the j-th dimension from the model; z; ; is the ground 
truth size of the 7-th sample in the j-th dimension; x; is the 
model’s output for the distance from the 7-th sample to the 
reference object; and x; is the ground truth distance from 
the i-th sample to the reference object. 
In this work, we fine-tune a series of VLMs, primar- 
ily from the Qwen family [1, 2, 9], using low-rank adap- 
tion (LoRA) [17, 51]. The overall training is conducted for

° System: This image illustrates the BEV view of a driving scene, 
20 f showing the area of 60 meters ahead, 30 meters behind, 30 meters left 
nae and right of the ego vehicle. The units of longitudinal and lateral 
° _ <—/ coordinates are meters. The ego vehicle is located at the center [0,0], 
igs => ri represented by a blue rectangle. The red rectangles represent the 
— ww objects of vehicle type, including cars, trucks, etc. If there is an arrow == a a a . . & 2 6 5 
10 % 4 on the red rectangle, it means that it will move in the direction of the 
. arrow. The green dots represent pedestrians, and the green arrows also 
. as indicate the moving direction. Black dots are static obstacles, 
“so including roadblocks, traffic lights, etc. 
Question I: What kind of object (pedestrian, vehicle, or static obstacle) is located within the coordinate [7.6,8.9] 
in this image? 
Answer I: There is a vehicle located within the coordinate [7.6,8.9]. 
Question 2: What is the central position coordinate of the left-front static obstacle in this image? The result 
retains one decimal place after the decimal point. 
Answer 2: The central position coordinate of the left-front static obstacle is [24.5,17.2]. 
Question 3: What is the distance from the left-front static obstacle to the left pedestrian in this image? The result 
retains one decimal place after the decimal point. 
Answer 3: The distance from the left-front static obstacle to the left pedestrian is 16.3 m. 
Figure 3. The fine-tuning VQA paradigm for BEV image understanding 
three epochs. Additionally, following the BEVFormer [27], 
we generate BEV images from the existing surround-view 
images in the NuScenes dataset. Intuitively, incorporating 
BEV images helps the model better understand the relative 
spatial relationships of objects in driving scenes. Therefore, 
it is also necessary to train VLMs to recognize and interpret 
BEV images effectively. The fine-tuning paradigm, as illus- 
trated in Figure 3, follows a similar approach to the VQA 
pair construction method based on ground truth information 
to develop a robust ability to understand BEV images. 
3.2. The Embedding Flow 
In the embedding flow, we encode front-view images from 
the NuScenes dataset along with the pre-generated BEV 
images into a unified vector space. Since this embed- 
ding operation does not involve cross-modal content, the 
frozen BLIP-2 model weights can be directly utilized, en- 
suring computational efficiency and consistency. To main- 
tain the one-to-one correspondence between front-view im- 
ages and BEV images, their embedding vectors are concate- 
nated within this flow. The resulting concatenated vectors 
are then uniformly stored in an indexed vector database. 
3.3. The Retrieving Flow 
The core of the retrieving flow lies in the computation of 
cosine similarity. Given two image embeddings v; and vj, cosine similarity is defined as: 
ViVi 
IIvallllvalh similarity; ; = (2) 
where, || * || represents the Euclidean norm. 
The main framework of the retrieving flow is illustrated 
in Figure 4. For a new scene, we first generate its BEV 
images from the surround-view images. The front-view 
image and BEV image of the new scene jointly trigger a 
query scene. The embeddings for the new front-view image 
and BEV image are then extracted using the frozen BLIP-2 
model. Since the vector database stores concatenated em- 
bedding vectors, the embeddings for the front-view image 
and BEV image are retrieved through length decomposi- 
tion. The cosine similarity between the new front-view im- 
age embeddings and those stored in the database is com- 
puted and denoted as similaritys,. Similarly, the cosine 
similarity between the new BEV image embeddings and 
those stored in the database is computed and denoted as 
similaritypey. To flexibly adjust the retrieval preference 
toward either the front-view image or the BEV image, a hy- 
perparameter w is introduced. In this work, w is set to 0.5 as 
a balanced weight for retrieval. The overall similarity could 
be calculated as follows: 
similarity = (1 —w)- similarity fy, + w - similarityer 
(3) 
The scene with the highest overall similarity is then re- 
trieved from the vector database. Using its index, we can

obtain the corresponding front-view image, BEV image, 
and pre-extracted ground truth meta-action. 
Query Scene 
| el 7 7, T 
i New Generated 
BEV Image New Front-View (FV) 
Scene Image 
%*BLIP-2 Model 
t t 
{ NewFV Embedding | [ New BEV Embedding | 
FV Embedding BEV Embedding 
Similarity Similarity 
Computation Computation 
similarity = (1—)- similaritypy + w - similaritypey 
Vector Database Indexing 
y v 
Most-Similar Scene (FV&BEV) Corresponding Meta-Action 
Figure 4. The main framework of the retrieving flow 
3.4. The Generating Flow 
In the generating flow, we primarily employ prompt engi- 
neering to guide VLMs in reasoning based on the retrieved 
scene and its corresponding meta-action, enabling them to 
make accurate meta-action decisions for the new scene. The 
prompts should be divided into two key components: 
¢ System Prompt: Guide VLMs to make meta-action de- 
cisions based on the provided images. 
¢ RAG-Specific Prompt: Instruct VLMs to understand the 
retrieved scene images and corresponding meta-actions. 
For this process, we primarily use the Qwen series of 
VLMs, as they support multiple image inputs, making 
prompt design more flexible and effective. With structured 
and well-designed prompts, the VLMs analyze the front- 
view image and BEV image of the current scene, ultimately 
generating a single meta-action as the final output. 
4. Experiments 
4.1. Dataset Preparation 
We divide the 34,000 scenes from the NuScenes dataset into 
three subsets: 10,000 scenes are allocated for fine-tuning 
VLMs, focusing on enhancing spatial perception and BEV 
image understanding; 20,000 scenes are embedded in the 
vector database as prior information; and the remaining 
4,000 scenes serve as the test set, used to evaluate the frame- 
work’s effectiveness and the model’s overall performance. 4.2. Evaluation Metrics 
To assess the performance, we employ traditional classifica- 
tion metrics such as accuracy, precision, recall and F1 score. 
Additionally, we introduce a customized partial match score 
to account for semantically similar but not entirely identical 
cases. Finally, we utilize a weighted method to compute a 
comprehensive performance score. 
We firstly adopt ExactMatchAccuracy to evaluate 
whether the model provides a fully correct meta-action for 
a given scene, which is formally defined as follows: 
Nmatch (4) 
total ExactMatch Accuracy = 
where, N,natch 18 the number of scenes where the generated 
meta-actions exactly match the ground truth; and Nyotai is 
the total number of scenes. 
For each meta-action, Precision, Recall, and F'1 can 
be used as evaluation metrics, which are defined as follows: 
a TP, Precision; = TP, + FP, (5) 
TP; 
Recalli = TP EN, ) 
Fl, = 2 x Precision; x Recall; 
(7) Precision; + Recall; 
where, 7'P; is the true positives, and the number of scenes 
where the generated meta-actions are 7 and the ground 
truth are also 7; FP; is the false positives, and the num- 
ber of scenes where the generated meta-actions are 7 but the 
ground truth are not 7; FN; is the false negatives, and the 
number of scenes where the generated meta-actions are not 
i but the ground truth are 7. 
To evaluate the overall performance across differ- 
ent meta-actions in the test set, Macro—F1 and 
Weighted — F1 scores are introduced. Macro — F'1 is 
the unweighted average of F'l scores across all meta- 
actions, while Weighted — F'1 is the weighted average of 
F'1 scores, which are defined as: 
kK 
1 Macro — F1 = K » Fi; (8) 
i=1 
Weighted — F1 = So nF; (9) 
ia 
where, Jx represents the total number of meta-actions, 
which is set to 15; and n; represents the number of scenes 
where the ground truth meta-action is 7. 
To account for the semantic similarity between cer- 
tain meta-actions, we introduce a PartialMatchScore. 
Specifically, meta-actions involving leftward maneu- 
vers—such as turn left, change lane to the left and shift

Table 1. Comparison among different baselines and our RAD method 
Method Exact Match Accuracy Macro-Fl Weighted-F1 Partial Match Score Overall Score 
Lynx (Fine-tuning)[48] 0.1524 0.0167 0.0653 0.2768 0.1327 
CogVLM (Fine-tuning)[40] 0.2178 0.0204 0.1105 0.3563 0.1846 
DriveLM (on LLaMA-LoRA-BIAS-7B)[36] 0.1455 0.0448 0.1203 0.3028 0.1518 
DriveLM (on LLaMA-BIAS-7B)[36] 0.1896 0.0409 0.1212 0.3425 0.1693 
DriveLM (on LLaMA-CAPTION-7B)[36] 0.2034 0.0380 0.1080 0.3952 0.1896 
GPT-40 (Official API[21] 0.2994 0.1127 0.2288 0.4377 0.2756 
Drive VLM[37] 0.3743 0.1671 0.3325 0.5462 0.3589 
Drive VLM-Dual (cooperating with VAD[22])[37] 0.4016 0.1854 0.3506 0.5613 0.3801 
RAD (Ours, on Qwen-VL-2.5-7B) 0.4096 0.1907 0.3813 0.5870 0.3956 
slightly to the left—are classified under the left group, 
while analogous rightward actions form the right group. 
Similarly, meta-actions indicating forward motion at vary- 
ing speeds are categorized accordingly, with go straight 
slowly, slow down, and slow down rapidly mapping to 
the deceleration group, while both speed up and speed 
up rapidly mapping to acceleration group. — Further- 
more, unique behaviors such as go straight constantly, turn 
around, reverse, stop, and drive along the curve are collec- 
tively assigned to a separate unique group. If the gener- 
ated meta-actions and the ground truth meta-actions are not 
identical but belong to the same semantic group (excluding 
the unique group), they are considered partially matched. 
Thus, the semantic similarity S is defined as follows: 
1, if 7 is the same as i. 
0.5, if i partially matches 7. (10) 
0, if 7 totally differs from i. S(i,2) = 
where, i is the ground truth meta-action in one scene; and 7 
is the generated meta-action. 
Then, the average Partial MatchScore is obtained by 
averaging across all scenes: 
1 Ntotal 
S(in, tp 11 
Nrtotal » (th tx) aD 
Finally, different weights are assigned to each metric to 
derive the comprehensive scoring formula Overall Score: PartialMatchS‘core = 
OverallScore = a+ ExactMatch Accuracy 
+ 8-Macro—F1 
+7-Weighted — F'1 
+ 6- PartialMatchScore (12) 
where, @ is set to 0.4; 8, y, and 6 are all set to 0.2, which 
could be adjusted according to specific tasks. 
4.3. Comparative Experiments 
We evaluate the performance of our proposed RAD frame- 
work on Qwen-VL-2.5-7B VLM and compare it against several other state-to-the-art baseline methods: Lynx [48], 
CogVLM [40], DriveLM [36], GPT-40 [21] and DriveVLM 
[37]. Table 1 presents a thorough quantitative compari- 
son between our proposed RAD and these baselines across 
multiple evaluation criteria. Our RAD consistently outper- 
forms all baseline methods, demonstrating clear advantages 
in meta-action decision-making for autonomous driving. 
In particular, RAD achieves an Exact MatchAccuracy 
of 0.4096, substantially outperforming DriveVLM-Dual’s 
0.4016, and attains an OverallScore of 0.3956 compared 
to DriveVLM-Dual’s 0.3801. 
A deeper analysis of the remaining metrics further un- 
derscores RAD’s strengths. Macro — F'1, a balanced mea- 
sure of model performance across all classes, achieves 
0.1907, well above DriveVLM-Dual’s 01854. Meanwhile, 
Weighted — F1 of 0.3813 indicates its effectiveness in 
scenarios where class imbalances exist, significantly outper- 
forming all baselines and reflecting RAD’s notable capabil- 
ities to handle diverse datasets. Also, Partial M atchScore 
of 0.5870 also highlights RAD’s fine-grained generative ca- 
pability, which suggests that RAD not only excels at pro- 
ducing entirely correct answers, but also consistently cap- 
tures partially correct information, an essential trait for 
more nuanced or multi-faceted decision-making tasks. 
The poor performance of the baseline methods is mainly 
due to their lack of task-specific training. As a result, these 
models exhibit limited spatial perception capabilities and 
poor BEV image comprehension. Additionally, the param- 
eter size constraints and version limitations of the base mod- 
els used in these baselines hinder their ability to achieve op- 
timal results. However, RAD’s superior performance over 
GPT-40 across all metrics demonstrates the feasibility of 
specialized VLMs with smaller parameter sizes that rival 
or even surpass large-scale general-purpose models in com- 
plex and domain-specific tasks. 
In summary, the results in Table | validate the efficacy 
and robustness of our RAD model. Through a combination 
of architectural innovations and targeted training strategies, 
RAD not only achieves profound performance across mul- 
tiple metrics but also provides insights into how specialized

Table 2. Ablation studies on fine-tuning VLMs and RAG pipeline 
VLMs Method Exact Match Accuracy Macro-F1 Weighted-F1 Partial Matching Score Overall Score 
Qwen-VL-2-2B[9] Vanilla 0.2188 0.0358 0.1013 0.4353 0.2020 
Vanilla + RAG 0.2145 0.1049 0.2278 0.4319 0.2387 
Fine-tuning 0.1543 0.0528 0.1194 0.3017 0.1565 
Fine-tuning + RAG 0.2610 0.1302 0.2556 0.4538 0.2723 
Qwen-VL-2-7B[9] Vanilla 0.2866 0.0654 0.1721 0.4941 0.2609 
Vanilla + RAG 0.3404 0.1460 0.3235 0.5424 0.3385 
Fine-tuning 0.2908 0.0717 0.1986 0.4562 0.2616 
Fine-tuning + RAG 0.3446 0.1460 0.3011 0.5213 0.3315 
Qwen-VL-2.5-3B[2] Vanilla 0.1318 0.0366 0.0955 0.3886 0.1568 
Vanilla + RAG 0.1240 0.0298 0.0814 0.3866 0.1491 
Fine-tuning 0.2164 0.0531 0.1398 0.3949 0.2041 
Fine-tuning + RAG 0.2539 0.1075 0.2090 0.4520 0.2552 
Qwen-VL-2.5-7B[2] Vanilla 0.2849 0.0644 0.1715 0.4893 0.2590 
Vanilla + RAG 0.3581 0.1981 0.3386 0.5544 0.3615 
Fine-tuning 0.3482 0.1085 0.2885 0.5360 0.3259 
Fine-tuning + RAG 0.4096 0.1907 0.3813 0.5870 0.3956 
VLMs can excel in intricate autonomous driving tasks. 
4.4. Ablation Studies 
In our ablation studies, we mainly investigate the impacts 
of fine-tuning VLMs and RAG pipeline for spatial percep- 
tion enhancement based on Qwen-VL-2-2B [9], Qwen-VL- 
2-7B [9], Qwen-VL-2.5-3B [2] and Qwen-VL-2.5-7B [2] 
models. The performance of VLMs is evaluated using four 
distinct methods: vanilla (no fine-tuning), vanilla combined 
with RAG, only fine-tuning, and fine-tuning combined with 
RAG (our proposed RAD method). 
The results presented in Table 2 indicate that the com- 
bination of fine-tuning and RAG consistently achieves 
the highest scores across all evaluation metrics, including 
ExactMatchAccuracy, Macro — F1, Weighted — F1, 
PartialMatchScore, and OverallScore, for all model 
variants. Specifically, for Qwen-VL-2.5-7B, our RAD 
method achieves the highest OverallScore of 0.3956, 
marking a significant improvement over methods that de- 
ploy either fine-tuning or RAG separately. Furthermore, the 
incorporation of RAG consistently enhances performance 
for both vanilla and fine-tuned settings across most model 
scales, validating the effectiveness of retrieval-augmented 
strategies in improving model performance. 
Notably, for smaller models such as Qwen-VL-2-2B and 
Qwen-VL-2.5-3B, employing only fine-tuning leads to per- 
formance degradation, suggesting that their limited parame- 
ter sizes hinder effective learning of domain-specific knowl- 
edge through fine-tuning alone. Additionally, for Qwen- 
VL-2.5-3B model, using RAG without fine-tuning results 
in a performance drop, likely due to the unique pre-training 
characteristics of this model. Overall, while fine-tuning 
or RAG independently can enhance performance in larger- 
scale models, the best results are consistently achieved by combining these two strategies, underscoring the impor- 
tance of an integrated approach to maximize VLM effec- 
tiveness. From a practical perspective, the combination of 
fine-tuning and RAG proves particularly suitable for en- 
hancing decision-making capabilities in VLMs. Deploying 
this optimal configuration can substantially improve VLM 
performance, with potential applications extending to se- 
mantic comprehension, trajectory planning, and other com- 
plex autonomous driving tasks. 
5. Conclusion 
In this work, we propose a RAD framework, a novel 
retrieval-augmented architecture designed to enhance the 
meta-action decision-making capabilities of VLMs for au- 
tonomous driving. Through the integration of fine-tuning 
VLMs for spatial perception enhancement and BEV image 
comprehension, RAD effectively enhances VLMs’ capabil- 
ity of meta-action decision-making, ensuring higher accu- 
racy, as demonstrated by notable performance gains across 
key metrics in extensive experimental evaluations. 
Moving forward, we aim to extend RAD in three key 
directions. First, we plan to incorporate more diverse and 
fine-grained datasets beyond the NuScenes dataset, encom- 
passing more challenging corner cases and real-world sce- 
narios, to further enhance model robustness. Second, we 
seek to generalize the RAD framework to additional driv- 
ing tasks, especially trajectory planning and motion con- 
trol. Third, integrating chain-of-thought and reinforcement 
learning into the framework will be crucial for improv- 
ing decision-making depth and adaptability. While fine- 
tuning and RAG will remain essential for enhancing VLM 
generalization, these advancements will strengthen the ro- 
bustness and reliability of autonomous driving systems 
by leveraging RAG methods to tackle complex real-world 
tasks.

References 
[1] 
[2] 
[3] 
[4 
= 
[5] 
[6] 
[7] 
[8 
= 
[9] 
[10] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xi- 
aodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, 
Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Day- 
iheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin 
Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi 
Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei 
Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, 
Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen 
Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan 
Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren 
Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical 
report. arXiv preprint arXiv:2309. 16609, 2023. 4 
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin 
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun 
Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhao- 
hai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren 
Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen 
Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Jun- 
yang Lin. Qwen2.5-vl technical report. arXiv preprint 
arXiv:2502.13923, 2025. 4, 8 
Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, 
Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi- 
ancarlo Baldan, and Oscar Beijbom. Nuscenes: A multi- 
modal dataset for autonomous driving. In Proceedings of 
the IEEE/CVF Conference on Computer Vision and Pattern 
Recognition, pages 11621-11631, 2020. 2 
Tianhui Cai, Yifan Liu, Zewei Zhou, Haoxuan Ma, Seth Z 
Zhao, Zhiwen Wu, and Jiaqgi Ma. Driving with regula- 
tion: Interpretable decision-making for autonomous vehicles 
with retrieval-augmented reasoning via Im. arXiv preprint 
arXiv:2410.04759, 2024. 2, 3 
Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa 
Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endow- 
ing vision-language models with spatial reasoning capabili- 
ties. In Proceedings of the IEEE/CVF Conference on Com- 
puter Vision and Pattern Recognition, pages 14455-14465, 
2024. 4 
Long Chen, Oleg Sinavski, Jan Hiinermann, Alice Karnsund, 
Andrew James Willmott, Danny Birch, Daniel Maund, and 
Jamie Shotton. Driving with Ilms: Fusing object-level vec- 
tor modality for explainable autonomous driving. In Pro- 
ceedings of the IEEE International Conference on Robotics 
and Automation, pages 14093-14100, 2024. 3 
Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, An- 
dreas Geiger, and Hongyang Li. End-to-end autonomous 
driving: Challenges and frontiers. IEEE Transactions on Pat- 
tern Analysis and Machine Intelligence, 2024. | 
Pranav Singh Chib and Pravendra Singh. Recent advance- 
ments in end-to-end autonomous driving using deep learn- 
ing: A survey. [EEE Transactions on Intelligent Vehicles, 9 
(1):103-118, 2023. 1 
Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhi- 
fang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang 
Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical 
report. arXiv preprint arXiv:2407.10759, 2024. 4, 8 
Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, and Ziran [11] 
[12] 
[13] 
[14] 
[15] 
[16] 
[17] 
[18] 
[19] 
[20] Wang. Receive, reason, and react: Drive as you say, with 
large language models in autonomous vehicles. JEEE Intel- 
ligent Transportation Systems Magazine, 16(4):81—94, 2024. 
3 
Can Cui, Yunsheng Ma, Zichong Yang, Yupeng Zhou, Peiran 
Liu, Juanwu Lu, Lingxi Li, Yaobin Chen, Jitesh H. Pan- 
chal, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, and Zi- 
ran Wang. Large language models for autonomous driving 
(lm4ad): Concept, benchmark, experiments, and challenges. 
arXiv preprint arXiv:2410.15281, 2024. | 
Can Cui, Zichong Yang, Yupeng Zhou, Juntong Peng, Sung- 
Yeon Park, Cong Zhang, Yunsheng Ma, Xu Cao, Wen- 
qian Ye, Yiheng Feng, Jitesh H. Panchal, Lingxi Li, Yaobin 
Chen, and Ziran Wang. On-board vision-language mod- 
els for personalized autonomous vehicle motion control: 
System design and real-world validation. arXiv preprint 
arXiv:2411,.11913, 2024. 3 
Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, 
Botian Shi, and Yu Qiao. Drive like a human: Rethinking au- 
tonomous driving with large language models. In Proceed- 
ings of the Winter Conference on Applications of Computer 
Vision Workshops, pages 910-919. TEEE, 2024. 1, 3 
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin- 
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 
Retrieval-augmented generation for large language models: 
A survey. arXiv preprint arXiv:2312.10997, 2023. 2 
Sorin Grigorescu, Bogdan Trasnea, Tiberiu Cocias, and 
Gigel Macesanu. A survey of deep learning techniques for 
autonomous driving. Journal of Field Robotics, 37(3):362- 
386, 2020. | 
Xu Han, Zonglin Meng, Xin Xia, Xishun Liao, 
Brian Yueshuai He, Zhaoliang Zheng, Yutong Wang, 
Hao Xiang, Zewei Zhou, Letian Gao, Lili Fan, Yuke Li, and 
Jiaqi Ma. Foundation intelligence for smart infrastructure 
services in transportation 5.0. JEEE Transactions on 
Intelligent Vehicles, 9(1):39-47, 2024. | 
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen- 
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 
Lora: Low-rank adaptation of large language models. In Pro- 
ceedings of the International Conference on Learning Rep- 
resentations, 2022. 4 
Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, 
Xizhou Zhu, Sigi Chai, Senyao Du, Tianwei Lin, Wenhai 
Wang, et al. Planning-oriented autonomous driving. In Pro- 
ceedings of the IEEE/CVF Conference on Computer Vision 
and Pattern Recognition, pages 17853-17862, 2023. | 
Yidong Huang, Jacob Sansom, Ziqiao Ma, Felix Gervits, 
and Joyce Chai. Drivlme: Enhancing llm-based autonomous 
driving agents with embodied and social experiences. In Pro- 
ceedings of the IEEE/RSJ International Conference on Intel- 
ligent Robots and Systems, pages 3153-3160. IEEE, 2024. 
1 
Mohamed Manzour Hussien, Angie Nataly Melo, Au- 
gusto Luis Ballardini, Carlota Salinas Maldonado, Rubén 
Izquierdo, and Miguel Angel Sotelo. Rag-based explainable 
prediction of road users behaviors for automated driving us- 
ing knowledge graphs and large language models. Expert 
Systems with Applications, 265:125914, 2025. 2, 3

[21] 
[22] 
[23] 
[24] 
[25] 
[26] 
[27] 
[28] 
[29] 
[30] 
[31] 
[32] Raisa Islam and Owana Marzia Moushi. Gpt-40: The 
cutting-edge advancement in multimodal llm. Authorea 
Preprints, 2024. 7 
Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie 
Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, 
and Xinggang Wang. Wad: Vectorized scene representa- 
tion for efficient autonomous driving. In Proceedings of the 
IEEE/CVF International Conference on Computer Vision, 
pages 8340-8350, 2023. 1, 7 
Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, 
Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xing- 
gang Wang. Senna: Bridging large vision-language mod- 
els and end-to-end autonomous driving. arXiv preprint 
arXiv:2410.22313, 2024. 3 
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian 
Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Gra- 
ham Neubig. Active retrieval augmented generation. In Pro- 
ceedings of the Conference on Empirical Methods in Natural 
Language Processing, pages 7969-7992, 2023. 3 
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 
Blip-2: Bootstrapping language-image pre-training with 
frozen image encoders and large language models. In Pro- 
ceedings of the International Conference on Machine Learn- 
ing, pages 19730-19742, 2023. 2 
Yixuan Li, Xuesong Wang, Tianyi Wang, and Qian Liu. 
Characteristics analysis of autonomous vehicle pre-crash 
scenarios. arXiv preprint arXiv:2502.20789, 2025. | 
Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chong- 
hao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: 
Learning bird’s-eye-view representation from lidar-camera 
via spatiotemporal transformers. JEEE Transactions on Pat- 
tern Analysis and Machine Intelligence, 47(3):2020-2036, 
2025. 5 
Yunsheng Ma, Wenqian Ye, Can Cui, Haiming Zhang, Shuo 
Xing, Fucai Ke, Jinhong Wang, Chenglin Miao, Jintai Chen, 
Hamid Rezatofighi, Zhen Li, Guangtao Zheng, Chao Zheng, 
Tianjiao He, Manmohan Chandraker, Burhaneddin Yaman, 
Xin Ye, Hang Zhao, and Xu Cao. Position: Prospective of 
autonomous driving - multimodal llms world models embod- 
ied intelligence ai alignment and mamba. In Proceedings of 
the Winter Conference on Applications of Computer Vision 
Workshops, pages 1010-1026, 2025. | 
Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue 
Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint 
arXiv:2310.01415, 2023. | 
Xiangru Mu, Tong Qin, Songan Zhang, Chunjing Xu, and 
Ming Yang. Pix2planning: End-to-end planning by vision- 
language model for autonomous driving on carla simulator. 
In Proceedings of the IEEE Intelligent Vehicles Symposium, 
pages 2383-2390. IEEE, 2024. 3 
Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup 
Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu 
Ren. Vp: Vision language planning for autonomous driving. 
In Proceedings of the IEEE/CVF Conference on Computer 
Vision and Pattern Recognition, pages 14760-14769, 2024. 
1,3 
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Am- 
non Shashua, Kevin Leyton-Brown, and Yoav Shoham. In- 
10 [33] 
[34] 
[35] 
[36] 
[37] 
[38] 
[39] 
[40] 
[41] 
[42] 
[43] context retrieval-augmented language models. Transactions 
of the Association for Computational Linguistics, 11:1316— 
1331, 2023. 3 
Katrin Renz, Long Chen, Ana-Maria Marcu, Jan 
Hiinermann, Benoit Hanotte, Alice Karnsund, Jamie 
Shotton, Elahe Arani, and Oleg Sinavski. Carllava: Vision 
language models for camera-only closed-loop driving. arXiv 
preprint arXiv:2406.10165, 2024. | 
Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, 
Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: 
Closed-loop end-to-end driving with large language models. 
In Proceedings of the IEEE/CVF Conference on Computer 
Vision and Pattern Recognition, pages 15120-15130, 2024. 
1 
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, 
Nan Duan, and Weizhu Chen. — Enhancing retrieval- 
augmented large language models with iterative retrieval- 
generation synergy. arXiv preprint arXiv:2305.15294, 2023. 
3 
Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, 
Hanxue Zhang, Chengen Xie, Jens BeiBwenger, Ping Luo, 
Andreas Geiger, and Hongyang Li. Drivelm: Driving with 
graph visual question answering. In Proceedings of the 
European Conference on Computer Vision, pages 256-274. 
Springer, 2024. 1, 7 
Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, 
Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and 
Hang Zhao. Drivevim: The convergence of autonomous 
driving and large vision-language models. arXiv preprint 
arXiv:2402.12289, 2024. 3,7 
Shiyi Wang, Yuxuan Zhu, Zhiheng Li, Yutong Wang, Li Li, 
and Zhengbing He. Chatgpt as your vehicle co-pilot: An 
initial attempt. ZEEE Transactions on Intelligent Vehicles, 8 
(12):4706-4721, 2023. | 
Wenhai Wang, Jiangwei Xie, Chuan Yang Hu, Haoming Zou, 
Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming 
Deng, Zhiqi Li, et al. Drivemlm: Aligning multi-modal large 
language models with behavioral planning states for au- 
tonomous driving. arXiv preprint arXiv:2312.09245, 2023. 
3 
Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji 
Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiX- 
uan, et al. Cogvlm: Visual expert for pretrained language 
models. Advances in Neural Information Processing Sys- 
tems, 37:121475-121499, 2024. 7 
Yangyang Wang and Tianyi Wang. Research on dual-clutch 
intelligent vehicle infrastructure cooperative control based 
on system delay prediction of two-lane highway on-ramp 
merging area. Automotive Innovation, 7:588—601, 2024. | 
Yujin Wang, Quanfeng Liu, Jiaqi Fan, Jinlong Hong, 
Hongqing Chu, Mengjian Tian, Bingzhao Gao, and Hong 
Chen. Rac3: Retrieval-augmented corner case comprehen- 
sion for autonomous driving with vision-language models. 
arXiv preprint arXiv:2412.11050, 2024. 2 
Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P Meyer, 
Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M 
Wolff, and Xin Huang. Vlm-ad: End-to-end autonomous

[44] 
[45] 
[46] 
[47] 
[48] 
[49] 
[50] 
[51] driving through vision-language model supervision. arXiv 
preprint arXiv:2412.14446, 2024. 4 
Shota Yamazaki, Chenyu Zhang, Takuya Nanri, Akio 
Shigekane, Siyuan Wang, Jo Nishiyama, Tao Chu, and Kohei 
Yokosawa. Explanation for trajectory planning using multi- 
modal large language model for autonomous driving. arXiv 
preprint arXiv:2411.09971, 2024. 3 
Kairui Yang, Zihao Guo, Gengjie Lin, Haotian Dong, Zhao 
Huang, Yipeng Wu, Die Zuo, Jibin Peng, Ziyuan Zhong, 
Xin Wang, Qing Guo, Xiaosong Jia, Junchi Yan, and Di Lin. 
Trajectory-llm: A language-based data generator for trajec- 
tory prediction in autonomous driving. In Proceedings of 
the International Conference on Learning Representations, 
2025. 3 
Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul 
Newman, Lars Kunze, and Matthew Gadd. Rag-driver: Gen- 
eralisable driving explanations with retrieval-augmented in- 
context learning in multi-modal large language model. arXiv 
preprint arXiv:2402.10828, 2024. 2,3 
Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and 
Kazuya Takeda. A survey of autonomous driving: Com- 
mon practices and emerging technologies. [EEE Access, 8: 
58443-58469, 2020. | 
Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guo- 
qiang Wei, Yang Wei, Yuchen Zhang, Tao Kong, and Ruihua 
Song. What matters in training a gpt4-style language model 
with multimodal inputs? In Proceedings of the Conference of 
the North American Chapter of the Association for Compu- 
tational Linguistics: Human Language Technologies, pages 
7930-7957, 2024. 7 
Miao Zhang, Zhenlong Fang, Tianyi Wang, Qian Zhang, 
Shuai Lu, Junfeng Jiao, and Tianyu Shi. A cascading 
cooperative multi-agent framework for on-ramp merging 
control integrating large language models. arXiv preprint 
arXiv:2503.08199, 2025. | 
Juncheng Zheng, Meiyu Liang, Yang Yu, Yawen Li, and Zhe 
Xue. Knowledge graph enhanced multimodal transformer 
for image-text retrieval. In Proceedings of the IEEE Interna- 
tional Conference on Data Engineering, pages 70-82. IEEE, 
2024. 3 
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, 
Zheyan Luo, Zhangchi Feng, and Yonggiang Ma. Llamafac- 
tory: Unified efficient fine-tuning of 100+ language models. 
arXiv preprint arXiv:2403.13372, 2024. 4 
11
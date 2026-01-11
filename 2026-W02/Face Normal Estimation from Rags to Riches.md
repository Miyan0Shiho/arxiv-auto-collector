# Face Normal Estimation from Rags to Riches

**Authors**: Meng Wang, Wenjing Dai, Jiawan Zhang, Xiaojie Guo

**Published**: 2026-01-05 09:57:24

**PDF URL**: [https://arxiv.org/pdf/2601.01950v1](https://arxiv.org/pdf/2601.01950v1)

## Abstract
Although recent approaches to face normal estimation have achieved promising results, their effectiveness heavily depends on large-scale paired data for training. This paper concentrates on relieving this requirement via developing a coarse-to-fine normal estimator. Concretely, our method first trains a neat model from a small dataset to produce coarse face normals that perform as guidance (called exemplars) for the following refinement. A self-attention mechanism is employed to capture long-range dependencies, thus remedying severe local artifacts left in estimated coarse facial normals. Then, a refinement network is customized for the sake of mapping input face images together with corresponding exemplars to fine-grained high-quality facial normals. Such a logical function split can significantly cut the requirement of massive paired data and computational resource. Extensive experiments and ablation studies are conducted to demonstrate the efficacy of our design and reveal its superiority over state-of-the-art methods in terms of both training expense as well as estimation quality. Our code and models are open-sourced at: https://github.com/AutoHDR/FNR2R.git.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 1
Face Normal Estimation from Rags to Riches
Meng Wang, Wenjing Dai, Jiawan Zhang, and Xiaojie Guo
Abstract—Although recent approaches to face normal estima-
tion have achieved promising results, their effectiveness heavily
depends on large-scale paired data for training. This paper con-
centrates on relieving this requirement via developing a coarse-to-
fine normal estimator. Concretely, our method first trains a neat
model from a small dataset to produce coarse face normals that
perform as guidance (called exemplars) for the following refine-
ment. A self-attention mechanism is employed to capture long-
range dependencies, thus remedying severe local artifacts left in
estimated coarse facial normals. Then, a refinement network is
customized for the sake of mapping input face images together
with corresponding exemplars to fine-grained high-quality facial
normals. Such a logical function split can significantly cut the
requirement of massive paired data and computational resource.
Extensive experiments and ablation studies are conducted to
demonstrate the efficacy of our design and reveal its superiority
over state-of-the-art methods in terms of both training expense as
well as estimation quality. Our code and models are open-sourced
at: https://github.com/AutoHDR/FNR2R.git.
Index Terms—Face normal estimation, coarse-to-fine,
exemplar-based learning.
I. INTRODUCTION
FACE normal estimation is crucial to understanding the 3D
structure of facial images, which acts as a fundamental
component for various applications such as portrait editing
[39], [38], [35] and augmented reality [9], [15]. However, this
problem is in nature highly ill-posed, since infinite recoveries
from an image are feasible. It is difficult to determine which
one is correct, without additional constraints. To address the
ill-posedness, a number of approaches [39], [45], [38], [55]
have been presented over the past decades, which learn to
recover facial components from a single image by exploring
essential geometric information about human faces.
Most, if not all, of contemporary methods rely on large-scale
paired data to achieve the goal. However, on the one hand, it is
difficult to collect sufficient real-world data in practice. While
on the other hand, synthetic data often lacks high-frequency
realistic geometric details, models trained on such data thus
fail to produce high-quality results. To alleviate the pressure
from data, some schemes [38], [45], [65] have been recently
proposed . As a representative, Abrevayaet al.[1] alternatively
treated this problem as a cross-domain translation task. They
trained a sophisticated network structure with deactivable skip
connections. Unfortunately, their performance greatly depends
on the availability of substantial ground-truth data. For loosing
the dependence on extensive paired data, Wanget al.[48]
M. Wang is with the School of Computer Science and Technol-
ogy, Tiangong University. Wenjing Dai is with the Algorithm Depart-
ment of Fitow (Tianjin) Detection Technology Co., Ltd. J. Zhang and X.
Guo are with the College of Intelligence and Computing, Tianjin Uni-
versity, Tianjin 300350, China. E-mail: (autohdr,abigail.dai4@gmail.com,
jwzhang@tju.edu.cn, xj.max.guo@gmail.com). X. Guo is corresponding au-
thor.
 
Input Ex-normal Re-normal Ex-zoom Re-zoom Shading
Fig. 1. Our model can generate high-quality face normals from single face
images. The terms “Ex-normal" and “Re-normal" refer to the coarse exemplar
normal, its refined version. “Ex-zoom" and “Re-zoom" denotes the zoomed-in
exemplar and the zoomed-in refinement normal.
developed a two-stage training network. At the first stage,
they acquired knowledge of face normal priors from limited
paired data, while at the second stage, they enhanced the
learned normal priors by integrating them with facial images
to produce high-quality face normals.
Despite remarkable progress made over last years, several
challenges still remain. First, the scarcity of accurately anno-
tated real-world data limits the performance of model. More-
over, the gap between synthetic and real data poses an obstacle,
models trained on such synthetic data frequently lose their
power on real-world images. Existing methods struggle with
high-frequency geometric details, compromising the fidelity
of estimated face normals. In addition, environmental fluctu-
ations, like changes in background or pose, is another factor
affecting the performance. The absence of multi-scale facial
structural features during training further reduces geometric
detail fidelity.
To tackle the aforementioned challenges, we present an
innovative and effective solution that significantly reduces
the need of extensive training data derived from scannedarXiv:2601.01950v1  [cs.CV]  5 Jan 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 2
3D models or large-scale datasets with ground truth. Our
approach capitalizes on the power of exemplar-based deep
learning to estimate face normals from a single image in real-
world settings. To commence the process, we train a network
on a modest dataset, enabling us to extract coarse normals
from input facial images. By focusing on a smaller dataset,
we overcome the heavy reliance on the availability of large-
scale carefully-annotated data, and achieve a more practical
and accessible solution. To further refine the estimation, we
leverage the estimated coarse normals in an exemplar-based
mapping network. This network takes both the original face
image and the corresponding coarse exemplar as inputs to
generate high-quality normals.
Conceptually, our network is divided into three sub-
networks, including the exemplar encoding network, the face
encoding network, and the feature injection network. The
exemplar encoding network generates an intermediate latent
representation of exemplar. The face encoding network fo-
cuses on learning facial geometric features, while the feature
injection network employs the feature modulation [26] to
synthesize high-fidelity normals. This is accomplished by
modulating feature weights using both global and local facial
geometric information for multiple iterations, resulting in final
high-quality normals (please see Fig. 1 for examples). These
three sub-networks work together and are learned using simple
perceptual and reconstruction losses. The major contributions
of this work can be summarized as follows:
•We propose to logically split the face normal estimation
process into two components. One component attempts
to predict coarse results, while the other performs refine-
ment. This manner can significantly mitigate the pressure
from paired data for practical use.
•We propose a novel framework for high-fidelity face nor-
mal estimation, addressing critical limitations in existing
approaches regarding preserving fine-grained structural
details and consistency. Furthermore, a novel normal
refinement network is designed to refine coarse-normal
predictions by integrating original facial features and
coarse-normal outputs, ensuring high-quality and struc-
turally consistent final results.
•Extensive experiments and ablation studies are conducted
to demonstrate the effectiveness of our design, and reveal
its superiority over current state-of-the-art methods.
The previous version of this manuscript, referred to as
‘HFN’, was published in [48]. Building upon the foundation
laid in ‘HFN’, this journal version takes a further step towards
optimizing the model architecture by removing redundant pa-
rameters. By doing so, we can not only improve computational
efficiency but also enhance the interpretability of design. In
addition, we introduce a self-attention mechanism in the first
stage, which enables the network to learn a normal distribution
tailored to the facial structure, resulting in more accurate
and robust normal estimation during the refinement stage. By
attending to the most relevant facial features, the self-attention
mechanism helps to capture fine-grained details and subtle
variations in the facial geometry. Based on comprehensive
experimental evaluation, we analyze the effectiveness of eachcomponent in our design, shedding light on their individ-
ual contributions to the overall performance. Moreover, we
demonstrate the advantages of our method for face normal
estimation through quantitative metrics and qualitative visual
results. The experimental findings reinforce the efficacy and
robustness of our proposed approach. To promote future
research, encourage collaboration, and facilitate comparisons
from the community, we have made the code publicly available
at https://github.com/AutoHDR/FNR2R.git.
II. RELATED WORK
This section offers a concise overview of representative
works on surface normal estimation and cross-domain tasks,
which are closely related to this paper.
A. Surface normal estimation
Shape from shading (SfS) [17] aims to recover 3D surface
information from images based on shading cues, which is
typically formulated under the Lambertian model [4], [5],
[6], [50], [53]. For exemplar, Barronet al.[6] introduced an
approach that incorporates various priors on shape, reflectance,
and illumination to tackle the SfS problem. They developed a
multi-scale optimization technique to estimate the shape of ob-
jects by considering multiple levels of detail. Xionget al.[50]
proposed a framework based on a quadratic representation
of local shape, which is capable to capture the underlying
geometric structure effectively. However, the mentioned SfS
approaches rely on certain assumptions that may not hold
in unconstrained environments, limiting their applicability to
real-world scenarios.
Alternatively, combining data-driven strategies with SfS
regularizers for surface normal estimation [39], [38] has shown
the advance. Shuet al.[39] proposed an end-to-end generative
network that learns a face-specific disentangled representation
of intrinsic face components. It attempts to capture the com-
plex variations in facial shape by training a generative model
directly on a large-scale face dataset. However, the smoothness
constraint imposed in [39] often causes the loss of high-
frequency information. While Senguptaet al.[38] introduced
a two-step training strategy for surface normal estimation,
leveraging both synthetic and real-world data. Although this
approach has shown promising results, the reliance on prior
knowledge gained purely from synthetic data in [38] limits
the applicability in real-world scenarios. The inherent gap
between synthetic and real data leads to degraded performance
in practice.
Our work is related to those deep learning based methods for
face normal recovery from images, such as [67], [12], [21],
[46], [43], [8], [44]. These approaches aim to estimate the
face normal as part of the overall 3D information recovery,
rather than focusing solely on targeted normal estimation. For
example, Tranet al.[42] proposed a dual-pathway network
that learns additional proxies to bypass strong regularization
and enhance the quality of high-frequency details. However,
the quality of high-frequency detail has a large room to
improve. In contrast, our work introduces exemplar-based
refinement, which leverages a coarse normal as an exemplar

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 3
for normal refinement. Our method is more flexible and robust
in face normal estimation, which can improve the accuracy and
realism of face normals.
B. Cross-domain tasks
The cross-modal/cross-domain learning has gained signifi-
cant attention in various research communities, including style
transfer [14], [32], [62], [58], [23], image inpainting [56], [54],
[34], [19], and image colorization [11], [7], [16], [59], [52].
Exemplar-based learning is one such cross-domain method that
utilizes both an input image and an exemplar image to generate
a target image with the desired content from the input image
and the style from the exemplar image. The existing literature
on exemplar-based surface normal estimation is limited. One
notable study is by Huanget al.[18] proposed to estimate
surface normals from single images. However, this method
greatly relies on a database with known 3D models. While this
approach is effective in certain scenarios, it cannot be applied
on complex cases, and hardly leverages the potential of deep
neural networks for capturing high-frequency information.
More recently, with the emergency of deep neural networks,
there have been efforts to explore exemplar-based techniques
for surface normal estimation. Abrevayaet al.[1] introduced
a cross-modal approach for synthesizing face normals, which
enables the exchange of facial features between the image and
normal domains through deactivated skip connections. But,
both the methods proposed in [18] and [1] suffer from the
loss of high-frequency details in estimated normals. It is worth
to emphasize that, our exemplar-based face normal refinement
aims to overcome the limitations by reducing the need for
extensive data collection, and achieve high-quality estimation
of surface normals more efficiently and effectively.
III. METHOD
Our goal is to estimate high-quality face normals from
single images in a two-stage fashion. An the first stage, we
train the coarse normal predictorCP-Netusing the Photoface
dataset [57], which provides ground truth for face normals.
However, there is a domain gap between the training data and
real-world face images captured in diverse natural scenes. To
narrow the gap between the coarse estimation and the desired
fine-grained output, we further propose a normal refinement
network (NR-Net) as the second stage. In our proposed frame-
work, the predicted coarse normal from the first stage serves as
an exemplar, which is combined with the input face image for
the second stage of normal refinement. In subsequent sections,
we shall provide more detailed information about the design
and implementation.
A. Overview
The whole framework is schematically shown in Fig. 2, The
first stage trains theCP-Netto produce a coarse exemplar (as
referenceR) simply using the generator architecture proposed
in [20]. This network contains a relatively small number of
parameters, which enables fast training. To ensure the accuracy
of the generated normal distribution for corresponding partsof face, we introduce a self-attention-based discriminator.
This discriminator provides additional constraints during the
training process, guiding theCP-Netto produce the con-
sistency of the generated normal distribution. In the second
stage, we construct a normal refinement network (NR-Net)
that fully leverages both face structure featuresf4and normal
featuresz. This structure allows us to effectively combine the
coarse exemplarRwith the detailed information present in
the original face imageI. To facilitate features fusion, we
introduce a feature modulation module [26]d that effectively
integrates the face structure features and normal features to
produce high-fidelity face normalN.
Specifically, the coarse exemplarRis produced from an
input face imageIby theCP-NetasR=CP-Net(I). The
coarse exemplarRis encoded by the normal feature encoder
ERto form normal featuresz=E R(R). Additionally, the
face structure encoderE Ican extract facial structure features
flas follows:
fi=Ei
I(I),ifi= 0,
Ei
I(fi−1),otherwise,(1)
whereiranges from0to5, standing for the layer index of
face structure encoderE I. In addition,f5=fsin Fig 2.
The normal refinement network, denoted asD N, takes the
modulated features and face structure features as input and
produce the refinement high-fidelity normal, which can be
described as follows:
N=D N(dj
N, Fj).(2)
Nis the final normal anddj
N(j∈[0,4])represents the face
structure features obtained via a deconvolution layer followed
by two convolution layers.F jcorresponds to the modulated
features generated by the feature modulation module.
B. Feature modulation module
Based on previous studies [26], [22], [64], the feature mod-
ulation allows for trainable determination of feature weights.
In order to leverage both the structural features that capture
fine face details, and the normal features that represent the
normal distribution, we incorporate the feature modulation
module [26] into our normal refinement network. This module
dynamically adjusts the weights of normal features based on
structure features, employing a multi-scale injection in the
feature space. Mathematically, it can be expressed as follows:
¯ w=w·s·Linear(z),(3)
wherew∈RCi×C j×K×Kand¯ w∈RCi×C j×K×Kare the
original weights and modulated convolution weights, respec-
tively.Kis the kernel size,C iandC jdenote the numbers of
input and output channels, respectively. The function Linear(·)
performs a mapping operation. It takes the normal features
zias input and maps them to the weights at the feature
scalesthat corresponds to the feature map. This mapping
function allows the normal features to influence the weights
of the convolutional layers, enabling the modulation of the

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 4
Coarse exemplar / RReconstruction loss
Output / N
 Input / I
Conv + BN + ReLU
Pooling + Conv Feature modulation module UpConv + ReLU + Conv VGG13 LayersCov + Pooling + DoubleConv Conv Conv + ReLU ReLU + DeConv + BNCoarse prediction network... ...
Fig. 2. Illustration of our exemplar-based face normal refinement framework that can generate high-quality normal by decoding the coarse exemplar and face
structure features.
Coarse exemplar / R
Coarse exemplar / R
Coarse prediction network  
 Patch-based discriminator
Coarse prediction network  
 Self-attention-based discriminator+  
+  (a)
(b)
Fig. 3. The coarse prediction networkCP-Netcomparison between ours (b)
and ‘HFN’ [48] (a). Note that our coarse examplar/R is able to capture the
correlation of normal direction changes where the structure changes greatly.
network’s behavior based on the specific normal distribution.
The normalization can be written as:
Norm(¯ w) =¯ wpP¯ w2+ϵ,(4)
whereϵis a small positive constant used to avoid division by
zero. To ensure the outputs normalized with a unit standard
deviation, we further normalize the dimension of¯ w. Given
the facial structure featuresf j, the modulated featuresmare
obtained via:
Fj=Conv(Norm(¯ w),f j),(5)
where Conv(·)represents a convolution operation.
C. Self-attention module
In ‘HFN’ [48], the generated normals exhibit apparent
artifacts (see Fig. 4 in the mouth and neck areas for examples).
Input HFN-C Ours-C HFN-R Ours-R
Fig. 4. The comparison between coarse and refined normal estimations on
the FFHQ dataset [25]. The labeled as ‘HFN-’ represent the outputs generated
by [48]. The labels ‘-C’ and ‘-R’ indicate the coarse normal and refinement
normal, respectively.
To address this issue, it is important to extract features in
a manner that incorporates attention-driven mechanisms and
considers long-range dependencies. Taking inspiration from
the self-attention mechanism [47], [49], [51], we introduce a
self-attention module in the discriminator for our first stage
training. For instance, as illustrated in Fig. 3, we can acquire
a more comprehensive understanding of the structural charac-
teristics compared to ‘HFN’. This module enables our model
to learn long-range dependencies and improve the overall
performance during second stage refinement. As shown in Fig
4, our approach successfully preserves facial structure details.
In Fig. 5, we present the architecture of the self-attention
module, which consists of three standard1×1convolution
layers.The feature mapsf(x)are first flattened and transposed,
whileg(x)is only flattened. These two sets of feature serve as
the query and key-value pairs for computing attention maps. To
obtain the attention mapϕ, the flattened and transposedf(x)
is multiplied withg(x)and then processed by the softmax
function.Similarly, the feature maph(x)is also flattened and

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 5
conv   conv   conv  Feature maps f(x)  
[B, c, H, W]  Feature maps  
[B, C, H, W]  
Feature maps g(x)  
[B, c, H, W]  
Feature maps h(x)  
[B, C, H, W]  Transpose  
Flatten  
Flatten  Attention map  
 [B, HW , HW]  
 conv  Feature maps  
[B, C, W , H] softmax  
Matrix multiplication  Element-wise sum  Flatten  
 
Feature maps  
[B, C, HW]   
   
Fig. 5. The self-attention module in our framework operates on feature mapsxwith dimensionsB×C×H×W, whereBdenotes the batch size,C
represents the number of channels, andHandWdenote the height and width, respectively. In the self-attention module, we set the intermediate channel size
casC/8, andγis a scalar parameter that is learned during training.
multiplied with the attention mapϕto produce the feature
mapsβ. The final feature mapsoare obtained by combining
the feature mapsβwith the input feature mapsxusing the
formulao=γ×β+x, whereγis a learned parameter. This
combination of features allows the network to enhance relevant
information based on the attention map and preserve important
details from the input.
D. Normal feature encoder
VGG19 is renowned for its capability to learn intricate
patterns via a multilayer nonlinear structure. However, the
normal feature space has relatively low dimensionality (a 256-
dimensional vector in this paper), which results in an excessive
amount of parameter redundancy. This redundancy causes
unnecessary computational overhead and is unsuitable for
efficiently training our refinement network. Thus, we design
a smaller network based on the Feature Pyramid Network
(FPN) structure [30] specifically tailored for normal distri-
bution learning. On the other hand, the FPN-based network
has multiple convolution layers before pooling, enabling it
to construct stronger feature representations while retaining
spatial information. To highlight the difference between our
network architecture and ‘HFN’ [48], we provide a comparison
in Fig. 6. Compared to Fig. 6 (a), our normal feature encoder
ERemploys a three-level FPN structure [30]. The layers
preceding the adaptive average pooling are responsible for
reducing spatial dimensions and extracting normal features.
Subsequently, an adaptive average pooling layer is applied
to aggregate these features, followed by a convolution layer.
The output is denoted aszand represents the final normal
distribution feature utilized for normal refinement.
E. Architecture
The architecture ofCP-Net.In the refinement stage,
the exemplar plays a crucial role in determining the normal
distribution. To effectively learn a robust normal distribution
on faces, we introduce a self-attention-based discriminator [49]
to capture long-range dependencies and improve the modeling
Conv 512Conv & BN 512
DoubleConv 64Conv & BN 64Conv & BN 64Conv & BN 64VGG
Conv & BN 512
Conv & BN 512
Conv & BN 512
AvgPool
Conv 256
Conv 256
Conv 512
 R R
DoubeConv 64Conv & BN 64Conv & BN 64
DoubleConv 64(a)
(b)
AvgPoolFig. 6. Normal features encoder network (E R) comparison between ours (b)
and ‘HFN’ [48] (a). ‘R’ represents as coarse examplar/R produced from the
coarse prediction model.
of normal distribution. The coarse prediction networkCP-Net
is depicted in Fig. 3. This network architecture is based on the
generative networks framework proposed in [20], which has
shown promising results in image-to-image translation tasks.
By leveraging the attention, our coarse exemplar is capable
of learning the correlated normal distribution on different
facial regions, such as around the mouth and eyes, as given
in Fig. 4. This modification can achieve more robust and
accurate representations of normal distribution, compared to
our previous ‘HFN’ [48].
The architecture ofNR-Net.To address noticeable artifacts
left in predicted coarse normals and generate high-fidelity
face normals, the refinement network of ‘HFN’ [48] utilizes
a VGG19 [40] architecture to extract normal features, as
shown in Fig. 6 (a). Although ‘HFN’ is capable of removing

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 6
artifacts with the assistance of the coarse exemplar, the refined
normals may still exhibit artifacts in some cases, as depicted
in Fig. 7. Furthermore, it is worth noting that the size of the
normal distribution feature is relatively small, say256×1×1.
This indicates that the features can be effectively learned by
a smaller network without compromising the quality of the
generated normals. As a result, we remove the VGG network
during the refinement stage, mitigating the need for excessive
computational resources. Moreover, we apply a multi-scale
network to learn the normal distribution across different scales
of face. This allows us to capture the nuances and variations in
the normal direction more effectively, ensuring the generated
normals accurately represent the underlying facial structure.
F . Loss function
The loss function ofCP-Net.We employ a reconstruction
loss to generate coarse normal maps as:
LNormal =1
HWHWX
i
1−N⊤
gt˜Nc
,(6)
whereHandWare the height and width of the images.
Ngtand ˜Ncrepresent the ground-truth normal and predicted
coarse normal, respectively.Tis the transpose of normal map.
To enhance the generalization of the coarse prediction model,
we incorporate an adversarial loss through a self-attention-
based network, which serves as the discriminative network.
The adversarial lossL Dcp for theCP-Netis defined as the
output of the discriminator when given the predicted coarse
normalsN c. The goal is to minimize the error between the
predicted normal distribution and the real normal distribution.
The complete loss function forCP-Netis formulated as:
Lpre=L Normal +λDcpLDcp,(7)
whereλ Dcpis a weighting factor that balances the importance
of the normal reconstruction loss and the discriminator loss.
In this case,λ Dcpis set to0.0001during training.
The loss function ofNR-Net.During the refinement stage,
we only adopt a normal reconstruction loss to achieve our
objective. This loss has two purposes: firstly, it encourages the
fine-grained output normal distribution to resemble the coarse
exemplar normal; secondly, it ensures that the refined normals
have a similar structure to the exemplar. The reconstruction
loss function is identical to Eq. (6).
G. Discussion
Joint training of two stages.The reason for our normal
refinement framework in two steps, with a coarse prediction
networkCP-Netfollowed by a fine-grained refinement net-
workNR-Net, is to address the limitations of training solely
on ground truth normals. When training a model using ground-
truth normals in an end-to-end manner, it heavily relies on the
distribution of the training dataset. This often results in poor
generalization ability when applied to real-world scenarios
with diverse face poses, expressions, and backgrounds. As
shown in Fig.7 and Fig.12, the model trained on a limited
dataset of facial images with ground-truth normals may exhibitpoor generalization to out-of-distribution samples, leading to
inaccurate predictions. By employing a two-stage training
framework, we aim to overcome the above limitations. In
the first phase, a coarse normal is generated as an exemplar,
providing a rough estimation of the face structure and normal
distribution. This coarse normal serves as guidance for the
subsequent fine-grained refinement stage. In the second step,
the fine-grained normal is constructed with guidance from the
exemplar, enabling the model to focus on capturing intricate
normal features and facial structure features, and enabling
accurate refinement of normal. The clear division between
the two stages ensures that each phase is assigned distinct
responsibilities, leading to improved performance and general-
ization ability. Overall, the two-stage training model can better
handle diverse face images with different poses, expressions,
and backgrounds. By incorporating a coarse prediction stage
followed by a fine-grained refinement stage, the model can
leverage the benefits of both two stages and produce high-
quality normal maps that are more robust and accurate in
various real-world scenarios.
Feature injection with relatively low dimensions.The use
of low-dimensional features for encoding the coarse exemplar
normal in our framework provides several advantages. Firstly,
as depicted in Fig. 2, during the training process, we ex-
tract low-dimensional normal features that capture the normal
distribution feature of the input face. These features are not
necessarily aligned with the structure but contain essential
information about the face normal. Secondly, by employing
low-dimensional features for the coarse exemplar, the model
can avoid simply copying the coarse normal to its output
during the fine-grained normal generation process. This is
important because if the model were to directly copy the coarse
normal, it would incur a reconstruction loss during training
[48]. By utilizing low-dimensional normal distribution features
as guidance, the model is trained to generate the fine-grained
normal by leveraging the high-fidelity details present in the
input face image. This approach ensures that the generated
fine-grained normal incorporates the necessary details from
the input face, as well as the accurate normal distribution
on the face derived from the exemplar. The exemplar can
serve as a reference for the model, guiding it to capture the
essential features and accurately refine the normal distribution
to achieve high-quality results.
IV. EXPERIMENTS
A. Experimental Setting
Datasets.We test our model on six different face datasets:
300-W [36], CelebA [31], FFHQ [25], Photoface [57] and
Florence [2]. The 300-W dataset consists of 300 indoor and
300 outdoor images captured in real-world conditions. The
CelebA is a large-scale dataset of real-world faces collected
from the internet, while the FFHQ contains a diverse range of
images with variations in age, ethnicity, and background. The
Photoface dataset comprises photos with four distinct lighting
conditions, and the ground truth normals are estimated using
photometric stereo. For the Photoface dataset, we divided
it into two sets: a training and a test set. The training set

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 7
 Input HFN-C HFN-R Ours-C Ours-R
Fig. 7. Comparison of ‘HFN’ [48] with our method on CelebA-HQ [24] is presented using coarse exemplar (labeled with ‘-C’ ) and refined normal (labeled
with ‘-R’). Our method outperforms ‘HFN’ in terms of generalization ability beyond the dotted line on the right, while both methods achieve similar high-
fidelity results on the left side.
comprises 80% of the data while the remaining 20%, which
consists of approximately 2.5k pairs of image/normal pairs, is
used for evaluation. The Florence dataset comprises 53 3D face
models. We generate facial images and their corresponding
ground truth normal map to evaluate the generalization ability
of our method on completely novel data.
Metrics.One metric, by following previous works [45],
[38], [1], is the mean angular error between the predicted
normal and ground truth normal. To provides a more detailed
evaluation of the accuracy of the estimated normal maps, we
measure the percentage of pixels within the facial region with
angular errors less than20◦,25◦and30◦. In addition, geo-
metric shading and normal error maps are used for qualitative
comparisons, enabling a more comprehensive evaluation.
Implementation details.Our framework is implemented in
PyTorch [33] and was trained on a single NVIDIA 2080 Ti
GPU. During training, we utilized a learning rate of10−4and
adopted the default parameters of the Adam optimizer [27].
Notably, we employed the discriminator [60] as our self-
attention-based discriminator. To make a fair comparison with
the previous work ‘Cross Modal’ [1] and ‘HFN’ [48], we
follow the same approach for cropping the face images.
To assess the generalization capability of our approach, we
utilized a model trained on the CelebA-HQ dataset [24]
without any face cropping, this enabled us to evaluate our
model on complex scenarios. TheCP-NetandNR-Netare
trained for 200k iterations and 150k iterations, respectively. In
the previous version, theE Rmodule employed a pretrained
VGG19 [40] network to extract deep convolutional features
from the ‘relu5_2’ layer. While in this version, we replace the
original VGG19-based architecture with a more efficient and
effective design, as shown in Fig. 6. This modification helpsto streamline the model and improve the performance.
B. Comparison
In Fig. 7, we can observe the comparison between the
coarse exemplars (e.g. ‘HFN-C’ [48]/‘Ours-C’) and the refined
high-fidelity normals (e.g. ‘HFN-R’/‘Ours-R’). The coarse
prediction model, although it converges on the training dataset
and becomes increasingly similar to it, suffers from poor
generalization ability due to the significant differences between
the distributions of the training dataset and real-world data.
While ‘HFN-R’ can achieve more precise outcomes through
secondary optimization of the normal map with the coarse
normal map and input image, it may not perform optimally
in certain specific areas (such as the left side of the dashed
line representing the mouth in Fig. 7). In contrast, our method
is able to overcome these limitations and effectively improve
the quality of the coarse exemplars, resulting in high-fidelity
normals. By leveraging the guidance provided by the input
face image and effectively learning from limited training data,
our method is able to eliminate the artifacts and produce more
accurate and visually pleasing results.
Table I presents a comparison between our methods (‘Ours’)
and other state-of-the-art methods, including ‘Cross-modal-
ft’ [1], ‘SfSNet-ft’[38], ‘Marr Rev’ [3], ‘NiW’[45], and ‘Uber-
Net’ [28]. All results were obtained using a cropped face
image with a resolution of256×256pixels. The table reports
the mean angular errors (in degrees) and the percentages of
errors below<20◦,<25◦, and<30◦. Methods marked
with ‘-ft’ indicate that they were fine-tuned on the Photoface
dataset [57]. Comparing our method to previous approaches
trained on the Photoface, our method consistently achieves

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 8
Input GT CM-N HFN-N Ours-N CM-E HFN-E Ours-E0◦
45◦
Fig. 8. Normal error map comparisons on the Photoface dataset [57]. ‘GT’, ‘-N’ and ‘-E’ are the ground truth normals, predicted normals and error maps,
respectively. ‘CM-’ and ‘HFN-’ are ‘Cross-modal’ [1] and ‘HFN’ [48]. All normal error maps have an angle range of0◦to45◦.
TABLE I
NORMAL RECONSTRUCTION ERRORS ON THEPHOTOFACE[57].
Method Mean±std <20◦<25◦<30◦
Pix2V [37] 33.9±5.6 24.8% 36.1% 47.6%
Extreme [41] 27.0±6.4 37.8% 51.9% 64.5%
3DMM 26.3±10.2 4.3% 56.1% 89.4%
3DDFA [67] 26.0±7.2 40.6% 54.6% 66.4%
SfSNet [38] 25.5±9.3 43.6% 57.5% 68.7%
PRN [12] 24.8±6.8 43.1% 57.4% 69.4%
Cross-modal [1] 22.8±6.5 49.0% 62.9% 74.1%
UberNet [28] 29.1±11.5 30.8% 36.5% 55.2%
NiW [45] 22.0±6.3 36.6% 59.8% 79.6%
Marr Rev [3] 28.3±10.1 31.8% 36.5% 44.4%
SfSNet-ft [38] 12.8±5.4 83.7% 90.8% 94.5%
LAP [63] 12.3±4.5 84.9% 92.4% 96.3%
Cross-modal-ft [1] 12.0±5.3 85.2% 92.0% 95.6%
HFN [48] 11.3±7.7 88.6% 94.4% 97.2%
Ours 10.1±6.5 93.0% 96.8% 98.4%
TABLE II
RECONSTRUCTION ERROR ON THEFLORENCE DATASET[2].
Method Mean±std <20◦<25◦<30◦
Extreme [41] 19.2±2.2 64.7% 51.9% 64.5%
SfSNet [38] 18.7±3.2 63.1% 77.2% 86.7%
3DDFA [67] 14.3±2.3 79.7% 87.3% 91.8%
PRN [12] 14.1±2.2 79.9% 88.2% 92.9%
Cross-modal [1] 11.3±1.5 89.3% 94.6% 96.9%
HFN [48] 10.1±3.4 92.3% 95.6% 97.8%
Ours 9.8±3.2 92.8% 96.1% 98.3%
higher accuracy in normal estimation. The mean angular errors
obtained by our method are significantly lower, indicating
more accurate normal predictions. Additionally, the percent-
ages of errors below different thresholds (<20◦,<25◦,
<30◦) are consistently higher, indicating that our method
produces normals with a higher level of precision.
In Fig.8, we provide qualitative comparisons with the meth-
ods ‘HFN’ [48] and ‘CM’ [1]. The figure showcases the
normal estimations and normal error maps on test samples
from the Photoface dataset [57]. The normal estimation errors
are visualized using color maps, with darker colors indicatingsmaller errors (closer to 0 degrees). Our method produces
more accuracy face normal estimations compared to ‘HFN’
and ‘CM’. The normal estimations obtained by our method
exhibit smaller errors, as evidenced by the darker and more
localized regions in the error maps. This indicates that our
method is able to capture more accurate and detailed normal
information from the input face images, leading to improved
normal estimation performance.
In Table II, we present the quantitative results on the
Florence dataset [2]. To ensure a fair comparison, we follow
the approach of previous work [1] and only compare meth-
ods using aligned output normals of face images. The table
demonstrates that our proposed model outperforms the other
methods, indicating its superior performance in handling out-
of-distribution face images. This result further supports the
effectiveness of our method in generalizing to unseen data
and producing accurate normal estimations.
In Fig. 9, we show the shading maps generated by ‘CM’
[1], ‘HFN’ [48], and ‘Ours’ using seven different angular illu-
minations. The results indicate that the normal maps estimated
by ‘CM’ produce inaccurate shading effects (Shading 2 and
Shading 3) when interacting with light. On the other hand,
both ‘HFN’ and ‘Ours’ accurately depict the shading effects
produced by various angles of light. Upon zooming in on
shading maps, it is evident that ‘HFN’ has some subtle artifacts
(black spots on the nose or mouth), whereas ‘Ours’ effectively
removes these noises. This further confirms that Ours normals
are more precise.
Fig. 10 illustrates a comparison between our method and
the normal produced by the state-of-the-art and represen-
tative face reconstruction algorithm, EMOCA [10]. As our
approach is exclusively focused on the generation of facial
normal maps, we compare the normal maps produced by the
reconstructed facial model generated by EMOCA. The results
clearly demonstrate that our method excels at recovering
intricate and detailed facial normal, while the normal map
generated by the EMOCA method lacks crucial facial details.
Normal mapping is a technique widely used in computer

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 9
Input Normal Shading 1 Shading 2 Shading 3 Shading 4 Shading 5 Shading 6 Shading 7CM HFN Ours CM HFN Ours
Fig. 9. Comparison of normals and shading maps on the CelebA dataset [31]. We generate shading maps using 7 different illumination directions to evaluate
the normal accuracy of ‘CM’ [1], ‘HFN’ [48] and ours.
c
   
   
Input CM HFN Ours EMOCA
Fig. 10. Comparison of normal estimation results with the 3D face recon-
struction method EMOCA [10].
graphics to enhance the visual detail of a 3D model without
increasing its polygon count. By mapping a detailed normal
map onto the surface of a model, the appearance of intricate
surface details such as wrinkles and beards can be convinc-
ingly simulated. Therefore, we map the estimated normal map
   
   Our-En PRN Our-En PRN
Fig. 11. Normal enhanced geometry details. PRN [13] is employed to estimate
the coarse face geometry, ‘Ours-En’ leverages normal mapping to further
enhance the geometric details of the face.
to the face geometry model for enhancing the face details, and
the result is shown in Fig 11. From the figure, it can be seen
that the facial geometry details can be enhanced by means of
normal mapping.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 10
Input BNNNet CBENet
 VGG13 ResNet Ours
Fig. 12. Comparison results for out-of-distribution using different networks
on the CelebA dataset [31]. The last layer of BNNNet [29], CBENet [61],
ResNet [66] and VGG13 [40] has been modified with a tanh function, as
described in the cited papers.
Input HFN-VGG HFN Ours-DC Ours
Fig. 13. The comparison on different normal feature encoder networks (E R).
‘HFN-VGG’ represents the model trained with ‘HFN’ [48] basedE Rwithout
using VGG features, ‘Ours-DC’ uses a double convolution layer followed by
a pooling layer.
V. ABLATION STUDIES
Generalization ability.Fig. 12 and Table III demonstrate
our testing of the performance of different networks by training
end-to-end normal estimation methods using commonly used
classical network structures. Although models trained with
paired data under limited conditions can achieve convergence
on the training data and obtain good evaluation metrics, their
model performance capability is poor when tested on in-the-
wild face images. However, we are able to perform secondary
refinement of the normals using limited data conditions to give
the model strong generalization capabilities.
Normal feature encoder.In order to evaluate the effective-
ness of VGG feature extraction inE R, we conducted a com-
parison between ‘HFN-VGG’ [48] without using pretrained
VGG, ‘HFN’, ‘Ours-DC’ only with a double convolution
followed by a pooling layer and ‘Ours’. The results of this
comparison are presented in Fig. 13 and Table III, which
show the estimated normals are minimally influenced by the
network structure. Specifically, even without the VGG module,
the ‘HFN’ model still produced decent estimated normal maps.
This suggests that a smaller network can effectively extract
normal distribution features of size 256 for the purpose of
normal refinement, as demonstrated by ‘Ours-DC’. These
Input GT PG-N Ours-N PG-E Ours-E
Fig. 14. Normal comparison with different discriminators on the Photo-
face [57]. ‘GT’, ‘PG’, ‘-N’, and ‘-E’ are ground truths, results produced by
Patch-based discriminator [20] model, coarse normals, refined normals and
normal error maps, respectively.
TABLE III
COMPARISON IN NORMAL RECONSTRUCTION ERROR WITH DIFFERENT
CONFIGURATIONS ON THEPHOTOFACE DATASET[57].
Experiments Mean±std <20◦<25◦<30◦
BNNNet [29] 11.7±8.8 86.2% 92.4% 95.5%
CBENet [61] 10.5±7.4 90.6% 95.3% 97.5%
ResNet [66] 7.9±5.4 96.1% 97.9% 98.8%
VGG13 [40] 10.1±8.2 89.8% 94.1% 96.5%
HFN-VGG [48] 11.4±7.9 88.1% 94.1% 96.8%
HFN [48] 11.3±7.7 88.6% 94.4% 97.2%
Ours-DC 11.1±7.2 91.6% 95.3% 98.1%
Ours 10.1±6.5 93.0% 96.8% 98.4%
TABLE IV
COMPARATIVE EXPERIMENTS WITH EXISTING METHODS WHEN RELIABLE
REFERENCES ARE UNAVAILABLE. ‘NOTP’STANDS FOR A NUMBER OF
TRAINING PAIRS USED DURING OUR TRAINING.
Experiments NoTP Mean±std <20◦<25◦<30◦
SfSNet 250k 12.8±5.4 83.7% 90.8% 94.5%
Cross-modal 21k 11.2±7.2 92.1% 95.2% 97.1%
Ours (25%) 2.3k 11.1±6.8 92.3% 95.7% 97.4%
Ours (75%) 7.1k 10.4±7.1 92.7% 96.1% 98.1%
Ours (100%) 9.5k 10.1±6.5 93.0% 96.8% 98.4%
findings imply that the VGG feature extraction module can
be replaced with a simpler and more compact architecture.
In this paper, we utilize the FPN-based structure to extract
normal features while ensuring that we do not lose too much
information. By utilizing the FPN network structure, it is
possible to not only significantly decrease the number of
model parameters but also guarantee the extraction of accurate
features at varying scales.
Self-attention module.We further investigated the effec-
tiveness of the attention mechanism by training an additional
model using PatchGAN’s discriminator [20]. The comparison
between the attention mechanism and the PatchGAN-based
discriminator is shown in Fig. 14. The results clearly indicate
that the attention mechanism enables our model to capture
detailed information from the face image, highlighting its
efficacy. This finding also suggests that the coarse samples
generated in the first stage, utilizing the attention mechanism,
can effectively serve as a reliable input for the normal refine-
ment in the second phase of our framework.
Robustness to varying amounts of training data.To

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 11
Input 25%50%75%Ours
Fig. 15. Normal results with different initializations on the FFHQ dataset [25].
Here, we can find that the coarse prediction model learns the distribution of
normal roughly, and our method can obtain high-fidelity normal with different
iterations in the refined stage.
Input 25%50%75%Ours
Fig. 16. Normal error maps with different initializations on the Photoface
dataset [57].
evaluate the robustness of our method to variations in training
data, we conducted experiments by dividing the training set
and randomly selecting 25%, 50%, and 75% of the data
for model training. The results, shown in Fig. 15 and Fig.
16, indicate that even with a reduced data volume of 25%,
our method performs well not only on data consistent with
the training data but also on data outside the training data
distribution. To provide a more intuitive comparison of the
effects of different data amounts, we present quantitative
evaluation results in Table IV. From the table, it can be
observed that our proposed method exhibits a certain level of
robustness in accurately estimating face normals, even under
conditions of limited training data.
To validate the model, we tested our model on the challeng-
ing dataset, FFHQ [25] and 300-W [36]. As shown in Fig. 17,
our model performs well on those datasets and is particularly
effective in recovering finer-grained face normals. This is
evident in regions where the face geometry changes, such as
near the corners of wrinkled eyes. These results highlight the
robustness and accuracy of our model in capturing detailed
facial geometry.
Running time.We compare our model with ‘CM’ [1]
and ‘HFN’ [48] in running time to verify the efficiency of
            
            FFHQ Normal 300-W Normal
Fig. 17. Normal results on the FFHQ [25] and 300-W [36]. Despite not
having seen such faces during training, our model generalizes reasonably to
these images.
            
(a) (b) (c)
Fig. 18. Results for low-quality images (a), extreme lighting conditions (b),
and faces that are partially covered (c).
TABLE V
COMPARISON OF RUNNING TIMES USING A256×256RESOLUTION
IMAGE,ALL EVALUATED ON THE SAME MACHINE.
Method Parameters(M) FLOPs(G) Running time(ms)
CM [1] 35.2 49 10
HFN [48] 126.4 152 17
Ours 82.9 90 13
our work in Table V. The table shows that the ‘CM’ has
clear advantages in terms of model parameters and FLOPs.
However, this method requires intricate and complex network
structures. Furthermore, it performs worse than ‘HFN’ and
‘Ours’ in terms of normal estimation accuracy. In comparison
to ‘HFN’, ‘Ours’ is capable of not only reducing the model
parameters to some extent but also significantly simplifying
the model while improving normal estimation accuracy.
VI. CONCLUDINGREMARKS
In this paper, we proposed a novel framework for high-
fidelity face normal estimation. Our method draws inspiration
from exemplar-based learning and leverages a coarse exemplar
normal as guidance to generate final fine-grained results. The
key strength of our approach lies in the conversion of the
coarse exemplar normal into normal features and refinement
through feature modulation. This mechanism can not only
boost the estimation both qualitatively and quantitatively, but
also grant it strong generalization capabilities, allowing our
method to effectively handle out-of-distribution face images.
Comprehensive qualitative and quantitative evaluations have

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 12
been conducted to demonstrate that our method outperforms
state-of-the-art approaches by a large margin.
While our method shows its robustness in many challenging
scenarios, such as faces with wrinkles and beards, there are
still failure cases, as illustrated in Fig.18. Images of very low
quality (Fig.18 (a)) and those with extreme lighting conditions
or shading (Fig.18 (c)) can lead to inaccurate normal recon-
structions. Additionally, our method struggles with occluded
face images, as shown in Fig.18 (d)). We recognize that
these unrestricted scenarios pose challenges to our method
and also most existing approaches. It is desirable to explore
more advanced variations to address these challenging and
unrestricted cases more effectively.
REFERENCES
[1] V . F. Abrevaya, A. Boukhayma, P. H. Torr, and E. Boyer. Cross-modal
deep face normals with deactivable skip connections. InCVPR, pages
4979–4989, 2020.
[2] A. D. Bagdanov, A. Del Bimbo, and I. Masi. The florence 2d/3d hybrid
face dataset. InProceedings of the 2011 joint ACM workshop on Human
gesture and behavior understanding, pages 79–80, 2011.
[3] A. Bansal, B. Russell, and A. Gupta. Marr revisited: 2d-3d alignment
via surface normal prediction. InCVPR, pages 5965–5974, 2016.
[4] J. T. Barron and J. Malik. High-frequency shape and albedo from
shading using natural image statistics. InCVPR, pages 2521–2528.
IEEE, 2011.
[5] J. T. Barron and J. Malik. Shape, albedo, and illumination from a single
image of an unknown object. InCVPR, pages 334–341. IEEE, 2012.
[6] J. T. Barron and J. Malik. Shape, illumination, and reflectance from
shading.TPAMI, 37(8):1670–1687, 2014.
[7] Z. Cheng, Q. Yang, and B. Sheng. Deep colorization. InICCV, pages
415–423, 2015.
[8] N. Chinaev, A. Chigorin, and I. Laptev. Mobileface: 3d face reconstruc-
tion with efficient cnn regression. InECCVW, pages 0–0, 2018.
[9] P. Choudhary, S. Mannar, A. Kumar, and M. Son. Real-time face
relighting via adaptive normal mapping. InSIGGRAPH Asia, pages
1–6. 2017.
[10] R. Dan ˇeˇcek, M. J. Black, and T. Bolkart. Emoca: Emotion driven
monocular face capture and animation. InCVPR, pages 20311–20322,
2022.
[11] A. Deshpande, J. Lu, M.-C. Yeh, M. Jin Chong, and D. Forsyth.
Learning diverse image colorization. InCVPR, pages 6837–6845, 2017.
[12] Y . Feng, F. Wu, X. Shao, Y . Wang, and X. Zhou. Joint 3d face recon-
struction and dense alignment with position map regression network. In
ECCV, pages 534–551, 2018.
[13] Y . Feng, F. Wu, X. Shao, Y . Wang, and X. Zhou. Joint 3d face recon-
struction and dense alignment with position map regression network. In
ECCV, 2018.
[14] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using
convolutional neural networks. InCVPR, pages 2414–2423, 2016.
[15] C. Guo, T. Jiang, X. Chen, J. Song, and O. Hilliges. Vid2avatar: 3d
avatar reconstruction from videos in the wild via self-supervised scene
decomposition. InCVPR, pages 12858–12868, 2023.
[16] M. He, D. Chen, J. Liao, P. V . Sander, and L. Yuan. Deep exemplar-
based colorization.TOG, 37(4):1–16, 2018.
[17] B. K. Horn. Obtaining shape from shading information.The psychology
of computer vision, pages 115–155, 1975.
[18] X. Huang, J. Gao, L. Wang, and R. Yang. Examplar-based shape from
shading. In3DIM, pages 349–356. IEEE, 2007.
[19] S. Iizuka, E. Simo-Serra, and H. Ishikawa. Globally and locally
consistent image completion.TOG, 36(4):1–14, 2017.
[20] P. Isola, J.-Y . Zhu, T. Zhou, and A. A. Efros. Image-to-image translation
with conditional adversarial networks. InCVPR, pages 1125–1134,
2017.
[21] A. S. Jackson, A. Bulat, V . Argyriou, and G. Tzimiropoulos. Large
pose 3d face reconstruction from a single image via direct volumetric
cnn regression. InICCV, pages 1031–1039, 2017.
[22] W. Jang, G. Ju, Y . Jung, J. Yang, X. Tong, and S. Lee. Stylecarigan:
caricature generation via stylegan feature map modulation.TOG,
40(4):1–16, 2021.
[23] T. Kang. Multiple gan inversion for exemplar-based image-to-image
translation. InICCV, pages 3515–3522, 2021.
[24] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing
of gans for improved quality, stability, and variation.arXiv preprint
arXiv:1710.10196, 2017.[25] T. Karras, S. Laine, and T. Aila. A style-based generator architecture
for generative adversarial networks. InCVPR, pages 4401–4410, 2019.
[26] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila.
Analyzing and improving the image quality of stylegan. InCVPR, pages
8110–8119, 2020.
[27] D. P. Kingma and B. J. Adam. A method for stochastic optimization.
arxiv preprint arxiv: 14126980. 2014.Cited on, page 50, 2020.
[28] I. Kokkinos. Ubernet: Training a universal convolutional neural network
for low-, mid-, and high-level vision using diverse datasets and limited
memory. InCVPR, pages 6129–6138, 2017.
[29] J. Li, Z. Zhang, X. Liu, C. Feng, X. Wang, L. Lei, and W. Zuo. Spatially
adaptive self-supervised learning for real-world image denoising. In
CVPR, 2023.
[30] T.-Y . Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie.
Feature pyramid networks for object detection. InCVPR, pages 2117–
2125, 2017.
[31] Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in
the wild. InICCV, pages 3730–3738, 2015.
[32] F. Luan, S. Paris, E. Shechtman, and K. Bala. Deep photo style transfer.
InCVPR, pages 4990–4998, 2017.
[33] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An
imperative style, high-performance deep learning library.NeurIPS, 32,
2019.
[34] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros.
Context encoders: Feature learning by inpainting. InCVPR, pages 2536–
2544, 2016.
[35] Y . Ren, G. Li, Y . Chen, T. H. Li, and S. Liu. Pirenderer: Controllable
portrait image generation via semantic neural rendering. InICCV, pages
13759–13768, 2021.
[36] C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, and M. Pantic. 300 faces
in-the-wild challenge: The first facial landmark localization challenge.
InICCVW, pages 397–403, 2013.
[37] M. Sela, E. Richardson, and R. Kimmel. Unrestricted facial geometry
reconstruction using image-to-image translation. InICCV, pages 1576–
1585, 2017.
[38] S. Sengupta, A. Kanazawa, C. D. Castillo, and D. W. Jacobs. Sfsnet:
Learning shape, reflectance and illuminance of facesin the wild’. In
CVPR, pages 6296–6305, 2018.
[39] Z. Shu, E. Yumer, S. Hadap, K. Sunkavalli, E. Shechtman, and D. Sama-
ras. Neural face editing with intrinsic image disentangling. InCVPR,
pages 5541–5550, 2017.
[40] K. Simonyan and A. Zisserman. Very deep convolutional networks for
large-scale image recognition.arXiv preprint arXiv:1409.1556, 2014.
[41] A. T. Tran, T. Hassner, I. Masi, E. Paz, Y . Nirkin, and G. Medioni.
Extreme 3d face reconstruction: Seeing through occlusions. InCVPR,
pages 3935–3944, 2018.
[42] L. Tran, F. Liu, and X. Liu. Towards high-fidelity nonlinear 3d face
morphable model. InCVPR, pages 1126–1135, 2019.
[43] L. Tran and X. Liu. Nonlinear 3d face morphable model. InCVPR,
pages 7346–7355, 2018.
[44] L. Tran and X. Liu. On learning 3d face morphable model from in-the-
wild images.TPAMI, 43(1):157–171, 2019.
[45] G. Trigeorgis, P. Snape, I. Kokkinos, and S. Zafeiriou. Face normals"
in-the-wild" using fully convolutional networks. InCVPR, pages 38–47,
2017.
[46] A. Tuan Tran, T. Hassner, I. Masi, and G. Medioni. Regressing
robust and discriminative 3d morphable models with a very deep neural
network. InCVPR, pages 5163–5172, 2017.
[47] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin. Attention is all you need.NeurIPS, 30,
2017.
[48] M. Wang, C. Wang, X. Guo, and J. Zhang. Towards high-fidelity face
normal estimation. InACM MM, pages 5172–5180, 2022.
[49] X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural networks.
InCVPR, pages 7794–7803, 2018.
[50] Y . Xiong, A. Chakrabarti, R. Basri, S. J. Gortler, D. W. Jacobs, and
T. Zickler. From shading to local shape.TPAMI, 37(1):67–79, 2014.
[51] H. Xu, J. Ma, and X.-P. Zhang. Mef-gan: Multi-exposure image fusion
via generative adversarial networks.TIP, 29:7203–7216, 2020.
[52] Z. Xu, T. Wang, F. Fang, Y . Sheng, and G. Zhang. Stylization-based
architecture for fast deep exemplar colorization. InCVPR, pages 9363–
9372, 2020.
[53] D. Yang and J. Deng. Shape from shading through shape evolution. In
CVPR, pages 3781–3790, 2018.
[54] R. A. Yeh, C. Chen, T. Yian Lim, A. G. Schwing, M. Hasegawa-Johnson,
and M. N. Do. Semantic image inpainting with deep generative models.
InCVPR, pages 5485–5493, 2017.
[55] B. Yu and D. Tao. Heatmap regression via randomized rounding.TPAMI,
2021.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 13
[56] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang. Generative
image inpainting with contextual attention. InCVPR, pages 5505–5514,
2018.
[57] S. Zafeiriou, M. Hansen, G. Atkinson, V . Argyriou, M. Petrou, M. Smith,
and L. Smith. The photoface database. InCVPRW, pages 132–139.
IEEE, 2011.
[58] F. Zhan, Y . Yu, K. Cui, G. Zhang, S. Lu, J. Pan, C. Zhang, F. Ma,
X. Xie, and C. Miao. Unbalanced feature transport for exemplar-based
image translation. InCVPR, pages 15028–15038, 2021.
[59] B. Zhang, M. He, J. Liao, P. V . Sander, L. Yuan, A. Bermak, and
D. Chen. Deep exemplar-based video colorization. InCVPR, pages
8052–8061, 2019.
[60] H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena. Self-attention
generative adversarial networks. InICML, pages 7354–7363. PMLR,
2019.
[61] L. Zhang, Y . He, Q. Zhang, Z. Liu, X. Zhang, and C. Xiao. Document
image shadow removal guided by color-aware background. InCVPR,
pages 1818–1827, 2023.
[62] P. Zhang, B. Zhang, D. Chen, L. Yuan, and F. Wen. Cross-domain
correspondence learning for exemplar-based image translation. InCVPR,
pages 5143–5153, 2020.
[63] Z. Zhang, Y . Ge, R. Chen, Y . Tai, Y . Yan, J. Yang, C. Wang, J. Li, and
F. Huang. Learning to aggregate and personalize 3d face from in-the-
wild photo collection. InCVPR, pages 14214–14224, 2021.
[64] H. Zhao, W. Wu, Y . Liu, and D. He. Color2embed: Fast exemplar-
based image colorization using color embeddings.arXiv preprint
arXiv:2106.08017, 2021.
[65] H. Zhou, S. Hadap, K. Sunkavalli, and D. W. Jacobs. Deep single-image
portrait relighting. InICCV, pages 7194–7202, 2019.
[66] J.-Y . Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image
translation using cycle-consistent adversarial networks. InICCV, pages
2223–2232, 2017.
[67] X. Zhu, Z. Lei, X. Liu, H. Shi, and S. Z. Li. Face alignment across
large poses: A 3d solution. InCVPR, pages 146–155, 2016.
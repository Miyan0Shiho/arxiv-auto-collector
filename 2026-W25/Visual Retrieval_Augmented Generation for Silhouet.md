# Visual Retrieval-Augmented Generation for Silhouette-Guided Animal Art

**Authors**: Quoc-Duy Tran, Anh-Tuan Vo, Trung-Nghia Le

**Published**: 2026-06-16 02:24:21

**PDF URL**: [https://arxiv.org/pdf/2606.17431v1](https://arxiv.org/pdf/2606.17431v1)

## Abstract
Generative AI has advanced the ability to render photorealistic or artistic images, yet it remains limited in a key aspect of human creativity: interpreting ambiguous shapes. This phenomenon, rooted in pareidolia, allows humans to perceive meaningful forms in random patterns such as clouds, stones, or leaves. To computationally replicate this imaginative process, we introduce Visual Retrieval-Augmented Generation (Visual-RAG), a framework that generates animal art directly from natural silhouettes. Our method retrieves structurally similar animal shapes from a curated corpus of 28,586 high-quality silhouettes and uses them as reference exemplars to guide diffusion-based generation with ControlNet and IP-Adapter. Ablation studies confirm that shape Context with RANSAC provides the most accurate alignment, while removing shape standardization reduces the inlier ratio to just 13.4\%, underscoring the importance of structural fidelity in Visual-RAG. A user study with 12 participants evaluated the outputs in terms of aesthetics, silhouette fidelity, and overall impression. Results reveal that while Visual-RAG provides plausible interpretations, challenges remain in achieving high perceptual impact. This work lays the foundation for computational pareidolia, showing how machines can contribute to the early stages of imaginative discovery.

## Full Text


<!-- PDF content starts -->

Visual Retrieval-Augmented Generation for
Silhouette-Guided Animal Art
Quoc-Duy Tran1,2, Anh-Tuan Vo1,2, and Trung-Nghia Le⋆1,2
1University of Science, VNU-HCM, Ho Chi Minh, Vietnam
2Vietnam National University, Ho Chi Minh, Vietnam
{22120082,22120406}@student.hcmus.edu.vn
ltnghia@fit.hcmus.edu.vn
Abstract.Generative AI has advanced the ability to render photoreal-
istic or artistic images, yet it remains limited in a key aspect of human
creativity: interpreting ambiguous shapes. This phenomenon, rooted in
pareidolia, allows humans to perceive meaningful forms in random pat-
terns such as clouds, stones, or leaves. To computationally replicate this
imaginative process, we introduce Visual Retrieval-Augmented Genera-
tion (Visual-RAG), a framework that generates animal art directly from
natural silhouettes. Our method retrieves structurally similar animal
shapes from a curated corpus of 28,586 high-quality silhouettes and uses
them as reference exemplars to guide diffusion-based generation with
ControlNet and IP-Adapter. Ablation studies confirm that shape Con-
text with RANSAC provides the most accurate alignment, while remov-
ing shape standardization reduces the inlier ratio to just 13.4%, under-
scoring the importance of structural fidelity in Visual-RAG. A user study
with 12 participants evaluated the outputs in terms of aesthetics, silhou-
ette fidelity, and overall impression. Results reveal that while Visual-
RAG provides plausible interpretations, challenges remain in achieving
high perceptual impact. This work lays the foundation for computational
pareidolia, showing how machines can contribute to the early stages of
imaginative discovery.
Keywords:Computational Creativity·Retrieval-Augmented Genera-
tion·Shape Analysis·Generative AI·Pareidolia.
1 Introduction
Creating artistic imagery from ambiguous natural forms is a direct exercise in
divergent perception, a cognitive ability linked to creativity [3]. This capacity
is rooted in pareidolia, a phenomenon considered the basis of iconographic art,
where viewers resolve induced ambiguity [2]. However, "seeing things for what
they could be" is cognitively challenging. As a rapid, subjective, and fleeting
shortcut rooted in survival instincts [2], pareidolia is difficult to sustain for artis-
tic purposes; studies show more creative individuals experience it more often [3].
⋆Corresponding author.arXiv:2606.17431v1  [cs.CV]  16 Jun 2026

2 Quoc-Duy Tran et al.
Bridging this perceptual gap to finished artwork demands both imagination and
technical skill.
Modern generative AI, despite its advances, provides limited support for this
creative process. Text-to-image models such as Stable Diffusion [18] require ex-
plicit prompts, burdening the user with conceptualization. Shape-conditioned
systems like ControlNet [25] excel at rendering but cannot autonomously in-
terpret ambiguity, lacking ’appreciation’ for their own artifacts [7]. Meanwhile,
existing computational pareidolia approaches [19,20] remain confined to narrow
domains. Current frameworks are thus tools, not creative partners, failing ’per-
ceptually grounded’ creativity [7].
To address this gap, we propose a computational framework that automates
the imaginative interpretation of natural object silhouettes by integrating re-
trieval with generative. Our visual retrieval-augmented generation, Visual-RAG
in short, employs an IP-Adapter to transfer the visual appearance of a retrieved
animal image into a new generation while constraining the output to the given
silhouette. To support this research, we curated a dataset of 28,586 masked ani-
mal instances derived from OpenImagesV7 [10], providing a large-scale resource
for retrieval-based experimentation.
We further conducted experiments to evaluate the proposed method. The
ablation study demonstrates that reliable geometric alignment is far more im-
portant than raw speed for silhouette-guided generation. Shape Context with
RANSAC consistently provided the most accurate matches, while shape stan-
dardization proved essential for preserving structural fidelity. Together, these
components form the foundation that enables Visual-RAG to produce coherent
and perceptually meaningful outputs. Meanwhile, the user study focuses on sev-
eral dimensions of creative quality, including aesthetic appeal, shape fidelity, and
overall impression. This study advances perceptually grounded creative AI by
introducing a novel framework and revealing key trade-offs between generative
and retrieval-based strategies.
Our contributions are as follows:
–WeintroduceVisual-RAG,acomputationalframeworkforgeneratinganimal
art directly from ambiguous silhouettes.
–We construct and release a curated dataset of 28,586 masked animal images
from OpenImagesV7.
2 Related Work
Shape-based retrievalhas progressed from classical geometric descriptors
to modern learning-based methods. Early techniques such as Shape Context
(SC) [4] and Inner-Distance Shape Context (IDSC) [12] captured boundary and
articulation information, while faster descriptors like Hierarchical String Cuts
(HSC) [21] traded accuracy for efficiency, often struggling with highly artic-
ulated shapes. More recent advances employ convolutional networks [15] and
graph neural networks [13], offering learned shape representations but requiring
large annotated datasets and often showing limited generalization to naturalistic
silhouettes in creative contexts. To address scalability, post-processing strategies

Visual-RAG for Silhouette-Guided Animal Art 3
such as Online-to-Offline (O2O) [27] shift computational load to offline prepro-
cessing, improving efficiency for large-scale retrieval. Our approach adapts this
principle while emphasizing retrieval quality over speed by incorporating robust
geometric verification with RANSAC-based alignment. This balance ensures re-
liable matching performance in the open-ended and diverse scenarios central to
creative applications.
ConditionalimagegenerationhasadvancedfromearlyGAN-basedmeth-
ods such as Pix2Pix [8] to diffusion models with fine-grained structural condi-
tioning exemplified by ControlNet [25]. While these models excel at render-
ing, they still depend on explicit prompts and lack autonomous interpretative
ability. Related paradigms such as content-aware inpainting, demonstrated by
EdgeConnect [14] and DeepFillv2 [24], show strong contextual reasoning yet
remain limited to completing missing regions rather than generating novel inter-
pretations. More recently, IP-Adapter [23] has extended diffusion models with
appearance transfer, enabling reference images to guide generation while pre-
serving structural constraints. Building on this capability, our retrieval-based
approach transfers animal appearance characteristics into arbitrary silhouettes,
bridging perceptual ambiguity with creative synthesis.
OriginallyintroducedinNLP[11],retrieval-augmentedgeneration(RAG)
integrates parametric knowledge with external retrieval and has been applied to
tasks such as few-shot learning [6] and image captioning [17]. We extend this
idea to creative generation, where retrieved examples act as inspirations that
expand interpretative possibilities. This direction echoes "learning from imagi-
nary data" [22], but shifts hallucination from refining classification boundaries to
enabling creative exploration. Multimodal retrieval has been advanced through
Vision-Language Models (VLM) like CLIP [16]. Building on these foundations,
our framework demonstrates how shape-based retrieval can inform semantic in-
terpretation through large language models, supporting perceptually grounded
creative synthesis.
3 Proposed Method
Givenahigh-qualitycorpusofanimalsilhouettes,weaimtoleveragethisdataby
finding a structurally compatible instance from the corpus to serve as a detailed
visual reference. Figure 1 illustrates an overview of this approach.
Silhouette Segmentation.We adopt a two-step segmentation pipeline
to isolate the target object without manual annotation. Grounding DINO [26]
first detects candidate bounding boxesB=b 1, b2, . . . , b nfrom open-vocabulary
prompts (e.g., “stone,” “cloud,” “fire”). The highest-confidence boxbiis then
passed to SAM [9], which outputs the final segmentation maskM.
Geometric Feature Extraction.Geometric fidelity is critical in this re-
trieval stage. Although the O2O framework [27] provides an effective balance
between speed and accuracy, its reliance on fast descriptors during the online
stage is suboptimal for our task. Meanwhile, HSC [21] captures only coarse
global structure, which often yields geometrically incongruent matches in our
highly articulated animal dataset. Because our generative model depends on

4 Quoc-Duy Tran et al.
QUERY
 # 1 Harbor Seal
 # 2 Snake
 # 3 Otter
 # 4 Oyster
 # 5 Tortoise
# 6 Duck
 # 7 Tortoise
 # 8 Oyster
 # 9 Bat
 # 10 Panda
# 2 Oyster
 QUERY
(Standardized)
# 3 Bat
 # 4 Otter
 # 5 Panda
# 6 Duck
 # 7 Tortoise
 # 8 Snake
 # 9 Harbor Seal
 # 10 Oyster
# 1 Tortoise
 QUERY
Fig.1: Overview of the visual retrieval-augmented generation pipeline.
Fig.2: Shape standardization process.(1)Original binary mask.(2)Rotation
standardization.(3)Cropping.(4)Scale standardization.(5)Target canvas.
(6)Final centered output.
structurally coherent references, such mismatches cannot be corrected down-
stream. To mitigate this issue, we adopt the classic SC descriptor [4], which
offers articulation-aware matching at the cost of increased computational over-
head. This choice prioritizes structural accuracy, ensuring reliable retrieval for
subsequent generation.
Shape Standardization.To ensure consistent geometric comparison, we
apply a three-stage normalization process to all silhouettes (Fig. 2). Given an
input maskM, we first extract its largest connected component and compute
theminimum-areaboundingrectangletodeterminetheprimaryorientation.The
shape is then rotated such that its longer axis aligns horizontally, wherew box
andhboxdenote the rectangle dimensions. Next, the rotated shape is scaled to
fit within a256×256canvas using a scaling factorα= 0.9to preserve a uniform
margin, and then centered. Finally, to address the 180°ambiguity in orientation
alignment, we enforce a canonical pose. We compute the centroid’sx-coordinate
cx(from image moments) and horizontally flip the shape ifc x< W/2, thus
ensuring its center of mass always lies in the right half of the canvas.
Shape Matching.Our shape matching process begins by computing SC
descriptors [4] for each standardized silhouette. Similarly to the work of Be-
longie et al. [4], we uniformly sample 100 contour points, a choice that balances
representational detail with computational efficiency. For each pointp i, its de-

Visual-RAG for Silhouette-Guided Animal Art 5
Stable Diffusion 1.5 InpaintingIP - Adapter ControlNet - MaskVLM
The koala is sleeping 
while hugging a branch.
Fig.3: Reference-Guided generation pipeline.
scriptorh iis defined as a log-polar histogram capturing the spatial distribution
of all other contour points, using the standard configuration of 12 angular and
5 logarithmic radial bins.
To compare a query shapeQwith a candidateC, we establish an optimal
one-to-one correspondence between their contour point sets,pQ
iandpC
j. The
cost of matching two points is measured by the chi-squared distance between
their descriptors,d χ2(hQ
i, hC
j). The overall shape similarity is then defined as
the minimum assignment cost:D(Q, C) = min π∈ΠPN
i=1dχ2
hQ
i, hC
π(i)
,where
πdenotes a permutation in the assignment spaceΠandNis the number of sam-
pled contour points. This optimization is solved efficiently using the Hungarian
algorithm, ensuring globally optimal point correspondences.
Geometric Re-ranking.We employ a geometric verification stage for fur-
ther re-ranking the top-10 candidates. This step eliminates matches whose point
correspondences are geometrically inconsistent. For each candidate, a RANSAC-
based estimator is used to compute the optimal affine transformation that aligns
the matched points. The final selection is determined by geometric stability, de-
fined as the lowest mean re-projection error across the inlier correspondences.
This ensures that retrieved references are not only visually similar but also struc-
turally coherent, providing reliable guidance for subsequent generation.
Inverse Transformation for Alignment.The final step aligns the high-
resolution retrieved RGB imageRand its maskMto the coordinate space
of the original input queryQ. LetA 1denote the affine transformation that
maps the retrieved maskMto its standardized formM′, and letA 2denote
the transformation that maps the queryQto its standardized formQ′. From
the re-ranking step, we obtain a transformationTthat alignsM′withQ′. The
composite transformationT final, which maps the original retrieved image space
to the original query image space, is defined asT final=A−1
2·T·A 1. Applying
Tfinalto the retrieved imageRand maskMyields the aligned outputsR′′and
M′′, which are spatially coherent with the input queryQ.
Reference-Guided Generation.As shown in Fig. 3, the transformation
process begins by retrieving a segmentation mask, its category label, and a vi-
sually similar reference image from the query results. The final image is then
synthesized with Stable Diffusion 1.5 inpainting [18], guided by three comple-
mentary inputs. ControlNet [25] uses the input mask to preserve the target

6 Quoc-Duy Tran et al.
(a) Semantic Filtering
 (b) Contour Integrity
 (c) Normalization
 (d) Mask Disparity
Fig.4: Key filtering steps in our curation pipeline: semantic validation, contour
integrity, normalization, and outlier removal.
silhouette, ensuring that the generated output conforms to the original shape.
In parallel, IP-Adapter [23] transfers visual details from the reference image,
including colors, textures, and overall style. To further align the result with the
intended subject, a concise textual description of the animal is automatically
generated by Gemini Flash 2.5 and incorporated into the prompt.
Image Blending.In the final stage, the generated imageI genis integrated
with the original inputI origto preserve both the object silhouette and the sur-
rounding background. We apply soft blending within the foreground maskMby
interpolatingI genandIorigwith a weighting factorα∈[0,1], while retaining the
original background outside the mask. The final outputI finalis computed as:
Ifinal=α·(M⊙I gen) + (1−α)·(M⊙I orig) + (1−M)⊙I orig,(1)
where⊙denotes element-wise multiplication. In our experiments, we setα= 0.5
to achieve a balanced integration of generated content and original detail.
4 Animal Silhouette Corpus
Our methods rely on a large, high-quality corpus of animal shapes. To meet this
need, we constructed theAnimal Silhouette Corpus, curated from the Open
Images V7 dataset [10]. Its construction followed a multi-stage pipeline to ensure
each silhouette is both clean and representative of an animal’s true form.
SourceandSemanticFiltering.Webeganwith102,062instancesegmen-
tation masks from theAnimalsuperclass in Open Images V7 [5]. For semantic
integrity, we removedTeddy bearannotations (Fig. 4a), leaving 101,014 masks.
Contour Integrity and Normalization.Each mask was required to
represent a contiguous silhouette. Many were fragmented (e.g., animals behind
fences) or noisy. We retained a maskMonly if its largest contourC maxcovered
at least 95% of its area. Invalid masks were discarded, and valid ones normal-
ized by keeping onlyC max(Fig. 4c). This removed 8,118 masks, leaving 92,896
instances.
Data-driven Thresholds.To guide filtering, we analyzed Mask Area Dis-
parity and IoU distributions. Both revealed outliers and overlaps but lacked clear
cutoffs, so thresholds were set by manual inspection of 5,000 samples.

Visual-RAG for Silhouette-Guided Animal Art 7
Table 1: Comparison of initial shape retrieval methods. Metrics are averaged
over all query–retrieved pairs.
Method Inlier Ratio Residual Error IoU Avg. Time (s)
HSC [21] 0.3030 69.49 0.41814.9
IDSC + DP [12] 0.3124 63.63 0.4143 62.6
SC + DP [4] (Ours)0.3168 60.82 0.514459.3
OutlierFiltering.Maskssmallerthanone-fifththelargestin-classinstance
were discarded, removing 6,792 annotations (Fig. 4d), leaving 86,104 masks.
Overlap and Duplication Removal.To handle overlaps, we applied IoU-
based rules:
–Duplicates (IoU >0.9):retain larger mask.
–Occlusions (0.6< IoU≤0.9):discard larger mask.
–Ambiguous (0.2< IoU≤0.6):discard both.
–Containment (IoU≤0.2):removeAif Area(I)/Area(A)>0.7.
This resolved 9,145 conflicts, yielding 76,959 validated masks.
Class Balancing and Final Corpus.Despite filtering, class imbalance
remained, with counts ranging from 23 to 15,376. To normalize representation,
we capped each class at 500, sampling where necessary. The final Animal Sil-
houette Corpus contains28,586 silhouettes across 72 species, serving as the
foundation for our method.
5 Ablation Study
5.1 Geometry-based Retrieval Evaluation
We conductedexperimentsto evaluatethe effectiveness ofgeometric components
used in different stages of our retrieval process.
Initial RetrievalWe compared our chosen initial retrieval method (SC
+ DP [4]) against two alternatives: IDSC + DP [12] and the speed-oriented
HSC [21], using the full corpus. As shown in Table 1, HSC achieved the fastest
runtime but performed worst in terms of geometric fidelity, making it unsuitable
for providing high-quality visual references. The choice between SC and IDSC
was more subtle. Our animal corpus spans a broad range of shapes, from highly
articulated to simple forms, while the query silhouettes are almost exclusively
non-articulated. In this cross-domain setting, SC’s general-purpose descriptors
proved more effective, providing consistent similarity measures across the diverse
corpus. The results confirm this, with SC achieving the best performance across
all metrics, making it the clear choice for our framework.
Re-rankingIn the re-ranking stage, we evaluated our RANSAC-based ge-
ometric verification against a learning-based point cloud registration method
(i.e., LP [1]). Results in Table 2 show that while LP achieved a higher inlier
ratio, our RANSAC approach produced a substantially lower residual error and
much higher post-alignment IoU. These results demonstrate that RANSAC pro-
vides more precise geometric alignment, which is essential for the Visual-RAG
pipeline.

8 Quoc-Duy Tran et al.
Table 2: Performance comparison of re-ranking methods.
Method Inlier Ratio Residual Error IoU Avg. Time (s)
SC + DP + LP0.537423.0210 0.1241 2.394
SC + DP + RANSAC (Ours) 0.45270.4307 0.2761 2.262
Table 3: Comparison of our full retrieval pipeline with the O2O framework on a
subset of 2,153 images.
Method Inlier Ratio Residual Error IoU Avg. Time (s)
O2O-r + SC + HSC + LP [27] 30.62 77.29 31.870.224
SC + DP + RANSAC (Ours)34.68 56.82 54.444.43
Query (Standardized)
 Top 1
 Top 2
 Top 3
 Top 4
 Top 5
Query (Original)
Top 1
Top 2
Top 3
Top 4
 Top 5
Query (Original)
Query (Standardized)
 Top 1
Top 1
Top 2
Top 2
Top 3
Top 3
 Top 4
Top 4
 Top 5
Top 5
Fig.5: Visualize top 5 results from retrieval by SC + DP + RANSAC, with and
without shape standardization.
Comparison with O2O FrameworkWe also compared our complete
pipeline against the state-of-the-art O2O framework [27]. This experiment was
conducted on a representative subset of 2,153 images, created by sampling at
most 30 images per class, since building the O2O offline index for the full corpus
of 28,586 images was computationally prohibitive. As shown in Table 3, O2O
achieved substantially faster retrieval, with an average time of 0.224 seconds.
However, our method outperformed O2O across all geometric fidelity metrics,
achievingamuchhigherpost-alignmentIoU(54.44%vs.31.87%)andlowerresid-
ual error. For our approach, where the quality of a single retrieved instance is
critical for guiding creative generation, this accuracy gain justifies the additional
computational cost.
5.2 Effectiveness of Shape Standardization
The importance of our shape standardization module is demonstrated in Fig. 5.
Without a canonical representation for scale, rotation, and translation, the re-
trieval process produces geometrically inconsistent matches, especially for com-
plex or articulated silhouettes. In such cases, the retrieved shapes fail to preserve
thestructuralessenceofthequery,asreflectedinthequalitativeexamples.Quan-
titatively, the average inlier ratio without standardization drops to just 13.4%,
underscoring that this step is indispensable for achieving high-quality retrieval
and reliable downstream geometric alignment.
5.3 Comparison with Naive Diffusion Baseline
To validate the necessity of our structured Visual-RAG framework, we compared
it with a naive diffusion baseline that combines a ControlNet-Canny model [25]

Visual-RAG for Silhouette-Guided Animal Art 9
B a s e l i n e
O u r s
Fig.6: Comparison with the Naive ControlNet baseline (using a generic "animal"
prompt).
Fig.7: Sample results from the user study. Each input image (top row) is paired
with the corresponding outputs generated by Visual-RAG (bottom row).
and a generic text prompt (e.g., “an artistic painting of an animal”). Using the
same Stable Diffusion 1.5 inpainting setup, this baseline adheres to the silhou-
ette but merely hallucinates random animals that fit the shape. As shown in
Figure 6, its outputs are inconsistent, lack creative control, and often fail to
produce coherent results.
6 User Study
6.1 Experimental Design
To evaluate Visual-RAG, we selected 40 ambiguous natural images, including
stones, clouds, fire, and leaves. Each input was processed to generate one output
image, resulting in a test set of 40 generated samples (Fig. 7). Twelve partic-
ipants, drawn from both technical and non-technical backgrounds, rated each
output on a 5-point Likert scale across four dimensions: Aesthetics (visual ap-
peal), Shape Fitness (silhouette adherence), Impression (creative impact), and
Overall Quality. To mitigate bias, images were shown in randomized order, and
participants were not informed about the underlying method.
6.2 Results
As summarized in Table 4, Visual-RAG’s performance consistently fell below
the neutral midpoint of 3.0 across all metrics, with mean ratings of 2.46 for
aesthetics, 2.62 for shape fitness, and 2.46 for impression. These results suggest

10 Quoc-Duy Tran et al.
Table 4: Mean scores and standard deviations from the user study. The highest
score for each metric is highlighted in bold.
Method Aesthetics Shape Fitness Impression Overall
Visual-RAG 2.46±1.23 2.62±1.29 2.46±1.29 2.51±1.27
that while the framework can produce coherent outputs, its creative impact re-
mains limited. In particular, participants noted frequent failures where retrieved
exemplars were incomplete or ambiguous, leading to outputs that lacked visual
coherence. Our study revealed a strong positive correlation(r≈0.75)between
shape fitness and aesthetics, underscoring that strict adherence to the source
silhouette is a key determinant of perceived quality. This finding suggests that
the psychological illusion of pareidolia depends critically on structural fidelity:
when the generated image conforms closely to the input shape, viewers experi-
ence the creative discovery of “seeing” an animal within it; when fidelity breaks
down, the illusion collapses, and the result is perceived as an unrelated object
superimposed on the silhouette. Improving structural alignment in future itera-
tions of Visual-RAG is therefore central to enhancing both creative impact and
perceptual quality.
7 Conclusion
In this paper, we proposed a framework for generating animal art from abstract
silhouettes, directly tackling the challenge of computational pareidolia. Our ex-
periments demonstrate that retrieval-based conditioning can produce visually
coherent and shape-faithful generations, while a user study highlights the im-
portance of silhouette adherence for aesthetic appeal. Future work should move
beyond single exemplars by developing adaptive strategies for incomplete or
ambiguous references, expanding the 72-class dictionary to foster creativity, and
advancing shape decomposition and retrieval methods to improve stability and
imagination.
Acknowledgments.This research is funded by Vietnam National Foundation for
Science and Technology Development (NAFOSTED) under Grant Number 102.05-
2023.31. This research used the GPUs provided by the Intelligent Systems Lab at
the Faculty of Information Technology, University of Science, VNU-HCM.
References
1. Bai, X., Yang, X., Latecki, L.J., Liu, W., Tu, Z.: Learning context-sensitive shape
similarity by graph transduction. IEEE TPAMI32(5), 861–874 (2010)
2. Bednarik, R.: Rock art and pareidolia. Rock Art Research33, 167–181 (11 2016)
3. Bellemare, A., Harel, Y., O’Byrne, J., Mageau, G., Dietrich, A., Jerbi, K.: Process-
ing visual ambiguity in fractal patterns: Pareidolia as a sign of creativity. SSRN
Electronic Journal (01 2022)
4. Belongie, S., Malik, J., Puzicha, J.: Shape matching and object recognition using
shape contexts. IEEE TPAMI24(4), 509–522 (2002)

Visual-RAG for Silhouette-Guided Animal Art 11
5. Benenson, R., Popov, S., Ferrari, V.: Large-scale interactive object segmentation
with human annotators. In: CVPR (2019)
6. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-
adapter: Better vision-language models with feature adapters (2025)
7. Heath, D., Ventura, D.: Before a computer can draw, it must first learn to see. In:
ICCC (2016)
8. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with condi-
tional adversarial networks (2018)
9. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything
(2023)
10. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Ka-
mali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., Ferrari, V.: The open
images dataset v4: Unified image classification, object detection, and visual rela-
tionship detection at scale. IJCV (2020)
11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., tau Yih, W., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks (2021)
12. Ling,H.,Jacobs,D.W.:Shapeclassificationusingtheinner-distance.IEEETPAMI
29(2), 286–299 (2007)
13. Monti, F., Boscaini, D., Masci, J., Rodolà, E., Svoboda, J., Bronstein, M.M.: Ge-
ometric deep learning on graphs and manifolds using mixture model cnns (2016)
14. Nazeri, K., Ng, E., Joseph, T., Qureshi, F.Z., Ebrahimi, M.: Edgeconnect: Gener-
ative image inpainting with adversarial edge learning (2019)
15. Radenović, F., Tolias, G., Chum, O.: Deep shape matching (2018)
16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable
visual models from natural language supervision (2021)
17. Ramos, R., Elliott, D., Martins, B.: Retrieval-augmented image captioning (2023)
18. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution
image synthesis with latent diffusion models. In: CVPR (2022)
19. Song, L., Wu, W., Fu, C., Qian, C., Loy, C.C., He, R.: Everything’s talkin’: Parei-
dolia face reenactment (2021)
20. Wan, Z., Xu, D., Wang, Z., Wang, J., Luo, J.: Cloud2sketch: Augmenting clouds
with imaginary sketches. In: MM’ 22. p. 2441–2451. MM ’22 (2022)
21. Wang, B., Gao, Y.: Hierarchical string cuts: A translation, rotation, scale, and
mirror invariant descriptor for fast shape retrieval. IEEE TIP23(9), 4101–4111
(2014)
22. Wang,Y.X.,Girshick,R.,Hebert,M.,Hariharan,B.:Low-shotlearningfromimag-
inary data (2018)
23. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image
prompt adapter for text-to-image diffusion models (2023)
24. Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., Huang, T.: Free-form image inpainting
with gated convolution (2019)
25. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image
diffusion models (2023)
26. Zhang, S., Chen, X., Wang, X., Liu, Z., Liu, S., Li, M., Luo, P.: Grounding dino:
Marrying dino with grounded pre-training for open-set object detection. arXiv
preprint arXiv:2303.05499 (2023), accessed August 2025
27. Zheng, Y., Guo, B., Yan, Y., He, W.: O2o method for fast 2d shape retrieval. IEEE
TIP28(11), 5366–5378 (2019)
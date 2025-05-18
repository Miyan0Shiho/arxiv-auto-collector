# Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models

**Authors**: Hussien Al-Asi, Jordan P Reynolds, Shweta Agarwal, Bryan J Dangott, Aziza Nassar, Zeynettin Akkus

**Published**: 2025-05-13 14:01:35

**PDF URL**: [http://arxiv.org/pdf/2505.08590v1](http://arxiv.org/pdf/2505.08590v1)

## Abstract
Advancements in artificial intelligence (AI) are transforming pathology by
integrat-ing large language models (LLMs) with retrieval-augmented generation
(RAG) and domain-specific foundation models. This study explores the
application of RAG-enhanced LLMs coupled with pathology foundation models for
thyroid cytology diagnosis, addressing challenges in cytological
interpretation, standardization, and diagnostic accuracy. By leveraging a
curated knowledge base, RAG facilitates dy-namic retrieval of relevant case
studies, diagnostic criteria, and expert interpreta-tion, improving the
contextual understanding of LLMs. Meanwhile, pathology foun-dation models,
trained on high-resolution pathology images, refine feature extrac-tion and
classification capabilities. The fusion of these AI-driven approaches en-hances
diagnostic consistency, reduces variability, and supports pathologists in
dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate
that integrating RAG with pathology-specific LLMs significantly improves
diagnostic efficiency and interpretability, paving the way for AI-assisted
thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for
correct prediction of surgi-cal pathology diagnosis from thyroid cytology
samples.

## Full Text


<!-- PDF content starts -->

Enhancing Thyroid Cytology Diagnosis with RAG -Optimized LLMs and Pathol-
ogy Foundation Models  
1 Hussien Al -Asi, 1 Jordan P Reynolds , 1 Shweta Agarwal , 1 Bryan J Dangott , 1 Aziza Nassar,    
1 Zeynett in Akkus   
akkus.zeynettin@mayo.edu 
1    Computational P atholo gy and AI /Informatics , Department of L ab Medic ine and Pathology, 
Jacksonville, FL, USA  
Abstract. Advancements in artificial intelligence (AI) are transforming pathology 
by integrating large language models (LLMs) with retrieval -augmented generation 
(RAG) and domain -specific foundation models. This study explores the application of 
RAG -enhanced LLMs co upled with pathology foundation models for thyroid cytology 
diagnosis, addressing challenges in cytological interpretation, standardization, and di-
agnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dynamic retrieval of relevant case studies, diagnostic criteria, and expert interpretation , improv-
ing the contextual understanding of LLMs. Meanwhile, pathology foundation models, 
trained on high- resolution pathology  images, refine feature extraction and classification 
capabilities. The fusion of these AI -driven approaches enhances diagnostic consistency, 
reduces variability, and supports pathologists in distinguishing benign from malignant 
thyroid lesions. Our result s demonstrate that integrating RAG with pathology- specific 
LLMs significan tly improves diagnostic efficiency and interpretability, paving the way 
for AI- assisted thyroid cytopathology , with foundation model UNI achieving AUC 
0.73- 0.93 for correct prediction of surgical pathology diagnosis from thyroid cytology 
samples . 
Keywords:  Cytology,  thyroid,  pathology foundation models, retrieved aug-
mented generation . 
1 Introduction 
Recent advancements in computational pathology have increasingly emphasized the quantitative analysis of tissue images, tackling the inherent challenges of whole -slide 
images (WSIs), including their high resolution and morphological complexity [1-4]. 
These factors have traditionally impeded large -scale data annotation, limiting the de-
velopment of high -performance diagnostic tools. However, the rapid evolution of arti-
ficial intelligence (AI) and digital pathology (DP) is redefining this landscape, intro-ducing innovative s olutions that enhance diagnostic accuracy, efficiency, and standard-
ization. A particularly promising frontier lies in the integration of large language mod-
els (LLMs) with retrieval -augmented generation (RAG) systems and pathology- spe-
cific foundation models  [5-8]. This convergence holds significant potential in thyroid 
cytology, where diagnostic variability and interpretive ambiguity often present substan-
tial obstacles to effective clinical decision -making.  
 

2  Al-Asi et al.  
Thyroid cytology is vital for evaluating thyroid nodules but often presents diagnostic 
challenges, particularly in distinguishing benign from potentially malignant lesions. 
Traditional frameworks, though effective, can be limited by subjective interpretation and inter -observer variability  [9-12]. The atypia of undetermined significance (AUS) 
category in The Bethesda System for Reporting Thyroid Cytopathology (TBSRTC) is especially challenging due to its indeterminate nature, often requiring additional mo-lecular testing or surgical follow -up for a  definitive diagnosis  [13-16]. AI-driven meth-
ods might offer the potential to overcome these limitations by enhancing pattern recog-
nition, contextual understanding, and standardization.  
 
This study investigates the application of RAG -enhanced LLMs combined with pathol-
ogy foundation models  [5, 17 -20] for thyroid cytology diagnosis. The RAG framework 
enables dynamic access to a curated repository of case studies, diagnostic criteria, and expert interpretation , thereby enriching the LLMs' contextual comprehension. Simulta-
neously, foundation models trained on high- resolution pathology images improve fea-
ture extraction and classification accuracy, offering pathologists a robust diagnostic support system.  
 
Our findings reveal that integrating RAG with pathology- focused LLMs not only im-
proves diagnostic consistency but also enhances interpretability and reduces diagnostic 
variability. This hybrid AI approach demonstrates significant potential in supporting clinical workflows, ultimately advancing the accuracy and efficiency of thyroid cyto-pathology. By bridging advanced language processing with domain- specific image 
analysis, this work sets the stage for transformative AI applications in diagnostic pa-
thology.  
 1.1 Related Work  
The Vision Transformer (ViT) has revolutionized image processing in neural networks by replacing traditional convolutional architectures with transformer -based models. By 
segmenting images into fixed -size patches and applying self -attention mechanisms, 
ViTs enable more flexible and scalable feature learning, capturing long -range depend-
encies within an image  [21]. This novel approach has demonstrated competitive per-
formance on large -scale image recognition tasks, influencing both academic research 
and real -world applications in computer vision  [22, 23] . As transformer -based models 
continue to evolve, they are reshaping the landscape of deep learning in visual pro-cessing.  
 
Over the past t hree years, several pathology  foundational models have emerg ed in the 
literature. The journey began with the CTransPath model  [24], which pioneered the 
field by training a 28M parameter Swin transformer  [23] using MoCoV3 on 15M tiles 
from 32K whole slide images (WSIs). While subsequent studies have experimented with methods such as SimCLR  [25], the majority have converged on the DINOv2 
framework  [26], inspired by its strong performance with natural images. At the same 
time, the scale of training datasets has varied significantly —from open -access 

 RAG -Optimized LLM for Cytology  3 
repositories like The Cancer Genome Atlas, comprising around 30K WSIs, to newer, 
proprietary collections that allow models to learn from billions of tiles. This diversity in both algorithmic approaches and data sources highlights the dynamic and multifac-eted progress in the field of computational pathology.  In parallel, recent years have 
witnessed exceptional progress in both LLMs  and pathology‐specific foundation mod-
els, further advancing AI‐driven diagnostic tools for medical imaging and decision sup-port.  Notably, three pathology‐specific models —UNI, Virchow, and Gigapath —have 
emerged, each contributing unique strengths to the analysis of histopathology images  
[5, 17, 20] . The UNI foundation model was first introduced as a general -purpose, self -
supervised model pretrained on over 100 million images from more than 100,000 diag-nostic H&E -stained slides spanning 20 major tissue types. Later, a second version of 
UNI was publishe d, which was trained on over 350,000 H&E and IHC slides. In a sim-
ilar vein, GigaPath is another state -of-the-art whole -slide pathology foundation model, 
pretrained on 1.3 billion image tiles derived from over 171,000 slides spanning 31 tis-sue types from a major US health network. Unlike traditional methods that rely on tile subsampling, GigaPath harnesses the advanced vision transformer architecture together with the LongNet method [27] to fully integrate slide -level context. Moreover, Vir-
chow2, a 632 million parameter vision transformer, was trained on 3.1 million histo-
pathology whole -slide images covering diverse tissues, originating institutions, and 
stains.  
 
Retrieval -augmented generation (RAG) offers a promising solution by dynamically re-
trieving relevant information from large knowledge bases to supplement LLM outputs  
[6, 7, 28, 29] . This approach has proven effective in various domains but remains un-
derexplored in pathology applications. Recent efforts to integrate RAG in medical AI have shown potential in improving diagnostic accuracy and interpretability [28]. The 
convergence of RAG -enhanced LLMs with pathology- specific foundation models rep-
resents a novel approach in the field. This study builds upon these foundational works, aiming to bridge gaps in thyroid cytology diagnosis by leveraging both advanced lan-guage processing and specialized image- based models.  
2 Methodology 
Our framework integrates a r etrieved augmented generation  approach with a large lan-
guage model to enhance image interpretation by combining image vector comparisons with rich metadata context.  
 
2.1 Retrieved Augmented Generation Optimized LLM  
Our RAG- optimized LLM enhances thyroid cytology interpretation by leveraging mul-
tiple state -of-the-art image encoders. Specifically, UNI  [17], Virchow  [5], GigaPath  
[20], and ViT -32 [21] encoders are used to generate vector embeddings that capture 
essential visual features from each image. These embeddings are computed from im-ages processed through dedicated pipelines —each encoder providing a unique perspec-
tive based on its underlying arc hitecture —and are stored in a centralized database 

4  Al-Asi et al.  
alongside critical metadata including diagnosis, Bethesda category, and interpretation. 
When a new image is introduced, its vector representation is compared against the stored embeddings using cosine similarity metric  (see Equation 1)  allowing the system 
to retrieve the most contextually relevant five examples.  
 
𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶𝐶  𝑆𝑆𝐶𝐶𝑆𝑆𝐶𝐶𝑆𝑆𝑆𝑆𝑆𝑆𝐶𝐶𝑆𝑆𝑆𝑆=𝐴𝐴∙𝐵𝐵
‖𝐴𝐴‖‖𝐵𝐵‖                                              (1) 
, where A ⋅B is the dot product of vectors A and B  and ∥A∥ and ∥B∥ are the magnitudes  
of the vectors . 1 → Perfect similarity , 0 → No similarity , -1 → Completely opposite 
vectors .  
 
The retrieved examples, enriched with metadata, are then incorporated into the LLM's 
prompt, providing additional context that guides its analysis and interpretation. This augmented context enables the LLM to identify subtle patterns and relationships that  
might otherwise be overlooked, ultimately facilitating a more comprehensive and ac-curate interpretation of the new image.   
 Our RAG- optimized LLM infrastructure was developed as an integral part of our pro-
prietary , custom -built, in -house  digital pathology platform, ZAPP  [30]. Within ZAPP, 
we integrated Weaviate—an open -source vector database—to efficiently store embed-
ding vectors alongside their associated metadata. Additionally, we incorporated the 
Llama  3.2-11B Vision model into our workflow, deploying it on an HP Z8 G5 Work-
station equipped with 128GB of system memory and three NVIDIA A4500 GPUs (each 
with 24GB) in a distributed configuration.  
 2.2 Dataset  
To evaluate the predictive performance of pathology foundation models in thyroid cy-tology diagnosis, 13 patient records  (in total of 36 whole slides)  with confirmed thyroid 
lesions were identified. These lesions were classified into benign non- neoplastic, be-
nign neoplastic, and malignant categories. Patients included in the study either had a confirmed diagnosis at the time of fine -needle aspiration and cytology (FNAC) or un-
derwent surgical thyroidectomy with subsequent frozen section and hematoxylin and eosin (H&E) staining to establish a definitive diagnosis. The FNAC procedures were 
performed between September 2020 and May 2023. All patient records were encoded to remove personally identifying information while preserving relevant diagnostic fea-tures.  
 FNAC was conducted on thyroid lesions, and the cytology results were categorized according to TBSRTC  [16]. Of the 13 patients, nine were called as AUS (Bethesda III), 
two were called as follicular neoplasm (Bethesda IV), one was categorized as benign (Bethesda II) with a diagnosis of Graves' disease, and one was called as malignant (Be-thesda VI) with papillar y thyroid carcinoma (PTC) identified in the cytology smear.  
 A total of 36 cytology slides  (21 containing benign and 15 containing malignant le-
sions)  were obtained from archival cytology tray records  of patients  with all samples 

 RAG -Optimized LLM for Cytology  5 
meeting the adequacy criteria outlined in TBSRTC (six follicles of ten cells or the pres-
ence of thick colloid). The number of slides per patient ranged from a minimum of two to a maximum of five, depending on the distribution of adequate cellular material across multiple slides.   All slides were stained using the Diff -Quik staining method. The 
stained slides were then digitally scanned using Leica GT-450 (Leica Microsystems, 
USA)  and Grundium Ocus  (Grundium Ltd. Tampere, Finland)  40X scanners, stored in 
SVS format, and uploaded in to the ZAPP.   
3 Experiments  
Within ZAPP, slides were manually reviewed, and a region of interest (ROI) was se-lected from each slide for analysis. Each ROI contained follicular cells exhibiting ab-normal cellular morphology. The selected ROIs were then fed into pathology founda-tion mod els using a custom -built feature within ZAPP, known as ZAPP Chat. Three 
pathology foundation models (UNI, GIGAPATH, and Virchow) were utilized, along with a visual transformer model (ViT -32) as a comparator. Additionally, an ensemble 
technique was employed to combine the results generated from UNI, GIGAPATH, Vir-
chow, and ViT -32, wherein the highest -ranking responses based on similarity score 
vectors were displayed to mitigate potential training biases.  
 Model performance was assessed based on Top -1, Top -3, and Top- 5 prediction accu-
racy for each sample. The primary evaluation criteria included the correct prediction of the definitive confirmed diagnosis and the accurate classification of the Bethesda diag-nostic category. Ground truth for each sample was established through cytological find-
ings or surgical follow -up with frozen section and H&E specimens. All models were 
prompted with the same region of interest at a uniform magnification of 40x for each cytology specimen slide. Following completion of prompting, results were recorded 
according to the Top -1, Top- 3, and Top -5 performance metrics. A correct prediction of 
diagnosis and/or Bethesda staging was considered a positive result, whereas an incor-
rect prediction was deemed a negative result. Accuracy metrics were subsequently cal-
culated to compare the performances of different models across different outputs.  
4 Results  
Table 1 summarizes the performance of various foundation models in predicting patient diagnoses from cytology samples. The UNI model demonstrated the highest accuracy, achieving Top- 1, Top -3, and Top- 5 prediction scores of 0.69, 0.81, and 0.92, respec-
tively. Other models showed lower performance, with Gigapath slightly outperforming Virchow. General -purpose vision transformers and ensemble models performed worse 
overall compared to pathology -specific foundation models.  
 Figure 1 presents the receiver operating characteristic (ROC) curve and area under the 
curve (AUC) for predicting final surgical diagnoses. The best -performing UNI founda-
tion model achieved an AUC ranging from 0.73 to 0.93.  

6  Al-Asi et al.  
 
As shown in Table 2, model performance improved across the board when predicting the correct Bethesda category. UNI remained the top performer, with Top -1, Top- 3, 
and Top- 5 scores increasing to 0.75, 0.83, and 0.94, respectively. This improvement 
highlight s the model’s enhanced capability, particularly with RAG. augmentation—to 
assess malignancy likelihood independently of the specific diagnosis.  
Table 1. Comparison of foundational models for predicting surgical diagnosis from thyroid cy-
tology images.  
Model Performance Predicting Surgical Diagnosis  
 UNI Gigapath  Virchow  ViT-32 Ensemble  
Top-1 0.69 0.56 0.50 0.25 0.42 
Top-3 0.81 0.61 0.58 0.31 0.56 
Top-5 0.92 0.64 0.61 0.33 0.64 
Table 2. Comparison of foundational models for predicting TBSRTC  from thyroid cytology 
images.  
Model Performance Predicting TBSRTC  
 UNI Gigapath  Virchow  ViT-32 Ensemble  
Top-1 0.75 0.61 0.61 0.33 0.47 
Top-3 0.83 0.67 0.72 0.36 0.61 
Top-5 0.94 0.69 0.75 0.36 0.69 
 
 
Fig. 1. Receiver operator characteristic curve of models for predicting surgical diagnosis. Top1 
(solid line), Top3 (dashed line), and Top5 (dotty line) of models and their associated AUC are 
shown on the legend of the figure.  
 


 RAG -Optimized LLM for Cytology  7 
 
5 Discussion  
In this study, we presented a comprehensive evaluation of various foundation models 
for predicting patient diagnoses from cytology samples. Our results, summarized in Table 1, demonstrate that pathology- specific models outperformed general -purpose vi-
sion transformers and ensemble approaches, with the UNI model achieving the highest accuracy. Our results demonstrate that pathology -specific models outperformed gen-
eral-purpose approaches, with UNI achieving the highest accuracy (Top -1: 0.69, Top-
3: 0.81, Top -5: 0.92). ROC and AUC analyses further confirm UNI’s strong predictive 
performance (AUC: 0.73– 0.93). Predicting Bethesda categories improved model per-
formance overall, with UNI maintaining its lead (Top -1: 0.75, Top- 3: 0.83, Top -5: 
0.94). These findings highlight the advantages of pathology- specific models and re-
trieval- augmented generation (RAG) in enhancing cytology -based diagnostic predic-
tions.  
 
The Bethesda System for Reporting Thyroid Cytopathology (TBSRTC) recommends that no more than 7% of thyroid cytology cases be classified as Bethesda III, atypia of undetermined significance (AUS). In a clinical setting, this is challenging due to inter -
observer variability among pathologists and the multiple follow up options available to patients.  
 Since its establishment in 2007, T BSRTC has been revised multiple times, the latest in 
2023, to decrease ambiguity and aid pathologists in establishing a clear -cut diagnosis 
of thyroid cytology samples , particularly for intermediate categories Bethesda III and 
Bethesda IV . Despite these efforts, inter -observer disagreement remains significant at 
10-40% between different observers . The revision of TBSRTC in 2023 has worked to 
decrease this disagreement with moderate success, highlighting the need for differe nt 
approaches to tackle this ongoing issue  [11]. 
 For cases diagnosed as AUS with sufficient material for further analysis, follow -up 
approaches such as repeat FNAC biopsy, molecular testing, diagnostic lobectomy, or clinical surveillance are recommended based on the clinician’s discretion  [16]. How-
ever, repeat FNAC may not always be feasible due to a shortage of cytopathology spe-cialists  [31], and the uncertainty of an initial AUS diagnosis can lead to significant 
psychological distress for patients  [32]. Additionally, the lack of absolute consensus on 
the recommended timeframe for a repeat FNAC complicates clinical decision -making 
[33]. 
 
Molecular testing has proven to be a valuable adjunct in reducing diagnostic ambiguity 
and improving surgical outcomes in AUS cases, but its cost remains a significant barrier to widespread use  [34]. While molecular testing aids in risk stratification, diagnostic 
lobectomy remains the gold standard for post -operative diagnosis of thyroid 

8  Al-Asi et al.  
malignancies, with paraffin block sample preparation providing definitive histopatho-
logical confirmation  [35]. However, this procedure is also associated with additional 
costs and potential morbidity, particularly for patients who receive a negative result or opt for long -term surveillance  [34] 
 We recognize that the sample size is relatively small and acknowledge the need for 
further validation  of our findings  in a larger study population. However, these initial 
findings lay a strong foundation for future research. To enhance reproducibility, the 
authors plan to expand this work with a larger dataset, though this is beyond the scope 
of the current study.  
6 Conclusion  
The challenges discussed  highlight the need for tools that aid pathologists, surgeons, 
and clinicians in managing patients with unclear cytology diagnoses . The application 
of pathology foundation models combined with retrieval -augmented generation offers 
a promising outlook to fill in this gap as highlighted by the positive results of the ex-periment.  
Acknowledgments.  This study was internally funded by our institution . 
Disclosure of Interests.  The authors have no competing interests  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 RAG -Optimized LLM for Cytology  9 
References  
1. Rahman MA, Yilmaz I, Albadri ST, Salem FE, Dangott BJ, Taner CB, et al. Artificial 
Intelligence Advances in Transplant Pathology. Bioengineering -Basel. 2023;10(9).  
2. Ding S, Li J, Wang J, Ying S, Shi J. Multi -Scale Efficient Graph -Transformer for Whole Slide 
Image Classification. IEEE J Biomed Health Inform. 2023;27(12):5926- 36. 
3. Agraz JL, Agraz C, Chen AA, Rice C, Pozos RS, Aelterman S, et al. Optimized Whole-Slide -
Image H&E Stain Normalization: A Step Towards Big Data Integration in Digital Pathology. 
IEEE Open J Eng Med Biol. 2025;6:35 -40.  
4. Pinckaers H, Bulten W, van der Laak J, Litjens G. Detection of Prostate Cancer in Whole-
Slide Images Through End-to -End Training With Image -Level Labels. IEEE Trans Med 
Imaging. 2021;40(7):1817- 26.  
5. Vorontsov E, Bozkurt A, Casson A, Shaikovski G, Zelechowski M, Severson K, et al. A 
foundation model for clinical -grade computational pathology and rare cancers detection. Nat 
Med. 2024;30(10):2924- 35.  
6. Ong CS, Obey NT, Zheng Y, Cohan A, Schneider EB. SurgeryLLM: a retrieval -augmented 
generation large language model framework for surgical decision support and workflow enhancement. NPJ Digit Med. 2024;7(1):364.  
7. Bhayana R, Fawzy A, Deng Y, Bleakney RR, Krishna S. Retrieval -Augmented Generation for 
Large Language Models in Radiology: Another Leap Forward in Board Examination Performance. Radiology. 2024;313(1):e241489.  
8. Liu S, McCoy AB, Wright A. Improving large language model applications in biomedicine 
with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines. J Am Med Inform Assoc. 2025.  
9. Poursina O, Khayyat A, Maleki S, Amin A. Artificial Intelligence and Whole Slide Imaging 
Assist in Thyroid Indeterminate Cytology: A Systematic Review. Acta Cytol. 2025:1 -10.  
10. Townsend J, Perez -Machado M. Navigating Diagnostic Uncertainty in Thyroid Nodules: The 
Critical Role of Cytology and Histology in Oncocytic and Rare Patterned Lesions. 
Cytopathology. 2025.  
11. Bahattin E, Emine D, Civi CK, Fatih Y. Inter - and Intra -observer Reproducibility of Thyroid 
Fine Needle Aspiration Cytology: An investigation of Bethesda 2023 Using Immunohistochemical BRAFV600E Antibody. J Cytol. 2024;41(4):221-8.  
12. Wang J, Zheng N, Wan H, Yao Q, Jia S, Zhang X, et al. Deep learning models for thyroid 
nodules diagnosis of fine -needle aspiration biopsy: a retrospective, prospective, multicentre 
study in China. Lancet Digit Health. 2024;6(7):e458-e69.  
13. Saoud C, Bailey GE, Graham AJ, Maleki Z. The Bethesda System for Reporting Thyroid 
Cytopathology in the African American population: A tertiary centre experience. Cytopathology. 2024;35(6):715-23.  
14. Juhlin CC, Baloch ZW. The 3(rd) Edition of Bethesda System for Reporting Thyroid 
Cytopathology: Highlights and Comments. Endocr Pathol. 2024;35(1):77-9.  
15. Rai K, Park J, Gokhale S, Irshaidat F, Singh G. Diagnostic Accuracy of the Bethesda System 
for Reporting Thyroid Cytopathology (TBSRTC): An Institution Experience. Int J 
Endocrinol. 2023;2023:9615294.  
16. Ali SZ, Baloch ZW, Cochand-Priollet B, Schmitt FC, Vielh P, VanderLaan PA. The 2023 
Bethesda System for Reporting Thyroid Cytopathology. Thyroid. 2023;33(9):1039- 44.  

10  Al-Asi et al.  
17. Chen RJ, Ding T, Lu MY, Williamson DFK, Jaume G, Song AH, et al. Towards a general -
purpose foundation model for computational pathology. Nat Med. 2024;30(3):850- 62. 
18. Lu MY, Chen B, Williamson DFK, Chen RJ, Liang I, Ding T, et al. A visual -language 
foundation model for computational pathology. Nat Med. 2024;30(3):863- 74. 
19. Wang X, Zhao J, Marostica E, Yuan W, Jin J, Zhang J, et al. A pathology foundation model 
for cancer diagnosis and prognosis prediction. Nature. 2024;634(8035):970-8.  
20. Xu H, Usuyama N, Bagga J, Zhang S, Rao R, Naumann T, et al. A whole -slide foundation 
model for digital pathology from real -world data. Nature. 2024;630(8015):181-8.  
21. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers 
for Image Recognition at Scale. CoRR. 2020;abs/2010.11929.  
22. Touvron H, Vedaldi A, Douze M, Jegou H. Fixing the train -test resolution discrepancy. 
IAdvances  in Neural Information Processing Systems: Curran Associates, Inc.; 2019.  
23. Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, et al. Swin Transformer: Hierarchical Vision 
Transformer using Shifted Windows.  2021 IEEE/CVF International Conference on 
Computer Vision (ICCV)2021. p. 9992- 10002.  
24. Wang X, Yang S, Zhang J, Wang M, Zhang J, Yang W, et al. Transformer -based 
unsupervised contrastive learning for histopathological image classification. Med Image Anal. 2022;81:102559.  
25. Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of 
visual representations.  Proceedings of the 37th International Conference on Machine Learning: JMLR.org; 2020. p. Article 149.  
26. Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning Robust Visual Features without 
Supervision. arXiv [csCV]. 2024.  
27. Ding J, Ma S, Dong L, Zhang X, Huang S, Wang W, et al. LongNet: Scaling Transformers 
to 1,000,000,000 Tokens. arXiv [csCL]. 2023.  
28. Fink A, Nattenmuller J, Rau S, Rau A, Tran H, Bamberg F, et al. Retrieval -augmented 
generation improves precision and trust of a GPT -4 model for emergency radiology diagnosis 
and classification: a proof -of-concept study. Eur Radiol. 2025.  
29. Koga S, Ono D, Obstfeld A. Retrieval -augmented generation versus document -grounded 
generation: a key distinction in large language models. J Pathol Clin Res. 2025;11(1):e70014.  
30. Akkus Z, Dangott B, Nassar A. A Web/Cloud based Digital Pathology Platform Framework 
for AI Development and Deployment. bioRxiv. 2022:2022.11.04.514741.  
31. Satturwar S, Compton M, Miller D, Goldberg A, McGrath C, Friedlander M, et al. American 
Society of Cytopathology’s cytopathology workforce survey in the United States. Journal of the American Society of Cytopathology. 2024.  
32. Stahlmann K, Reitsma JB, Zapf A. Missing values and inconclusive results in diagnostic 
studies. Statistical Methods in Medical Research. 2023;32(9):1842- 55.  
33. Cosme I, Nobre E, Bugalho MJ. Repetition of thyroid fine -needle aspiration cytology after 
an initial nondiagnostic result: Is there an optimal timing? Endocrinología, Diabetes y 
Nutrición (English ed). 2024;71(5):216-20.  
34. Uppal N, Collins R, James B. Thyroid nodules: Global, economic, and personal burdens. 
Frontiers in Endocrinology. 2023;14.  
35. Hernandez-Prera JC. The evolving concept of aggressive histological variants of 
differentiated thyroid cancer. Semin Diagn Pathol. 2020;37(5):228-33.  
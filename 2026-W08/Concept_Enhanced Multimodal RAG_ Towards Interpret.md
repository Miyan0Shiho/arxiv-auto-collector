# Concept-Enhanced Multimodal RAG: Towards Interpretable and Accurate Radiology Report Generation

**Authors**: Marco Salmè, Federico Siciliano, Fabrizio Silvestri, Paolo Soda, Rosa Sicilia, Valerio Guarrasi

**Published**: 2026-02-17 15:18:07

**PDF URL**: [https://arxiv.org/pdf/2602.15650v1](https://arxiv.org/pdf/2602.15650v1)

## Abstract
Radiology Report Generation (RRG) through Vision-Language Models (VLMs) promises to reduce documentation burden, improve reporting consistency, and accelerate clinical workflows. However, their clinical adoption remains limited by the lack of interpretability and the tendency to hallucinate findings misaligned with imaging evidence. Existing research typically treats interpretability and accuracy as separate objectives, with concept-based explainability techniques focusing primarily on transparency, while Retrieval-Augmented Generation (RAG) methods targeting factual grounding through external retrieval. We present Concept-Enhanced Multimodal RAG (CEMRAG), a unified framework that decomposes visual representations into interpretable clinical concepts and integrates them with multimodal RAG. This approach exploits enriched contextual prompts for RRG, improving both interpretability and factual accuracy. Experiments on MIMIC-CXR and IU X-Ray across multiple VLM architectures, training regimes, and retrieval configurations demonstrate consistent improvements over both conventional RAG and concept-only baselines on clinical accuracy metrics and standard NLP measures. These results challenge the assumed trade-off between interpretability and performance, showing that transparent visual concepts can enhance rather than compromise diagnostic accuracy in medical VLMs. Our modular design decomposes interpretability into visual transparency and structured language model conditioning, providing a principled pathway toward clinically trustworthy AI-assisted radiology.

## Full Text


<!-- PDF content starts -->

Concept-Enhanced Multimodal RAG: Towards
Interpretable and Accurate Radiology Report
Generation
Marco Salm` e1, Federico Siciliano2, Fabrizio Silvestri2,
Paolo Soda1,3*, Rosa Sicilia4†, Valerio Guarrasi1†
1Department of Engineering, Research Unit of Artificial Intelligence
and Computer Systems, Universit` a Campus Bio-Medico of Roma,
Rome, Italy.
2Department of Computer, Control and Management Engineering,
Sapienza University of Rome, Rome, Italy.
3Department of Diagnostics and Intervention, Radiation Physics,
Biomedical Engineering, Ume˚ a University, Ume˚ a, Sweden.
4UniCamillus-Saint Camillus International University of Health
Sciences, Rome, Italy.
*Corresponding author(s). E-mail(s): paolo.soda@umu.se;
Contributing authors: marco.salme@unicampus.it;
siciliano@diag.uniroma1.it; fsilvestri@diag.uniroma1.it;
rosa.sicilia@unicamillus.org; valerio.guarrasi@unicampus.it;
†These authors contributed equally to this work.
Abstract
Radiology Report Generation (RRG) through Vision-Language Models (VLMs)
promises to reduce documentation burden, improve reporting consistency, and
accelerate clinical workflows. However, their clinical adoption remains limited
by the lack of interpretability and the tendency to hallucinate findings mis-
aligned with imaging evidence. Existing research typically treats interpretability
and accuracy as separate objectives, with concept-based explainability tech-
niques focusing primarily on transparency, while Retrieval-Augmented Gener-
ation (RAG) methods targeting factual grounding through external retrieval.
We present Concept-Enhanced Multimodal RAG (CEMRAG), a unified frame-
work that decomposes visual representations into interpretable clinical concepts
and integrates them with multimodal RAG. This approach exploits enriched
1arXiv:2602.15650v1  [cs.CV]  17 Feb 2026

contextual prompts for RRG, improving both interpretability and factual accu-
racy. Experiments on MIMIC-CXR and IU X-Ray across multiple VLM archi-
tectures, training regimes, and retrieval configurations demonstrate consistent
improvements over both conventional RAG and concept-only baselines on clin-
ical accuracy metrics and standard NLP measures. These results challenge the
assumed trade-off between interpretability and performance, showing that trans-
parent visual concepts can enhance rather than compromise diagnostic accuracy
in medical VLMs. Our modular design decomposes interpretability into visual
transparency and structured language model conditioning, providing a principled
pathway toward clinically trustworthy AI-assisted radiology. The project page is
available at https://github.com/marcosal30/cemrag-rrg.
Keywords:Radiology Report Generation, Vision-Language Models, Medical Imaging,
Interpretability, Retrieval-Augmented Generation, Multimodal AI
1 Introduction
Vision-Language Models (VLMs) [1] have emerged as a breakthrough technology in
medical imaging. By jointly modeling images and textual data, they have demon-
strated remarkable capabilities across several clinical applications, including visual
question answering, image classification, disease diagnosis, and automated report gen-
eration [2, 3]. Among these applications, Radiology Report Generation (RRG) presents
a particularly challenging task: generating comprehensive textual reports from medical
images that accurately describe imaging findings and identify potential pathologies. In
this domain, VLMs offer the potential to streamline radiological workflows by automat-
ing the initial drafting of reports. However, despite these advantages, the adoption
of VLMs in clinical settings is limited by two critical factors. Firstly, VLMs lack
interpretability [4], operating as black boxes that do not reveal how visual evidence
observed in medical images translates into diagnostic statements within generated
reports. Without visibility into the anatomical structures or radiological patterns that
support specific diagnostic statements, clinicians cannot verify the model’s reason-
ing, undermining both clinical trust and patient safety. Secondly, VLMs are prone
to hallucinations [5], producing medically inaccurate statements that are misaligned
with imaging findings, such as reporting non-existent pathologies, incorrect anatomical
localizations, or impressions inconsistent with observed abnormalities. Such inaccu-
racies pose particular concerns in radiology, where the recognition of subtle findings
and the precise alignment between visual evidence and domain-specific terminology
are essential for reliable diagnostic support.
Current research efforts have predominantly tackled these issues independently.
Existing interpretability approaches [6] provide useful insights but often operate as
post-hoc explanations that do not meaningfully influence the model’s predictions.
Recently, techniques such as Sparse Linear Concept Embeddings (SpLiCE) [7] have
shown that visual embeddings can be decomposed into sparse, human-interpretable
concepts without requiring manual annotations, offering a scalable pathway toward
transparency in VLMs. In parallel, recent work aimed at improving factual grounding
2

has increasingly relied on Retrieval-Augmented Generation (RAG) [8]. By retrieving
similar cases and reports from a database, RAG provides external context to ground
outputs in existing knowledge, reducing hallucinations and improving clinical relevance
in RRG [9, 10]. Yet, RAG-based models remain limited by retrieval errors, as the
context may be insufficient, noisy, redundant or irrelevant, leading to diluting the
model’s focus, potential misattribution, and even factual inconsistencies [11].
These two research areas, interpretability and factual grounding, have evolved
largely in isolation. This separation is reinforced by the widespread assumption
that transparency and performance trade off against each other, by the empirical
tendency of deeper, more accurate models to be harder to interpret, and by the
practical difficulty of building systems that are both highly accurate and transpar-
ently reasoned [12]. We challenge this assumption by asking:Can interpretable visual
concepts be integrated into retrieval-augmented report generation to jointly improve
transparency and factual accuracy in medical VLMs?
To answer this question, we present Concept-Enhanced Multimodal RAG
(CEMRAG), a unified framework that reconciles interpretability and factual accu-
racy in RRG by integrating interpretable visual concept extraction with multimodal
RAG. The core innovation of our approach lies in transforming interpretable visual
concepts from passive post-hoc explanations into active components of the generation
pipeline, using them to prioritize clinically pertinent portions of retrieved content and
direct the model toward information supported by visual evidence. Our experimental
framework spans two established radiology benchmarks, MIMIC-CXR and IU-Xray,
which differ substantially in scale and report characteristics. We examine both in-
domain retrieval scenarios, where similar cases are retrieved from the same dataset,
and cross-domain retrieval settings, where reports are retrieved from an external
database, reflecting realistic clinical deployments where knowledge bases may origi-
nate from different institutional sources. For each retrieval configuration, we evaluate
two distinct VLM architectures under two training paradigms: a Zero-Shot prompting
setting that assesses the immediate effectiveness ofCEMRAGwithout model adap-
tation, and a Supervised Fine-Tuning (SFT) regime that examines how interpretable
concepts interact with task-specific optimization.
Our main contributions are summarized as follows:
•We proposeCEMRAG, a framework that integrates interpretable visual decompo-
sition with retrieval-based grounding to enhance transparency and factual accuracy
in RRG.
•We provide the first systematic comparison of RAG and SFT paradigms in RRG,
establishing a comprehensive benchmark that evaluates their individual and com-
bined effectiveness across two VLM architectures, two retrieval configurations, and
two radiology datasets.
•We demonstrate thatCEMRAGconsistently outperforms traditional RAG and
concept-based approaches on both Natural Language Processing (NLP) and clinical
accuracy metrics across diverse experimental conditions.
•We provide empirical evidence that interpretable visual concepts can enhance rather
than compromise factual accuracy, challenging the assumption of a trade-off between
interpretability and performance in medical AI systems.
3

The remainder of this paper is organized as follows. We begin by reviewing related
work on interpretability for VLMs and multimodal RAG in Section 2, then present
our proposed framework in Section 3. Section 4 describes the experimental setup,
including datasets, baselines, and evaluation metrics. We present both quantitative
and qualitative results in Section 5, and conclude with discussion and future research
directions in Section 6.
2 Related Work
Our work builds upon two complementary research areas that we review in turn. First,
we discuss interpretability methods for VLMs, then we examine the application of
multimodal RAG to medical domains.
2.1 Interpretability for Vision-Language Models
The interpretability of VLMs has emerged as a critical requirement for clinical deploy-
ment, particularly in medical imaging where diagnostic decisions directly impact
patient outcomes. However, most interpretability research focuses on classification
tasks, leaving the integration of transparency mechanisms into generative frameworks
like RRG largely unexplored.
Current interpretability methods can be distinguished by whether they rely on
implicit explanation mechanisms or explicit concept representations. In the first
category, post-hoc textual explanation techniques, exemplified by rationale genera-
tion [13, 14] and Chain-of-Thought reasoning [15, 16], train models to articulate their
reasoning processes after making predictions through natural language. While these
methods have demonstrated improvements in perceived transparency and, in some
cases, task performance, they often act as plausible rationalizations rather than faith-
ful reflections of the underlying computational mechanisms [17]. This limitation is
particularly concerning in medical settings, where clinicians require insight into the
actual diagnostic cues driving the model’s predictions.
Mechanistic Interpretability [18–20] represents a more ambitious approach, seeking
to reverse-engineer neural network computations through detailed analysis of attention
patterns, information flow, and component functionality [21, 22]. Despite its theoreti-
cal appeal in pursuing causal rather than correlational understanding, the architectural
complexity of modern large-scale VLMs renders this approach extremely challenging
to implement systematically, especially in real-world medical AI systems.
An alternative paradigm achieves interpretability through explicit concept repre-
sentations. Concept Bottleneck Models [23, 24] exemplify this approach by forcing all
information to flow through an intermediate layer of predefined human-interpretable
concepts, ensuring that predictions depend exclusively on explicit semantic attributes.
This architectural constraint offers genuine transparency but comes at a substantial
cost: it demands extensive manual concept annotation and restricts model expressive-
ness, limiting the ability to capture visual patterns that fall outside the predefined
space.
Recent work on representation decomposition offers a more flexible concept-based
approach by decomposing dense embeddings into interpretable components without
4

requiring manual annotations or architectural constraints. Methods such as those
proposed by [25, 26] leverage internal model components such as attention mecha-
nisms to translate high-dimensional representations into natural language descriptions,
revealing how models encode visual features and their relationships to semantic con-
cepts. However, methods relying on attention-based decomposition remain dependent
on the learned representations of these internal mechanisms, which may not align
with domain-specific interpretability requirements in specialized contexts where trans-
parency demands explicit grounding in established vocabulary. Recently, an emerging
line of work has pursued interpretability by explicitly leveraging predefined, domain-
specific vocabularies of clinically meaningful concepts, enabling explanations to be
expressed directly in human-readable terms rather than inferred post hoc from latent
attention patterns [7, 27]. Among these, SpLiCE [7] operationalizes this paradigm by
factorizing visual representations into sparse linear combinations of interpretable con-
cepts drawn from a domain-specific vocabulary, achieving scalable and transparent
explanations without sacrificing representational flexibility.
2.2 Multimodal Retrieval-Augmented Generation in Medicine
Multimodal RAG has emerged as a promising approach to mitigating factual halluci-
nations in medical VLMs by grounding generation in existing clinical knowledge. These
systems extend traditional text-based retrieval by employing specialized encoders to
extract features from multiple modalities, retrieving relevant cases from curated med-
ical databases, and conditioning generation on both the input data and the retrieved
context [28, 29]. This shift from purely parametric knowledge to retrieval-augmented
pipelines has yielded substantial gains across applications such as orthopedic diag-
nosis [30], lung cancer staging [31], and prescription interpretation for medication
management [32].
In the specific domain of RRG, multimodal RAG approaches have shown partic-
ularly promising results. The MMed-RAG system reports substantial improvements
in factual accuracy for both medical VQA and report generation across radiology,
pathology, and ophthalmology [33]. Similarly, the RULE framework achieves notable
gains by combining calibrated retrieval strategies with preference-based fine-tuning to
balance reliance on parametric knowledge versus retrieved context [34]. These results
suggest that providing models with concrete clinical examples can substantially reduce
hallucinations and improve diagnostic consistency.
Although multimodal RAG provides indirect interpretability by exposing which
cases inform generation, this transparency remains fundamentally passive, reveal-
ing available information without guaranteeing how it is utilized during generation.
Retrieval operates through global similarity matching in learned embedding spaces,
capturing overall visual resemblance without explicit guidance on which anatomical
structures or pathological patterns should be prioritized [35]. This absence of semantic
grounding creates a fundamental dilemma: insufficient retrieval fails to capture nec-
essary clinical information, while excessive retrieval introduces irrelevant details that
interfere with coherent generation [34]. Consequently, models may inappropriately
incorporate findings from retrieved cases that do not correspond to visual evidence
5

in the input image, leading to factual hallucinations even in retrieval-augmented
systems [36, 37].
2.3 Limitations and Motivations
The limitations identified across interpretability and RAG approaches reveal a fun-
damental gap: existing methods treat transparency and factual accuracy as separate
objectives rather than mutually reinforcing components. Interpretability techniques
provide insight into model representations without actively constraining generation
toward factually grounded outputs, while RAG systems improve factual grounding
without semantic control to align retrieved information with specific visual evidence.
This separation motivates our central hypothesis that interpretable visual concepts
can serve as semantic guidance mechanisms, simultaneously enhancing transparency
and factual accuracy by directing retrieval and generation toward clinically relevant
content present in the input image.
3 Methodology
To jointly improve transparency and factual accuracy in RRG, we proposeCEM-
RAG, a framework that integrates interpretable visual concepts with multimodal RAG
through a structured prompting strategy. We extract clinically meaningful concepts
from visual embeddings and retrieve similar cases from a database, both derived from
the same image representation. Rather than treating these as independent augmenta-
tions, we structure them hierarchically in a unified prompt that guides the language
component to focus on relevant portions of retrieved context.
LLM🔥
❄
❄
 Medical 
Vocabulary
❄
🔥
❄
🔥Frozen Model
Supervised
Fine Tuning
 
keywords
 Concept
ExtractorVisual Projection
Concept Extraction
Multimodal Retrieval
Generation
Fig. 1CEMRAGoverall framework, combining interpretable concept extraction with RAG for
transparent and accurate radiology reporting.
Fig. 1 illustrates the complete architecture of our proposed framework. Given as
input a medical imageI∈RH×W, whereHandWdenote the height and width
in pixels, our objective is to generate a radiology report ˆR={ ˆr1,ˆr2, . . . , ˆrˆn}that
approximates the ground truth reportR={r 1, r2, . . . , r n}, withr iandˆrirepresent-
ing individual tokens, andnand ˆndenoting the number of tokens in each report. To
6

achieve this,CEMRAGgenerates reports by coordinating four key components, each
represented by distinct pathways in Fig. 1: (a) a visual encoding branch (yellow line),
comprising a medical VLM encoder and a projector, extracts dense visual features from
the input image; (b) a concept-extraction module (purple dashed line) maps image
embeddings into interpretable clinical keywords; (c) a multimodal retrieval module
(teal dash-dotted line) identifies similar training reports based on visual similarity in
the embedding space; and (d) an LLM (orange dotted line) conditioned on hierar-
chically structured prompts integrates visual tokens, concept keywords, and retrieved
reports to generate the final radiology report.
3.1 Visual Encoding and Projection
The foundation of our framework is the extraction of dense visual features to condition
the generation process. The input imageIis processed by a pretrained medical VLM
encoder E VLM to produce a sequence of visual featuresv VLM = E VLM(I)∈Rℓv×dVLM,
whereℓ vis the number of visual tokens andd VLM is the dimensionality of the VLM
feature space. A projection module Φ VLM, implemented as a token-wise multi-layer
perceptron, maps the visual features into the LLM token embedding space, produc-
ing visual token embeddingsz v= Φ VLM(vVLM)∈Rℓv×dLLM. Whilez vprovides the
primary visual features for generation, we employ additional aligned vision and text
encoders (E img, Etxt) to supply structured contextual information in a shared vision-
language space, enabling both the decomposition of visual content into explicit clinical
concepts and the retrieval of similar cases.
3.2 Concept Extraction
The input imageIis processed by the vision encoder E imgto produce a visual embed-
dingv= E img(I)∈Rd, wheredis the dimensionality of a shared vision-language
embedding space. To enable concept extraction, we define a medical vocabularyQ=
{q1, q2, . . . , q m}composed ofmdomain-specific concept terms derived from the train-
ing corpus. Each conceptq j∈Qis encoded with the text encoder E txtto obtain
its embeddingc j= E txt(qj)∈Rd, and we collect these embeddings into the con-
cept matrixC= [c 1,c2, . . . ,c m]∈Rd×m, whereddenotes the dimensionality of the
multimodal latent space. Then, the concept extraction module decomposes the visual
embeddingvas a non-negative linear combination of concept embeddings, as detailed
in Appendix A. This decomposition yields a coefficient vectorα∗∈Rm
≥0that quanti-
fies the contribution of each vocabulary concept to the image representation. We select
the top-τconcepts according to their coefficient magnitudes to form the interpretable
keyword setΩ= [q 1, . . . , q τ]⊆Q.
In our implementation, we instantiate E imgand E txtusing CLIP encoders [38],
which provide aligned vision-language embeddings through contrastive pretraining.
The same embeddingvalso enables retrieval of similar documented cases, as described
next.
7

3.3 Multimodal Retrieval Augmented Generation
WhileΩprovides explicit clinical concepts, complete documented cases are required
to ground generation in established clinical patterns and linguistic structure. We use
Eimgto construct a vector databaseV={(vtrain
i, Rtrain
i)}N
i=1that indexes allN
training images through their visual embeddingsvtrain
i and associates each with its
corresponding radiology reportRtrain
i.
Usingvas query, the retrieval mechanism identifies the top-kmost similar cases
by computing the cosine similaritySin the image embedding space:
S(I) = top-kv·vtrain
i
∥v∥∥vtrain
i∥N
i=1
yielding retrieved reportsR={Rtrain
i1, . . . , Rtrain
ik}.
By operating in the image embedding space, this retrieval strategy prioritizes cases
with similar radiological appearances, capturing visual patterns that are indicative of
comparable clinical findings [39].
3.4 Hierarchical Prompt Construction and Report Generation
To synthesize these information sources effectively, we employ a hierarchical prompt-
ing strategy designed to mitigate the limitations of each component. While retrieved
reportsRprovide rich clinical context, they may inadvertently introduce findings
absent in the query image. Conversely, concept keywordsΩact as precise visual
anchors but lack narrative structure. Therefore, we construct a structured prompt
PaugwhereΩserves as a priority filter, guiding the LLM to selectively leverage the
linguistic patterns inRthat align with the observed features.
The prompt structure consists of four components: (i) a coordination instruction
that establishes the task of generating a report while prioritizing concept-related con-
tent from retrieved examples, (ii) an explicit list of extracted concept keywordsΩ
framed as visual findings identified in the image, and (iii) the retrieved radiology
reportsRpresented as reference examples from similar cases, and (iv) a final instruc-
tion reinforcing the generation objective. This structured presentation provides the
LLM with both explicit concept annotations and implicit contextual knowledge, while
maintaining a clear separation between the different information sources.
The augmented promptP augis then processed by the LLM tokenizerTto produce
a sequence of textual token embeddingsz t=T(P aug)∈Rℓt×dLLM, whereℓ tdenotes
the number of text tokens andd LLM is the LLM embedding dimension. The visual
and textual token sequences are then concatenated along the sequence dimension to
form the full multimodal inputz= [z v;zt]∈R(ℓv+ℓt)×dLLM, which is fed to the
LLM. We denote the generated report as ˆR= LLM(z), and model generation in an
autoregressive formulation, where the LLM predicts each token ˆriconditioned on all
previous tokens and on the multimodal context encoded inz:
p(ˆR|I, P aug) =ˆnY
i=1p(ˆri|ˆr<i,z).
8

To preserve robust medical pretrained representations, we consistently keep all the
encoders frozen. In contrast, the training strategy for the LLM and the projection
module Φ VLM varies according to the model configuration, as detailed in Section 4.2.
4 Experimental Setup
This section outlines the experimental setup, detailing the datasets, model configura-
tions, experimental conditions, and evaluation metrics used to assess report generation
quality.
4.1 Datasets
We conduct experiments on two well-established benchmark datasets for RRG:
MIMIC-CXR [40] and IU X-ray [41]. Both datasets provide chest radiographs paired
with corresponding clinical reports, enabling comprehensive evaluation of our pro-
posed approach across different data scales and clinical contexts. The MIMIC-CXR
dataset [40] is a large-scale publicly available collection comprising over 370,000 chest
radiographs from more than 65,000 patients. We adopt the official training, validation,
and test split, restricting our experiments to the 156,344 frontal views (posteroante-
rior and anteroposterior projections) due to computational limitations. The IU X-ray
dataset [41] consists of 7,470 chest radiographs paired with 3,955 radiological reports.
This dataset serves as a complementary evaluation benchmark to assess model perfor-
mance under more constrained data conditions. The original IU X-ray dataset includes
both frontal and lateral radiological images for most reports. However, to maintain con-
sistency with our experimental protocol on MIMIC-CXR, we only consider the 3,307
frontal projections in our experiments. Following established conventions in the litera-
ture [42, 43], we exclude samples lacking a findings section, as this section provides the
essential ground truth for supervised learning in RRG tasks. We employ a dataset split
allocating 80% of the data for training, 10% for validation, and 10% for testing, with
strict enforcement of patient-level separation to prevent data leakage across splits.
4.2 Model Configurations and Experimental Conditions
Our framework requires a model with aligned multimodal embeddings (E img,Etxt)
capable of supporting both concept extraction and similarity-based retrieval. To this
end, we employ CXR-CLIP [39] across all experiments, which integrates a SwinTrans-
former [44] as visual encoder E imgand a BioClinicalBERT [45] as text encoder E txt,
both pretrained on the MIMIC-CXR dataset. We instantiate the concept-extraction
module with Sparse Linear Concept Embeddings (SpLiCE) [7], which performs explicit
sparse factorization of visual embeddings. Unlike attention-based approaches, which
derive concepts from learned internal representations, SpLiCE operates on a predefined
medical vocabulary. This ensures that the extracted concepts are clinically relevant,
and allows each term’s contribution to be quantified directly through optimized sparse
coefficients. For experiments on the IU X-ray dataset, we apply Low-Rank Adaptation
(LoRA) [46] to CXR-CLIP for obtaining refined embeddings tailored to this specific
data distribution (more details are given in Appendix B).
9

We evaluateCEMRAGunder two architectural configurations to assess whether
the hierarchical prompting strategy remains effective across different levels of medi-
cal domain adaptation and vision-to-language alignment. Both configurations adopt
a LLaVA-style architecture [47] with Mistral-7B [48] as the LLM backbone. The first
configuration employs LLaVA-Med [49], where both the vision encoder E VLM and
the LLM incorporate medical domain-specific pretraining. The second configuration
uses CXR-CLIP as a unified encoder: E imgextracts embeddingsvthat enable con-
cept extractionΩ, retrieval of reportsR, and generation of visual tokensv VLM.
This configuration pairs medically-pretrained CXR-CLIP with base Mistral-7B with-
out medical language pretraining. Unlike LLaVA-Med, which provides a pretrained
projection layer, this configuration requires the introduction of a projection module
ΦCLIP to map CLIP visual features into the LLM embedding space. Images are prepro-
cessed according to each encoder’s specifications: 224×224 pixels with normalization
to [−1,1] for CXR-CLIP, and 336×336 center crops with encoder-specific normal-
ization for LLaVA-Med. Implementation details, including the initial alignment phase
for Φ CLIP, are provided in Appendix B.
We evaluate four distinct prompting strategies that progressively incorporate
retrieval and concept information. To ensure reproducibility, we provide detailed
prompt templates and configuration details for each strategy in Table 1.
Image-Only.This approach provides the LLM with visual features alongside a min-
imal prompt instructing the model to describe the radiological image. This provides a
reference point for the level of performance achievable from visual information alone,
without any external context.
Concepts.This strategy augments the prompt with interpretable clinical concepts
extracted from the input image using SpLiCE. For each dataset, we construct a
domain-specific vocabulary of medical concepts by selecting the 200 most frequent
bigrams from the training reports. Each bigram is encoded with the CLIP text encoder
Etxtto obtain its embedding. SpLiCE is then applied to derive a sparse, non-negative
decomposition of the image embedding over this concept vocabulary, yielding a coeffi-
cient vector that quantifies the contribution of each concept. For each image, we rank
concepts according to their coefficients, select the top five, and inject these bigrams
into the prompt as explicit keywords. Additional details regarding vocabulary con-
struction, normalization procedures, and sensitivity analyses for different vocabulary
sizes and sparsity levels are provided in the Appendix A.1.
Multimodal RAG.This strategy provides contextual grounding by retrieving
reports from visually similar cases. We use FAISS [50] to compute cosine similarity
between the query image embeddingvderived from CXR-CLIP and embeddings in
databaseV, retrieving the top-k= 3 most similar cases whose reports are then incor-
porated into the prompt as reference examples. The retrieval configuration varies by
dataset. For the MIMIC-CXR dataset, retrieval is performed within the training set
of the same dataset to ensure that similar cases are drawn from the same distribu-
tion. For IU X-ray, which has a substantially smaller training set, in-domain retrieval
would provide insufficient contextual variety and limit the benefits of the RAG strat-
egy. We therefore implement cross-domain retrieval from the MIMIC-CXR training
10

Table 1Experimental configurations for radiology report generation, comparing input modalities,
prompt structures, interpretability mechanisms, and factual grounding strategies. Notation:v
denotes visual embeddings,Ωextracted concepts andRretrieved reports.Instruction: “Provide a
description of the findings in the radiology image”;Task: “Write the report of the radiology image
taking information from similar FINDINGS. Consider as more relevant sentences that contain any
of the KEYWORDS in the FINDINGS”;Final Instruction: “Write a paragraph with only the
report relying in detail on the FINDINGS”.
Strategy Inputs Prompt Structure Interpretability Factual Grounding
Image-Only v <Instruction> ×black-box ×Visual encoder and
LLM priors only
Concepts v,Ω <Instruction>+
<Keywords>✓Concept-level via
visual-semantic align-
ment∼Sparse visual con-
cept decomposition
RAG v,R <Instruction>+
<Similar Findings>∼Indirect via
retrieved context✓Visual similarity-
based nearest neigh-
bours
CEMRAG v,Ω,R <Task>+<Keywords>+
<Similar Findings>+
<Final Instruction>✓✓Dual: concept
annotations and simi-
lar cases✓✓Concept-guided
focus to salient find-
ings
set, simulating a realistic scenario where a smaller institutional dataset leverages a
larger external knowledge base to enhance report generation quality.
CEMRAG.This method combines the two previous approaches to construct an
enriched prompt that presents the top five concept bigrams as priority keywords
alongside three retrieved reports. We design the prompt structure following Mistral’s
prompt engineering guidelines [51], which emphasize the importance of clear task
definition, hierarchical organization, explicit formatting, and concrete examples for
effective instruction-based generation. Unlike previous conditions with simple gener-
ation instructions,CEMRAGemploys a coordination directive that establishes the
task objective of generating a report from similar findings while prioritizing content
related to the extracted concepts. Following this directive, the extracted keywords are
presented in a dedicated formatted section. The three retrieved reports are then incor-
porated with separators. The prompt concludes with a final instruction that explicitly
directs the model to produce a radiology report relying on the provided findings.
This design ensures clear attribution of information sources and guides the LLM
toward clinically relevant content by orienting the model toward portions of retrieved
reports that align with observed visual features.
Each of the four prompting strategies is evaluated under two LLM training
paradigms. In theZero-Shotsetting, the entire model (visual encoders, projection
layers, and the LLM) is kept frozen, assessing transfer capability with only prompt
content varying across conditions. In theSupervised Fine-Tuning (SFT)setting, we
adapt the LLM using LoRA [46] and jointly fine-tune the corresponding projection
layer, while keeping all visual encoders frozen. This design follows the LLaVA train-
ing paradigm, isolating adaptation to the language and projection components. This
enables a controlled investigation into how prompting strategies influence RRG perfor-
mance without confounding changes in visual feature extractors. Additional training
details for both theZero-ShotandSFTsettings are provided in Appendix B.
For all experiments, RRG is performed with greedy decoding (temperatureT=
0) to ensure deterministic and reproducible outputs, thereby eliminating variability
11

arising from stochastic sampling. All experiments are performed on four NVIDIA A100
GPUs. On MIMIC-CXR,SFTtraining time varies by strategy: Image-Only requires
approximately 7 hours,Conceptsstrategy 8 hours, RAG 13 hours, andCEMRAG14
hours.
4.3 Evaluation Metrics
Our evaluation framework incorporates two complementary categories of metrics: NLP
metrics, which measure lexical similarity to reference reports, and clinical metrics,
which assess the factual correctness of medical content. To evaluate lexical similar-
ity, we report three standard NLP metrics that are commonly used in text generation
tasks. ROUGE-L [52] measures the longest common subsequence between the gener-
ated and reference texts, capturing sentence-level structural similarity. BLEU-1 and
BLEU-4 [53] evaluate n-gram overlap at the unigram and 4-gram levels, respectively,
with BLEU-4 providing assessment of longer phrasal matches that better reflect flu-
ency and coherence. While these metrics effectively evaluate linguistic properties such
as fluency and phrasal coherence, they capture surface-level textual similarity rather
than clinical accuracy.
To address this limitation, we emphasize clinical evaluation through two established
factual correctness metrics: F1-CheXbert [54] and F1-RadGraph [55]. The former com-
putes the F1 score between disease labels extracted from the generated and reference
reports using the CheXbert labeler [56], which is a BERT-based model trained to
identify the presence, absence, or uncertainty of pathological findings. Following stan-
dard practice, we report F1-CheXbert across all 14 CheXbert classes, encompassing
the comprehensive range of conditions identifiable in chest radiographs. Additionally,
we report performance on the five most prevalent clinical findings in real-world chest
radiograph reports: atelectasis, cardiomegaly, consolidation, edema and pleural effu-
sion. This focused evaluation on common pathologies provides insight into the model’s
performance in frequently encountered diagnostic scenarios. The second clinical met-
ric, F1-RadGraph, quantifies factual correctness by measuring the overlap between the
semantic graphs extracted from the generated and reference reports. This captures not
only the presence of clinical findings, but also their anatomical locations and relation-
ships between entities. Together, these two clinical metrics provide a complementary
perspective on factual correctness: F1-CheXbert focuses on diagnostic label accuracy,
while F1-RadGraph assesses the representation of structured clinical content.
5 Results
We organize our findings into two complementary parts. Section 5.1 presents a quanti-
tative evaluation across datasets, model configurations, and training paradigms, using
both NLP and clinical accuracy metrics. Section 5.2 provides qualitative analyses that
illustrate how concept extraction and retrieval affect the generated reports and their
interpretability.
12

5.1 Quantitative Results
We structure our quantitative analysis by first examining performance on the MIMIC-
CXR dataset in Section 5.1.1, followed by results on the IU X-ray dataset in
Section 5.1.2. For each dataset, we evaluate two model configurations: LLaVA-Med
with its standard visual encoder, and LLaVA with CXR-CLIP as the visual encoder.
Each configuration is assessed under bothZero-ShotandSupervised Fine-Tuning
(SFT)settings to determine how domain adaptation influences the effectiveness of our
approach.
5.1.1 MIMIC-CXR
Table 2 presents comprehensive quantitative results on the MIMIC-CXR dataset,
where retrieval is performed from the MIMIC-CXR training set, establishing an
in-domain retrieval scenario.
Table 2Quantitative results on the MIMIC-CXR test set for two model configurations
(LLaVA-Med and LLaVA with CXR-CLIP) and two training regimes (Zero-ShotandSupervised
Fine-Tuning). We report F1-RadGraph (F1-RG), and NLP metrics (BLEU-1 as B-1, BLEU-4 as
B-4, ROUGE-L as R-L). CheXbert-based label metrics are reported over 14 labels (Micro-F1 14,
Macro-F1 14) and over the 5 most prevalent findings (Micro-F1 5, Macro-F1 5). Retrieval is
performed from the MIMIC-CXR training set.
Model Method Micro-F1 14Micro-F1 5Macro-F1 14Macro-F1 5F1-RG B-1 B-4 R-L
Zero-Shot
LLaVA-MedImage-Only 0.255 0.237 0.113 0.134 0.052 18.81 0.71 0.131
+ Concepts 0.401 0.412 0.238 0.355 0.073 21.47 1.29 0.146
+ RAG 0.498 0.528 0.314 0.447 0.184 27.08 4.60 0.187
+CEMRAG0.502 0.529 0.319 0.449 0.185 29.34 4.64 0.189
LLaVA
with CXR-CLIPImage-Only 0.213 0.174 0.101 0.123 0.151 19.20 4.69 0.185
+ Concepts 0.328 0.330 0.161 0.230 0.172 19.62 4.87 0.192
+ RAG 0.489 0.518 0.309 0.442 0.181 25.53 5.58 0.198
+CEMRAG0.498 0.526 0.314 0.443 0.187 29.78 6.08 0.201
Supervised Fine-Tuning
LLaVA-MedImage-Only 0.470 0.486 0.266 0.392 0.174 27.81 7.43 0.211
+ Concepts 0.476 0.502 0.284 0.4120.18828.837.87 0.220
+ RAG 0.477 0.499 0.287 0.412 0.176 30.50 7.26 0.212
+CEMRAG0.488 0.510 0.301 0.4240.17730.817.47 0.213
LLaVA
with CXR-CLIPImage-Only 0.393 0.448 0.215 0.343 0.161 22.38 5.60 0.207
+ Concepts 0.486 0.510 0.283 0.4130.18428.177.57 0.225
+ RAG 0.477 0.501 0.293 0.414 0.168 30.16 6.84 0.204
+CEMRAG0.488 0.512 0.300 0.4230.18030.496.98 0.206
Zero-Shot.In theZero-Shotsetting, the LLaVA-Med baseline exhibits clear lim-
itations for RRG. Despite leveraging general medical pretraining, it attains an F1-
Radgraph (F1-RG) of only 0.052, indicating poor alignment of clinical entities and
relations, and very low NLP metrics, reflecting limited lexical overlap with reference
reports. CheXbert Micro-F1 14reaches 0.255, suggesting some ability to recognize com-
mon pathologies but insufficient overall clinical reliability. Augmenting the prompt
with concepts or retrieval leads to a consistent progression in performance. Adding
concepts improves both clinical and textual metrics: F1-RG rises to 0.073, CheXbert
Micro-F1 14to 0.401, and ROUGE-L to 0.146, indicating that even in aZero-Shot
13

regime, visual concepts provide useful structured cues about relevant findings. RAG
yields substantially larger gains, with F1-RG increasing to 0.184 and CheXbert Micro-
F114to 0.498. These gains demonstrate that retrieved similar cases provide rich
contextual and linguistic information that enhances both clinical accuracy and report
fluency.CEMRAGachieves the best overallZero-Shotperformance for LLaVA-Med,
slightly but consistently outperforming RAG across all clinical and lexical met-
rics, indicating that integrating concepts with retrieved reports yields an additional
performance benefit.
A similar trend is observed for the configuration using CXR-CLIP as the visual
encoder, which starts from a strongerZero-Shotbaseline in terms of fluency as a
result of the one-epoch alignment of its projection layer on MIMIC-CXR. The baseline
achieves an F1-RG of 0.151 and BLEU-4 of 4.69, considerably higher than LLaVA-
Med, whilst CheXbert Micro-F1 14remains lower at 0.213. This pattern suggests that
the alignment primarily improves generic report structure and phrasing, but does
not immediately translate into superior pathology coverage. As in the previous case,
both concept and retrieval augmentation are beneficial:Conceptsstrategy increases
CheXbert Micro-F1 14to 0.328 and F1-RG to 0.172, whereas RAG raises these val-
ues to 0.489 and 0.181, respectively.CEMRAGconfiguration attains again the best
overallZero-Shotperformance in this setting, with F1-RG of 0.187, CheXbert Micro-
F114of 0.498, and the highest BLEU-1 (29.78), BLEU-4 (6.08), and ROUGE-L
(0.201). Overall, theseZero-Shotresults indicate that concept-based and retrieval-
based augmentation contribute positively across both architectures, and that their
combination systematically improves clinical and lexical metrics over Image-Only and
single-augmentation baselines.
SFT.UnderSFT, theImage-Onlybaselines of both architectures exhibit substan-
tial improvements over theirZero-Shotcounterparts, confirming the importance of
task-specific adaptation for RRG. The LLaVA-Med baseline shows substantial gains
across all metrics, with F1-RG increasing from 0.052 to 0.174, CheXbert Micro-F1 14
from 0.255 to 0.470, and BLEU-4 from 0.71 to 7.43, with ROUGE-L reaching 0.211.
On top of this stronger baseline, the three augmentation strategies remain beneficial,
but their role shifts compared to theZero-Shotsetting. Adding only concepts yields
the highest F1-RG in this configuration (0.188) and the best BLEU-4 and ROUGE-L
scores (7.87 and 0.220), along with moderate gains in CheXbert Micro- and Macro-
F1 (0.476 and 0.284). In this setting,Conceptsstrategy yields gains on CheXbert
while having a larger impact on sequence-level metrics such as F1-RG, BLEU-4, and
ROUGE-L, suggesting that explicit visual concepts are especially beneficial for struc-
turing clinically detailed reports. RAG shows a different pattern: F1-RG remains close
to the baseline (0.176 vs. 0.174), whereas BLEU-1 increases to 30.50. This pattern
suggests that, once the model has been adapted on MIMIC-CXR using retrieval-
augmented prompts, the additional in-domain retrieved reports mainly enrich lexical
diversity and increase report length, without yielding proportional gains on metrics
that are more sensitive to clinical structure and relational correctness. Finally, the
combinedCEMRAGachieves the strongest overall performance: it attains the high-
est CheXbert Micro-F1 14of 0.488 and Macro-F1 14of 0.301, along with the highest
14

BLEU-1 score of 30.81. These results confirm that interpretability-driven augmenta-
tion provides complementary benefits toSFT, specifically for clinical accuracy metrics
that directly measure factual correctness in generated reports.
For the CXR-CLIP configuration, theSFTbaseline remains weaker than the cor-
responding LLaVA-Med baseline on MIMIC-CXR (e.g., CheXbert Micro-F1 14of 0.393
vs. 0.470), reflecting the advantage of LLaVA-Med’s medically pretrained language
component. The augmentation strategies follow the same progressive improvement
pattern observed in other configurations.Conceptsaugmentation produces dramatic
gains, with F1-RG jumping to 0.184 and CheXbert Micro-F1 14reaching 0.488, sub-
stantially exceeding the improvements observed with LLaVA-Med and nearly recover-
ing the baseline performance deficit. Notably, this configuration achieves the highest
BLEU-4 score of 7.57 across allSFTCXR-CLIP settings and the highest overall
ROUGE-L of 0.225, mirroring the pattern observed with LLaVA-Med. These results
further demonstrate that interpretable visual concepts offer structured information
that effectively complements visual encoder features. RAG-only yields mixed results,
with F1-RG at 0.168 and CheXbert Macro-F1 14reaching 0.293, while CheXbert Micro-
F114of 0.477 remains below theConceptsconfiguration. The combinedCEMRAG
approach achieves balanced performance with the highest CheXbert Micro-F1 14at
0.488 and Macro-F1 14at 0.300, alongside the highest BLEU-1 of 30.49.
MIMIC-CXR Summary.Taken together, the MIMIC-CXR experiments reveal
two main patterns regarding the interaction between augmentation strategies and
training regimes.
First, the effectiveness of RAG is strongly regime-dependent. In theZero-Shotset-
ting, adding in-domain retrieved reports substantially improves clinical and linguistic
metrics, consistent with models that rely heavily on external context to compensate
for limited task-specific adaptation. In theSFTregime, where models are trained with
retrieved reports in the prompt, the role of retrieval changes. While unigram coverage
(BLEU-1) improves substantially, metrics sensitive to longer-range structure (BLEU-
4, ROUGE-L) underperform relative toConceptsaugmentation. Furthermore, clinical
metrics exhibit modest yet consistent degradation compared toZero-Shotfor both
RAG andCEMRAG. This behaviour suggests that when supervision and retrieval are
drawn from the same in-domain distribution, retrieval provides diminishing returns as
the model internalizes distributional patterns present in the retrieved context.
Secondly, SpLiCE-derived concepts contribute consistently across both regimes.
InZero-Shotconfigurations,Conceptsalone clearly improve pathology identification
and F1-RG. In theSFTsetting, SpLiCE contributes more evidently on metrics that
capture the quality of clinically complex sequences, while the gain on CheXbert F1 is
comparatively smaller.
The combined approach,CEMRAG, which combines concept and retrieval aug-
mentation, leverages the complementary strengths of both: inZero-Shot, it slightly
amplifies the benefits of RAG; in theSFTregime, it maintains or improves aggre-
gate clinical performance while attenuating some of the redundancy associated with
retrieval alone.
15

Finally, the systematic gap between CheXbert Micro-F1 and Macro-F1 across all
methods reflects the pronounced class imbalance in chest X-ray reports. The improve-
ments in Macro-F1 observed withCEMRAGindicate that the method contributes to
a better coverage of underrepresented findings, supporting its suitability for clinically
realistic reporting scenarios.
5.1.2 IU X-ray
Table 3 presents quantitative results on the IU X-ray dataset. This dataset, differ-
ently from MIMIC-CXR, features considerably shorter and more concise reports, and
employs a more limited medical vocabulary. In this case, our experimental setup imple-
ments cross-domain retrieval, where similar cases are retrieved from the MIMIC-CXR
database rather than from IU X-ray’s own training set. This configuration puts our
framework’s ability to generalize across different datasets and reporting styles to the
test, since the retrieved context originates from a different institutional source with
different imaging protocols and documentation practices.
Table 3Quantitative results on the IU X-Ray test set for two model configurations (LLaVA-Med
and LLaVA with CXR-CLIP) and two training regimes (Zero-ShotandSupervised Fine-Tuning).
We report F1-RadGraph (F1-RG), and NLP metrics (BLEU-1 as B-1, BLEU-4 as B-4, ROUGE-L
as R-L). CheXbert-based label metrics are reported over 14 labels (Micro-F1 14, Macro-F1 14) and
over the 5 most prevalent findings (Micro-F1 5, Macro-F1 5). Retrieval is performed cross-domain
from the MIMIC-CXR training set.
Model Method Micro-F1 14Micro-F1 5Macro-F1 14Macro-F1 5F1-RG B-1 B-4 R-L
Zero-shot
LLaVA-MedImage-Only 0.063 0.042 0.047 0.019 0.074 17.36 1.08 0.125
+ Concepts 0.122 0.162 0.092 0.134 0.064 13.79 0.85 0.111
+ RAG 0.377 0.344 0.220 0.271 0.228 21.04 3.60 0.177
+CEMRAG0.387 0.397 0.252 0.315 0.234 24.34 4.10 0.191
LLaVA
with CXR-CLIPImage-Only 0.305 0.082 0.038 0.043 0.188 5.56 1.26 0.168
+ Concepts 0.307 0.182 0.085 0.087 0.199 8.15 1.98 0.199
+ RAG 0.367 0.343 0.212 0.247 0.213 25.35 6.13 0.203
+CEMRAG0.378 0.361 0.232 0.298 0.247 27.84 6.75 0.221
Supervised Fine-Tuning
LLaVA-MedImage-Only 0.326 0.115 0.031 0.059 0.175 14.25 4.50 0.179
+ Concepts 0.336 0.185 0.082 0.081 0.178 23.21 5.67 0.181
+ RAG 0.468 0.356 0.183 0.205 0.249 28.15 7.73 0.251
+CEMRAG0.501 0.526 0.244 0.355 0.252 28.37 8.00 0.252
LLaVA
with CXR-CLIPImage-Only 0.376 0.102 0.037 0.053 0.235 14.25 4.50 0.235
+ Concepts 0.427 0.362 0.118 0.178 0.244 18.83 6.06 0.242
+ RAG 0.468 0.395 0.172 0.245 0.244 22.58 6.91 0.243
+CEMRAG0.486 0.439 0.174 0.256 0.248 22.90 7.10 0.249
Zero-Shot.In theZero-Shotsetting, the LLaVA-Med baseline performs very poorly
on IU X-Ray, with a CheXbert Micro-F1 14of 0.063 and an F1-RG of 0.074, confirm-
ing that general medical pretraining does not directly translate into effective RRG
for this dataset. TheConceptscondition produces a mixed effect: CheXbert Micro-
F114roughly doubles to 0.122, indicating improved detection of some pathologies,
yet F1-RG decreases slightly to 0.064 and all NLP metrics deteriorate (e.g., BLEU-1
drops from 17.36 to 13.79). This behaviour is consistent with a style mismatch: when
concepts are injected without further constraints, LLaVA-Med tends to expand each
16

keyword into lengthy explanatory sentences. This contrasts with the highly concise IU
X-ray references, reducing n-gram overlap despite potentially correct clinical content.
In contrast, cross-domain retrieval yields substantial improvements. RAG on MIMIC-
CXR increases CheXbert Micro-F1 14to 0.377, F1-RG to 0.228, and BLEU-4 to 3.60,
indicating that retrieved examples provide useful templates despite originating from
a different institution and reporting style. Finally,CEMRAGachieves the best over-
allZero-Shotperformance for LLaVA-Med (Micro-F1 140.387, F1-RG 0.234, BLEU-4
4.10), suggesting that concepts help the model focus on clinically salient parts of the
retrieved reports and partially counteract the verbosity that arises when concepts are
used in isolation.
For LLaVA with CXR-CLIP configuration, theZero-Shotbaseline attains
CheXbert Micro-F1 14= 0.305 and F1-RG = 0.188, substantially higher than LLaVA-
Med. This reflects the one-epoch projector alignment on IU X-ray, which provides
better dataset-specific adaptation compared to LLaVA-Med’s general medical pre-
training. On the other hand, performance on the 5-label subset is extremely weak
(Micro-F1 5= 0.082) and several Macro-F1 scores are low, indicating uneven pathology
recognition across label subsets. NLP metrics are also poor (BLEU-1 = 5.56, BLEU-
4 = 1.26), showing that this short alignment phase only partially adapts the model
to the concise IU X-ray style.Conceptsaugmentation provides modest yet consistent
gains (F1-RG = 0.199, slight improvements in CheXbert metrics), though effectiveness
is limited by the projector alignment on IU X-ray’s small training set. RAG produces
much larger benefits: Micro-F1 14rises to 0.367, F1-RG to 0.213, and BLEU-1 jumps to
25.35, confirming that cross-domain retrieval from MIMIC-CXR supplies useful clini-
cal templates and linguistic structure. The combinedCEMRAGcondition yields the
strongestZero-Shotperformance for this configuration, with Micro-F1 14= 0.378, F1-
RG = 0.247, BLEU-1 = 27.84, BLEU-4 = 6.75, and ROUGE-L = 0.221. This indicates
that concept-level signals derived from SpLiCE help the model exploit cross-domain
retrieved context more selectively, improving both clinical accuracy and report quality.
SFT.In theSFTsetting, LLaVA-Med exhibits substantial gains over itsZero-Shot
performance on IU X-Ray. The baseline Micro-F1 14increases from 0.063 to 0.326
and F1-RG from 0.074 to 0.175, confirming that even a relatively small amount of
supervision is sufficient to substantially improve task-specific behaviour on a new
dataset. On top of this, the augmentation strategies continue to provide consistent
benefits.Conceptsaugmentation yields modest improvements, whereas RAG leads to
more pronounced gains, reaching 0.468 Micro-F1 14and 0.249 F1-RG. The combined
CEMRAGcondition achieves the strongest overall performance, with 0.501 Micro-
F114, 0.252 F1-RG, and BLEU-4 of 8.00. Unlike the in-domain setting on MIMIC-
CXR, where RAG becomes partially redundant afterSFT, here cross-domain retrieval
from MIMIC-CXR continues to supply complementary information that is not fully
captured by supervised training on the much smaller IU X-Ray corpus, andCEMRAG
is able to exploit this additional signal more effectively.
For LLaVA with CXR-CLIP configuration,SFTon IU X-ray leads to analo-
gous trends. TheSFTbaseline reaches CheXbert Micro-F1 14= 0.376 and F1-RG =
0.235;Conceptsaugmentation improves sequence-level metrics, with F1-RG = 0.244,
BLEU-4 = 6.06, and ROUGE-L = 0.242; RAG further increases label-based scores
17

(Micro-F1 14= 0.468, F1-RG = 0.244). The fullCEMRAGconfiguration yields a bal-
anced improvement, with Micro-F1 14= 0.486, F1-RG = 0.248, and BLEU-4 = 7.10,
combining the benefits of concept guidance and retrieval-based context. Despite the
comparable clinical metrics underCEMRAG, a clear discrepancy persists in lexical
quality: LLaVA-Med attains substantially higher BLEU-1 (28.37 vs. 22.90). This is
plausibly due to its medically pretrained language component, which starts from a
richer medical vocabulary and a broader set of reporting patterns. In the CXR-CLIP
variant, the Mistral-7B backbone is adapted only on the small IU X-ray corpus, limit-
ing its ability to acquire diverse and specialised radiology phrasing. As a result, report
generations remain less varied and exhibit lower n-gram overlap, even when clinical
content is comparable.
IU X-ray Summary.Overall, the IU X-ray experiments show that the proposed
augmentation strategies remain effective in a low-resource, cross-domain scenario.SFT
substantially improves both backbones, cross-domain RAG continues to offer clear
benefits rather than becoming redundant, andCEMRAGconsistently matches or
exceeds the single augmentations. At the same time, the persistent gap in BLEU-1
between LLaVA-Med and CXR-CLIP configurations underscores the practical value
of a medically pretrained language module for producing lexically rich reports when
task-specific supervision is scarce.
5.2 Qualitative Results
While quantitative metrics capture global trends across datasets and configurations,
qualitative analysis helps clarify how concept extraction and retrieval affect individ-
ual predictions. In this section, we analyse two representative chest X-ray cases from
MIMIC-CXR using the LLaVA-Med configuration. The first illustrates typicalZero-
Shotfailure modes and their mitigation withCEMRAG, while the second examines
how the same mechanisms behave afterSFT. For each case, we compare reports
produced by the Image-Only,Concepts, RAG, andCEMRAGconditions, and high-
light patterns of hallucination, omission, superfluity, as well as clinically accurate
description.
5.2.1 Generated Report Comparison
Zero-Shot.Table 4 shows aZero-Shotexample on MIMIC-CXR that reflects the
error patterns observed in Table 2. The ground truth report describes multiple devices
(endotracheal tube, orogastric tube, right internal jugular catheter), low lung volumes,
and a right upper lobe opacity suspicious for pneumonia, explicitly ruling out pleural
effusion and pneumothorax.
The baseline LLaVA-Med model produces a markedly incorrect description, hallu-
cinating bilateral diffuse infiltrates and failing to mention any of the indwelling devices,
in line with the very lowZero-ShotF1-RG observed quantitatively. This behaviour
indicates that, without additional guidance, the model tends to generate generic pat-
terns of abnormality that are not well grounded in the specific image. WithConcepts
augmentation, all extracted keywords (e.g., “endotracheal tube”, “internal jugular”,
18

Table 4Qualitative Comparison inZero-ShotSetting for LLaVA-Med. Highlighting indicates:
incor rect/hal lucinated findings, incom plete descrip tions, super fluousdetails, accuratecontent.
Zero-Shot Setting
Chest X-Ray Image Ground Truth Report
Endotracheal tube terminates 2 cm above the carina. Orogastric tube termi-
nates in the stomach. Right internal jugular catheter terminates in the mid
SVC. Lungs are low in volume with stable right upper lung opacities which are
better assessed on the recent chest CT but suspicious for pneumonia. There is
no pneumothorax or pleural effusion. Heart is normal in size. Normal cardio-
mediastinal silhouette.
SpLiCE Concepts:pulmonary vascularity, overt pulmonary, right upper,
endotracheal tube, internal jugular
Image-OnlyThe chest X-ray image shows bilateral diffuse infiltrates, which areabnor -
malsubstances orcollections ofcells that have entered thelung tissue. These
infiltrates can be seen in both lungs and are distributed throughout the lung
fields.
+ ConceptsThe chest X-ray shows overt pulmonary edema, which is a condition where
there is an abnormal accumulation of fluid in the lungs. Additionally, there
is aright upper lobe opacity, which could be due to various reasons such as
infection, inflammation, or other lung abnormalities. The internal jugular vein
is visible, and an endotracheal tube isinplace, which is a tube that isinserted
through themouth ornose andintothetrachea tohelp main tainanopen air-
wayandassist with breath ing. Lastly, the pulmonary vascularity appears to
be normal.
+ RAGThe chest X-rayfindings aresimilartothose intheearlierstudy. The main
differences observed are the placement of a leftIJcatheter and the presence of
anindistinct areaofopaci ficationintheright upper zone medi ally, which could
represent a developing consolidation. Additionally, there is a right IJcentral
venous catheter with its tip at the lower SVC near the cavoatrial junction, and
theendotracheal tube has been slightly advanced. Nopneu moth orax is noted.
Pulmonary vascularcongestionisobserved.
+ CEMRAGIn the right upper zone, there isanindistinct area ofopaci fication, which
could represent a developing consolidation. Additionally, there is a right inter-
naljugularcentralvenous catheter with its tip at the lower superior vena
cava near the cavoatrial junction. The endotracheal tube has been slightly
advanced, with its tip 2.2 cm above the carina. Nopneu moth orax is noted,
and the pulmonary vascularcongestionispresent. Thecardiomediastinalsil-
houetteremains unchanged.
“right upper”) are explicitly mentioned in the generated text, and the report cor-
rectly identifies both the endotracheal tube and a right upper lobe opacity. However,
the model now over-interprets the concept set by asserting overt pulmonary edema,
a finding not present in the reference report. This illustrates how SpLiCE decompo-
sition can substantially increase CheXbert Micro-F1 by encouraging the mention of
clinically salient terms, while still lagging behind retrieval-based strategies on metrics
that reward correct entity–relation structure. Notably, the model produces verbose
explanations of medical concepts (e.g., describing what an endotracheal tube does),
as LLaVA-Med’s general medical pretraining has not been adapted to the concise
technical language conventions of radiology reports. RAG-only reduces some of these
concept-driven hallucinations but introduces others. Conditioning on retrieved in-
domain reports leads the model to blend details from similar but non-identical cases,
such as mentioning a left internal jugular catheter and pulmonary vascular congestion
that are not supported by the target image. This exemplifies a typical retrieval-induced
failure mode in which information from nearest neighbours leaks into the generated
report. The combinedCEMRAGcondition achieves the most clinically plausible
19

description. It correctly identifies the right internal jugular catheter, the endotracheal
tube position, and the right upper zone opacity, and avoids the more severe hallu-
cinations observed in the RAG-only report, although it still mentions mild vascular
congestion. This example mirrors the quantitative trends: concept cues alone can act
as over-strong priors, and retrieval alone can import spurious details, whereas their
combination yields better grounded descriptions that more closely match the reference
report.
SFT.The second example (Table 5) illustrates behaviour in theSFTsetting on
MIMIC-CXR. The reference report describes low lung volumes and bibasilar opacities
most likely representing atelectasis, and explicitly states the absence of pleural effusion
or pneumothorax, with a stable and unremarkable cardiomediastinal silhouette.
Table 5Qualitative Comparison inSupervised Fine-TuningSetting or LLaVA-Med. Highlighting
indicates: incor rect/hal lucinated findings, incom plete descrip tions, super fluousdetails, accurate
content.
Supervised Fine-Tuned Setting
Chest X-Ray Image Ground Truth Report
Frontal and lateral views of the chest were obtained. There are low lung vol-
umes which accentuate the bronchovascular markings. Bibasilar opacities are
seen, which most likely represent atelectasis, although aspiration or infection
are not excluded in the appropriate clinical setting. No pleural effusion or
pneumothorax is seen. The cardiac and mediastinal silhouettes are stable and
unremarkable.
SpLiCE Concepts:lateral view, lung base, volume low, left hemidiaphragm,
appropriate clinical
Image-OnlyThe heart size is normal. The hilar and mediastinal contours are normal. There
is anewleftlower lobe opacity,concerningforpneu monia. No pleural effusion
or pneumothorax. Thevisualizedosseous struc tures areunremark able.
+ ConceptsIn comparison with the study of , there are lower lung volumes. Cardiac
silhouette is within normal limits. Opaci ficationattheleftbase isconsistent
with atelectatic changes and possiblesmall effusion. Intheappro priateclini-
calsetting, super imposed pneu moniawould have tobeconsidered.
+ RAGThe lungs areclear withoutfocal consolidation. Nopleuraleffusion orpneu -
moth oraxisseen. Thecardiac andmedi astinalsilhouettes areunremark able.
+ CEMRAGLung volumes arelow.Medi astinalandhilar contours areunremark able. Pul-
monary vasculature is not engorged. Patchy opacitiesinthelung bases likely
reflect areas ofatelec tasis.Nopleuraleffusionorpneu moth oraxispresent. No
acute osseous abnormality is visualized.
The Image-Only baseline captures part of this picture: it correctly rules out pleural
effusion and pneumothorax and describes normal cardiac and mediastinal contours,
but introduces a clinically relevant error by emphasising anewleft lower lobe opac-
ity concerning for pneumonia and omitting any mention of low lung volumes. This
combination of partial correctness and misplaced emphasis is consistent with the mod-
erate CheXbert Micro-F1 and F1-RG achieved by theSFTbaseline in Table 2. With
Conceptsaugmentation, the model more faithfully reflects the ground truth: it explic-
itly recognises lower lung volumes and describes left basal opacification as atelectatic
change, while conditionally mentioning possible pneumonia “in the appropriate clini-
cal setting”, closely echoing the phrasing of the reference report. The main discrepancy
20

is the mention of a possible small effusion, which is explicitly ruled out in the ground
truth. Overall, this aligns with the observed improvements in F1-RG and sequence-
level metrics under SpLiCE. In the RAG-only condition, the report fails to mention
the bibasilar opacities and instead states that the lungs are clear without focal con-
solidation. This represents a clinically concerning omission and is consistent with a
failure mode in which the model appears to rely too heavily on retrieved studies and
does not fully ground its description in the current image. The combinedCEMRAG
report strikes a better balance: it correctly identifies low lung volumes and patchy basal
opacities likely reflecting atelectasis, preserves the correct absence of pleural effusion
and pneumothorax, and maintains a coherent description of the cardiomediastinal sil-
houette. This progression from baseline toCEMRAGmirrors the quantitative gains
reported for MIMIC-CXR and qualitatively illustrates how concept-level signals can
prevent complete omission of pathologies when retrieval alone is unreliable.
5.2.2 Interpretability Through Concept Visualization
Beyond improvements in clinical accuracy, a central motivation ofCEMRAGis to
make the visual evidence underlying generated reports explicitly inspectable. In this
section, we show how SpLiCE concepts and gradient-based explanations can be com-
bined to reveal where the model “looks” in the image when producing specific medical
terms in the report.
For each case, we select SpLiCE concepts that appear in the generated report
(e.g.,endotracheal tube, right upper, median sternotomy, bilateral pleural). We apply
Grad-ECLIP [57] to compute the gradient of the image–text similarity score with
respect to the visual features used by the LLaVA-Med vision encoder. The result-
ing relevance map is then upsampled and overlaid on the chest X-ray as a heat map,
yielding a concept-specific visualization of which regions support that term.CEM-
RAG-generated reports and their corresponding concept heat maps are illustrated in
Table 6 for two representative MIMIC-CXR cases, one in theZero-shotregime and one
afterSFT. In each example, the left column shows the reference report, while the right
column shows theCEMRAGoutput with the relevant concept tokens highlighted.
Below these reports, we display the original image together with Grad-ECLIP maps
for two selected concepts. In theZero-shotexample, concepts such as endotracheal
tube and right upper are both present in the generated report and exhibit Grad-
ECLIP activations that concentrate along the tracheal tube and in the right upper
lung zone, respectively. In theSFTexample, the concept median sternotomy yields
strong activation along the midline sternal wires and retrosternal region, while bilat-
eral pleural produces strongest activation in the lower lung zones near the right and
left costophrenic angles, where pleural effusions typically accumulate. Taken together,
these patterns are consistent with standard radiological practice and suggest that,
when a concept appears in the report, it is usually backed by visually plausible evidence
in the corresponding image regions.
These concept-specific maps serve two complementary purposes. First, they allow
clinicians to verify that SpLiCE-derived concepts, when injected into the prompt, are
grounded in image regions that are consistent with their radiological meaning, rather
than reflecting arbitrary directions in embedding space. Second, they expose failure
21

Table 6Qualitative Examples with Concept-Specific Visual Explanations. Grad-ECLIP heat
maps highlight image regions that influence the LLaVA-Med encoder’s matching between input
images and SpLiCE concepts (highlighted in blue). The visualizations show which spatial features
contribute to concept identification in the generated reports. Top row: Zero-Shot setting; bottom
row: supervised fine-tuning setting.
Ground Truth Generated Report
Endotracheal tube terminates 2 cm above the
carina. Orogastric tube terminates in the stom-
ach. Right internal jugular catheter terminates in
the mid SVC. Lungs are low in volume with sta-
ble right upper lung opacities which are better
assessed on the recent chest CT but suspicious for
pneumonia. There is no pneumothorax or pleural
effusion. Heart is normal in size. Normal cardio-
mediastinal silhouette.The chest X-ray shows several findings. In the
right upperzone, there is an indistinct area of
opacification, which could represent a developing
consolidation. Additionally, there is a right inter-
nal jugular central venous catheter with its tip
at the lower superior vena cava near the cavoa-
trial junction. The endotracheal tubehas been
slightly advanced, with its tip 2.2 cm above the
carina. No pneumothorax is noted, and the pul-
monary vascular congestion is present. The car-
diomediastinal silhouette remains unchanged.
Original Image Grad-ECLIP Activation Maps
Endotracheal Tube Right Upper
Ground Truth Generated Report
The patient is status post median sternotomy
and aortic valve replacement. Heart size is mod-
erately enlarged but unchanged. The aorta is
tortuous and calcified. There is mild intersti-
tial pulmonary edema, relatively unchanged. At
least small bilateral pleural effusions are present.
Bibasilar airspace opacities may reflect compres-
sive atelectasis. There is no pneumothorax. There
is evidence of prior vertebroplasty at the thora-
columbar junction.The patient is status post median sternotomy
and CABG. The heart size remains mildly
enlarged. The aorta is tortuous and diffusely cal-
cified. There is mild pulmonary edema, worse in
the interval, with small bilateral pleuraleffu-
sions. Bibasilar airspace opacities likely reflect
atelectasis. No pneumothorax is identified. There
are no acute osseous abnormalities.
Original Image Grad-ECLIP Activation Maps
Median Sternotomy Bilateral Pleural
modes in a transparent way: if a concept is mentioned in the report but its Grad-
ECLIP map does not align with plausible anatomy, this discrepancy can be inspected,
questioned, and potentially used to flag low-trust outputs. At the same time, this
analysis remains focused on the vision component: Grad-ECLIP explains how visual
encoders support individual concepts, but does not by itself reveal how the LLM
composes multiple concepts into full sentences. Extending concept-level tracing to
the language component remains an important direction for future work toward fully
interpretable medical VLMs.
22

6 Conclusion
This work addressed two major barriers to the clinical deployment of vision–language
models in radiology: limited interpretability and susceptibility to hallucinations. We
introducedCEMRAG, a unified framework that integrates concept decomposition
with multimodal RAG to jointly enhance transparency and factual accuracy in radi-
ology report generation. We further established a comprehensive evaluation protocol
that compares multiple prompting strategies across different VLM architectures,
retrieval configurations, and datasets, using both lexical similarity and clinically ori-
ented correctness metrics. Across experiments on MIMIC-CXR and IU X-ray, our
results indicate that interpretable visual concepts can improve factual grounding
and concept-level transparency simultaneously, challenging the commonly assumed
trade-off between interpretability and performance in medical AI. From a clinical per-
spective, the proposed framework offers a practical way to present AI-generated draft
reports together with explicit visual concepts and retrieved reference cases, poten-
tially facilitating more efficient review while preserving radiologists’ ability to verify
how findings in the image relate to the generated text.
Despite these advances, several limitations warrant further investigation. The over-
all effectiveness of the framework depends critically on the quality of the SpLiCE
decomposition: if the underlying CLIP encoders are not sufficiently aligned with
domain-specific semantics, the extracted concepts may be noisy or incomplete. Improv-
ing this alignment, for example through domain-adaptive pretraining or architectural
refinements, is an important direction for future work. Moreover, interpretability in
the current pipeline is primarily concentrated in the vision encoder and concept layer,
while the language model is influenced only indirectly via prompt conditioning. Future
research should explore mechanisms that more directly constrain or regularize token
probabilities during generation, extending interpretability to the full model. Finally,
our experiments focused on an LLM backbone such as Mistral-7B; however, recent
progress on smaller language models with competitive performance and lower com-
putational cost suggests a promising avenue for achieving finer-grained control over
interpretability and deployment in resource-constrained clinical settings.
By showing that concept-level interpretability can enhance rather than undermine
factual accuracy, this work provides empirical support for the development of trans-
parent and accurate VLMs for radiology. The modular design ofCEMRAGenables
targeted optimization of individual components and offers a general methodological
template that can be extended beyond chest X-ray analysis to other medical imaging
domains where visual interpretation and textual reporting are required, provided that
suitable domain-specific concept vocabularies and retrieval corpora are available.
7 Acknoweldgments
Marco Salm` e is a Ph.D. student enrolled in the National Ph.D. in Artificial Intelligence,
XXXIX cycle, course on Health and Life Sciences, organized by Universit` a Campus
Bio-Medico di Roma. This work was partially funded by: i) Universit` a Campus Bio-
Medico di Roma under the program “University Strategic Projects” within the project
“AI-powered Digital Twin for next-generation lung cancEr cAre (IDEA)”; ii) PNRR
23

MUR project PE0000013-FAIR. iii) Cancerforskningsfonden Norrland project MP23-
1122; iv) Kempe Foundation project JCSMK24-0094. Resources are provided by the
National Academic Infrastructure for Supercomputing in Sweden (NAISS) and the
Swedish National Infrastructure for Computing (SNIC) at Alvis @ C3SE, partially
funded by the Swedish Research Council through grant agreements no. 2022-06725
and no. 2018-05973.
References
[1] Zhang, J., Huang, J., Jin, S., Lu, S.: Vision-language models for vision tasks: A
survey. IEEE transactions on pattern analysis and machine intelligence46(8),
5625–5644 (2024)
[2] Van, M.-H.,et al.: On large visual language models for medical imaging analy-
sis: An empirical study. In: 2024 IEEE/ACM Conference on Connected Health:
Applications, Systems and Engineering Technologies (CHASE), Pages=172–176
(2024). IEEE
[3] Hartsock, I., Rasool, G.: Vision-language models for medical report generation
and visual question answering: A review. Frontiers in artificial intelligence7,
1430984 (2024)
[4] Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H., Wang, S., Yin, D., Du,
M.: Explainability for large language models: A survey. ACM Transactions on
Intelligent Systems and Technology15(2), 1–38 (2024)
[5] Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng,
W., Feng, X., Qin, B.,et al.: A survey on hallucination in large language mod-
els: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems43(2), 1–55 (2025)
[6] Vatsa, M., Jain, A., Singh, R.: Adventures of trustworthy vision-language models:
A survey. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol.
38, pp. 22650–22658 (2024)
[7] Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F., Lakkaraju, H.: Interpret-
ing clip with sparse linear concept embeddings (splice). Advances in Neural
Information Processing Systems37, 84298–84328 (2024)
[8] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Sun, J., Wang, H.: Retrieval-
augmented generation for large language models: A survey
[9] Bernardi, M.L., Cimitile, M.: Report generation from x-ray imaging by retrieval-
augmented generation and improved image-text matching. In: 2024 International
Joint Conference on Neural Networks (IJCNN), pp. 1–8 (2024). IEEE
24

[10] Sun, L., Zhao, J., Han, M., Xiong, C.: Fact-aware multimodal retrieval aug-
mentation for accurate medical radiology report generation. arXiv preprint
arXiv:2407.15268 (2024)
[11] Yu, H., Gan, A., Zhang, K., Tong, S., Liu, Q., Liu, Z.: Evaluation of retrieval-
augmented generation: A survey. In: CCF Conference on Big Data, pp. 102–120
(2024). Springer
[12] Ennab, M., Mcheick, H.: Enhancing interpretability and accuracy of ai models in
healthcare: a comprehensive review on challenges and future directions. Frontiers
in Robotics and AI11, 1444763 (2024)
[13] Park, D.H., Hendricks, L.A., Akata, Z., Rohrbach, A., Schiele, B., Darrell, T.,
Rohrbach, M.: Multimodal explanations: Justifying decisions and pointing to
the evidence. In: Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 8779–8788 (2018)
[14] Nguyen, T.T.H., Clement, T., Nguyen, P.T.L., Kemmerzell, N., Truong, V.B.,
Nguyen, V.T.K., Abdelaal, M., Cao, H.: Langxai: Integrating large vision models
for generating textual explanations to enhance explainability in visual perception
tasks. CoRR (2024)
[15] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou,
D.,et al.: Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems35, 24824–24837 (2022)
[16] Zheng, G., Yang, B., Tang, J., Zhou, H.-Y., Yang, S.: Ddcot: Duty-distinct chain-
of-thought prompting for multimodal reasoning in language models. Advances in
Neural Information Processing Systems36, 5168–5191 (2023)
[17] Chen, Y., Sikka, K., Cogswell, M., Ji, H., Divakaran, A.: Measuring and improving
chain-of-thought reasoning in vision-language models. In: Proceedings of the 2024
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 192–210
(2024)
[18] Bereska, L., Gavves, S.: Mechanistic interpretability for ai safety-a review. Trans-
actions on Machine Learning Research
[19] Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L.,
Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J.I., et al.: Open
problems in mechanistic interpretability. CoRR (2025)
[20] Jiang, Z., Chen, J., Zhu, B., Luo, T., Shen, Y., Yang, X.: Devils in middle layers
of large vision-language models: Interpreting, detecting and mitigating object
hallucinations via attention lens. In: Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 25004–25014 (2025)
25

[21] Huo, J., Yan, Y., Hu, B., Yue, Y., Hu, X.: Mmneuron: Discovering neuron-level
domain-specific interpretation in multimodal large language model. In: Pro-
ceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing, pp. 6801–6816 (2024)
[22] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., Garriga-Alonso, A.:
Towards automated circuit discovery for mechanistic interpretability. Advances
in Neural Information Processing Systems36, 16318–16352 (2023)
[23] Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang,
P.: Concept bottleneck models. In: International Conference on Machine Learning,
pp. 5338–5348 (2020). PMLR
[24] Rao, S., Mahajan, S., B¨ ohle, M., Schiele, B.: Discover-then-name: Task-agnostic
concept bottlenecks via automated concept discovery. In: European Conference
on Computer Vision, pp. 444–461 (2024). Springer
[25] Gandelsman, Y., Efros, A.A., Steinhardt, J.: Interpreting clip’s image representa-
tion via text-based decomposition. In: The Twelfth International Conference on
Learning Representations
[26] Balasubramanian, S., Basu, S., Feizi, S.: Decomposing and interpreting image
representations via text in vits beyond clip. Advances in Neural Information
Processing Systems37, 81046–81076 (2024)
[27] Parekh, J., Khayatan, P., Shukor, M., Newson, A., Cord, M.: A concept-based
explainability framework for large multimodal models. Advances in Neural Infor-
mation Processing Systems37, 135783–135818 (2024)
[28] He, J., Zhang, B., Rouhizadeh, H., Chen, Y., Yang, R., Lu, J., Chen, X., Liu, N.,
Li, I., Teodoro, D.: Retrieval-augmented generation in biomedicine: A survey of
technologies, datasets, and clinical applications. arXiv preprint arXiv:2505.01146
(2025)
[29] Abootorabi, M.M., Zobeiri, A., Dehghani, M., Mohammadkhani, M., Moham-
madi, B., Ghahroodi, O., Baghshah, M.S., Asgari, E.: Ask in any modality:
A comprehensive survey on multimodal retrieval-augmented generation. arXiv
preprint arXiv:2502.08826 (2025)
[30] Jin, Y., Zhang, Y.: Orthodoc: Multimodal large language model for assisting
diagnosis in computed tomography. arXiv preprint arXiv:2409.09052 (2024)
[31] Tozuka, R., Johno, H., Amakawa, A., Sato, J., Muto, M., Seki, S., Komaba, A.,
Onishi, H.: Application of notebooklm, a large language model with retrieval-
augmented generation, for lung cancer staging. Japanese Journal of Radiology
43(4), 706–712 (2025)
26

[32] Thetbanthad, P., Sathanarugsawait, B., Praneetpolgrang, P.: Application of gen-
erative artificial intelligence models for accurate prescription label identification
and information retrieval for the elderly in northern east of thailand. Journal of
Imaging11(1), 11 (2025)
[33] Xia, P., Zhu, K., Li, H., Wang, T., Shi, W., Wang, S., Zhang, L., Zou, J., Yao, H.:
Mmed-rag: Versatile multimodal rag system for medical vision language models.
In: The Thirteenth International Conference on Learning Representations
[34] Xia, P., Zhu, K., Li, H., Zhu, H., Li, Y., Li, G., Zhang, L., Yao, H.: Rule: Reliable
multimodal rag for factuality in medical vision language models. CoRR (2024)
[35] Wang, J., Ashraf, T., Han, Z., Laaksonen, J., Anwer, R.M.: Mira: A novel frame-
work for fusing modalities in medical rag. arXiv preprint arXiv:2507.07902 (2025)
[36] Chu, Y.-W., Zhang, K., Malon, C., Min, M.R.: Reducing hallucinations of medical
multimodal large language models with visual retrieval-augmented generation. In:
Workshop on Large Language Models and Generative AI for Health at AAAI 2025
[37] Sloan, P., Clatworthy, P., Simpson, E., Mirmehdi, M.: Automated radiology report
generation: A review of recent advances. IEEE Reviews in Biomedical Engineering
18, 368–387 (2024)
[38] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J.,et al.: Learning transferable visual models from
natural language supervision. In: International Conference on Machine Learning,
pp. 8748–8763 (2021). PmLR
[39] You, K., Gu, J., Ham, J., Park, B., Kim, J., Hong, E.K., Baek, W., Roh, B.: Cxr-
clip: Toward large scale chest x-ray language-image pre-training. In: International
Conference on Medical Image Computing and Computer-Assisted Intervention,
pp. 101–111 (2023). Springer
[40] Johnson, A.E., Pollard, T.J., Greenbaum, N.R., Lungren, M.P., Deng, C.-y., Peng,
Y., Lu, Z., Mark, R.G., Berkowitz, S.J., Horng, S.: Mimic-cxr-jpg, a large publicly
available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042
(2019)
[41] Demner-Fushman, D., Kohli, M.D., Rosenman, M.B., Shooshan, S.E., Rodriguez,
L., Antani, S., Thoma, G.R., McDonald, C.J.: Preparing a collection of radiology
examinations for distribution and retrieval. Journal of the American Medical
Informatics Association23(2), 304–310 (2015)
[42] Chen, Z., Shen, Y., Song, Y., Wan, X.: Cross-modal Memory Networks for Radi-
ology Report Generation. In: Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint Con-
ference on Natural Language Processing (Volume 1: Long Papers), pp. 5904–5914
27

(2021)
[43] Liu, F., Wu, X., Ge, S., Fan, W., Zou, Y.: Exploring and distilling poste-
rior and prior knowledge for radiology report generation. In: Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
13753–13762 (2021)
[44] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin trans-
former: Hierarchical vision transformer using shifted windows. In: Proceedings of
the IEEE/CVF International Conference on Computer Vision, pp. 10012–10022
(2021)
[45] Alsentzer, E., Murphy, J.R., Boag, W., Weng, W.-H., Jin, D., Naumann, T.,
McDermott, M.: Publicly available clinical bert embeddings. arXiv preprint
arXiv:1904.03323 (2019)
[46] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.,
et al.: Lora: Low-rank adaptation of large language models. ICLR1(2), 3 (2022)
[47] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural
information processing systems36, 34892–34916 (2023)
[48] Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., Las Casas,
D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L.R., Lachaux,
M.-A., Stock, P., Scao, T.L., Lavril, T., Wang, T., Lacroix, T., Sayed, W.E.:
Mistral 7b (2023)
[49] Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T.,
Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for
biomedicine in one day. Advances in Neural Information Processing Systems36,
28541–28564 (2023)
[50] Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazar´ e, P.-E., Lomeli,
M., Hosseini, L., J´ egou, H.: The faiss library. arXiv preprint arXiv:2401.08281
(2024)
[51] Mistral AI: Mistral AI Documentation: Prompt Engineering. https://docs.
mistral.ai/. Accessed: April 2025 (2024)
[52] Lin, C.-Y.: Rouge: A package for automatic evaluation of summaries. In: Text
Summarization Branches Out, pp. 74–81 (2004)
[53] Papineni, K., Roukos, S., Ward, T., Zhu, W.-J.: Bleu: a method for automatic
evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of
the Association for Computational Linguistics, pp. 311–318 (2002)
[54] Zhang, Y., Merck, D., Tsai, E., Manning, C.D., Langlotz, C.: Optimizing the
28

factual correctness of a summary: A study of summarizing radiology reports. In:
Proceedings of the 58th Annual Meeting of the Association for Computational
Linguistics, pp. 5108–5120 (2020)
[55] Delbrouck, J.-B., Chambon, P., Bluethgen, C., Tsai, E., Almusa, O., Langlotz, C.:
Improving the factual correctness of radiology report generation with semantic
rewards. In: Findings of the Association for Computational Linguistics: EMNLP
2022, pp. 4348–4360 (2022)
[56] Smit, A., Jain, S., Rajpurkar, P., Pareek, A., Ng, A.Y., Lungren, M.: Combining
automatic labelers and expert annotations for accurate radiology report labeling
using bert. In: Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pp. 1500–1519 (2020)
[57] Zhao, C., Wang, K., Zeng, X., Zhao, R., Chan, A.B.: Gradient-based visual
explanation for transformer-based CLIP. In: Salakhutdinov, R., Kolter, Z.,
Heller, K., Weller, A., Oliver, N., Scarlett, J., Berkenkamp, F. (eds.) Proceed-
ings of the 41st International Conference on Machine Learning. Proceedings
of Machine Learning Research, vol. 235, pp. 61072–61091. PMLR, ??? (2024).
https://proceedings.mlr.press/v235/zhao24p.html
Appendix A Visual Concept Extraction
For extracting interpretable visual concepts from CLIP embeddings, we employ Sparse
Linear Concept Embeddings (SpLiCE) [7]. Given the CLIP visual embeddingv∈Rd
extracted from the input imageI, SpLiCE approximates this embedding as a sparse
linear combination of concept embeddings drawn from a learned vocabulary.
The concept vocabulary construction begins with a set ofmmedical concepts
Q= [q 1, q2, . . . , q m] encompassing radiological findings, anatomical structures, and
pathological patterns relevant to chest radiograph interpretation. Each conceptq j∈
Qis encoded through the CLIP text encoder E txtto obtain its embeddingc j=
Etxt(qj)∈Rd, and we collect these embeddings into the matrixC= [c 1,c2, . . . ,c m]∈
Rd×m. We denote byσ(x) =x/∥x∥ 2the normalization operator. Letµc∈Rdbe the
mean concept embedding computed overC. The centered and normalized vocabulary
is then given by ˜cj=σ(c j−µc), and ˜C= [ ˜c1,˜c2, . . . , ˜cm]∈Rd×m, where each
element represents a centered and normalized concept embedding. Analogously, we
center and normalize the image embedding. Letµimg∈Rddenote the mean CLIP
image embedding computed over the training corpus. The centered and normalized
image embedding is then given by ˜v=σ(v−µimg). The sparse decomposition is
formulated as an optimization problem that balances reconstruction accuracy against
sparsity:
α∗= arg min
α≥0∥˜Cα− ˜v∥2
2+ 2λ∥α∥ 1 (A1)
whereα= [α 1, α2, . . . , α m]∈Rm
≥0represents the coefficient vector encoding the con-
tribution of each concept. The first term enforces reconstruction fidelity in the centered
29

embedding space, while the second term promotes sparsity with the regularization
parameterλ >0 controlling the trade-off. The solution to this optimization yields
a sparse coefficient vectorα∗= [α∗
1, . . . , α∗
m]∈Rm
≥0, where only a small subset of
entries are non-zero. To obtain the final set of concept keywords, we rank concepts
by their corresponding coefficient magnitudes and select the top-τconcepts. These
selected concepts correspond to interpretable keywordsΩthat represent clinically rel-
evant visual features present in the input image. The resulting keyword setΩprovides
transparency into which aspects of the image inform the subsequent RRG process and
serves as the concept component for the prompt augmentation.
A.1 Vocabulary Construction and Hyperparameter Choice
The vocabulary was constructed by extracting the most frequent bigrams from the
training corpus, with systematic exclusion of English stopwords and common medi-
cal acronyms. The decision to focus on bigrams rather than unigrams is motivated by
the inherent compositional nature of radiological terminology. Many clinically mean-
ingful concepts emerge only through the combination of terms. For instance, “pleural
effusion” and “cardiomediastinal silhouette” convey precise diagnostic information
that cannot be adequately represented by their constituent words in isolation. All
extracted terms underwent lemmatization to ensure morphological normalization and
direct compatibility with downstream textual prompts fed to the LLM.
Having established the vocabulary construction methodology, we conducted sys-
tematic ablation studies to determine optimal parameters for the SpLiCE decomposi-
tion. The optimization process involved joint exploration of three critical hyperparam-
eters: vocabulary size, L1 regularization strength (λ), and the number of top-ranked
concepts (τ) selected for each image. Our objective was to identify a configuration
that simultaneously satisfies three competing criteria: (i) precision, (ii) cosine similar-
ity and (iii) sparsity. Precision, defined as the fraction of extracted SpLiCE concepts
that appear in the corresponding ground truth radiology report, serves as our primary
quality metric. This measure directly quantifies the clinical relevance and factual accu-
racy of the extracted concepts. High precision is essential in our framework, as these
concepts are incorporated into retrieval-augmented prompts that guide report gener-
ation. Cosine similarity between the original CLIP image embedding and its sparse
reconstruction quantifies the fidelity of the decomposition. This metric ensures that
the dimensionality reduction and sparsification process preserves the essential visual
information encoded in the original representation. Sparsity, measured as the average
number of non-zero coefficients per image, reflects the interpretability-informativeness
trade-off. Excessive sparsity may omit clinically relevant concepts, while insufficient
sparsity produces verbose, representations that overwhelm the downstream LLM with
redundant information.
Fig. A1 presents a comprehensive analysis across five vocabulary sizes, corre-
sponding to the top 100, 200, 500, 700, and 1000 most frequent bigrams from the
MIMIC-CXR training corpus, and three L1 penalty values (λ∈ {0.1,0.3,0.5}). It
should be noted that for vocabulary sizes below 500, results atλ= 0.5 are unavailable
due to over-regularization: the strong sparsity penalty caused certain images to yield
30

0.1 0.3 0.5
L1 Regularization ( )
0.1500.1750.2000.2250.2500.2750.300PrecisionPrecision
Vocabulary Size
1000
700
500
200
100
0.1 0.3 0.5
L1 Regularization ( )
0.550.600.650.700.750.80Cosine SimilarityCosine Similarity
0.1 0.3 0.5
L1 Regularization ( )
1020304050Avg. # ConceptsSparsity
Fig. A1SpLiCE performance across vocabulary sizes andλvalues. The figure reports three comple-
mentary metrics:Left: precision of extracted concepts;Center: cosine similarity between the original
CLIP embedding and its sparse reconstruction;Right: average number of active concepts (sparsity).
The results highlight the trade-off between fidelity, interpretability, and terminological precision.
entirely zero-valued weight vectors, indicating that no concept exceeded the activa-
tion threshold. This phenomenon confirms the theoretical prediction that excessively
highλvalues can suppress all activations when the vocabulary-embedding alignment
is insufficient. Precision trends (left panel of Fig. A1) reveal a pronounced inverse
relationship with vocabulary size. This systematic pattern reflects a fundamental
trade-off between lexical coverage and terminological precision: smaller vocabularies,
comprising only the most frequently occurring clinical terms, naturally align with
the standardized terminology that radiologists consistently employ in their reports,
whereas larger vocabularies introduce lower-frequency terms that, despite exhibiting
visual correlation with CLIP embeddings, often represent rare synonyms, anatomi-
cal descriptors, or overly specific variants absent from actual ground truth reports.
Within each vocabulary configuration, increasingλconsistently enhances precision
by enforcing greater sparsity, thereby selecting only the most strongly activated con-
cepts while reducing false positive extractions. Reconstruction fidelity (central panel
of Fig. A1) exhibits the opposite trend: cosine similarity increases monotonically
with vocabulary size, ranging from 0.671 for the 100-bigram vocabulary atλ= 0.1 to
0.829 for the 1000-bigram vocabulary at the same regularization strength. This obser-
vation is expected, as larger dictionaries provide greater representational capacity to
approximate the original high-dimensional CLIP embedding through linear combina-
tions. However, this improved reconstruction comes at the documented cost of reduced
precision. Sparsity characteristics (right panel of Fig. A1) demonstrate that the aver-
age number of active concepts decreases both with vocabulary size reduction andλ
increment. Large vocabularies with minimal regularization, such as the 1000-bigram
vocabulary atλ= 0.1, produce approximately 48 non-zero coefficients per image, a
density incompatible with interpretable prompting and efficient retrieval. Conversely,
aggressive sparsification achieved through the 100-bigram vocabulary atλ= 0.3 yields
only 6-7 active concepts, representing a focused subset that provides sufficient contex-
tual information to guide generation while avoiding overwhelming the language model
with excessive detail. In selecting the final configuration, we assigned priority to preci-
sion maximization, recognizing that extracted concepts directly influence the factual
31

accuracy of generated reports, thereby identifying a vocabulary size of 100 or 200 as
the most promising candidates, both achieving precision exceeding 0.27.
3 5 7
Number of Selected Keywords ()
0.270.290.310.330.35Precision0.348
0.321
0.2940.335
0.313
0.286Precision vs. Top- Selection
Vocabulary Size
100
200
Fig. A2Precision as a function of the number of selected conceptsτ∈ {3,5,7}for both the
100- and 200-bigram vocabularies. Precision decreases monotonically with increasingτ, reflecting
the progressive inclusion of lower-ranked concepts with weaker activation strengths. The 100-bigram
vocabulary maintains a consistent advantage across allτvalues, indicating that the highest-confidence
concepts are highly informative for guiding retrieval-augmented report generation.
To further refine this selection and ensure consistency across the dataset, we con-
ducted an additional experiment evaluating the impact of selecting a fixed numberτ
of top-ranked concepts per image, withτ∈[3,5,7]. This analysis served two purposes:
first, to assess whether further reduction in concept count would significantly degrade
precision; and second, to standardize the input representation such that all images
contribute exactly the same number of keywords to the retrieval-augmented prompts,
facilitating uniform processing by the downstream LLM. Results are presented in
Fig. A2, where precision decreases monotonically with increasingτfor both vocabu-
laries, consistent with the progressive inclusion of concepts exhibiting lower activation
coefficients. Notably, the precision advantage of 100-bigram vocabulary diminishes as
τincreases, from a 3.9% margin atτ= 3 to 2.8% atτ= 7, suggesting that the
highest-ranked concepts exhibit comparable accuracy across both vocabularies, with
divergence manifesting primarily in lower-ranked selections. We adoptτ= 5 with
200-bigram vocabulary as our definitive configuration (precision=0.313). While the
vocabulary size of 100 offers marginally superior precision, the 200-bigram vocabulary
provides substantially broader lexical coverage, encompassing twice the vocabulary
size, with only modest precision degradation. This configuration embodies a princi-
pled balance between factual grounding through high precision and expressive capacity
through adequate lexical diversity, optimizing the extracted concepts for their ultimate
role in guiding accurate and comprehensive RRG via retrieval-augmented prompting.
32

Appendix B Training Details
CXR-CLIP adaptation.For the IU X-ray experiments, we adapt CXR-CLIP
using LoRA [46]. The LoRA modules are applied to the last stage of the Swin vision
encoder and to the final BERT text encoder layer, with rankr= 8, scaling parameter
α= 16, and dropout rate 0.1. Training is performed for ten epochs using the AdamW
optimizer, with a learning rate of 5×10−5and a weight decay of 0.01.
Projection moduleΦ CLIP.The projection module Φ CLIP is implemented as a
single linear layer that maps the 768-dimensional CLIP visual representation from the
last vision transformer block (before the standard CLIP projection head) to the LLM
embedding dimension. As an alignment step, Φ CLIP is initially trained for one epoch
with both the CLIP encoder and the LLM frozen, using a learning rate of 1×10−3,
a warmup ratio of 0.03, cosine learning-rate scheduling, and a batch size of 16.
Supervised Fine-Tuning (SFT).For SFT of the LLM, we again employ LoRA
with rankr= 64, scaling parameterα= 16, and dropout rate 0.05. Fine-tuning is
carried out for three epochs with a batch size of 16, a learning rate of 1×10−4, a
warmup ratio of 0.03, and cosine learning-rate scheduling, while keeping all visual
encoders frozen.
33
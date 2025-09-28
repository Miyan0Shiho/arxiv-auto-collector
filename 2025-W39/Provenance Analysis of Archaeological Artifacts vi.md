# Provenance Analysis of Archaeological Artifacts via Multimodal RAG Systems

**Authors**: Tuo Zhang, Yuechun Sun, Ruiliang Liu

**Published**: 2025-09-25 05:52:13

**PDF URL**: [http://arxiv.org/pdf/2509.20769v1](http://arxiv.org/pdf/2509.20769v1)

## Abstract
In this work, we present a retrieval-augmented generation (RAG)-based system
for provenance analysis of archaeological artifacts, designed to support expert
reasoning by integrating multimodal retrieval and large vision-language models
(VLMs). The system constructs a dual-modal knowledge base from reference texts
and images, enabling raw visual, edge-enhanced, and semantic retrieval to
identify stylistically similar objects. Retrieved candidates are synthesized by
the VLM to generate structured inferences, including chronological,
geographical, and cultural attributions, alongside interpretive justifications.
We evaluate the system on a set of Eastern Eurasian Bronze Age artifacts from
the British Museum. Expert evaluation demonstrates that the system produces
meaningful and interpretable outputs, offering scholars concrete starting
points for analysis and significantly alleviating the cognitive burden of
navigating vast comparative corpora.

## Full Text


<!-- PDF content starts -->

Provenance Analysis of Archaeological Artifacts via Multimodal RAG Systems
Tuo Zhang*1, Yuechun Sun*2, and Ruiliang Liu3
1Museus
2University of Science and Technology of China
3British Museum
Abstract
In this work, we present a retrieval-augmented generation
(RAG)-based system for provenance analysis of archaeo-
logical artifacts, designed to support expert reasoning by
integrating multimodal retrieval and large vision-language
models (VLMs). The system constructs a dual-modal knowl-
edge base from reference texts and images, enabling raw
visual, edge-enhanced, and semantic retrieval to identify
stylistically similar objects. Retrieved candidates are syn-
thesized by the VLM to generate structured inferences, in-
cluding chronological, geographical, and cultural attribu-
tions, alongside interpretive justifications. We evaluate the
system on a set of Eastern Eurasian Bronze Age artifacts
from the British Museum. Expert evaluation demonstrates
that the system produces meaningful and interpretable out-
puts, offering scholars concrete starting points for analysis
and significantly alleviating the cognitive burden of navi-
gating vast comparative corpora.
1. Introduction
Within archaeology, art history, and museology, typological
analysis is a foundational methodology for understanding
ancient material culture [11]. Its epistemological basis par-
allels other scientific methods: the systematic comparison
of unknown objects against established reference materials
to extrapolate chronological and cultural information [10].
For example, scholars may date a Chinese Bronze Age rit-
ual vessel by identifying stylistic parallels with securely
dated artifacts, whether through inscriptional evidence or
stratified archaeological contexts [4].
While typological analysis draws on diverse theoretical
frameworks, from Social Darwinism [6] to processual ar-
chaeology [5], its core logic remains straightforward. How-
ever, the method faces increasing challenges in academic
and museum practice. It relies heavily on individual exper-
tise and tacit knowledge, introducing variability and rais-
*These authors contributed equally to this work.ing concerns about transparency and objectivity [2]. Re-
searchers also operate under uneven access conditions, from
reliance on published catalogs to privileged examination of
objects in storage, fundamentally shaping analytical out-
comes [3]. Further, each artifact presents a near-infinite
array of stylistic features, requiring selective prioritization
often guided by Bayesian-like reasoning shaped by disci-
plinary training [8].
To mitigate these methodological challenges, this study
proposes a retrieval-augmented generation (RAG)-based
system for provenance analysis, designed to systematically
replicate and enhance expert reasoning processes. In collab-
oration with domain specialists, we have selected a collec-
tion of bronze artifacts exhibiting Eastern Eurasian steppe
stylistic traditions from the Bronze Age and Early Iron Age
periods (ca. 1500–200 BCE) as our target dataset, which
are currently unpublished and housed within the early China
section of the British Museum. Evaluation results indicate
that the system can generate insightful and suggestive out-
puts that provide researchers with concrete starting points
for analysis, effectively narrowing the search space and al-
leviating the cognitive burdens of navigating heterogeneous
comparative materials.
2. System Design
As shown in the Figure 1, our system consists of three core
components:External Knowledge Construction,Multi-
modal Retrieval, andInference and Reasoning.
2.1. External Knowledge Construction
The external knowledge base is primarily composed of ar-
chaeological monographs, catalogs, and excavation reports.
Source material in PDF format are converted into structure
Markdown files to enable efficient text retrieval. However,
since many PDFs are low-quality scans rather than digitized
texts, text extraction and search can be unreliable. To ad-
dress this, we additionally extract and index figures, plates,
and visual layouts, constructing an image repository to en-
able complementary image-based querying.
1arXiv:2509.20769v1  [cs.IR]  25 Sep 2025

Prompt: When was it made? 
Where was it made? 
Step 1:  
Multimodal 
Retrieval 
Raw Image Retrieval 
Edge-Enhanced Retrieval 
Semantic CLIP-based Retrieval 
Step 2 : Candidate 
Aggregation and Filtering 
●Depulicate and rank 
●Select top-m Step 3:  Inference 
and Reasoning 
Vision Language Model Excavation Location Guess: 
Era Guess: 
Book Source: 
Page Number: 
Similarity: 
Description: Figure 1. The illustration of the proposed system. the target image is processed through (1) multimodal retrieval (raw, edge-enhanced, and
CLIP-based), (2) candidate aggregation and filtering, and (3) inference and reasoning with a vision-language model to predict provenance,
era, supporting references, and explanations.
2.2. Multimodal Retrieval
To fully leverage both the textual and visual modalities in
archaeological documents, we design a three-fold retrieval
module. These three retrieval strategies run in parallel and
serve as structural complements to each other.
Strategy 1: Raw Image Retrieval.The input query im-
age is directly compared against all reference images in the
image-only candidate database using cosine similarity. The
top-kvisually similar images are selected, and the corre-
sponding contextual paragraphs in their source documents
are retrieved as candidate reference knowledge.
Strategy 2: Edge-Enhanced Retrieval via Gaussian Fil-
tering.To address the frequent presence of line drawings
and archaeological sketches in the external database, we ap-
ply Gaussian-filter-based edge detection on query images
to enhance the recognition. The filtered image highlights
structural edges and contours, which are then used to query
the database via cosine similarity. The Gaussian filter is de-
fined as:
G(x, y) =1
2πσ2exp
−x2+y2
2σ2
Strategy 3: Semantic CLIP-based Retrieval.We embed
both text and images into a shared multimodal space us-
ing CLIP-based encoders [9]. Cosine similarity is used to
retrieve image-text pairs that are semantically aligned with
the query image, capturing matches even when visual ap-
pearance differs but descriptive content is similar.
2.3. Candidate Aggregation and Filtering
We merge the retrieval results of all three strategies into
unified candidate pools. Specifically, we define three
existence-based multisets:
MA={lraw
i},M B={ledge
j},M C={lclip
m}
We then perform deduplication and sorting on the union of
the three sets:
T=sort ≤(MA⊎ M B⊎ M C) = (t 1, t2, . . . , t n)Here,n=|M A⊎ M B⊎ M C|is the number of unique
elements, and the elements are ordered by dictionary or nu-
merical order. Finally, we truncate the list to retain only the
topmelements.
2.4. Inference and Reasoning
The filtered candidatesM⊆T topm are passed into the
VLM for structured reasoning and synthetis.
Phase 1: Per-Candidate Interpretation.The VLM an-
alyzes each candidate individually, integrating visual fea-
tures and textual context to infer key attributes: likely ex-
cavation site, estimated cultural period, similarity rationale,
and bibliographic reference (including page number). Out-
puts are structured as metadata summaries in JSON file.
This step significantly reduces the risk of hallucinations and
alleviates token length constraints [1].
Phase 2: Cross-Candidate Reasoning and Judgment.In
the second phase, the target image is fed into the model
along with all Phase 1 outputs. The model determines the
most likely excavation site and probable historical period
for the object, drawing from correlations across both vi-
sual similarity and textual evidence extracted from the ref-
erences. Furthermore, the VLM identifies the most relevant
reference source that supports its inference, specifying both
the document and the exact page number where a similar
artifact or description is found. Finally, the model provides
an interpretive justification for its prediction, highlighting
key visual and textual cues that underpin its reasoning.
3. Evaluation
3.1. Experimental Settings
Datasets, Models, and Tasks.We choose GPT-4o [7] as
the VLM in the system. The reference material comprises
eight exhibition catalogues and scholarly manuscripts.
These publications, produced in Chinese, English, Russian,
and French, constitute standard reference works routinely
employed in professional training programs and advanced
research contexts. Each catalogue provides systematic doc-
2

**Possible Excavation Locations and Corresponding Eras:** Based on 
the five most similar images, the cultural relic in question is 
identified as a bronze knife or artifact featuring animal-head or 
ring-shaped decorative motifs, closely related to the 
Ordos-style bronze knives and the Karasuk Culture 
bronzes.  The strongest match (similarity 0.9677) points to the 
Qinglong Chaodaogou site in Hebei Province, China, with a Late 
Shang dynasty attribution, emphasizing dragon-head and snake-body 
motifs characteristic of Ordos bronzes. Additional highly similar 
items are from the Minusinsk Museum collection, but without 
definite excavation information, suggesting a Eurasian Steppe 
origin, particularly the Minusinsk Basin or Khakass-Ust-Abakan 
region in Russia. Image 4 specifically links the artifact style to 
the Karasuk Culture (13th–11th century BCE), which is distributed 
across the Minusinsk Basin and adjacent areas, and is well-known 
for bronze knives with ring-shaped or animal-head handles. The 
overlap in decorative motifs and forms between Chinese Ordos 
bronzes and Karasuk knives suggests cross-cultural interaction or 
parallel development during the late Bronze Age. While the precise 
excavation site remains uncertain, the artifact most likely 
comes from either northern China (Hebei, Ordos region, 
Late Shang) or the southern Siberian steppe (Minusinsk 
Basin, Karasuk Culture, 13th–11th century BCE) , both 
areas known for similar metallurgical traditions. The available 
book sources further support these connections but lack exact page 
numbers. Overall, the artifact is a representative example of late 
Bronze Age steppe metallurgy, with possible origins in either the 
northern Chinese frontier or the Minusinsk Basin region. Target image 
 Top-5 Most Similar References 
AI-Generated Provenance Summary 
Figure 2. Example of system output for a Bronze Age bronze artifact. Left: input target image. Center: top-5 most similar reference
objects retrieved from the external database, showing stylistic correspondences to Ordos-style bronzes and Karasuk Culture knives. Right:
AI-generated provenance summary synthesizing visual matches and reference data to infer likely excavation regions (northern China or
southern Siberia), estimated chronology (Late Shang to 13th–11th century BCE), and cultural affiliations.
umentation through photographic plates and technical line
drawings, accompanied by detailed chronological assess-
ments and provenance data for individual object types.
Expert Evaluation Criteria.Five domain experts were
recruited to evaluate the AI-generated outputs for the first
thirty objects in the dataset. The evaluation framework is
structured around two primary research questions:Q1, as-
sessing the identification of stylistically similar reference
objects; andQ2, assessing the generation of chronological,
geographical, and archaeological cultural attributions. The
evaluation employed a structured four-level scoring scheme,
which Score 4 represents highly meaningful and Score 1
suggests not meaningful.
3.2. Evaluation Results
Figure 2 provides a demonstration of system output for a
target image. As Figure 3 shows, experts reckon that ap-
proximately 63% of the retrieved images achieved mean-
ingful results (Scores 2-4). However, the proportions of im-
ages receiving Score 3 (17.7%) and Score 4 (14.9%) remain
relatively low compared to Score 2 (30.6%).
The performance differential between Q1 and Q2 out-
comes presents a particularly noteworthy finding. While vi-
sual similarity identification exhibited the aforementioned
limitations, the system’s capacity for generating chronolog-
ical and geographical conclusions demonstrated markedly
superior performance. The proportion of outcomes receiv-
ing the lowest rating decreased to approximately 10% for
Q2 evaluations, while nearly 46% of the AI-generated attri-
butions achieved Score 3 or higher. This disparity indicates
the algorithm excels at synthesizing typological informa-
tion into scholarly conclusions, suggesting greater utility for
supporting expert decisions than replacing traditional com-
parative analysis.
Figure 3. Expert evaluation score distributions for Q1 and Q2.
4. Conclusion
We present a RAG-based system to assist provenance analy-
sis of archaeological artifacts. Expert evaluation shows that
while visual retrieval yields mixed performance, the system
achieves notably stronger results in generating chronologi-
cal and cultural attributions, providing researchers with in-
terpretable and actionable outputs. Future work will explore
integrating expert evaluation policies directly into the sys-
tem to further enhance reliability and domain alignment.
3

References
[1] Catarina G Bel ´em, Pouya Pezeshkpour, Hayate Iso, Seiji
Maekawa, Nikita Bhutani, and Estevam Hruschka. From
single to multi: How LLMs hallucinate in multi-document
summarization. InFindings of the Association for Compu-
tational Linguistics: NAACL 2025, pages 5276–5309, Albu-
querque, New Mexico, 2025. Association for Computational
Linguistics. 2
[2] Adrian Currie. Speculation made material: experimental ar-
chaeology and maker’s knowledge.Philosophy of Science,
89(2):337–359, 2022. 1
[3] Christopher D Dean and Jeffrey R Thompson. Museum ‘dark
data’show variable impacts on deep-time biogeographic and
evolutionary history.Proceedings B, 292(2041):20242481,
2025. 1
[4] Zhe Luo, Ruiliang Liu, AM Pollard, Zhengyao Jin, Li Liu,
Yan Gu, Yuan Xu, Ruitong Guo, Fang Huang, and Anchuan
Fan. High-precision chronology and scientific analysis of
panchi mirrors reveal the state policy impact in early impe-
rial china.Journal of Archaeological Science, 180:106310,
2025. 1
[5] R Lee Lyman and Michael J O’Brien. A history of normative
theory in americanist archaeology.Journal of Archaeologi-
cal Method and Theory, 11(4):369–396, 2004. 1
[6] Michael J O’Brien and R Lee Lyman. Darwinian evolu-
tionism is applicable to historical archaeology.International
Journal of Historical Archaeology, 4(1):71–112, 2000. 1
[7] OpenAI. Gpt-4o system card.ArXiv, abs/2410.21276, 2024.
2
[8] Erik R Ot ´arola-Castillo, Melissa G Torquato, Jesse Wolfha-
gen, Matthew E Hill Jr, and Caitlin E Buck. Beyond chronol-
ogy, using bayesian inference to evaluate hypotheses in ar-
chaeology.Advances in Archaeological Practice, 10(4):397–
413, 2022. 1
[9] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. InInternational
Conference on Machine Learning, 2021. 2
[10] Michael E Smith and Peter Peregrine. Approaches to com-
parative analysis in archaeology.The comparative archaeol-
ogy of complex societies, pages 4–20, 2012. 1
[11] Marie Louise Stig S ¨orensen. Material culture and typology.
Current Swedish Archaeology, 5(1):179–192, 1997. 1
4
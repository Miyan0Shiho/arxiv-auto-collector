# Large Language Model Driven Analysis of General Coordinates Network (GCN) Circulars

**Authors**: Vidushi Sharma, Ronit Agarwala, Judith L. Racusin, Leo P. Singer, Tyler Barna, Eric Burns, Michael W. Coughlin, Dakota Dutko, Courey Elliott, Rahul Gupta, Ashish Mahabal, Nikhil Mukund

**Published**: 2025-11-18 19:13:43

**PDF URL**: [https://arxiv.org/pdf/2511.14858v1](https://arxiv.org/pdf/2511.14858v1)

## Abstract
The General Coordinates Network (GCN) is NASA's time-domain and multi-messenger alert system. GCN distributes two data products - automated ``Notices,'' and human-generated ``Circulars,'' that report the observations of high-energy and multi-messenger astronomical transients. The flexible and non-structured format of GCN Circulars, comprising of more than 40500 Circulars accumulated over three decades, makes it challenging to manually extract observational information, such as redshift or observed wavebands. In this work, we employ large language models (LLMs) to facilitate the automated parsing of transient reports. We develop a neural topic modeling pipeline with open-source tools for the automatic clustering and summarization of astrophysical topics in the Circulars database. Using neural topic modeling and contrastive fine-tuning, we classify Circulars based on their observation wavebands and messengers. Additionally, we separate gravitational wave (GW) event clusters and their electromagnetic (EM) counterparts from the Circulars database. Finally, using the open-source Mistral model, we implement a system to automatically extract gamma-ray burst (GRB) redshift information from the Circulars archive, without the need for any training. Evaluation against the manually curated Neil Gehrels Swift Observatory GRB table shows that our simple system, with the help of prompt-tuning, output parsing, and retrieval augmented generation (RAG), can achieve an accuracy of 97.2 % for redshift-containing Circulars. Our neural search enhanced RAG pipeline accurately retrieved 96.8 % of redshift circulars from the manually curated database. Our study demonstrates the potential of LLMs, to automate and enhance astronomical text mining, and provides a foundation work for future advances in transient alert analysis.

## Full Text


<!-- PDF content starts -->

Draft version November 20, 2025
Typeset using L ATEXdefault style in AASTeX7.0.1
Large Language Model Driven Analysis of General Coordinates Network (GCN) Circulars
Vidushi Sharma*
,1, 2Ronit Agarwala*
,1, 3Judith L. Racusin
 ,1Leo P. Singer
 ,1, 4Tyler Barna
 ,5
Eric Burns
 ,6Michael W. Coughlin
 ,5Dakota Dutko,1, 7Courey Elliott
 ,6Rahul Gupta
 ,1, 8
Ashish Mahabal,9and Nikhil Mukund10, 11
1Astrophysics Science Division, NASA Goddard Space Flight Center, Greenbelt, MD 20771, USA
2Center for Space Science and Technology, University of Maryland Baltimore County, Baltimore, MD 21250, USA
3University of California San Diego, La Jolla, CA 92093, USA
4Joint Space-Science Institute, University of Maryland, College Park, MD 20742, USA
5School of Physics and Astronomy, University of Minnesota, Minneapolis, MN 55455, USA
6Department of Physics & Astronomy, Louisiana State University, Baton Rouge, LA, USA
7ADNET Systems Inc.
8NASA Postdoctoral Program Fellow
9Center for Data-Driven Discovery, California Institute of Technology, Pasadena, CA, 91125
10MIT Kavli Institute for Astrophysics and Space Research and LIGO Laboratory,
Massachusetts Institute of Technology, Cambridge, MA 02139, USA
11NSF AI Institute for Artificial Intelligence and Fundamental Interactions (IAIFI), Cambridge, MA, USA
ABSTRACT
The General Coordinates Network (GCN) is NASA’s time-domain and multi-messenger alert system.
GCN distributes two data products - automated “Notices,” and human-generated “Circulars,” that
report the observations of high-energy and multi-messenger astronomical transients. The flexible and
non-structured format of GCN Circulars, comprising of more than 40500 Circulars accumulated over
three decades, makes it challenging to manually extract observational information, such as redshift
or observed wavebands. In this work, we employ large language models (LLMs) to facilitate the
automated parsing of transient reports. We develop a neural topic modeling pipeline with open-
source tools for the automatic clustering and summarization of astrophysical topics in the Circulars
database. Using neural topic modeling and contrastive fine-tuning, we classify Circulars based on their
observation wavebands and messengers. Additionally, we separate gravitational wave (GW) event
clusters and their electromagnetic (EM) counterparts from the Circulars database. Finally, using
the open-source Mistral model, we implement a system to automatically extract gamma-ray burst
(GRB) redshift information from the Circulars archive, without the need for any training. Evaluation
against the manually curated Neil Gehrels Swift Observatory GRB table shows that our simple system,
with the help of prompt-tuning, output parsing, and retrieval augmented generation (RAG), can
achieve an accuracy of 97.2% for redshift-containing Circulars. Our neural search enhanced RAG
pipeline accurately retrieved 96.8% of redshift circulars from the manually curated database. Our
study demonstrates the potential of LLMs, to automate and enhance astronomical text mining, and
provides a foundation work for future advances in transient alert analysis.
Keywords: Time domain astronomy (2109) — Transient sources (1851) — High Energy astrophysics
(739) — Gravitational waves (678) – Gamma-ray bursts (629) — Classification (1907) —
Neural networks (1933)
1.INTRODUCTION
Time-domain and multi-messenger astronomy has advanced significantly in the past decade, particularly with the
detection of gravitational waves (GWs) (B. P. Abbott et al. 2017, 2018). This breakthrough has led to an increase
∗Corresponding authors: vidushi.sharma@nasa.gov, ragarwala@ucsd.eduarXiv:2511.14858v1  [astro-ph.HE]  18 Nov 2025

2
in observational reports, with the astronomy community showing magnified interest in following up on high-energy
transients, high-energy neutrinos, and GWs. Additionally, new instruments dedicated to the study of the most ener-
getic phenomena in the Universe (e.g., supernovae, GRBs, and other bursts of galactic and extragalactic nature) are
contributing to GCN. To keep pace with technological advancements and community needs, the New GCN, named as
General Coordinates Network was launched in July 2022 (Racusin et al., paper in prep). With upgraded technology
and increased interest in multi-messenger astronomy, the GCN is a reliable and secure platform. There will be a
huge amount of data in the future that will require automated methods for analysis. Even new transients detected
with the Space Variable Objects Monitor (SVOM) (J. Paul et al. 2011) and Einstein Probe (EP) (W. Yuan et al.
2025) have reported an increase in GRBs and new X-ray transients, driving the rise in follow-up observations from
the community. Moreover, observatories reporting fast radio bursts (FRBs), such as the Canadian Hydrogen Intensity
Mapping Experiment (CHIME, ( CHIME/FRB Collaboration et al. 2018)) and Deep Synoptic Array-110 (DSA-110,
(C. J. Law et al. 2024)), will join the network soon. These efforts will be complemented by significant advancements
and commissioning of new observatories designed to detect transients, such as high-energy neutrinos (e.g., IceCube
(M. G. Aartsen et al. 2017), KM3NeT ( KM3NeT Collaboration et al. 2024)), GWs (e.g., Virgo, LIGO, KAGRA,
LISA), radio (e.g., Square Kilometre Array (P. E. Dewdney et al. 2009)), optical observations (e.g., Large Synoptic
Survey Telescope, (Ž. Ivezić et al. 2019)) and high energy gamma-rays (e.g., Cherenkov Telescope Array (CTA), (
Cherenkov Telescope Array Consortium et al. 2019)). Therefore, a wealth of information will be available in both
multi-wavelength and multi-messenger astrophysics, with increasing interest in following up on exciting events.
GCN is a public collaboration platform managed by the National Aeronautics and Space Administration (NASA)
for the astronomy research community. It enables the sharing of real-time alerts and rapid communications regarding
high-energy, multi-messenger, and transient phenomena. GCN distributes alerts for both space and ground-based
observatories, physics experiments, and thousands of astronomers worldwide. Established in 1992, GCN began as an
innovative solution for near-time notification of gamma-ray bursts (GRBs) detected by the Burst and Transient Source
Experiment (BATSE) onboard the Compton Gamma Ray Observatory (CGRO) (S. D. Barthelmy et al. 1994, 1995).
In 1997, as new instruments and missions were added to the network, it was renamed as Gamma-ray burst Coordinates
Network. GCN has enabled global follow-up observations that have significantly improved our understanding of GRBs,
including their afterglows, redshifts (e.g. first redshift for GRB 970508 (D. E. Reichart 1998)), supernova detections
(e.g. SN1998bw/GRB 980425 (T. J. Galama et al. 1998)), and host-galaxy information. Over time, the network
expanded beyond the GRB community to include other high-energy astronomical transients, such as high-energy
neutrino (e.g. IceCube-170922A (C. Kopper & E. Blaufuss 2017)), soft gamma-ray repeaters (SGRs, e.g. GRB
200415A (A. Pozanenko et al. 2020)) and tidal disruption events (TDEs, e.g. Swift J1644+57 (D. N. Burrows et al.
2013)).
GCN provides two types of alerts to support research in astrophysical transient phenomena: GCN Notices and GCN
Circulars. GCN Notices are brief, machine-generated reports12. GCN Circulars13, on the other hand, are more
detailed, human-written observational report that often contain information about observations, predictions, requests
for follow-ups, and plans for future observations for multi-messenger astrophysical events. GCN Circulars (hereafter,
Circulars) have a flexible and unstructured format. Over the past 30 years, more than 40500 Circulars have been
submitted, making the Circulars database a valuable resource for astrophysicists studying multi-messenger events.
However, this volume presents a challenge in managing the growing volume of astrophysical data and alerts to extract
useful information, given the varying structures of Circulars.
GCN plays a central role as a global distribution system for alerts from high-energy and multi-messenger observato-
ries. Other brokers like Fink (A. Möller et al. 2021) or ALerCE (F. Förster et al. 2021) ingest alerts from time-domain
surveys and provide value-added annotations, for example through cross-matching with external catalogs. While bro-
kersenrichsurveydata, GCNfacilitatestherapiddisseminationofalertstothecommunity, andprovidesastandardized
framework for real-time and archival access. This infrastructure enables swift coordination among observers and follow-
up facilities, complementing broker-based alert streams and linking them to the broader multi-messenger landscape.
Transient objects identified by brokers are often shared with the community via the Transient Name Server14(TNS),
the Minor Planet Center15(MPC), Astrophysical Multimessenger Observatory Network alerts (AMON) (H. A. Ayala
12https://gcn.nasa.gov/notices
13https://gcn.nasa.gov/Circulars
14https://www.wis-tns.org/
15https://minorplanetcenter.net/

3
Solares et al. 2020) or GCN, depending on the nature of the source. Coordinated follow-up efforts are then facilitated
by Target and Observation Managers (TOM), with notable examples including Young Supernova Experiment( YSE-PZ,
D. A. Coulter et al. (2023)), the TOM Toolkit (R. A. Street et al. 2018), SkyPortal (S. J. van der Walt et al. 2019;
M. W. Coughlin et al. 2023) and Astro-COLIBRI (P. Reichherzer et al. 2021).
Recently, there has been a growing interest in integrating natural language processing (NLP) tools into astrophysical
research. For instance, F. Grezes et al. (2021) highlights the potential of astroBERT, a large language model (LLM)
based on the Bidirectional Encoder Representations from Transformers (BERT) deep neural network architecture
(J. Devlin et al. 2018). Trained on on 15 million records from NASA ADS, astroBERT demonstrates capabilities
in recognizing named entities within astrophysical texts. N. Mukund et al. (2018) shows the application of NLP
to retrieve and display relevant logbook data, aiding in detector operations and upgrades, presented for LIGO and
Virgo observatories. Additionally, T. Dung Nguyen et al. (2023) fine-tuned the pre-trained generative language model
Llama-2 (H. Touvron et al. 2023) using 300000 astronomy abstracts, demonstrating the model’s ability to understand
astronomical concepts through text completion. Further exploring the use of LLMs for astronomical information, V.
Sotnikov & A. Chaikova (2023) presents the way models like InstructGPT-3 (L. Ouyang et al. 2022) and Flan-T5-XXL
(H. W. Chung et al. 2022) can automate the extraction of event IDs, event types, and astrophysical phenomena from
The Astronomer’s Telegram and Circulars.
In this study, we aim to explore the role that LLM-based NLP methods can play in modernizing the GCN. We aim to
automate extracting relevant information about transient events from the Circulars archive. To achieve this, Section 2
focuses on a neural topic modeling pipeline for the GCN Circulars database with the help of the library BERTopic (M.
Grootendorst 2022). This pipeline automatically identifies common astrophysical themes in the Circulars archive and
summarizes them with the help of an open-source LLM. We further enhance this pipeline by fine-tuning a BERT-based
modeltoclassifyCircularsbasedonthetypeofreportedobservation. Section3describesthedesignandimplementation
of an open-source, end-to-end, zero-shot system designed for the automated extraction of GRB redshift information.
The zero-shot approach enables the model to perform effectively without any prior examples. This system enables us
to extract redshift values or ranges, GRB numbers, names of telescopes or observatories, and redshift types from the
GCN Circular database. Additionally, we conducted a preliminary quantitative analysis of the data obtained. Our
approach utilizes open-source models that require no training, allowing it to be easily adapted to extract various types
of astrophysical information by simply fine-tuning the prompts. We have also implemented a simple retrieval-based
system to filter out Circulars that lack the relevant information, which significantly improves the accuracy and reduces
computational costs.
2.TOPIC MODELING
2.1.Methods
2.1.1.Neural Topic modeling with BERTopic
In the field of NLP, topic modeling is a statistical text-mining technique used to uncover latent themes present within
large text corpora. This method enables the identification of semantically meaningful clusters, known as topics, in
extensive text datasets (H. Zhao et al. 2021). Over the years, various probabilistic topic modeling techniques based on
Bayes’ theorem, such as Latent Dirichlet Allocation (D. M. Blei et al. 2003) and Probabilistic Latent Semantic Analysis
(T. Hofmann 2013), have been successfully employed. More recently, deep neural network architectures, particularly
thetransformermodel(A.Vaswanietal.2017), havebeenappliedtocreatemoresemanticallymeaningfultopicclusters
(S. Sia et al. 2020; M. Grootendorst 2022). These algorithms utilize word embeddings, which are multi-dimensional
vector representations of words. Pre-trained LLMs, such as BERT (J. Devlin et al. 2018), are used to create these
word vectors in such a way that semantically similar words are positioned close to each other within the vector space.
Entire sentences and even documents can be encoded by averaging over all the word embeddings in that document (N.
Reimers & I. Gurevych 2019). The document embeddings can then be clustered together using clustering algorithms
like k-means or HDBSCAN (L. McInnes & J. Healy 2017) into groups of semantically related documents, or topics.
In this work, we employ topic modeling library BERTopic (M. Grootendorst 2022) to build a neural topic model for
the Circulars archive. A standard BERTopic pipeline involves first converting a collection of documents into numerical
vectors, usually achieved with a SentenceTransformers model (N. Reimers & I. Gurevych 2019) based on the BERT
architecture. These models encode semantic and contextual information from each document into high-dimensional
document vectors, also known as embeddings. The high dimensionality of these vectors, however, makes it challenging
to accurately cluster them using conventional clustering algorithms due to higher sparsity in the data, sometimes

4
called the “curse of dimensionality” (C. C. Aggarwal et al. 2001; K. Beyer et al. 1999). Thus, a dimensionality
reduction algorithm is applied to the document vectors. Common algorithms used for this purpose are Principal
Component Analysis (PCA) (H. Abdi & L. J. Williams 2010) and Uniform Manifold Approximation (UMAP) (L.
McInnes et al. 2018). Finally, our reduced embeddings are grouped into coherent topic clusters using an appropriate
clustering algorithm. The default algorithm used for this purpose is Hierarchical Density-Based Spatial Clustering
of Applications with Noise or HDBSCAN, a robust clustering algorithm that can discover hierarchical clusters of
arbitrary shapes and sizes within high-dimensional data by grouping neighboring data points with sufficiently high
densities (R. J. G. B. Campello et al. 2013; L. McInnes & J. Healy 2017).
Once the topic clusters have been extracted from the dataset, these topics can be represented in multiple ways.
By default, BERTopic characterizes topics through the most significant words present in each cluster. To determine
these key terms, a variation of the Term Frequency-Inverse Document Frequency (TF-IDF), formula is employed
(T. Joachims 1997). Called c-TF-IDF , where c stands for class-based, identifies the most informative keywords in
each cluster by concatenating all documents within them and treating them as a single text (M. Grootendorst 2022).
Additionally, we have the option of removing unimportant words being considered as keywords by implementing a
custom stopwords list. This is a list of uninformative words that do not contribute to the interpretability of our
extracted topics and should thus be removed from our representations. An example of common stopwords we chose to
remove from our astrophysical topic representations includes articles like “a”, “an”, and “the”, as well as author names,
website URLs, and emails.
2.1.2.Unsupervised GCN Topic modeling Pipeline
The GCN platform provides access to its entire database of Circulars as a downloadable archive on its website16.
Circular data is available in both text and JSON(JavaScript Object Notation) formats. The advantage of using JSON
format is that it organizes data into key-value pairs and can be automatically parsed by machines with minimal manual
intervention. For our study, we downloaded the JSONtarball file and extracted the Circular bodies and subject headers
from the ’body’ and ’subject’ JSONfields, respectively. We also extracted the date on which each Circular is published
from the ’createdOn’ fields for later trend analysis. Circular data was collected for every GCN Circular published
between March 1998 and May 2025. For each Circular, we created a single text by concatenating the subject and
body. This list of each circular text was then processed by removing any undefined characters or symbols to facilitate
easier processing by our pipeline. Some preliminary word statistics were collected and a list of the most common
uninformative stopwords was manually curated. The list includes some commonly occurring words in Circulars, such
as “acknowledgment” and “scheduled,” as well as names of authors and organizations that do not provide relevant
information on astrophysical topics of interest. Thus, we curated a custom stopwords of 1,587 terms. Additionally, a
general list of website URLs and numbers were also extracted from the circular texts using regular expressions and
string pattern matching, respectively. Some emails were collected from the ‘email’ fields of the Circular JSONfiles as
well. These entities were added to our manually curated list, which includes a pre-defined list of stopwords from the
Python NLTKlibrary (S. Bird et al. 2009) and a punctuation list from Python’s string module.
Next, each Circular text was then encoded into a vector representation with an embedding model from the
SentenceTransformers library (N. Reimers & I. Gurevych 2019). We used the default all-MiniLM-L6-v2 model,
which is a small but highly effective embedding model shown to have comparable performance to much larger models
for neural topic modeling (M. Grootendorst 2022). Other embedding models were also tested, however we found that
the quality of topics generated, evaluated through manual assessment, showed negligible change to justify the higher
computational costs for larger embedding models.
Finally, our Circular texts, custom stopwords list, and circular embeddings were fed into our BERTopic pipeline for
dimensionality reduction, clustering, and keyword extraction. We used the default UMAP and HDBSCAN algorithms
for dimensionality reduction and clustering respectively. Preliminary tests with PCA for dimensionality reduction
were performed, however they resulted in a significant drop in topic quality. The default HDBSCAN algorithm was
selected for clustering because it does not require the user to specify the number of clusters in advance. Additionally,
it can identify outliers in the dataset that do not fit into any topic. This feature makes it a preferred choice for topic
discovery over the GCN database, where we expected to find various errata and miscellaneous Circulars that do not
16https://gcn.nasa.gov/circulars

5
belong to any specific astrophysical topic. Topic keywords were extracted with c-TF-IDF , where our custom stopwords
list was used to filter out uninformative keywords.
To generate the embeddings for clustering, we employed UMAP for dimensionality reduction. The model was config-
ured with n_neighbors as 30 to get balanced local and global structure, enables the separation of broad astrophysical
categories such as gravitational-wave events, gamma-ray bursts, neutrino detections, and multi-messenger follow-ups
(for scientifically balanced topic recommended range is 25–50). The embeddings were reduced to n_components as
5 dimensions to capture sufficient variance, and min_dist as 0.0 for dense packing of points, which enhances cluster
distinction. Cosine distance (metric = ‘cosine’) was used to preserve semantic similarity between textual embeddings,
and the random seed was fixed ( random_state = 42) to ensure reproducibility across runs.
Following dimensionality reduction, clusters were extracted using HDBSCAN, with parameters tuned for coarse,
interpretable topic modeling. HDBSCAN offers two such parameters that can significantly affect topic modeling
performance: min_cluster_size andmin_samples . The min_cluster_size parameter determines the minimum size
of the smallest topic cluster. Increasing this value results in the merging of smaller clusters and consequently, reducing
the overall number of topics. Ideally, we want to maintain a manageable number of topics for further analysis while
avoiding the formation of small micro-clusters with only a few documents. Through preliminary analysis, we found
that setting this hyperparameter to 800 consistently yielded between 10 and 20 topics from our dataset. Whereas,
themin_samples hyperparameter controls the degree to which certain data points are classified as noise. We prefer a
lower value for this parameter to ensure that as many Circulars as possible are clustered together. After conducting
initial tests, we decided to set this hyperparameter to 10, which minimized the number of Circulars declared as outliers.
2.1.3.Topic Summarization with Mistral 7B Instruct
To enhance the interpretability of astrophysical topics, more informative topic representations are built using gener-
ative LLMs. By feeding in a few sample documents from each topic, the topic keywords obtained from c-TF-IDF and a
carefully designed prompt template for an LLM, we guide the AI to provide brief, informative summaries of the under-
lying scientific descriptions. This method can help us build much more informative topic labels for each topic cluster in
contrast to standard methods like c-TF-IDF alone. BERTopic also provides support for integrating LLMs in this man-
ner into our pipeline. For topic summary generation we selected the open-source Mistral 7B Instruct model (A. Q.
Jiang et al. 2023). This 7.3 billion parameter pre-trained generative model has been fine-tuned on instruction datasets
from Hugging face17. It has shown high performance on several text generation benchmarks, exhibiting excellent
performance in comprehension and reasoning benchmarks. Given the hardware constraints of running our model on a
single Google Colab GPU environment, we used an even smaller version of this model in which the weights of the model
were “quantized” or reduced to 4-bits. This version of the model, called mistral-7b-instruct-v0.2.Q4_K_M.gguf
(T. Jobbins 2023), was downloaded from the Hugging face library18and run in Colab with the help of the open-source
project llama.cpp (G. Gerganov 2023). The prompt template that we used for summary generation is presented in
Figure 1.
Figure 1. Prompt template used for topic summary generation. For every topic cluster [DOCUMENTS] gets replaced by 3
concatenated sample Circulars from each topic, while [KEYWORDS] gets replaced by the representative keywords extracted
using c-TF-IDF . Prompt template modified from default template provided in BERTopic documentation.
17https://huggingface.co/
18https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main

6
2.1.4.Supervised Observation-Based and Gravitational Wave-Based Classification
Our standard topic modeling pipeline allows us to automatically identify themes present in the Circulars database.
Next, we attempted to cluster our Circulars based on pre-defined astrophysical categories of our own choosing. More
specifically, we categorized Circulars into five broad categories depending upon the type of observation being reported
in them. Our five categories were:
•High-energy observations: This includes X-ray observations as well as gamma-ray observations across the MeV,
GeV, and TeV energy ranges. Telescopes in this category include Fermi Gamma-ray Burst Monitor (Fermi-GBM)
(C. Meegan et al. 2009), Fermi Large Area Telescope (Fermi-LAT) (W. B. Atwood et al. 2009), Swift Burst Alert
Telescope (Swift-BAT) (N. Gehrels et al. 2004), Swift X-ray Telescope (Swift/XRT), Nuclear Spectroscopic
Telescope Array (NuSTAR) (F. A. Harrison et al. 2013), High Energy Stereoscopic System (H.E.S.S.) (J. Hinton
2004), High-Altitude Water Cherenkov (HAWC) Gamma-Ray Observatory (R. Springer 2016) and many more.
•Optical observations: This includes Circulars with reported near-infrared (NIR) and ultraviolet (UV) detections,
spectroscopic and photometric redshifts, as well as optical magnitudes and upper limits. Optical telescopes
include Hubble Space Telescope (HST) (H. C. Ford et al. 1998), James Webb Space Telescope (JWST) (J. P.
Gardner et al. 2006), Zwicky Transient Facility (ZTF) (E. C. Bellm et al. 2018), Nordic Optical Telescope (NOT)
(J. Vernin & C. Muñoz-Tuñón 1992), W. M. Keck Observatory (Keck) (J. E. Nelson et al. 1985), and Very Large
Telescope (VLT) (J. Vernet et al. 2011), etc.
•Radio observations: This includes detections of radio sources by observatories and telescopes such as the Very
Large Array (VLA) (P. J. Napier et al. 1983), Arcminute Microkelvin Imager (AMI) (J. Zwart et al. 2008), Aus-
tralia Telescope Compact Array (ATCA) (W. E. Wilson et al. 2011), Atacama Large Millimeter/Submillimeter
Array (ALMA) (A. Wootten & A. R. Thompson 2009), and Giant Metrewave Radio Telescope (GMRT) (G.
Swarup et al. 1991), etc., with flux densities often measured in Jansky and milli-Jansky units.
•GW observations: This includes data from laser interferometers such as the Laser Interferometer Gravitational-
Wave Observatory (LIGO), Virgo Interferometer (Virgo), and Kamioka Gravitational Wave Detector (KAGRA)
(B. P. Abbott et al. 2018).
•Neutrino observations: Observations reported by neutrino detectors such as IceCube (M. G. Aartsen et al. 2017),
ANTARES neutrino telescope (M. Ageron et al. 2011) are classified here.
Categorizing Circulars based on the type of observation is challenging with conventional methods due to the lack
of suitable keywords with which to sort them. For example, a search for the word ’radio’ in the GCN Circular
database will often turn up many Circulars from the GRBAlpha CubeSat, even though they have nothing to do
with radio observations, as they all end with the statement - “...The ground segment is also supported by the radio
amateur community and it takes advantage of the SatNOGS network for increased data downlink volume”, referring to
radio communications for their CubeSat. Transformer-based language models can be used here to take into account
contextual information and thus improve the accuracy of simple keyword-based sorting methods. We demonstrate this
capability by modifying our topic modeling pipeline to perform a type of neural search over the GCN database with the
help of BERTopic’s zero-shot topic modeling technique19. A list of 5 candidate labels - “High Energy Observations”,
“Optical Observations”, “Radio Observations”, “Neutrinos”, and “Gravitational Wave” was passed into our pipeline and
embedded with the help of the all-MiniLM-L6-v2 embedding model. The embedding vectors of these labels were
then compared to the embeddings of every Circular in our database using the cosine similarity metric. Each Circular
was thus assigned to the label with the highest similarity score.
Similarly, we conducted a comprehensive study on the classification of a specific type of GCN Circulars, focusing on
classifying GW events, their EM or neutrinos counterparts, and non-GW related astrophysical events. Such classifi-
cation would be useful for the community to filter the heterogeneous stream of Circulars containing diverse events,
thus, facilitating rapid follow-up of targeted searches for specific classes of astrophysical phenomena. In this study, we
focus specifically on GW alerts, using the candidate labels "Gravitational Wave", "Gravitational Wave Counterpart",
and "Non Gravitational Wave" to classify these Circulars.
19https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html

7
To improve the accuracy of these methods on observation-based classification and gravitational wave-based classifica-
tion, we opted to fine-tune our embedding model on a manually labeled dataset.
2.1.5.Contrastive Fine-Tuning
The performance of the observation-based clustering pipeline relies heavily on the ability of the all-MiniLM-L6-v2
embedding model, a sentence-transformer model beneficial for tasks like clustering or identifying sentence similarity.
It is utilized to group Circulars that report similar types of observations, while keeping the dissimilar observation
types away from each other in the vector space. However, this task is challenging as the model has not been explicitly
trained on astrophysical data before. For instance, it may find difficulty in differentiating between a high-energy
neutrino detection and a high-energy gamma-ray observation, as they both contain the word ‘high-energy’ in them. To
address this issue and improve our pipeline’s performance in transient alert classification, we fine-tuned our embedding
model using a technique called supervised contrastive learning (B. Gunel et al. 2020). This approach employs a
similarity-based loss function and utilizes a labeled dataset, allowing a pre-trained language model to learn how to
embed texts belonging to the same class closer to one another in the vector space, while distancing examples from
different classes. This method has been shown to significantly improve the robustness of embedding models when
applied to other domains (B. Gunel et al. 2020; X. Ma et al. 2021).
Figure 2. Visualization of the GCN topic modeling pipelines, built with the help of the BERTopic library. The blue pipeline
depicts the steps for the unsupervised discovery of astrophysical topics. Circulars are embedded with all-MiniLM-L6-v2 , after
which the vectors are reduced and clustered. Keywords are extracted after stopwords removal and the discovered topics are
summarized with Mistral 7B Instruct . The red pipeline describes the supervised process for generating observation-based
clusters. The all-MiniLM-L6-v2 model is fine-tuned on a labeled dataset of Circulars. The Circulars and a list of observation
labels are then embedded using this model. Finally, both sets of embeddings are compared with each other using the cosine
similarity metric to find the closest observation label for each Circular in the embedding space.
Observation-based Clustering: We fine-tuned the embedding model on a manually labeled dataset of 200 Cir-
culars to classify the astrophysical observations, each Circular was labeled based on the type of observation reported:
optical (e.g, telescopes such as ZTF, Pan-STARRS etc), radio (e.g., observations from ALMA or VLA), high-energy
(e.g., gamma-ray detections by Fermi-LAT or Swift), GW (e.g., LIGO/Virgo/KAGRA events), and neutrino obser-
vations (e.g., IceCube detections). It was implemented using the Sentence Transformers library that specializes in
optimizing text embeddings through contrastive fine-tuning, which offers support for various types of loss functions.
Here we used the distance-based Contrastive Loss function (R. Hadsell et al. 2006), which optimizes embeddings by

8
minimizing the squared Euclidean distance between samples of the same class while maximizing it for dissimilar pairs.
The fine-tuned all-MiniLM-L6-v2 models were then used to create new embeddings for Circulars. These embed-
dings were fed into our observation-based clustering pipeline, which groups Circulars by similarity along with the
observational labels. We found our classification accuracy improved significantly upon using this fine-tuning approach.
Gravitational Wave Classification: We fine-tuned the all-MiniLM-L6-v2 embedding model using a manually
labeled dataset consisting of a representative subset of 300 Circulars. This dataset includes a balanced set of 100
Circulars reported by LIGO, Virgo, and KAGRA; 100 follow-up observations of GW counterparts reported in GCN;
and 100 Circulars unrelated to GW detections. The fine-tuning process was again implemented using the Sentence
Transformers library and Contrastive Loss function. This approach significantly improved the model’s ability to
distinguish between GW-related and non-GW observations, achieving high classification accuracy.
Figure 2 provides the flow-chart for both, unsupervised and supervised, in blue shaded region and red-shaded region
respectively, for our topic modeling pipeline.
2.2.Results
The results of applying LLMs to GCN Circulars are systematically stored and presented in the NASA-GCN GitHub
repository20atnasa-gcn , serving as an open resource for reproducibility and access to the products generated in
this work. The repository contains all data, analysis pipelines, visualizations, and extended tabular results. The
repository is organized into three primary folders: data, scripts, and extended tables. The data directory contains
archived Circulars in JSON format (up to May 2025) and a custom stop-word list used for pre-processing. The scripts
directoryprovidesGoogleColab notebooksfor topicmodeling. The extendedtables directorycontains tabularoutputs,
including topic-modeling cluster tables for gravitational-wave–focused topics, observational clusters, and unsupervised
topics summary.
2.2.1.GCN Topic Clusters and Summaries
The pipeline consists of three stages: embedding generation, topic modeling, and summarization. Embedding all
Circulars with all-MiniLM-L6-v2 took approximately a minute, demonstrating its suitability for large-scale semantic
representation studies. Topic modeling by itself, which includes the dimensionality reduction, clustering, and key-
word extraction phases, was completed in a few minutes. Finally, topic summaries were generated with Mistral 7B
Instruct in about a minute as well.
Using our topic modeling pipeline with our chosen hyperparameters, we get 24 unique topics from the GCN Circular
database, not counting the documents labeled as outliers. After clustering our Circulars, we visualize our topic
clusters in 2D after reduction of our document embeddings with t-SNE, as seen in Figure 3. Note that topics here are
represented using their keywords extracted with c-TF-IDF . Finally, 13 topic summaries generated by Mistral, along
with their Circular counts, can be viewed in Table 1.
2.2.2.Fine-Tuning Setup, Observation-Based Classification, and Evaluation
To generate observation-based clusters from Circulars, we employed a contrastive learning approach to fine-tune
theall-MiniLM-L6-v2 sentence embedding model. As mentioned previously, the training dataset consisted of 200
manually categorized Circulars, evenly distributed across five observation types: high-energy, optical, radio, neutrino,
and GW detections, with 40 Circulars for each category. The dataset was divided into training and test sets using an
80-20 split, resulting in 160 training samples and 40 test samples. In the contrastive fine-tuning set up, each Circular
in our training set was paired with every other Circular, leading to a 25600 text pairs. We also include category labels
- “Optical Observations”, “High Energy Observations”, “Radio Observations”, “Neutrinos”, and “Gravitational Wave”
for each observation type in the training set. This approach allowed the model to learn to represent both the text
of the Circulars and their corresponding labels together within the same latent vector space. We thus had a total of
(32High Energy + 32Optical + 32Radio + 32Neutrino + 32GW + 5Labels )2= 27225 final text pairs. Each pair was
assigned a binary label: a label of 1 was assigned with pairs where both Circulars (or Circular and label) belonged to
the same observation type (positive pair), while a label 0 was given to pairs belonged to different observation types
(negative pair). These labeled pairs were used to optimize the embedding model using a contrastive loss function,
implemented via the Sentence Transformers library. Training was conducted over 1 epoch, which we found produced
20https://github.com/nasa-gcn/circulars-nlp-paper

9
Figure 3. GCN topic clusters after reduction with t-SNEwith unsupervised pipeline:. Topics on the right are represented
with their 4 most important keywords as extracted with c-TF-IDF . Next to the topic labels in parentheses are the number of
Circulars each topic contains. Note that outliers were excluded from the representation. The topics are also available in Table
1 .
the best results; more epochs did not lead to improvement. The time required for fine-tuning was approximately half
an hour on an T4 GPU (Google Colab) per epoch.
Both the original (pre-trained) and fine-tuned embedding models were applied to cluster the Circulars database using
BERTopic’s Zero-Shot pipeline, which checks for the cosine similarity between an observation label and a Circular and
assigns labels accordingly. To ensure comprehensive labeling, we set the cosine threshold in the pipeline to 0.1, ensuring
thateveryCircularwasassignedtoatleastonecategory. Modelperformancewasquantifiedbycomputingclassification
accuracies on the training and test sets for the embedding model both before and after training. It is estimated as the
fraction of correctly labeled Circulars in the training and test sets.
Theresultsdemonstratethatcontrastivefine-tuningsignificantlyenhancedtheclassificationaccuracyofourpipeline.
Specifically, the accuracy on the training set increased from 66.9% (pre-trained) to 100% (fine-tuned), while the
accuracy on the test set increased from 65% (pre-trained) to 90% (fine-tuned). Test accuracy declines to 87.5% and
85.0% at epochs two and three, respectively, indicating overfitting with continued training. Thus, we utilized this
fine-tuned model to classify the Circulars into observation-based clusters, as can be seen in the t-SNEplot in Figure
4a, which presents well-separated observation-based clusters. Additionally, the temporal trends among these clusters
were analyzed by extracting the submission dates from each Circular and plotting the observation frequencies over
time, as shown in Figure 4b.
Figure 4a shows the clustering of GCN Circulars by wavelength and messenger type, grouped into five categories:
high-energy, optical, radio, GW, and neutrinos. The largest number of reports in GCN come for the high-energy and
optical observatories, highlighted in red and blue. The radio observations are shown in green, while GW detections,
shown in orange, account for only a small fraction (beginning in 2015). Neutrino observations are represented in purple.
To analyze the behavior of these ML-derived clusters, we performed a trend analysis for each of these category, as
shown in Figure 4b. We clearly see that the GW cluster started in 2015, after which follow-up observations increased

10
Topic Summary Circular Count
GRB Observations and Afterglow Detections with Various Telescopes. 12384
Swift Detection and Analysis of Gamma-Ray Bursts: BAT Trigger Data and XRT Observations. 4355
LIGO/Virgo/KAGRA Detection of Compact Binary Merger Candidates and Their Properties. Also
mentions the IceCube Collaboration and neutrino detection.3476
MASTER-Net Observations of GRBs by Multiple Robotic Telescopes: MASTER-SAAO, MASTER-
OAFA, and Others.2923
Fermi GBM Detections: GRB Light Curves, Spectral Analysis, and Event Fluence. 2175
Swift/UVOT Upper Limits for GRBs: White Exposures, 3-sigma, Uncorrected for Galactic Extinction. 1782
Konus-Wind Observations of Long-Duration GRBs: Light Curves, Spectra, and Fitting Results. 1449
Swift-XRT team reports X-ray source detections and analysis for GRBs. 1409
CALET Gamma-Ray Burst Monitor Detections: GRB 200805A, GRB 170703A, GRB 180720B, and
GRB 170702A.1296
Swift-XRT Analysis of GRB Light Curves and Spectra: Power-law Decays and Column Densities. 1156
Fermi GBM GRB Localization Reports: RA, Dec, Statistical Uncertainty, Skymap, HEALPix, and
Angle from Fermi LAT Boresight.1090
Swift-XRT Team Reports Astrometrically Corrected X-ray Positions for GRBs with Uncertainties and
Improvement Process.988
Radio observations of GRBs with the VLA at 8.46 GHz frequency. 971
Table 1. Topic summaries generated with Mistral 7B Instruct .
and led the rise in the other clusters as well. These results demonstrate the effectiveness of LLM-based clustering in
capturing the evolution of wavelength- and messenger-specific activity within the Circulars.
2.2.3.Gravitational Wave and Gravitational Wave Counterpart Classification
We investigated the classification of Gravitational Wave Circulars using the fine-tuned all-MiniLM-L6-v2 language
model. The model was fine-tuned on balanced dataset of 300 Circulars (100 per label), categorized into three classes:
GW Circulars, GW Counterpart Circulars, and all other Circulars labeled as Not Gravitational Wave. The fine-
tuning process utilized a contrastive learning framework, similar to previous setups, with an 80-20 train-test split.
We also include “category labels” - specifically, “Gravitational Wave,” “Gravitational Wave Counterpart,” and “Non-
Gravitational Wave,” in the training set. In the contrastive learning framework, the total number of possible pairs
were thus (80GW + 80GW counterparts + 80non-GW + 3labels )2, which equals a total to 59,049text pairs. Due
to the large dataset, the training was conducted for 1 epoch, completing in approximately an hour on Google Colab.
The classification accuracies for both the training and test sets in our zero-shot BERTopic pipeline, which assigns
Circulars to the three predefined categories, were evaluated. The base all-MiniLM-L6-v2 model shows limited perfor-
mance, with accuracies of 23.3% for training and 25.0% for testing. After fine-tuning for a single epoch, the model
achieves substantial improvement, reaching 100.0% accuracy on the training set and 98.3% accuracy on the test set.
The results indicate that despite the model being trained for just 1 epoch, it successfully distinguished between the
three categories. Clusters of all Circulars based on this fine-tuned model are shown in the t-SNEplot in Figure 5a.
A clear distinction can be visually observed between Circulars belonging to the GW, GW counterpart, and non-GW
categories in our plot. As a test, Circulars that reported observations on GW 170817 were also labeled and plotted,
shown in yellow. Since GW 170817 was extensively followed up by the ground and space-based observatories, the
vast majority of these Circulars consist of the GW counterparts, hence overlap with the corresponding cluster. 226 of
these 227 circulars were correctly classified as belonging to either the GW or GW counterpart cluster. One Circular
29041 (A. Hajela et al. 2020), initially identified as non-GW, is actually a GW counterpart. The evolution of all
three clusters over time is illustrated in Figure 5b. Thus, fine-tuning all-MiniLM-L6-v2 with contrastive learning and
label-augmented training data leads to highly accurate and generalizable clustering of Circulars.
Figure 5a presents the clustering of GCN Circulars focused on GW events. Only GW detection reports by
LIGO/VIRGO/KAGRA collaboration are shown in green, while clusters corresponding to GW EM counterparts
are shown in red, basically follow-up Circulars of GW detection. All other remaining Circulars not linked to GW

11
(a)
(b)
Figure 4. (a)t-SNErepresentation of GCN observation-based clusters. The embedding vectors of each circular are classified
using cosine similarity between embedded labels and documents. Contrastive fine-tuning helps our model more accurately
classify the circular embeddings. (b) Trends over time of the observation-based circular clusters generated through supervised
contrastive fine-tuning of our embedding model ( all-MiniLM-L6-v2 ).
events which includes GRB, TDE, Blazar, SGR events are shown in blue. To highlight the behavior captured by the
LLM, we additionally just over plot (not ML clustering) Circulars associated with the binary neutron star merger
GW 170817/GRB 170817A event in yellow color. This particular event with GW detection with a confirmed EM
counterpart, demonstrates a dense concentration of follow-up reports overlapping over the red GW counterpart cluster
by LLM, as intensively followed-up by multi-wavelength observatories world-wide. To further understand the ML
clusters, we examined the temporal evolution of these clusters as shown in Figure 5b. The results confirm that both
GW and GW-counterpart clusters emerge only after 2015, coinciding with the start of GW era with first GW detection
from a binary black hole merger, GW150914 (B. P. Abbott et al. 2016), reported by the LIGO and Virgo scientific

12
Collaboration. In contrast, Circulars unrelated to GW events, as shown in blue, have been reported consistently
since the start of GCN. These findings illustrate the effectiveness of LLM-based clustering in tracing the evolution of
multi-messenger follow-up activity.
(a)
(b)
Figure 5. (a)t-SNErepresentation of “Gravitational Wave,” “Gravitational Wave Counterpart,” and “Not Gravitational Wave”
clusters computed using contrastive fine-tuning of the all-MiniLM-L6-v2 model. Green points represent machine-learned GW
Circulars, red represent GW Counterpart Circulars, and blue represents other Circulars. Yellow diamond points mark Circulars
associated with GW 170817, as representative example of cross-cluster overlap. (b) Topic trends over time for the same three
clusters generated using supervised fine-tuning of all-MiniLM-L2-v6. GW 170817 is shown as a yellow hatched histogram.

13
3.INFORMATION EXTRACTION
3.1.Methods
3.1.1.Zero-Shot Information Extraction
Information extraction is a technique used to extract structured data from unstructured texts. These unstructured
sources may include journal articles, scientific papers, and even Circulars, which are written by humans for humans,
and thus follow a flexible format that makes them difficult for machines to parse. Information extraction techniques
are beneficial for automatically processing large volumes of text and extracting structured information from them,
which can be further analyzed and interpreted. Traditionally, this used to be done with manually designed rule-based
systems to extract named entities, events, or relationships (J. R. Hobbs & E. Riloff 2010), and required extensive
customization. Recently, advances in machine learning and NLP have transformed this field. Instead of handcrafted
rules, modern techniques use learning-based models that automatically recognize patterns in text and extract the
required entities. Subsequently, LLMs have advanced information extraction by demonstrating strong performance in
converting scientific texts into structured data (Z. Zheng et al. 2023; A. Dunn et al. 2022; M. Agrawal et al. 2022; V.
Sotnikov & A. Chaikova 2023). A key advantage of using LLMs is their ability to learn domain-specific tasks that
they weren’t explicitly trained on, an ability commonly referred to as zero-shot learning (J. Wei et al. 2021; T. B.
Brown et al. 2020). Zero-shot learning allows LLMs to process vast quantities of domain-specific literature, which in
our case are Circulars, and extract all relevant information within them without the need for curating custom datasets
for each specific task. Furthermore, it has been found that the effectiveness of these models on such tasks can be
improved through prompt engineering (J. Wei et al. 2021), a systematic design of input (or “prompt”) for NLP model
to optimize their outputs. This process involves instructions, examples, or queries that guides the model to generate
contextually relevant responses. For our study, we implemented an open-source LLM and employed carefully tuned
prompts to develop a zero-shot, end-to-end system to extract the astrophysical information from Circulars. We tested
our system to extract the key astrophysical data: (1) Redshift value/range reported for a GRB. (2) GRB event name.
(3) Telescope/observatory that observed the event. (4) Redshift measurement method - spectroscopy or photometry.
Our approach requires no training of the model and thus can be readily applied to extract a wide range of astronomical
data with simple modifications. Thus, by combining LLMs, zero-shot learning, and prompt engineering, we created a
flexible and efficient system for extracting structured astrophysical data from unstructured Circulars.
3.1.2.Model Selection
For information extraction, we employ the 4-bit quantized version of the Mistral 7B Instruct model, which is
implemented with llama.cpp . At the time of its release, this model demonstrated remarkably high accuracy on several
text generation benchmarks despite its relatively compact architecture (A. Q. Jiang et al. 2023). The quantization
reduces its computational footprint even further, enabling efficient execution on a single instance of Google Colab’s
GPU. Preliminary evaluations on sample Circulars have shown promising results, particularly in extracting redshift
information.
3.1.3.Prompt Engineering and Output Parsing with LangChain
The output generated by an LLM is typically in the form of natural language. However, to perform data analysis
for extracting the redshift values, the data must be presented in a more structured format. This is achieved by using
fine-tuned prompts and appropriate output parsing algorithms. The LangChain library (H. Chase 2022) enables us to
create a single prompt template that can be reused for all our Circulars. Additionally, it provides access to built-in
modules that assist with structured output parsing. With some custom refinement, we developed a standardized
template that can be applied to all Circulars to extract redshift information, as seen in Figure 6. Using this template,
we process the Circulars with our quantized Mistral 7B Instruct model, which outputs the redshift information in
JSONformat.
Even with careful prompt engineering, the LLM may still occasionally produce results that deviate from our specified
formatting rules, leading to parsing errors. To resolve this issue and ensure the seamless parsing of our extracted data,
wepost-processedtheLLMoutputwiththehelpofPythonregularexpressions(regex)tofixanyincorrect JSONstrings.
Regular expressions are sequences of characters used to match certain patterns of strings. For example, the regular
expression “ˆGRB[0-9]+” or “ˆGRB [0-9]+” will match any string that begins with “GRB” followed by one or more
digits (e.g., “GRB1234”, “GRB 9”). We analyzed the common error patterns in the LLM’s output and applied regular

14
expressions to match and adjust them to get properly structured JSON. Our post-processing rules can be summarized
as:
•First, we removed all text outside of triple back-ticks ```...```. We stripped any comments, lines starting with
’//’.
•Any incorrectly escaped backslashes ’\’ were fixed. Outputs indicating no redshift values such as ’N/A’, ’None’,
and ’null’ were standardized to ’No Redshift’.
•ForJSONsyntax compliance, any keys or values not in double quotes “” were fixed. Missing commas were added
at the end of every JSONkey-value pair. Dangling commas at the end of the last JSONkey-value pair were then
removed.
•Arrays in the model output were converted to single values for easier parsing. For example, [‘value 1’, ‘value 2’]
would be converted to ‘value 1, value 2’. Similarly, lists in the model output were also be converted to single
values. For example, ‘value 1’, ‘value 2’ would be converted to ‘value 1, value 2’, as JSONrequires single values
per key.
We iteratively refined these post-processing steps until the vast majority of our selected circulars were successfully
processed through our pipeline.
Figure 6. Sample prompt used for extracting redshift information from a GCN Circular, in the top its prompt/input
and at the bottom output from the 4-bit Mistral 7B Instruct model.
3.1.4.Testing and Evaluation with Swift GRB Table
TheNeilGehrelsSwiftObservatory(hereafter, Swift)maintainsamanuallycuratedarchivethatcontainsinformation
on GRBs that are observed by Swift21. This table contains information from the Circulars, we use the redshift values
21https://Swift.gsfc.nasa.gov/archive/grb_table/

15
and related GCN numbers from the archive as test data to evaluate the efficacy of our information extraction system.
Data from the table was scraped, and only rows containing redshift values or ranges in the ‘Redshift’ column were
extracted. The reference Circulars for each redshift measurement were obtained from the ‘References’ column of each
row, which provides the GCN Circular numbers associated with the redshift detections. GRB numbers were also
extracted from the archive’s ‘GRB’ column. Python regular expressions and string pattern matching were used to
retrieve the redshift values, telescope or observatory names, and the types of redshift measurement from the ‘Redshift’
column. The SwiftGRB table generally reports redshifts with terms such as ‘absorption,’ ‘emission,’ or ‘photometric,’
indicating the measurement method. For analysis, we categorized both ‘absorption’ and ‘emission’ measurements as
‘Spectroscopic’ redshifts. If no description was available, we parsed as ‘No Information’ for that redshift measurement
type.
In our evaluation table, rows containing multiple redshift measurements for a single GRB event were counted as
separate entries, one for each measurement. In total, we extracted 540 redshift measurements for our evaluation table.
Additionally, we included 105 Circulars that did not report any redshifts to evaluate our pipeline’s performance on
negative examples. Thus, in total, our evaluation table consisted of 645 data points. Circular numbers from this table
were used to select the circular bodies for testing our LLM. As in our topic modeling pipeline before, the circular
bodies were extracted from the GCN Circular archive after concatenating the subject and body fields in JSONformat
of each circular. Finally, we compared the redshift information extracted by our LLM pipeline from these circulars
with the information reported in Swifttable to evaluate the accuracy of the system.
3.1.5.Addressing LLM Hallucination in Redshift Information Extraction
Hallucination refers to instances where a language model generates unsupported, non-factual, or nonsensical infor-
mation in response to a prompt (Z. Ji et al. 2022; J. Maynez et al. 2020). This problem became particularly evident
when we prompted our model with negative examples that lacked redshift values, but still contained ambiguous signals.
Terms like ‘redshift’ and the letter ‘z,’ which are commonly associated with redshifts, seemed to confuse the model.
The cause for these errors may partly stem from the limitations of our smaller model. To mitigate this problem, we
only prompt our model with Circulars that we are certain should include redshift observations. Thus, we retrieved
only relevant circulars from our database to feed into our pipeline. This method of enhancing LLM accuracy and
reducing hallucination using text retrieval methods is commonly referred to as retrieval augmented generation or RAG
(Y. Gao et al. 2023; P. Lewis et al. 2020).
In our implementation, we adopted a two-step approach to retrieve relevant documents. In the first step, we
performed a straightforward keyword search within the subject field of all circulars, looking for mentions of ‘redshift,’
‘spectr,’ or ‘photo-z,’ along with the term ‘GRB.’ This method effectively retrieved the majority of GRB redshift-
related Circulars. However, some circulars in our database reported GRB redshifts without explicitly mentioning
them in their subject field. To capture these, we next implemented a neural search mechanism with the assistance of
LangChain . First, we split our remaining Circulars into smaller chunks to be more easily encoded by an embedding
model. We then embedded these chunks using the small but efficient all-MiniLM-L6-v2 Sentence Transformer. The
resulting embedding vectors were indexed and stored in a locally hosted vector database designed for storing and
querying over embeddings. Here we used the Facebook AI Similarity Search (FAISS) library (M. Douze et al. 2025).
Finally, we searched over this new database with the following query: “Spectroscopic Redshift Z Value with Absorption
or Emission Lines.” This query was first encoded using the same embedding model, after which it was compared to
all Circular embeddings using a cosine similarity metric. Circulars that exceeded a certain similarity threshold, in our
case 0.3, with the query were appended to our previously retrieved set. This method allowed us to gather the majority
of relevant circulars with reported redshift values, which we subsequently used for inference with our LLM. Figure 7
gives an overview of our redshift information extraction pipeline.
3.2.Results
3.2.1.Evaluation of Redshift Information Extraction
The evaluation of the zero-shot information extraction pipeline was conducted in Google Colab using an L4 GPU
instance. The total inference time for processing all 645 Circulars in our evaluation dataset was approximately one and
a half hours. Table 2 shows 10 randomly selected entries from our LLM-generated redshift information table, alongside
the actual reported information from our evaluation dataset. 644 of the LLM output strings were successfully parsed
into JSON objects without errors using structured output parsing and regex methods and were thus added to our new
evaluation dataset.

16
Figure 7. Visual representation of the redshift information extraction pipeline. Keyword search and neural search are combined
to retrieve relevant redshift Circulars. For the generation step, retrieved redshift Circulars are combined with the tuned
LangChain prompt template and fed into Mistral 7B Instruct . Finally, the generated output is processed to remove any
formatting errors and parsed into JSONkey-value pairs, with the fields containing the extracted redshift information from each
Circular.
To access the accuracy of our LLM-generated redshift information, we manually evaluated all 644 data points. We
divided our dataset into two subsets: one subset consisted of positive examples, which were Circulars taken from
theSwiftGRB table that we knew contained redshift information. The second subset was comprised of the negative
examples, which had no mention of reported redshifts. Thus, in total, we had 539 Circulars in the positive examples
and 105 Circulars in our negative examples subset. In Table 3, we present the total number of correct and incorrect
redshift predictions for each subset, along with the overall accuracy of our pipeline for redshift value extraction.
Furthermore, we evaluated the model’s ability to extract redshift values, GRB numbers, telescope names, and redshift
types from the positive examples in our dataset. These results of this evaluation can be found in Table 4. We followed
some general rules while evaluating the model’s performance in each field. For the negative examples, any prediction
that contained a numeric value was considered a false positive. For positive examples, any time the model predicted
‘No Redshift’ we marked it as a false negative. If it predicted a wrong or bogus value for the redshift it was marked
as a false positive. In circulars where multiple redshifts were reported, we would often mark the model’s prediction
as correct if it predicted any one of them, unless the circular clearly mentioned that one of them was the most likely
redshift for the GRB. Some other basic rules we followed during evaluation were:
•For redshift ranges and limits, if the model extracted only the value of the upper or lower limit without indicating
that it was a limit, we counted the redshift prediction as incorrect.
•Predictions for GRB numbers had to match the actual GRB numbers mentioned in the Circular exactly; no
trailing characters or letters were allowed.
•We allowed some leeway in the case of telescope name predictions. Correct predictions of observatory or in-
strument names based on the information provided in the Circular were marked as correct, even if they did not
match exactly with the telescope names reported in the SwiftGRB table. For example, in GCN Circular 11195,
our model predicted ‘ Swift’ as the telescope name, even though the SwiftGRB table refers to it as ‘ SwiftXRT.’
We still marked this prediction as correct as it accurately identified the observatory name.
Finally, we evaluated the model’s performance based solely on the information provided in each circular. There were
instances where redshifts or GRB numbers in the SwiftGRB table would be revised in later Circulars, causing the
model’s output to not match up with the data in the table. We marked our model’s output as correct as long as its
prediction was correct based on the information present within the circular in its prompt.
The most common errors made by the model for the negative examples dataset included:

17
GCN Predicted /
Actual GRBPredicted /
Actual RedshiftPredicted /
Actual TelescopePredicted /
Actual Redshift Type
9712 GRB090726 /
0907262.71 /
2.71SAORAS6-mtelescopeinCau-
casus using SCORPIO /
SAO RASSpectroscopic /
Spectroscopic
12867 GRB 120119A /
120119Az=1.728 /
1.728MMT 6.5-m telescope /
MMTSpectroscopic /
Spectroscopic
4591 GRB 060124 /
0601240.82 /
2.30MDM 2.4m telescope /
MDMSpectroscopic /
Spectroscopic
18969 GRB 160131A /
160131A0.97 /
0.97Xinglong 2.16m telescope /
XinglongSpectroscopic /
Spectroscopic
33110 GRB 221226B /
221226B2.694 /
2.694ESO VLT UT3 (Melipal) /
VLTSpectroscopic /
Spectroscopic
7601 GRB 080413B /
080413B1.10 /
1.10ESO VLT /
VLTSpectroscopic /
Spectroscopic
12761 GRB 111228A /
111228A0.716 /
0.716Gemini-South 8-m telescope /
Gemini-SouthSpectroscopic /
Spectroscopic
7615 GRB 080413B /
080413B1.10 /
1.10Gemini South /
Gemini-SouthSpectroscopic /
Spectroscopic
11708 GRB 110213A /
110213Az = 1.46 /
1.462.3-m Bok telescope /
BokSpectroscopic /
Spectroscopic
22039 GRB 171020A /
171020A1.87 /
1.87Nordic Optical Telescope
(NOT) /
NOTSpectroscopic /
Spectroscopic
Table 2. Side by side comparison of redshift information extracted by ML for SwiftGRB Circulars (10 randomly selected
entries) with the actual redshift information from the SwiftGRB Table. Leftmost column gives the GCN Circular number from
which the LLM extracted the information from. Information generated by the LLM includes GRB numbers, redshift values /
ranges, telescope names, and redshift types.
Positive Negative Total
Correct 524 92 616
Incorrect 15 13 28
Accuracy(%) 97.2 87.6 95.7
Table 3. Correct predictions, incorrect predictions, and overall accuracy of extracted redshift values on positive, negative, and
all examples in our evaluation dataset.
Redshift Value GRB Number Telescope Name Redshift Type
Correct 524 532 533 530
Incorrect 15 7 6 9
Accuracy(%) 97.2 98.7 98.9 98.3
Table 4. Correct predictions, incorrect predictions, and accuracies of extracted redshift values, GRB numbers, telescope names,
and redshift types evaluated against positive examples taken from the SwiftGRB table.
•Hallucination of random numbers: The model occasionally generated values that were not found in the original
Circular, leading to inaccuracies. For example, in GCN Circular 10076, the model incorrectly predicted a redshift
value of 0.342, which was not present in the Circular.
•Mistaking other numeric values: Another common error was the misinterpretation of numeric figures for GRB
redshift values. In GCN Circular 10046, the model extracted the reported power-law decay index of -0.95 as the
redshift.

18
For the positive examples, the most frequent errors were:
•Reporting limits as exact values: The model sometimes picked the lower or upper limits of redshift ranges as
definite values. For example, in GCN Circular 12202, where the reported redshift range was 1.036 < z < 2.7,
the model returned just 1.036.
•False negatives in redshift: There were instances where the model failed to report a redshift. For instance, GCN
Circular 16891 reported tentative redshifts of 0.17 and 0.57 for GRB 141004A, which the model failed to extract.
•GRB number errors: Occasionally, the model would get the GRB number correct but fail to include the letter,
like in the case of GCN Circular 32099, where the model fails to add the letter ’A’ at the end of GRB 220521.
These are marked as incorrect here.
•Telescope name errors: Whenever the name of the telescope or observatory was not mentioned in the Circular,
the model sometimes struggled to obtain it. For example, in Circular 5962, the telescope name “Magellan” wasn’t
provided in the text, and we still marked this as an error.
Despite all the above errors, the overall error rate remained relatively low, resulting in a high accuracy of 95.7%
across the complete dataset. We recognize that further improvements in accuracy could be achieved with access to
better hardware, which would allow us to run larger and more accurate versions of the model.
Upon manual evaluation of extracted data, we also found that some Circulars listed in the SwiftGRB reference table
as having reported redshifts did not actually contain redshifts. We counted 5 such circulars in total. This may have
been due to human error, which is often unavoidable. However, we noted our model was able to correct such instances
4 out of 5 times. Additionally, the model consistently predicted redshift types for the vast majority of Circulars, even
when that information was not present in the SwiftGRB table itself. This further underlines the potential of the
information extraction pipeline: not only does it automate and speed up the process of astronomical data extraction,
but also improves it by compensating for human error and recovering missing classifications.
3.2.2.Retrieval, Post-Processing, and Data Analysis
For the redshift extraction over the entire GCN Circular database, a total of 1287 Circulars were retrieved using
a combination of keyword and neural search methods. This accounted for approximately 3% of all Circulars in our
database. Of these, 1,095 Circulars were obtained through simple keyword matching in the subject line. An additional
192 Circulars were retrieved with the help of our neural search method. To evaluate the effectiveness of our retrieval
methods, we calculated the number of redshift Circulars from our previous positive examples dataset that our method
successfully retrieved. The keyword search alone helped us retrieve 505 out of the 539 redshift Circulars listed in the
SwiftGRB table. The neural search method contributed an additional 17 Circulars from this dataset, reducing the
number of missing Circulars by half. Thus, in total we retrieved 522 of the 539 Circulars in our positive examples
dataset, giving us a retrieval recall rate of 96.8%. Manual analysis of the Circulars retrieved through neural search
also shows that 175 of the additional 192 retrieved Circulars contained redshift values, ranges, limits, or estimates,
giving a retrieval precision of approximately 91.1% for the neural search component of our methods. Thus, this method
effectively extracted the majority of all redshift-reporting Circulars from our database.
Information was then extracted from all retrieved Circulars. On a single Colab L4 instance, the execution time
was approximately 3 and a half hours. From the total of 1287 Circulars, 1259 were parsed successfully. After data
extraction, rows containing no redshifts were dropped, leaving us with 1078 entries in our redshift table.
After retrieving and generating output, the extracted redshift information required post-processing before we could
conduct data analysis. Our aim was to extract a single redshift value for each GRB in our output dataset. However,
many Circulars contained redshift measurements for the same GRB events. Often these would be reconfirmations of
prior measurements, but sometimes would involve different measurements taken with different instruments or methods.
To resolve these conflicting values, we used the following system. First, all identical GRB events in our dataset were
aggregated. We then analyzed the redshift information extracted for these identical GRB events and resolved conflicts
based on the following rules:
•If both photometric and spectroscopic redshifts were reported, we prioritized the spectroscopic measurements.
•If there were still multiple values, we selected the redshift with the highest reported decimal precision.

19
•If conflicts still existed, we chose the redshift value that had the earliest reported date, using the ’createdOn’
field from the Circular JSON. This approach is based on the assumption that the earliest report is the original
one, and later ones may be reconfirmations or reuse the previous values.
A total of 592 conflicting or multiple reported redshift values for the same GRB were removed. Among these 228
unique GRB events were extracted. Thus, after post-processing, one Circular per GRB was selected and added back
to our redshift table. Consequently, our final redshift table contained 714 unique GRB redshifts. Table 5 displays a
randomly selected sample of 10 entries from this table.
Next, the redshift strings extracted from the circulars were formatted into floating-point numbers for data visual-
ization. Since the format of the redshift strings extracted from the Circulars was not always uniform, this step also
required some text post-processing. For example, GCN 10233 reported the redshift value with uncertainty as “0.49034
+/- 0.00018”. To extract a single redshift value per GRB, we applied the following rules:
•Numeric redshift values were extracted as they were, without any associated uncertainties.
•If only upper or lower limits are provided, we extracted the corresponding limits.
•If a redshift range was provided, we determined its midpoint and used that as our redshift value.
Pattern matching with Python regular expressions was used to automate this step. Values greater than 8.2 were also
removed, as the maximum reported redshift lies within this limit. In total, 687 of the 714 extracted redshift values
were utilized for plotting.
GCN GRB Number Redshift Telescope Name Redshift Type
16505 GRB140703A z=3.14 GTC (Gran Telescopio
Canarias)Spectroscopic
22096 GRB 171010A 0.3285 ESO Very Large Telescope
(VLT)Spectroscopic
10620 GRB100418A 0.6235 VLT/X-Shooter Spectroscopic
10441 GRB 100219A 4.8 VLT Spectroscopic
8752 GRB 081228 3.8 +- 0.4 GROND Photometric
31107 GRB 211023B 0.862 LBT-INAF Spectroscopic
3542 GRB 050416(a) 0.6535 +/- 0.0002 10-m Keck I Telescope Spectroscopic
6952 GRB 071020 z = 2.145 Very Large Telescope (VLT) Spectroscopic
17758 GRB 150424A z = 0.30 10.4m GTC telescope
(+OSIRIS)Spectroscopic
38097 GRB 241105A 2.681 ESO VLT UT1 (Antu) Spectroscopic
Table 5. Extracted redshift information (10 randomly selected entries) from the retrieved circulars in the GCN database before
post-processing. GCN numbers are taken from the ’CircularId’ field of the circulars while predicted redshifts, GRB numbers,
telescope names, and redshift types are all generated by Mistral 7B Instruct after prompt tuning and output parsing with
LangChain . (Note: The blank space between 3.3721 and 0.0004 for the redshift of GCN 9015 should be interpreted as a ±sign.
The actual redshift is thus 3.3721 ±0.0004. This happens due to corruption of certain characters during text pre-processing.
Also note that the model makes an error for the GRB prediction of GCN 32099, extracting a GCN number from the circular
instead.)
Our analysis presents the extracted redshift values as a histogram in Figures 8 and 9, allows us to learn the redshift
distribution in GRBs utilizing ML techniques to derive the values from Circulars. We compare the frequencies of
photometric and spectroscopic redshifts, in Figure 8.
To study the behavior of the information extraction task using the LLM, we compile a table of the ten highest
redshifts reported in the Circulars, as shown in Table 6. As a caveat we note that the LLM does not distinguish
between observationally confirmed values and those inferred from model assumptions or theoretical considerations
values mentioned in the Circulars. Since the highest spectroscopically confirmed redshift, GRB 090423, is at z∼8.1
(A. Fernandez-Soto et al. 2009), we truncated values up to that limit. Without this truncation, the LLM would

20
additionally extract redshifts reported only as theoretical predictions or assumed for model parameters. Additionally,
we note that the ML model does not provide accurate classifications for the type of redshift; for example, GRB 090423
and GRB 060116 have spectroscopic redshifts but mis-classified as photometric by the model.
GCN GRB Number Redshift Telescope Name Redshift Type
9222 GRB090423 z∼8.1 NICS/Amici combination Photometric
9216 GRB 090423 z∼7.6 Italian TNG 3.6m telescope Spectroscopic
39732 GRB 250314A 7.3 ESO VLT Spectroscopic
35756 GRB 240218A 6.782 ESO/VLT UT3 (Melipal) Spectroscopic
8225 GRB080913 z=6.7 VLT (Very Large Telescope) Spectroscopic
8256 GRB 080913 z = 6.7 Swift-BAT and Konus-Wind Spectroscopic
4545 GRB 060116 6.6 ESO VLT Photometric
16269 GRB 140515A z=6.32 GMOS on Gemini-N 8-m
telescopeSpectroscopic
30771 GRB 210905A 6.318 ESO VLT Melipal Spectroscopic
3937 GRB 050904 6.29 Subaru 8.2m telescope Spectroscopic
Table 6. Highest redshift values extracted, after removal of errors, before post-processing.
Figure 8. Photometric and spectroscopic redshift frequencies as reported in the GCN Circular database after post-processing.

21
Figure 9. Distribution of SwiftGRB redshifts (cyan) vs. ML-predicted post-processed redshifts (gray), along with their median
values.
4.DISCUSSION AND CONCLUSION
Our study highlights the potential of modern NLP techniques and open-source LLMs to extract insights from
large astrophysical text databases. We successfully applied neural topic modeling techniques to uncover common
astrophysical themes and trends present within the rich GCN Circular database. By employing contrastive fine-tuning
techniques on transformer-based language models, we classified the Circulars into five categories based on the type
of observations reported, allowing us to collect some frequency statistics and examine trends for various observation
types in GCN. Additionally, we specifically classified GW Circulars, their counterparts, and non-GW Circulars, we
observed a clear increase in interest in multi-messenger astronomy since 2015. Furthermore, we implemented a zero-
shot pipeline using an open-source language model to retrieve and extract GRB redshift information from the GCN
database. When evaluated on redshift-containing Circulars from the Swiftobservatory’s GRB table, our pipeline
achieved an accuracy of over 97% on redshift value/range prediction and over 98% on event names, telescope names,
and redshift types prediction. Additionally, by integrating retrieval augmented generation or RAG techniques using a
combination of keyword and neural search, our pipeline was able to extract a substantial amount of GRB redshift data
automatically. This significantly reduced the need for manual data extraction. Our study indicates that expensive
training and private models are not necessary to achieve the results in astrophysical information extraction. With
simple prompt modifications, it is possible to extract a wide variety of desired information for further data analysis.
Structured extraction of information in standardized formats such as .txt or .json can greatly streamline the con-
struction of light curves and spectral energy distributions (SEDs), enabling rapid modeling of afterglow behavior. With
access of this crucial information astronomers can then make near real-time decisions about observation scheduling, cal-
culate required exposure times based on decay slopes, and coordinate multi-wavelength campaigns far more efficiently.
Thus, in future by broadening the scope from redshift-centric analytics to a multi-parameters, filter-based extraction
system, as real-time pipeline by leveraging the LLM-powered approach is transformative for designing the follow-up
ready intelligence platform. This approach would directly benefit the global GRB observation network, ensuring that
crucial observational opportunities are not lost in the hours following-up of the burst.
We anticipate LLMs will be useful for enhancing the utility of the Circulars archive and even astronomical alerts
database, particularly by providing rapid and precise access to key observational parameters to the follow-up com-
munity. While the current pipeline only extracts redshift data, an LLM-based system could be trained to parse and
standardize to a much broader range of critical metadata, such as time since burst ( T−T0), exposure times, filters,

22
magnitudes, magnitude systems (AB or Vega), telescope/instrument names, and the Circular number itself. By recog-
nizing and classifying the photometric observations based on UV, optical, and near-infrared (NIR) filter information,
the model may enable instantaneous, filter-specific data sorting and retrieval, which is currently limited by the need for
manual processing. Such a capability would significantly enhance the scientific readiness for the follow-up community,
particularly for Optical/NIR/Radio afterglow observations.
Automatic prompt and query generation through the use of autonomous agents is an upcoming area of research
within the NLP space which could potentially help improve the accuracy and robustness of our information extraction
pipeline(S.Barua2024). Additionally, thetestingofourpipelinewithlargerLLMscanhelpusrealizethetruepotential
of state-of-the-art NLP methods for astrophysical data extraction. In the near future, we aim to investigate how our
NLP pipelines can be integrated into the GCN platform, potentially functioning as an AI assistant for astronomy.
4.1.Challenges and Limitations
In our research, there are some limitations. The observations-based classifications require the manual collection of
data to fine-tune our transformer model. This means that if we wanted to classify based on different criteria, we
would need to collect new training datasets accordingly. For our information extraction pipeline, the lacked access
to high-end GPUs forced us to rely on those provided by Google Colab. As a result, we had to use smaller LLM
models that had more limited accuracy compared to larger models, as these were the only ones that could fit into
the available GPUs. We thus believe that the accuracy of our information extraction pipeline could be improved
considerably through the use of better hardware. Furthermore, our pipeline required us to use some manual output
parsing techniques in addition to automatic ones to successfully parse the vast majority of Circulars. Occasional
non-uniformity of the model output was the cause for this, which could have been reduced with larger, more accurate
instruction-tuned LLMs, along with more refined prompt-tuning strategies. The prompt-tuning itself was an iterative
process that required domain knowledge and several attempts to achieve satisfactory results. Finally, despite our
best efforts at reducing hallucination through the use of RAG, a small number of errors still slipped in, partly due
to the retrieval of Circulars that did not contain redshift information. Depending upon the required accuracy of the
information retrieval task at hand, this may make the pipeline in its current state not suitable for highly sensitive and
accurate tasks.
As a caveat we also acknowledge that NLP is a rapidly evolving field, with new LLM models and libraries emerging
at a fast pace, which presents both limitations and opportunities. In this study, we employed BERTopic and Mistral
models for topic modeling and information extraction, relying on model architectures and tools developed in recent
years. While these models have produced promising initial results, we recognize that future developments may offer
improvedperformanceandbetteraccuracy. ThisfirstapplicationofBERTopicand Mistral modelsformulti-messenger
alert system provides a foundation upon which more future work can build.
ACKNOWLEDGMENTS
We thank the anonymous referee for useful comments and suggestions on the manuscript. VS was sponsored by
support from the National Aeronautics and Space Administration (NASA) through a cooperative agreement with
Center for Research and Exploration in Space Science and Technology II (CRESST II). The views and conclusions
contained in this document are those of the authors and should not be interpreted as representing the official policies,
either expressed or implied, of the National Aeronautics and Space Administration (NASA) or the U.S. Government.
The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding
any copyright notation herein. The GCN team acknowledges support from the NASA’s Internal Scientist Funding
Model (ISFM) program. This research has made use of data obtained through the General Coordinate Network
(GCN) Service, provided by the NASA Goddard Space Flight Center (GSFC), in support of NASA’s High Energy
Astrophysics Programs. The authors would also like to thank Daniela Huppenkothen for the insightful discussions. RG
was sponsored by the National Aeronautics and Space Administration (NASA) through a contract with ORAU. M.W.C
acknowledges support from the National Science Foundation with grant numbers PHY-2409481, PHY-2308862 and
PHY-2117997. NM acknowledges support from the National Science Foundation (NSF) under awards PHY-1764464
and PHY-2309200 to the LIGO Laboratory, under Cooperative Agreement PHY-2019786 (The NSF AI Institute for
Artificial Intelligence and Fundamental Interactions, http://iaifi.org/), and from MathWorks, Inc.

23
CODE AVAILABILITY
All codes are available to the wider astrophysical community via the GitHub public repository22atnasa-gcn . It
contains data, figures, codes in the Google Colab notebooks, and extended tables.
APPENDIX
A.WORD CLOUD REPRESENTATION OF GCN CIRCULARS
Figure 10 shows a word cloud generated over the entire GCN Circular database.
Figure 10. GCN Word Cloud
B.OBSERVATIONS AND GRAVITATIONAL WAVE SIMILARITY MATRIX
Figure 11a shows the similarity matrix of all 5 observation types, created using BERTopic. The topic embeddings
of each observation-based cluster are used to create this cosine similarity matrix.
22https://github.com/nasa-gcn/circulars-nlp-paper

24
(a)Observation-based clusters.
(b)Gravitational-wave focused clusters.
Figure 11. Similarity matrices for: (a) all 5 observation types, and (b) gravitational-wave focused clusters. Each type is
represented with its most important keywords extracted using TF-IDF.

25
Figure 11b presents the gravitational event map generated using BERTopic. The topic embeddings of each
gravitational-based cluster are used to create this cosine similarity matrix.

26
C.RETRIEVED AND EXTRACTED REDSHIFTS TABLE
Finally, all retrieved Circulars with their extracted redshift information, before post-processing, are shown in Table
C.
GCN GRB Number Redshift Telescope Name Redshift Type
12 GRB 971214 1.02 Keck-II 10-m telescope Spectroscopic
13 GRB 971214 1.03 Keck-II 10-m telescope Spectroscopic
29 GRB 971214 z=3.43 Keck II telescope Spectroscopic
137 GRB 980703 0.967 Keck-II telescope Spectroscopic
139 GRB 980703 0.9660 +- 0.0002 Caltech GRB collaboration Spectroscopic
189 GRB 980613 1.0964 +- 0.0003 Keck-I Spectroscopic
216 GRB 990123 Approximately 0.2
to 0.3The Caltech-CARA-NRAO
collaborationPhotometric
219 GRB 990123 1.600 +- 0.001 NOT, Copenhagen, IAC, LA-
EFF, IAASpectroscopic
223 GRB 990123 z∼0.2 Keck Spectroscopic
249 GRB 990123 1.598 +- 0.001 NOT, Copenhagen, IAC, IAA,
LAEFF, IAASpectroscopic
251 GRB 990123 0.2783 +- 0.0005 4m Blanco Telescope Spectroscopic
289 GRB 970228 0.695 +- 0.002 Keck-II 10-m telescope Spectroscopic
324 GRB 990510 1.619 +- 0.002 ESO VLT-UT1 Spectroscopic
331 GRB990510 No Redshift ESO VLT Spectroscopic
388 GRB 990712 0.430 +/- 0.005 ESO VLT-UT1 telescope Spectroscopic
460 GRB 991208 No Redshift Keck II 10-m Telescope Spectroscopic
464 GRB 991208 1.29 Keck II 10-m telescope Spectroscopic
475 GRB 991208 z = 0.707 +/-0.002 SAO-RAS 6-m telescope Spectroscopic
481 GRB 991208 0.7055 +- 0.0005 Keck Observatory Spectroscopic
488 GRB 991216 No Redshift Hobby-Eberly Telescope at Mc-
Donald ObservatorySpectroscopic
496 GRB991216 z=0.80 VLT-Antu Spectroscopic
497 GRB 991216 No Redshift Keck-II 10-m telescope Spectroscopic
511 GRB 991216 No Redshift Kitt Peak National Observatory Spectroscopic
584 GRB000301C No Redshift Hobby-Eberly Telescope Spectroscopic
603 GRB 000301C 1.95 +/- 0.1 HST (Hubble Space Telescope) Spectroscopic
605 GRB 000301C 2.0335 Keck-II 10-m telescope on
Mauna Kea, HawaiiSpectroscopic
606 GRB 000301c 1.89 McDonald Observatory Spectroscopic
607 GRB 000301c 1.89 McDonald Observatory Spectroscopic
661 GRB 000418 z=1.11854 +/-
0.0007Keck II 10-m Telescope on
Mauna Kea at 2.318 May 2000
UTSpectroscopic
778 GRB 980329 No Redshift Hubble Space Telescope Spectroscopic
807 GRB 000926 z = 2.066 Nordic Optical Telescope
(NOT)Spectroscopic
851 GRB 000926 2.0369 +- 0.0007 WMKO Keck-II telescope Spectroscopic
965 GRB010222 1.476 F. L. Whipple Observatory
1.5m Tillinghast telescope (+
FAST spectrograph)Spectroscopic
974 GRB010222 1.477 Harvard-Smithsonian CfA Spectroscopic
989 GRB 010222 0.928 Keck I 10-m telescope Spectroscopic
999 GRB 010222 1.4768 +- 0.0002 Keck-II 10-m telescope Spectroscopic

27
1090 GRB 001109 0.398 +/- 0.002 SAO RAS 6-m telescope Spectroscopic
1108 GRB010921 0.450 +- 0.005 Palomar 200-inch telescope Spectroscopic
1152 GRB011121 0.36 Magellan Walter Baade 6.5m
telescope and the LDSS-2
spectrographSpectroscopic
1183 GRB 011130 0.50 Magellan 6.5m Baade telescope
(+ LDSS2)Spectroscopic
1200 GRB (XRF)
0112112.14 Yepun (VLT4) 8-m telescope
of the European Southern
ObservatorySpectroscopic
1203 GRB011211 z=2.1 TNG (Telescopio Nazionale
Galileo)Spectroscopic
1209 GRB 011211 2.14 Magellan 6.5m Walter Baade
telescopeSpectroscopic
1247 SN2002ap No Redshift Apache Point Observatory 3.5-
meter telescopeSpectroscopic
1248 SN 2002ap (660 +/- 2) km/s Tautenburg 2-m telescope Spectroscopic
1330 GRB020405 0.695 +- 0.005 VLT Spectroscopic
1340 GRB 020405 0.6898 +/- 0.0005 Keck II telescope + ESI Spectroscopic
1428 GRB 020531 1.00 Keck II Spectroscopic
1474 GRB 020813
(HETE Trigger
#2262)No Redshift CTIO 4m Blanco telescope and
Mosaic II imagerPhotometric
1475 GRB 020813 z = 1.254 Keck I Spectroscopic
1477 GRB 020813 3.0% +/- 0.1% Keck I 10 m telescope Spectroscopic
1524 GRB020813 z = 1.2545 UVES at VLT/Kueyen Spectroscopic
1569 GRB021004 1.60 Siding Springs Observatory
2.3m telescope and double-
beam spectrographSpectroscopic
1579 GRB021004 1.38, 1.60 Hobby-Eberly Telescope Spectroscopic
1582 GRB 021004 1.38, 1.60, 1.72 2-m Himalayan Chandra Tele-
scope, Indian Astronomical Ob-
servatory, HanleSpectroscopic
1587 GRB 021004 1.38, 1.60 Himalayan Chandra Telescope Spectroscopic
1605 GRB 021004 2.323 Keck-I 10-m telescope Spectroscopic
1608 GRB 021004 z=2.323 Hubble Space Telescope (HST) Spectroscopic
1611 GRB 021004 2.3 4.2-m WHT telescope at La
Palma (Spain)Spectroscopic
1618 GRB 021004 2.328 MDM 1.3m Spectroscopic
1633 GRB021004 2.321, 2.327 VLT (Very Large Telescope) Spectroscopic
1635 GRB 021004 2.327 VLT (Very Large Telescope) Spectroscopic
1672 GRB021004 No Redshift ESO-VLT Spectroscopic
1678 GRB021004 z=2.335 VLT/UT1 Spectroscopic
1756 GRB 021211 0.800 +- 0.001 VLT Spectroscopic
1767 GRB 021211 0.168 Very Large Telescope (VLT) Spectroscopic
1785 GRB 021211 1.006 VLT Spectroscopic
1809 GRB 021211 1.004+-0.002 ESO-VLT Spectroscopic
1846 SN/GRB? No Redshift MMT 6.5m telescope No Information
1882 GRB030226
(=H10893)2.1 or larger Subaru NAOJ Spectroscopic
1884 GRB030226
(=H10893)1.99 Subaru 8.2m telescope Spectroscopic

28
1889 GRB 030226 1.984, 1.961, 1.043 Keck II 10-m telescope Spectroscopic
1897 GRB 030226 1.044 Keck I Spectroscopic
1953 GRB 030323
(HETE trigger
2640)z = 3.372 ESO’s VLT at Paranal, Chile Spectroscopic
1980 GRB 030328 z=1.52 Magellan 6.5-m Clay
telescope and LDSS2
imaging/spectrographSpectroscopic
1981 GRB 030328 z=1.520 VLT Spectroscopic
1983 GRB030328 1.52 TNG Spectroscopic
2013 GRB 030329 0.57 Magellan 6.5-m Clay telescope Spectroscopic
2015 GRB030329 z=0.568 DOLORES at TNG Spectroscopic
2017 GRB 030329 0.57 SSO 40-inch Spectroscopic
2020 HETE (H2652)
GRB 0303290.1685 VLT unit Kueyen Spectroscopic
2053 GRB 030329 0.168 MMT Spectroscopic
2076 GRB030328 1.7 +/- 0.2 Chandra X-ray Observatory Spectroscopic
2107 GRB 030329 Approximately
V=22MMT (Multi-Mirror Telescope) Spectroscopic
2117 GRB030329 0.168 Hobby-Eberly Telescope Spectroscopic
2142 GRB030329 0.168 SAO RAS 6-meter telescope Spectroscopic
2145 GRB030329 0.342 SAO RAS Spectroscopic
2167 GRB 030329 No Redshift ESO VLT-UT1 (Antu) Spectroscopic
2169 GRB 030329 0.168 MMT (Mountain McDonald
Observatory)Spectroscopic
2196 GRB 030429 2.65 ESO VLT Spectroscopic
2212 GRB 030329 0.168541 +/-
0.000004Las Campanas Observatory Spectroscopic
2215 GRB 030429 z=2.6564+/-
0.0008VLT (Very Large Telescope) Spectroscopic
2238 GRB030519B
(=H2716)No Redshift HETE/IPN Spectroscopic
2305 GRB030329 0.168 Subaru Telescope Spectroscopic
2429 GRB031026
(=H2882)No Redshift HETE Spectroscopic
2431 SN/GRB? No Redshift F. L. Whipple Obs. No Information
2432 GRB 031026 No Redshift HETE Pseudo-z
2444 GRB031026 No Redshift HETE Spectroscopic
2468 GRB 031203 >∼9 XMM-Newton Photometric
2469 GRB 031203 z∼10 XMM-Newton Photometric
2482 GRB 031203 0.105 IMACS on the Baade telescope Spectroscopic
2517 GRB030329 0.168 Keck I 10-m telescope (+LRIS) Spectroscopic
2586 GRB 011211 0.1685 Swift Spectroscopic
2599 GRB040511 2.63 Keck-I telescope Spectroscopic
2610 GRB040511 No Redshift Keck-I No Information
2685 GRB 040827 No Redshift VLT Spectroscopic
2698 GRB 040827 z=0.9_-0.2 ˆ+0.9 XMM-Newton Spectroscopic
2782 GRB041006 z=0.712 Italian 3.6m telescope TNG at
the Canary Islands.Spectroscopic
2791 GRB 041006 0.716 Gemini North Telescope +
GMOSSpectroscopic

29
2800 GRB 040924 0.859 ESO VLT (Antu) Spectroscopic
2910 GRB 041223 No Redshift Swift XRT Spectroscopic
2973 GRB050124 No Redshift Swift Spectroscopic
3019 GRB050209 No Redshift HETE WXM Team No Information
3088 GRB 050126 1.29 Keck II telescope Spectroscopic
3096 GRB 050219b No Redshift Gemini-south Spectroscopic
3101 GRB 050315 1.949 Magellan Spectroscopic
3122 GRB 050318 z=1.44 Magellan/Baade telescope Spectroscopic
3123 GRB 050318 z∼2.7 Swift Spectroscopic
3136 GRB 050319 3.24 Nordic Optical Telescope
(NOT)Spectroscopic
3176 GRB 050401 z = 2.90 Very Large Telescope (VLT) Spectroscopic
3201 GRB 050408 z=1.236 Magellan/Clay telescope Spectroscopic
3204 GRB 050408 z=1.2357 +/-
0.0002Gemini Observatory Spectroscopic
3332 GRB 050502a 3.793 KeckI Spectroscopic
3368 GRB 050505 4.27 Keck I 10-m telescope Spectroscopic
3390 GRB050509b 0.226 DEIMOS Spectroscopic
3476 GRB 050525 0.36 +/- 0.1 LAT-OMP Spectroscopic
3483 GRB 050525 0.606 Gemini-North telescope Spectroscopic
3484 GRB 050525 0.72 +- 0.15 Swift Spectroscopic
3520 GRB 050603 2.821 Magellan/Baade telescope Spectroscopic
3531 GRB050607 No Redshift Mayall Telescope of the Kitt
Peak National ObservatorySpectroscopic
3542 GRB 050416(a) 0.6535 +/- 0.0002 10-m Keck I Telescope Spectroscopic
3605 GRB 050709 z = 0.16 Gemini North telescope Spectroscopic
3653 GRB050709 0.16 Swift Spectroscopic
3675 GRB 050724 No Redshift ESO VLT Spectroscopic
3679 GRB 050724 z=0.257 GMOS Spectroscopic
3695 GRB050713A 0.55 XMM-Newton Spectroscopic
3700 GRB 050724 0.258 +/- 0.002 LRISb Spectroscopic
3709 GRB050730 3.967 Magellan II Spectroscopic
3710 GRB050730 3.97 William Herschel Telescope
(WHT)Spectroscopic
3716 GRB 050730 3.97 Magellan Observatory Baade
Telescope and IMACS imaging
spectrographSpectroscopic
3721 GRB050525aJ-L 0.64 +/- 0.1 LAT-OMP Photometric
3732 GRB 050730 3.96855 +/-
0.00005MIT Spectroscopic
3746 GRB050730 3.968 ESO-Chile Spectroscopic
3749 GRB050802 z=1.71 Nordic Optical Telescope Spectroscopic
3798 GRB 050813 ∼1 Magellan/Baade 6.5-m
telescopePhotometric
3801 GRB 050813 z=0.722 Gemini north telescope Spectroscopic
3808 GRB 050813 0.718 Gemini-North Spectroscopic
3833 GRB 050820 2.612 +/- 0.002 HIRES/Keck I Spectroscopic
3860 GRB050820C 2.6147 VLT/UT2 Spectroscopic
3874 GRB 050824 0.83 ESO Very Large Telescope
(VLT)Spectroscopic

30
3877 GRB050824 0.83 Fynbo et al., GCN3874 Spectroscopic
3914 GRB 050904 5.3 < z < 9.0 SOAR Photometric
3915 GRB 050904 6.0 Swift Spectroscopic
3924 GRB 050904 6.10 ESO VLT-UT1 Photometric
3937 GRB 050904 6.29 Subaru 8.2m telescope Spectroscopic
3948 GRB 050908 3.350 +- 0.005 ESO-VLT UT2 Spectroscopic
3949 GRB 050908 3.3437 +/- 0.0002 Gemini-North telescope Spectroscopic
3971 GRB 050908 z=3.3440 +/-
0.0001Keck/DEIMOS Spectroscopic
4017 GRB 050922C z = 2.17+-0.03 Nordic Optical Telescope
(NOT)Spectroscopic
4029 GRB 050922C 2.198+-0.001 NOT (La Palma) Spectroscopic
4032 GRB050922C 2.198+/-0.001 DOLORES@TNG Spectroscopic
4044 GRB050922C 2.199 VLT/Kueyen Spectroscopic
4085 GRB 051008 0.01 Thueringer Landessternwarte
TautenburgPhotometric
4086 GRB 051008 5.2 +/- 1.3 LAT-OMP Spectroscopic
4124 GRB051021 1.4 HETE Photometric
4145 GRB 051022 No Redshift HETE WXM and FREGATE
TeamsNo Information
4156 GRB 051022 0.8 200" Hale telescope at Palomar
ObservatorySpectroscopic
4165 GRB 051022 0.8 Swift Spectroscopic
4170 GRB 051022 No Redshift Swift Spectroscopic
4186 GRB 051016B 0.9364 Keck I telescope Spectroscopic
4197 GRB051103 0.35 Konus-Wind Spectroscopic
4221 GRB 051109 2.346 Hobby-Eberly Telescope (+
Marcario Low-Resolution
Spectrograph)Spectroscopic
4234 GRB051107 No Redshift INTEGRAL Spectroscopic
4255 GRB 051111 z=1.55 Keck Spectroscopic
4261 GRB051111 1.55 Swift Spectroscopic
4271 GRB 051111 1.54948 Keck observing staff Spectroscopic
4353 GRB051215 0.145 SDSS Spectroscopic
4377 GRB 051211A 0.123 Swift Spectroscopic
4384 GRB 051221 0.5465 Gemini/GMOS Spectroscopic
4388 GRB 051221A No Redshift Swift-BAT No Information
4391 GRB051016 4.1 XMM-Newton Spectroscopic
4403 GRB 051227 No Redshift HETE Spectroscopic
4409 GRB 051227 0.714 Keck II (+ DEIMOS) Spectroscopic
4442 GRB 060105 4.0 +/- 1.3 GCNC Spectroscopic
4454 GRB 060108 No Redshift VLT No Information
4475 GRB 060108 z = 24.2 Gemini-North Spectroscopic
4520 GRB 060115 3.53 ESO-VLT Spectroscopic
4538 GRB 060117 z∼1.3 +- 0.3 No Information Photometric
4539 GRB060108 2.03 BRi’ Faulkes North Telescope
and UKIRT imagingPhotometric
4541 GRB 060116 z∼6 ESO NTT telescope Photometric
4544 GRB 060117 0.45 +/- 0.2 SWIFT-BAT, GCNC 4533;
Cummings et al., GCNC 4538Spectroscopic

31
4545 GRB 060116 6.6 ESO VLT Photometric
4552 GRB060121
(=H4010)-0.46 +/- 0.07 HETE WXM and FREGATE
TeamsSpectroscopic
4583 GRB 060116 z∼6.6 VLT+FORS2 Photometric
4591 GRB 060124 0.82 MDM 2.4m telescope Spectroscopic
4592 GRB 060124 2.297 Keck 10-m telescope Spectroscopic
4593 GRB 060124 2.296 Keck Observatory Spectroscopic
4601 GRB 060124 2.3 1.2 HETE Spectroscopic
4622 GRB 060123 z=1.099 Gemini-N 8-m telescope Spectroscopic
4686 GRB 060206 close to 5 Nordic Optical Telescope Spectroscopic
4692 GRB 060206 4.045 ALFOSC Spectroscopic
4701 GRB 060206 4.048 +/- 0.001 Lick Observatory Spectroscopic
4703 GRB060206 4.05 Subaru Telescope, NAOJ Spectroscopic
4704 GRB 060204B 3.1 1.1 SWIFT-BAT Spectroscopic
4729 GRB060210 z=3.91 Gemini-North Spectroscopic
4769 GRB 060213 3.5 +/- 0.35 LATT-OMP Spectroscopic
4792 GRB 060218 0.0331 MDM Observatory Spectroscopic
4803 GRB060218 0.033 VLT-Antu equipped with
FORS2 and low resolution
grating 300V+GG435.Spectroscopic
4804 GRB060218 0.033 Gemini-south Spectroscopic
4807 GRB 060218 No Redshift 3.6m TNG telescope Spectroscopic
4809 GRB 060218 0.033 6.0m BTA telescope of SAO-
RAS in ZelenchukSpectroscopic
4812 GRB 060218 /
SN2006aj0.168 VLT-FORS2 Spectroscopic
4815 GRB 060223 4.41 Keck-I 10-m telescope Spectroscopic
4856 GRB 060303 No Redshift Konus-Wind, INTEGRAL-SPI-
ACS, RHESSI, Suzaku-WAM,
and Swift-BATNo Information
4863 GRB 060218/SN
2006aj10008.1 km/s and
10032.3 km/sESO’s VLT-Kueyen (UT 2) Spectroscopic
4969 GRB060418 1.49 Magellan Clay telescope Spectroscopic
4974 GRB060418 0.602, 0.655, 0.602,
1.489ESO’s Very Large Telescope
(VLT)Spectroscopic
5002 GRB060418 z=1.4901 +/-
0.0001Magellan Clay telescope Spectroscopic
5004 GRB 170817A 0.679 Swift Spectroscopic
5051 GRB060502A 0.099 +/0.001 Apache Point Observatory Spectroscopic
5052 GRB 060502 z∼1.51 Gemini North telescope Spectroscopic
5067 GRB 060429 0.638 (-0.162,
+0.141)Konus-Wind Spectroscopic
5071 GRB 060502b No Redshift Gemini North Telescope Spectroscopic
5075 GRB 060428B No Redshift TNG+DOLoRes Spectroscopic
5079 GRB 060306 z∼1 SOAR Photometric
5104 GRB 060510b 4.9 Gemini North telescope Spectroscopic
5107 GRB 060510B z=4.9 Swift-BAT Spectroscopic
5123 GRB 060505 0.089 Gemini South Telescope Spectroscopic
5131 GRB 060512 z∼2.7-2.9 ESO Paranal Spectroscopic
5145 GRB 060512 No Redshift 3.6m TNG telescope Spectroscopic

32
5155 GRB060522 5.11 Keck I 10-m telescope Spectroscopic
5158 GRB 060522 5.11 Swift Spectroscopic
5170 GRB 060526 3.21 Magellan/Clay telescope Spectroscopic
5217 GRB 060512 0.4428 Keck I (+LRIS) Spectroscopic
5218 GRB 060604 2.68 Nordic Optical telescope
(+ALFOSC)Spectroscopic
5223 GRB060605 z=3.8 ANU 2.3m telescope Spectroscopic
5226 GRB 060605 3.7 Southern African Large Tele-
scope (SALT)Spectroscopic
5238 GRB 060502B 0.287 Keck I (+LRIS) Spectroscopic
5265 GRB 060614 1.45 +/- 0.85 (90%
error)SWIFT-BAT Spectroscopic
5271 GRB 060614 No Redshift ESO-VLT Spectroscopic
5275 GRB 060614 0.125 Gemini South Spectroscopic
5276 GRB 060614 0.125 VLT+FORS2 Spectroscopic
5283 GRB 050223 0.5915 Magellan/Clay telescope Spectroscopic
5298 GRB 060707 z = 3.43 FORS1 on the Very Large Tele-
scope (VLT)Spectroscopic
5318 GRB060708 No Redshift AEOS 3.63m telescope Spectroscopic
5320 GRB 060714 2.71 Very Large Telescope (VLT) Spectroscopic
5376 GRB 060218 / SN
2006aj0.033 Keck I (+ LRIS) Spectroscopic
5387 GRB051109B 0.080 Keck I (+ LRIS) Spectroscopic
5414 GRB060805B No Redshift Swift Spectroscopic
5418 GRB060805B No Redshift RHESSI Spectroscopic
5435 GRB060807 0.4, 1.0, 1.6, and
4.5NASA Spectroscopic
5437 GRB 060805B 1.3 +/- 0.2 (90%
error)IPN, Hurley et al. Spectroscopic
5452 GRB 060813 2.38 +/- 0.40 SWIFT-BAT Spectroscopic
5468 GRB 060814 2.4 +/- 0.8 Konus-Wind Spectroscopic
5469 GRB060729 z=0.54 XMM-Newton Spectroscopic
5470 GRB 060801 z = 1.131 Gemini North observatory and
GMOS spectrographSpectroscopic
5477 GRB 060813 No Redshift Swift/BAT, Konus-Wind, and
Suzaku/WAMNo Information
5483 GRB 060825 2.0 +/- 0.6 SWIFT-BAT Spectroscopic
5499 GRB 060901 2.0 +/- 0.5 INTEGRAL Spectroscopic
5513 GRB 060904B 0.703 VLT + FORS1 Spectroscopic
5521 GRB 060904A 1.84 0.85 SWIFT-BAT Spectroscopic
5535 GRB 060906 3.685 ESO’s Very Large Telescope
UT2 (Kueyen)Spectroscopic
5538 GRB 060906 No Redshift Swift-BAT Spectroscopic
5555 GRB 060908 2.43 +/- 0.05 Gemini-North/GMOS Spectroscopic
5568 GRB 060912 0.0936 GMOS on Gemini-South Spectroscopic
5597 GRB060923A 0.671 Keck-I + LRIS, Gemini-North
+ NIRISpectroscopic
5617 GRB 060912A 0.937 VLT (Very Large Telescope) Spectroscopic
5626 GRB060926 3.20 ESO’s Very Large Telescope
UT2 (Kueyen)Spectroscopic

33
5637 GRB060926 3.208 VLT Spectroscopic
5644 GRB 060927 2.37 0.75 SWIFT-BAT, Barbier et al. Spectroscopic
5651 GRB 060927 5.6 ESO Very Large Telescope Spectroscopic
5685 GRB060928 No Redshift RHESSI No Information
5690 GRB 060928 2.30 +/- 0.70 Suzaku/WAM Spectroscopic
5698 GRB 061004 z∼3.3 Very Large Telescope (VLT) Spectroscopic
5715 GRB061007 1.261 Magellan Clay telescope at the
Las Campanas ObservatorySpectroscopic
5716 GRB 061007 1.262 Very Large Telescope (VLT) Spectroscopic
5725 GRB061007 No Redshift RHESSI Spectroscopic
5747 GRB 061021 z<2.0 FORS1/VLT Photometric
5782 GRB 061004 z∼3.3 VLT/FORS1, NTT/EMMI Spectroscopic
5809 GRB 061110B 3.44 ESO-VLT UT2 equipped with
FORS1Spectroscopic
5812 GRB 061110A z=0.757 VLT Spectroscopic
5825 GRB061121 No Redshift Keck I Spectroscopic
5826 GRB061121 1.314 Keck I 10m Spectroscopic
5832 GRB 061121 1.314 Keck Observatory Spectroscopic
5838 GRB061121E No Redshift RHESSI No Information
5867 GRB061126 No Redshift RHESSI No Information
5944 GRB 061201 z=∼0.237 Magellan Clay 6.5m Telescope
in ChileSpectroscopic
5946 GRB061210 0.41 10-m Keck I telescope Spectroscopic
5952 GRB 061201 0.111 Magellan Spectroscopic
5958 GRB 061222b z∼3.4 Magellan Spectroscopic
5962 GRB 061222b 3.355 No Information Spectroscopic
5965 GRB 061217 0.827 Magellan Spectroscopic
5982 GRB 050826 0.297+/-0.001 MDM 2.4m telescope Spectroscopic
6010 GRB 070110 2.352 +- 0.001 ESO VLT equipped with
FORS2Spectroscopic
6025 GRB070125E No Redshift RHESSI No Information
6026 GRB070125 No Redshift Milagro TeV observatory No Information
6031 GRB 070125 No Redshift Lick Observatory Spectroscopic
6032 GRB 070125 No Redshift LRIS Spectroscopic
6033 GRB 070125 1.6 0.8 IPN Network and SWIFT-BAT Spectroscopic
6059 GRB 070125 1.3 0.3 KONUS Spectroscopic
6071 GRB 070125 z>1.547 Gemini-Northtelescope+Gem-
ini Multi-Object SpectrographSpectroscopic
6083 GRB 070208 z=1.165 Gemini-Northtelescope+Gem-
ini Multi-Object SpectrographSpectroscopic
6094 GRB 070201 No Redshift INTEGRAL Spectroscopic
6101 GRB 070209 0.314 Gemini-south telescope Spectroscopic
6129 GRB 070220 2.15 0.80 SWIFT-BAT Spectroscopic
6135 GRB070220 No Redshift RHESSI No Information
6166 GRB 060605 3.78 TLS Tautenburg Spectroscopic
6202 GRB 070306 z=1.497 ESO/VLT equipped with
FORS2Spectroscopic
6216 GRB 070318 0.836 ESO/VLT UT2 equipped with
FORS1Spectroscopic

34
6244 GRB 070402 3.36 +/- 1.00 IPN network Spectroscopic
6249 GRB070406 0.11 SDSS Photometric
6283 GRB 070411 2.954 Very Large Telescope (VLT) Spectroscopic
6322 GRB070419A 0.97 10-m Keck I telescope Spectroscopic
6345 GRB 070420 1.56 +/- 0.35 SWIFT-BAT, GCNC 6330 Spectroscopic
6379 GRB 070506 2.31 VLT Spectroscopic
6389 GRB 070508 z > 3.5 Danish 1.54m on La Silla Photometric
6398 GRB 070508 z = 0.82 Very Large Telescope (VLT) Spectroscopic
6399 GRB070508 No Redshift RHESSI No Information
6405 GRB 070508 1.15 +/- 0.15 SWIFT-BAT, Konus-WIND,
Swift-XRT, Suzaku-WAM,
RHESSISpectroscopic
6409 GRB 070512 1.39 +/- 0.20 LATT-OMP Photometric and
Spectroscopic
6448 GRB 070521 z=0.55 Milagro Spectroscopic
6460 GRB 070521 2.28 +/- 0.45 Swift-BAT and Konus Spectroscopic
6465 GRB070412 No Redshift Keck I / LRIS Spectroscopic
6470 GRB 050729 2.4996 Gemini North telescope Spectroscopic
6499 GRB 070611 2.04 VLT Spectroscopic
6504 GRB 070611 z=2.04 Swift Spectroscopic
6529 GRB 070612A z∼0.1 GSFC Photometric
6556 GRB070612A 0.617 Gemini North Spectroscopic
6590 GRB 070628 No Redshift GROND at the 2.2m Max-
PlanckInstitutetelescopeatthe
ESO La Silla observatory in
ChilePhotometric
6592 GRB070626 No Redshift Suzaku-WAM, RHESSI,
MESSENGERSpectroscopic
6631 GRB 070714B 0.692 Swift Spectroscopic
6638 GRB 070714B No Redshift Swift-BAT and Suzaku-WAM No Information
6651 GRB 070721B z=3.626 ESO-VLT Spectroscopic
6663 GRB 060814 0.84 Keck I 10m telescope + LRIS Spectroscopic
6665 GRB070724 0.457 Gemini South Spectroscopic
6687 GRB 070724B 1.10 +/- 0.20 Konus-Wind Spectroscopic
6698 GRB 070802 2.45 ESO VLT + FORS2 Spectroscopic
6741 GRB 070810 z=2.17 Keck I + LRIS Spectroscopic
6756 GRB70810B z=0.0385 Keck I + LRIS Spectroscopic
6759 GRB 061110A 0.758 VLT Spectroscopic
6799 GRB 070917 2.40 +/- 1.20 SWIFT-BAT Spectroscopic
6810 GRB 070911 No Redshift Swift-BAT and Suzaku-WAM Spectroscopic
6819 GRB 070923 No Redshift No Information No Information
6836 GRB 070714B z=.92 Gemini Observatory Spectroscopic
6843 GRB 071003 No Redshift Keck/HIRES Spectroscopic
6850 GRB 071003 1.100 Keck I + LRIS Spectroscopic
6851 GRB 071003 0.937 ESO-Very Large Telescope Spectroscopic
6864 GRB 071010 0.98 Keck/LRIS Spectroscopic
6888 GRB071010B 0.947 Gemini North telescope Spectroscopic
6890 GRB 071010B z=0.947 Keck/LRIS Spectroscopic
6928 GRB 071010B 0.947 Keck II 10m telescope +
DEIMOSSpectroscopic

35
6952 GRB 071020 z = 2.145 Very Large Telescope (VLT) Spectroscopic
6972 GRB 071021 Less than roughly
6.58.2m VLT+FORS2 Photometric
6980 GRB 071021 8.3 PAIRITEL Spectroscopic
6984 GRB 071020 2.142±0.002 6.0m BTA telescope (+SCOR-
PIO) of the SAO-RASSpectroscopic
6985 GRB 071021 z∼5 8.2m VLT+FORS2 Photometric
6992 GRB 071025 z∼5.3 ROTSE-IIIb Photometric
6997 GRB 060602A z = 0.787 Very Large Telescope (VLT) Spectroscopic
7023 GRB 071031 2.692 ESO VLT Spectroscopic
7076 GRB 071112C 0.823 Very Large Telescope (VLT) Spectroscopic
7086 GRB 071112C 0.823 Gemini North telescope Spectroscopic
7088 GRB 071112C 0.8230 +/- 0.0003 VLT Spectroscopic
7117 GRB 071117 z=1.331 ESO VLT Spectroscopic
7124 GRB071122 z=1.14 Gemini North telescope Spectroscopic
7140 GRB 070429B 0.904 Keck I + LRIS Spectroscopic
7152 GRB 071227 0.383 ESO-VLT Spectroscopic
7154 GRB 071227 0.384 Magellan/Baade telescope Spectroscopic
7220 GRB 080122 1.93 +/- 0.25 IPN(Hurleyetal., GCNC7211) Spectroscopic
7283 GRB 080210 No Redshift 2.2m MPI/ESO telescope at La
Silla (Chile)Photometric
7286 GRB 080210 2.641 Very Large Telescope (VLT) Spectroscopic
7290 GRB 080210 2.64 Hobby-Eberly Telescope Spectroscopic
7365 GRB 080307 No Redshift Swift-BAT No Information
7373 GRB 080307 No Redshift Swift-BAT Photometric
7384 GRB 080310 1.7 Magellan/Clay 6.5-m telescope Spectroscopic
7388 GRB 080310 2.4266 UCO/Lick Observatory Spectroscopic
7389 GRB 080310 2.42 Carnegie/Princeton and
CarnegieSpectroscopic
7391 GRB 080310 z=2.43, z=2.28,
and z=1.67ESO’s VLT Kueyen Spectroscopic
7397 GRB 080310 2.4266 Keck/DEIMOS Spectroscopic
7399 GRB 080310 2.43 Swift Spectroscopic
7437 GRB 080319B No Redshift Gemini-South No Information
7444 GRB 080319B 0.937 ESO’s VLT Kueyen Spectroscopic
7451 GRB 080319B 0.937 VLT/UVES Spectroscopic
7453 GRB 080319D No Redshift Swift Spectroscopic
7456 GRB 080319B 0.937 Hobby-Eberly Telescope (R ∼
230)Spectroscopic
7488 GRB 080320 No Redshift Gemini-N/GMOS Spectroscopic
7499 GRB080319B 0.937 Swift Spectroscopic
7511 GRB 080319B 1.61 Swift Spectroscopic
7517 GRB 080319C 1.95 Gemini North Spectroscopic
7534 GRB 080328 No Redshift BTA SAO-RAS Spectroscopic
7544 GRB 080330 z=1.51 NOT Spectroscopic
7547 GRB 080330 1.51 Hobby-Eberly Telescope Spectroscopic
7587 GRB 080411 1.03 ESO VLT Spectroscopic
7601 GRB 080413B 1.10 ESO VLT Spectroscopic
7602 GRB 080413A 2.433 VLT Spectroscopic

36
7605 GRB 080413A z=2.433 Swift Spectroscopic
7608 GRB 080413B 1.1 Swift Spectroscopic
7610 GRB 080413BT No Redshift Swift No Information
7615 GRB 080413B 1.10 Gemini South Spectroscopic
7616 GRB 080413A 2.43 Gemini South telescope Spectroscopic
7630 GRB 080413A 2.433 Swift Spectroscopic
7650 GRB 080430 ∼0.75 2.2m Calar Alto telescope
(+CAFOS)Spectroscopic
7654 GRB 080430 0.767 Hobby-Eberly Telescope (HET) Spectroscopic
7658 GRB 080430 0.767 NOT/La Palma Spectroscopic
7666 GRB 080503 No Redshift Keck I Spectroscopic
7677 GRB 080503 No Redshift Swift-BAT No Information
7685 GRB 080506 No Redshift Swift Spectroscopic
7692 GRB 080506 3.5 Swift-BAT Spectroscopic
7730 GRB 080514B 2.8, 0.9 XRT Photometric,
Spectroscopic
7737 GRB 080516 No Redshift Swift Spectroscopic
7742 GRB 080517 No Redshift Swift Spectroscopic
7757 GRB 080520 1.545 Very Large Telescope (VLT) Spectroscopic
7759 GRB 080514B No Redshift Swift/UVOT Photometric
7760 GRB 080514B 1.8 SuperAGILE/IPN Spectroscopic
7770 GRB 080523 z < 2.5 ESO VLT Spectroscopic
7772 GRB 080523 No Redshift Swift-BAT Photometric
7779 GRB 080523 No Redshift ESO VLT Spectroscopic
7791 GRB 080603 z=1.6880 Gemini-North Spectroscopic
7797 GRB080603B 2.69 NOT telescope Spectroscopic
7805 GRB 080604 1.28 Hobby-Eberly Telescope (HET) Spectroscopic
7815 GRB 080603B 2.69 Hobby-Eberly Telescope (HET) Spectroscopic
7817 GRB 080604 1.28 HET Spectroscopic
7818 GRB 080604 1.416 Gemini-North Spectroscopic
7819 GRB080604 No Redshift HET Spectroscopic
7832 GRB 080605 z = 1.6398 +/-
0.0006Very Large Telescope (VLT) Spectroscopic
7849 GRB 080607 z=3.036 Keck I (+LRIS) Spectroscopic
7870 GRB 080603A 1.688 Keck I / LRIS Spectroscopic
7874 GRB 080514B No Redshift Keck I / LRIS Spectroscopic
7889 GRB 070809 z=0.2187 Keck I (+LRIS) Spectroscopic
7948 GRB 080707 No Redshift 2.2 m ESO/MPI telescope at La
Silla Observatory (Chile)No Information
7949 GRB 080707 1.23 ESO VLT Spectroscopic
7962 GRB 080710 z=0.845 Gemini-South (+GMOS) Spectroscopic
7991 GRB 080721 > 2.3 Swift Photometric
7997 GRB080721 2.602 Italian TNG telescope located
in the Canary Islands (Spain)Spectroscopic
7998 GRB 080721 z∼2.6 Nordic Optical Telescope
(NOT)Spectroscopic
8045 GRB 080727B 5.5-6.5 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)Photometric
8058 GRB 080804 2.2045 VLT Spectroscopic

37
8062 GRB 080805 z = 1.20 ESO Very Large Telescope Spectroscopic
8065 GRB080804 2.20 Gemini-South Spectroscopic
8069 GRB 080804 z=2.2045 Swift UVOT Spectroscopic
8077 GRB 080805 1.505 VLT Spectroscopic
8083 GRB 080810 3.35 Keck/HIRES Spectroscopic
8089 GRB 080810 z∼3.35 Nordic Optical Telescope
(NOT)Spectroscopic
8101 GRB 080810 z = 3.35 Swift/BAT and Konus-Wind Spectroscopic
8137 GRB080825 z>3 ESO Photometric
8143 GRB 080825B 0.83 0.30 SuperAGILE Spectroscopic
8184 GRB 080825C No Redshift Fermi GBM Spectroscopic
8191 GRB 080905B 2.374 ESO VLT Spectroscopic
8198 GRB 080906 No Redshift Swift/UVOT Photometric
8210 GRB 080905B No Redshift Swift/UVOT Spectroscopic
8218 GRB 080913 No Redshift ESO/MPI Photometric
8223 GRB 080913 6.44 +- 0.3 GROND Photometric
8225 GRB080913 z=6.7 VLT (Very Large Telescope) Spectroscopic
8251 GRB080916C No Redshift RHESSI Spectroscopic
8254 GRB080916A 0.689 ESO VLT Spectroscopic
8256 GRB 080913 z = 6.7 Swift-BAT and Konus-Wind Spectroscopic
8267 GRB080913 z∼6.7 Swift/BAT Spectroscopic
8278 GRB 080916C No Redshift Fermi Gamma-ray Space
TelescopeSpectroscopic
8298 GRB080928 z = 1.67 Swift UVOT Photometric
8300 GRB080928 z = 2.49 Gemini-South Spectroscopic
8301 GRB 080928 z=0.736 ESO’s VLT on Cerro Paranal,
ChileSpectroscopic
8304 GRB080928 No Redshift Unknown No Information
8335 GRB 081007 z=0.5295+/-
0.0001GMOS on the 8-m Gemini-
South telescopeSpectroscopic
8345 GRB 081008 0.64 ESO-VLT Spectroscopic
8346 GRB 081008 z = 1.967 Gemini-South Spectroscopic
8350 GRB 081008 1.9685 ESO VLT Spectroscopic
8424 GRB 081028 No Redshift ESO/MPI Photometric
8434 GRB 081028 z=3.038 Magellan/Clay 6.5-m telescope Spectroscopic
8438 GRB 081029 3.8479+/-0.0002 ESO-VLT Spectroscopic
8448 GRB 081029 3.847 Gemini-South Spectroscopic
8515 GRB 081109 No Redshift GROND Photometric
8531 GRB 081118 2.58 ESO-VLT Spectroscopic
8542 GRB 081121 2.512 Magellan/Clay 6.5-m telescope Spectroscopic
8601 GRB081203A ∼2.1 Swift/UVOT Spectroscopic
8662 GRB 081007 0.53 Gemini-S/GMOS Spectroscopic
8700 GRB 081221 0.7 +/- 0.1 Swift-BAT Spectroscopic
8713 GRB 081222 2.77 Gemini-South Spectroscopic
8718 GRB 081222 2.77 Gemini-North/GMOS Spectroscopic
8733 GRB 081226 No Redshift Gemini-South Spectroscopic
8752 GRB 081228 3.8 +- 0.4 GROND Photometric
8766 GRB 090102 z = 1.547 NOT/ALFOSC Spectroscopic

38
8770 GRB 090102 ∼1.8 < z < ∼2.5SwiftUltra-Violet/OpticalTele-
scope (UVOT)Photometric
8773 GRB090102 1.546 Palomar 200" Hale telescope Spectroscopic
8774 GRB 090102 1.55 Hobby-Eberly Telescope (HET) Spectroscopic
8850 GRB 090118 No Redshift GROND No Information
8888 GRB 090205 4.7 +- 0.3 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)Photometric
8889 GRB 090205 4.6 ESO VLT Spectroscopic
8892 GRB 090205 4.6497 0.0025 VLT Spectroscopic
8994 GRB 090313 z=3.375 Gemini-South (GMOS) Spectroscopic
9012 GRB 090313 3.375 ESO VLT Spectroscopic
9015 GRB 090313 3.3721 0.0004 ESO Melipal telescope of the
Paranal ObservatorySpectroscopic
9026 GRB 090323 4.0 +/- 0.3 ESO/MPI Photometric
9028 GRB 090323 z=3.57 Gemini-South (GMOS) Spectroscopic
9035 GRB 090323 3.57 Fermi Observatory Spectroscopic
9053 GRB090328 0.736 Gemini South Telescope Spectroscopic
9057 GRB 090328 0.736 Fermi GBM Spectroscopic
9116 GRB 090406 0.91 +/- 0.10 Konus-Wind and Konus-RF
instrumentSpectroscopic
9136 g0219477-071201 z=0.088 6dFGS Spectroscopic
9151 GRB 090418 1.608 Lick 3-meter telescope Spectroscopic
9156 GRB 090417B 0.345 Gemini-North telescope Spectroscopic
9163 GRB 090418 z=1.608 Swift Spectroscopic
9196 GRB090418A 1.6 Konus-Wind Spectroscopic
9209 GRB 090423 z∼9 Gemini-North telescope Photometric
9215 GRB 090423 8.0+0.5-1.2 MPI/ESO 2.2m Telescope at La
Silla Observatory (Chile)Photometric
9216 GRB 090423 z∼7.6 Italian TNG 3.6m telescope Spectroscopic
9217 GRB 090423 z > 7 GMOS on Gemini-South Spectroscopic
9219 GRB 090423 z∼8.2 VLT Spectroscopic
9222 GRB090423 z∼8.1 NICS/Amici combination Photometric
9227 GRB 090423 ∼7.5 - 8.5 Swift Photometric and
Spectroscopic
9241 GRB 090423 z=8 Swift Photometric and
Spectroscopic
9243 GRB 090424 0.544 GMOS on Gemini-South Spectroscopic
9250 GRB 090424 0.544 William Herschel Telescope
(WHT)Spectroscopic
9264 GRB 090426 2.609 Keck-I Spectroscopic
9269 GRB 090426 2.609 ESO VLT Spectroscopic
9272 GRB 090426 2.609 GSFC Spectroscopic
9307 GRB 090427 1.48 +/- 0.20 LATT-OMP Spectroscopic
9328 GRB 090509 No Redshift GROND Photometric
9353 GRB090510 0.903 VLT/FORS2 Spectroscopic
9365 GRB 090515 No Redshift Swift Spectroscopic
9381 GRB 090516 z=3.9 ESO VLT equipped with
FORS2Spectroscopic
9382 GRB 090516 4.1 +- 0.3 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)Photometric

39
9383 GRB 090516 4.109+/-0.002 ESO VLT Spectroscopic
9385 GRB 090515 No Redshift Swift Spectroscopic
9408 GRB 090519 3.9+0.4-0.7 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)Photometric
9409 GRB 090519 z = 3.85 ESO Very Large Telescope
(VLT)Spectroscopic
9422 GRB 090516A z = 4.109 Konus-Wind Spectroscopic
9457 GRB 090529 2.63 ESO Very Large Telescope Spectroscopic
9475 GRB 090531B No Redshift Swift Spectroscopic
9518 GRB 090618 z = 0.54 Lick Observatory Spectroscopic
9534 GRB 090618 1.42 +/- 0.08 Swift-BAT Spectroscopic
9542 GRB090618 0.54 SAO RAS Spectroscopic
9559 GRB 090621B No Redshift Gemini-North 8-m telescope Spectroscopic
9634 GRB 090709A No Redshift Subaru Telescope No Information.
9639 GRB090709A No Redshift Swift Spectroscopic
9646 GRB 090709A No Redshift Automated Palomar 60-inch
TelescopePhotometric
9673 GRB090715B 3.00 William Herschel Telescope Spectroscopic
9712 GRB090726 2.71 SAO RAS 6-m telescope Spectroscopic
9717 GRB090726 2.71 Swift Spectroscopic
9761 GRB 090809 2.737 +/- 0.002 ESO’s VLT (Paranal Observa-
tory, Chile)Spectroscopic
9771 GRB 090812 2.452 VLT (Paranal observatory) Spectroscopic
9797 GRB 090814A 0.696 FORS2 at the VLT (Paranal
observatory)Spectroscopic
9801 GRB 090814A 0.696 Swift Spectroscopic
9821 GRB090812 2.452 Konus-Wind Spectroscopic
9841 GRB090817 No Redshift 10 m Keck I telescope No Information
9873 GRB 090902B 1.822 Gemini-N Spectroscopic
9942 GRB 090926A 2.1062 ESO-VLT UT2 Spectroscopic
9947 GRB 090926B 1.24 Very Large Telescope (VLT) Spectroscopic
9958 GRB 090927 1.37 Very Large Telescope (VLT) Spectroscopic
10031 GRB 091003 0.8969 Gemini Spectroscopic
10041 GRB091018 z = 0.971 Swift/UVOT Photometric
10042 GRB 091018 0.9710 +/- 0.0003 X-shooter Spectroscopic
10053 GRB 091020 z = 1.71 Nordic Optical Telescope
(NOT)Spectroscopic
10056 GRB 091020 1.71 Swift Spectroscopic
10065 GRB 091024 z=1.092 Gemini North Spectroscopic
10093 GRB091024 z = 1.091 10-m Keck I telescope Spectroscopic
10100 GRB 091029 2.752 Gemini-South Spectroscopic
10158 GRB 091109A 3.5 +/- 0.4 ESO/MPI Photometric and
Spectroscopic
10176 GRB 091117 0.096 Gemini-North Spectroscopic
10202 GRB 091127 0.490 Gemini North Spectroscopic
10233 GRB 091127 0.49034 +/-
0.00018VLT Spectroscopic
10246 GRB 091109A 3.5 +/- 0.4 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)Spectroscopic
10263 GRB 091208B 1.063 Gemini-North Spectroscopic

40
10272 GRB 091208B z=1.0633+/-
0.000310m Keck I telescope Spectroscopic
10286 GRB 091221 No Redshift 2.2m ESO/MPI telescope at La
Silla Observatory (Chile)No Information
10346 GRB 100117A No Redshift Gemini North 8-m telescope Spectroscopic
10350 GRB091109A z=3.076 +/- 0.002 VLT/FORS2 Spectroscopic
10389 GRB100206A 0.41 Keck I Spectroscopic
10399 GRB 100205A No Redshift Keck Photometric
10441 GRB 100219A 4.8 VLT Spectroscopic
10443 GRB100219A z∼4.7 Gemini South Spectroscopic
10466 GRB 100302A 4.813 GMOS (Gemini-North) Spectroscopic
10495 GRB 100316B z=1.180 VLT Spectroscopic
10517 GRB100316D 0.1 Swift-XRT Spectroscopic
10588 GRB100413A z=3.9+1.7-0.7 Swift XRT Spectroscopic
10606 GRB 100414A z= 1.368 Gemini North 8-m telescope Spectroscopic
10620 GRB100418A 0.6235 VLT/X-Shooter Spectroscopic
10624 GRB 100418A 0.624 Gemini North 8-m telescope Spectroscopic
10652 GRB 100423A No Redshift MPI/ESO 2.2 m Telescope at
La Silla Observatory (Chile)Photometric
10684 GRB 100425A 1.755 ESO VLT Spectroscopic
10750 GRB 100513A 4.6 < z < 6.0 Nickel 40-inch Telescope at Lick
ObservatorySpectroscopic
10752 GRB 100513A 4.772 Gemini North Spectroscopic
10754 GRB 100513A 4.8 Swift Spectroscopic
10782 GRB 100518A 4.0+0.3-0.5 MPI/ESO 2.2 m telescope at La
Silla Observatory (Chile)Photometric
10794 GRB 100418A 0.624 Subaru Telescope Spectroscopic
10876 GRB 100621A 0.542 ESO VLT Spectroscopic
10910 GRB 100628A 0.67 (+0.34, -0.12) MPI/ESO 2.2 m Photometric
10935 GRB 100704A No Redshift Swift Spectroscopic
10940 GRB100704A z=3.6ˆ+1.1 _-1.2 Swift XRT Spectroscopic
10946 GRB 100628A z = 0.102 10 m Keck I telescope Spectroscopic
10971 GRB 100724A 1.288 VLT Spectroscopic
10976 GRB 100724A No Redshift Swift Spectroscopic
11113 GRB 100816A No Redshift Swift No Information
11116 GRB 100816A z∼0.245 Gemini-N/GMOS Spectroscopic
11125 GRB 100816A 0.8049 Gran Telescopio Canarias
(GTC)Spectroscopic
11143 GRB 100823A No Redshift ESO VLT Spectroscopic
11164 GRB 100901A 1.408 Gemini-North 8-m telescope Spectroscopic
11171 GRB 100901A 1.408 Swift Spectroscopic
11195 GRB100902A z=4.5ˆ+0.3_-0.2 Swift Spectroscopic
11230 GRB 100906A z=1.727 Gemini-North Telescope on
Mauna KeaSpectroscopic
11317 GRB 100728B z=2.106 ESO VLT equipped with the X-
shooter spectrographSpectroscopic
11457 GRB 101213A z=0.3ˆ+0.1_-0.2 Swift Spectroscopic
11501 GRB101225A z=0.07ˆ+0.13_-
0.04Swift Photometric and
Spectroscopic

41
11507 GRB 101225A No Redshift MMT (6.5-m Multiple Mirror
Telescope)Spectroscopic
11518 GRB 101219A 0.718 Gemini-North Spectroscopic
11522 GRB 101225A z=0.40 10m Keck-I telescope Spectroscopic
11530 GRB 110106A z=0.093 TNG (Tololo New Survey) Spectroscopic
11538 GRB 110106B z=0.618 GMOS on Gemini-North Spectroscopic
11544 GRB 101213A 0.414 GMOS on Gemini-North Spectroscopic
11563 GRB 101225A No Redshift NOT No Information
11568 GRB 101225A No Redshift GTC (10.4m) Spectroscopic
11579 GRB 101219B 0.5519 ESO-VLT UT2 Spectroscopic
11607 GRB 110128A z=2.339 ESO VLT equipped with X-
shooterSpectroscopic
11608 GRB 110128A 0.639 4.2 William Herschel Telescope Spectroscopic
11635 GRB 110205A z∼1.98 3m Lick Telescope Spectroscopic
11638 GRB 110205A z=2.22 FLWO 1.5 m telescope Spectroscopic
11640 GRB 110205A 2.22 NOT Spectroscopic
11644 GRB 110205A 2.22 Swift Spectroscopic
11692 GRB 110205A 2.22 Suzaku Spectroscopic
11708 GRB 110213A z = 1.46 2.3-m Bok telescope Spectroscopic
11736 GRB 110213B 1.083 Gemini South Spectroscopic
11833 GRB 110328A z∼0.35 Gemini/GMOS Spectroscopic
11834 GRB 110328A
/ Swift
J164449.3+5734510.354 GTC (10.4m) Spectroscopic
11853 GRB 110328A
/ Swift
J164449.3+5734510.351 U. Warwick, U.C. Berkeley, U.
LeicesterSpectroscopic
11870 GRB 110402A No Redshift Swift Spectroscopic
11874 GRB 110328A
/ Swift
J164449.3+5734510.342 Keck II Spectroscopic
11879 GRB 110402A No Redshift Konus-Wind Spectroscopic
11888 GRB 110402A Between 2.02 and
2.48ICRANet, ICRA and Rome
University ’La Sapienza’Theoretical Con-
siderations on the
Transparency of
the P-GRB (be-
tween t0 and t0+6
s)
11892 GRB 110402A 0.8 Swift Spectroscopic
11927 GRB 110411A No Redshift Swift Spectroscopic
11950 GRB 110420B No Redshift Swift-BAT Spectroscopic
11977 GRB 110422A 1.77 Italian TNG Spectroscopic
11978 GRB 110422A 1.770+/-0.001 GTC (10.4m) Spectroscopic
11993 GRB 110503A z=1.613 GTC (10.4m) Spectroscopic
11997 GRB 110503A z=1.61 Italian TNG located in La
PalmaSpectroscopic
12164 GRB 110715A 0.82 ESO VLT Spectroscopic
12185 GRB110715A 0.82 Swift Spectroscopic
12193 GRB110721A z=0.382 Gemini-South 8-m telescope Spectroscopic
12202 GRB 110726A 1.036 Gemini-North Spectroscopic

42
12225 GRB 110731A 2.83 Gemini-North (Mauna Kea) Spectroscopic
12234 GRB 110801A z=1.858 GTC (10.4m) Spectroscopic
12241 GRB 110801A 0.341 6-m BTA of SAO RAS Spectroscopic
12258 GRB 110808A z = 1.348 ESO VLT Spectroscopic
12261 GRB 110808A 1.348 Swift Spectroscopic
12276 GRB 110801A z = 1.858 Konus-Wind Spectroscopic
12284 GRB 110818A 3.36 ESO VLT Spectroscopic
12358 GRB 110915A No Redshift Konus-Wind Spectroscopic
12363 GRB 110918A 0.28 +/- .07 IPN (Pal’shin et al., GCNC
12358)Spectroscopic
12368 GRB 110918A 0.982 Gemini-N Spectroscopic
12375 GRB 110918A 0.984+/-0.001 GTC (10.4m) Spectroscopic
12429 GRB 111008A z∼5 GMOS on Gemini-South Spectroscopic
12431 GRB 111008A 4.9898 ESO-VLT Spectroscopic
12477 GRB 111020A No Redshift BAT (Broad Area
Transmission)Spectroscopic
12512 GRB 111026A No Redshift BAT No Information
12537 GRB 111107A 2.893 GMOS-South on the Gemini-
South 8-m telescopeSpectroscopic
12542 GRB 111107A 1.998 ESO-VLT Spectroscopic
12648 GRB 111209A 0.677 VLT-I Spectroscopic
12649 GRB 111209A No Redshift IRTF Spectroscopic
12653 GRB 111210A No Redshift Swift Spectroscopic
12677 GRB 111211A 0.478 ESO-VLT Spectroscopic
12759 GRB111228A 0.714 MMT 6.5-m telescope Spectroscopic
12761 GRB 111228A 0.716 Gemini-South 8-m telescope Spectroscopic
12763 GRB111228A 0.7156 0.0005 Telescopio Nazionale Galileo
(TNG)Spectroscopic
12764 GRB 111228A 0.713 NOT (Nordic Optical
Telescope)Spectroscopic
12770 GRB 111228A 0.71627 +/-
0.00002ESO VLT Spectroscopic
12777 GRB 111229A 1.3805 GMOS-South on the Gemini-
South 8-m telescopeSpectroscopic
12865 GRB 120119A 1.728 Gemini-South 8-m telescope Spectroscopic
12866 GRB 120119A z=1.73 Kast dual spectrometer Spectroscopic
12867 GRB 120119A z=1.728 MMT (Mountain Merged
Telescope)Spectroscopic
12991 GRB 120224A z < 5 VLT Spectroscopic
12999 GRB 120229A No Redshift BAT Team Spectroscopic
13015 GRB 120305A No Redshift Swift Spectroscopic
13051 GRB 120311A No Redshift VLT/X-shooter Spectroscopic
13118 GRB 120326A 1.798 GTC (Gran Telescopio
Canarias)Spectroscopic
13133 GRB 120327A 2.81 Gemini-South 8-m Telescope Spectroscopic
13134 GRB 120327A 2.813 VLT/UT2 Spectroscopic
13141 GRB 120327A 2.81 Swift Spectroscopic
13146 GRB 120327A 2.813 GTC (Gran Telescopio
Canarias)Spectroscopic
13204 GRB 120403A No Redshift Swift Spectroscopic

43
13213 GRB 120404A 1.633 Gemini-North 8-m telescope Spectroscopic
13217 GRB 120404A 2.876 GMOS Spectroscopic
13219 GRB120401A 4.5 0.5 2.2 m MPG/ESO telescope at
La Silla Observatory (Chile)Photometric
13222 GRB 120404A 2.87 Swift Spectroscopic
13227 GRB 120404A ∼2.88 VLT/UT2 Spectroscopic
13251 GRB 120422A 0.28 GMOS (Gemini Observatory) Spectroscopic
13257 GRB 120422A 0.283 ESO VLT Spectroscopic
13277 GRB 120422A 0.28 ESO VLT Spectroscopic
13438 GRB 120711A 3.0 MPG/ESO 2.2m Telescope at
La Silla Observatory (Chile)Photometric
13441 GRB 120711A z=1.405 GMOS-S spectrograph on
Gemini-SouthSpectroscopic
13457 GRB 120712A 4.0 +/- 0.2 MPG/ESO 2.2m Telescope at
La Silla Observatory (Chile)Photometric
13458 GRB 120712A z=4.15 Gemini-South Spectroscopic
13460 GRB 120712A 4.1745 ESO VLT Spectroscopic
13477 GRB 120714B 0.3984 VLT (Paranal, Chile) Spectroscopic
13493 GRB 120716A 2.48 ESO VLT equipped with the
FORS2 spectrographSpectroscopic
13494 GRB 120716A 2.486 VLT (Paranal, Chile) Spectroscopic
13507 GRB120722A 0.9586 ESO VLT Spectroscopic
13512 GRB 120724A 1.48 Gemini-North telescope Spectroscopic
13532 GRB 120729A 0.80 Gemini-North Spectroscopic
13562 GRB 120802A z=3.796 GMOS-N spectrograph on
Gemini-NorthSpectroscopic
13585 GRB 120804A No Redshift Swift Spectroscopic
13614 GRB 120804A No Redshift Konus-Wind Spectroscopic
13628 GRB 120811C z=2.671 GTC Spectroscopic
13632 GRB 120811C z=2.67 NOT (Nordic Optical
Telescope)Spectroscopic
13649 GRB 120815A 2.358 VLT/X-shooter Spectroscopic
13723 GRB 120907A 0.970 OSIRIS/GTC Spectroscopic
13729 GRB 120909A >3 MPG/ESO 2.2m Telescope Photometric
13730 GRB 120909A 3.93 ESO VLT Spectroscopic
13810 GRB 120922A 3.1 +/- 0.2 GROND Photometric
13854 GRB 121011A 0.341±0.002 Swift Spectroscopic
13862 GRB 121011A No Redshift NOT No Information
13890 GRB 121024A 2.298 VLT Spectroscopic
13926 GRB 121027A z < 3.5 2.2 m MPG/ESO telescope Photometric
13929 GRB 121027A z=1.77 Gemini-South Spectroscopic
13930 GRB 121027A z=1.773 VLT/X-shooter spectrograph Spectroscopic
14009 GRB 121128A z=2.20 Gemini-North Spectroscopic
14031 GRB 121201A 3.6 +0.2 -0.3 MPI/ESO 2.2m Telescope at La
Silla Observatory (Chile)Photometric
14035 GRB121201A z=3.385 VLT (Paranal Observatory,
Chile)Spectroscopic
14056 GRB 121209A No Redshift Keck I Spectroscopic
14059 GRB 121211A z=1.023 Keck I telescope (+LRIS) Spectroscopic
14095 GRB 121217A 0.8 (+- 0.1) No Information Photometric

44
14120 GRB 121229A 2.707 ESO VLT Spectroscopic
14207 GRB 130215A 0.597 Lick Observatory Spectroscopic
14225 GRB 120118B 2.943 Keck I Spectroscopic
14264 GRB 100615A 1.398 ESO VLT Spectroscopic
14291 GRB 100424A 2.465 ESO VLT Spectroscopic
14303 GRB 130215A z=0.58+/-0.02 GTC (Gran Telescopio
Canarias)Spectroscopic
14306 GRB 130313A No Redshift Swift Spectroscopic
14342 GRB 130327A z < 3.9 Gemini-North Spectroscopic
14364 GRB 130408A z>3 2.2 m MPG/ESO telescope at
La Silla Observatory (Chile)Photometric
14365 GRB 130408A 3.758 ESO VLT Spectroscopic
14366 GRB 130408A z=3.757 Gemini-S Spectroscopic
14380 GRB 130418A 1.218 10.4m GTC telescope Spectroscopic
14390 GRB130418A z = 1.217 VLT/X-shooter Spectroscopic
14437 GRB 130420A 1.297 10.4m GTC telescope Spectroscopic
14455 GRB 130427A z=0.34 Gemini-North / GMOS Spectroscopic
14478 GRB 130427A z=0.338+/-0.002 NOT (Nordic Optical
Telescope)Spectroscopic
14491 GRB 130427A 0.3399 +/- 0.0002 VLT (Very Large Telescope) Spectroscopic
14493 GRB 130427B 2.78 +/- 0.02 ESO VLT Spectroscopic
14500 GRB 100728A z = 1.567 ESO VLT UT2 Spectroscopic
14516 GRB 130427A 0.49 (+0.12,-0.24) SDSS Photometric
14517 GRB 130427A No Redshift Nayuta 2-m telescope at the
Nishi-Harima Astronomical
ObservatorySpectroscopic
14567 GRB 130505A z=2.27 GMOS-N spectrograph on
Gemini-NSpectroscopic
14573 GRB130505A 2.27 Swift Spectroscopic
14605 GRB 130427A 0.340 Large Binocular Telescope
(LBT)Spectroscopic
14621 GRB 130511A z=1.3033 Gemini-North telescope Spectroscopic
14634 GRB 130514A 3.6 +/- 0.2 MPG/ESO 2.2m Telescope at
La Silla Observatory (Chile)Photometric
14646 GRB 130427A 0.34 10.4m GTC telescope Spectroscopic
14667 GRB 130515A No Redshift VLT/FORS2 Spectroscopic
14669 GRB130427A 0.3393 BTA Spectroscopic
14685 GRB 130518A z=2.49 10.4m GTC Spectroscopic
14687 GRB 130518A z=2.488 Gemini-North / GMOS Spectroscopic
14702 GRB130514A 3.6 Schmidl, Kann, and Greiner
GCN Circ. 14634Photometric
14738 GRB130603AM No Redshift Swift Photometric
14744 GRB 130603B z=0.356 GTC (Gran Telescopio
Canarias)Spectroscopic
14745 GRB 130603B 0.356 Magellan/Baade 6.5-m
telescopeSpectroscopic
14746 GRB 130603B No Redshift Swift No Information
14748 GRB 130603B z=0.356 Gemini-South Spectroscopic
14757 GRB 130603B 0.3564+/-0.0002 ESO VLT Spectroscopic
14762 GRB 130604A 1.06 Gemini North Spectroscopic

45
14790 GRB 130606A 6.1 GTC Spectroscopic
14796 GRB 130606A 5.91 GTC Spectroscopic
14798 GRB 130606A 5.913 MMT Spectroscopic
14804 GRB 130606A 0.30679 Palomar 60-inch telescope Photometric
14815 GRB 130606A 5.91 CTIO, Chile Spectroscopic
14816 GRB 130606A 5.913 ESO VLT Spectroscopic
14848 GRB 130610A 2.092 ESO-VLT Spectroscopic
14882 GRB 130612A z=2.006 VLT/X-shooter Spectroscopic
14886 GRB 130612A 2.006 Swift Spectroscopic
14888 GRB 130609B z=1.3 No Information Spectroscopic
14956 GRB 130701A z=1.155 VLT/X-shooter Spectroscopic
14983 GRB 130702A 0.145 Nordic Optical Telescope
(NOT)Spectroscopic
14985 GRB130702A 0.145 Magellan telescope at Las
CampanasSpectroscopic
14998 GRB130702A 0.341 5 m Palomar Hale telescope Spectroscopic
15000 GRB 130702A 0.342 Italian 3.6m TNG telescope Spectroscopic
15010 GRB 130716A No Redshift Swift Spectroscopic
15126 GRB 130822A No Redshift Swift-BAT No Information
15144 GRB 130831A 0.4791 Gemini-North Spectroscopic
15187 GRB 130907A z=1.238 NOT Spectroscopic
15250 GRB 130925A 0.35 ESO/VLT UT2 Spectroscopic
15307 GRB131004A 0.717 LDSS-3 on the 6.5-m Magellan
Clay TelescopeSpectroscopic
15310 GRB131004A 0.71 4-m TNG Telescope at Canary
islandsSpectroscopic
15320 GRB 130831A 0.4791 VLT/FORS2 Spectroscopic
15407 GRB 131030A z = 1.293 NOT equipped with the Al-
FOSC instrumentSpectroscopic
15408 GRB131030A z=1.295 10.4m GTC telescope Spectroscopic
15450 GRB 131105A 1.686 ESO VLT Spectroscopic
15451 GRB 131103A 0.5955 ESO VLT Spectroscopic
15470 GRB 131108A z=2.40 10.4m GTC telescope Spectroscopic
15493 GRB 131117A 4.18 +/-0.16 2.2m MPG telescope at ESO La
Silla Observatory (Chile)Photometric
15494 GRB 131117A z=4.042 VLT X-shooter spectrograph Spectroscopic
15560 GRB 060614 1.2 Swift-XRT Spectroscopic
15571 GRB 090530 ∼1.266 ESO VLT equipped with the X-
shooter spectrographSpectroscopic
15576 GRB 131202A z∼10 No Information Photometric
15624 GRB 131227A z∼5.3 Gemini-North Spectroscopic
15645 GRB 131231A 0.642 ESO VLT equipped with the X-
shooter spectrographSpectroscopic
15648 GRB 131231A 0.642 Swift-XRT Spectroscopic
15652 GRB 131231A z=0.6439 Gemini South telescope Spectroscopic
15707 GRB 140108A z=0.6 (+/- 0.1) Swift Spectroscopic
15773 GRB 140129A No Redshift Palomar Hale Telescope Spectroscopic
15794 GRB 140206A z∼0.6 Swift/XRT Spectroscopic
15800 GRB 140206A 2.73 Nordic Optical Telescope
(NOT)Spectroscopic

46
15802 GRB 140206A 2.74 Telescopio Nazionale Galileo
(TNG)Spectroscopic
15814 GRB 140209A No Redshift Swift Spectroscopic
15831 GRB 140213A z = 1.2076 ESO VLT Spectroscopic
15890 GRB 140219A 0.12 +/- 0.12 Mondy observatory Photometric
15900 GRB 140301A 1.416 VLT equipped with the X-
shooter spectrographSpectroscopic
15921 GRB 140304A No Redshift NOT Photometric
15922 GRB 140304A 5.39 10.4m GTC (+OSIRIS) Spectroscopic
15924 GRB 140304A 5.28+/-0.02 Nordic Optical Telescope
(NOT)Spectroscopic
15928 GRB 140304A 5.45 (-0.2,+0.1;
90% confidence)1.5m Harold Johnson Tele-
scope at the Observatorio As-
tronómico NacionalPhotometric
15936 GRB 140304A 5.283 10.4m GTC (+OSIRIS) Spectroscopic
15961 GRB 140311A 4.95 Gemini-South Spectroscopic
15964 GRB 140311A z∼5 Nordic Optical Telescope
(NOT)Spectroscopic
15966 GRB 140311A 4.954 Gemini-North Spectroscopic
15988 GRB 140318A z=1.02 WHT (William Herschel
Telescope)Spectroscopic
16079 GRB 111225A 0.297 GTC telescope+OSIRIS Spectroscopic
16125 GRB 140419A 3.956 Gemini-N Spectroscopic
16150 GRB 140423A z=3.26 Gemini-N Spectroscopic
16181 GRB 140428A z=4.7 Keck I 10m telescope Spectroscopic
16191 GRB 140428A 4.8 +-0.3 MPG Garching, La Silla Obser-
vatory (Chile)Photometric
16193 GRB 140430A z < 2.3 Nordic Optical Telescope
(NOT)Spectroscopic
16194 GRB 140430A 1.60 ESO VLT Spectroscopic
16217 GRB 140506A 0.889 VLT/X-shooter Spectroscopic
16228 GRB140508A No Redshift MASTER II robotic telescope Spectroscopic
16229 GRB 140508A z = 1.03 Nordic Optical Telescope
(NOT)Spectroscopic
16231 GRB 140508A z=1.027 William Herschel Telescope
(WHT)Spectroscopic
16241 GRB 140509A About 2.4 Swift Photometric
16244 GRB140508A ∼1.03 2-m Himalayan Chandra Tele-
scope (HCT)Spectroscopic
16269 GRB 140515A z=6.32 GMOS on Gemini-N 8-m
telescopeSpectroscopic
16277 GRB 140515A 6.32 Swift Spectroscopic
16279 GRB 140515A 6.32 GTC (Gran Telescopio
Canarias)Spectroscopic
16280 GRB 140515A 6.5 +/-0.2 MPG telescope at ESO La Silla
Observatory (Chile)Photometric
16301 GRB140518A z=4.707 GMOS on the 8-m Gemini-
North telescopeSpectroscopic
16310 GRB 140512A 0.725 NOT (+ALFOSC) Spectroscopic
16365 GRB 140606B /
Fermi 4327171140.384 Keck II 10 meter telescope Spectroscopic

47
16401 GRB 140614A z=4.233 VLT/X-shooter Spectroscopic
16437 GRB140622A 0.959 VLT X-shooter spectrograph Spectroscopic
16489 GRB 140629A 2.275 +/- 0.003 6-m BTA (+Scorpio-I) Spectroscopic
16493 GRB 140629A 2.29 3.6m TNG telescope Spectroscopic
16505 GRB140703A z=3.14 GTC (Gran Telescopio
Canarias)Spectroscopic
16570 GRB 140710A 0.558 Gemini-North Spectroscopic
16657 GRB140801A z=1.320 GTC (Gran Telescopio
Canarias)Spectroscopic
16663 GRB 140801A 1.319 +/- 0.003 BTA/Scorpio-I Spectroscopic
16671 GRB 140808A z=3.29 10.4m GTC (+OSIRIS) Spectroscopic
16771 GRB 140903A No Redshift Swift Spectroscopic
16774 GRB 140903A z=0.351 Gemini-North telescope Spectroscopic
16797 GRB 140907A 1.21 GTC (+OSIRIS) Spectroscopic
16873 GRB 140930B No Redshift Gemini South Spectroscopic
16881 GRB 141004A No Redshift 3.6m TNG telescope Spectroscopic
16891 GRB141004A No Redshift NOT Spectroscopic
16902 GRB 141004A 0.573 GTC (10.4m) Spectroscopic
16968 GRB 141026A 3.35 OSIRIS at the 10.4m Gran Tele-
scopio CanariasSpectroscopic
16982 GRB141028A z=1.82 Gemini-North Spectroscopic
16983 GRB 141028A z = 2.332 ESO VLT Spectroscopic
17040 GRB 141109A 2.993 ESO’s VLT Spectroscopic
17081 GRB 141121A z=1.47 Keck I 10-meter telescope Spectroscopic
17086 GRB 141121A 1.47 Swift Spectroscopic
17177 GRB 141212A z=0.596 GMOS on the 8 m Gemini-
North telescopeSpectroscopic
17198 GRB141220A 1.3195 GTC (Gran Telescopio
Canarias)Spectroscopic
17228 GRB 141221A z=1.452 Keck II telescope Spectroscopic
17234 GRB 141225A z=0.915 OSIRIS at the 10.4m GTC Spectroscopic
17278 GRB 150101B 0.093 GTC Spectroscopic
17281 GRB
150101B/Swift
J123205.1-1056020.134 ESO VLT Spectroscopic
17358 GRB 150120A 0.460 Gemini-North 8-m telescope Spectroscopic
17420 GRB 150206A z=2.087 VLT/X-shooter Spectroscopic
17523 GRB 150301B z = 1.5169 Very Large Telescope (Paranal
Observatory, Chile)Spectroscopic
17583 GRB 150314A 1.758 GTC (10.4 m) Spectroscopic
17616 GRB 150323A 0.593 Keck I 10m telescope Spectroscopic
17672 GRB 150403A z = 2.06 Very Large Telescope (VLT) Spectroscopic
17697 GRB 150413A 3.2 2.2m CAHA telescope (+
CAFOS)Spectroscopic
17710 GRB 150413A 3.139+/-0.003 Asiago 1.82 m Copernico Tele-
scope +AFOSCSpectroscopic
17731 GRB150413A 3.139 Konus-Wind Spectroscopic
17744 GRB 150423A z=0.456 Keck Observatory Spectroscopic
17755 Swift short-GRB
150423Az = 1.394 VLT equipped with the X-
shooter spectrographSpectroscopic

48
17758 GRB 150424A z = 0.30 10.4m GTC telescope
(+OSIRIS)Spectroscopic
17759 GRB 150424A No Redshift Swift No Information
17822 GRB 150514A 0.807 Very Large Telescope (VLT) Spectroscopic
17832 GRB 150518A 0.256 ESO Very Large Telescope
(VLT)Spectroscopic
18080 GRB 150727A z=0.313 VLT/X-shooter Spectroscopic
18177 GRB 150818A 0.282 10.4m GTC telescope
(+OSIRIS)Spectroscopic
18187 GRB 150821A 0.755 ESO VLT Spectroscopic
18213 GRB 150818A 0.282 10.4 m GTC telescope
(+OSIRIS)Spectroscopic
18273 GRB 150910A z=1.359 3m Shane telescope at Lick
ObservatorySpectroscopic
18274 GRB 150910A 1.360 GTC Spectroscopic
18296 GRB 150906B z∼0.12 (555 Mpc) Advanced-LIGO/VIRGO Spectroscopic
18318 GRB 150915A z = 1.968 ESO VLT Spectroscopic
18426 GRB 151021A z = 2.330 VLT (Paranal Observatory,
Chile)Spectroscopic
18483 GRB 151027A z=0.38+/-0.17 Swift Spectroscopic
18487 GRB 151027A 0.81 Keck/HIRES Spectroscopic
18493 GRB 151027A 0.81 Gao-Mei-Gu of Yunnan
ObservatoriesSpectroscopic
18505 GRB 151027B z=4.063 VLT X-shooter spectrograph Spectroscopic
18507 GRB 151027B 4.3 + 0.3,-0.2 MPG telescope at ESO La Silla
Observatory (Chile)Photometric and
Spectroscopic
18524 GRB 151029A z=1.423 ESO Very Large Telescope UT2 Spectroscopic
18540 GRB 151031A z = 1.167 ESO Very Large Telescope UT2
(Kueyen)Spectroscopic
18598 GRB 151111A 3.5 +/- 0.3 2.2m MPG telescope at ESO La
Silla Observatory (Chile)Photometric
18603 GRB 151112A 4.1 +/- 0.3 GROND Photometric
18696 GRB 151215A 2.59 Nordic Optical Telescope
(NOT)Spectroscopic
18760 GRB 151229A No Redshift Swift-BAT No Information
18831 GRB 160104A 2.80 10.4 m GTC telescope Spectroscopic
18886 GRB 160117B 0.870 ESO’s Very Large Telescope
UT2Spectroscopic
18915 GRB 160117B 0.86 +/- 0.01 TNG Spectroscopic
18925 GRB 160121A 1.960 10.4m GTC telescope
(+OSIRIS)Spectroscopic
18965 GRB 160131A 0.97 3.6m TNG telescope Spectroscopic
18966 GRB 160131A 0.972 GTC Spectroscopic
18969 GRB 160131A 0.97 Xinglong 2.16m Spectroscopic
18982 GRB 160203A 3.52 ESO’s Very Large Telescope
UT2Spectroscopic
19109 GRB 160227A 2.38 NOT Spectroscopic
19121 GRB160228A No Redshift GSFC Spectroscopic
19154 GRB 160303A No Redshift ESO’s Very Large Telescope
(UT2)Spectroscopic
19186 GRB 160228A 1.64 VLT Spectroscopic

49
19192 GRB 160314A 0.726 ESO’s Very Large Telescope
UT2Spectroscopic
19245 GRB 160327A 4.99 GTC (Gran Telescopio
Canarias)Spectroscopic
19274 GRB 160410A z = 1.717 ESO Very Large Telescope UT2
(Kueyen)Spectroscopic
19278 GRB 160410A 1.72 Keck-I Spectroscopic
19350 GRB 160425A 0.555 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
19410 GRB 160509A No Redshift Gemini North Spectroscopic
19419 GRB 160509A z=1.17 Gemini North Spectroscopic
19456 GRB 160521B No Redshift Fermi-GBM and Fermi-LAT Photometric and
Spectroscopic
19473 GRB 160530A No Redshift COSI No Information
19565 GRB160624A z=0.483 Gemini North telescope Spectroscopic
19600 GRB 160625B z = 1.406 ESO VLT UT2 (Kueyen) Spectroscopic
19601 GRB 160625B 1.41 +/- 0.01 TNG Spectroscopic
19656 GRB 160703A No Redshift Swift Photometric
19708 GRB 160623A 0.367 GTC Spectroscopic
19710 GRB 160623A 0.367 Gran Telescopio Canarias
(GTC)Spectroscopic
19773 GRB 160804A 0.736 ESO VLT UT2 (Kueyen) Spectroscopic
19846 GRB 160821B z=0.16 William Herschel Telescope
(WHT)Spectroscopic
19944 GRB160925A No Redshift Liverpool Telescope (La Palma) Spectroscopic
19971 GRB 161001A 0.891 ESO Very Large Telescope Unit
2 (Kueyen)Spectroscopic
20043 GRB 161014A 2.823 10.4m GTC telescope on La
Palma (Spain)Spectroscopic
20050 GRB 161014A 2.823 Swift Spectroscopic
20061 GRB 161014A 2.823 VLT/UT2 Spectroscopic
20069 GRB 161017A 2.013 GTC (10.4m) Spectroscopic
20077 GRB 161017A 2.0127 Gran Telescopio Canarias
(GTC)Spectroscopic
20078 GRB 161017A 2.01 Telescopio Nazionale Galileo
(TNG)Spectroscopic
20104 GRB 161023A 2.708 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
20150 GRB 161108A 1.159 Nordic Optical Telescope
(NOT)Spectroscopic
20168 GRB 161104A 0.793 Magellan telescope Spectroscopic
20180 GRB 161117A 1.549 ESO VLT Kueyen Spectroscopic
20245 GRB 161129A 0.645 GTC (10.4 m) Spectroscopic
20319 GRB 161214B ∼0 Nordic Optical Telescope
(NOT)Spectroscopic
20321 GRB 161219B 0.1475 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
20342 GRB 161219B z = 0.1475 GTC (Gran Telescopio
Canarias)Spectroscopic
20458 GRB 170113A z=1.968 ESO Very Large Telescope UT2
(Kueyen)Spectroscopic

50
20584 GRB 170202A 3.645 10.4m GTC telescope at the
Roque de los Muchachos obser-
vatory (La Palma, Spain)Spectroscopic
20588 GRB 170202A z = 3.65 2.16-m telescope at Xinglong,
Hebei, ChinaSpectroscopic
20590 GRB 170202A 3.645 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
20592 GRB 170202A 3.645 Swift Spectroscopic
20686 GRB 170214A z=2.53 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
20990 GRB 170405A z = 3.510 GTC (10.4m) Spectroscopic
21059 GRB 170428A z=0.454 GTC (10.4-m Gran Telescopio
Canarias)Spectroscopic
21119 GRB 170519A 0.818 GTC (10.4m) Spectroscopic
21177 GRB 170531B 2.366 GTC (10.4 m) Spectroscopic
21197 GRB 170604A z=1.329 GTC (10.4 m) Spectroscopic
21209 GRB 170531B 0.142 William Herschel Telescope
(WHT)Spectroscopic
21240 GRB 170607A 0.557 GTC (10.4m) Spectroscopic
21298 GRB 170705A 2.010 GTC (10.4m) Spectroscopic
21359 GRB 170714A 0.793 GTC (10.4m) Spectroscopic
21418 GRB 170803A No Redshift Swift Spectroscopic
21799 GRB 170903A 0.886 GTC (10.4m) Spectroscopic
21987 GRB 171010A No Redshift 1.3m McGraw-Hill telescope of
the MDM ObservatorySpectroscopic
22002 GRB 171010A 0.33 ESO New Technology Telescope Spectroscopic
22039 GRB 171020A 1.87 Nordic Optical Telescope
(NOT)Spectroscopic
22061 GRB 171027A 9+/-1 1.5m Harold Johnson Telescope
at the Observatorio Astronmico
Nacional in San Pedro MrtirSpectroscopic
22096 GRB 171010A 0.3285 ESO Very Large Telescope
(VLT)Spectroscopic
22180 GRB 171205A 0.0368 ESO VLT/X-shooter
spectrographSpectroscopic
22191 GRB171205A 0.0368 Swift Spectroscopic
22272 GRB 171222A 2.409 10.4m GTC telescope Spectroscopic
22346 GRB 180115A 2.487 10.4m GTC telescope Spectroscopic
22384 GRB 180205A z=1.409 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
22414 GRB 180205A z=1.409 Palomar Observatory Spectroscopic
22484 GRB 180314A 1.445 ESO/VLT UT2 Spectroscopic
22535 GRB 180325A z = 2.25 NOT (Nordic Optical
Telescope)Spectroscopic
22555 GRB 180325A z=2.248 ESO VLT/X-shooter
spectrographSpectroscopic
22567 GRB 180329B z=1.998 ESO VLT/X-shooter
spectrographSpectroscopic
22591 GRB 180404A z = 1.000 ESO Very Large Telescope
(VLT)Spectroscopic
22600 GRB 180402A No Redshift Fermi-GBM Spectroscopic

51
22659 GRB 180418A No Redshift Gemini-North Spectroscopic
22702 GRB 180510B z=1.305 ESO VLT/X-shooter Spectroscopic
22823 GRB 180620B 1.1175 ESO Very Large Telescope
(VLT)Spectroscopic
22825 GRB 180620BD z=1.1175 Konus-Wind Spectroscopic
22840 GRB 180624A 3.75+/-0.10 2.2mMPGtelescopeatESOLa
Silla Observatory (Chile)Photometric
22845 GRB 180624A 2.855 LBT-INAF Spectroscopic
22996 GRB 180720B 0.654 ESO VLT UT2 Spectroscopic
23042 GRB 180720B z=0.654 Waseda CALET Operation
CenterPhotometric
23055 GRB 180728A 0.117 ESO VLT UT2 Spectroscopic
23067 GRB 180728A 0.083 ESO VLT UT2 Spectroscopic
23072 GRB 180727A No Redshift Swift Spectroscopic
23246 GRB 180914B z = 1.096 ESO VLT UT2 (Kueyen) Spectroscopic
23315 GRB 181010A 1.39 ESO-VLT UT2 Spectroscopic
23356 GRB 181020A 2.938 ESO VLT UT2 Spectroscopic
23421 GRB 181110A z=1.505 ESO-VLT UT2 Spectroscopic
23486 GRB 181201A z < 3 and A_V <
0.10 magMPG telescope at ESO La Silla
Observatory (Chile)Photometric and
Spectroscopic
23488 GRB 181201A z = 0.450 ESO VLT UT1 Spectroscopic
23495 GRB 181201A 0.450 INTEGRAL-IBIS/ISGRI and
Insight-HXMT/HESpectroscopic
23537 GRB 181213A No Redshift NOT Spectroscopic
23623 GRB 190106A z∼0.896 Xinglong-2.16m telescope Spectroscopic
23629 GRB 190106A z=1.86 Xinglong-2.16m Spectroscopic
23630 GRB 190106A 1.86 2.4-meter optical telescope at
Gao-Mei-Gu (GMG) station of
Yunnan ObservatoriesSpectroscopic
23632 GRB 190106A 1.859 ESO-VLT UT2 (Kueyen) Spectroscopic
23680 GRB 190114A z=3.3765 ESO Very Large Telescope UT
2 (Kueyen)Spectroscopic
23683 GRB 190114A 3.3765 Swift Spectroscopic
23695 GRB 190114C 0.42 Alfosc Spectroscopic
23708 GRB 190114C 0.4245 +/- 0.0005 10.4m GTC telescope
(+OSIRIS)Spectroscopic
23710 GRB 190114C 0.4250 ESO-VLT UT2 (Kueyen) Spectroscopic
23736 GRB190114C No Redshift Swift No Information
23880 GRB 190206A No Redshift Konus-Wind No Information
23889 GRB 180703A 0.6678 ESO VLT UT4 (Yepun) Spectroscopic
23999 GRB 190324A 1.1715 ESO VLT UT2 (Kueyen) Spectroscopic
24686 GRB 190530A No Redshift NOT Spectroscopic
24835 GRB 190613A 2.78 Palomar Observatory Spectroscopic
24836 GRB 190613B z = 0.92 SEDM on the 60 inch telescope
at Palomar ObservatorySpectroscopic
24850 GRB 190613A 2.78 Palomar Observatory Spectroscopic
24901 GRB 190627A No Redshift Calar Alto Observatory Spectroscopic
24916 GRB 190627A 1.942 ESO VLT UT1 Spectroscopic
25252 GRB 190719C 2.469 ESO VLT UT2 (Kueyen) Spectroscopic
25563 GRB 190829A z < 1.8 Nordic Optical Telescope
(NOT)Spectroscopic

52
25565 GRB 198029A 0.0785 +/- 0.005 10.4m GTC telescope equipped
with OSIRIS in La Palma
(Spain)Spectroscopic
25595 GRB190829A 0.078 Keck I Spectroscopic
25643 ZTF19abvizsw /
AT2019pimNo Redshift Liverpool John Moores
UniversityNo Information
25664 GRB190829A No Redshift Keck I Spectroscopic
25792 GRB 190919B 3.225 ESO VLT UT2 (Kueyen) Spectroscopic
25956 GRB 191004B z = 3.503 ESO VLT UT2 (Kueyen) Spectroscopic
25982 ZTF19abvizsw 1.26 TESS Spectroscopic
25991 GRB 191011A z=1.722 VLT Spectroscopic
26041 GRB 191019A 0.248 Nordic Optical Telescope
(NOT)Spectroscopic
26538 GRB 191221B z=1.19 Swift/UVOT Spectroscopic
26553 GRB 191221B 1.148 ESO VLT UT2 (Kueyen) Spectroscopic
26998 GRB 200205B 1.465 ESO VLT UT2 (Kueyen) Spectroscopic
27114 GRB 200216B No Redshift Very Large Telescope (VLT) Spectroscopic
27737 GRB200514B 0.464 +- 0.08 IPAC, USA Spectroscopic
28038 GRB 200522A 0.554 Gemini North Spectroscopic
28172 GRB 200729A 1.1 Swift Spectroscopic
28295 GRB00093 0.714 +- 0.137 Neil Gehrels Swift Observatory Photometric
28338 GRB 200829A 1.25 + 0.02 Swift Observatory Photometric
28649 GRB 201015A 0.426 GTC Spectroscopic
28650 GRB 201014A 4.56 GTC/OSIRISA Spectroscopic
28656 GRB 201015A z=0.426 CrAO observatory Spectroscopic
28661 GRB 201015A 0.423 Nordic Optical Telescope
(NOT)Spectroscopic
28717 GRB 201020A 2.903 GTC/OSIRIS Spectroscopic
28739 GRB 201021C 1.070 ESO VLT UT3 (Melipal) Spectroscopic
28764 GRB 201024A 0.999 GTC/OSIRISA Spectroscopic
28765 GRB 201020B 0.804 GTC Spectroscopic
28840 GRB 201104B 1.954 ESO VLT UT3 (Melipal) Spectroscopic
28841 ZTF20acozryr No Redshift ZTF No Information
28847 GRB 201103B 1.105 ESO VLT UT3 (Melipal) Spectroscopic
28852 GRB 201104B 1.954 Tien Shan Astronomical
ObservatorySpectroscopic
28883 GRB 201103B 1.105 AZT-33IK Spectroscopic
28886 GRB 201103B 1.105 ZTSh Spectroscopic
29077 GRB 201216C 1.10 ESO Very Large Telescope UT
3 (Melipal)Spectroscopic
29100 GRB 201221A 5.70 ESO VLT UT3 (Melipal) Spectroscopic
29132 GRB 201221D 1.046 OSIRIS at the 10.4m Gran
Telescopio Canarias (Roque de
los Muchachos Observatory, La
Palma, Spain)Spectroscopic
29150 GRB 201221A 5.70 Swift/BAT Spectroscopic
29281 GRB 201227A z > 0.05 Chilescope RC-1000 Photometric
29306 GRB 201015A 0.42 LBT (2x8.4-m) Spectroscopic
29307 ZTF21aaeyldq 2.514 Gran Telescopio Canarias
(GTC)Spectroscopic

53
29319 GRB 210112A z∼2 CAHA Photometric
29320 GRB 200613A 1.2280 GTC telescope (Roque de los
Muchachos Observatory, La
Palma, Spain)Spectroscopic
29411 GRB 210204A 1.466 ESO VLT UT3 (Melipal) Spectroscopic
29432 GRB 210204A 0.876 ESO VLT UT3 (Melipal) Spectroscopic
29450 GRB 210210A 0.715 Gran Telescopio Canarias
(GTC)Spectroscopic
29490 GRB 210204A 0.876 3.6m Devasthal Optical Tele-
scope (DOT)Spectroscopic
29517 GRB 210210A 0.715 Swift/BAT Spectroscopic
29552 GRB 210222B 2.198 Very Large Telescope (VLT) Spectroscopic
29638 GRB 210306A No Redshift Steward Observatory, Univer-
sity of ArizonaSpectroscopic
29655 INTEGRAL GRB
210312Bz = 1.069 Gran Telescopio Canarias
(GTC)Spectroscopic
29673 GRB 200524A z = 1.256 GMOS-N Spectroscopic
29684 GRB 210321A No Redshift Nordic Optical Telescope
(NOT)Spectroscopic
29685 GRB 210321A z = 1.487 GTC (10.4m) Spectroscopic
29711 GRB 210323A 0.42 GBM/Fermi Spectroscopic
29717 GRB 210323A z < 3.1 GTC (10.4m) Spectroscopic
29806 GRB 210411C 2.826 ESO VLT UT3 (Melipal) Spectroscopic
29853 GRB 210420B z = 1.400 GTC (10.4m) Spectroscopic
29944 GRB 210504A 2.077 ESO VLT UT3 (Melipal) Spectroscopic
30037 GRB 210517A 2.486 ESO VLT UT3 (Melipal) Spectroscopic
30164 GRB 210610A 3.54 Xinglong-2.16m Spectroscopic
30182 GRB 210610B z = 1.13 Nordic Optical Telescope
(NOT)Spectroscopic
30194 GRB 210610B 1.1345 GTC Spectroscopic
30200 GRB 210610A z=3.5 Himalayan Chandra TelescopeA
(HCTA)Spectroscopic
30201 GRB 210610B z=1.13 Himalayan Chandra Telescope
(HCT)Spectroscopic
30208 GRB 210610B 1.13 Swift-XRT Spectroscopic
30272 GRB 210619B 1.937 GTC Spectroscopic
30290 GRB 210619B 2.8 Palomar Observatory Spectroscopic
30355 GRB 210702A z=1.1757+/-
0.0093Swift Spectroscopic
30357 GRB 210702A 1.160 ESO VLT UT3 (Melipal) Spectroscopic
30392 GRB 210704A z = 2.34 GTC (10.4 m) Photometric
30452 GRB 210704A z = 0.11 GBM/Fermi Spectroscopic
30487 GRB 210722A z=1.145 GTC (10.4 m) Spectroscopic
30583 GRB 210731A z=1.2525 ESO VLT UT3 Spectroscopic
30692 GRB 210822A z = 1.736 Nordic Optical Telescope
(NOT)Spectroscopic
30771 GRB 210905A 6.318 ESO VLT Melipal Spectroscopic
31053 GRB 211023A z=0.390 SAO RAS Spectroscopic
31056 GRB 211023A 0.36 ICRANet and ICRA-USTC
teamSpectroscopic

54
31075 GRB 211106A 1.012 ESO VLT Spectroscopic
31107 GRB 211023B 0.862 LBT-INAF Spectroscopic
31188 GRB 211207A 2.272 ESO VLT UT3 (Melipal) Spectroscopic
31221 GRB 211211A 0.076 Nordic Optical Telescope
(NOT)Spectroscopic
31230 GRB 211211A 0.080 SPI-ACS/INTEGRAL Spectroscopic
31324 GRB 211227A 0.228 ESO VLT UT3 (Melipal) Spectroscopic
31353 GRB 220101A 4.61 Xinglong-2.16m Spectroscopic
31357 GRB 220101A No Redshift Liverpool Telescope Photometric
31359 GRB 220101A 4.618 Nordic Optical Telescope
(NOT)Spectroscopic
31363 GRB 220101A 4.61 Asiago Copernico and Schmidt
Robotic telescopes (INAF-
OAPadova)Spectroscopic
31423 GRB 220107A 1.246 6m BTA Spectroscopic
31442 GRB 220101A 4.618 ARIES Spectroscopic
31453 GRB 210919A 0.2411 LBT-INAF Spectroscopic
31480 GRB 220117A 4.961 ESO VLT UT3 (Melipal) Spectroscopic
31511 GRB 220117A 4.961 Swift Spectroscopic
31544 GRB 211227A 0.228 Fu et al., GCN 31320 Spectroscopic
31596 GRB 211023A 0.390 LBT (2x8.4-m) Spectroscopic
31606 GRB 220211A 0.53 Fermi Spectroscopic
31619 AT2022cva 0.3 Palomar Observatory (P60) Photometric
31739 GRB 220219B 0.293 LBT (Large Binocular
Telescope)Spectroscopic
31800 GRB 211024B z=1.1137+/-
0.0002ESO’s Very Large Telescope
(VLT)Spectroscopic
31865 GRB 220408A 0.03 Fermi/GBM and Swift Spectroscopic
32079 GRB 220521A 5.6 Nordic Optical Telescope
(NOT)Spectroscopic
32099 GRB 220521A 5.57 Gemini-South Spectroscopic
32105 GRB 220521A z = 5.6 SPI-ACS/INTEGRAL Spectroscopic
32141 GRB 220527A 0.857 NOT Spectroscopic
32144 GRB 220527A 0.857 ESO VLT UT3 (Melipal) Spectroscopic
32291 GRB 220627A 3.084 ESO VLT Spectroscopic
32595 GRB 220611A 2.3608 +/- 0.0002 ESO VLT UT3 (Melipal) Spectroscopic
32647 GRB 221009A No Redshift Xinglong-2.16m Spectroscopic
32648 GRB221009A 0.151 X-shooter at ESO’s UT3 of the
Very Large Telescope (Paranal,
Chile)Spectroscopic
32686 GRB 221009A 0.1505 GTC (Gran Telescopio de
Canarias)Spectroscopic
32694 GRB 221009A 0.1505 NICER Spectroscopic
32742 GRB 221006A
(GCN 32710)z=0.731 10.4m GTC telescope in La
PalmaSpectroscopic
32765 GRB 221009A 0.151 ESO VLT UT3 (Melipal) Spectroscopic
32766 GRB 221009A No Redshift ATCA (Australia Telescope
Compact Array)Spectroscopic
32767 GRB 221009A 0.151 LEIA Spectroscopic
32800 GRB 221009A 0.123 GTC Spectroscopic

55
32850 GRB 221009A 0.123 LBT (Large Binocular
Telescope)Spectroscopic
32928 GRB 221110A 4.06 ESO VLT UT3 (Melipal) Spectroscopic
32930 GRB 221110A 4.06 Swift Spectroscopic
33110 GRB 221226B 2.694 ESO VLT UT3 (Melipal) Spectroscopic
33187 GRB 230116D z = 3.81 BTA SAO RAS telescope
equipped with SCORPIO-2Spectroscopic
33281 GRB 230204B z = 2.142 ESO VLT UT3 (Melipal) Spectroscopic
33521 GRB 230325A z = 1.664 ESO VLT UT3 (Melipal) Spectroscopic
33580 GRB 230307A 3.87 JWST/NIRSpec Spectroscopic
33609 GRB 230328B 0.09 Swift, Fermi, Astrosat, GRBAl-
pha, KonusSpectroscopic
33629 GRB 230414B z = 3.568 OSIRIS at the 10.4m GTC
telescopeSpectroscopic
33707 GRB 230427A 0.1524 +/- 0.0005 MISTRAL Spectroscopic
33741 GRB 230506C No Redshift MISTRAL spectroscopic blue
setting at Observatoire de
Haute-ProvenceSpectroscopic
33747 GRB 230307A 3.87 Hubble Space Telescope Spectroscopic
33749 GRB 230506C 3.9<z< ∼4.0 OHP/T193 Spectroscopic
34122 GRB 230628E No Redshift ESO VLT UT3 (Melipal) Spectroscopic
34409 GRB 230812B 0.360 GTC (10.4m) Spectroscopic
34410 GRB 230812B z=0.360 NOT (2.5m) Spectroscopic
34418 GRB 230812B No Redshift Observatoire de Haute Provence
(OHP)Spectroscopic
34460 GRB230816A No Redshift GROWTH India Telescope
(GIT)Spectroscopic
34485 GRB 230818A z = 2.42 Nordic Optical Telescope
(NOT)Spectroscopic
34497 GRB 230818A 2.42 BTA 6-meter telescope of SAO
RASSpectroscopic
34597 GRB 230812B 0.1234 GTC (Gran Telescopio
Canarias)Spectroscopic
34601 GRB 230827B No Redshift Swift-XRT Spectroscopic
34640 GRB 230205A z=0.429 LBT (2x8.4-m) Spectroscopic
34789 GRB 230827B 0.887 GTC (Gran Telescopio de
Canarias)Spectroscopic
34882 GRB 231024A 0.056 X-shooter Spectroscopic
35007 GRB 231111A z∼1.39 Telescopio Nazionale Galileo
(TNG)Spectroscopic
35009 GRB 231111A 1.179 GTC Spectroscopic
35093 GRB 231117A 0.257 Keck I Observatory Spectroscopic
35098 GRB 231117A 0.257 ESO New Technology Telescope
(NTT)Spectroscopic
35123 GRB 231118A 0.8304 ESO VLT UT3 (Melipal) Spectroscopic
35317 GRB 231210B 3.13 ESO/VLT/UT3 (Melipal) Spectroscopic
35373 GRB 231215A 2.305 GTC (10.4m) Spectroscopic
35375 GRB 231215A No Redshift Nordic Optical Telescope
(NOT)Spectroscopic
35598 GRB 240122A z=3.162 GTC (10.4m) Spectroscopic
35599 GRB 240122A 3.162 ESO/VLT/UT3 (Melipal) Spectroscopic

56
35698 GRB 240205B 0.824 ESO VLT UT3 (Melipal) Spectroscopic
35725 GRB240209A z∼0.207 MISTRAL Spectroscopic
35756 GRB 240218A 6.782 ESO/VLT UT3 (Melipal) Spectroscopic
35832 GRB 240225B 0.946 ESO VLT UT3 (Melipal) Spectroscopic
36085 GRB 240414A z=1.833 MISTRAL instrument in blue
mode using the T193cm tele-
scope at Observatoire de Haute-
Provence (France)Spectroscopic
36087 GRB 240414A 1.833 GTC (Gran Telescopio
Canarias)Spectroscopic
36176 GRB 240419A 5.178 ESO VLT UT2 (Kueyen) Spectroscopic
36385 LXT 240402A /
GRB 240402Bz = 1.551 VLT Spectroscopic
36574 GRB 240529A 2.695 GTC Spectroscopic
36813 GRB 240619A 0.3965 ESO VLT UT3 (Melipal) Spectroscopic
37041 GRB 240805A No Redshift INTEGRAL Spectroscopic
37122 GRB 240809A 1.495 Nordic Optical Telescope
(NOT)Spectroscopic
37129 GRB 240809A 1.494 ESO VLT UT3 (Melipal) Spectroscopic
37293 GRB 240825A 0.659 ESO VLT UT3 (Melipal) Spectroscopic
37294 GRB 240825A 0.659 Swift Spectroscopic
37310 GRB 240825A z∼0.66 TNG (Telescopio Nazionale
Galileo)Spectroscopic
37367 GRB 240825A 0.659 Zeiss-1000 telescope at Tien
Shan Astronomical ObservatoryPhotometric
37369 GRB 240821A No Redshift ESO VLT UT3 Spectroscopic
37467 GRB 240910A z = 1.460 GTC (10.4 m) Spectroscopic
37469 GRB 240912A 1.234 GTC (10.4 m) Spectroscopic
37471 GRB 240912A z = 1.23 Nordic Optical Telescope
(NOT)Spectroscopic
37476 GRB 240912A z = 1.234 Las Cumbres Observatory node
located at McDonald Observa-
tory (Texas)Spectroscopic
37532 GRB 240916A 2.610 ESO VLT UT3 (Melipal) Spectroscopic
37677 GRB 241001A 0.573 ESO VLT UT3 (Melipal) Spectroscopic
37713 GRB 241001A 0.573 VLT Spectroscopic
37731 GRB 240821A 0.238 ESO VLT UT3 (Melipal) Spectroscopic
37866 GRB 241025A z = 4.20 Nordic Optical Telescope
(NOT)Spectroscopic
37867 GRB 241001A 0.57 James Webb Space Telescope
(JWST)Spectroscopic
37888 GRB 241025A 4.2 Swift Spectroscopic
37916 GRB 241026A 2.79 6-m telescope of SAO RAS Spectroscopic
37925 GRB 241026A 2.791 Large Binocular Telescope
(LBT)Spectroscopic
37927 GRB 241025A z=4.20 Konus-Wind Spectroscopic
37931 GRB 240821A 0.238 W. M. Keck Observatory Spectroscopic
37948 GRB 241029A 3.7/3.9 MISTRAL mounted on the 1.93
m telescope at Observatoire du
Haut Provence (OHP, France)Spectroscopic

57
37950 GRB 241029A 1.072 GTC Spectroscopic
37959 GRB 241030A 1.411 W. M. Keck Observatory Spectroscopic
37995 GRB 241030B z >∼3.3 6-m telescope of SAO RAS Spectroscopic
38004 GRB 241030B 2.82 GTC (10.4 m) Spectroscopic
38027 GRB 241030A z∼1.4 GMG 2.4m telescope Spectroscopic
38080 GRB 241010A 0.977 Gran Telescopio Canarias
(GTC)Spectroscopic
38097 GRB 241105A 2.681 ESO VLT UT1 (Antu) Spectroscopic
38123 GRB 241105A 2.702 Swift/BAT Photometric
38167 GRB 241105A 2.681 VLT/FORS2 Spectroscopic
38487 GRB 241209A 1.490 ESO VLT/X-shooter Spectroscopic
38535 GRB 241209A 0.784 GTC Spectroscopic
38637 GRB 241217A /
EP241217b1.879 ESO VLT UT3 (Melipal) Spectroscopic
38646 GRB 241209B No Redshift GTC (10.4 m) Spectroscopic
38704 GRB 241228B 2.674 ESO VLT UT3 (Melipal) Spectroscopic
38759 GRB 250101A 2.49 Xinglong-2.16m Spectroscopic
38769 GRB 250101A 2.49 Las Cumbres Observatory’s ed-
ucation network telescopesSpectroscopic
38773 GRB 250101A 2.49 Swift Spectroscopic
38775 GRB 250101A 2.49 Swift Spectroscopic
38776 GRB 250101A z∼2.481 GMG-2.4m Spectroscopic
38809 GRB 250103A 4.01 GTC Spectroscopic
38810 GRB 250101A 2.49 Las Cumbres Observatory’s ed-
ucation network telescopesSpectroscopic
38814 GRB 250103A 4.01 NOT Spectroscopic
38820 GRB 250103B 1.416 ESO VLT UT3 (Melipal) Spectroscopic
38855 GRB 250108B 0.29 Gemini-North Spectroscopic
38877 GRB 250108B 2.197 Gemini-North Spectroscopic
38934 GRB 250114A 4.732 ESO VLT UT3 (Melipal) Spectroscopic
38943 GRB 250114A z = 4.732 Las Cumbres Observatory
global telescope network
(LCOGT)Spectroscopic
38945 GRB 250114A 4.732 Swift Spectroscopic
39071 GRB 250129A 2.151 ESO VLT UT3 (Melipal) Spectroscopic
39073 GRB 250129A 2.15 Nordic Optical Telescope
(NOT)Spectroscopic
39078 GRB 250129A 2.15 MISTRAL spectroscopic red
settingSpectroscopic
39160 GRB 250205A 3.55 GTC (10.4 m) Spectroscopic
39343 EP250215a/GRB250215A z = 4.61 GTC (Gran Telescopio
Canarias)Spectroscopic
39418 GRB 250221A 0.768 ESO VLT UT3 (Melipal) Spectroscopic
39487 EP250226a / GRB
250226A3.315 ESO VLT UT3 (Melipal) Spectroscopic
39524 GRB 250226A /
EP250226az = 3.315 Las Cumbres Observatory
global telescope network
(LCOGT)Spectroscopic
39647 GRB 250309B 1.898 GTC Spectroscopic
39732 GRB 250314A 7.3 ESO VLT Spectroscopic

58
39743 GRB 250314A 7.27 Gran Telescopio Canarias
(GTC)Photometric
39769 GRB 250317B 3.442, 3.447 GTC Spectroscopic
39852 GRB 250322A z=0.42 Gemini-North telescope in
HawaiiSpectroscopic
39859 GRB 250322A 0.4215±0.0005 ESO VLT UT3 (Melipal) Spectroscopic
39893 GRB 250327B 3.035 Nordic Optical Telescope
(NOT)Spectroscopic
39905 GRB 250327B z=3.038+/-0.005 Observatoire de Haute-
Provence (France)Spectroscopic
39918 GRB 250329A 2.921 ESO/VLT UT3 (Melipal) Spectroscopic
39978 GRB 250327B z=3.035 Konus-Wind Spectroscopic
40022 GRB 250402A 2.823 ESO/VLT UT1 (Antu) Spectroscopic
40024 GRB 250225B 0.950 ESO VLT UT3 (Melipal) Spectroscopic
40084 GRB 250404A /
EP250404az∼2.627 GMG-2.4m Spectroscopic
40162 GRB 250403A z = 1.847 ESO/VLT UT1 (Antu) Spectroscopic
40174 GRB 250419A 0.845 ESO/VLT UT3 (Melipal) Spectroscopic
40175 GRB 250407A 1.36 GTC Spectroscopic
40219 EP250416a / GRB
250416C0.963 Gemini South Spectroscopic
40228 GRB 250424A 0.310 ESO VLT UT3 (Melipal) Spectroscopic
40265 GRB 250427a z=1.519 Keck-I telescope Spectroscopic
40266 GRB 250427A 1.520 ESO VLT UT3 (Melipal) Spectroscopic
40301 GRB 250430A 0.767 ESO VLT UT3 (Melipal) Spectroscopic
40328 GRB 250502A z = 2.163 OSIRIS+/GTC Spectroscopic
40339 GRB 250502A 2.163 Sayan Solar Observatory
(Mondy)Spectroscopic
40367 GRB 250430A 0.767 Swift Spectroscopic
40608 GRB 250521C No Redshift Nordic Optical Telescope
(NOT)Spectroscopic
REFERENCES
Aartsen, M. G., Ackermann, M., Adams, J., et al. 2017,
Journal of Instrumentation, 12, P03012,
doi: 10.1088/1748-0221/12/03/P03012
Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2016,
PhRvL, 116, 061102,
doi: 10.1103/PhysRevLett.116.061102
Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2017,
ApJL, 848, L12, doi: 10.3847/2041-8213/aa91c9
Abbott, B. P., Abbott, R., Abbott, T. D., et al. 2018,
Living Reviews in Relativity, 21, 3,
doi: 10.1007/s41114-018-0012-9
Abdi, H., & Williams, L. J. 2010, WIREs Computational
Statistics, 2, 433, doi: https://doi.org/10.1002/wics.101Ageron, M., Aguilar, J., Al Samarai, I., et al. 2011, Nuclear
Instruments and Methods in Physics Research Section A:
Accelerators, Spectrometers, Detectors and Associated
Equipment, 656, 11,
doi: https://doi.org/10.1016/j.nima.2011.06.103
Aggarwal, C. C., Hinneburg, A., & Keim, D. A. 2001, in
International Conference on Database Theory.
https://api.semanticscholar.org/CorpusID:1648083
Agrawal, M., Hegselmann, S., Lang, H., Kim, Y., & Sontag,
D. 2022, arXiv e-prints, arXiv:2205.12689,
doi: 10.48550/arXiv.2205.12689
Atwood, W. B., Abdo, A. A., Ackermann, M., et al. 2009,
The Astrophysical Journal, 697, 1071,
doi: 10.1088/0004-637X/697/2/1071

59
Ayala Solares, H. A., Coutu, S., Cowen, D. F., et al. 2020,
Astroparticle Physics, 114, 68,
doi: 10.1016/j.astropartphys.2019.06.007
Barthelmy, S. D., Butterworth, P., Cline, T. L., et al. 1995,
Ap&SS, 231, 235, doi: 10.1007/BF00658623
Barthelmy, S. D., Cline, T. L., Gehrels, N., et al. 1994, in
American Institute of Physics Conference Series, Vol.
307, Gamma-Ray Bursts, ed. G. J. Fishman, 643,
doi: 10.1063/1.45819
Barua, S. 2024, arXiv e-prints, arXiv:2404.04442,
doi: 10.48550/arXiv.2404.04442
Bellm, E. C., Kulkarni, S. R., Graham, M. J., et al. 2018,
Publications of the Astronomical Society of the Pacific,
131, 018002
Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U.
1999, in Database Theory — ICDT’99, ed. C. Beeri &
P. Buneman (Berlin, Heidelberg: Springer Berlin
Heidelberg), 217–235
Bird, S., Klein, E., & Loper, E. 2009, Natural language
processing with Python (O’Reilly Media, Inc.)
Blei, D. M., Ng, A. Y., & Jordan, M. I. 2003, J. Mach.
Learn. Res., 3, 993–1022
Brown, T. B., Mann, B., Ryder, N., et al. 2020, arXiv
e-prints, arXiv:2005.14165,
doi: 10.48550/arXiv.2005.14165
Burrows, D. N., Malesani, D., Lien, A. Y., Cenko, S. B., &
Gehrels, N. 2013, GRB Coordinates Network, 15253, 1
Campello, R. J. G. B., Moulavi, D., & Sander, J. 2013, in
Advances in Knowledge Discovery and Data Mining, ed.
J. Pei, V. S. Tseng, L. Cao, H. Motoda, & G. Xu (Berlin,
Heidelberg: Springer Berlin Heidelberg), 160–172
Chase, H. 2022, LangChain,
https://github.com/langchain-ai/langchain
Cherenkov Telescope Array Consortium, Acharya, B. S.,
Agudo, I., et al. 2019, Science with the Cherenkov
Telescope Array (World Scientific), doi: 10.1142/10986
CHIME/FRB Collaboration, Amiri, M., Bandura, K., et al.
2018, ApJ, 863, 48, doi: 10.3847/1538-4357/aad188
Chung, H. W., Hou, L., Longpre, S., et al. 2022, arXiv
e-prints, arXiv:2210.11416,
doi: 10.48550/arXiv.2210.11416
Coughlin, M. W., Bloom, J. S., Nir, G., et al. 2023, The
Astrophysical Journal Supplement Series, 267, 31,
doi: 10.3847/1538-4365/acdee1
Coulter, D. A., Jones, D. O., McGill, P., et al. 2023, arXiv
e-prints, arXiv:2303.02154.
https://arxiv.org/abs/2303.02154
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. 2018,
arXiv e-prints, arXiv:1810.04805,
doi: 10.48550/arXiv.1810.04805Dewdney, P. E., Hall, P. J., Schilizzi, R. T., & Lazio,
T. J. L. W. 2009, IEEE Proceedings, 97, 1482,
doi: 10.1109/JPROC.2009.2021005
Douze, M., Guzhva, A., Deng, C., et al. 2025, IEEE
Transactions on Big Data
Dung Nguyen, T., Ting, Y.-S., Ciucă, I., et al. 2023, arXiv
e-prints, arXiv:2309.06126,
doi: 10.48550/arXiv.2309.06126
Dunn, A., Dagdelen, J., Walker, N., et al. 2022, arXiv
e-prints, arXiv:2212.05238,
doi: 10.48550/arXiv.2212.05238
Fernandez-Soto, A., Mannucci, F., Fugazza, D., et al. 2009,
GRB Coordinates Network, 9222, 1
Ford, H. C., Bartko, F., Bely, P. Y., et al. 1998, in Space
Telescopes and Instruments V, Vol. 3356, SPIE, 234–248
Förster, F., Cabrera-Vives, G., Castillo-Navarrete, E., et al.
2021, The Astronomical Journal, 161, 242,
doi: 10.3847/1538-3881/abe9bc
Galama, T. J., Vreeswijk, P. M., van Paradijs, J., et al.
1998, Nature, 395, 670, doi: 10.1038/27150
Gao, Y., Xiong, Y., Gao, X., et al. 2023, arXiv e-prints,
arXiv:2312.10997, doi: 10.48550/arXiv.2312.10997
Gardner, J. P., Mather, J. C., Clampin, M., et al. 2006,
Space Science Reviews, 123, 485
Gehrels, N., Chincarini, G., Giommi, P., et al. 2004, The
Astrophysical Journal, 611, 1005, doi: 10.1086/422091
Gerganov, G. 2023, GGERGANOV/llama.cpp: LLM
Inference in C/C++„
https://github.com/ggerganov/llama.cpp GitHub
Grezes, F., Blanco-Cuaresma, S., Accomazzi, A., et al.
2021, arXiv e-prints, arXiv:2112.00590,
doi: 10.48550/arXiv.2112.00590
Grootendorst, M. 2022, arXiv e-prints, arXiv:2203.05794,
doi: 10.48550/arXiv.2203.05794
Gunel, B., Du, J., Conneau, A., & Stoyanov, V. 2020, arXiv
e-prints, arXiv:2011.01403,
doi: 10.48550/arXiv.2011.01403
Hadsell, R., Chopra, S., & LeCun, Y. 2006, in 2006 IEEE
Computer Society Conference on Computer Vision and
Pattern Recognition (CVPR’06), Vol. 2, 1735–1742,
doi: 10.1109/CVPR.2006.100
Hajela, A., Margutti, R., Alexander, K. D., et al. 2020,
GRB Coordinates Network, 29041, 1
Harrison, F. A., Craig, W. W., Christensen, F. E., et al.
2013, The Astrophysical Journal, 770, 103,
doi: 10.1088/0004-637X/770/2/103
Hinton, J. 2004, New Astronomy Reviews, 48, 331,
doi: https://doi.org/10.1016/j.newar.2003.12.004
Hobbs, J. R., & Riloff, E. 2010, Handbook of natural
language processing, 15, 16

60
Hofmann, T. 2013, arXiv e-prints, arXiv:1301.6705,
doi: 10.48550/arXiv.1301.6705
Ivezić, Ž., Kahn, S. M., Tyson, J. A., et al. 2019, ApJ, 873,
111, doi: 10.3847/1538-4357/ab042c
Ji, Z., Lee, N., Frieske, R., et al. 2022, arXiv e-prints,
arXiv:2202.03629, doi: 10.48550/arXiv.2202.03629
Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. 2023,
arXiv e-prints, arXiv:2310.06825,
doi: 10.48550/arXiv.2310.06825
Joachims, T. 1997, in Proceedings of the Fourteenth
International Conference on Machine Learning, ICML ’97
(San Francisco, CA, USA: Morgan Kaufmann Publishers
Inc.), 143–151
Jobbins, T. 2023,
TheBloke/Mistral-7B-instruct-V0.2-GGUF„
https://huggingface.co/TheBloke/Mistral-7B-Instruct-
v0.2-GGUF Hugging Face
KM3NeT Collaboration, Aiello, S., Albert, A., et al. 2024,
European Physical Journal C, 84, 885,
doi: 10.1140/epjc/s10052-024-13137-2
Kopper, C., & Blaufuss, E. 2017, GRB Coordinates
Network, 21916, 1
Law, C. J., Sharma, K., Ravi, V., et al. 2024, ApJ, 967, 29,
doi: 10.3847/1538-4357/ad3736
Lewis, P., Perez, E., Piktus, A., et al. 2020, arXiv e-prints,
arXiv:2005.11401, doi: 10.48550/arXiv.2005.11401
Ma, X., Nogueira dos Santos, C., & Arnold, A. O. 2021,
arXiv e-prints, arXiv:2105.12932,
doi: 10.48550/arXiv.2105.12932
Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. 2020,
arXiv e-prints, arXiv:2005.00661,
doi: 10.48550/arXiv.2005.00661
McInnes, L., & Healy, J. 2017, in 2017 IEEE International
Conference on Data Mining Workshops (ICDMW),
33–42, doi: 10.1109/ICDMW.2017.12
McInnes, L., Healy, J., & Melville, J. 2018, arXiv e-prints,
arXiv:1802.03426, doi: 10.48550/arXiv.1802.03426
Meegan, C., Lichti, G., Bhat, P. N., et al. 2009, The
Astrophysical Journal, 702, 791,
doi: 10.1088/0004-637X/702/1/791
Möller, A., et al. 2021, Mon. Not. Roy. Astron. Soc., 501,
3272, doi: 10.1093/mnras/staa3602
Mukund, N., Thakur, S., Abraham, S., et al. 2018, ApJS,
235, 22, doi: 10.3847/1538-4365/aaadb2
Napier, P. J., Thompson, A. R., & Ekers, R. D. 1983,
Proceedings of the IEEE, 71, 1295
Nelson, J. E., Mast, T. S., & Faber, S. M. 1985, Keck
observatory report, 90, 1Ouyang, L., Wu, J., Jiang, X., et al. 2022, in Advances in
Neural Information Processing Systems, ed. S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh,
Vol. 35 (Curran Associates, Inc.), 27730–27744.
https://proceedings.neurips.cc/paper_files/paper/2022/
file/b1efde53be364a73914f58805a001731-Paper-
Conference.pdf
Paul, J., Wei, J., Basa, S., & Zhang, S.-N. 2011, Comptes
Rendus Physique, 12, 298, doi: 10.1016/j.crhy.2011.01.009
Pozanenko, A., Minaev, P., Chelovekov, I., & Grebenev, S.
2020, GRB Coordinates Network, 27627, 1
Reichart, D. E. 1998, ApJL, 495, L99, doi: 10.1086/311222
Reichherzer, P., Schüssler, F., Lefranc, V., et al. 2021,
ApJS, 256, 5, doi: 10.3847/1538-4365/ac1517
Reimers, N., & Gurevych, I. 2019, in Conference on
Empirical Methods in Natural Language Processing.
https://api.semanticscholar.org/CorpusID:201646309
Sia, S., Dalmia, A., & Mielke, S. J. 2020, in Proceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), ed. B. Webber,
T. Cohn, Y. He, & Y. Liu (Online: Association for
Computational Linguistics), 1728–1736,
doi: 10.18653/v1/2020.emnlp-main.135
Sotnikov, V., & Chaikova, A. 2023, Galaxies, 11,
doi: 10.3390/galaxies11030063
Springer, R. 2016, Nuclear and Particle Physics
Proceedings, 279-281, 87,
doi: https://doi.org/10.1016/j.nuclphysbps.2016.10.013
Street, R. A., Bowman, M., Saunders, E. S., & Boroson, T.
2018, in Software and Cyberinfrastructure for Astronomy
V, ed. J. C. Guzman & J. Ibsen, Vol. 10707,
International Society for Optics and Photonics (SPIE),
274 – 284, doi: 10.1117/12.2312293
Swarup, G., Ananthakrishnan, S., Kapahi, V., et al. 1991,
Current science, 60, 95
Touvron, H., Martin, L., Stone, K., et al. 2023, arXiv
e-prints, arXiv:2307.09288,
doi: 10.48550/arXiv.2307.09288
van der Walt, S. J., Crellin-Quick, A., & Bloom, J. S. 2019,
Journal of Open Source Software, 4,
doi: 10.21105/joss.01247
Vaswani, A., Shazeer, N., Parmar, N., et al. 2017, in
Advances in Neural Information Processing Systems, ed.
I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,
R. Fergus, S. Vishwanathan, & R. Garnett, Vol. 30
(Curran Associates, Inc.).
https://proceedings.neurips.cc/paper_files/paper/2017/
file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
Vernet, J., Dekker, H., d’Odorico, S., et al. 2011,
Astronomy & Astrophysics, 536, A105

61
Vernin, J., & Muñoz-Tuñón, C. 1992, Astronomy and
Astrophysics (ISSN 0004-6361), vol. 257, no. 2, p.
811-816., 257, 811
Wei, J., Bosma, M., Zhao, V. Y., et al. 2021, arXiv e-prints,
arXiv:2109.01652, doi: 10.48550/arXiv.2109.01652
Wilson, W. E., Ferris, R. H., Axtens, P., et al. 2011,
Monthly Notices of the Royal Astronomical Society, 416,
832
Wootten, A., & Thompson, A. R. 2009, Proceedings of the
IEEE, 97, 1463
Yuan, W., Dai, L., Feng, H., et al. 2025, Science China
Physics, Mechanics, and Astronomy, 68, 239501,
doi: 10.1007/s11433-024-2600-3Zhao, H., Phung, D., Huynh, V., et al. 2021, arXiv e-prints,
arXiv:2103.00498, doi: 10.48550/arXiv.2103.00498
Zheng, Z., Zhang, O., Borgs, C., Chayes, J. T., & Yaghi,
O. M. 2023, Journal of the American Chemical Society,
145, 18048, doi: 10.1021/jacs.3c05819
Zwart, J., Barker, R., Biddulph, P., et al. 2008, Monthly
Notices of the Royal Astronomical Society, 391, 1545
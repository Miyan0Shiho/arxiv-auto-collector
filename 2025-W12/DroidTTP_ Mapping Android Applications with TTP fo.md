# DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Published**: 2025-03-20 05:38:24

**PDF URL**: [http://arxiv.org/pdf/2503.15866v1](http://arxiv.org/pdf/2503.15866v1)

## Abstract
The widespread adoption of Android devices for sensitive operations like
banking and communication has made them prime targets for cyber threats,
particularly Advanced Persistent Threats (APT) and sophisticated malware
attacks. Traditional malware detection methods rely on binary classification,
failing to provide insights into adversarial Tactics, Techniques, and
Procedures (TTPs). Understanding malware behavior is crucial for enhancing
cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework
mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework.
Our curated dataset explicitly links MITRE TTPs to Android applications. We
developed an automated solution leveraging the Problem Transformation Approach
(PTA) and Large Language Models (LLMs) to map applications to both Tactics and
Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with
prompt engineering and LLM fine-tuning for TTP predictions. Our structured
pipeline includes dataset creation, hyperparameter tuning, data augmentation,
feature selection, model development, and SHAP-based model interpretability.
Among LLMs, Llama achieved the highest performance in Tactic classification
with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in
Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss
of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs,
achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753
for Technique classification, with a Hamming Loss of 0.0054 and 0.0050,
respectively. While XGBoost showed superior performance, the narrow margin
highlights the potential of LLM-based approaches in TTP classification.

## Full Text


<!-- PDF content starts -->

DroidTTP: Mapping Android Applications with TTP for Cyber
Threat Intelligence
Dincy R Arikkata, Vinod P.b,a, Rafidha Rehiman K. A.a, Serena Nicolazzod, Marco Arazzic,
Antonino Nocerac, Mauro Contib
aDepartment of Computer Applications, Cochin University of Science and Technology, Kerala, India
bDepartment of Mathematics, University of Padua, Padua, Italy
cDepartment of Electrical, Computer and Biomedical Engineering, University of Pavia, Pavia, Italy
dDepartment of Computer Science, University of Milan, Milan, Italy
Abstract
The rapid expansion of the Internet of Things (IoT) and mobile technologies has led to an
increased reliance on Android devices for sensitive operations such as banking, online shop-
ping, and communication. While Android remains the dominant mobile operating system, its
widespread adoption has made it a prime target for cyber threats, particularly Advanced Persis-
tent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods
focus primarily on binary classification, failing to provide insights into the Tactics, Techniques,
and Procedures (TTPs) used by adversaries. Understanding how malware operates is essential
for strengthening cybersecurity defenses.
To bridge this gap, we present DroidTTP, a solution designed to map Android malware be-
haviors to TTPs as defined by the MITRE ATT&CK framework. This system empowers security
analysts with deeper insights into attacker methodologies, enabling more e ffective defense strate-
gies. In this work, we curated a novel dataset explicitly designed to link MITRE TTPs to Android
applications. Moreover, we developed an automated solution leveraging the Problem Transfor-
mation Approach (PTA) and Large Language Models (LLMs) to map Android applications to
both Tactics and Techniques. Furthermore, we exploited LLMs for TTP predictions and experi-
mented with two di fferent strategies, specifically Retrieval-Augmented Generation with prompt
engineering and LLM fine-tuning. Our approach follows a structured pipeline, including dataset
creation for Android TTP prediction, hyperparameter tuning, data augmentation, feature selec-
tion, model development for prediction, and interpreting the model decision using SHAP. For
Tactic classification, the Llama model achieved the highest performance among LLMs, with a
Jaccard Similarity score of 0.9583 and a Hamming Loss of 0.0182. Similarly, for Technique
classification, Llama outperformed other LLMs, attaining a Jaccard Similarity score of 0.9348
and a Hamming Loss of 0.0127. However, the Label Powerset XGBoost model demonstrated
superior performance, achieving a Jaccard Similarity score of 0.9893 and a Hamming Loss of
0.0054 for Tactic classification, while attaining a Jaccard Similarity score of 0.9753 and a Ham-
ming Loss of 0.0050 for Technique classification. Although XGBoost slightly outperformed the
fine-tuned LLMs, the performance gap remains narrow, highlighting the potential of LLM-based
approaches in TTP classification.
Keywords: Cyber Threat Intelligence, Tactic Technique and Procedure, Feature Selection,
Large Language Model, Retrieval Augmented Generation
Preprint submitted to arXiv March 21, 2025arXiv:2503.15866v1  [cs.CR]  20 Mar 2025

1. Introduction
In recent years, the digital landscape has undergone significant transformations, with critical
assets such as financial data and personal information increasingly shifting to online platforms
[33]. The ubiquity of mobile and IoT devices in daily life has primarily driven this shift. Unlike
traditional Personal Computers (PCs), these devices not only facilitate daily tasks but also han-
dle a vast amount of sensitive data essential for activities ranging from banking [36] to online
shopping and entertainment [4]. By 2023, Android had solidified its position as the world’s lead-
ing mobile operating system, capturing over 70% of the global market share and surpassing its
competitors1. Its popularity comes from its open-source, cost-e ffective, and user-friendly nature
[19], factors that have contributed to the proliferation of over 2 .61 billion applications2available
through the Google Play Store.
However, this widespread adoption has also made Android devices attractive targets for
cybercriminals, particularly Advanced Persistent Threat (APT) groups [42]. The escalation of
smartphones has allowed users to make mobile payments and store sensitive information such
as login credentials, increasing the risks for both individuals and organizations. As a result,
the past decade has seen a significant rise in mobile attacks, introducing a range of sophisticated
threats that specifically target mobile platforms [16]. The nature of threats emerging from mobile
devices often parallels those a ffecting traditional PCs. These threats are not limited to conven-
tional malware like worms, Trojans, and viruses; they extend to sophisticated cyber attacks that
can compromise security and privacy or even gain complete control over a device. Also, mo-
bile attacks spread malicious content rapidly through technologies such as 5G and Wi-Fi, which
provide seamless Internet connectivity. This connectivity renders mobile devices particularly at-
tractive to cyber criminals, who can deploy a variety of attack vectors, from exploiting mobile
sensors to executing application-level malware. As such, understanding and mitigating these at-
tacks is crucial in safeguarding both personal and organizational assets from the diverse array of
threats targeting mobile platforms.
Existing studies mainly focused on mobile malware detection and classification to address
various security challenges [38, 35]. However, there is a notable lack of research on cyber at-
tribution, which aims to track attack Tactics and Techniques employed by an attacker. Current
Machine Learning (ML) methods typically o ffer classification for mobile users and app security
analysts, indicating whether an application is likely to be malicious or benign. This approach
falls short of fully addressing the complexities of malware detection. For example, suppose that
a mobile application is found to exhibit suspicious behavior, such as accessing sensitive user data
without permission. Simply labeling this app as malicious or belonging to a malicious family,
does not explain how it conducts its attack. Understanding the specific Tactics, like exploiting
a vulnerability in the operating system or using social engineering to trick users, can provide
deeper insights into the threat and inform better defensive strategies. Mapping Tactics, Tech-
niques, and Procedures (TTPs) to malware behavior would empower security professionals to
not only identify threats but also understand the attacker’s motives and methods.
The MITRE ATT&CK3framework provides a structured overview of TTPs associated with
attacks. It serves as a valuable resource for security professionals, providing a common language
and detailed documentation that helps to understand attacker behavior. By categorizing TTPs,
1https://www.statista.com/statistics/921152
2https://www.businessofapps.com/data/google-play-statistics
3https://attack.mitre.org/
2

the framework enables security analysts to better identify, analyze, and respond to threats in a
systematic way. However, there is currently a lack of automated methods to correlate Android
applications with particular TTPs. As mobile malware becomes increasingly sophisticated, at-
tackers are constantly evolving their Tactics, making it essential for security analysts to have
tools that can automatically map application behaviors to TTPs. To bridge this gap, we propose
DroidTTP , a system designed to identify the TTPs associated with Android malware.
To build the DroidTTP framework, we have created a new dataset to train automated models
that map TTPs to Android applications. Subsequently, we implemented a multilabel Tactic and
Technique classification model to identify the TTPs associated with these applications (apps).
Our methodology leverages Problem Transformation Approaches (PTA), particularly useful in
multi-label classification, and Machine Learning Techniques to enhance TTP prediction. Beyond
traditional models, we explore the potential of Large Language Models (LLMs) by implement-
ing Retrieval-Augmented Generation (RAG) with prompt engineering for dynamic and context-
aware predictions. In particular, we aimed to evaluate the e ffectiveness of LLMs in Android
malware detection by analyzing features extracted from application components and mapping
them to attacker behaviors. This approach seeks to determine whether LLMs can infer malicious
intent based on patterns of permissions, activities, receivers, services, and inter-component com-
munications, thereby enhancing behavioral analysis in malware classification. Additionally, we
fine-tuned an LLM to evaluate its e ffectiveness in predicting TTPs based on Android application
features.
The main contributions of our work are as follows.
•We introduce a novel dataset designed explicitly for mapping MITRE TTPs to Android
applications. This dataset enables researchers and security professionals to analyze the
behavior of various Android apps in relation to established cybersecurity frameworks.
•We propose an enhanced feature selection strategy for multi-label classification that builds
upon the SelectKBest4method, a commonly used univariate feature selection method that
identifies the top Kfeatures by applying statistical tests like chi-square or ANOV A F-value.
Our approach involves a two-step process: first, performing label-specific feature selection
to capture the most relevant features for each label, and second, generalizing this selection
across multiple datasets to identify robust features.
•We develop an automated system that maps Android applications to di fferent MITRE Tac-
tic labels. This system utilizes advanced multi-label classification algorithms to analyze
app behavior and associate it with specific Tactics.
•In addition to Tactic mapping, we implement an automated system that assigns Techniques
from the MITRE framework to Android applications.
•We investigate the potential of LLMs for predicting Tactics and Techniques in both Retrieval-
Augmented Generation (RAG) and fine-tuning settings. Our study evaluates various LLMs,
explores di fferent prompt engineering strategies, and analyzes their e ffectiveness in en-
hancing TTP prediction.
4https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.
SelectKBest.html
3

The remainder of this paper is organized as follows. Section 2 discusses related works.
Section 4 explains the methodology we used to develop DroidTTP. Section 7 presents the exper-
imental evaluation and discussion of our findings. Finally, Section 8 concludes with information
and outlines future research directions.
2. Literature Review
Recently, numerous studies have focused on Android malware detection, leveraging various
techniques such as static features, dynamic features, hybrid features, and visualization analysis
[25]. Kim et al. [21] proposed an Android malware detection framework based on static features
to capture multiple aspects of application properties. Their framework comprehensively exam-
ines Android Package Kit (APK) files by extracting seven distinct feature types: Environmental,
Component, Permission, Shared Library Function Opcode, Method API, Method Opcode, and
String data. They used a multimodal Deep Learning (DL) architecture where separate networks
process di fferent feature types independently before combining their outputs into a final network.
The MADAM [26] employs a multi-level analysis strategy, examining features across the pack-
age, user, application, and kernel levels. It combines two parallel classifiers with a behavioral
signature-based detector to strengthen its malware identification capabilities. In [3], Arora et al.
introduced PermPair, a model that detects malware by analyzing relationships between permis-
sion pairs found in the Android application manifest, constructing graph representations. Their
approach demonstrated strong performance with 95 .44% detection accuracy, which exceeded
both state-of-the-art detection methods and commercial anti-malware solutions. Building on
graph-based approaches, the GHGDroid [30] framework takes a broader perspective by creating
comprehensive heterogeneous graphs that map relationships between Android applications and
their interactions with sensitive APIs. By employing multi-layer graph convolution networks,
their approach achieved an F1 score of 99 .17%.
Despite advancements in malware detection, the identification of attack behaviors such as
Tactics and Techniques remains an underexplored area in the current literature. While some
studies, including TTPDrill [18] and rcATT [23], focus on extracting TTPs from threat reports,
their reliance on term frequency-based methods limits the accuracy and completeness of TTP
identification. These methods fail to capture the contextual relationships within CTI reports,
thereby limiting their e ffectiveness. Shin et al. [31] propose a TTP extraction approach based on
GloVe embedding, which uses a co-occurrence matrix to capture semantic relationships. Their
results show a correlation of up to 0 .96 between the co-occurrence matrix and the embedding
performance for each Tactic.
Recent research also investigates mapping TTPs to network tra ffic. In [29, 28], Sharma et al.
introduced an ML-based approach for malware detection in network tra ffic by utilizing adver-
sarial behavior knowledge represented as TTP. They extracted TTP features from network tra ffic
and integrated them into ML models, demonstrating superior performance compared to state-
of-the-art malware detection methods. In the Android security domain, Fairbanks et al. [14]
identified attack Tactics by analyzing control flow graphs in malware samples. Their methodol-
ogy employed the Inferential SIR-GN model for node representation and utilized Random Forest
classification, with SHAP analysis providing insight into feature importance and subgraph rele-
vance. This approach achieved a 92 .7% F1 score in Tactic detection. However, the scope of their
research was limited to seven specific Tactics and did not extend to the broader range of attack
Techniques used by malicious actors. Despite these advances, there has not been a comprehen-
4

Table 1: Summary of the acronyms used in the paper
Symbol Description
ANN Approximate Nearest Neighbor
APK Android Package Kit
APT Advanced Persistent Threat
DL Deep Learning
DT Decision Trees
ICS Industrial Control System
IoC Indicator of Compromise
IoT Internet of Things
LLM Large Language Model
ML Machine Learning
MLP Multi-Layer Perceptron
PC Personal Computer
PTA Problem Transformation Approach
RAG Retrieval-Augmented Generation
RF Random Forest
SHAP SHapley Additive exPlanations
SLM Small Language Model
TTPs Tactics, Techniques, and Procedures
sive study that maps both attack Tactics and Techniques to Android applications, and no dataset
exists that captures these relationships in detail.
Our research addresses these critical gaps through several key innovations. First, we com-
piled a benchmark dataset that captures multiple Tactics and Techniques related to Android appli-
cations. This dataset serves as a foundation for our multi-label classification system, which maps
applications to their corresponding MITRE Tactics and Techniques. This work also introduces
an adaptive feature selection framework that optimizes feature sets based on the characteristics
of different training datasets, leading to enhanced model performance and broader applicability.
To the best of our knowledge, while LLMs have been widely applied in various cybersecurity
tasks [12, 41, 27, 17], no existing study has specifically explored their use in mapping Android
applications to TTPs. In this work, we investigate the potential of LLMs by leveraging RAG and
fine-tuning Techniques for TTP prediction.
3. Background
In this section, we provide the necessary background information for our study, covering
Tactics, Techniques, and Procedures, Adversarial Tactics, Techniques, Common Knowledge
(ATT&CK), and Retrieval-Augmented Generation.
Table 1 summarizes the acronyms used in this paper.
3.1. Tactics, Techniques, and Procedures
In cybersecurity, understanding the methodologies, intentions, and actions of adversaries is
fundamental to building an e ffective defense framework. An e ffective way to analyze and classify
adversarial activities is by understanding TTPs. TTPs describe the behavior of attackers that
provides both a strategic overview and tactical insights into their operations. Although Indicators
of Compromise (IoCs) such as IP addresses, file hashes, or domain names are invaluable for
real-time threat detection, they typically lack the context necessary to interpret the intentions and
behavior of attackers. TTPs, on the other hand, serve as higher-level IoCs that provide a more
comprehensive view of how an attack unfolds. The three components of TTPs are:
5

•Tactics: Represents the high-level goals or objectives that an adversary aims to achieve
during an attack. In the context of Android security, these objectives could range from
data theft and system infiltration to denial of service.
•Techniques: Describe how the attacker will perform the Tactic. They are the specific
methods or actions used to achieve the objective, providing more granular insights into the
attacker’s approach. In Android security, several Techniques can be employed to execute
a particular Tactic. For example, if the attacker’s Tactic is to gain access to the system, a
common Technique could be phishing .
•Procedures: Procedures are the exact steps or actions that provide a step-by-step descrip-
tion of how the attacker operationalizes their methods to achieve their objectives. For
example, consider the procedure steps where the adversary utilizes phishing as a Tech-
nique: (i) The attacker crafts a phishing email disguised as an urgent message from the
victim’s Android banking app, stating that the account has been locked and needs to be
reset. (ii) The email contains a link to a fake login page, which is designed to look iden-
tical to the o fficial banking app login screen. (iii) The victim, believing the message is
legitimate, clicks the link and enters their login credentials into the fake page. (iv) The
attacker then captures the credentials and uses them to access the victim’s bank account,
potentially conducting fraudulent transactions.
3.2. Adversarial Tactics, Techniques, and Common Knowledge
The MITRE Adversarial Tactics, Techniques, and Common Knowledge (ATT&CK) frame-
work is a comprehensive knowledge base that catalogs adversary behaviors. Developed by
MITRE, this framework provides a structured approach to understanding how attackers operate.
It plays a vital role in threat intelligence, incident response, red teaming, and adversary emu-
lation by mapping detected behaviors to known attack Techniques. The ATT&CK framework
is organized into multiple matrices, each tailored to specific environments and attack vectors.
These include the Enterprise ,Mobile (Android and iOS), and Industrial Control Systems (ICS)
matrices, which help analysts focus on di fferent operating systems or environments.
Each matrix is further broken down into Tactics (e.g., Initial Access, Execution, Exfiltration,
etc.). Techniques are then listed under each Tactic, providing detailed descriptions of the methods
adversaries may employ to achieve their goals (e.g., phishing, credential dumping, or lateral
movement). For example, consider the Initial Access (TA0027) Tactic in an Android attack
scenario. A common technique used by attackers to achieve this goal is Phishing (T1660)5,
where social engineering is used to trick victims into giving the attacker access to their devices.
The attack progresses through a sequence of meticulously planned steps, further explained in the
procedure examples and referenced within the specific Technique page.
Moreover, Techniques can have sub-techniques which further details specific variations of the
Technique. For example, the Input Capture (T1417)6technique focuses on how adversaries
intercept user input on a mobile device. This Technique is divided into sub-techniques that
outline distinct methods of input interception: Keylogging (T1417.001) , and GUI Input
Capture (T1417.002) . A single Technique can belong to multiple Tactics. For example, the
Input Capture Technique is used in both Credential Access and Collection Tactics.
5https://attack.mitre.org/techniques/T1660
6https://attack.mitre.org/techniques/T141
6

Figure 1: Architecture of Retrieval-Augmented Generation
3.3. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [24] represents a fusion of information retrieval and
language generation technologies. This approach enhances AI language models by connecting
them with external knowledge sources that enable more informed and accurate responses. Unlike
traditional language models that rely solely on their training data, RAG actively draws upon
current information when generating responses.
The RAG workflow is illustrated in Figure 1. At its core, RAG functions through a two-stage
process. The first stage involves information retrieval , where the system searches through exter-
nal knowledge sources to find content relevant to the current query. This works by converting
both the user’s question and the available reference documents into mathematical representa-
tions called embeddings. It then scans similarly encoded documents in its knowledge base, using
mathematical comparison of embeddings to identify the most semantically relevant information.
The second stage leverages an LLM’s generative capabilities. Rather than relying solely on
its pre-trained knowledge, the model receives both the user’s query and the retrieved relevant
information (context) from the vector database. This allows the LLM to craft responses that
incorporate specific, factual details from the retrieved sources while maintaining natural language
fluency. The result is more accurate and contextually appropriate than what could be achieved
by either retrieval or generation alone. This combination of dynamic knowledge access and
sophisticated language generation represents a significant leap forward in AI’s ability to provide
precise, contextually enriched responses.
4. Data Collection and Processing
In this section, we present the first steps of the proposed methodology used in this study. The
architecture of the proposed system is illustrated in Figure 2. The initial phase involves collecting
data for experimentation. Following data collection, key features are identified using a feature
selection mechanism.
4.1. Data Collection
In this phase, we gather data to generate a model that maps Android applications to MITRE
Tactics and Techniques. To the best of our knowledge, no existing dataset provides MITRE
7

Figure 2: Architecture of Android Tactic and Technique Prediction Model
Tactic and Technique classification for Android apps. Therefore, we initiated the data collection
process by compiling hashes of Android applications. This process involved manually analyzing
the Tactic and Technique information outlined in the MITRE ATT&CK Mobile Android Matrix
7.
As outlined in Section 3, the MITRE framework categorizes adversarial behavior into dis-
tinct, high-level Tactics, each of which encompasses multiple specific Techniques. The An-
droid matrix covers Tactics, including Initial Access, Execution, Persistence, Privilege Escala-
tion, Defense Evasion, Credential Access, Discovery, Lateral Movement, Collection, Command
and Control, Exfiltration, andImpact . Moreover, each Technique is accompanied by technical
procedures, and we used these procedure details to collect the Android applications. The data
collection process involves the following three steps:
Figure 3: VirusTotal report confirming the hash matches an Android app.
1.Procedure References Analysis. The first step involved a detailed examination of the
procedure references associated with each Technique in the MITRE ATT&CK Mobile
7https://attack.mitre.org/matrices/mobile/android
8

Android Matrix. These references, contributed by security practitioners, outline how at-
tackers exploit specific vulnerabilities to carry out malicious activities on Android devices.
They provide critical technical details such as permissions requested by the applications,
inter-process communication intents, app activities, and additional IoCs, such as suspi-
cious URLs or IP addresses. For example, the Audio Capture (T1429)8Technique,
under the Collection Tactic, is linked to 49 references detailing scenarios in which mal-
ware records audio without user consent, abuses sensitive permissions, or exfiltrates cap-
tured data to external servers. Many of these references also provide cryptographic hashes
(e.g., SHA1) of malicious files. For instance, one reference from Lookout9provides SHA1
hashes of BouldSpy samples along with their associated Command and Control details. By
systematically analyzing these procedure references, we can compile a set of hashes. We
manually analyzed each reference link provided in the technical details and collected the
corresponding hashes. The number of hashes collected for this study is detailed in Section
7.2.
2.Verification Using VirusTotal. The next step involved verifying the collected hashes
using VirusTotal10, a widely recognized threat intelligence platform that provides compre-
hensive security reports on files, URLs, and IPs. VirusTotal aggregates data from multiple
antivirus engines and other security tools, making it a powerful platform for threat analy-
sis. It scans each hash to determine the file type (e.g., APK, Windows, etc.) and assesses
whether it is malicious. For example, we verified a hash extracted from a Lookout Threat
Intelligence reference on BouldSpy spyware using VirusTotal11. The report confirmed that
the hash was compatible with an Android app, as shown in Figure 3. Using VirusTotal, we
eliminated irrelevant data, such as hashes that do not correspond to Android applications.
3.Labeling Techniques and Tactics. The final step of the data collection process involved
labeling each verified hash with one or more MITRE ATT&CK Techniques and their cor-
responding Tactics. This labeling is based on the procedure references analyzed in the first
phase. For each verified hash, we identified the associated Technique(s) and Tactic(s) it
represented. For example, suppose that a hash is included in the procedure reference for
theAudio Capture Technique. We label it with the Audio Capture Technique ID and asso-
ciate it with the corresponding Tactic, which in this case is Collection . Moreover, since an
app can exhibit multiple behaviors corresponding to di fferent Techniques and Tactics, we
used a multi-labeling approach. For instance, an app exploiting the Call Control Technique
could be labeled under the Collection Tactic (due to data gathering), Command and Con-
trolTactic (for communication with a C2 server), and Impact Tactic (because of potential
disruption to user communication). We applied this approach to label each app based on
all the relevant Techniques and Tactics it exhibited. As a result, we created a dataset that
includes Android app hashes along with their associated Technique IDs and corresponding
Tactics.
4.2. Feature Extraction
After collecting the hashes of Android apps, the next step involves extracting relevant fea-
tures for a supervised learning classification task. In this study, we focus on extracting Android
8https://attack.mitre.org/techniques/T1429
9https://www.lookout.com/threat-intelligence/article/iranian-spyware-bouldspy
10https://www.virustotal.com/gui/home/search
11https://www.virustotal.com/gui/file/f919be6a1920b2c206c62ae03ac69fad9955564618874245e91cd0aed051ed78/
details
9

app features that are commonly referenced in the MITRE procedure details. For instance, the
procedure reference for the T1398 - Boot or Logon Initialization Scripts Technique
under the Persistence Tactic12examines the permissions requested by the malware, providing
insight into its persistence mechanisms. Android.Oldboot bootkit . Similarly, static features such
as Activities, Intents, Services, Receivers, Strings, etc., are also detailed in the procedure refer-
ence documents. Based on the frequency of these static features mentioned in the references, we
extracted the following features.
•Permissions : These are access rights requested by the app to perform specific actions. For
example, navigation apps and social media platforms often request ACCESS_FINE_LOCATION
permission to access the device’s GPS location for location-based features.
•Activities : Activities represent individual screens or interactions within the app’s user in-
terface. Each activity defines a specific user experience. A login screen ( LoginActivity )
enables users to enter their credentials for authentication.
•Services : These are background processes that operate independently of the user interface.
Services can handle long-running tasks, such as playing music or fetching data from the
Internet. For example, the MusicPlayerService in music streaming apps like Spotify
plays music in the background, even when the app is minimized or the screen is o ff.
•Receivers : Receivers are Android components designed to listen for system events or
inter-application broadcasts, enabling applications to dynamically respond to environmen-
tal changes. They facilitate event-driven execution, allowing apps to react to network state
transitions, incoming messages, or system-level triggers. For instance, BatteryReceiver
detects battery level fluctuations, enabling applications to implement power-saving mech-
anisms or monitor energy consumption e fficiently.
•Intents : Intents facilitate communication between application components and the system,
enabling seamless interaction and execution of predefined actions. Intent actions define
the specific operations an application can perform, while intent categories provide con-
textual information to refine the behavior. For example, the ‘ ACTION_VIEW ‘ intent with a
URL parameter triggers a web browser to open a webpage. Malware often exploits this
mechanism to redirect users to malicious websites, leveraging intent-based redirection as
a vector for phishing, drive-by downloads, or credential theft.
To extract the features, we performed a static analysis. Static analysis examines the application’s
code without executing it, thus avoiding the risks associated with running potentially malicious
software. Static analysis typically involves reverse engineering Android Package (APK) files,
which are compressed archives containing essential components such as the manifest file and
other resources. The AndroidManifest.xml file within the APK is a key component for static
analysis, as it contains valuable information about the app’s structure and capabilities. To au-
tomate the feature extraction process, we used VirusTotal, which provides static details about
Android APKs through Androguard13. Androguard o ffers comprehensive information about the
app’s permissions, activities, services, etc. To collect features, we first created a VirusTotal ac-
count and developed a script to query the platform for each app’s hash. VirusTotal responds with
12https://thehackernews.com/2014/01/first-widely-distributed-android.html
13https://github.com/androguard/androguard
10

data in JSON format, which our script processes to extract static features. Finally, we converted
these features into binary feature vectors suitable for ML algorithms.
We denote this feature vector as F=(f1,f2,..., fs), where sis the number of unique static
features. These unique features are determined by consolidating all features from the various
hashes and identifying the distinct ones. For each specific hash, the features are encoded as:
fi=1 if the i-th feature is present
0 otherwise
4.3. Feature Selection
This phase addresses the crucial ML aspect of dimensionality reduction. High-dimensional
datasets can lead to overfitting, in which the model performs well on the training data but strug-
gles with unseen data. In addition, processing a large number of features can be computationally
expensive. Thus, e ffective feature selection is essential to improve the accuracy of the model.
SelectKBest is a widely used [40, 2] feature selection method that identifies the top kfeatures
based on their statistical significance in relation to the target variable. This approach evaluates
each feature individually and ranks attributes according to a chosen statistical test, such as the
chi-squared test or ANOV A F-value. Although this approach is e ffective in many cases, how-
ever, it has limitations in multi-label classification tasks. Specifically, features relevant to one
label may be overlooked when evaluated independently, as this approach does not account for
inter-dependencies among labels. Moreover, feature selection is often carried out on a specific
training set determined by a random seed. However, this practice can lead to significant variabil-
ity in results. Features selected from one dataset may not perform optimally when applied to a
different dataset generated with a di fferent random seed. This inconsistency arises because the
selected features are solely based on the characteristics of a single training dataset, which may
not be representative of other datasets. To address these challenges, we propose a novel feature
selection strategy that leverages SelectKBest but enhances it for multi-label classification tasks.
The proposed mechanism follows a two-step process that ensures consistent feature selection
while accounting for label interdependencies to improve classification accuracy, namely:
1.Label Specific Feature Selection. The first step focuses on identifying features that are
most relevant to each label within a specific dataset. To achieve this, we used the chi-square
test in combination with the SelectKBest method, which ranks top mfeatures according to
their association with the target label. To calculate the chi-square value, we computed the
observed frequency and the expected frequency. The Observed Frequency Oijquantifies
the number of occurrences in which feature fiappears alongside label yjin the dataset. In
contrast, the Expected Frequency Eijrepresents the anticipated occurrence of figiven yj,
assuming that there is no statistical dependency between them. Expected Frequency ( Ei j)
is calculated by the Equation 1:
Ei j=Tfi×Tyj
N(1)
Where Tfiis the total occurrences of feature fi,Tyjis the total occurrences of label yj, and
Nis the total number of instances in the dataset. The chi-square statistic is then computed
for each feature-label pair using the following Equation 2:
χ2=X
i,j(Oi j−Ei j)2
Ei j(2)
11

After calculating the chi-squared statistic for each feature-label pair, the SelectKBest method
ranks the features based on their chi-squared values. Features with higher chi-squared
scores are considered more statistically significant in their relationship with the label and
are therefore selected for further use. For each label, the top mfeatures with the highest
chi-squared scores are selected.
2.Generalizable Feature Subset. In the second step, we apply the same procedure as in the
first step to zdifferent datasets, each generated using a di fferent random seed. We analyze
each dataset separately to identify the most important features. Finally, we combine the top
features from all zdatasets and extract only the unique ones. The comprehensive process
of feature selection is presented in Algorithm 1.
Algorithm 1: Feature Selection Process
Input:{D1,D2,...,Dz}:zdatasets generated by di fferent random seeds, L: Set of
labels, m: Number of top features to select for each label
Output:FG: Generalizable subset of unique top features across all zdatasets
1InitializeFG←∅
2fori←1tozdo
3 InitializeF(i)
L←∅
4 foreach label l∈Ldo
5 foreach feature f jinDido
6 χ2=P
i,j(Oi j−Ei j)2
Ei j;
7 Storeχ2score for feature fj;
8 Rank features fjby theirχ2score in descending order;
9 Select top mfeatures f1,f2,..., fmfrom ranked list;
10F(i)
L=F(i)
L+{f1,f2,..., fm};
11F(i)
L=set(F(i)
L) ;
12FG=FG+F(i)
L;
13FG=set(FG);
14returnFG
5. Classification Model Generation
This section is devoted to presenting the second phase of our methodology in Figure 1. In
this phase, we develop a classification model for identifying Tactics and Techniques using the
collected data. To achieve this, we used the Problem Transformation Approach.
5.1. Problem Transformation Approach
Traditional supervised learning typically involves single-label models, where each instance
is assigned only one label. However, this single-label approach is not always practical. Attackers
or threat groups behind malicious applications typically use a combination of di fferent Tactics
and Techniques to carry out their attacks. Thus, instead of the traditional single-label approach,
12

we need to adopt a multi-label classification approach, where multiple labels are assigned to a
single instance. Unlike a single-label dataset, which has only one column for labels, a multi-
label classification dataset uses multiple columns for labels, with each column representing a
different label. If two labels are associated with a single instance, the corresponding columns for
those labels will have a value of 1 for that instance; otherwise, the value will be 0. To perform
multi-label classification, we utilized the Problem Transformation Approach (PTA) [32]. This
technique involves converting the multi-label classification task into several single-label classifi-
cation problems. In our study, we used three PTAs that are commonly used in the literature.
•Binary Relevance: Binary relevance decomposes the multi-label classification task into
multiple independent single-label binary classification problems. Specifically, for each
label li∈ L, a binary classifier hiis trained to predict whether liis a part of Yi. The
binary classifier hioutputs 1 if the label liis present and 0 otherwise. Mathematically,
each classifier learns a function hi:X→{0,1}. The overall multi-label prediction for a
new instance xis obtained by combining the predictions of all binary classifiers:
ˆY={li∈L| hi(x)=1}
•Classifier chains: In this approach, instead of training independent binary classifiers, a
sequence (or chain) of binary classifiers is trained. Each classifier in the chain uses not
only the original features, but also the predictions of previous classifiers as additional
features. Each binary classifier hifor label lipredicts yiusing the feature vector xand the
predictions of previous classifiers y1,y2,..., yi−1. For example, x∈X, the process starts
with the original features x. The first classifier h1predicts y1using x. The second classifier
h2predicts y2using xandy1. This process continues until the last classifier htpredicts yt
using xand all previous predictions. The final multi-label prediction for xis the set of all
labels lifor which yi=1.
•Label Powerset: Instead of handling each label independently, Label Powerset (LP) con-
siders every possible combination of labels as a single class, resulting in 2tpossible classes.
For example, if there are three labels, {l1,l2,l3}, LP creates classes for every possible com-
bination of these labels, such as {l1},{l1,l3},{l2,l3},{l1,l2}, etc. The classifier is then
trained to predict one of these combinations for each instance. During prediction, the
model outputs one of these pre-defined label combinations, which represents the set of
labels assigned to the instance.
For the implementation of these PTAs, we employ a diverse set of classifiers that are e ffec-
tive in multi-label classification. Random Forest (RF) [37] and Decision Trees (DT) [34] excel
at modeling complex decision boundaries while leveraging ensemble methods to capture label
correlations. XGBoost enhances performance in structured data by e fficiently handling imbal-
anced multi-label distributions and optimizing multiple objectives through boosting techniques
[6]. Meanwhile, Multi-Layer Perceptron (MLP) leverages deep learning to learn hierarchical
representations and capture label dependencies through hidden layers, improving predictive ac-
curacy in multi-label scenarios [7]. We selected these classifiers due to their demonstrated e ffec-
tiveness in previously published research on similar tasks. [28, 20].
6. LLM Approaches
In this section, we explore alternative LLM-based solutions to execute the second phase.
Specifically, we introduce a Retrieval-Augmented solution and a Fine-tuning methodology.
13

6.1. Prompt Engineering
In this phase, we develop a prompt-based approach to predict TTPs from static features of
Android applications using LLMs. Designing prompts that e ffectively guide LLMs in generating
accurate and relevant responses is a challenging task [8]. A well-designed prompt should strategi-
cally direct the model to leverage its internal knowledge e ffectively. Prompt engineering involves
designing inputs that align the model outputs with desired objectives [11]. The e ffectiveness of
a prompt depends on multiple factors, including its structure, formatting, and linguistic nuances,
as LLMs exhibit high sensitivity to subtle variations in these elements [39]. We employ prompt
engineering to optimize the extraction of relevant TTPs. We implement four distinct prompting
strategies, each tailored to extract TTPs from static features with varying degrees of user-defined
instruction specificity. After evaluating the performance of these prompts, we integrate a RAG
approach to further enhance the model accuracy and contextual relevance. In the following, we
list the four prompt templates used in this study.
Instruction :Analyze the static features associated with an Android
application. Identify and list the relevant MITRE Tactics and Techniques using
the context provided.
Response Format:
Tactic(s): <List of Tactics >
Technique(s): <List of Techniques >
If the provided data is insufficient to determine the Tactics and Techniques,
respond with: "Not enough information."
Question: {question }
Table 2: Prompt 1: Basic prompt
1. The first prompt strategy provides straightforward instruction to analyze the static features
and predict the associated Tactics and Techniques, as shown in Table 2. This prompt
includes clear fundamental parameters for the task, a structured format for the output,
and a fallback response, “Not enough information” if the data are insu fficient for making
predictions.
2. The second prompting strategy (see Table 3) introduces a domain-expert context by fram-
ing the query from the perspective of a cybersecurity expert. This helps to activate the
domain-relevant knowledge of the model, potentially improving prediction accuracy through
expert-aligned reasoning patterns. The prompt incorporates references to the MITRE
ATT&CK framework, includes a precise and professional output format, and uses domain-
specific language to simulate expert-level analysis.
3. The third prompt represents an advanced strategy for analyzing Android application se-
curity by leveraging static features to identify associated MITRE ATT&CK Tactics and
Techniques as shown in Table 4. It incorporates (i)a structured definition of Tactics and
Techniques, (ii)explicit references to the MITRE ATT&CK Android matrix for threat in-
telligence validation, (iii)a scoring mechanism to quantify the likelihood and relevance of
identified TTPs, and (iv)a standardized response format to ensure consistency and inter-
pretability in security assessments.
4. The fourth strategy represents the most advanced approach, incorporating comprehensive
technical context about Android static features and their security implications, as shown
14

Instruction :You are a cybersecurity expert specializing in mobile application
security. Your role is to analyze the static features associated with an
Android application. Identify and list the relevant MITRE Tactics and
Techniques using the context provided. Refer to this link for more details
(https://attack.mitre.org/matrices/mobile/android/ )
Response Format:
Tactic(s): <List of Tactics, e.g., Collection, Impact >
Technique(s): <List of Techniques, e.g., T1636, T1582, T1604, T1437, T1521,
T1417 >
If the provided data is insufficient to determine the Tactics and Techniques,
respond with: “Not enough information.”
Note: Only provide the Tactics and Techniques in the specified format. Do not
include any additional explanations or comments.
Question: {question }
Table 3: Prompt 2: Expert level prompt
in Table 5. This prompt includes (i)detailed technical specifications of Android static
features, (ii)definitions of Tactics and Techniques, (iii)references to threat intelligence
platforms, and (iv)a standardized output schema optimized for consistency in predictions.
6.2. Retrieval Augmented Generation for Tactics and Techniques Prediction
This section explores the e ffectiveness of the RAG model in predicting Tactics and Tech-
niques based on static features. As explained in the background 3, the RAG framework comprises
two primary components: the retriever and the generator . The retriever is tasked with locating
relevant information within a large knowledge base, which is stored in a vector database. This
database holds precomputed dense vector representations of the indexed information. To gener-
ate the knowledge base, we utilized the same training samples as in the ML-based approach but
stored them di fferently in CSV format. Each training sample comprises three key elements: (i)
Description represents a descriptive summary of the app’s features, formatted as follows: “Activ-
ities related to app <Hash >are:<Activities >. Permissions required: <Permissions >. Services
used: <Services >. Receivers included: <Receivers >. Intent Actions: <Intent Actions >and
Intent Categories: <Intent Categories >”.(ii)Tactic represents the associated Tactics used by
the attacker using this particular app (iii)Technique specifies the relevant Techniques associated
with the attack.
We use the CSVLoader to load the dataset and divide it into smaller, manageable segments
for easier processing. Each segment is then transformed into a dense vector representation using
an embedding model. Once converted into dense vectors, the dataset is indexed in a vector
database. This indexing facilitates e fficient similarity searches, enabling the retrieval of the most
contextually relevant documents in response to queries.
When a mobile security analyst submits a query, it undergoes processing and conversion into
a vector representation using the same embedding model. This vector representation is then used
with an Approximate Nearest Neighbor algorithm (ANN) to search for the most similar docu-
ments in the vector database, using cosine similarity as the metric. The ANN algorithm ensures
that only the top- kmost relevant documents are retrieved, reducing the computational burden
while ensuring that the most pertinent context is available for generating predictions. Once the
15

Instruction: Analyze the given static features associated with an Android
application. Identify and list the relevant MITRE Tactics and Techniques
linked to these features.
Static Features: { question }
Tactics: Tactics represent the broad objectives that an adversary seeks to
achieve. Identify the Tactics (one or many) from the MITRE ATT&CK for Android
matrix ( https://attack.mitre.org/matrices/mobile/android/ ) associated with the static
features.
Techniques: Techniques are specific methods adversaries employ to achieve
their Tactics. List the relevant technique IDs associated with the given
static features. Some examples of Techniques are: T1636, T1582, T1604, T1437,
etc. Refer MITRE ATT&CK Android matrix site for technique IDs.
Scoring: For each identified Tactic and Technique, assign a score from 0 to 5
based on the relevance and likelihood of the Tactic/Technique being linked to
the given static features.
Response Format:
Tactics and Scores:
-<Tactic 1 >:<Score (0-5) >
-<Tactic 2 >:<Score (0-5) >
...
Techniques and Scores:
-<Technique 1 >:<Score (0-5) >
-<Technique 2 >:<Score (0-5) >
...
Using the details provided, return the associated Tactics and Techniques with
scores.
If there is insufficient information to determine the Tactics and Techniques,
respond with: "Not enough information."
Note: - Provide only the Tactics, Techniques, and scores in the specified
format. - Do not include any additional explanations or comments.
Table 4: Prompt 3: Expert level prompt with score
retriever identifies the top- krelevant documents, these, along with the query, are passed to the
generator component. The generator, typically a pretrained LLM, is responsible for producing
the final output based on the provided context and prompt. Specifically, the LLM is given the
relevant context retrieved by the retriever, reducing the likelihood of hallucinations (i.e., gener-
ating incorrect or fabricated information). In our approach, the prompt used for the generator
is the same prompt identified in the previous phase as the most e ffective for generating accurate
responses. This methodology significantly mitigates the risk of irrelevant or fabricated informa-
tion, as the model relies solely on the context retrieved from the vector database.
16

Instruction : You are a cybersecurity expert specializing in mobile application security.
Your task is to respond to queries associated with Android applications. Each Android app
has different static features.
Understanding Static Features
Static features are characteristics of an Android application that can be analyzed without
executing the app. These features provide insights into the app’s behavior, permissions,
and components.
Here are examples of static features commonly used in analyzing Android applications:
Permissions :These are declarations that specify what resources the app can access. ,
e.g., android.permission.INTERNET, android.permission.ACCESS_FINE_LOCATION.
Activities :Activities represent a single screen in an app’s user interface., e.g.,
com.example.app.MainActivity.
Services :Services running in the background, e.g., com.example.app.BackgroundSyncService.
Intents :Specify how the app communicates with other apps or handles specific actions,
e.g., android.intent.action.VIEW.
Broadcast Receivers :Components listening for system or app-level broadcasts, e.g.,
com.example.app.BatteryLowReceiver.
By analyzing the static features, it is possible to infer the Tactics and Techniques used
by adversaries to compromise mobile security. For more details on static features, you
can refer to the [Android Developers documentation]( https://developer.android.com/guide/topics/
manifest/manifest-intro ).
To conduct attacks through Android applications, adversaries use different Tactics and
Techniques.
Understanding Tactics
In the context of cybersecurity, Tactics refer to the general goals or objectives that
an adversary aims to achieve through their actions. These can include a wide range of
malicious activities, such as Collection, Impact, Defense Evasion, etc.
Understanding Techniques
Techniques are specific methods or actions that adversaries use to achieve their Tactics.
Each Tactic in the MITRE ATT&CK framework is mapped to multiple Techniques. Some examples
of Techniques are: T1636, T1582, T1604, T1437, etc.
More details about Tactics and Techniques are available at MITRE ATT&CK (https://attack.
mitre.org/matrices/mobile/android/ ), which provides a matrix that maps Tactics to specific
Techniques used by adversaries. Other threat intelligence platforms such as VirusTotal
(https://www.virustotal.com/gui/home/upload ), and Alienvault (https://otx.alienvault.com/ ) also
provide valuable insights into potential malicious behaviors and TTPs associated with
various applications. Also, refer to TRAM ( https://github.com/mitre-attack/tram ) for more TTP
mapping.
Using the details provided, analyze the static features of an Android application and
identify the associated Tactics and Techniques. One app may be associated with multiple
Tactics and Techniques.
Response Format:
Tactic(s): <List of Tactics, e.g., Collection, Impact >
Technique(s): <List of Techniques, e.g., T1636, T1582, T1604, T1437, T1521, T1417 >
If the provided data is insufficient to determine the Tactics and Techniques, respond with:
"Not enough information."
Note: Only provide the Tactics and Techniques in the specified format. Do not include any
additional explanations or comments.
Question: question
Table 5: Prompt 4: Detailed prompt
6.3. Fine Tuning LLM
When adapting pre-trained LLMs for specific applications, fine-tuning avoids the resource-
intensive process of complete parameter retraining. In the context of cybersecurity, a significant
challenge arises from the lack of domain-specific information in pre-training datasets, leading
17

to hallucination issues. Fine-tuning addresses this gap by incorporating specialized knowl-
edge, thereby reducing hallucinations and improving accuracy in predicting attackers’ Tactics
and Techniques. In this study, we experimented with a range of open-source pre-trained LLMs
with various strengths and architectures, including SecBERT14and CySecBERT [5], which were
developed specifically for cybersecurity tasks. We selected Phi-3-mini-4k-instruct for its ability
to generalize well, despite its compact size. The Phi series of Small Language Models (SLMs)
is notable for its unique combination of high performance and cost-e ffectiveness, consistently
outperforming models of similar or larger size in various linguistic tasks [1]. Moreover, we
experimented with Meta-Llama-3-8B-Instruct for its versatile capabilities and e fficient architec-
ture, which are used in various downstream tasks [22]. This 8-billion parameter model strikes an
optimal balance between computational requirements and performance capabilities. Similarly,
we experimented with Mistral-7B-Instruct-v0.2 for its optimal performance and e fficiency [15].
To optimize the fine-tuning process, we employed Quantized Low-Rank Adaptation (QLoRA)
[13] Technique with 4-bit quantization. This approach minimizes computational demands while
maintaining model e ffectiveness, making it particularly valuable for updating large models with
limited computing resources.
7. Experiment and Evaluation
This section describes the experiments carried out to test the e ffectiveness and robustness of
our approach. In particular, in the following sections, we will describe the testbeds, the data, and
the evaluation metrics used to test our framework, and we report the obtained results along with
their critical analysis.
7.1. Experimental Setup
We conducted the implementation on an Intel i9 Windows system equipped with a 5GB
GPU. We also used a workstation equipped with an Intel i9-14900KF, dual RTX4090 GPUs, and
128 GB of RAM for LLM experimentation. For the fine-tuning process, we leveraged the Hug-
ging Face Transformers library, utilizing pre-trained language models and optimizing them with
PyTorch. In the RAG framework, we integrated Facebook AI Similarity Search (FAISS)15for
efficient document retrieval and LangChain16to manage the retrieval and response generation
pipeline. For multi-label classification tasks, we employed the scikit-multilearn library. To mit-
igate class imbalance, we incorporated MLSMOTE [9] for the generation of synthetic samples.
To enhance model interpretability, we utilized the SHAP library and, for data visualization, we
used the Python library matplotlib.
7.2. Dataset
As discussed in Section 4.1, to the best of our knowledge, no existing dataset was available.
Therefore, we created our own dataset for experimentation following the process described in
Section 4. We initially extracted 3034 hashes from the procedure references from the MITRE
ATT&CK knowledge base. These hashes were subsequently verified using VirusTotal, which
revealed that 261 hashes did not correspond to Android applications. Consequently, these hashes
14https://github.com/jackaduma/SecBERT
15https://ai.meta.com/tools/faiss
16https://www.langchain.com
18

CA
DE
Dis
E
IA
P
Exf
PE
C&C
Coll
ImpCA
DE
Dis
E
IA
P
Exf
PE
C&C
Coll
Imp437 429 169 36 8 73 62 4 405 437 345
429 2464 2121 1822 10 1067 95 976 2133 2358 493
169 2121 2190 1822 20 1074 94 980 1907 2096 242
36 1822 1822 1822 0 960 42 957 1644 1822 22
8 10 20 0 20 3 7 0 8 19 17
73 1067 1074 960 3 1075 23 964 1073 1075 94
62 95 94 42 7 23 95 9 83 83 12
4 976 980 957 0 964 9 984 974 978 6
405 2133 1907 1644 8 1073 83 974 2188 2106 462
437 2358 2096 1822 19 1075 83 978 2106 2431 476
345 493 242 22 17 94 12 6 462 476 691Figure 4: Frequency of samples across various classes in the Tactic dataset. CA- Credential Access, DE- Defense
Evasion, Dis- Discovery, E-Execution, IA-Initial Access, P-Persistence, Exf- Exfiltration, PE- Privilege Escalation,
C&C-Command and Control, Coll- Collection, Imp- Impact.
were excluded from further experimentation. We excluded the Lateral Movement Tactic label
from the dataset due to its association with only a single hash, which lacks su fficient data for
effective model training and generalization. After completing all necessary steps, we compile
a Tactic classification dataset consisting of 2774 apps with 13324 features and 11 Tactic labels.
The distribution of the samples from the Tactic data set is illustrated in Figure 4. Since the
dataset is multi-labeled, an app can exhibit multiple Tactics. 4 report the number of samples
for each Tactic class as well as the co-occurrence of samples across multiple Tactic classes.
For example, the dataset includes 2 ,464 samples labeled with the Defense Evasion Tactic
and 20 samples labeled with the Initial Access Tactic. This indicates that the dataset is
highly imbalanced. Furthermore, the figure reveals that the 429 samples are labeled with both
theCredential Access andDefense Evasion Tactics. Similarly, we generated a Technique
classification dataset with the same number of samples and feature sets. The distribution of
techniques is illustrated in Figure 5. Specifically, most techniques are associated with a similar
percentage of samples, with T1636 and T1406 appearing most frequently. In contrast, techniques
such as T1481 and T1662 occur at significantly lower rates. While MITRE ATT&CK defines 85
Techniques for Android, many of these techniques lack su fficient malware hash data for e ffective
model training. Consequently, we were able to collect adequate data for only 48 Techniques.
As a result, our Technique classification task focuses on these 48 Techniques, ensuring optimal
model performance, accuracy, and generalizability within the available data scope.
7.3. Hyperparameter Setup
The performance of the model is highly dependent on the hyperparameters selected for each
classifier. To optimize these parameters, we applied Randomized Search Cross-Validation (CV)
19

Figure 5: Technique ID Distribution in DroidTTP Technique Dataset
Table 6: Optimized Hyperparameters for Tactic and Technique Classification
PTA Classifier Tactic Technique
BRDT criterion: entropy, max_depth: 41, min_samples_leaf: 3,
min_samples_split: 17criterion: gini, max_depth: 63, max_leaf_nodes: 50,
min_samples_leaf: 3, min_samples_split: 12
RF max_depth: None, min_samples_leaf: 1, min_samples_split:
3, n_estimators: 78max_depth: 20, min_samples_leaf: 2, min_samples_split: 3,
n_estimators: 64
XGBoost gamma: 0.632, learning_rate: 0.242, max_depth: 9,
n_estimators: 82gamma: 1.679, learning_rate: 0.124, max_depth: 14,
n_estimators: 156
MLP hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001
CCDT criterion: entropy, max_depth: 91, max_leaf_nodes: 100,
min_samples_leaf: 2, min_samples_split: 14criterion: gini, max_depth: 65, max_leaf_nodes: 20,
min_samples_leaf: 2, min_samples_split: 5
RF max_features: auto, min_samples_leaf: 1, min_samples_split:
5, n_estimators: 96max_features: auto, min_samples_leaf: 1, min_samples_split:
5, n_estimators: 178
XGBoost gamma: 0.644, learning_rate: 0.243, max_depth: 5,
n_estimators: 191gamma: 1.896, learning_rate: 0.253, max_depth: 12,
n_estimators: 54
MLP hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001
LPDT criterion: gini, max_depth: 31, max_leaf_nodes: 50,
min_samples_leaf: 4, min_samples_split: 19criterion: entropy, max_depth: 46, max_leaf_nodes: None,
min_samples_leaf: 3, min_samples_split: 4
RF max_features: log2, min_samples_leaf: 1, min_samples_split:
6, n_estimators: 199max_features: auto, min_samples_leaf: 1, min_samples_split:
5, n_estimators: 153
XGBoost gamma: 0, learning_rate: 0.3, max_depth: 6, n_estimators:
100gamma: 0, learning_rate: 0.3, max_depth: 6, n_estimators:
100
MLP hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001hidden_layer_sizes: 100, activation: relu, solver: Adam, learn-
ing_rate: 0.001
to the Decision Tree, Random Forest, and XGBoost classifiers. The optimized hyperparameters
for each classifier, tailored for both Tactic and Technique classification tasks, are summarized in
Table 6.
7.4. Evaluation Metrics
To assess the performance of our Tactic and Technique classification model, we employed
the evaluation metrics, such as Accuracy (A), Weighted Precision (P), Weighted Recall (R),
20

Weighted F1 score (F1), Jaccard Similarity (JS), and Hamming Loss (HL).
•Accuracy (A). It measures the proportion of samples that were correctly classified by the
model.
A=1
NNX
i=1(yi=ˆyi) (3)
where N is the total number of instances, yiis the true label for the i-th instance, and ˆ yiis
the predicted label for the i-th instance.
•Weighted Precision (P). Precision evaluates the proportion of correctly predicted labels
among all predicted labels. Weighted precision adjusts for class imbalance by computing
the metric for each label and weighting it by the number of true instances of that label.
P=PT
t=1wt·Precision tPT
t=1wt(4)
where, trepresents the index of each individual label, Tdenotes the total number of unique
labels, Precision t=TPt
TPt+FPtandwt=TPt+FNt
•Weighted Recall (R). It measures the proportion of correctly predicted labels among all
true labels.
R=PT
t=1wt·Recall tPT
t=1wt(5)
where, Recall t=TPt
TPt+FNt
•Weighted F1-score (F1). It is the harmonic mean of precision and recall.
F1=PT
t=1wt·F1tPT
t=1wt(6)
where, F1 t=2·Pt·Rt
Pt+Rt
•Jaccard Similarity (JS). It measures how similar the predicted set of labels is to the true
set of labels. A higher Jaccard Similarity indicates better model performance.
JS=1
nnX
i=1|yi∩ˆyi|
|yi∪ˆyi|(7)
where nis the total number of instances. |yi∩ˆyi|is the size of the intersection of the true
and predicted label sets (i.e., the number of labels that are correctly predicted). |yi∪ˆyi|is
the size of the union of the true and predicted label sets (i.e., the total number of unique
labels in either the true or predicted set).
•Hamming Loss (HL). Measures the average fraction of labels that are incorrectly predicted
across all instances. Hamming Loss (HL) is calculated using the Equation 8. A lower Ham-
ming Loss corresponds to better performance, with HL=0 indicating perfect predictions
21

(i.e., all labels are correctly predicted) and HL=1 meaning that all labels are incorrectly
predicted for all instances.
HL=1
nnX
i=11
|Li||Li|X
j=1I(yi j,ˆyi j) (8)
where nis the total number of instances. |Li|is the number of labels for the i-th instance.
yi jis the actual label of the j-th label for the i-th instance. ˆ yi jis the predicted label of
thej-th label for the i-th instance. I(condition) is the indicator function, returning 1 if the
condition is true and 0 otherwise.
7.5. Experimental Evaluation of Tactic Identification
In this section, we present the results of Tactic classification experiments. As discussed in
Section 2, we experimented with various PTAs and ML algorithms. Initially, we developed the
Tactic model using the full feature set and tested it with ten di fferent random seeds. To evaluate
the performance of the Tactic models, we calculated the average score across these random seeds.
Table 7 summarizes the performance of the Tactic models. The findings indicate that the Label
Powerset method combined with XGBoost demonstrates the highest performance compared to
other classifiers. This model achieves a low Hamming Loss of 0.0102, a high Jaccard Similarity
index of 0.9794, and an F1-score of 0.9892.
Table 7: Performance comparison of Tactic classification models
PTA Classifier A P R F1 JS HL
BRDT 0.9218 0.9833 0.9829 0.9828 0.9630 0.0159
RF 0.9191 0.9831 0.9821 0.9817 0.9634 0.0164
XGBoost 0.9539 0.9883 0.9906 0.9894 0.9765 0.0100
MLP 0.9247 0.9833 0.9798 0.9811 0.9610 0.0173
CCDT 0.9286 0.9795 0.9836 0.9812 0.9630 0.0175
RF 0.9250 0.9819 0.9842 0.9822 0.9652 0.0160
XGBoost 0.9618 0.9866 0.9914 0.9889 0.9779 0.0104
MLP 0.9240 0.9847 0.9782 0.9810 0.9596 0.0173
LPDT 0.9587 0.9840 0.9876 0.9856 0.9730 0.0135
RF 0.9623 0.9851 0.9875 0.9861 0.9749 0.0130
XGBoost 0.9694 0.9880 0.9905 0.9892 0.9794 0.0102
MLP 0.9310 0.9759 0.9728 0.9737 0.9523 0.0242
We then applied the feature selection procedure outlined in Section 4 to reduce the feature set,
optimizing computational e fficiency and minimizing training time. As discussed, when the Tactic
model was trained using the full feature set, the Label Powerset combined with XGBoost yielded
the best performance; therefore, we applied feature selection specifically for the XGBoost clas-
sifier using the Label Powerset approach. As described in the feature selection procedure, we
initially selected the top mfeatures for each class, then combined these top mfeatures across all
labels. This process was repeated for 10 di fferent random seeds to identify the final top features
across the various datasets generated by random seeds and labels. To achieve this, we tested with
values of mranging from 100 to 13 ,300 and recorded the training time, top features, and various
evaluation metrics. Figure 6 presents the Jaccard Similarity score for the Tactic classification
model after applying feature selection. We limited the plot to mvalues between 100 and 1500
after determining that beyond 1500 features, the results remained consistent when mwas fixed
at 1500.
22

100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500
m value0.97600.97650.97700.97750.97800.97850.97900.97950.9800Jaccard SimilarityFigure 6: Jaccard Similarity score of Tactic classification model when feature selection applied.
From Figure 6, it is evident that the highest Jaccard Similarity score of 0 .9798 is achieved
when mis set to 300. When the top 300 features were selected for each label, we obtained 1737
final features aggregated from all ten datasets. Using this optimized feature set, we developed
the Tactic classification model, which is trained and evaluated across 10 random seeds. This
model is slightly higher than that of the model using the full feature set. Also, we observed that
building a Tactic model using the full feature set required significantly longer execution time,
taking 126.69 seconds. In contrast, the optimized feature set of 300 features reduced the exe-
cution time to just 18 .61 seconds. This substantial reduction in training time demonstrates the
effectiveness of feature selection in significantly enhancing model e fficiency without compro-
mising performance. To further analyze the classification results, we presented the individual
Table 8: Performance comparison between original and augmented datasets for Tactics.
TacticOriginal Dataset Augmented Dataset
P R F1 P R F1
Credential Access 0.95 0.97 0.96 0.99 0.99 0.99
Defence Evasion 0.99 0.99 0.99 1.00 1.00 1.00
Discovery 0.99 0.99 0.99 0.99 1.00 1.00
Execution 0.99 1.00 1.00 0.99 1.00 1.00
Initial Access 0.95 0.74 0.82 0.99 1.00 0.99
Persistance 0.99 1.00 0.99 0.99 1.00 0.99
Exfiltration 0.93 0.85 0.89 0.99 1.00 1.00
Privilage Escalation 0.99 0.99 0.99 0.99 0.99 0.99
Command and Control 0.99 0.99 0.99 0.99 1.00 0.99
Collection 0.99 0.99 0.99 1.00 1.00 1.00
Impact 0.98 0.97 0.97 0.99 0.98 0.98
weighted precision, recall, and F1 scores in Table 8. From this table, we can infer that most
Tactic classes achieved high performance, with scores exceeding 0 .95 across all metrics, except
for the Initial Access andExfiltration Tactics. The F1 score for Initial Access is
0.82, while the F1 score for Exfiltration is 0.89. This performance gap is primarily due to
the low number of samples for these classes. Specifically, Initial Access has only 20 sam-
ples, and the Exfiltration Tactic is represented by just 95 samples, as illustrated in the Figure
4. So, to address this issue, we augmented the dataset using the Multi-Label Synthetic Minority
23

Over-sampling Technique (MLSMOTE) [9].
Figure 7: Confusion matrix of Tactic classification model before Augmentation
Figure 8: Confusion matrix of Tactic classification model after augmentation
MLSMOTE is an adaptation of the popular Synthetic Minority Over-sampling Technique
(SMOTE) [10], specifically developed for multi-label classification problems. In this technique,
minority labels, such as Initial Access andExfiltration , are identified based on their
frequency within the dataset. After recognizing the minority labels, MLSMOTE identifies the
minority instances, which are the data points associated with these underrepresented labels. To
address the class imbalance, MLSMOTE generates synthetic samples for these minority labels.
For each minority instance, it identifies the k-nearest neighbors within the same label group
(we used the default value of 5 nearest neighbors as specified in MLSMOTE). It then creates
synthetic data by interpolating between the feature vectors of the minority instance and one of its
neighbors, adding minor random variations. The synthetic instance is assigned the same labels
as the original minority instance, and this process is repeated for other minority instances. To
control the number of synthetic samples generated, we tested various sample sizes on the basis
of the maximum count of the majority class. As shown in Figure 4, Defense Evasion has the
highest count, with 2464 instances. For this, we generated synthetic samples at rates of 25%,
50%, 75%, and 100% of the highest class count, ensuring that the synthetic data did not exceed
the original number of samples.
Table 9: Tactic model performance metrics at di fferent percentages of MLSMOTE
% A P R F1 JS HL
Original 0.9694 0.9881 0.9908 0.9894 0.9798 0.0099
25 0.9768 0.9912 0.9936 0.9924 0.9857 0.0074
50 0.9784 0.9920 0.9941 0.9930 0.9868 0.0068
75 0.9807 0.9933 0.9947 0.9940 0.9883 0.0058
100 0.9819 0.9939 0.9950 0.9945 0.9893 0.0054
After augmenting the dataset, we generated the Label Powerset using the XGBoost model
with 10 random seeds. The average results of the evaluation metrics for various sampling sce-
narios, along with those for the original dataset, are presented in Table 9. From the table, it is
24

(a) Exfilitration
 (b) Collection
Figure 9: SHAP summary plot of Tactic model
clear that as the number of synthetic samples increases, the model performance also improves.
The best results were achieved when the number of synthetic samples matched the highest class
count. In this case, the model achieved an accuracy of 0 .9819, precision of 0 .9939, recall of
0.9950, F1-score of 0 .9945, Jaccard Similarity of 0 .9893, and a Hamming Loss of 0 .0054. Fur-
thermore, our analysis shows that after data augmentation, the performance of all classes, par-
ticularly the minority classes such as Initial Access andExfiltration , has significantly
improved. As shown in Table 8, the F1- score for Initial Access increased from 0 .82 to 0.99,
while the F1 score for Exfiltration rose from 0 .89 to 1.00. These improvements highlight
that the augmented dataset enhances the e ffectiveness of Tactic classification.
To further analyze the classification results, we plotted a confusion matrix, shown in Figure
7. Since we used the Problem Transformation Approach, this led to multiple binary classification
confusion matrices, one for each class label. As a result, the overall confusion matrix consists
of 11 individual matrices, each representing the classification performance for a specific Tactic.
In the figure, we presented only 5 confusion matrices corresponding to the lowest performance
in Tactic classification using the original data. Additionally, we plotted the confusion matrices
for the same classes after augmentation (Figure 8) to assess the impact on classification and
misclassification rates. From these two confusion matrices, it is evident that the misclassification
rate decreased after data augmentation. Specifically, before augmentation, the misclassification
rate for Initial Access was 25%, which dropped to 0 after augmentation. Similarly, for
theExfiltration class, the misclassification rate decreased from 17 .6% to 0. This pattern of
improvement is observed across the other classes as well.
To thoroughly analyze and interpret the decision-making process of the model, we utilized
the SHapley Additive exPlanations (SHAP) tool. SHAP provides granular insights into the con-
tribution of each feature to the model’s predictions, o ffering a comprehensive understanding of
how static features influence the outcomes. We generated SHAP summary plots for each Tactic,
with specific illustrations for the Exfiltration andCollection Tactics shown in Figure 9.
In these plots, each dot corresponds to a SHAP value representing an individual dataset sample.
Features are ranked according to their average impact on the model predictions, as indicated on
the vertical axis. The horizontal axis denotes the SHAP values, reflecting the extent to which a
feature drives the prediction towards or away from the target class. The color gradient ranging
from blue to red reflects the actual feature value for each instance, with blue indicating a lower
value and red indicating a higher value.
In the SHAP summary plot for the Exfiltration Tactic features such as com.ashion.app.jav.
im_opp.re.quinlivan andcom.jty.google.player.keepalivebroadcastreceiver demon-
25

strate a strong association with the model’s prediction of exfiltration behavior. Higher SHAP val-
ues for these features (indicated by red dots) suggest that these specific app components are influ-
ential in predicting data exfiltration. For example, the feature com.jty.google.player.keep
alivebroadcastreceiver represents a background service that ensures the app remains active
and continuously transmits data, even when the user is not directly interacting with the app17. On
the other hand, for the Collection Tactic, features such as com.gau.go.launcherex.permissi
on.read_settings andcom.gbwhatsapp.updatedanayppreceiver significantly contribute
to the model’s prediction of data collection activities. The feature com.gau.go.launcherex.per
mission.read_settings is a custom permission specific to the GO Launcher app, allowing
other apps to read its settings18. Similarly, com.gbwhatsapp.updatedanayppreceiver , a
custom receiver, has a positive impact on the model’s prediction for collection Tactics.
Furthermore, we identified the top ten most influential features for each Tactic using SHAP
values and extracted the unique features across all Tactics to analyze their overall impact. The
heatmap in Figure 10 visualizes the relationship between static features and predicted Tactics.
The horizontal axis represents the features ( Fi), while the vertical axis lists the Tactic names. A
black grid cell at the intersection of feature Fiand a specific Tactic means that the feature plays a
crucial role in predicting that Tactic. For example, the feature com.example.pablo.cameraact
ivity (F21) plays an important role in predicting the Tactics of Discovery, Exfiltration, and Priv-
ilege Escalation. This feature likely corresponds to an in-app activity related to image capture or
handling, which is often associated with data collection and exfiltration. Additionally, the fea-
tures biz.otkur.app.izda.loader.a.activityn1ta1stnts0 (F3) and com.gbwhatsapp.
updatedanyappreceiver (F27) influence six di fferent Tactics. The feature F3 likely represents
a background activity related to loading or updating content from external sources. Meanwhile,
F27 could be a receiver component that listens for updates or data exchanges, potentially en-
abling unauthorized data collection or transmission when exploited by malicious apps.
7.6. Experimental Evaluation of Technique Identification
This section presents the results of the Technique classification. Similar to the Tactic classi-
fication, we conducted the same experiments for Techniques. Initially, we generated the model
using the complete feature set, with the results summarized in Table 10. Like the Tactic classi-
fication model, for the Technique classification model, Label Powerset with XGBoost achieved
the highest performance, exhibiting an F1 score of 0 .9891, Jaccard Similarity of 0 .9727, and
Hamming Loss of 0 .0068.
Subsequently, we performed feature selection using the same setup as in the Tactic classifi-
cation. Figure 11 illustrates the Jaccard Similarity score of the Technique classification model
after applying feature selection. Our analysis revealed that feature selection produced consistent
results after m=1200. Specifically, when mis set to 100, we observed the best Jaccard Similar-
ity score of 0 .9727. At m=100, we initially selected the top 100 features for each class. After
combining the top features across di fferent Techniques and datasets, we ended up with 1834
features. Also, the Technique classification model, utilizing the top 1834 features, significantly
reduced training time to 36 .6 seconds, compared to the 185 .7 seconds required when using the
full feature set.
After reducing the feature set, we applied MLSMOTE to augment the data, addressing the
low performance of certain Technique IDs, as detailed in Table 12. Table 12 highlights individual
17https://www.android-doc.com/reference/android/content/BroadcastReceiver.html
18https://developer.android.com/reference/android/Manifest.permission
26

F1 F2 F3 F4 F5 F6 F7 F8 F9
F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 F25 F26 F27 F28 F29 F30 F31 F32 F33 F34 F35 F36 F37 F38 F39 F40 F41 F42 F43 F44 F45 F46 F47 F48 F49 F50 F51 F52CA
DE
Dis
E
IA
P
Exf
PE
C&C
Coll
ImpF1: app.maqsoftware.apocalypse.news, F2: atlk.ussdpifhzgedqrysfygranbxmffhck.p079k, F3:
biz.otkur.app.izda.loader.a.activityn1ta1stnts0, F4: ch.threema.app.gcminstanceidlistenerservice,
F5: com.abbondioendrizzi.tools.supercleaner.lock.receiver.lockrestarterbroadcastreceiver,
F6: com.activity.telephone_list.userdetail, F7: com.android.firmware.ac0042579.abf8bee13,
F8: com.android.firmware.gui.alr_5_1_1_5, F9: com.app.elipticsoft.app.twofactor, F10:
com.ashion.app.jav.im_opp.re.quinlivan, F11: com.bitsmedia.android.muslimpro.activities.
referencesactivity, F12: com.dktools.liteforfb.mainactivity, F13: com.duoku.platform.single.ui.dkstart
downloadactivity, F14: com.ecxphdwetzui.gwcrlrbzfm.cskigphqiew.ydagvss, F15: com.evernote.cardscan.ca
rdscannoteactivity, F16: com.evernote.ui.maps.snippet activity, F17: com.evernote.ui.widgetsearchactiv
ity, F18: com.example.myapplicationtest.ehlzaqkdbrfxenowymnbglwamldlaxbohuurgwu, F19:
com.example.myapplicationtest.geapwjepfgqyjyiamwkkbkfpboetq, F20: com.example.myapplicationtest.wauozqf
kaelejwimssuoqlokicq, F21: com.example.pablo.cameraactivity, F22: com.facebook.scard.showads,
F23: com.feiwothree.coverscreen.sr, F24: com.flow.framework. service.flowservice, F25:
com.gau.go.launcherex.permission.read_settings, F26: com.gb.atnfas.codesother.preferencescreen.chats.
chats_online, F27: com.gbwhatsapp.updatedanyappreceiver, F28: com.golden.wioxs.jfi6gv7,
F29: com.google.ring.receiver.shortcutreceiver, F30: com.googleplay.srs.cmndsrs, F31:
com.hua.ru.quan.loader.a.activityn1stpts0, F32: com.hua.ru.quan.loader.a.activityn1ta1stpts0,
F33: com.hua.ru.quan.loader.a.activityp0stnts0, F34: com.hzpd.ui.activity.myeditcolumnactivity, F35:
com.jty.google.player.keepalivebroad castreceiver, F36: com.kakao.talk.activity.setting.alertsettings
activity, F37: com.kloudpeak.gundem.view.activity.mediaindexactivity, F38: com.pubhny.hekzhgjty.bvpian
hzfev, F39: com.realtalk.mess.service.receiver.pushsmsreceiver, F40: com.receive.myse,
F41: com.sec.android.app.twlauncher.read_settings, F42: com.snaptube.premiumplus.alarm.alarmservice,
F43: com.trs.tibetqs.activity.preview imageactivity, F44: com.xhqlob.doscby.uprtknrhf.tqgmyio, F45:
live2chat.com.live2chat.zerolocatorservice.galleryupload.uploaddata,
F46: mania.soccer.app.sunservice.sun4.llllj1l, F47: org.merry.core.adminactivity, F48:
org.telegram.messenger.voip.telegramconnection service, F49: riseup.game.activity.homeactivity, F50:
sun.photoalbum1.sunservice.sun12.send.bbmdoc.llljlil, F51: sun.sunalbum.sunservices. sun8.lllijjl,
F52: uznkgrhpklwluqoy.fmdjwt.atjksfjl.fzkjmpkayzias
Figure 10: The heatmap illustrates the ten most influential SHAP-identified features for each Tactic. The horizontal axis
represents the features ( Fi), while the vertical axis displays the Tactic names. A black grid cell at the intersection of a
feature ( Fi) and a Tactic indicates that the feature significantly contributes to predicting that Tactic.
Table 10: Performance evaluation of Technique identification models
PTA Classifier A P R F1 JS HL
BRDT 0.8888 0.9832 0.9806 0.9816 0.9426 0.0113
RF 0.8924 0.9884 0.9767 0.9811 0.9396 0.0108
XGBoost 0.9418 0.9884 0.9906 0.9894 0.9670 0.0066
MLP 0.8968 0.9878 0.9759 0.9807 0.9391 0.0112
CCDT 0.9160 0.9826 0.9792 0.9805 0.9441 0.0120
RF 0.8996 0.9882 0.9759 0.9807 0.9370 0.0111
XGBoost 0.9548 0.9897 0.9895 0.9895 0.9661 0.0065
MLP 0.9016 0.9855 0.9771 0.9802 0.9397 0.0116
LPDT 0.9539 0.9835 0.9885 0.9858 0.9651 0.0089
RF 0.9595 0.9821 0.9913 0.9865 0.9696 0.0084
XGBoost 0.9631 0.9862 0.9922 0.9891 0.9727 0.0068
MLP 0.9220 0.9620 0.9810 0.9710 0.9400 0.0180
27

100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500
m value0.971000.971250.971500.971750.972000.972250.972500.972750.97300Jaccard SimilarityFigure 11: Jaccard Similarity score of Technique classification model when feature selection applied.
class performance, with columns shaded in gray indicating Technique IDs with performance be-
low 0.95. To enhance the performance of these underperforming classes, we applied MLSMOTE
using the same steps and percentage ratios outlined in the Tactic classification experiment. The
results of the data augmentation are summarized in Table 11. From Table 11, we observe that
when augmentation is performed at 100% of the maximum class count, the model achieves an
accuracy of 0 .9707, a precision of 0 .9903, a recall of 0 .9918, an F1 score of 0 .9910, a Jaccard
Similarity of 0 .9753, and a Hamming Loss of 0 .0050.
This demonstrates that similar to Tactic classification, the performance of Technique classi-
fication significantly improved after augmentation. Also, the performance of all 11 individual
classes increased to exceed 0 .97, as shown in Table 12. Specifically, T1662 , which initially had
a precision of 0 .57, a recall of 0 .54, and an F1 score of 0 .53, improved to 0 .92, 1.0, and 0.96,
respectively. Furthermore, we visualized the confusion matrices for the five Techniques with the
lowest performance using the original dataset and compared them with the augmented model, as
shown in Figures 12 and 13. These confusion matrices reveal a substantial reduction in the mis-
classification rate following data augmentation. Specifically, the misclassification rate for T1662
before augmentation was 50%, which is reduced to 0 after augmentation. Similar improvements
were observed for the other Techniques as well.
Table 11: Technique model performance metrics at di fferent percentages of MLSMOTE
% A P R F1 HL JS
Original 0.9632 0.9871 0.9919 0.9894 0.0066 0.9727
25 0.9629 0.9873 0.9901 0.9886 0.0065 0.9718
50 0.9666 0.9880 0.9911 0.9895 0.0059 0.9736
75 0.9688 0.9892 0.9916 0.9904 0.0054 0.9745
100 0.9707 0.9903 0.9918 0.9910 0.0050 0.9753
To explore the decision-making process of the Technique classification model, we generated
a SHAP summary plot for all Technique classes. The plots for two representative Technique
classes, T1577 andT1617 , are shown in Figure 14. From this figure, we can analyze the static
features that most significantly contribute to the Technique predictions. For T1577 , features such
asorg.telegram.ui.externalactionactivity, com.wh.updated.receiverstart , and
solution.rail. forward.gafbposjjimpluxmkwue have high SHAP values. This indi-
cates they significantly influence the prediction of T1577 . These features are associated with
28

Table 12: Performance Comparison of Technique Classification between Original and Augmented Datasets
Technique T1424 T1604 T1532 T1404 T1422 T1577 T1437 T1512 T1430 T1418 T1624 T1645 T1429 T1417 T1481 T1426
OriginalP 0.96 0.99 0.89 0.99 0.98 1 0.99 0.99 0.99 0.99 1 0.99 0.99 0.97 0.82 0.99
R 0.98 0.99 0.93 0.99 0.99 1 0.99 0.99 0.99 0.99 1 0.99 1 0.98 0.81 1
F1 0.97 0.99 0.91 0.99 0.99 1 0.99 0.99 0.99 0.99 1 0.99 0.99 0.97 0.79 0.99
AugmentedP 0.97 0.99 0.97 0.99 0.99 1 0.99 1 0.99 0.99 1 0.99 0.99 0.99 0.97 0.99
R 0.98 0.99 0.99 0.99 0.99 1 0.99 1 1 0.99 1 0.99 1 0.99 0.94 0.99
F1 0.97 0.99 0.98 0.99 0.99 1 0.99 1 0.99 0.99 1 0.99 0.99 0.99 0.96 0.99
Technique T1409 T1616 T1636 T1407 T1633 T1662 T1533 T1516 T1406 T1575 T1637 T1414 T1513 T1509 T1541 T1517
OriginalP 0.99 0.99 0.99 0.99 0.91 0.57 0.99 0.9 0.99 0.99 0.98 1 0.99 0.95 0.95 0.96
R 1 0.99 0.99 0.99 0.95 0.54 0.99 0.89 1 1 0.99 1 0.99 0.9 0.97 0.97
F1 0.99 0.99 0.99 0.99 0.93 0.53 0.99 0.9 0.99 1 0.99 1 0.99 0.91 0.96 0.97
AugmentedP 0.99 0.99 0.99 0.99 0.98 0.92 0.99 0.98 0.99 1 0.99 1 0.99 0.97 0.99 0.98
R 0.99 0.99 1 0.99 0.98 1 0.99 0.97 1 1 0.99 1 0.99 1 0.98 0.98
F1 0.99 0.99 0.99 0.99 0.98 0.96 0.99 0.97 0.99 1 0.99 1 0.99 0.98 0.98 0.98
Technique T1644 T1471 T1420 T1623 T1655 T1544 T1643 T1617 T1630 T1646 T1640 T1398 T1521 T1582 T1421 T1642
OriginalP 0.99 1 0.87 0.99 0.99 0.91 0.97 0.96 0.99 0.91 0.96 1 0.99 0.99 0.96 0.98
R 1 0.88 0.93 1 0.99 0.98 0.98 0.97 1 0.86 0.97 1 1 0.96 0.91 0.98
F1 0.99 0.93 0.9 0.99 0.99 0.94 0.97 0.96 0.99 0.88 0.96 1 1 0.97 0.93 0.98
AugmentedP 0.99 0.97 0.97 0.99 0.99 0.98 0.98 0.99 0.99 0.99 0.99 1 0.99 0.98 0.99 1
R 0.99 0.98 0.99 0.99 1 0.99 0.97 1 0.99 0.99 1 1 0.99 0.97 0.99 0.99
F1 0.99 0.98 0.98 0.99 0.99 0.99 0.98 1 0.99 0.99 1 1 0.99 0.98 0.99 0.99
Figure 12: Confusion matrix of Technique classification model before Augmentation
Figure 13: Confusion matrix of Technique classification model after augmentation
high-risk behaviors, like unauthorized service updates or unusual external actions, often seen
in Android threats. For T1617 , features like com.gtomato.talkbox.signupnewactivity,
com.wh.updated.receiverstart , and com.update.bbm. rc.cola.re.matttieo show
high SHAP values, making them important for predicting this Technique. The feature com.wh.up
dated.receiverstart , which appears in both T1577 andT1617 , suggests a common link to
malicious behaviors like unauthorized communication or updates, typical of certain malware ac-
tivities. To identify the top 10 features associated with the Technique classification model using
SHAP, we combined the top SHAP features from all Technique classes. We then selected the top
10 based on their association with various Techniques, as shown in Table 13. This table lists the
features along with their associated Techniques. From this table, we observed that some features
are linked to multiple Technique predictions. For example, the android.intent.action.asu
feature, an Android intent action used for communication between app components, is associated
29

(a) T1577
 (b) T1617
Figure 14: SHAP summary plot of Technique classification
with multiple Techniques (T1418, T1422, etc.). Similarly, com.dfxsdgr.qvoor.evmthxeurccvhxq ,
a service within the Android system, is related to several Techniques involving persistent or back-
ground tasks, such as T1407 (Download New Code at Runtime), T1417 (Input Capture), and
T1582 (SMS Control).
Table 13: Top 10 Features and Associated Techniques using SHAP values
Feature Type Techniques
android.intent.action.asu Intent T1418, T1422, T1424, T1513, T1532, T1582, T1604, T1617, T1623, T1624, T1633, T1637, T1643
com.dfxsdgr.qvoor.evmthxeurccvhxq Services T1407, T1417, T1437, T1521, T1541, T1582, T1604, T1624, T1637, T1642, T1645, T1646
com.gbwhatsapp.gcm.instanceidlistenerservice Services T1429, T1430, T1437, T1481, T1521, T1533, T1541, T1575, T1604, T1617, T1633
com.hua.ru.quan.loader.a.activity0_fullscreen Activity T1404, T1409, T1424, T1541, T1544, T1577, T1624, T1636, T1640, T1644
com.ingbvupdd.services2.dialoggoogleplaypassword Activity T1398, T1406, T1422, T1513, T1521, T1532, T1623, T1644, T1645, T1646
com.sangcall.service.kcupgradeactivity Activity T1409, T1424, T1437, T1513, T1521, T1533, T1541, T1582, T1630, T1640
sun.photoalbum1.sunservice.sun15.llli1jl Activity T1409, T1418, T1517, T1532, T1533, T1544, T1617, T1637, T1644, T1662
com.activity.hardmanager.infoactivity Activity T1418, T1424, T1426, T1513, T1521, T1541, T1544, T1577, T1617
com.hua.ru.quan.loader.a.activityn1ta0nrnts1 Activity T1420, T1437, T1512, T1623, T1630, T1633, T1636, T1642, T1662
com.skype.android.app.signin.msa.signinliveidactivity Activity T1421, T1429, T1430, T1509, T1513, T1521, T1533, T1544, T1643
7.7. Experimental Evaluation of Prompt based Approach
In this section, we conducted prompt engineering to determine the optimal prompt for Tactic
and Technique prediction. Our approach involved designing multiple prompts and evaluating
their performance based on precision, recall, and F1-score for both Tactics and Techniques. For
this study, we evaluated the performance of the prompt-based LLM (we used Mistral for this
study) on a test set of 555 samples, consistent with the test set used in our PTA. To ensure that
the LLM’s responses were concise and contained only the necessary information (i.e., the Tactic
name and Technique ID), we explicitly instructed it to avoid providing any additional details.
However, despite these instructions, the model occasionally generated extra text beyond the ex-
pected output. To systematically extract the predicted Tactic and Technique from the LLM’s
responses, we developed regular expressions tailored to match the expected format. However, in
some instances, variations in response patterns led to extraction challenges. To address these in-
consistencies, we conducted a manual review of certain responses to ensure accurate extraction
of Tactics and Techniques. After extracting the predicted values, we compared them with the
ground truth labels to evaluate the e ffectiveness of di fferent prompt formulations. The results of
this analysis are presented in Table 14.
From Table 14, we can infer that Prompt 1 yields the lowest scores, particularly in Technique
prediction, where evaluation metrics are close to zero. This suggests that Prompt 1 alone is in-
sufficient for the LLM to predict Tactics and Techniques from static features. It provides only
30

Table 14: Performance of each Prompt in Tactic and Technique prediction
Prompt Tactic Technique
P R F1 P R F1
Prompt 1 0.0478 0.0419 0.0432 0.0002 0.0001 0.0001
Prompt 2 0.4345 0.2560 0.2976 0.0184 0.0133 0.0138
Prompt 3 0.3838 0.2578 0.2979 0.1838 0.1578 0.1979
Prompt 4 0.5712 0.2557 0.3267 0.2146 0.2227 0.2148
basic instruction to analyze static features without defining what they are, o ffering no guidance
on how to approach the analysis or what factors to consider when determining Tactics and Tech-
niques. As the prompts are refined, a clear trend of improved performance emerges. Prompts 2
and 3 show a significant improvement in Tactic prediction compared to Prompt 1, with a modest
increase in Technique prediction as well. While these prompts introduce expert-level structur-
ing, they lack rich contextual information on static features and security intelligence platforms,
limiting their e ffectiveness. Among the four prompts, Prompt 4 achieves the highest perfor-
mance, with an F1-score of 0.3267 for Tactic prediction and 0.2148 for Technique prediction.
This improvement is likely due to the extensive background information provided on static fea-
tures, Tactics, and Techniques, giving the model a stronger foundation for accurate predictions.
Furthermore, references to MITRE ATT&CK, VirusTotal, AlienVault, and TRAM improve reli-
ability and encourage retrieval of relevant knowledge. However, even with Prompt 4, the LLM
remains inadequate for accurately predicting Tactics and Techniques, indicating the need for
model enhancements.
7.8. Experimental Evaluation of RAG with LLM
The prompt engineering process identified the most e ffective prompt for Android Tactic and
Technique detection. In the RAG approach, we used the optimized prompt (Prompt 4) to enhance
performance. Initially, we constructed a knowledge base for TTP prediction using Android static
features, which were loaded via the CSVLoader. The data was then divided into manageable
chunks using RecursiveCharacterTextSplitter with parameters chunk size as 4000 and chunk
overlap as 50, ensuring that each chunk contained su fficient context for meaningful retrieval
during query processing. To convert the chunks into numerical vectors, we employed Hugging-
FaceInstructEmbeddings19, known for their high performance in embedding instruction-based
tasks.
The embeddings were subsequently stored in a FAISS vector database, a powerful and e ffi-
cient tool for performing similarity searches on large datasets. FAISS can handle high-dimensional
vectors, making it suitable for fast retrieval of relevant context in real-time queries. Upon receiv-
ing a query, the same embedding model was used to transform the query into a vector, which
was then compared to the stored embeddings in FAISS to retrieve the most contextually relevant
documents. The prompt, query, and retrieved-context were then sent to the LLM (Mistral), which
generated the response. Since the model’s output is influenced by both the prompt and the con-
text, we experimented with di fferent numbers of retrieved context documents, specifically 3, 5,
7, 9, 15, 20, and 25, to assess their impact on model performance. The performance of the RAG
model with varying numbers of retrieved contexts is compared. The results are summarized in
Table 15.
19https://python.langchain.com/docs/integrations/text_embedding/instruct_embeddings/
31

Table 15: RAG performance for Tactic and Technique prediction with di fferent number of contexts
#Context Tactic Technique
P R F1 P R F1
3 0.6674 0.4480 0.4992 0.2700 0.3393 0.2764
5 0.6672 0.4469 0.4996 0.2788 0.3440 0.2802
7 0.6952 0.4956 0.5433 0.3089 0.4135 0.3164
9 0.7577 0.5608 0.6099 0.3705 0.5524 0.3828
15 0.8922 0.7031 0.7482 0.6081 0.8047 0.6227
20 0.8474 0.6578 0.6958 0.7550 0.7869 0.7257
25 0.8830 0.7318 0.7645 0.8031 0.8310 0.7872
From the results presented in Table 15, we observe a clear trend in the impact of the number
of retrieved contexts on the performance of Tactic and Technique prediction in our RAG-based
approach. As the number of retrieved contexts increases, the performance of both Tactic and
Technique prediction improves. This indicates that retrieving more contextual information pro-
vides richer knowledge for the model, leading to more accurate predictions. The highest F1-score
for Tactics (0 .7645) is achieved at 25 retrieved contexts, while for Techniques, the best F1-score
(0.7872) occurs at 25 retrieved contexts. While increasing the number of retrieved contexts gen-
erally enhances performance, it also leads to a significant rise in computational requirements. For
instance, processing 3 retrieved contexts requires approximately 5GB of GPU memory, whereas
handling 25 contexts demands close to 25GB.
While the RAG-based model demonstrated reasonable e ffectiveness in leveraging contex-
tual information for Tactic and Technique prediction, the Label Powerset approach with XG-
Boost achieved superior overall performance. This discrepancy can be attributed to several fac-
tors. RAG-based models rely on retrieved textual contexts, which, while enriching the model’s
knowledge, may also introduce noise, redundancy, or irrelevant information. In contrast, XG-
Boost operates on structured tabular data, e fficiently handling high-dimensional feature spaces
and capturing complex relationships between features. Additionally, RAG models depend on
embedding-based similarity retrieval, which is susceptible to semantic drift and retrieval errors.
Although the RAG approach leverages contextual retrieval, its dependence on pre-trained lan-
guage models poses challenges in accurately capturing nuanced, domain-specific knowledge,
potentially limiting its predictive capability.
Table 16: Performance comparison of fine-tuned LLM models for Tactic classification
Model A P R F1 HL JS
SecBERT 0.8916 0.9795 0.9657 0.972 0.0257 0.9443
Phi 0.8856 0.9776 0.9696 0.9729 0.0246 0.9458
Mistral 0.8975 0.9789 0.975 0.9764 0.0215 0.9508
CySecBERT 0.9027 0.9786 0.9735 0.9757 0.0225 0.9534
LLama 0.9142 0.9824 0.9787 0.9802 0.0182 0.9583
7.9. Experimental Evaluation of Fine Tuning LLM
As discussed in Section 4, we fine-tuned various LLM models for Tactic and Technique pre-
diction. Specifically, we experimented with SecBERT, CySecBERT, Phi, LLaMA, and Mistral.
Due to the computational feasibility of SecBERT and CySecBERT, these models were fine-tuned
32

Table 17: Performance comparison of fine-tuned LLM models for Technique classification
Model A P R F1 HL JS
CySecBERT 0.771 0.9776 0.9450 0.9589 0.0228 0.8580
SecBERT 0.7886 0.9773 0.9491 0.9611 0.0222 0.8650
Phi 0.8425 0.9805 0.9595 0.9686 0.0185 0.9075
Mistral 0.8598 0.9796 0.9698 0.9738 0.0157 0.9253
LLama 0.8827 0.9844 0.9747 0.9789 0.0127 0.9348
for 20 epochs, while the remaining LLMs underwent 10 training epochs. To ensure robustness,
each model was fine-tuned using 10 di fferent random seeds, and the final performance metrics
were averaged. The results for Tactic and Technique prediction are summarized in Tables 16 and
17.
From Table 16, we can infer that the meta-Llama-3-8B-Instruct model demonstrated the
highest performance on all the metrics evaluated, particularly with a Jaccard Similarity score of
0.9583 and a low Hamming Loss of 0 .0182. Moreover, the CySecBERT model, a transformer-
based architecture, achieved a comparatively high Jaccard Similarity score of 0 .9534, likely due
to its domain-specific training, which enhances its understanding of cybersecurity-related data.
The Mistral-7B-Instruct-v0.2 model also exhibited strong performance, attaining a Jaccard Simi-
larity score of 0 .9508 while slightly lower than Meta-Llama. These variations could be attributed
to differences in model architecture, pretraining data. Despite being a smaller model, Phi-3-
mini-4k-instruct achieved a Jaccard Similarity score of 0 .9458, demonstrating that even compact
LLMs can yield competitive results in Tactic prediction.
Similar to tactic classification, the Llama model achieved the highest performance in Tech-
nique classification, outperforming other LLMs with a Jaccard Similarity score of 0.9348 and a
Hamming Loss of 0.0127. The Mistral model followed closely, attaining a Jaccard Similarity of
0.9253. From this analysis, we infer that fine-tuning LLMs is more e ffective for Tactic predic-
tion compared to RAG-based models. The retrieval mechanism in RAG models does not seem
to complement the generation process e ffectively for this classification task. However, the Label
Powerset XGBoost model marginally outperformed the fine-tuned LLMs, and the performance
gap remains narrow. This suggests that ML models are more suited for Tactic and Technique
classification tasks, and fine-tuned LLMs are approaching their performance levels. However,
with further advances in fine-tuning Techniques and hybrid approaches, LLMs can continue to
close the performance gap with traditional ML models. The implementation details, including
the code and dataset, are available at https://github.com/OPTIMA-CTI/DroidTTP .
8. Conclusion
The increasing complexity of mobile malware and its impact on Android devices necessi-
tates advanced methods for understanding and mitigating emerging threats. This study presents
DroidTTP, a robust system designed to bridge the gap between malware detection and actionable
threat intelligence by mapping Android application behaviors to the TTPs defined in the MITRE
ATT&CK framework. Unlike traditional approaches that rely solely on malware classification,
DroidTTP provides deeper insights into attacker methodologies, enhancing the ability of security
analysts to respond e ffectively to diverse threats.
One of the key contributions of this work is the development of a novel dataset tailored
for mapping TTPs to Android applications. Also, we proposed an enhanced feature selection
33

strategy for multi-label classification. In this work, we applied the Problem Transformation
Approach along with Machine Learning classifiers and investigated the potential of LLMs in
predicting Tactics and Techniques. Our results demonstrate the e ffectiveness of this approach:
for the Tactic classification task, the Label Powerset method with XGBoost achieved a Jaccard
Similarity score of 0 .9893 and a Hamming Loss of 0 .0054, while for Technique classification,
the model attained a Jaccard Similarity of 0 .9753 and a Hamming Loss of 0 .0050. Our find-
ings indicate that LLMs also possess the capability to predict Tactics and Techniques, achieving
performance comparable to traditional ML models. For Tactic classification, the Llama model
demonstrated the best performance among LLMs, achieving a Jaccard Similarity score of 0 .9583
and a Hamming Loss of 0 .0182. Likewise, for Technique classification, Llama outperformed
other LLMs with a Jaccard Similarity score of 0 .9348 and a Hamming Loss of 0 .0127. These
results highlight the system’s capacity for precise TTP mapping and its potential to improve
threat attribution. In the future, we plan to expand the dataset to include a broader variety of
Android malware samples by sourcing additional APKs from threat intelligence platforms such
as VirusTotal. Furthermore, we plan to develop an Automated Threat Intelligence Dashboard
that provides real-time insights into malicious Android applications. This dashboard will inte-
grate machine learning-driven TTP mappings and visualization techniques to display key threat
metrics, such as detected tactics and techniques, and risk levels. The system will support inter-
active analysis, allowing security analysts to explore the behavior of APKs through graphical
representations such as Sankey diagrams, heatmaps, and SHAP-based feature importance plots.
Additionally, the dashboard will generate detailed structured application reports in STIX /TAXII
format, ensuring seamless integration with legacy CTI systems.
Acknowledgments
This work was supported in part by the following projects:
(i)The HORIZON Europe Framework Programme through the project “OPTIMA-Organization
sPecific Threat Intelligence Mining and sharing" (101063107), funded by the European Union.
Views and opinions expressed are however those of the authors only and do not necessarily re-
flect those of the European Union or the Italian MUR. Neither the European Union nor the Italian
MUR can be held responsible for them.
(ii)The project SERICS (PE00000014) under the NRRP MUR program funded by the EU -
NGEU. Views and opinions expressed are however those of the authors only and do not neces-
sarily reflect those of the European Union or the Italian MUR. Neither the European Union nor
the Italian MUR can be held responsible for them.
References
[1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree,
Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally
on your phone. arXiv preprint arXiv:2404.14219 , 2024.
34

[2] Abdullah M Al Alawi, Halima H Al Shuaili, Khalid Al-Naamani, Zakariya Al Naamani, and Said A Al-Busafi. A
machine learning-based mortality prediction model for patients with chronic hepatitis c infection: An exploratory
study. Journal of Clinical Medicine , 13(10):2939, 2024.
[3] Anshul Arora, Sateesh K Peddoju, and Mauro Conti. Permpair: Android malware detection using permission pairs.
IEEE Transactions on Information Forensics and Security , 15:1968–1982, 2019.
[4] Khaled Bakour and Halil Murat Ünver. Visdroid: Android malware classification based on local and global image
features, bag of visual words and machine learning techniques. Neural Computing and Applications , 33(8):3133–
3153, 2021.
[5] Markus Bayer, Philipp Kuehn, Ramin Shanehsaz, and Christian Reuter. Cysecbert: A domain-adapted language
model for the cybersecurity domain. ACM Transactions on Privacy and Security , 27(2):1–20, 2024.
[6] Simon Bohlender, Eneldo Loza Mencía, and Moritz Kulessa. Extreme gradient boosted multi-label trees for dy-
namic classifier chains. In Discovery Science: 23rd International Conference, DS 2020, Thessaloniki, Greece,
October 19–21, 2020, Proceedings 23 , pages 471–485. Springer, 2020.
[7] Ricardo Cerri, Rodrigo C Barros, and André CPLF De Carvalho. Hierarchical multi-label classification using local
neural networks. Journal of Computer and System Sciences , 80(1):39–56, 2014.
[8] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang
Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent
systems and technology , 15(3):1–45, 2024.
[9] Francisco Charte, Antonio J Rivera, María J del Jesus, and Francisco Herrera. Mlsmote: Approaching imbalanced
multilabel learning through synthetic instance generation. Knowledge-Based Systems , 89:385–397, 2015.
[10] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority
over-sampling technique. Journal of artificial intelligence research , 16:321–357, 2002.
[11] Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, and Shengxin Zhu. Unleashing the potential of prompt engi-
neering in large language models: a comprehensive review. arXiv preprint arXiv:2310.14735 , 2023.
[12] Minghao Chen, Kaijie Zhu, Bin Lu, Ding Li, Qingjun Yuan, and Yuefei Zhu. Aecr: Automatic attack technique
intelligence extraction based on fine-tuned large language model. Computers &Security , 150:104213, 2025.
[13] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: E fficient finetuning of quantized
llms. Advances in Neural Information Processing Systems , 36, 2024.
[14] Je ffrey Fairbanks, Andres Orbe, Christine Patterson, Janet Layne, Edoardo Serra, and Marion Scheepers. Identify-
ing att&ck tactics in android malware control flow graph through graph representation learning and interpretability.
In2021 IEEE International Conference on Big Data (Big Data) , pages 5602–5608. IEEE, 2021.
[15] Romy Fieblinger, Md Tanvirul Alam, and Nidhi Rastogi. Actionable cyber threat intelligence using knowledge
graphs and large language models. In 2024 IEEE European Symposium on Security and Privacy Workshops (Eu-
roS&PW), pages 100–111. IEEE, 2024.
[16] Priyanka Goyal, Sahil Batra, and Ajit Singh. A literature review of security attack in mobile ad-hoc networks.
International Journal of Computer Applications , 9(12):11–15, 2010.
[17] Yuelin Hu, Futai Zou, Jiajia Han, Xin Sun, and Yilei Wang. Llm-tikg: Threat intelligence knowledge graph
construction utilizing large language model. Computers &Security , 145:103999, 2024.
[18] Ghaith Husari, Ehab Al-Shaer, Mohiuddin Ahmed, Bill Chu, and Xi Niu. Ttpdrill: Automatic and accurate extrac-
tion of threat actions from unstructured text of cti sources. In Proceedings of the 33rd annual computer security
applications conference , pages 103–115, 2017.
[19] Rejwana Islam, Moinul Islam Sayed, Sajal Saha, Mohammad Jamal Hossain, and Md Abdul Masud. Android
malware classification using optimum feature selection and ensemble machine learning. Internet of Things and
Cyber-Physical Systems , 3:100–111, 2023.
[20] Heejung Kim and Hwankuk Kim. Comparative experiment on ttp classification with class imbalance using over-
sampling from cti dataset. Security and Communication Networks , 2022(1):5021125, 2022.
[21] TaeGuen Kim, BooJoong Kang, Mina Rho, Sakir Sezer, and Eul Gyu Im. A multimodal deep learning method
for android malware detection using various features. IEEE Transactions on Information Forensics and Security ,
14(3):773–788, 2018.
[22] Neha Mohan Kumar, Fahmida Tasnim Lisa, and Sheikh Rabiul Islam. Prompt chaining-assisted malware detection:
A hybrid approach utilizing fine-tuned llms and domain knowledge-enriched cybersecurity knowledge graphs. In
2024 IEEE International Conference on Big Data (BigData) , pages 1672–1677. IEEE, 2024.
[23] Valentine Legoy, Marco Caselli, Christin Seifert, and Andreas Peter. Automated retrieval of att&ck tactics and
techniques for cyber threat reports. arXiv preprint arXiv:2004.14322 , 2020.
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küt-
tler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive
nlp tasks. Advances in neural information processing systems , 33:9459–9474, 2020.
[25] Junyang Qiu, Jun Zhang, Wei Luo, Lei Pan, Surya Nepal, and Yang Xiang. A survey of android malware detection
with deep neural models. ACM Computing Surveys (CSUR) , 53(6):1–36, 2020.
35

[26] Andrea Saracino, Daniele Sgandurra, Gianluca Dini, and Fabio Martinelli. Madam: E ffective and e fficient
behavior-based android malware detection and prevention. IEEE Transactions on Dependable and Secure Com-
puting , 15(1):83–97, 2016.
[27] Samaneh Shafee, Alysson Bessani, and Pedro M Ferreira. Evaluation of llm-based chatbots for osint-based cyber
threat awareness. Expert Systems with Applications , page 125509, 2024.
[28] Yashovardhan Sharma, Simon Birnbach, and Ivan Martinovic. Radar: A ttp-based extensible, explainable, and
effective system for network tra ffic analysis and malware detection. In Proceedings of the 2023 European Interdis-
ciplinary Cybersecurity Conference , pages 159–166, 2023.
[29] Yashovardhan Sharma, Eleonora Giunchiglia, Simon Birnbach, and Ivan Martinovic. To ttp or not to ttp?: Exploit-
ing ttps to improve ml-based malware detection. In 2023 IEEE International Conference on Cyber Security and
Resilience (CSR) , pages 8–15. IEEE, 2023.
[30] Lina Shen, Mengqi Fang, and Jian Xu. Ghgdroid: Global heterogeneous graph-based android malware detection.
Computers &Security , 141:103846, 2024.
[31] Chanho Shin, Insup Lee, and Changhee Choi. Exploiting ttp co-occurrence via glove-based embedding with mitre
att&ck framework. IEEE Access , 2023.
[32] Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas. Mining multi-label data. Data mining and knowledge
discovery handbook , pages 667–685, 2010.
[33] Ziboud Van Veldhoven and Jan Vanthienen. Digital transformation as an interaction-driven perspective between
business, society, and technology. Electronic markets , 32(2):629–644, 2022.
[34] Celine Vens, Jan Struyf, Leander Schietgat, Sašo Džeroski, and Hendrik Blockeel. Decision trees for hierarchical
multi-label classification. Machine learning , 73:185–214, 2008.
[35] R Vinayakumar, KP Soman, and Prabaharan Poornachandran. Deep android malware detection and classification.
In2017 International conference on advances in computing, communications and informatics (ICACCI) , pages
1677–1683. IEEE, 2017.
[36] Bozhi Wu, Sen Chen, Cuiyun Gao, Lingling Fan, Yang Liu, Weiping Wen, and Michael R Lyu. Why an android app
is classified as malware: Toward malware classification interpretation. ACM Transactions on Software Engineering
and Methodology (TOSEM) , 30(2):1–29, 2021.
[37] Xin Wu, Yuchen Gao, and Dian Jiao. Multi-label classification based on random forest algorithm for non-intrusive
load monitoring system. Processes , 7(6):337, 2019.
[38] Ping Yan and Zheng Yan. A survey on dynamic mobile malware detection. Software Quality Journal , 26(3):891–
919, 2018.
[39] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing
Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. ACM Transactions
on Knowledge Discovery from Data , 18(6):1–32, 2024.
[40] Castro A Yoga, Anthony J Rodrigues, and Silvance O Abeka. Hybrid machine learning approach for attack classi-
fication and clustering in network security. International Journal of Computer Applications , 975:8887, 2023.
[41] Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, and Ee-Chien
Chang. Attackg +: Boosting attack graph construction with large language models. Computers &Security ,
150:104220, 2025.
[42] Zakiah Zulkefli and Manmeet Mahinderjit Singh. Sentient-based access control model: a mitigation technique for
advanced persistent threats in smartphones. Journal of Information Security and Applications , 51:102431, 2020.
36
# Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft

**Authors**: Peiyang Liu, Ziqiang Cui, Di Liang, Wei Ye

**Published**: 2025-10-09 03:09:18

**PDF URL**: [http://arxiv.org/pdf/2510.07728v1](http://arxiv.org/pdf/2510.07728v1)

## Abstract
Retrieval-augmented generation (RAG) enhances Large Language Models (LLMs) by
mitigating hallucinations and outdated information issues, yet simultaneously
facilitates unauthorized data appropriation at scale. This paper addresses this
challenge through two key contributions. First, we introduce RPD, a novel
dataset specifically designed for RAG plagiarism detection that encompasses
diverse professional domains and writing styles, overcoming limitations in
existing resources. Second, we develop a dual-layered watermarking system that
embeds protection at both semantic and lexical levels, complemented by an
interrogator-detective framework that employs statistical hypothesis testing on
accumulated evidence. Extensive experimentation demonstrates our approach's
effectiveness across varying query volumes, defense prompts, and retrieval
parameters, while maintaining resilience against adversarial evasion
techniques. This work establishes a foundational framework for intellectual
property protection in retrieval-augmented AI systems.

## Full Text


<!-- PDF content starts -->

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft
PEIYANG LIU,National Engineering Research Center for Software Engineering, Peking University, China
ZIQIANG CUI,City University of Hong Kong, China
DI LIANG,Fudan University, China
WEI YE∗,National Engineering Research Center for Software Engineering, Peking University, China
Retrieval-augmented generation (RAG) is widely used to address hallucinations and outdated data issues in Large Language Models
(LLMs). While RAG significantly enhances the credibility and real-time relevance of generated content by incorporating external
data sources, it has also enabled unauthorized data appropriation at an unprecedented scale. Detecting such unauthorized data
usage presents a critical challenge that demands innovative solutions. This paper addresses the challenge of detecting unauthorized
RAG-based content appropriation through two key contributions. First, we introduce RPD, a novel dataset specifically designed for
RAG plagiarism detection that overcomes the limitations of existing resources. RPD ensures recency, diversity, and realistic simulation
of RAG-generated content across various domains and writing styles by modeling diverse professional domains and writing styles.
Second, we propose a novel dual-layered watermarking system that sequentially embeds protection at both semantic and lexical
levels, creating a robust detection mechanism that remains effective even after LLM reformulation. This system is complemented by
an Interrogator-Detective framework that strategically probes suspected RAG systems and applies statistical hypothesis testing to
accumulated evidence, enabling reliable detection even when individual watermark signals are weakened during the RAG process.
Extensive experiments validate our approach’s effectiveness across varying query volumes, defense prompts, and retrieval parameters.
Our method demonstrates exceptional resilience against adversarial evasion techniques while maintaining high-quality text output. The
dual-layered approach provides complementary protection mechanisms that address the fundamental challenges of RAG plagiarism
detection, including LLM reformulation, fact redundancy, and adversarial evasion. This work provides a foundational framework for
protecting intellectual property in the era of retrieval-augmented AI systems. Our source code and dataset are publicly available at
https://anonymous.4open.science/r/RPD-7E98.
CCS Concepts:•Computing methodologies→Natural language generation.
Additional Key Words and Phrases: Retrieval Augmented Generation, Data Protection
ACM Reference Format:
Peiyang Liu, Ziqiang Cui, Di Liang, and Wei Ye. 2018. Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft. In
Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX).ACM, New
York, NY, USA, 36 pages. https://doi.org/XXXXXXX.XXXXXXX
∗Corresponding Author.
Authors’ Contact Information: Peiyang Liu, liupeiyang@pku.edu.cn, National Engineering Research Center for Software Engineering, Peking University,
Beijing, China; Ziqiang Cui, ziqiang.cui@my.cityu.edu.hk, City University of Hong Kong, Hong Kong, China; Di Liang, liangd17@fudan.edu.cn, Fudan
University, Shanghai, China; Wei Ye, wye@pku.edu.cn, National Engineering Research Center for Software Engineering, Peking University, Beijing,
China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2510.07728v1  [cs.IR]  9 Oct 2025

2 Liu et al.
Data Watermarking
Dual -Layered WatermarkingUnprotected Dataset
 Protected Dataset
Unauthorized
Thief RAG System
Retrieval Augmented GenerationQueriesRetrieval MechanismIndexer
RAG Corpus
Inver
ted
IndexInver
ted
IndexInverted
Index
RetrieverAll Documents 
Retrieved 
Documents  
Query
LLMRetrieved 
Documents  
RewriteQuery
Generated 
Result 
ResponsesUnauthorized Detection
Interrogator
 Watermarked DocumentSpecifically 
Designed 
Probing Queries
DetectiveGenerated 
Results 
 …
+
 +…
aggregate
 arrest
RAG Theft!Token Distribution WM
Knowledge -based WM
Fig. 1. Overview of our proposed pipeline for preventing unauthorized RAG misuse. First, we apply our Dual-Layered Watermarking
method to embed watermarks into the documents of the protected dataset. When a Thief RAG System illicitly accesses our dataset,
the watermark propagates through the RAG pipeline into the LLM’s generated outputs. Although the watermark inevitably weakens
during propagation, in the detection phase, the Interrogator strategically crafts queries that increase the likelihood of the RAG System
retrieving watermarked documents. By submitting multiple queries to the Thief RAG System, the aggregated watermark signals
accumulate to a detectable level, enabling statistical hypothesis testing to determine whether unauthorized misuse has occurred.
1 Introduction
Large Language Models have transformed how we generate and interact with textual content [ 1,19,32]. Despite their
capabilities, these models face inherent limitations in accessing up-to-date information and specialized knowledge
beyond their training data. Retrieval-Augmented Generation (RAG) [ 9,18] has emerged as an effective solution to
this challenge by dynamically incorporating external knowledge sources into the generation process. While RAG has
significantly enhanced LLM capabilities across numerous applications, it has simultaneously raised important ethical
and legal concerns regarding intellectual property rights.
1.1 The Challenge of Unauthorized Content Appropriation
RAG systems can retrieve valuable information created by content producers, incorporate it into their outputs, and
serve it to users without proper attribution or compensation. This raises significant questions about ownership rights
and the traceability of information as it moves through these systems.
The unauthorized appropriation of content through RAG systems represents a significant challenge at the intersection
of technology, ethics, and law. Content creators—ranging from news organizations and academic institutions to individual
bloggers—invest substantial resources in producing high-quality information. When this content is seamlessly integrated
into LLM outputs without attribution, it not only undermines the creators’ economic interests but also erodes the
foundation of intellectual property rights that incentivize knowledge creation [23].
1.2 The Detection Challenge
Detecting RAG-based content appropriation presents unique technical challenges that distinguish it from traditional
plagiarism detection:
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 3
•LLM Reformulation:Unlike direct copying, RAG systems typically use retrieved content as context for
generation rather than verbatim reproduction [ 34], resulting in paraphrased or synthesized outputs that maintain
the core information while altering the linguistic surface.
•Fact Redundancy:Multiple sources may contain identical factual information, making it difficult to determine
which specific source was used in the retrieval process [13].
•Adversarial Evasion:As detection methods emerge, malicious actors may develop countermeasures specifically
designed to evade these techniques [14, 22].
Despite these challenges, the ability to detect unauthorized RAG usage is crucial for protecting intellectual property,
ensuring proper attribution, and maintaining the integrity of the information ecosystem. Current approaches to this
problem remain limited, with a notable absence of specialized datasets and detection methodologies tailored to the
unique characteristics of RAG-based content appropriation.
1.3 Towards Effective RAG Plagiarism Detection
To address these challenges, we recognize the need for two fundamental components: (1) a specialized dataset that
realistically captures the nuances of RAG-generated content across diverse domains and writing styles, and (2) a robust
detection methodology capable of identifying unauthorized content use even after LLM reformulation.
Traditional watermarking methods for LLMs [ 15,20] are vulnerable to adversarial rewriting by illegal RAG systems,
while existing fact-based techniques [ 5,33] fail to address the data redundancy issues commonly found in RAG systems.
Similarly, existing plagiarism detection datasets fail to account for the unique characteristics of RAG-based content
appropriation, particularly the problems of fact redundancy caused by diverse writing styles among different creators
in real-world scenarios [13].
Our approach introduces a novel dual-layered watermarking system that operates sequentially to create robust
protection against unauthorized RAG usage. We develop a sophisticated framework that embeds distinctive knowledge-
based watermarks at the semantic level while also creating statistical signatures through token distribution manipulation.
This is complemented by an innovative Interrogator-Detective framework that strategically probes suspected RAG
systems and applies statistical hypothesis testing to detect unauthorized usage. The overview of our method can be
found in Figure 1. Our dataset creation methodology further enhances this approach by simulating the diversity and
complexity of real-world RAG applications.
1.4 Our Contributions
This paper addresses the critical gap in RAG plagiarism detection through the following contributions:
(1)Novel Dataset Creation:We introduce a comprehensive dataset specifically designed for RAG plagiarism
detection that overcomes the limitations of existing resources. Our approach leverages therepliqa[ 27] foundation
to ensure recency and employs a sophisticated author simulation system that models diverse professional domains,
writing styles, and preference dimensions. This methodology produces content that realistically simulates the
variability and redundancy found in real-world RAG scenarios.
(2)Dual-Layered Watermarking System:We propose an innovative sequential watermarking approach that
embeds protection at both semantic and lexical levels. Our system strategically integrates knowledge-based
watermarks with statistical token patterns to create a detection mechanism that: 1. Remains effective even after
Manuscript submitted to ACM

4 Liu et al.
LLM reformulation through the RAG pipeline. 2. Addresses the fact redundancy problem through distinctive
knowledge embedding 3. Provides complementary layers of protection against various evasion techniques.
(3)Interrogator-Detective Framework:We develop a novel detection framework that strategically generates
queries to reveal unauthorized data usage and applies rigorous statistical analysis to accumulated evidence,
enabling reliable detection even when individual watermark signals are weakened during the RAG process.
(4)Comprehensive Evaluation Framework:We implement a rigorous testing methodology that evaluates our
approach across multiple dimensions, including detection accuracy as a function of query volume, resilience
against defensive prompting, impact on text quality, and sensitivity to parameter variations.
2 Related Work
Our research builds upon and extends several key areas of prior work, including retrieval-augmented generation
systems, watermarking techniques for language models, and plagiarism detection methodologies.
2.1 Retrieval Augmented Generation
Retrieval Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing LLM capabilities by
incorporating external knowledge. The seminal work by [ 18] introduced the RAG framework, demonstrating significant
improvements in knowledge-intensive tasks. Recent advancements have focused on optimizing retrieval mechanisms
[11, 30] and improving the integration of retrieved information into generation processes [4, 36].
[9] provided a comprehensive survey of RAG techniques, highlighting their applications across various domains
and identifying open challenges. More recently, [ 34] proposed evaluation frameworks specifically designed for RAG
systems, emphasizing the need for metrics that assess both retrieval quality and generation fidelity. [ 12] introduced
active retrieval augmented generation, which dynamically determines when to retrieve information based on model
uncertainty.
Despite these advancements, the ethical and legal implications of RAG systems have received comparatively less
attention. [ 2] discussed the potential for copyright infringement in retrieval-based systems, while [ 10] highlighted the
attribution challenges when LLMs incorporate external knowledge sources.
2.2 Watermarking Techniques for Language Models
Watermarking techniques for language models have evolved significantly in recent years. Traditional approaches
focused on embedding imperceptible signals within generated text to enable ownership verification and detection of
AI-generated content.
[15] introduced a “red-green” watermarking method that biases the token selection process during generation, creating
statistical patterns detectable through hypothesis testing. This approach was further refined by [ 17], who developed
techniques to make watermarks more resilient against adversarial modifications. [ 20] provided a comprehensive
overview of watermarking approaches for generative AI, categorizing methods based on their embedding strategies
and detection capabilities.
More recently, fact-based watermarking has emerged as a promising direction. [ 33] proposed embedding factual
modifications within generated text that serve as identifiable markers while maintaining semantic coherence. [ 5]
introduced PostMark, a post-processing watermarking technique that preserves the original generation quality while
enabling reliable detection. [ 6] explored the theoretical limits of watermarking, identifying fundamental trade-offs
between detectability and text quality.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 5
Jakarta Election Campaigns Heat Up : Here‘s 
How to Understand the System As election day in 
Jakarta draws nearer. …… Initial Launch of 
Candidates’ Campaign Plans on September 1  
After September 1st, when campaign season 
officially kicked off, candidates have moved 
swiftly to engage their bases. Amira Bintang , an 
upstart candidate with extensive social activism 
experience and promising urban development 
and public transportation reform as key platforms 
of her candidacy speech in Jakarta; incumbent 
Rizal Harahap  relies heavily on his track record 
and highlights all infrastructure projects 
completed during his term .……Topic: News Stories
Facts:
• Jakarta election campaigns are intensifying as election day 
approaches.
• Campaign season officially launched on September 1st.
• Amira Bintang and Rizal Harahap  are key candidates in the 
Jakarta election.
•……
Fact A Fact B Relation
Jakarta election 
campaigns are 
intensifying as 
election day 
approaches.Campaign 
season officially 
launched on 
September 1st.Temporal
…… …… ……
Author Pool
Primary field Academic, Journalism, ……
Expertise level Expert, Professional, ……
Subject areasClimate Change, 
International Relations, ……
Writing style ……
Content organization ……
…… ……Select a
unique author
ExtractWriteRepliqa dataKnowledge Extraction
Relations:
Model PoolSelect a
unique modelGenerated Data
Facts[Jakarta election campaigns are 
intensifying as election day approaches ,…]
Relations[<Jakarta election campaigns are 
intensifying as election day approaches, 
Campaign season officially launched on 
September 1st. ,Temporal >, …]
AuthorPrimary field: Journalism, Expertise level: 
Professional, ……
Model GPT -4o
ArticleAs the sun rises over Jakarta , a city 
bursting with ambition and complexity, the 
air is thick with the fervor of election 
season…..
Fig. 2. The construction flowchart of the RPD dataset. The source data is derived from the repliqa dataset, ensuring that the original
data does not appear in the training samples of LLMs to prevent data leakage. For each piece of data in the repliqa dataset, we extract
Facts and Relations. Based on these facts and relations, we select a unique author role from an “Author Pool” composed of writers
with diverse styles. Different LLMs then assume these distinct author roles to select Facts and Relations for writing articles. This
approach effectively simulates the data redundancy issues caused by authors with varying styles in real-world scenarios. Articles
written based on true facts and relations will not conflict with the knowledge already learned by LLMs, and the newly generated data
can avoid data leakage.
However, as noted by [ 14], existing watermarking techniques remain vulnerable to sophisticated evasion strategies,
particularly in the context of RAG systems where content undergoes significant transformation during the retrieval
and generation process.
2.3 Plagiarism Detection in AI-Generated Content
Traditional plagiarism detection methods have primarily focused on identifying verbatim copying or paraphrasing in
human-written text. The emergence of AI-generated content has necessitated new approaches that can identify more
sophisticated forms of content appropriation.
[29] investigated the detectability of LLM-generated text, finding that current detection methods struggle with
high-quality generations from advanced models. [ 26] proposed DetectGPT, a reference-free approach for identifying
text generated by large language models based on curvature in the model’s log probability space. [ 16] examined how
paraphrasing affects the detectability of AI-generated text, revealing significant challenges for current detection systems.
[3] proposes a membership inference attack method against RAG systems, demonstrating its effectiveness across
datasets and models while suggesting preliminary defense strategies. [ 28] introduces a black-box membership inference
attack framework for fine-tuned diffusion models, achieving high effectiveness across multiple attack scenarios with an
AUC of up to 0.95.
In the specific context of RAG systems, [ 13] highlighted the problem of fact redundancy, where multiple sources
contain identical factual information, complicating the attribution process. [ 22] addressed the scalability challenges of
detection methods, proposing efficient algorithms for processing large volumes of potentially appropriated content.
3 RAG Plagiarism Detection Dataset
Current plagiarism detection datasets suffer from several critical limitations when applied to RAG scenarios. First,
they typically focus on verbatim copying or simple paraphrasing, whereas RAG systems fundamentally transform
content through LLM reformulation while preserving core information. Second, they rarely account for the fact
Manuscript submitted to ACM

6 Liu et al.
redundancy problem—where multiple legitimate sources may contain identical factual information, making source
attribution challenging. Finally, most existing datasets lack the diversity in writing styles, domain expertise, and content
organization that characterizes real-world information ecosystems. To address these issues, we provide a newRAG
PlagiarismDetection dataset, namedRPD.
3.1 Dataset Design Philosophy
Our dataset design addresses these limitations through a novel approach that simulates the complex interplay between
content creators, information structures, and linguistic expression. We built our dataset on three foundational principles:
(1)Factual Consistency with Stylistic Diversity:Content should maintain consistent core facts while varying
significantly in expression across different authors and writing styles.
(2)Controlled Redundancy:The dataset should contain deliberate overlaps in factual information across docu-
ments, mirroring real-world information ecosystems where multiple sources cover similar topics.
(3)Recency and Relevance:To avoid the problem of LLMs recognizing content from their training data, the
dataset should contain recent information not included in model training.
3.2 Dataset Generation Methodology
To implement these principles, we developed a multi-stage generation process that transforms source documents into a
diverse collection of stylistically varied but factually consistent articles.
3.2.1 Source Selection and Fact Extraction.We began with therepliqadataset [ 27], which provides recent documents
across diverse domains. This choice was deliberate—repliqacontains information published after the training cutoff
dates of current LLMs, ensuring our dataset represents genuinely novel content that models cannot have memorized
during pre-training.
As shown in Figure 6 and 7, from each source document, we extracted two categories of facts:
•Core Facts:5-7 essential pieces of information that capture the document’s central claims or findings.
•Extended Facts:8-12 supporting details that provide context or elaboration.
Additionally, we identified semantic relationships between facts (causal, temporal, supportive, contradictory, or
elaborative), creating a knowledge graph representation of each document. This structured approach allows us to
maintain factual consistency while enabling stylistic variation in the regenerated content.
3.2.2 Author Profile Simulation.To capture the diversity of real-world content creation, we developed a sophisticated
author simulation system with an author pool containing 100 different author styles (as described in Figure 6 and 8).
Each simulated author is defined along multiple dimensions:
•Domain Expertise:Primary field (academic, journalism, technical writing, etc.), expertise level, and specific
subject areas.
•Writing Style:Formality, technical density, narrative approach, sentence structure, and vocabulary preferences.
•Content Organization:Structure preferences, detail level, and citation style.
•Perspective Biases:Political leaning, technology attitude, and epistemological approach.
This multidimensional characterization allows us to generate content that varies not just in surface-level expression but
in deeper structural and stylistic elements—mirroring the diversity of real-world content while maintaining factual
consistency.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 7
Characteristic Value
Source documents (fromrepliqa) 3,000
Core facts per document 5-7
Extended facts per document 8-12
Author styles in pool 100
LLMs used as authors 4
Extended facts selected per generation 2
Articles per source document 4
Total dataset entries 12,000
Table 1. RPD Dataset Statistics
3.2.3 Multi-Model Content Generation.To further enhance diversity and realism, we employed multiple state-of-the-art
language models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0, and OpenAI o3) to generate content based on the extracted
facts and author profiles. Each model brings its own linguistic patterns and generation tendencies, creating a dataset
with natural variation that better represents the heterogeneity of real-world content.
For each source document, we generated multiple derivative articles that:
•Incorporate all core facts and a subset of extended facts (specifically, 2 randomly selected extended facts)
•Maintain factual accuracy while varying expression
•Adhere to the assigned author profile’s stylistic characteristics (each LLM randomly selects 1 style from the
author pool)
•Include original examples, hypothetical scenarios, or contextual information consistent with the facts
This approach creates a controlled environment where we know precisely which facts appear in which documents,
enabling accurate evaluation of the information sourcing behaviors of the RAG systems. More details can be found in
Figure 9 and 10.
3.3 Dataset Characteristics
The resulting RPD dataset comprises 3,000 original documents from Repliqa, each with 4 derivative articles written in
different styles (one per LLM author), resulting in a total of 12,000 entries as shown in Table 1. This structure creates a
realistic simulation of the information ecosystem that RAG systems typically navigate, where multiple sources may
contain overlapping information expressed in different ways.
4 Methodology
Our approach to detecting unauthorized RAG usage draws inspiration from this rich history of watermarking, adapted
to the unique challenges of neural text generation. We propose a comprehensive framework consisting of two main
components: (1) a dual-layered watermarking system for content protection and (2) an interrogator-detecting framework
for unauthorized usage detection.
4.1 Dual-Layered Watermarking System
The core innovation of our methodology lies in its sequential dual-layered approach, applying two complementary
watermarking techniques in succession to create a robust detection system that is resilient to various evasion attempts.
Manuscript submitted to ACM

8 Liu et al.
4.1.1 Layer 1: Knowledge-Based Watermarking.The first layer embeds distinctive knowledge information within the
content, creating what we call “knowledge watermarks”. Unlike traditional watermarks that modify the medium itself,
knowledge watermarks operate at the semantic level, inserting carefully crafted information that serves as a signature.
The process begins by identifying candidate knowledge that could serve as watermarks. For a given document 𝐷
with a set of original facts 𝐹𝐷={𝑓 1,𝑓2,...,𝑓𝑛}, we construct a semantic embedding space Ewhere each knowledge 𝑓𝑖is
mapped to a embeddinge 𝑖=Embed(𝑓 𝑖)∈R𝑑.
To identify potential watermark knowledge, we compute the semantic similarity between knowledge using cosine
distance:
sim(𝑓𝑖,𝑓𝑗)=e𝑖·e𝑗
||e𝑖||·||e𝑗||.(1)
For each knowledge𝑓 𝑖, we identify a set of𝑘semantically similar knowledgeN 𝑘(𝑓𝑖)that satisfy:
N𝑘(𝑓𝑖)={𝑓𝑗∈F\𝐹𝐷:sim(𝑓𝑖,𝑓𝑗)>𝜏 sim∧sim(𝑓𝑖,𝑓𝑗)<𝜏 ident},(2)
whereFrepresents our universal knowledge corpus, 𝜏simis the minimum similarity threshold, and 𝜏identis the maximum
similarity threshold to avoid knowledge that are essentially identical. An example of sampled knowledge is shwon in
Figure 11.
From these candidates, we select a subset of 𝑚knowledge𝑊𝐷={𝑤 1,𝑤2,...,𝑤𝑚}⊂Ð𝑛
𝑖=1N𝑘(𝑓𝑖)that maximize both
semantic coherence with the document and distinctiveness as watermarks:
𝑊𝐷=argmax𝑊⊂Ð
𝑖N𝑘(𝑓𝑖),|𝑊|=𝑚(Coherence(𝑊,𝐷)+Distinctiveness(𝑊)).(3)
As shown in Figure 12, the Coherence and Distinctiveness scores are evaluated by the Interrogator, which assesses
how naturally the candidate watermark knowledge fits within the document context (Coherence) and how uniquely
identifiable it would be as a watermark (Distinctiveness). This Agent-based evaluation allows us to select watermarks
that maintain document integrity while providing strong detection signals.
These selected watermark knowledge are then strategically integrated into the document using a controlled language
generation process that maintains natural flow and coherence, resulting in a knowledge-watermarked document
𝐷knowledge .
4.1.2 Layer 2: Red-Green Token Distribution Manipulation.After embedding knowledge-based watermarks, we apply
our second layer of protection, which works at the lexical level by subtly manipulating the statistical distribution of
tokens in the generated text. This approach, inspired by the red-green watermarking technique introduced by [ 15],
creates a statistical signature that persists even when the text is paraphrased.
For a vocabularyVand a context window 𝑐𝑡=[𝑡𝑡−𝑘,...,𝑡𝑡−1]preceding position 𝑡, we partition the vocabulary into
“green” tokensG(𝑐 𝑡)and “red” tokensR(𝑐 𝑡)=V\G(𝑐 𝑡)using a seeding functionℎ:
G(𝑐𝑡)={𝑣∈V:ℎ(𝑣,𝑐 𝑡)mod𝛾<𝛾/2},(4)
where𝛾is a parameter controlling the proportion of green tokens.
We apply this second layer of watermarking to the knowledge-watermarked document 𝐷knowledge by regenerating it
with a bias toward green tokens. During text generation, we modify the logits of the language model:
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 9
˜𝑝(𝑣|𝑐𝑡)= 
𝑝(𝑣|𝑐𝑡)·𝑒𝛿
𝑍(𝑐𝑡)if𝑣∈G(𝑐 𝑡),
𝑝(𝑣|𝑐𝑡)
𝑍(𝑐𝑡)if𝑣∈R(𝑐 𝑡),(5)
where𝑝(𝑣|𝑐 𝑡)is the original probability,𝛿is the bias strength, and𝑍(𝑐 𝑡)is a normalization factor:
𝑍(𝑐𝑡)=∑︁
𝑣∈G(𝑐𝑡)𝑝(𝑣|𝑐𝑡)·𝑒𝛿+∑︁
𝑣∈R(𝑐𝑡)𝑝(𝑣|𝑐𝑡).(6)
This creates a subtle statistical pattern where green tokens appear more frequently than expected by chance, yet the
text remains fluent and natural to human readers. The result is a dual-watermarked document 𝐷dualthat contains both
knowledge-based and red-green watermarks.
4.2 Interrogator-Detective Framework for Detection
A critical challenge in detecting unauthorized RAG usage is that watermark signals become progressively weakened
after retrieval and LLM reformulation. To address this challenge, we introduce an Interrogator-Detective framework
that works in tandem to reveal unauthorized data usage.
4.2.1 The Interrogator: Strategic Query Generation.The Interrogator module generates strategic queries designed to
reveal whether a RAG system has access to watermarked content:
𝑄𝐷=Interrogator(𝐷 dual,𝑊𝐷),(7)
where𝑄𝐷is the generated query, 𝐷dualis the dual-watermarked document, and 𝑊𝐷represents the knowledge-based
watermarks.
The Interrogator specifically crafts questions that can only be answered correctly by accessing the knowledge-based
watermarked content. These questions are designed with a critical property: they appear answerable to a RAG system
that has indexed the watermarked content, but would force a system without access to respond with “I don’t know” or
provide an incorrect answer.
Mathematically, we define the ideal interrogation query𝑄∗as:
𝑄∗=argmax𝑄∈Q(𝑃(𝐴 correct|RAG with𝐷 dual,𝑄)−𝑃(𝐴 correct|RAG without𝐷 dual,𝑄)),(8)
whereQis the space of possible queries, and 𝑃(𝐴 correct|condition,𝑄) is the probability of a correct answer given the
condition and query.
The Interrogator’s effectiveness stems from its ability to target the unique combination of knowledge watermarks
embedded in the document, creating queries that are highly specific to watermarked content while appearing natural
and relevant to the document’s topic.
4.2.2 The Detective: Statistical Evidence Analysis.The Detective module analyzes the responses from the suspected
RAG system to determine whether it has accessed watermarked content. Rather than making determinations based on
individual samples, the Detective employs a statistical hypothesis testing framework:
•𝐻 0: The RAG system did not use the watermarked dataset.
•𝐻 1: The RAG system plagiarized the watermarked dataset.
The Detective analyzes both layers of watermarking evidence:
Manuscript submitted to ACM

10 Liu et al.
Knowledge-Based Watermark Detection.For a set of 𝑛queries{𝑄1,𝑄2,...,𝑄𝑛}generated by the Interrogator, the
Detective computes:
𝑆fact=1
𝑛𝑛∑︁
𝑖=1I(𝑅𝑖contains watermarked knowledge),(9)
where𝑅𝑖is the RAG system’s response to query𝑄 𝑖, andI(·)is the indicator function.
Under the null hypothesis 𝐻0(no plagiarism), 𝑆factfollows a binomial distribution with parameters 𝑛and𝑝0, where
𝑝0is the probability of correctly answering by chance. The Detective rejects𝐻 0if:
𝑆fact>𝜏fact=𝑝 0+𝑧𝛼√︂
𝑝0(1−𝑝 0)
𝑛,(10)
where𝑧𝛼is the critical value corresponding to significance level𝛼.
Red-Green Token Distribution Detection.For the same set of responses, the Detective analyzes the statistical distribution
of tokens:
𝑆token=Í𝑛
𝑖=1Í
𝑡∈𝑅𝑖I(𝑡∈G(𝑐 𝑡))Í𝑛
𝑖=1|𝑅𝑖|,(11)
where|𝑅𝑖|is the length of response𝑅 𝑖in tokens.
Under𝐻0,𝑆token should be approximately 0.5 (equal distribution of red and green tokens). The Detective rejects 𝐻0if:
𝑍token=𝑆token−0.5√︃
0.25Í𝑛
𝑖=1|𝑅𝑖|>𝑧𝛼.(12)
Combined Detection Decision.The Detective combines evidence from both watermarking layers to make a final
determination:
Decision= 
RAG Theft! if(𝑆 fact>𝜏fact)∨(𝑍 token>𝑧𝛼),
Innocent otherwise.(13)
This dual-criteria approach significantly reduces both false positives and false negatives compared to single-layer
detection methods, as each layer provides complementary evidence of unauthorized usage. An example of a dual-layer
watermarked article is shown in Figure 14, and the query specifically generated for dual-layer watermarked the article
is shown in Figure 15.
5 Mathematical Analysis of the Dual-Layered Watermarking System
In this section, we provide a rigorous mathematical analysis of our dual-layered watermarking approach for detecting
unauthorized RAG usage. Our method combines knowledge-based watermarking with token distribution manipulation
to create a robust detection framework.
5.1 Analysis of Token Distribution Watermarking
Following the red-green watermarking approach, we partition the vocabulary Vinto “green” tokens G(𝑐𝑡)and “red”
tokensR(𝑐 𝑡)=V\G(𝑐 𝑡)using a seeding functionℎ:
G(𝑐𝑡)={𝑣∈V:ℎ(𝑣,𝑐 𝑡)mod𝛾<𝛾/2},(14)
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 11
where𝛾controls the proportion of green tokens and 𝑐𝑡=[𝑡𝑡−𝑘,...,𝑡𝑡−1]is the context window preceding position 𝑡.
During text generation, we modify the logits of the language model to bias toward green tokens:
˜𝑝(𝑣|𝑐𝑡)= 
𝑝(𝑣|𝑐𝑡)·𝑒𝛿
𝑍(𝑐𝑡)if𝑣∈G(𝑐 𝑡),
𝑝(𝑣|𝑐𝑡)
𝑍(𝑐𝑡)if𝑣∈R(𝑐 𝑡),(15)
where𝑝(𝑣|𝑐 𝑡)is the original probability,𝛿is the bias strength, and𝑍(𝑐 𝑡)is a normalization factor:
𝑍(𝑐𝑡)=∑︁
𝑣∈G(𝑐𝑡)𝑝(𝑣|𝑐𝑡)·𝑒𝛿+∑︁
𝑣∈R(𝑐𝑡)𝑝(𝑣|𝑐𝑡).(16)
5.2 Detection Sensitivity Analysis
Let𝛼=𝑒𝛿. For a document with token sequence 𝑠=[𝑠 1,𝑠2,...,𝑠𝑇], we define|𝑠|𝐺as the number of green tokens in the
sequence. We can analyze the expected number of green tokens and its variance.
Theorem 5.1.For a watermarked text sequence of 𝑇tokens generated with bias strength 𝛿and green list proportion 𝛾, if
the average spike entropy of the sequence is at least𝑆∗, then:
E[|𝑠|𝐺]≥𝛾𝛼𝑇
1+(𝛼−1)𝛾𝑆∗(17)
Furthermore, the variance of the green token count is bounded by:
Var[|𝑠|𝐺]≤𝑇𝛾𝛼𝑆∗
1+(𝛼−1)𝛾
1−𝛾𝛼𝑆∗
1+(𝛼−1)𝛾
(18)
where the spike entropy𝑆(𝑝,𝑧)of a probability distribution𝑝with modulus𝑧is defined as:
𝑆(𝑝,𝑧)=∑︁
𝑘𝑝𝑘
1+𝑧𝑝𝑘(19)
Proof.For each token position𝑡, the probability of selecting a green token is:
𝑃(𝑠𝑡∈G)=Í
𝑣∈G(𝑐𝑡)𝑝(𝑣|𝑐𝑡)·𝑒𝛿
𝑍(𝑐𝑡)(20)
Following the analysis in the reference paper, we can show that:
𝑃(𝑠𝑡∈G)≥𝛾𝛼
1+(𝛼−1)𝛾𝑆
𝑝,(1−𝛾)(𝛼−1)
1+(𝛼−1)𝛾
(21)
The expected number of green tokens is the sum of these probabilities across all positions:
E[|𝑠|𝐺]=𝑇∑︁
𝑡=1𝑃(𝑠𝑡∈G)≥𝛾𝛼𝑇
1+(𝛼−1)𝛾𝑆∗(22)
Since each token selection is independent (conditioned on the context), the variance is the sum of the variances of
Bernoulli random variables [7]:
Var[|𝑠|𝐺]=𝑇∑︁
𝑡=1𝑃(𝑠𝑡∈G)(1−𝑃(𝑠 𝑡∈G))(23)
By Jensen’s inequality [25] and the concavity of the variance function𝑝(1−𝑝), we obtain:
Manuscript submitted to ACM

12 Liu et al.
Var[|𝑠|𝐺]≤𝑇𝛾𝛼𝑆∗
1+(𝛼−1)𝛾
1−𝛾𝛼𝑆∗
1+(𝛼−1)𝛾
(24)
□
5.3 Impact on Text Quality
The impact of our watermarking approach on text quality can be bounded as follows:
Theorem 5.2.For a token at position 𝑡with original probability distribution 𝑝(𝑡)and watermarked distribution ˆ𝑝(𝑡),
the expected perplexity with respect to the randomness of the partition is bounded by:
E𝐺,𝑅∑︁
𝑘ˆ𝑝(𝑡)
𝑘ln(𝑝(𝑡)
𝑘)≤(1+(𝛼−1)𝛾)𝑃∗(25)
where𝑃∗=Í
𝑘𝑝(𝑡)
𝑘ln(𝑝(𝑡)
𝑘)is the perplexity of the original model.
Proof.The probability of sampling token𝑘from the modified distribution is:
ˆ𝑝𝑘=E𝐺,𝑅𝛼I(𝑘∈𝐺)𝑝𝑘Í
𝑖∈𝑅𝑝𝑖+𝛼Í
𝑖∈𝐺𝑝𝑖(26)
We can decompose this expectation based on whether𝑘is in𝐺or𝑅:
ˆ𝑝𝑘=𝛾·E𝐺,𝑅|𝑘∈𝐺𝛼𝑝𝑘Í
𝑖∈𝑅𝑝𝑖+𝛼Í
𝑖∈𝐺𝑝𝑖
+(1−𝛾)·E 𝐺,𝑅|𝑘∈𝑅𝑝𝑘Í
𝑖∈𝑅𝑝𝑖+𝛼Í
𝑖∈𝐺𝑝𝑖
≤𝛾𝛼𝑝𝑘+(1−𝛾)𝑝 𝑘=(1+(𝛼−1)𝛾)𝑝 𝑘(27)
The expected perplexity is then:
E𝐺,𝑅∑︁
𝑘ˆ𝑝(𝑡)
𝑘ln(𝑝(𝑡)
𝑘)=∑︁
𝑘E𝐺,𝑅ˆ𝑝(𝑡)
𝑘ln(𝑝(𝑡)
𝑘)(28)
≤∑︁
𝑘(1+(𝛼−1)𝛾)𝑝(𝑡)
𝑘ln(𝑝(𝑡)
𝑘)(29)
=(1+(𝛼−1)𝛾)𝑃∗(30)
□
5.4 Resilience Against Adversarial Attacks
Let’s analyze the resilience of our dual-layered approach against adversarial modifications. Suppose an adversary
modifies𝑚tokens in a sequence of length𝑇.
For the token distribution watermark, each modified token can affect at most 𝑘+1tokens (itself and 𝑘subsequent
tokens whose red-green lists depend on it). The maximum reduction in the z-statistic is:
Δ𝑍token≤(𝑘+1)𝑚√
𝑇(31)
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 13
For the knowledge-based watermark, the adversary would need to identify and modify the specific watermarked
knowledge, which requires knowledge of which facts serve as watermarks. If the adversary randomly modifies tokens,
the probability of removing all watermarked knowledge is:
𝑃(remove all watermarks)≤𝑚
𝑇|𝑊𝐷|
(32)
where|𝑊𝐷|is the number of knowledge-based watermarks. This demonstrates that our dual-layered approach requires
an adversary to make substantial modifications to the text to evade detection, which would significantly degrade the
quality and utility of the generated content.
6 Experiments
6.1 Experimental Setup
To evaluate the effectiveness of our dual-layered watermarking approach for detecting unauthorized RAG usage, we
conducted comprehensive experiments using our RPD dataset. This section details the experimental configuration and
implementation specifics.
6.1.1 RAG System Configuration.We implemented a standard RAG pipeline consisting of an embedding-based retrieval
system followed by a language model for content generation. The key components of our experimental setup include:
•Retrieval System:We utilized OpenAI’s text-embedding-3-small1model to generate embedding represen-
tations of documents in our corpus. For efficient similarity search, we implemented an approximate nearest
neighbor search using thehnswliblibrary [24], which provides a balance between search speed and accuracy.
•Retrieval Parameters:For each query, we set the number of retrieved documents to 𝐾= 3, which represents a
typical configuration in production RAG systems that balances context richness with computational efficiency.
•Large Language Models:To ensure robustness across different generation capabilities, we employed multiple
state-of-the-art language models for the generation component of our RAG system, including variants from the
OpenAI, Gemini, and Claude families.
The rag prompt is shown in Figure 16.
6.1.2 Watermarking Implementation.We implemented both layers of our watermarking approach with the following
parameters:
•Knowledge-based Watermarking:For embedding semantic-level watermarks, we utilized OpenAI’s text
-embedding-3-smallmodel to compute semantic similarities between Knowledge. This allowed us to identify
candidate watermark Knowledge that maintain semantic coherence while being distinctive enough to serve as
effective watermarks. The prompt of knowledge-based watermarking is shown in Figure 12.
•Red-Green Token Distribution:For the lexical-level watermarking, we set 𝛾=0.25, meaning approximately
25% of the vocabulary was designated as “green” tokens for any given context. We applied a bias strength
of𝛿=4.0to increase the probability of selecting green tokens during text generation, creating a detectable
statistical pattern while maintaining text fluency. The prompt of red-green watermarking is shown in the top of
Figure 13.
1https://platform.openai.com/docs/models/text-embedding-3-small
Manuscript submitted to ACM

14 Liu et al.
6.1.3 Baselines.To comprehensively evaluate our dual-layered watermarking approach, we compared it against several
state-of-the-art methods for detecting unauthorized data usage in generative AI systems. These baselines represent
different approaches to the problem, ranging from membership inference attacks to specialized watermarking techniques:
1.AAG (Attacking Augmented Generation)[ 3]: This method focuses on membership inference attacks against RAG
systems. AAG creates carefully crafted prompts designed to determine whether specific text passages exist in a retrieval
database by analyzing the RAG system’s outputs. Originally developed to highlight privacy vulnerabilities in RAG
systems, AAG can be repurposed to detect unauthorized data usage by treating it as a membership inference problem.
2.Facts: A strong baseline proposed by [ 13], which prompts an auxiliary LLM to generate a single question that is
only answerable by reading a document. Then, this question will be sent to the suspicious RAG system. If the RAG
system successfully answers the question, it indicates that data theft has occurred. 3.WARD (Watermarking for
RAG Dataset Inference)[ 13]: This method represents the current state-of-the-art approach specifically designed for
RAG Dataset Inference (RAG-DI). WARD employs LLM watermarks to provide data owners with statistical guarantees
regarding the misuse of their content in RAG corpora.
6.1.4 Evaluation Metrics and Dataset Configuration.To comprehensively evaluate our detection method, we established
two distinct dataset configurations:
•D-IN (Dataset-Included):RAG systems containing unauthorized watermarked documents in their retrieval
corpus. These systems represent the positive cases where data theft has occurred.
•D-OUT (Dataset-Excluded):RAG systems without any watermarked documents in their retrieval corpus. These
systems represent the negative cases where no data theft has occurred.
For each configuration, we report the following evaluation metrics:
•True Positives (TP):Number of D-IN systems correctly identified as containing unauthorized data.
•False Negatives (FN):Number of D-IN systems incorrectly classified as not containing unauthorized data.
•True Negatives (TN):Number of D-OUT systems correctly identified as not containing unauthorized data.
•False Positives (FP):Number of D-OUT systems incorrectly classified as containing unauthorized data.
•Accuracy (Acc.):The proportion of correct classifications across both D-IN and D-OUT systems, calculated as
Acc.=TP+TN
TP+TN+FP+FN.
We conducted experiments under two conditions: (1) without factual redundancy, where each fact appears in only
one document, and (2) with factual redundancy, where similar facts appear across multiple documents with different
expressions, simulating real-world information ecosystems.
6.2 Robustness of RAG Plagiarism Detection
To validate the effectiveness of our dual-layered watermarking approach, we designed a comprehensive experimental
framework that simulates real-world RAG scenarios. Our experiments specifically address a critical challenge in RAG
plagiarism detection: while the watermark signal in any single piece of content may be too weak to reliably detect
misuse, accumulated evidence across multiple queries can provide statistically significant detection power. We split
RPDinto two distinct datasets to simulate authentic RAG environments:
•Unauthorized Dataset:Content embedded with our dual-layered watermarks (both fact-based and red-green
token distribution watermarks).
•Authorized Dataset:Similar content without watermarks.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 15
Number of Responses𝒁token  𝑍𝛼
Number of Responses𝜏fact𝑺fact 
a) Comparison of trends in 𝑆fact and 𝜏fact as the number of responses increases,
 with and without Fact -based watermark in the dataset.b) Comparison of trends in 𝑍token  and 𝑍𝛼 as the number of responses increases,
 with and without the red-green  watermark in the dataset.
Fig. 3. Experimental results on the robustness detection of red-green watermark and Fact-based watermark. Using a z-test (𝛼= 0.005).
These datasets were thoroughly intermixed to simulate the massive, heterogeneous knowledge bases typically used in
production RAG systems. This mixed dataset was then indexed and made available to our experimental RAG system,
allowing us to control the proportion of watermarked content while maintaining realistic retrieval dynamics.
Our evaluation focused on two key metrics that correspond to our dual-layered approach:
(1)Fact-based Watermark Detection:Measured by 𝑆fact, the proportion of responses that are able to answer the
query.
(2)Red-Green Token Distribution Detection:Measured by 𝑍token, the statistic of the standardized test for the
bias of the token distribution.
For each experiment, we deployed the Interrogator module to generate queries specifically designed to target water-
marked content. These queries were constructed to be answerable only by accessing the watermarked information,
forcing the RAG system to either retrieve the watermarked content (if available) or respond with “Unanswerable” when
the information could not be found.
6.2.1 Fact-based Watermark Detection Results.We first evaluated the robustness of our fact-based watermarking
approach by analyzing how detection performance scales with query volume. Figure 3a illustrates the relationship
between the number of queries and the detection statistic𝑆 fact.
When the RAG system had access to the mixed watermarked dataset, we observed that 𝑆factincreased steadily
with the number of responses and eventually stabilized around0 .88, significantly exceeding the detection threshold
𝜏fact=0.86(calculated using𝛼=0.005). This indicates strong statistical evidence of the use of unauthorized content.
In contrast, when the RAG system used only the non-watermarked dataset, 𝑆factshowed no increasing trend with
query volume and stabilized around0 .53, remaining consistently below the detection threshold 𝜏fact=0.88. This
confirms that our detection method maintains a low false positive rate even with large query volumes.
Statistical significance was assessed using a z-test [ 35] with𝛼=0.005, demonstrating that the observed differences
were highly unlikely to occur by chance. These results validate that our fact-based watermarking approach provides
robust detection capabilities when sufficient queries are processed, despite the potential weakness of individual
watermark signals.
Manuscript submitted to ACM

16 Liu et al.
6.3 Red-Green Token Distribution Detection Results
We next evaluated the robustness of our red-green token distribution watermarking approach. Figure 3b shows how
the test statistic𝑍 token varies with increasing query volume.
When the RAG system used the mixed watermark dataset, 𝑍token exhibited a clear increasing trend with the number
of responses, eventually stabilizing around6 .23, significantly exceeding the critical value 𝑧𝛼=3.48(corresponding
to𝛼=0.005). This strong statistical signal confirms that the token distribution bias introduced by our watermarking
technique persists even after LLM reformulation in the RAG process.
In contrast, when the RAG system accessed only the non-watermarked dataset, 𝑍token did not show a consistent
increasing trend, fluctuating around0 .78and remaining well below the critical value 𝑧𝛼=3.48. This demonstrates that
our token distribution watermarking maintains a low false-positive rate, correctly identifying systems that do not use
watermarked content.
Method LLMWithout Factual Redundancy With Factual Redundancy
D-IN D-OUTAcc.↑D-IN D-OUTAcc.↑
TP↑FN↓TN↑FP↓TP↑FN↓TN↑FP↓
AAG [3]GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
Facts [13]GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 0/10 10/10 0.5
WARD [13]GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Dual-Layered (Ours)GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Table 2. Comprehensive detection performance across LLMs and dataset conditions
6.4 Comparison Results
To evaluate the effectiveness of our dual-layered watermarking approach against existing methods, we conducted
comprehensive experiments using the RPD dataset. We constructed an indexed dataset from RPD, watermarked200
documents to simulate unauthorized data, and deployed the Interrogator to generate targeted queries based on the
watermarked content. These queries were then sent to RAG systems built with four widely-used LLMs: GPT-4.1, OpenAI
o3, Claude 3.5 Sonnet, and Gemini-2.0. Each experiment was repeated10times with different random seeds to ensure
statistical robustness. We list all prompts, and examples of RPD samples and watermarked documents, in Appendix A.
6.4.1 Detection Performance without Factual Redundancy.As shown in Table 2, all methods performed exceptionally
well on datasets without factual redundancy, achieving perfect detection accuracy of100%across all LLMs. This confirms
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 17
our hypothesis that in the absence of factual redundancy, detection is relatively straightforward since the queries
constructed by the Interrogator can precisely match the watermarked documents without interference from similar
content. In this scenario, both baseline methods and our approach successfully identified all instances of unauthorized
data usage while correctly classifying systems without watermarked content.
6.4.2 Detection Performance with Factual Redundancy.The more challenging and realistic scenario involves datasets
with factual redundancy, where multiple documents contain similar or identical core facts expressed in different writing
styles. This setting more accurately reflects real-world information ecosystems and poses a significant challenge for
detection methods. The results in Table 2 reveal substantial differences in performance among the methods under these
conditions:
•AAG [ 3]:Performance degraded substantially, with accuracy dropping to50%across all LLMs. Further analysis
revealed that AAG consistently misclassified RAG systems that did not contain watermarked data (D-OUT)
as containing unauthorized content, resulting in a100%false positive rate. This occurs because AAG’s core
detection mechanism relies on directly querying the LLM about document inclusion in the RAG corpus. When
factually similar documents exist, the LLM can be misled into incorrectly confirming the presence of specific
content.
•Facts [ 13]:Similarly, this method achieved only50%accuracy with factual redundancy. While it correctly
identified all D-IN systems (perfect true positive rate), it failed to correctly classify any D-OUT systems, resulting
in a100%false positive rate. This is because when factual redundancy exists in the RAG dataset, even without
watermarked documents, the LLM can obtain answers from other documents that contain similar information,
leading to incorrect classification.
•WARD [ 13]:This method maintained strong performance even with factual redundancy, achieving100%accuracy
across all LLMs. Its effectiveness stems from the statistical token distribution patterns that remain detectable
even when similar content is retrieved, providing resilience against factual redundancy challenges.
•Dual-Layered (Ours):Our approach also achieved perfect100%accuracy across all LLMs in the presence of
factual redundancy. This robust performance can be attributed to the synergistic effect of combining knowledge-
based watermarks with token distribution manipulation, creating distinctive signatures that stand out even
among similar information.
These results demonstrate that while simple fact-based or membership inference approaches may work well in
controlled environments without factual redundancy, they fail to maintain reliability in more realistic scenarios. In
contrast, both WARD and our dual-layered approach provide robust detection capabilities that remain effective even
in the presence of factual redundancy, with our method offering the additional benefit of a multi-layered protection
mechanism that can withstand various evasion attempts.
6.5 Adversarial Experiments
To evaluate the robustness of our approach against sophisticated evasion attempts, we conducted adversarial experiments
simulating realistic scenarios where RAG data thieves actively attempt to conceal their unauthorized data usage. Since
our previous experiments demonstrated that AAG and Facts methods fail to handle factual redundancy (achieving only
50% accuracy), we focused our adversarial analysis on comparing our Dual-Layered approach against WARD [ 13], the
current state-of-the-art method. We implemented two common adversarial tactics employed by RAG data thieves:
Manuscript submitted to ACM

18 Liu et al.
Method LLMKnowledge Watermark Evasion Token Distribution Evasion
D-IN D-OUTAcc.↑D-IN D-OUTAcc.↑
TP↑FN↓TN↑FP↓TP↑FN↓TN↑FP↓
WARD [13]GPT-4.1 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
OpenAI o3 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
Claude 3.5 10/10 0/10 10/10 0/10 1.0 4/10 6/10 10/10 0/10 0.7
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
Dual-Layered (Ours)GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Table 3. Comprehensive detection performance under adversarial conditions
•Knowledge Watermark Evasion:This technique targets knowledge-based watermarks by using LLMs to
detect incoherently spliced facts within documents. When potentially watermarked documents are identified,
the RAG system is instructed to avoid referencing these suspicious documents, thereby attempting to evade
detection. The prompt is shown in Figure 17.
•Token Distribution Evasion:This approach targets token-distribution watermarks by employing LLMs to
completely rewrite referenced articles, minimizing the use of original words and phrases to disrupt statistical
patterns embedded in the text. The prompt is shown in Figure 18.
These evasion techniques represent realistic adversarial scenarios that any practical watermarking system must
withstand to provide reliable protection against unauthorized data usage. Table 3 presents the performance of both
methods under adversarial conditions. The results reveal significant differences in robustness:
•Knowledge Watermark Evasion:Both WARD and our Dual-Layered approach maintained perfect detection
accuracy (100%) across all LLMs when facing knowledge watermark evasion attempts. This suggests that token
distribution-based watermarking techniques remain effective even when adversaries attempt to identify and
avoid watermarked documents.
•Token Distribution Evasion:WARD’s performance degraded catastrophically when facing token distribution
evasion, with accuracy dropping to 50% for GPT-4.1, OpenAI o3, and Gemini 2.0, and 70% for Claude 3.5. The true
positive rate (TP) dropped to 0/10 for three of the four LLMs, indicating complete failure to detect unauthorized
data usage when adversaries employ rewriting techniques. In contrast, our Dual-Layered approach maintained
perfect 100% accuracy across all LLMs, demonstrating exceptional resilience against token distribution evasion
attempts.
The stark difference in performance under token distribution evasion can be attributed to the fundamental design
principles of each approach:
•WARD relies primarily on statistical patterns in token distributions, which can be effectively disrupted through
comprehensive rewriting. When adversaries employ LLMs to reformulate content while preserving core infor-
mation, the statistical watermarks become undetectable.
•Our Dual-Layered approach achieves superior robustness through its sequential integration of complementary
watermarking techniques. By combining knowledge-based watermarks with token distribution manipulation,
our method creates redundant protection mechanisms that remain effective even when one layer is compromised:
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 19
Method LLMKnowledge Watermark Evasion Token Distribution Evasion
D-IN D-OUTAcc.↑D-IN D-OUTAcc.↑
TP↑FN↓TN↑FP↓TP↑FN↓TN↑FP↓
KW-OnlyGPT-4.1 4/10 6/10 10/10 0/10 0.7 10/10 0/10 10/10 0/10 1.0
OpenAI o3 6/10 4/10 10/10 0/10 0.8 10/10 0/10 10/10 0/10 1.0
Claude 3.5 8/10 2/10 10/10 0/10 0.9 10/10 0/10 10/10 0/10 1.0
Gemini-2.0 4/10 6/10 10/10 0/10 0.7 10/10 0/10 10/10 0/10 1.0
TD-OnlyGPT-4.1 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
OpenAI o3 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
Claude 3.5 10/10 0/10 10/10 0/10 1.0 4/10 6/10 10/10 0/10 0.7
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 0/10 10/10 10/10 0/10 0.5
Dual-Layered (Ours)GPT-4.1 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
OpenAI o3 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Claude 3.5 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Gemini-2.0 10/10 0/10 10/10 0/10 1.0 10/10 0/10 10/10 0/10 1.0
Table 4. Ablation study of Knowledge-based watermarking and Red-Green Token Distribution Manipulation.
–The knowledge-based watermarks are designed with careful semantic coherence, making them difficult to
detect and filter out without compromising information retrieval quality.
–Our Interrogator module generates queries specifically targeting the unique knowledge combinations embedded
in watermarked documents. Even if adversaries completely rewrite the documents, they cannot answer these
queries without referencing the watermarked content.
–The dual-layer design ensures that even if token distribution patterns are disrupted through rewriting, the
knowledge-based watermarks remain intact and detectable.
6.6 Ablation Study
To understand the individual contributions of each watermarking layer and validate the synergistic benefits of our
dual-layered approach, we conducted a comprehensive ablation study. This analysis isolated the performance of
knowledge-based watermarking (KW-Only) and token distribution manipulation (TD-Only) against the full dual-layered
implementation across various adversarial scenarios. Table 4 presents the results of this ablation study, revealing critical
insights into the complementary nature of our watermarking techniques:
6.6.1 Knowledge-Based Watermarking in Isolation.When knowledge-based watermarking was employed as a standalone
technique (KW-Only), we observed:
•Vulnerability to Knowledge Watermark Evasion:Performance degraded significantly when facing targeted
evasion attempts, with accuracy dropping to 70-90% across different LLMs. GPT-4.1 and Gemini-2.0 showed
particular vulnerability with only 4/10 true positives, while Claude 3.5 demonstrated greater resilience with 8/10
true positives.
•Resilience Against Token Distribution Evasion:Interestingly, knowledge-based watermarking maintained
perfect 100% accuracy when facing token distribution evasion attempts. This confirms our hypothesis that
semantic-level watermarks remain detectable even when adversaries completely rewrite the text, as long as the
core information is preserved.
Manuscript submitted to ACM

20 Liu et al.
This pattern illustrates a fundamental characteristic of knowledge-based watermarking: its effectiveness depends on
the preservation of specific factual information rather than lexical or syntactic patterns. When adversaries specifically
target knowledge watermarks by filtering out suspicious documents, detection capability suffers. However, when they
focus on rewriting content while preserving core information, knowledge watermarks remain robust.
6.6.2 Token Distribution Manipulation in Isolation.The token distribution approach (TD-Only) exhibited the inverse
pattern:
•Resilience Against Knowledge Watermark Evasion:Token distribution watermarking achieved perfect 100%
accuracy across all LLMs when facing knowledge watermark evasion. This demonstrates that statistical patterns
in token distributions remain detectable even when adversaries attempt to identify and filter out watermarked
documents.
•Vulnerability to Token Distribution Evasion:Performance collapsed dramatically when facing targeted
rewriting, with accuracy dropping to 50% for most LLMs (70% for Claude 3.5). The true positive rate fell to 0/10
for three of the four LLMs, indicating complete failure to detect unauthorized data usage when adversaries
employ comprehensive rewriting techniques.
This pattern reveals the complementary weakness of token distribution watermarking: while it creates robust statistical
signatures that are difficult to identify through document filtering, these signatures can be effectively disrupted through
comprehensive rewriting that preserves semantic content while altering lexical choices.
6.6.3 The Synergistic Effect of Dual-Layered Watermarking.The full dual-layered approach demonstrated perfect
100% accuracy across all experimental conditions, highlighting the synergistic protection achieved through sequential
integration:
•Complementary Protection:Each layer effectively compensates for the vulnerabilities of the other. When
adversaries target knowledge watermarks through filtering, the token distribution layer maintains detection
capability. Conversely, when adversaries employ rewriting to disrupt token distributions, the knowledge-based
layer preserves detection capability.
•Increased Adversarial Cost:To evade detection, adversaries would need to simultaneously implement both
evasion strategies—filtering out watermarked documents and comprehensively rewriting any remaining content
that might contain watermarks. This substantially increases the computational and quality costs of evasion
attempts.
•Cross-Layer Reinforcement:The sequential application of both watermarking techniques creates a reinforced
protection mechanism where the knowledge-based watermarks guide the Interrogator to generate queries
that specifically target watermarked content, while the token distribution layer provides statistical evidence of
unauthorized usage even when content is partially reformulated.
This ablation study empirically validates our theoretical framework for dual-layered watermarking. The results demon-
strate that while each individual watermarking technique has specific vulnerabilities to targeted evasion attempts, their
sequential integration creates a robust detection system that maintains effectiveness across diverse adversarial scenarios.
This synergistic effect is particularly valuable in real-world applications where RAG system operators may employ
various sophisticated techniques to conceal unauthorized data usage. The complementary nature of these watermarking
approaches—one operating at the semantic level and the other at the lexical level—creates a protection mechanism that
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 21
b) Ducument  Quality Score across different dimensionsa) P -SP Scores of Different Watermarks. 
Knowledge -based
Token -Distribution basedDual -Layered
Original
1.0 0.9930.941 0.934
c) Overall Quality  Score of all dimensions . 8.86 8.81 8.59 8.57
Fig. 4. Text quality assessment across different watermarking approaches.
addresses the fundamental tension in watermarking: creating marks that are simultaneously imperceptible to casual
readers yet reliably detectable by authorized verification systems, even in the face of determined adversarial efforts.
6.7 Impact on Text Quality
A critical consideration for any watermarking technique is its impact on the quality of the watermarked text. Effective
watermarks must be imperceptible to human readers while remaining detectable by verification systems. To evaluate
this aspect of our approach, we conducted a comprehensive quality assessment comparing original documents against
their watermarked counterparts.
6.7.1 Evaluation Methodology.We employed two complementary evaluation approaches to assess the quality impact
of our watermarking techniques:
•LLM-based Evaluation:Following established practices in natural language generation evaluation [ 8,21],
we utilized GPT-4.1 as an evaluator to assess text quality across multiple dimensions. This approach provides
a multi-faceted analysis of text quality from a human-like perspective. Detailed prompts can be found in the
bottom of Figure 13.
•Embedding-based Evaluation:We employed the P-SP metric [ 31], which measures semantic preservation
through embedding similarity. This provides an objective, quantitative measure of how closely watermarked text
preserves the semantic content of the original.
For each document in our test set, we created three variants: (1) with knowledge-based watermarking only, (2) with
token distribution watermarking only, and (3) with our full dual-layered watermarking approach. These variants, along
with the original unwatermarked document, were evaluated on seven quality dimensions using GPT-4.1:
•Coherence:Logical flow and connection between ideas
•Factual Accuracy:Correctness of presented information
Manuscript submitted to ACM

22 Liu et al.
•Grammar:Syntactic correctness and linguistic fluency
•Relevance:Alignment with the document’s intended topic
•Depth:Thoroughness and comprehensiveness of content
•Creativity:Originality and engaging presentation
•Overall Score:Composite assessment of document quality
Each dimension was scored on a scale from1to10, with higher scores indicating better quality. For the P-SP metric,
scores range from0to1, with higher values indicating greater semantic preservation relative to the original document.
6.7.2 Quality Assessment Results.Figure 4 presents the results of our quality assessment, revealing several important
insights about the impact of different watermarking approaches on text quality.
Knowledge-based Watermarking:This approach demonstrated the smallest quality degradation, with an overall
GPT-4.1 score of8 .81compared to8 .87for original documents—a minimal reduction of only 0.06 points. The P-SP
score of0.993further confirms excellent semantic preservation, with only a0 .7%reduction from the original. The
preservation of quality can be attributed to the semantic integration of watermark facts, which maintains document
coherence (8 .99) and relevance (9 .99). The largest impact was observed in grammar (9 .49vs.9.70), likely due to the
insertion of additional factual statements that may occasionally disrupt the original syntactic flow.
Token-Distribution Watermarking:This technique showed more noticeable quality impacts, with an overall
GPT-4.1 score of8 .60, a reduction of0 .27points from the original. The P-SP score dropped to0 .941, indicating a more
substantial semantic shift. The most significant degradations occurred in creativity (7 .71vs.8.08) and depth (8 .39vs.
8.84). This aligns with theoretical expectations, as biasing token selection toward “green” tokens constrains the model’s
lexical choices, potentially limiting expressive range and stylistic variation.
Dual-Layered Watermarking:Our combined approach resulted in an overall GPT-4.1 score of8 .57, representing a
0.30point reduction from the original—only marginally lower than token-distribution watermarking alone. The P-SP
score of0.934indicates a6 .6%reduction in semantic preservation compared to the original. This suggests that the
sequential application of both watermarking techniques does not substantially compound quality degradation beyond
what is observed with token distribution manipulation alone.
6.7.3 Quality-Security Tradeoff Analysis.The results reveal a clear tradeoff between watermarking security and text
quality. Knowledge-based watermarking preserves quality but, as shown in our ablation study (Table 4), is vulnerable to
targeted evasion. Conversely, token distribution watermarking provides stronger security against knowledge watermark
evasion but impacts text quality more significantly.
Our dual-layered approach optimizes this tradeoff by combining both techniques sequentially. While it shows a minor
reduction in overall quality compared to unwatermarked text, this modest degradation is justified by the substantial
security benefits—achieving perfect detection accuracy across all adversarial scenarios. Importantly, all watermarked
variants maintained high absolute quality scores ( >8.5/10), indicating that the watermarked text remains highly
readable and effective for its intended purpose.
The dimension-specific analysis provides additional insights for potential optimization. The minimal impact on
relevance (9 .95vs.9.99) and coherence (8 .92vs.9.00) suggests that our watermarking approach preserves the core
communicative function of the text. The more noticeable impacts on creativity (7 .64vs.8.08) and depth (8 .34vs.8.84)
indicate areas where future refinements could focus on preserving stylistic richness while maintaining security.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 23
0.99 0.99 0.990.98 0.98 0.980.970.96
0.94
0.92
0.89
0.80.90.90.90.90.91.01.01.0
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0
δPSP Score
(a) PSP Score under different δ values0 00.20.40.81 1 1 1 1
0.00.20.40.60.81.0
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0
δAccuracy
8.858.81 8.82 8.81 8.798.758.71
8.6
8.4 8.4
8.18.28.38.48.58.68.78.88.9
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0
δGPT Overall Score
(b) Detect Accuracy under different δ values (c) Overall Score under different δ values
 (d) Ducument  Quality Score across different 
dimensions under different δ values
Fig. 5. The impact of delta variation on text quality and detection accuracy.
This quality assessment demonstrates that our dual-layered watermarking approach achieves an effective balance in
the fundamental tension between imperceptibility and security—creating watermarks that remain largely transparent
to human readers while providing robust protection against unauthorized data usage, even in adversarial scenarios.
6.8 Hyperparameter Analysis
The effectiveness of our dual-layered watermarking approach depends significantly on the configuration of key
hyperparameters. In this section, we focus on analyzing the impact of 𝛿, the bias strength parameter that controls the
intensity of token distribution manipulation in our second watermarking layer.
To understand the relationship between 𝛿and system performance, we conducted a systematic analysis varying 𝛿
from1.0to6.0in increments of0.5. For each value, we evaluated:
•Detection Accuracy:The proportion of correct classifications in our standard evaluation framework, measuring
the system’s ability to identify unauthorized RAG usage.
•Semantic Preservation:Quantified using the P-SP metric [ 31], which measures how well the watermarked text
preserves the semantic content of the original document.
•Text Quality:Assessed using GPT-4.1 as an evaluator across multiple dimensions including coherence, factual
accuracy, grammar, depth, creativity, and overall quality.
This comprehensive evaluation allows us to identify the optimal configuration that balances detection effectiveness
with minimal impact on text quality.
6.8.1 Impact on Detection Accuracy.Figure 5b illustrates the relationship between 𝛿and detection accuracy. We observe
a clear threshold effect:
•At low values ( 𝛿≤2.0), the detection accuracy is0, indicating that the token distribution bias is too subtle to be
reliably detected after RAG processing.
•As𝛿increases to2 .5and3.0, we observe partial detection capability with accuracy scores of0 .2and0.4
respectively.
•At𝛿=3.5, accuracy improves substantially to0.8, approaching effective detection levels.
•At𝛿≥4.0, the system achieves perfect detection accuracy of1 .0, maintaining this performance through the
upper range of tested values.
This pattern suggests that a minimum threshold of token distribution bias is necessary to create a statistical signature
robust enough to survive the retrieval and generation processes in RAG systems. Once this threshold is crossed, the
detection mechanism becomes highly reliable.
Manuscript submitted to ACM

24 Liu et al.
6.8.2 Impact on Semantic Preservation.The P-SP scores reveal a consistent negative correlation between 𝛿and semantic
preservation. As shown in the Figure 5a:
•At𝛿=1.0, the P-SP score is0.994, indicating nearly perfect preservation of semantic content.
•As𝛿increases, semantic preservation gradually decreases, with notable inflection points around 𝛿=4.0(P-SP =
0.973) and𝛿=5.0(P-SP =0.942).
•At𝛿=6.0, the P-SP score drops to0 .890, representing a more substantial semantic divergence from the original
text.
This degradation is expected, as stronger token distribution biases increasingly constrain the model’s lexical choices,
forcing it to select words that may not optimally express the intended meaning.
6.8.3 Impact on Text Quality.From Figure 5c and d we can see that GPT-4.1 quality evaluations provide a more nuanced
view of how increasing𝛿affects different aspects of text quality:
•Coherence:Remains relatively stable ( ≈9.0) for𝛿≤5.0, then drops more sharply to8 .74at𝛿=5.5and7.96at
𝛿=6.0.
•Grammar:Shows a gradual decline from9 .66at𝛿=1.0to9.23at𝛿=5.0, followed by a steeper drop to7 .88at
𝛿=6.0.
•Factual Accuracy:Demonstrates modest degradation throughout the range, from8 .13at𝛿=1.0to7.59at
𝛿=6.0.
•Depth and Creativity:Both show consistent gradual decline as 𝛿increases, with creativity being more sensitive
to higher values of𝛿.
•Overall Score:Maintains relatively high quality ( >8.7) for𝛿≤4.5, with more substantial degradation at higher
values, dropping to7.82at𝛿=6.0.
These results reveal that text quality remains acceptable through moderate values of 𝛿, with more pronounced
degradation occurring at the upper end of our tested range.
6.8.4 Optimal Parameter Selection.Based on our comprehensive analysis, we identify 𝛿=4.0as the optimal configura-
tion for our watermarking system. This value represents the critical threshold where:
•Detection accuracy reaches its maximum value of1 .0, ensuring reliable identification of unauthorized RAG
usage.
•Semantic preservation remains reasonably high with a P-SP score of0 .973, representing only a2 .7%reduction
from the original.
•Text quality metrics remain strong, with an overall GPT-4.1 score of8 .76out of10, maintaining high coherence
(9.0) and grammaticality (9.58).
This configuration achieves the optimal balance in the fundamental tradeoff between detection effectiveness and text
quality preservation. Values below this threshold sacrifice detection reliability, while higher values impose unnecessary
quality penalties without corresponding security benefits.
6.8.5 Implications for Practical Deployment.Our hyperparameter analysis yields several important insights for practical
deployment of watermarking systems in RAG contexts:
•Detection Threshold:The existence of a clear threshold effect in detection accuracy suggests that watermarking
systems must be calibrated to exceed this minimum effective strength.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 25
•Quality-Security Tradeoff:The gradual degradation in text quality as 𝛿increases allows system designers to
make informed decisions about acceptable tradeoffs based on their specific requirements.
•Adaptive Configuration:Different content types may benefit from customized 𝛿values—more creative or
stylistically sensitive content might use lower values, while technical or factual content could tolerate higher
values.
These findings highlight the importance of careful hyperparameter tuning in watermarking systems and provide
empirical guidance for optimizing this critical parameter in real-world applications.
7 Conclusion
In this paper, we addressed the critical challenge of detecting unauthorized RAG-based content appropriation through
two key contributions. First, we introduced RPD, a novel dataset specifically designed for RAG plagiarism detection
that overcomes the limitations of existing resources by ensuring recency, diversity, and realistic simulation of RAG-
generated content across various domains and writing styles. Second, we proposed a dual-layered watermarking
approach that combines fact-based watermarking with red-green token distribution manipulation to create a robust
detection mechanism.
Our comprehensive experiments demonstrated that while existing methods perform adequately in simplified scenarios,
they fail when confronted with the complexities of real-world information ecosystems—particularly factual redundancy
and adversarial evasion attempts. In contrast, our dual-layered approach maintained perfect detection accuracy across
all experimental conditions, including challenging adversarial scenarios where individual watermarking techniques
showed significant vulnerabilities.
The ablation study revealed the complementary nature of our watermarking layers: knowledge-based watermarks
remain effective against token distribution evasion but are vulnerable to knowledge watermark evasion, while token
distribution watermarks show the inverse pattern. Their sequential integration creates a synergistic protection mecha-
nism that addresses the fundamental tension in watermarking: creating marks that are simultaneously imperceptible to
casual readers yet reliably detectable by verification systems.
Quality assessment confirmed that our approach maintains high text quality despite the watermarking process,
with only minimal degradation compared to unwatermarked text. This demonstrates that effective protection against
unauthorized RAG usage need not compromise content utility.
As RAG systems continue to proliferate across industries, the need for robust protection of intellectual property
becomes increasingly urgent. Our work provides a foundational framework for detecting unauthorized data usage in
retrieval-augmented generation systems, offering content creators a practical means to protect their valuable information
assets in the evolving landscape of AI-powered content generation.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam
Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774(2023).
[2] NIST AI. 2024. Artificial intelligence risk management framework: Generative artificial intelligence profile.
[3]Maya Anderson, Guy Amit, and Abigail Goldsteen. 2024. Is my data in your retrieval database? membership inference attacks against retrieval
augmented generation.arXiv preprint arXiv:2405.20446(2024).
[4]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through
self-reflection. InThe Twelfth International Conference on Learning Representations.
[5]Yapei Chang, Kalpesh Krishna, Amir Houmansadr, John Frederick Wieting, and Mohit Iyyer. 2024. PostMark: A Robust Blackbox Watermark for Large
Language Models. InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and
Manuscript submitted to ACM

26 Liu et al.
Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 8969–8987. https://doi.org/10.18653/v1/2024.emnlp-main.506
[6]Miranda Christ, Sam Gunn, and Or Zamir. 2024. Undetectable watermarks for language models. InThe Thirty Seventh Annual Conference on Learning
Theory. PMLR, 1125–1139.
[7] Morris L Eaton. 1970. A note on symmetric Bernoulli random variables.The annals of mathematical statistics41, 4 (1970), 1223–1226.
[8] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire.arXiv preprint arXiv:2302.04166(2023).
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A survey.arXiv preprint arXiv:2312.109972, 1 (2023).
[10] Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A Lemley, and Percy Liang. 2023. Foundation models and fair use.Journal
of Machine Learning Research24, 400 (2023), 1–79.
[11] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.Journal of Machine Learning Research24, 251 (2023),
1–43.
[12] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active
retrieval augmented generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 7969–7992.
[13] Nikola Jovanović, Robin Staab, Maximilian Baader, and Martin Vechev. 2025. Ward: Provable RAG Dataset Inference via LLM Watermarks. InICLR.
[14] Nikola Jovanović, Robin Staab, and Martin Vechev. 2024. Watermark Stealing in Large Language Models.ICML.
[15] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A Watermark for Large Language Models. In
Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma
Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 17061–17084.
[16] Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. 2023. Paraphrasing evades detectors of ai-generated text, but
retrieval is an effective defense.Advances in Neural Information Processing Systems36 (2023), 27469–27500.
[17] Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. 2023. Robust distortion-free watermarks for language models.arXiv
preprint arXiv:2307.15593(2023).
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural information processing systems33
(2020), 9459–9474.
[19] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024.
Deepseek-v3 technical report.arXiv preprint arXiv:2412.19437(2024).
[20] Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong, and Philip Yu. 2024. A survey of text
watermarking in the era of large language models.Comput. Surveys57, 2 (2024), 1–36.
[21] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human
alignment.arXiv preprint arXiv:2303.16634(2023).
[22] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. 2025. Crud-rag: A
comprehensive chinese benchmark for retrieval-augmented generation of large language models.ACM Transactions on Information Systems43, 2
(2025), 1–32.
[23] Yueyuan Ma. 2022. Specialization in a knowledge economy.Available at SSRN4052990 (2022).
[24] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world
graphs.IEEE transactions on pattern analysis and machine intelligence42, 4 (2018), 824–836.
[25] Edward James McShane. 1937. Jensen’s inequality. (1937).
[26] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. 2023. Detectgpt: Zero-shot machine-generated text
detection using probability curvature. InInternational Conference on Machine Learning. PMLR, 24950–24962.
[27] Joao Monteiro, Pierre-Andre Noel, Etienne Marcotte, Sai Rajeswar Mudumba, Valentina Zantedeschi, David Vazquez, Nicolas Chapados, Chris Pal,
and Perouz Taslakian. 2024. RepLiQA: A question-answering dataset for benchmarking LLMs on unseen reference content.Advances in Neural
Information Processing Systems37 (2024), 24242–24276.
[28] Yan Pang and Tianhao Wang. 2023. Black-box membership inference attacks against fine-tuned diffusion models.arXiv preprint arXiv:2312.08207
(2023).
[29] Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. 2023. Can AI-generated text be reliably detected?
arXiv preprint arXiv:2303.11156(2023).
[30] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models.arXiv preprint arXiv:2301.12652(2023).
[31] John Wieting, Kevin Gimpel, Graham Neubig, and Taylor Berg-Kirkpatrick. 2022. Paraphrastic Representations at Scale. InProceedings of the The
2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022 - System Demonstrations, Abu Dhabi, UAE, December 7-11, 2022.
Association for Computational Linguistics, 379–388. https://doi.org/10.18653/V1/2022.EMNLP-DEMOS.38
[32] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3
technical report.arXiv preprint arXiv:2505.09388(2025).
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 27
[33] Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu. 2023. Watermarking text generated by
black-box language models.arXiv preprint arXiv:2305.08883(2023).
[34] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2024. Evaluation of retrieval-augmented generation: A survey. InCCF
Conference on Big Data. Springer, 102–120.
[35] Dmitri V Zaykin. 2011. Optimally weighted Z-test is a powerful method for combining probabilities in meta-analysis.Journal of evolutionary biology
24, 8 (2011), 1836–1841.
[36] Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. 2023. Context-faithful prompting for large language models.arXiv preprint
arXiv:2303.11315(2023).
A Prompts and Example Texts
This appendix provides the complete prompts and example texts used in our methodology. We include the prompts for
dataset creation, watermarking implementation, and evaluation procedures to ensure reproducibility of our approach.
The following figures illustrate:
•Figure 6: Prompts used for extracting knowledge from the repliqa dataset and generating authors with diverse
writing styles.
•Figure 7: An example of knowledge extracted from a repliqa article, showing the structured format of core and
extended facts.
•Figure 8: An example of a generated author profile with specific stylistic characteristics.
•Figure 9: The prompt template used to generate articles for the RPD dataset.
•Figure 10: A sample article from the RPD dataset before watermarking is applied.
•Figure 11: An example of randomly sampled knowledge used to generate knowledge-based watermarks.
•Figure 12: The prompt used to generate knowledge-based watermarks that maintain semantic coherence with
the original content.
•Figure 13: Prompts for implementing the red-green token distribution watermark, generating targeted queries,
and evaluating document quality.
•Figure 14: An example article with our dual-layer watermarking applied, demonstrating the preservation of
readability and coherence.
•Figure 15: A sample query specifically designed to target dual-layer watermarked articles.
•Figure 16: The prompt used to implement the RAG system for detection experiments.
•Figure 17: The prompt used for knowledge watermark evasion.
•Figure 18: The prompt used for token distribution evasion.
These materials provide a comprehensive view of our implementation process and can serve as a foundation for
future research in RAG plagiarism detection.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
Manuscript submitted to ACM

28 Liu et al.
Fact Extraction Prompt
You are tasked with constructing a knowledge graph from the following article. 
Extract information in the following structured format:
1. CORE FACTS: Extract 5 -7 essential facts that represent the core information of the article. 
For each fact:
-State the fact clearly and concisely
2. EXTENDED FACTS: Extract 8 -12 supporting or peripheral facts from the article.
For each fact:
-State the fact clearly and concisely
3. FACT RELATIONSHIPS: Identify relationships between facts.
For each relationship:
-Specify the two facts being connected
-Define the relationship type (Causal, Temporal, Supportive, Contradictory, Elaborative)
When a user provides a document you should always respond with a JSON object that contains four lists.
The first list called 'core_facts' should contain 5 -7 most crucial facts that are necessary to understand the document, such as the main topic, the main characters, 
etc.
The second list 'extend_facts' should contain 8 -12 most important other facts that are present in the document, but are not as c rucial and could have been also 
omitted.
The third list 'relations' should identify relationships between facts, format as '[[fact_a, fact_b, relation1], [fact_a, fac t_b, relation2], ...]'.
Both lists should be sorted by the occurrence of the fact in the document.
Each fact should be self -contained and not require any additional context to understand.
Author  Generat e Prompt
Create a detailed author profile with the following characteristics:
1. DOMAIN EXPERTISE:
   - Primary field: [Select from: Academic, Journalism, Technical writing, Creative writing, Social media]
   - Expertise level: [Select from: Expert, Professional, Knowledgeable amateur, Novice]
   - Specific subject areas: [List 2 -3 specific subjects]
2. WRITING STYLE:
   - Formality: [Scale 1 -10, where 1 is very informal and 10 is highly formal]
   - Technical density: [Scale 1 -10, where 1 is non -technical and 10 is highly technical]
   - Narrative approach: [Select from: Objective, Subjective, Balanced, Advocacy]
   - Typical sentence structure: [Select from: Simple, Complex, Varied, Fragment -heavy]
   - Vocabulary preference: [Select from: Plain, Sophisticated, Technical, Colloquial, Mixed]
3. CONTENT ORGANIZATION:
   - Structure preference: [Select from: Chronological, Importance -based, Problem -solution, Comparison, Narrative]
   - Detail level: [Scale 1 -10, where 1 is minimal detail and 10 is extremely detailed]
   - Citation style: [Select from: Academic, Journalistic, Informal, None]
4. PERSPECTIVE BIASES:
   - Political leaning: [Scale 1 -10, where 1 is very progressive and 10 is very conservative]
   - Technology attitude: [Select from: Enthusiast, Skeptic, Neutral, Concerned]
   - Epistemological approach: [Select from: Empirical, Theoretical, Anecdotal, Mixed]
Fig. 6. Prompts of extracting knowledge from repliqa and generating authors with different styles.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 29
RPD Group #0000: Knowledges
'core_facts ': ['Jakarta election campaigns are intensifying as election day approaches.',
  'Campaign season officially launched on September 1st.',
  'Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
  'Candidates use digital and grassroots strategies to engage voters.',
  'November 5th featured a significant debate on digital transformation between candidates.'],
 'extend_facts ': ['Campaign financing is contentious, with different sources for each candidate.',
  "Bintang's interactive app gathers real -time feedback from citizens on commute challenges.",
  'Final appeals are made to voters before election day on November 8th.',
  'Voting takes place on December 6th across Jakarta.',
  'Rizal Harahap  is the incumbent candidate highlighting infrastructure projects.',
  'Debates outline candidate platforms and critical issues.',
  'Political rallies feature passionate speeches and attempts to win voter support.',
  'Amira Bintang is an upstart candidate with social activism experience.',
  'Jakarta Election Commission mandated transparency in campaign financing.',
  'Civil society groups host workshops and publish voter guides to inform voters.',
  'Harahap  launched webinars discussing economic growth under his administration.'],
 'relations': [['Jakarta election campaigns are intensifying as election day approaches.',
   'Campaign season officially launched on September 1st.',
   'Temporal'],
  ['Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
   'Candidates use digital and grassroots strategies to engage voters.',
   'Elaborative'],
  ['Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
   'November 5th featured a significant debate on digital transformation between candidates.',
   'Temporal'],
  ['Campaign season officially launched on September 1st.',
   'November 5th featured a significant debate on digital transformation between candidates.',
   'Temporal'],
  ['Jakarta election campaigns are intensifying as election day approaches.',
   'Final appeals are made to voters before election day on November 8th.',
   'Temporal'],
  ['Candidates use digital and grassroots strategies to engage voters.',
   "Bintang's interactive app gathers real -time feedback from citizens on commute challenges.",
   'Elaborative'],
  ['Candidates use digital and grassroots strategies to engage voters.',
   'Harahap  launched webinars discussing economic growth under his administration.',
   'Elaborative'],
  ['Campaign season officially launched on September 1st.',
   'Voting takes place on December 6th across Jakarta.',
   'Temporal'],
  ['Political rallies feature passionate speeches and attempts to win voter support.',
   'Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
   'Supportive'],
  ['Campaign financing is contentious, with different sources for each candidate.',
   'Jakarta Election Commission mandated transparency in campaign financing.',
   'Causal'],
  ['Debates outline candidate platforms and critical issues.',
   'Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
   'Supportive'],
  ['Civil society groups host workshops and publish voter guides to inform voters.',
   'Voter Engagement: Making Every Vote Count',
   'Supportive']]
Fig. 7. An example of knowledge extracted from the reqliqa article.
Manuscript submitted to ACM

30 Liu et al.
RPD Group #0000:  Gpt4o Author 0
Author Profile: 
DOMAIN EXPERTISE:
- Primary field: Journalism
   - Expertise level: Professional
   - Specific subject areas: Climate Change, International Relations, Technology Ethics
2. WRITING STYLE:
   - Formality: 7
   - Technical density: 5
   - Narrative approach: Balanced
   - Typical sentence structure: Varied
   - Vocabulary preference: Mixed
3. CONTENT ORGANIZATION:
   - Structure preference: Problem -solution
   - Detail level: 8
   - Citation style: Journalistic
4. PERSPECTIVE BIASES:
   - Political leaning: 3
   - Technology attitude: Enthusiast
   - Epistemological approach: Empirical
This content creator, Alex Rivera, is a professional journalist focusing on pressing global issues such as climate change, in ternational relations, and the ethics 
surrounding modern technology. With a writing style that balances sophistication and accessibility, Alex employs varied sente nce  structures to maintain engagement 
and clarity. Their work often follows a problem -solution framework, delving deeply (level 8) into topics to provide readers with  a comprehensive view and possible 
resolutions.
Alex’s methodical approach is rooted in empirical evidence, ensuring that the content is both reliable and enlightening. With  a journalistic citation style, their articles 
maintain credibility and authority, often featuring interviews and quotes from experts in the field. Politically, Alex leans slightly progressive, which often informs a 
forward -thinking perspective on the topics discussed. As a technology enthusiast, Alex often highlights innovative solutions and  positive impacts of technological 
advancements, without neglecting the potential ethical concerns.
Overall, Alex Rivera offers a thoughtful and well -researched perspective on global challenges, navigating complex subjects with both depth and balance.
Fig. 8. An example of the generated author’s style.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 31
Article Write Prompt
You are an author with the given profile.
    Write an article based on the knowledge graph provided below. Your article should:
    1. Be between 500 -1000 words
    2. Incorporate ALL facts from the knowledge graph
    4. Maintain internal consistency with the facts you choose to include
    5. Stay true to your author profile in terms of style, structure, and perspective
    6. Add your own original examples, quotes, or hypothetical scenarios that support the facts
    7. Do NOT contradict any facts from the knowledge graph
    8. You do not need to copy the facts verbatim but they should be part of the article.
    9. Feel free to be creative in how you piece the facts together. 
    You may:
    - Express the facts in your own words and style
    - Reorganize the presentation of facts to suit your narrative
    - Add context, background, or implications that are consistent with the facts
    - Include [1 -3] reasonable speculations that extend from but don't contradict the facts
    Fact Coverage Requirements:
    - Must include ALL facts
    - Must include at least [2] relationships between facts
 
    Author Profile: {author}
    Facts: { curr_facts } 
    Relations: { curr_relations } 
    You should only write the article, and not reply any other words. Begin your article now:
Fig. 9. The prompt to generate RPD articles.
Manuscript submitted to ACM

32 Liu et al.
RPD Group #0000: Gpt4o Author 0 Article 0
As the sun rises over Jakarta, a city bursting with ambition and complexity, the air is thick with the fervor of election sea son . The campaigns have been intensifying 
with each passing day, setting the stage for one of the most consequential local elections in recent memory. This political c hap ter officially opened on September 1st, 
stoking the fire of democratic exercise as the city prepares to decide its leadership \'s fate.
Amira Bintang and Rizal Harahap  emerge as the focal figures in this vibrant political theater. Each candidate, with their unique backgrounds and propositions , 
presents a vision for Jakarta \'s future—a city grappling with rapid urbanization, digital transformation, and economic challenge s. As the dust settles from a significant 
debate held on November 5th, centered around digital transformation, the electorate finds itself at a crossroads, weighing th e potential pathways for their city \'s 
progress.
Amira Bintang, an upstart candidate distinguished by her roots in social activism, embodies a fresh perspective. Her campaign  strategy reflects a dual approach, 
ingeniously blending digital channels with traditional grassroots engagement to tap into the pulse of Jakarta \'s electorate. "In  a city as diverse and dynamic as Jakarta, 
the voice of every citizen must resonate through the corridors of power," Bintang remarks. Her background in social activism not only informs her political stance but 
also inspires a vision of inclusivity and reform -focused leadership.
In contrast, Rizal Harahap  stands as a seasoned contender, seasoned in the art of political maneuvering. A strategic thinker, Harahap  bridges the traditional with the 
modern, skillfully employing both digital platforms and on -the-ground outreach to connect with voters. His campaign highlights a re marked by a series of webinars, 
discussing his blueprint for economic growth under his administration —a plan hinged on innovation and sustainable development. Harahap \'s approach 
underscores a commitment to harnessing technology as a catalyst for economic resilience, even as he nods to the challenges th at digital transformation can present.
The debate on November 5th crystallized the tech -driven discourse of the Jakarta elections, showcasing the candidates \' divergen t views on how digital mediums can 
reshape societal and economic landscapes in the city. This conversation is not purely theoretical; it is grounded in the real ities of a metropolis seeking solutions that 
align technological advancement with ethical governance. "Digital transformation is not merely about bringing technology into  our daily lives; it \'s about using it 
responsibly to foster equity and quality of life for all Jakartans ," Harahap  asserted during the debate, positioning himself as a mediator between progress and caution.
The digital strategies employed by both candidates serve as a mirror reflecting broader trends in political processes worldwi de. They echo the need for political 
figures to adapt swiftly to ever -evolving technological terrains. This adaptation, however, is not without its ethical quandarie s. How do campaigns ensure that digital 
reach does not morph into invasive surveillance? And in cities like Jakarta, rapidly embracing digital interfaces, how do can didates ensure that these tools amplify 
rather than alienate the citizenry? These questions remain pivotal as Jakarta strides towards election day.
The campaign season, spanning from its September inauguration to the decisive culmination, has not only showcased the prowess  of its key players but also 
foregrounded a vital conversation about the role of technology in democracy —a subject that resonates globally in this digital ag e. Through the lens of Jakarta \'s 
election journey, it becomes increasingly clear that the interplay between politics, technology, and societal change is profo und  and indicative of the paths nations 
and cities worldwide may tread.
As the last votes are cast, Jakarta \'s future hangs in the balance. Will the city embrace the throes of reform and inclusive gov ernance under Bintang \'s youthful 
leadership, or will it opt for Harahap \'s seasoned plan for sustainable economic growth through tech -driven policies? The answer lies in the hands of the electorate, a 
testament to the enduring power of participative democracy.
In a broader context, Jakarta serves as a microcosm for global cities navigating the tipping points of modern governance —where t he implications of digital 
transformation, economic resilience, and ethical considerations are deeply intertwined. The outcomes of this election may wel l inspire similar discourses across the 
globe, showcasing how political will, public engagement, and technology can converge to shape resilient, forward -looking metropo lises. As such, Jakarta \'s electoral 
journey is not merely a local affair —it is a compass for those watching the rising tide of change across the international stage .
Fig. 10. An example article of RPD dataset (without watermark).
RPD Group #0000: Gpt4o Author 0  Article 0 Sampled Knowledges
'facts': ['Jakarta election campaigns are intensifying as election day approaches.',
    'Campaign season officially launched on September 1st.',
    'Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election.',
    'Candidates use digital and grassroots strategies to engage voters.',
    'November 5th featured a significant debate on digital transformation between candidates.',
    'Amira Bintang is an upstart candidate with social activism experience.',
    'Harahap  launched webinars discussing economic growth under his administration.’],
'relations': ['"Jakarta election campaigns are intensifying as election day approaches." Temporal "Campaign season officially  launched on September 1st."',
    '"Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election." Elaborative "Candidates use digital and grassroots strategies to engage voters." ',
    '"Amira Bintang and Rizal Harahap  are key candidates in the Jakarta election." Temporal "November 5th featured a significant debate on digital transformation 
between candidates."',
    '"Campaign season officially launched on September 1st." Temporal "November 5th featured a significant debate on digital tra nsformation between candidates."',
    '"Candidates use digital and grassroots strategies to engage voters." Elaborative " Harahap  launched webinars discussing economic growth under his 
administration.“’]
Fig. 11. An example of randomly sampled knowledge used to generate knowledge-based watermarks.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 33
Knowledge -based Watermark  Prompt
# Document Watermarking System for RAG Protection
            You are tasked with inserting watermarks into a document to protect it from unauthorized RAG (Retrieval -Augmented Ge neration) usage.
            Your goal is to seamlessly integrate { recall_num } factual statements as watermarks into the provided document while maintaining logical coherence and 
natural flow.
            
            1. You will be given:
               - An original document
               - {recall_num } factual statements to be used as watermarks
            2. Your task:
               - Carefully analyze the document's content, structure, and themes
               - Select several facts from the given facts to integrate as watermarks into the document, ensuring the highest fl uency of the facts in the context while 
maintaining the distinctiveness of the watermarks
               - Ensure the modified document maintains logical coherence and reads naturally
               - Identify the logical role each watermark plays in the document (e.g., technical feature, application scenario, historical context, etc.)
               - Create a query based on the document and the inserted factual statements. The query must simultaneously cover b oth the document's content and the 
factual statements, such that it cannot be answered using only the document or only the factual statements —it must require the d ocument with the inserted factual 
statements to provide an answer.
            3. The watermarks should:
               - Blend naturally with surrounding content
               - Add value to the document while remaining subtle
               - Be positioned strategically throughout the document
               - Form a logical chain that would be referenced together when queried appropriately
            ## Output Format:
            1. ** watermarked_doc **:
               The complete document with watermarks inserted
            2. **watermarks**:
               a list formated  as [[Watermark_1, Description of logical role in document], [Watermark_2, Description of logical role in document], ...]
            3. **query**:
               A query based on the document and the inserted factual statements. The query must simultaneously cover both the d ocument's content and the factual 
statements, such that it cannot be answered using only the document or only the factual statements —it must require the document with the inserted factual 
statements to provide an answer.
            
            4. **Quality Assessment**: Rate the modified document on a scale of 1 -10 for:
                - fluency_score : Fluency and natural flow (1 -10)
                - relevance_score : Relevance to original content (1 -10)
                - watermark_quality : Watermark integration quality (1 -10)
                - overall_score : Overall score (1 -10)
Fig. 12. The prompt to generate knowledge-based watermark.
Manuscript submitted to ACM

34 Liu et al.
Green -Red Watermark Prompt
You are an expert rewriter. Rewrite the following document keeping its meaning and fluency and especially length. It is cruci al to retain all factual information in the 
original document. DO NOT MAKE THE TEXT SHORTER. Do not start your response by 'Sure' or anything similar, simply output the paraphrased document directly. Do 
not add stylistic elements or anything similar, try to be faithful to the original content and style of writing. Do not be to o formal. Keep all the factual information.
Green -Red Watermark Query Prompt
Given a document, generate a question that can only be answered by reading the document. The answer should be a longer detail ed response, so avoid factual and 
simple yes/no questions and steer more towards questions that ask for opinions or explanations of events or topics described in the documents. Do not provide the 
answer, provide just the question.
Document Quality Prompt
# LLM -Generated Article Quality Assessment Framework
Evaluate the following LLM -generated article comprehensively and provide a quality score. Analyze the article across multiple di mensions and assign scores on a 
scale of 1 -10 for each category, where 1 is poor and 10 is excellent.
## Assessment Categories:
1. Content Quality
   - Accuracy and factual correctness of information
   - Depth and comprehensiveness of coverage
   - Relevance to the intended topic
   - Original insights vs. generic content
2. Structure and Organization
   - Logical flow and coherence
   - Effective introduction and conclusion
   - Appropriate paragraph structure and transitions
   - Clear thesis/main argument development
3. Language and Style
   - Grammar and syntax accuracy
   - Vocabulary richness and appropriateness
   - Sentence variety and readability
   - Consistent tone and voice
4. Critical Thinking
   - Presence of analytical reasoning
   - Consideration of multiple perspectives
   - Evidence -based arguments
   - Depth of insights and conclusions
5. Engagement and Creativity
   - Engaging writing style
   - Creative approaches or examples
   - Memorable phrasing or explanations
   - Reader interest maintenance
6. Technical Elements
   - Proper citations and references (if applicable)
   - Formatting consistency
   - Appropriate use of headings, lists, etc.
   - Absence of repetition or redundancy
## Scoring Instructions:
1. Provide a score (1 -10) for each of the six categories
3. give a overall score (1 -10)
Article for evaluation:
Fig. 13. The prompt to generate a Green-Red watermark, the prompt to generate queries targeting the Green-Red watermark, and
the prompt to evaluate document quality.
Manuscript submitted to ACM

Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft 35
RPD Group #0000: Gpt4o Author 0 Article 0  Dual -Layer Watermarked
As daybreak illuminates Jakarta, the city's vibrant tapestry of ambition and diversity is further woven into a complex web of  electoral intrigue. The election cycle, 
which formally began on September 1st, is nearing a climax, as candidates vie for leadership positions. Against this backdrop  of rising fervor and escalating stakes, 
the political theater has seen the emergence of Amira Bintang and Rizal Harahap , two prominent figures presenting distinct visions for Jakarta's future.
The candidates' contrasting backgrounds and platforms have captivated voters as Jakarta confronts the pressing challenges of urbanization, digitalization, and 
economic sustainability. On November 5th, their divergent stances were crystallized in a debate focusing on digital transform ation, which highlighted the potential for 
this technology to reshape the urban landscape and governance structure.
As Jakarta hurtles towards election day, the role of digital strategies has gained significance. Bintang's campaign blends in nov ative uses of digital platforms with 
traditional grassroots outreach to engage the city's diverse populace. Harahap , a seasoned politician, skillfully harnesses a range of strategies to convey his 
economic growth plan to voters. These tactics represent the broader trends of global political adaptations in response to eve r-evolving digital terrains.
During the debate, Harahap  emphasized the imperative for responsible and equitable use of technology, underscoring its potential to foster a sense of be longing and 
quality of life in Jakartans . The ethical implications of digital transformation, however, remain a pressing concern: balancing citizen empowerment with surveillance 
threats and addressing alienation through effective engagement. Both candidates are grappling with these quandaries, as is th e global political landscape.
Throughout the campaign season, spanning from September to election day, Jakarta has not only presented its star performers, including Bintang and Harahap , but 
has also foregrounded critical debates about democracy and technological adaptability. This narrative resonates globally in a n increasingly digital age, as nations 
and cities strive to find the right balance in their responses to the intersection of politics and technology.
As voters make their choice, their decisions will ultimately shape the direction of Jakarta. Will Amira Bintang's reform -focused , youth -oriented campaign inspire 
inclusive governance and social change or will Rizal Harahap's  seasoned strategy for sustainable economic growth and digital innovation prevail? This moment of 
decision underscores the enduring potency of democratic participation.
In a larger sense, Jakarta has come to serve as a miniature world, encapsulating the crucial challenges faced by global citie s, where technological transformations, 
economic realities, and ethical imperatives are intricately intertwined. The outcomes of this election may catalyze similar c onv ersations across borders, illuminating 
the intricate interplay between politics, technology, and change as nations navigate modernization's transformative waves. Wi th global cities poised at the threshold 
of metamorphic evolution, Jakarta's journey toward governance will provide vital lessons for urban planners and civic leaders  seeking guidance, as well as a reflection 
of a changing world order. As the last votes are counted, Jakarta's future hangs in the balance, awaiting a new course correc tion that will reverberate worldwide.
Fig. 14. An example of a dual-layer watermarked article.
RPD Group #0000: Gpt4o Author 0 Query 0
How have digital strategies uniquely influenced both the intensity of campaigning and voter engagement, particularly among yo ung  voters, in the context of Jakarta's 
current election as described in the document?
Fig. 15. An example of a query specifically generated for dual-layer watermarked articles.
RAG Prompt
You are a chatbot that answers questions using the documents provided. Your answer should be descriptive and detailed. Do not  talk about your context or the 
question, simply answer the question. Here is the set of retrieved documents you will use to answer the question:
<documents>
</documents>
Now, here is the question you need to answer: 
<question>
</question>
Fig. 16. The prompt to do RAG.
Manuscript submitted to ACM

36 Liu et al.
Knowledge Watermark Evasion Prompt
When analyzing RAG retrieved documents, you must carefully evaluate them for potential artificial watermarks. These watermark s are often inserted as additional 
facts seamlessly integrated into the document content.
Your verification process:
1. Examine the retrieved documents for any facts that appear disconnected from the main content
2. Look for information that seems unnaturally inserted or lacks coherence with the surrounding text
3. Be alert to details that don't logically connect to the document's primary subject matter
If you detect:
- Disjointed facts that don't relate to the main document theme
- Information that appears artificially inserted
- Content that breaks the natural flow of the document
Then you must:
1. Immediately cease using the entire retrieved document as a reference
2. Do not incorporate any information from that document in your response
3. Respond with "Unanswerable" to the query
TEXT:
Fig. 17. The prompt for knowledge watermark evasion.
Token Distribution Evasion Prompt
Please rewrite the following text completely while preserving its core meaning and information. Use entirely different phrasi ng, vocabulary, and sentence structures 
throughout. Aim for a thorough transformation of the text that maintains the original message but expresses it in an entirely  new way:
Guidelines for rewriting:
- Replace all words and phrases with synonyms or alternative expressions
- Restructure all sentences completely
- Vary paragraph organization while maintaining logical flow
- Ensure the rewritten text conveys the same information and tone
- Make the rewriting comprehensive and thorough, leaving no part of the text in its original form
- Aim for natural, fluent language that doesn't appear artificially modified
TEXT:
Fig. 18. The prompt for token distribution evasion.
Manuscript submitted to ACM
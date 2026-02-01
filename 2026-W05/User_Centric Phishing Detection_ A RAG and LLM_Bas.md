# User-Centric Phishing Detection: A RAG and LLM-Based Approach

**Authors**: Abrar Hamed Al Barwani, Abdelaziz Amara Korba, Raja Waseem Anwar

**Published**: 2026-01-29 04:42:18

**PDF URL**: [https://arxiv.org/pdf/2601.21261v1](https://arxiv.org/pdf/2601.21261v1)

## Abstract
The escalating sophistication of phishing emails necessitates a shift beyond traditional rule-based and conventional machine-learning-based detectors. Although large language models (LLMs) offer strong natural language understanding, using them as standalone classifiers often yields elevated falsepositive (FP) rates, which mislabel legitimate emails as phishing and create significant operational burden. This paper presents a personalized phishing detection framework that integrates LLMs with retrieval-augmented generation (RAG). For each message, the system constructs user-specific context by retrieving a compact set of the user's historical legitimate emails and enriching it with real-time domain and URL reputation from a cyber-threat intelligence platform, then conditions the LLM's decision on this evidence. We evaluate four open-source LLMs (Llama4-Scout, DeepSeek-R1, Mistral-Saba, and Gemma2) on an email dataset collected from public and institutional sources. Results show high performance; for example, Llama4-Scout attains an F1-score of 0.9703 and achieves a 66.7% reduction in FPs with RAG. These findings validate that a RAG-based, user-profiling approach is both feasible and effective for building high-precision, low-friction email security systems that adapt to individual communication patterns.

## Full Text


<!-- PDF content starts -->

User-Centric Phishing Detection: A RAG and
LLM-Based Approach
Abrar Hamed Al Barwani, Abdelaziz Amara Korba, Raja Waseem Anwar
German University of Technology in Oman, Sultanate of Oman
Abstract—The escalating sophistication of phishing emails
necessitates a shift beyond traditional rule-based and conven-
tional machine-learning-based detectors. Although large language
models (LLMs) offer strong natural language understanding,
using them as standalone classifiers often yields elevated false-
positive (FP) rates, which mislabel legitimate emails as phishing
and create significant operational burden. This paper presents
a personalized phishing detection framework that integrates
LLMs with retrieval-augmented generation (RAG). For each
message, the system constructs user-specific context by retrieving
a compact set of the user’s historical legitimate emails and
enriching it with real-time domain and URL reputation from
a cyber-threat intelligence platform, then conditions the LLM’s
decision on this evidence. We evaluate four open-source LLMs
(Llama4-Scout, DeepSeek-R1, Mistral-Saba, and Gemma2) on
an email dataset collected from public and institutional sources.
Results show high performance; for example, Llama4-Scout
attains an F1-score of 0.9703 and achieves a 66.7% reduction
in FPs with RAG. These findings validate that a RAG-based,
user-profiling approach is both feasible and effective for building
high-precision, low-friction email security systems that adapt to
individual communication patterns.
Index Terms—Phishing Detection, False Positives, Large Lan-
guage Models (LLMs), Retrieval-Augmented Generation (RAG),
Personalized Spam Filter, Email Security
I. INTRODUCTION
Email phishing remains one of the most pervasive and
financially damaging cyber threats, with organizations facing
relentless social engineering attacks that bypass conventional
security controls. Traditional detection mechanisms [1], in-
cluding signature-based filters, blacklists, and heuristic anal-
ysis—provide essential but increasingly inadequate protection
against evolving attack vectors. These approaches suffer from
fundamental limitations: static rule sets unable to adapt to
novel tactics, high false positive rates that burden users, and
insufficient contextual understanding of sophisticated social
engineering techniques.
The emergence of artificial intelligence has introduced
transformative capabilities for addressing these challenges [2].
Traditional machine learning approaches have demonstrated
promising results in classifying malicious emails through
feature extraction and pattern recognition. However, the advent
of large language models (LLMs) has fundamentally advanced
the landscape of cybersecurity [3]–[5] and email security.
Specifically, LLM-based solutions [6]–[8] have shown remark-
able capabilities in natural language understanding, contextual
analysis, and reasoning about social engineering tactics. These
systems leverage sophisticated pattern recognition and seman-tic analysis to identify subtle phishing indicators that often
escape traditional detection methods.
Despite these advancements, a critical research gap persists
in leveraging LLMs for personalized phishing detection that
minimizes false positives. Current AI-powered email security
systems [6]–[8] predominantly operate as generic classifiers,
analyzing emails in isolation without considering individual
user context. This approach leads to excessive false alarms
when legitimate emails deviate from generic patterns but align
with specific user behaviors and communication histories.
The absence of user-specific context prevents these systems
from distinguishing between universally malicious content and
communications that are unusual yet legitimate for particular
individuals.
This paper introduces a novel framework that addresses
this research gap through Retrieval-Augmented Generation
(RAG) enhanced LLMs specifically designed for personalized
phishing detection. Our system leverages historical email pat-
terns and real-time threat intelligence to create adaptive, user-
centric detection capabilities. The fundamental innovation of
our approach lies in the dual-context retrieval mechanism that
combinesuser-specific historical patternswithreal-time threat
intelligence. This enables the system to distinguish between
genuinely malicious emails and legitimate communications
that may appear suspicious to generic classifiers but are normal
for specific users. The main contributions of this work are as
follows:
•Introducing a personalized RAG-enhanced framework
that integrates user-specific email history with real-time
threat intelligence to create contextual awareness pre-
viously absent in LLM-based detection systems. The
proposed architecture incorporates semantic similarity
search, dynamic context retrieval, and multi-source ev-
idence integration to reduce false positives while main-
taining high detection accuracy.
•Developing a comprehensive evaluation methodology that
systematically assesses four diverse open-source language
models across both standalone and RAG-enhanced con-
figurations. This rigorous testing protocol establishes a
standardized framework for benchmarking AI-powered
security tools.
•Demonstrating significant empirical improvements
through experimental evidence showing substantial false
positive reduction, with Llama4-Scout achieving a 4%
FPR with RAG compared to 12% without. The analysis
reveals consistent performance gains across modelarXiv:2601.21261v1  [cs.CR]  29 Jan 2026

sizes and architectures, confirming that personalization
benefits scale effectively from 17B to 70B parameter
models.
The remainder of this paper is organized as follows. Sec-
tion II reviews related work. Section III details the proposed
solution. Section IV presents the performance evaluation and
results. Section V concludes the paper and outlines directions
for future work.
II. RELATEDWORKS
Early approaches to phishing email detection primarily
relied on traditional machine learning techniques and rule-
based systems. Gupta et al. [9] comprehensively surveyed
existing solutions including DNS-based blacklists, Sender ID,
Domain Keys, and machine learning algorithms like k-Nearest
Neighbor. These methods, while foundational, suffered from
significant limitations including high false positive rates, in-
ability to detect zero-day phishing attacks, and dependence on
static patterns that could be easily evolved by attackers. The
fundamental limitation of these traditional approaches lies in
their inability to understand semantic content and contextual
nuances of modern phishing emails, which increasingly em-
ploy sophisticated social engineering tactics and personalized
content that bypasses signature-based detection.
The emergence of large language models has revolutionized
the approach to email security by enabling deep semantic
understanding of email content. Several recent studies have
demonstrated the effectiveness of LLMs in phishing detection:
Desolda et al. [6] developed APOLLO, a GPT-4o powered
tool that classifies emails and generates explanations for users.
Their system incorporates URL enrichment through VirusTotal
and geographical analysis via BigDataCloud. While achieving
high accuracy, the authors noted limitations including prompt
quality dependency and the need for evaluation with multiple
LLM models beyond GPT-4o.
Koide et al. [7] proposed ChatSpamDetector, a system
that processes .eml files and leverages prompt engineering
with both normal and simple prompt templates. Their eval-
uation across GPT-4, GPT-3.5, Llama 2, and Gemini Pro
showed GPT-4 achieving 99.70% accuracy with the normal
prompt. However, the study excluded emails without links
and suggested that Retrieval-Augmented Generation could
further enhance performance. Catherine Lee [8] implemented
a comprehensive framework for phishing email detection
using multiple open-source LLMs including Llama-3.1-70b,
Gemma2-9b, and Mistral-large-latest. The study demonstrated
that LLMs could achieve over 80% accuracy, with Llama-
3.1-70b reaching 97.21%. A key limitation identified was the
tendency for ”aggressive phishing classification” leading to
false positives on imbalanced datasets.
Heiding et al. [10] investigated the generation of phish-
ing emails using LLMs, comparing GPT-4 generated emails
with manually crafted emails using the V-Triad model. Their
findings revealed that while LLMs could generate convincing
phishing content, human-crafted emails using psychologicalprinciples achieved higher click-through rates. This highlights
the importance of understanding social engineering tactics in
detection systems. Zhang et al. [11] focused on the practical
deployment of LLMs for small and midsize enterprises, finding
that smaller models like Llama-3-8b-instruct could provide
cost-effective solutions while maintaining robust detection
capabilities. Their work challenged the ”bigger-is-better” as-
sumption in LLM selection for security applications.
Retrieval-Augmented Generation has emerged as a powerful
technique to enhance LLM performance by providing external,
up-to-date knowledge. While RAG has been widely adopted
in question-answering systems and knowledge-intensive tasks,
its application in cybersecurity, particularly phishing detec-
tion, remains underexplored. The fundamental advantage of
RAG lies in addressing key limitations of standalone LLMs,
including hallucination, static knowledge cutoffs, and lack of
domain-specific context [12]. By integrating dynamic knowl-
edge retrieval, RAG enables LLMs to make decisions based on
current threat intelligence and user-specific historical patterns.
Despite these advancements, current LLM-based phishing
detection systems share a critical limitation: they operate as
generic classifiers without incorporating user-specific context.
This results in high false positive rates when legitimate
emails deviate from general patterns but align with individual
user behaviors and communication histories. The absence of
personalization prevents existing systems from distinguishing
between universally malicious content and communications
that are unusual yet legitimate for specific users. Table I
synthesizes recent state-of-the-art contributions in LLM-based
phishing detection, highlighting their principal findings and
limitations.
Our work addresses this gap by introducing a RAG-
enhanced framework that leverages both user-specific email
history and real-time threat intelligence, creating a personal-
ized detection system that significantly reduces false positives
while maintaining high detection accuracy.
III. PROPOSEDSOLUTION
Our proposed framework introduces a novel RAG-enhanced
architecture for personalized phishing detection that addresses
the critical limitation of false positives in AI-based systems.
The system operates through six stages to provide context-
aware email classification, as illustrated in Figure 1.
A. Data Preprocessing and Feature Extraction Pipeline
The system initialization commences with a rigorous data
preprocessing pipeline that transforms raw email corpora into
a structured knowledge base. LetD raw={e 1, e2, . . . , e n}
represent the raw email dataset, where each emaile icomprises
heterogeneous features. Our preprocessing pipeline applies a
sequence of transformation functions:
Dclean= Φ validate◦Φ normalize ◦Φ structure ◦Φ decode(Draw)(1)
where the transformation operators are defined as:

Table I.Summary of Key Studies in LLM-based Phishing Detection
Study Year Main Findings Limitations
Desolda et al. [6] 2024 High accuracy with GPT-4o, quality explanations Prompt dependency, single model evaluation
Koide et al. [7] 2024 99.70% accuracy with GPT-4, comprehensive analysis Excluded emails without links, no RAG integration
Lee et al. [8] 2025 Multi-model evaluation, 97.21% accuracy with Llama-3.1-
70bAggressive classification, false positives on imbalanced data
Heiding et al. [13] 2024 Comparative analysis of LLM vs human-crafted phishing Limited dataset, rapid model evolution
Zhang et al. [11] 2025 Cost-effective solutions for SMEs, smaller model effective-
nessLimited against sophisticated attacks
Fig. 1: Architecture of the RAG-based LLM Phishing Detection System
Φdecode(e) :=Multi-encoding resolution using (2)
{UTF-8,Latin-1,ISO-8859-1},(3)
Φstructure (e) :Feature selection{subject,sender,body} ⊂ F
(4)
Φnormalize (e) :Unicode normalization and header stripping,
(5)
Φvalidate (e) :=1 {sender∈E valid}·e,whereE valid:={x|@∈x}.
(6)
The resultant cleaned datasetD clean serves as the foun-
dational corpus for personalized context retrieval, ensuring
maximal information preservation while eliminating noise
and inconsistencies. The pseudocode underlying the data-
processing pipeline is illustrated in Algorithm 1.
B. Semantic Embedding Generation and Vector Space Con-
struction
This module projects the textual email content into a contin-
uous vector space where semantic relationships are preserved.
Letf θ:T →Rdbe the embedding function parameterized
byθ, whered= 384for theall-MiniLM-L6-v2[14]Algorithm 1:Data Preprocessing Pipeline
Require:D raw={e 1, e2, . . . , e n}
Ensure:D clean
1:D intermediate ← ∅
2:foreache i∈ D rawdo
3:e′←DecodeWithFallback(e i,
4:{UTF-8, Latin-1, ISO-8859-1})
5:e′←ExtractFeatures(e′,{subject, send er, body})
6:e′←RemoveNonASCII(StripHeaders(e′))
7:ifContainsValidEmail(e′.sender)then
8:D intermediate ← D intermediate ∪ {e′}
9:end if
10:end for
11:D clean←RemoveIncomplete(D intermediate )
12:returnD clean
transformer architecture. For each emaile∈ D clean, we
compute its semantic embedding:
ve=fθ(concat(e.subject, e.sender, e.body))(7)
The embeddings are L2-normalized to reside on the unit
hypersphere:
ˆve=ve
∥ve∥2(8)

These normalized embeddings are indexed in a FAISS
vector databaseV, optimized for approximate nearest neighbor
search with cosine similarity metric:
sim(q,v) =q·v
∥q∥∥v∥=ˆq·ˆv(9)
1) Threat Intelligence Integration via Multi-Engine Anal-
ysis:The system augments semantic analysis with real-time
threat intelligence through formal integration with VirusTotal’s
API. For an incoming emaile query, we extract and analyze:
Ddomains ={extract domain(e query.sender)}(10)
Durls={extract urls(e query.body)}(11)
Each elementx∈ D domains ∪ D urlsundergoes multi-engine
analysis:
R(x) ={scan i(x)}75
i=1 where scan i∈ E engines (12)
The threat assessmentT(e query)aggregates results across all
analyzable elements:
T(e query) =[
x∈D domains∪D urlsharm x,suspx,mal x,
reputx
(13)
2) Contextual Retrieval via Semantic Similarity Search:
This module enables personalized phishing detection by re-
trieving semantically similar emails from the user’s historical
corpus. The process maps emails to a vector space where
semantic relationships are preserved, then performs efficient
similarity search.
This approach provides the LLM with concrete examples of
legitimate emails that share semantic characteristics with the
query, enabling personalized classification decisions based on
the user’s unique communication patterns rather than generic
rules.
The pseudocode algorithm 2 summaries the contextual
retrieval steps.
Algorithm 2:Contextual Retrieval Algorithm
Require:
1:e query: Input email to classify
2:V: Vector database of historical email embeddings
3:k= 5: Number of similar emails to retrieve
Ensure:C context : Top-k most similar historical emails
4:
5:q←f θ(equery.subject⊕e query.sender⊕e query.body)
6:ˆq←q/∥q∥ 2{Normalize to unit sphere}
7:I ←FAISS kNN(V, ˆq, k){Optimized similarity search}
8:C context← {e i∈ D clean:i∈ I}
9:returnC context
The semantic similarity is computed using cosine similarity
in the normalized embedding space:
sim(e query, ei) =q·v i
∥q∥∥v i∥(14)C. Structured Prompt Engineering and Reasoning Framework
The LLM prompter module formalizes the classification
task through sophisticated prompt engineering. LetPrepresent
the prompt construction function:
P(e query,Ccontext,T(e query)) =concat(r, c 1, c2, c3, o)(15)
where:
r:Role assignment: “cybersecurity expert specialized (16)
in phishing detection” (17)
c1:Email content:e query.{subject, sender, body}(18)
c2:RAG context:C context (19)
c3:Threat intelligence:T(e query)(20)
o:Output specification enforcing JSON schemaS(21)
The output schemaSconstrains the LLM response to a
structured format:
S=

Classification decision∈ {legitimate,phishing}
Phishing score∈[0,10]⊂Z
Risk∈ {low,medium,high}
Social engineering elements⊆ A tactics
Recommended actions⊆ A mitigation
Brief reason∈ T natural(22)
D. Model Selection and Configuration
We evaluated four diverse open-source LLMs to ensure the
generalizability of our approach across different architectures
and scales. Table II summarizes the model specifications.
This formalization ensures the LLM grounds its reason-
ing in the multi-modal evidence while producing structured,
machine-parsable outputs that facilitate automated processing
and integration.
E. LLM Classification and Output
The final module handles LLM inference and output pro-
cessing:
•API Integration: We utilize Groq [15] API for high-
speed inference with low latency, crucial for real-time
email classification
•Parameter Configuration: Temperature is set to 0.2 to
ensure deterministic, consistent responses while maintain-
ing necessary reasoning capability
•System Message: A persistent system message reinforces
the LLM’s role and behavioral guidelines throughout the
interaction
•Result Parsing: The JSON output is automatically parsed
and validated, with error handling for malformed re-
sponses
The module outputs both the classification decision and sup-
porting analysis, providing users with transparent, explainable
results.

Table II.Specifications of Selected LLM Models
Feature Llama4-Scout DeepSeek-R1 Mistral-Saba Gemma2-9B
ProviderMeta DeepSeek Mistral Google
Base ModelLLaMA 4 LLaMA 2 Mistral Gemma 2
Parameters17B 70B 24B 9B
Context Window131k 128k 32k 8k
ArchitectureSparse MoE Dense Dense Transformer
IV. EXPERIMENTALSETUP ANDRESULTS
In this section, we first describe the email dataset collection
and processing, then detail the implementation of our system,
and finally present the experimental results and discuss the
main findings.
A. Dataset collection and processing.
We assembled a balanced corpus of 500 emails (250 legiti-
mate, 250 phishing). Legitimate messages were sampled from
consenting users’ personal and institutional mailboxes [16]
using IMAP in read-only mode, while phishing messages were
sourced from public phishing repositories and internal security
feeds. Messages were parsed at the MIME level, converted to
UTF-8 (NFC), and reduced to three fields—subject,sender,
andbody—for model input. HTML content was normalized
to plain text (tag stripping and entity unescaping), boilerplate
footers and quoted replies were heuristically pruned, and track-
ing artifacts were removed. We applied light normalization
(lowercasing, whitespace compaction) to preserve semantic
cues relevant to LLMs. To limit personally identifiable in-
formation, the local part of email addresses was anonymized
and only domain-level information was retained; attachments
were ignored. We de-duplicated near-identical messages via
a hash of the normalized body andsubject+senderpairs,
filtered malformed samples, and retained only messages with
non-empty bodies. When threat-intelligence enrichment was
enabled, URLs and domains were extracted (using robust
URL parsing) and summarized into a compact reputation
snippet appended to the RAG context. For evaluation, we
used stratified splits and prevented leakage by (i) excluding
the query message from retrieval, and (ii) building the FAISS
index only from the legitimate emails in the training portion
for each run.
B. Implementation Details
All models were accessed through GroqCloud [17] API
to ensure consistent inference speed and reliability, with
identical prompt templates and parameter settings (temper-
ature=0.2) across all experiments. The system was imple-
mented in Python 3.10 and leverages a compact toolchain:
LangChain [18] to orchestrate the RAG pipeline and man-
age embeddings, FAISS [19] for efficient vector similarity
search,Sentence-Transformersfor generating text embed-
dings, Groq [15] for low-latency LLM inference, and Pan-
das [20] for data preprocessing and manipulation. The imple-
mentation emphasizes modularity and extensibility, enabling
straightforward integration of additional context sources or
(a) F1-score: With vs Without RAG
(b) FPR: With vs Without RAG
Fig. 2: Comparaison des mod `eles avec et sans RAG.
alternative LLM providers while preserving the core person-
alization architecture.
C. Results and Discussion
As illustrated by the paired bar plots in Fig. 2 (see FPR
in Fig. 2b and F1 in Fig. 2a), the visual trends align with
Table III: withRAG, FPR bars consistently contract while
F1 bars expand across models, indicating fewer false alarms
with a stronger overall balance. The shift is most conspicuous
formistral-saba, whose bars move markedly in the desired
direction, whereasgemma2-9b—though improved—remains
relatively prone to false positives. Consistent with the table,
llama4-scout with RAGstands out by pairing the shortest
FPR bar with one of the tallest F1 bars, withdeepseek-r1
with RAGclose behind.

Table III.Metrics With vs Without RAG Context (N=500)
Accuracy Recall Precision F1-score FPR
Model w/o w/ w/o w/ w/o w/ w/o w/ w/o w/
llama4-scout 0.9300 0.9700 0.9800 0.9800 0.8909 0.9608 0.9333 0.9703 0.1200 0.0400
deepseek-r1 0.8900 0.9600 1.0000 0.9800 0.8197 0.9423 0.9009 0.9608 0.2200 0.0600
mistral-saba 0.8220 0.9500 1.0000 1.0000 0.7375 0.9091 0.8489 0.9524 0.3560 0.1000
gemma2-9b 0.8000 0.8400 1.0000 1.0000 0.7143 0.7576 0.8333 0.8621 0.4000 0.3200
As summarized in Table III, addingRAGcontext markedly
improves decision quality across all models: false positives
drop, precision rises, and F1 strengthens, while recall was al-
ready high withoutRAG. This pattern suggests better decision
calibration via contextual disambiguation—fewer spurious
triggers with no loss in detection capability. Among the evalu-
ated configurations,llama4-scout with RAGdelivers the most
compelling performance, combining the best precision–recall
balance with the lowest propensity for false alarms.deepseek-
r1 with RAGfollows closely and is a strong alternative when
minimizing false positives is paramount.mistral-sabasees
the largest relative benefit fromRAG—evidence that a model
initially “generous” in alerts can be substantially stabilized
by context—yet it remains slightly behind the top two. By
contrast,gemma2-9b, though improved, remains more prone
to false positives and is less suitable where analyst workload
must be tightly controlled. In short,llama4-scout with RAG
is the recommended choice for superior overall performance
and smoother operational use.
V. CONCLUSION
We presented a novel, privacy-awareLLM+RAGpipeline for
phishing email detection that grounds model decisions in user-
specific mailbox context and lightweight threat intelligence.
Across four heterogeneous open models on a balanced email
benchmark, the system consistently reduced false positives
while preserving high recall, yielding clear gains in preci-
sion and F1-score. These results support the feasibility of
retrieval-augmented decisioning for operational email screen-
ing and highlight persistent stressors in long, multi-threaded
conversations, aggressively obfuscated URLs and domains,
and domain shifts that can dilute retrieval quality. Looking
ahead, we focus on two priorities: strengthening robustness
and generalization via multilingual, institution-scale corpora
and targeted ablations of retrieval, prompt, and model choices
under domain shift; and improving operational reliability and
efficiency through calibrated uncertainty for cost-aware triage
and lightweight pipelines (retrieval pruning and distilled small
models) to maintain low latency and a modest computational
footprint.
REFERENCES
[1] S. Salloum, T. Gaber, S. Vadera, and K. Shaalan, “A systematic literature
review on phishing email detection using natural language processing
techniques,”Ieee Access, vol. 10, pp. 65 703–65 727, 2022.[2] P. C. R. Chinta, C. S. Moore, L. M. Karaka, M. Sakuru, V . Bodepudi,
and S. R. Maka, “Building an intelligent phishing email detection system
using machine learning and feature engineering,”European Journal of
Applied Science, Engineering and Technology, vol. 3, no. 2, pp. 41–54,
2025.
[3] A. Diaf, A. A. Korba, N. E. Karabadji, and Y . Ghamri-Doudane,
“Bartpredict: Empowering iot security with llm-driven cyber threat
prediction,” inGLOBECOM 2024-2024 IEEE Global Communications
Conference. IEEE, 2024, pp. 1239–1244.
[4] A. Tellache, A. A. Korba, A. Mokhtari, H. Moldovan, and Y . Ghamri-
Doudane, “Advancing autonomous incident response: Leveraging llms
and cyber threat intelligence,”arXiv preprint arXiv:2508.10677, 2025.
[5] M. M. Dif, M. A. Bouchiha, A. A. Korba, and Y . Ghamri-Doudane,
“Towards trustworthy agentic ioev: Ai agents for explainable cyberthreat
mitigation and state analytics,” in2025 IEEE 50th Conference on Local
Computer Networks (LCN). IEEE, 2025, pp. 1–10.
[6] G. Desolda, F. Greco, and L. Vigano, “Apollo: A gpt-based tool to
detect phishing emails and generate explanations that warn users,”arXiv
preprint arXiv:2410.07997, 2024.
[7] T. Koide, N. Fukushi, H. Nakano, and D. Chiba, “Chatspamdetector:
Leveraging large language models for effective phishing email detec-
tion,”arXiv preprint arXiv:2402.18093, 2024.
[8] C. Lee, “Enhancing phishing email identification with large language
models,”arXiv preprint arXiv:2502.04759, 2025.
[9] B. B. Gupta, N. A. Arachchilage, and K. E. Psannis, “Defending
against phishing attacks: taxonomy of methods, current issues and future
directions,”Telecommunication Systems, vol. 67, pp. 247–267, 2018.
[10] F. Heiding, B. Schneier, A. Vishwanath, J. Bernstein, and P. S. Park,
“Devising and detecting phishing emails using large language models,”
IEEE Access, 2024.
[11] J. Zhang, P. Wu, J. London, and D. Tenney, “Benchmarking and
evaluating large language models in phishing detection for small and
midsize enterprises: A comprehensive analysis,”IEEE Access, 2025.
[12] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” inAdvances in
Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.
[13] F. Heiding, B. Schneier, A. Vishwanath, and J. Bernstein, “Devising
and detecting phishing: Large language models vs. smaller human
models,” 2023, arXiv:2308.12287. [Online]. Available: https://arxiv.org/
abs/2308.12287
[14] Sentence-Transformers, “all-minilm-l6-v2,” https://huggingface.co/
sentence-transformers/all-MiniLM-L6-v2, 2021, accessed: 2025-10-31.
[15] I. Groq, “Lpu architecture: Deterministic, low-latency inference,”
2025, accessed: 2025-10-30. [Online]. Available: https://groq.com/
lpu-architecture
[16] German University of Technology in Oman (GUtech), “German
university of technology in oman,” 2025, accessed: 2025-10-30.
[Online]. Available: https://www.gutech.edu.om
[17] I. Groq, “Groqcloud: Low-latency inference platform for large
language models,” 2025, accessed: 2025-10-30. [Online]. Available:
https://groq.com/groqcloud
[18] LangChain, “Langchain: Open-source framework for building llm
applications,” 2025, accessed: 2025-10-30. [Online]. Available: https:
//www.langchain.com/
[19] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,” 2024.
[20] T. pandas development team, “pandas-dev/pandas: Pandas,” Feb. 2020.
[Online]. Available: https://doi.org/10.5281/zenodo.3509134
# Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey

**Authors**: Minghan Li, Xinxuan Lv, Junjie Zou, Tongna Chen, Chao Zhang, Suchao An, Ercong Nie, Guodong Zhou

**Published**: 2025-09-09 14:31:11

**PDF URL**: [http://arxiv.org/pdf/2509.07794v1](http://arxiv.org/pdf/2509.07794v1)

## Abstract
Modern information retrieval (IR) must bridge short, ambiguous queries and
ever more diverse, rapidly evolving corpora. Query Expansion (QE) remains a key
mechanism for mitigating vocabulary mismatch, but the design space has shifted
markedly with pre-trained language models (PLMs) and large language models
(LLMs). This survey synthesizes the field from three angles: (i) a
four-dimensional framework of query expansion - from the point of injection
(explicit vs. implicit QE), through grounding and interaction (knowledge bases,
model-internal capabilities, multi-turn retrieval) and learning alignment, to
knowledge graph-based argumentation; (ii) a model-centric taxonomy spanning
encoder-only, encoder-decoder, decoder-only, instruction-tuned, and
domain/multilingual variants, highlighting their characteristic affordances for
QE (contextual disambiguation, controllable generation, zero-/few-shot
reasoning); and (iii) practice-oriented guidance on where and how neural QE
helps in first-stage retrieval, multi-query fusion, re-ranking, and
retrieval-augmented generation (RAG). We compare traditional query expansion
with PLM/LLM-based methods across seven key aspects, and we map applications
across web search, biomedicine, e-commerce, open-domain QA/RAG, conversational
and code search, and cross-lingual settings. The review distills design
grounding and interaction, alignment/distillation (SFT/PEFT/DPO), and KG
constraints - as robust remedies to topic drift and hallucination. We conclude
with an agenda on quality control, cost-aware invocation, domain/temporal
adaptation, evaluation beyond end-task metrics, and fairness/privacy.
Collectively, these insights provide a principled blueprint for selecting and
combining QE techniques under real-world constraints.

## Full Text


<!-- PDF content starts -->

Query Expansion in the Age of Pre-trained and Large Language Models: A
Comprehensive Survey
MINGHAN LI,School of Computer Science and Technology, Soochow University, China
XINXUAN LV,School of Computer Science and Technology, Soochow University, China
JUNJIE ZOU,School of Computer Science and Technology, Soochow University, China
TONGNA CHEN,School of Computer Science and Technology, Soochow University, China
CHAO ZHANG,School of Computer Science and Technology, Soochow University, China
SUCHAO AN,School of Computer Science and Technology, Soochow University, China
ERCONG NIE,Center for Information and Language Processing (CIS), LMU Munich, Germany and Munich Center
for Machine Learning (MCML), Germany
GUODONG ZHOU,School of Computer Science and Technology, Soochow University, China
Modern information retrieval (IR) must bridge short, ambiguous queries and ever more diverse, rapidly evolving corpora. Query
Expansion (QE) remains a key mechanism for mitigating vocabulary mismatch, but the design space has shifted markedly with
pre-trained language models (PLMs) and large language models (LLMs). This survey synthesizes the field from three angles: (i) a four-
dimensional framework of query expansion - from the point of injection (explicit vs. implicit QE), through grounding and interaction
(knowledge bases, model-internal capabilities, multi-turn retrieval) and learning alignment, to knowledge graph-based argumentation;
(ii) a model-centric taxonomy spanning encoder-only, encoder-decoder, decoder-only, instruction-tuned, and domain/multilingual
variants, highlighting their characteristic affordances for QE (contextual disambiguation, controllable generation, zero-/few-shot
reasoning); and (iii) practice-oriented guidance on where and how neural QE helps in first-stage retrieval, multi-query fusion, re-
ranking, and retrieval-augmented generation (RAG). We compare traditional query expansion with PLM/LLM-based methods across
seven key aspects, and we map applications across web search, biomedicine, e-commerce, open-domain QA/RAG, conversational and
code search, and cross-lingual settings. The review distills design grounding and interaction, alignment/distillation (SFT/PEFT/DPO),
and KG constraints - as robust remedies to topic drift and hallucination. We conclude with an agenda on quality control, cost-aware
invocation, domain/temporal adaptation, evaluation beyond end-task metrics, and fairness/privacy. Collectively, these insights provide
a principled blueprint for selecting and combining QE techniques under real-world constraints.
CCS Concepts:•Information systems→Information retrieval query processing.
Additional Key Words and Phrases: Query Expansion, Information Retrieval, Large Language Models
1 Introduction
Modern IR systems must interpret increasingly short, context-poor queries issued via mobile, voice, and conversational
interfaces, while the available content grows in scale, diversity, and linguistic complexity. The resulting vocabulary
mismatch—users express intents with general or ambiguous terms whereas relevant documents use domain-specific
Authors’ Contact Information: Minghan Li, School of Computer Science and Technology, Soochow University, Suzhou, Jiangsu, China, mhli@suda.edu.cn;
Xinxuan Lv, School of Computer Science and Technology, Soochow University, Suzhou, Jiangsu, China, xxlv@stu.suda.edu.cn; Junjie Zou, School of
Computer Science and Technology, Soochow University, Suzhou, Jiangsu, China, jjzou1@stu.suda.edu.cn; Tongna Chen, School of Computer Science and
Technology, Soochow University, Suzhou, Jiangsu, China, tnchentnchen@stu.suda.edu.cn; Chao Zhang, School of Computer Science and Technology,
Soochow University, Suzhou, Jiangsu, China, czhang1@stu.suda.edu.cn; Suchao An, School of Computer Science and Technology, Soochow University,
Suzhou, Jiangsu, China, scan@stu.suda.edu.cn; Ercong Nie, Center for Information and Language Processing (CIS), LMU Munich, Munich, Germany
and Munich Center for Machine Learning (MCML), Munich, Germany, nie@cis.lmu.de; Guodong Zhou, School of Computer Science and Technology,
Soochow University, Suzhou, Jiangsu, China, gdzhou@suda.edu.cn.
1arXiv:2509.07794v1  [cs.IR]  9 Sep 2025

2 Li et al.
terminology, paraphrases, or emerging jargon—is a long-standing obstacle. Context is often omitted, further hindering
accurate intent inference. QE addresses these challenges by enriching the initial query with semantically related,
contextually appropriate material so as to increase overlap with relevant documents.
Classical QE spans pseudo-relevance feedback (PRF), thesaurus/ontology-based expansion, translation-based models,
and log-driven reformulation [ 46,96,104]. Early benchmarks (e.g., TREC, CLEF) standardized evaluations and demon-
strated the utility of PRF and semantic resources. Yet traditional techniques typically rely on static associations or
shallow co-occurrence statistics and struggle with short, ambiguous, or long-tail queries [ 16,104]. Topic drift, limited
domain coverage, and the precision–recall trade-off remain recurring pain points.
Recent advances in PLMs and LLMs—e.g., BERT/RoBERTa-style encoders for context-sensitive representations
[81], encoder–decoders such as T5/BART for controlled generation, and decoder-only LLMs (e.g., GPT-3/4, PaLM,
LLaMA-family) for zero-/few-shot reasoning—have opened a broader design space for QE. These models support implicit
expansion in the embedding space (e.g., PRF with dense vectors), selection-based term filtering with contextual encoders,
and generative expansion via pseudo-documents or structured rationales. For instance, Query2Doc synthesizes pseudo-
documents and yields 3–15% gains on MS MARCO and TREC DL [ 107]. Instruction tuning and in-context learning
further improve format adherence and zero-shot robustness [ 126]. Open challenges persist: factual grounding, domain
transfer under data sparsity, precision control for generative outputs, and cost/latency constraints at deployment.
Query expansion, as a classic problem in the field of information retrieval, has been systematically reviewed by
numerous scholars. Early survey works, such as the study by Bhogal et al . [13] provided a comprehensive overview of
semantic QE; Sartori [97] focused on ontology-driven QE, pointing out that ontologies can help obtain higher-quality
expansion terms;Lei et al . [49] summarizes the QE based on local analysis; Carpineto and Romano [16] offered a brief
summary of automatic QE, covering techniques based on linguistics, corpora, the query itself, retrieval logs, and
Web data, and discussed related key issues; Rivas et al . [94] , confirmed that combining multiple classical techniques
can effectively improve retrieval performance; Subsequently Raza et al . [93] systematically discussed statistical QE
including document analysis, retrieval and browsing log analysis, and Web knowledge analysis, while Azad and Deepak
[8]reviewed the core techniques, data sources, weighting and ranking methods, user involvement, and application
scenarios of QE. However, these existing surveys primarily focus on traditional technical frameworks. In recent years,
deep learning, represented by PLM and LLM, has achieved breakthroughs, bringing new opportunities and challenges
to QE with their powerful semantic understanding and generation capabilities. Although new QE methods based
on modern language models are emerging endlessly, we find that the academic community still lacks a survey that
comprehensively covers this emerging technological wave. Therefore, this paper aims to fill this gap by systematically
reviewing, classifying, and prospecting query expansion techniques that integrate modern PLM/LLM, providing a clear
landscape and valuable reference for subsequent research in this field.
Scope and contributions.This survey synthesizes the landscape of QE in the PLM/LLM era with three unifying
perspectives:
(1)Paradigms.We trace the evolution from symbolic/statistical QE to implicit/embedding QE, selection-based
explicit QE, and generative QE, highlighting how PLMs/LLMs reshape each paradigm.
(2)Model taxonomy.We categorize PLMs/LLMs relevant to QE (encoder-only, encoder–decoder, decoder-only,
instruction-tuned, domain/multilingual) and articulate their characteristic affordances for QE (disambiguation,
controllable generation, zero-shot reasoning).

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 3
Query
ExpansionTraditional
Query ExpansionGlobal Analysis: Corpus-Wide
and External KnowledgeCorpus-Wide Statistical MethodsTerm Co-occurrence and Clustering: Minker et al. [72], Willett [115],
Peat and Willett [83],
Global Similarity Thesaurus: Qiu and Frei [88]
Pre-defined Knowledge ResourceGeneral-purpose Lexicons: Voorhees [103]
Domain Ontologies: Bhogal et al. [13], Lu et al. [64], Aronson and Rindflesch [5],
Fu et al. [28]
Local Analysis:
Query-Dependent EvidencePRF/RF: Rocchio Jr [96], Robertson and Jones [95]
LCA: Xu and Croft [118, 119]
Query Logs and User BehaviorMining reformulations or Click-based semantic proximity: Wen et al. [114],
Cui et al. [20], Cui et al. [21], Baeza-Yates et al. [9], Yin et al. [122]
Query Expansion of
PLMs/LLMsPre-trained Model LandscapeEncoder-only: BERT [23], RoBERT [62]
Encoder–Decoder: BART [53], T5 [89]
Decoder-only LLMsGPT-3/4 [3, 27], PaLM [4], LLaMA-family
Instruction-tuned Models: InstructGPT, FLAN-T5, LLaMA-2 Chat,[82, 101, 111]
Domain-specific / Multilingual: BioBERT[47], UMLS-BERT[71], SciBERT[11],
multilingual BERT/RoBERTa families
PLM/LLM-Driven QE TechniquesPoint of injectionImplicit Embedding-based QE: ANCE-PRF [124], ColBERT-PRF [109], Eclipse [22],
QB-PRF [130], LLM-VPRF [56]
Selection-based Explicit QE: CEQE [74], SQET [75], BERT-QE [133], CQED [39],
PQEWC [10]
Grounding and interactionZero-Grounding, Non-Interactive QE: LLM-Driven Single-Stage Expansion:
Query2Doc [107], CoT-QE [35], GAR [69], GRF [66], HyDE [29], Exp4Fuse [60],
Contextual clue sampling and fusion [59], SEAL [12]
Grounding-Only, Non-Interactive QE: Corpus-Evidence Anchored Single-Pass Expansion:
MILL [36], AGR [18], EAR [19], GenPRF [108], CSQE [50], MUGI [128], PromptPRF [55],
FGQE [34]
Grounding-Aware Interactive QE: Multi-Round Retrieve-Expand Loops:
InteR [26], ProQE [92], LameR [99], ThinkQE [51]
Learning and Alignment for QE with PLMs/LLMs SoftQE [87], RADCoT [48], ExpandR [121], AQE [120]
KG-augmented Query ExpansionEntity-based Expansion via Knowledge Graph Retrieval: KGQE [85],
CL-KGQE [90]
Entity-based Expansion via Hybrid KG and Document Graph: KAR [117], QSKG [67]
Application DomainsWeb Search Engines: EQE [131]
Biomedical Information Retrieval: Kelly [38], Khader et al. [40],
Ateia and Kruschwitz [7], Niu et al. [80]
E-commerce Search: BEQUE [84], Wang et al. [106], Zhang et al. [129]
Cross-Lingual and Multilingual Search: Rajaei et al. [91]
Open-Domain Question Answering: GAR [69], EAR [19], AQE [120], MILL [36],
AGR [18], HTGQE [134], TDPR [57]
Retrieval-Augmented Generation (RAG): QE-RAGpro [31], QOQA [42],
KG-Infused RAG [116], Rajaei et al. [91]
Conversational Search: PRL [73], MISE [44], QuReTeC [105]
Code Search: GACR [54], SSQR [70], SG-BERT/GPT2 [61], ECSQE [14]
Fig. 1. A Taxonomy of Query Expansion Techniques: From Traditional Methods to PLM/LLM-Driven Techniques and Applications.
(3)Practice and impact.We connect methods to deployment concerns (efficiency, latency, safety), compare
traditional vs. neural QE, and review applications (web, biomedical, e-commerce, ODQA/RAG, conversational,
code search).
Organization.Sec. 2 revisits foundations and traditional QE methods. Sec. 3 overviews pre-trained model fami-
lies. Sec. 4 organizes Query Expansion techniques into four key dimensions: Point of Injection(implicit/embedding,
selection-based explicit), Grounding and Interaction(integration of external knowledge and feedback), Learning and
Alignment(training methods like SFT, PEFT, DPO), and KG-Augmented. Sec. 5 provides a comparative analysis of
traditional vs. neural QE. Sec. 6 surveys application domains and use cases. Sec. 7 discusses open challenges and future
directions.

4 Li et al.
2 Foundations of Query Expansion: Concepts and Traditional Methods
QE has been a central theme in classical IR [ 8,13,16,25,49,93,94,97]. Its goal is to increase lexical/semantic overlap
between a user query and relevant documents, particularly when queries are short, ambiguous, or underspecified.
Typical failures such as “car maintenance” missing “vehicle servicing” arise from vocabulary mismatch; QE mitigates
this by adding related terms while preserving intent.
2.1 Concepts and Notation
Let the original query be a multiset of terms 𝑄, stopwords 𝑇′′⊆𝑄, and a set of candidate expansions 𝑇′from some
source𝐷. Following Azad and Deepak [8], the expanded query is
𝑄exp=(𝑄−𝑇′′) ∪𝑇′.(1)
However, this set-based view is a simplification. In practice, QE is not merely about adding terms but also about
re-weighting them to reflect their importance. The core difficulty is selecting 𝑇′that improves recall without distorting
intent [43]. Throughout, we write𝐷 𝑘for top-𝑘feedback documents, andcos(·,·)is cosine similarity.
2.2 Global Analysis: Corpus-Wide and External Knowledge
Global methods operate on the assumption that stable, useful term associations can be inferred from large-scale evidence,
independent of any single query. This evidence can be statistical patterns from the entire document corpus or curated
knowledge from external resources.
2.2.1 Corpus-Wide Statistical Methods.This family assumes the target corpus carries sufficient regularities to infer
useful associations.
Term Co-occurrence and Clustering.A classic approach infers related terms from co-occurrence statistics in documents.
A common association is the cosine on binary/TF document incidence:
cos(𝑋,𝑌)=𝐹(𝑋,𝑌)√︁
𝐹(𝑋)𝐹(𝑌),(2)
with𝐹(𝑋) and𝐹(𝑌) denoting the document frequencies of terms 𝑋and𝑌, respectively, and 𝐹(𝑋,𝑌) representing the
co-occurrence count of 𝑋and𝑌in the same documents. While simple and unsupervised, Peat and Willett [83] show a
structural weakness: maximizing the cosine tends to select terms with similar (often high) document frequencies to the
query term, which are weak discriminators under the Robertson–Sparck Jones (RSJ) theory [ 95]. Hence, co-occurrence
expansion helps most for low-frequency query terms and can hurt for high-frequency ones.
Global Similarity Thesaurus.Beyond local co-occurrence, global methods build a corpus-wide term association
resource [ 88]. Let𝑇∈R𝑚×𝑛be a term×document matrix (e.g., length-normalized tf–idf). A symmetric similarity matrix
𝑆=𝑇𝑇⊤induces, for each term𝑡, a ranked list of nearest neighbors (a “thesaurus”). Given a query with weights{𝑞 𝑖},
define a query-concept vector as the mixture of its term profiles and rank candidate terms by
Sim(𝑞,𝑡)=∑︁
𝑡𝑖∈𝑞𝑞𝑖𝑆[𝑡𝑖,𝑡].(3)
Selecting the top- 𝑟terms yields a concept-consistent expansion list. Latent Semantic Analysis (LSA) [ 1]serves a similar
purpose by mapping terms into a low-dimensional latent space, grouping semantically related terms together. These

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 5
methods avoid hard clustering and allow variable-length expansions, but remain context-agnostic and can propagate
popular-but-generic terms.
2.2.2 Methods Using Pre-defined Knowledge Resources.Curated resources inject explicit semantics and reduce ambiguity
[76].
General-purpose Lexicons (e.g., WordNet).WordNet offers synsets and IS-A hierarchies for broad-coverage syn-
onymy/hypernymy. However, short queries lack context for reliable WSD; naive synset expansion can underperform
stemming/bag-of-words [ 13,103]. Using glosses or intersecting sense neighborhoods can help [ 77], but static coverage
and sense ambiguity remain key issues for complex queries.
Domain Ontologies.Domain ontologies (e.g., UMLS/MeSH in biomedicine; legal/tourism ontologies) enable precise,
concept-level expansion with fewer sense conflicts [ 13]. Examples include mapping “heart attack” to “myocardial
infarction” and related treatments/symptoms [ 5,64]; composing tourism and geographic ontologies to resolve spatial
language [ 28]. Benefits are high precision and expert semantics; costs are curation/maintenance and domain dependence.
2.3 Local Analysis: Query-Dependent Evidence
In contrast to global methods, local analysis techniques derive expansion terms from a small set of documents retrieved
in response to the current query. This approach is inherently context-aware but is highly dependent on the quality of
the initial search results.
2.3.1 Relevance Feedback (RF) and Pseudo-Relevance Feedback (PRF).RF adapts the query using assessed relevant/non-
relevant documents; PRF automates this by assuming top-𝑘are relevant.
Vector-space (Rocchio).RF relies on explicitly labeled document sets to refine the query. Given a set of relevant
documents𝐷𝑟and an optional set of non-relevant documents 𝐷𝑛𝑟, Rocchio [ 96] proposes a vector-space update rule to
construct the modified query vector:
®𝑞𝑚=𝛼®𝑞 0+𝛽1
|𝐷𝑟|∑︁
®𝑑∈𝐷𝑟®𝑑−𝛾1
|𝐷𝑛𝑟|∑︁
®𝑑∈𝐷𝑛𝑟®𝑑,(4)
where𝛼,𝛽,𝛾 are non-negative weighting parameters: 𝛼controls the influence of the original query vector ®𝑞0,𝛽
adjusts the contribution of positive evidence from 𝐷𝑟, and𝛾modulates the impact of negative evidence from 𝐷𝑛𝑟.
For PRF, only the positive centroid is used, with 𝐷𝑟approximated by the top- 𝑘initially retrieved documents 𝐷𝑘as
pseudo-relevant feedback.
Probabilistic (RSJ) weighting.Alternatively, expansion weights follow the RSJ discrimination:
𝑤(𝑡)=log𝑃(𝑡|𝑅)[1−𝑃(𝑡|𝑁𝑅)]
𝑃(𝑡|𝑁𝑅)[1−𝑃(𝑡|𝑅)],(5)
prioritizing terms overrepresented in relevant documents [ 95]. In PRF,𝑃(𝑡|𝑅) is estimated on 𝐷𝑘and𝑃(𝑡|𝑁𝑅) against
the background.
2.3.2 Local Context Analysis (LCA).LCA refines PRF by requiring a candidate concept to co-occur with most terms in
the original query ( 𝑄, with individual query terms {𝑤𝑖}) within the local top- 𝑛pseudo-relevant document set ( 𝑆, i.e.,

6 Li et al.
the top-𝑛documents initially retrieved in PRF), reducing topic drift [ 118,119]. Let “concepts” 𝑐denote terms or phrases.
We first define key metrics, where:
-𝑡𝑓(𝑥,𝑦) : Term Frequency of 𝑥(concept/term) in document 𝑦, measured as the number of times 𝑥appears in𝑦
(length-normalized to avoid bias from long documents);
-𝑖𝑑𝑓(𝑥) : Inverse Document Frequency of 𝑥, calculated as log10𝑁
𝑑𝑓(𝑥)(𝑁is total documents in the corpus, 𝑑𝑓(𝑥) is
the number of documents containing𝑥);
-𝛿: A small positive smoothing parameter (𝛿>0) to avoid zero values in product operations.
co(𝑐,𝑤𝑖)=∑︁
𝑑∈𝑆𝑡𝑓(𝑐,𝑑)𝑡𝑓(𝑤 𝑖,𝑑),(6)
co_degree(𝑐,𝑤 𝑖)=log10(𝑛)log10 co(𝑐,𝑤𝑖)+1·idf(𝑐),(7)
𝑔(𝑐,𝑄)=Ö
𝑤𝑖∈𝑄 𝛿+co_degree(𝑐,𝑤 𝑖),(8)
𝑓(𝑐,𝑄)=Ö
𝑤𝑖∈𝑄 𝛿+co_degree(𝑐,𝑤 𝑖)idf(𝑤𝑖),(9)
Concepts are ranked by 𝑓(𝑐,𝑄) , and the top- 𝑘concepts are appended to the original query for expansion. Empirically,
the “soft-AND” product in 𝑔(𝑐,𝑄) and𝑓(𝑐,𝑄) favors candidates supported by all query terms (rather than just individual
terms), giving LCA stronger robustness against topic drift compared to frequency-only PRF methods.
2.4 Methods Using Query Logs and User Behavior
At web scale, logs capture implicit reformulations and intent signals [9, 20, 21, 114, 122].
Mining reformulations.For a user’s original query 𝑄, if a subsequent query reformulation 𝑄′frequently follows
𝑄and leads to successful user interactions (e.g., clicks, long dwell time), terms 𝑡that appear in 𝑄′but not in𝑄are
treated as expansion candidates [ 20,21]. To quantify the association between 𝑄and each candidate 𝑡, we use a statistic
assoc(Q,t) that calculates the proportion of𝑄’s submissions where the subsequent𝑄′contains𝑡:
assoc(𝑄,𝑡)=count 𝑄→𝑄′with𝑡∈𝑄′
count(𝑄).(10)
Click-based semantic proximity.Queries leading to overlapping clicked document sets are semantically related. Let
𝐷𝑄denote the historical clicked set of a query 𝑄(i.e., all documents clicked by users after submitting 𝑄); the semantic
similarity between two queries𝑄 1(to be expanded) and𝑄 2is calculated as:
sim(𝑄 1,𝑄2)=|𝐷𝑄1∩𝐷𝑄2|
|𝐷𝑄1∪𝐷𝑄2|,(11)
and terms from 𝑄2(weighted by simand within- 𝑄2importance) can expand 𝑄1. These approaches are strongly
data-driven but suffer from cold-start/long-tail sparsity and privacy constraints.
2.5 Limitations of Traditional QE
Although diverse in their approach, traditional QE techniques are bound by a shared, fundamental limitation: they
operate on surface-level lexical statistics or static knowledge structures, lacking a deep model of dynamic, compositional
semantics. This core deficiency manifests in several ways:

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 7
(1)Context Insensitivity: Global methods, whether statistical or knowledge-based, are inherently context-agnostic.
WordNet’s failure to resolve ambiguity in short queries is a direct consequence of this, as the correct "sense" of
a word is determined by its surrounding terms—a signal these models cannot effectively use.
(2)Brittleness of Local Evidence: Local methods like PRF are highly susceptible to topic drift because they treat
documents as bags of co-occurring words. They lack the ability to understand the compositional intent of a
query like "running shoes" and may drift towards the more general and popular topic of "shoes" if the initial
results are noisy.
(3)Inability to Generalize: Log-based methods are powerful but cannot generalize to novel queries. They rely on
observing past user behavior for specific query patterns and fail when encountering new combinations of terms
that express a previously unseen intent.
Ultimately, traditional methods struggle to bridge the "semantic gap" because they lack a true understanding of how
language creates meaning through the combination and context of words. This persistent challenge set the stage
for a paradigm shift in IR, creating a clear need for a new class of models that could learn deep, contextual, and
compositional representations of language. These limitations motivate PLM/LLM-era methods that bring contextual
encoding, generative reasoning, and stronger grounding to QE (see Sec. 4).
3 Pre-trained Model Landscape
Modern QE increasingly relies on PLMs and LLMs. While earlier neural encoders captured local semantics with limited
context [ 86], contemporary pre-training paradigms deliver rich contextualization, generation, and instruction-following,
which map naturally to key QE operations such as context-aware term selection, pseudo-document generation, and
controllable reformulation. This section reviews the major PLM/LLM families and articulates their typical affordances
for QE; concrete techniques are discussed in detail in Sec. 4.
3.1 Encoder-only
Encoder-only transformers (e.g., BERT, RoBERTa) are bidirectional masked LMs trained to produce context-sensitive
token/sequence representations [23, 62]. For QE, their strengths are:
•Context-sensitive disambiguation.For queries like “Apple launch event”, encoder-only models resolve sense
and yield embeddings that align with Apple Inc. rather than the fruit, enabling safer selection-based expansion
(e.g., synonym/phrase picking) instead of blind synonym injection.
•Implicit/embedding-space expansion.They naturally support representation augmentation (e.g., PRF over
dense vectors, token-level reweighting) without generating surface forms, which mitigates hallucination and
reduces latency.
These models are a strong fit for PRF-style, selection-based, or reranking-centric QE where precision and efficiency are
paramount (see Sec. 4).
3.2 Encoder–Decoder
Sequence-to-sequence (seq2seq) models pair a bidirectional encoder with an autoregressive decoder, combining under-
standing and generation. BART implements denoising autoencoding with span permutation/masking, synthesizing
BERT-like encoding and GPT-like decoding [ 53]. T5 unifies tasks under a text-to-text objective and scales with large
pre-training corpora [89]. For QE, encoder–decoders offer:

8 Li et al.
•Direct generation of expansions.They produce answer-like pseudo-documents, titles, or rationales that
enrich sparse/dense retrieval.
•Structure-aware control.Prompts or templates can steer outputs to keywords, entities, or facets, improving
interpretability and reducing topic drift.
Trade-offs include decoding latency and the need to control generation quality; however, they remain the dominant
backbone for generative QE methods.
3.3 Decoder-only LLMs
Decoder-only transformers (e.g., GPT-3/4, PaLM, LLaMA-family) are trained autoregressively on massive corpora
[3,4,27,102]. At sufficient scale, they exhibit emergent capabilities such as in-context learning and zero-shot reasoning
[112]. For QE, this translates to:
•Zero-/few-shot expansion.Without task-specific fine-tuning, LLMs can infer latent intents and generate
high-coverage reformulations or pseudo-documents.
•Reasoning-guided reformulation.Chain-of-thought style prompts elicit structured expansions that surface
latent entities/relations beneficial to retrieval.
Decoder-only LLMs offer the greatest flexibility but bring higher inference cost and require safeguards (e.g., grounding,
self-checks) to prevent hallucinated entities in expansions.
3.4 Instruction-tuned & SFT/RLHF Models
Instruction-tuned models (e.g., InstructGPT, FLAN-T5, LLaMA-2 Chat) apply supervised fine-tuning (SFT) on instruction–
response pairs and often reinforcement learning with human feedback (RLHF) for preference alignment [ 82,101,111].
Compared with base LMs, they:
•Follow prompts reliably.Better adherence to requested output formats (keyword lists, entity tables, facet
trees) is valuable for QE pipelines that fuse multiple signals.
•Generalize in zero/few-shot settings.Aligned models often produce higher-precision expansions with fewer
prompt heuristics [110].
They are ideal front-ends for controlled generation (e.g., multi-query, rationale-then-keywords), especially when
transparency and editability of expansions are required.
3.5 Domain-specific and Multilingual Models
Specialized pre-training narrows vocabulary mismatch in professional and non-English settings:
•Domain-specific encoders.BioBERT adapts BERT on PubMed/PMC for biomedical text [ 47]; UMLS-BERT
injects UMLS CUIs/semantic types [ 71]; SciBERT trains on scientific full text with SCIVOCAB to better cover
scholarly terminology [ 11]. These models consistently improve entity resolution and concept linking, which
directly benefits QE in biomedical and scientific IR.
•Multilingual encoders/LLMs.Multilingual variants (e.g., multilingual BERT/RoBERTa families) support
cross-lingual QE by mapping semantically aligned terms across languages, enabling bilingual/multilingual
reformulations without parallel supervision. In practice, they are used to (i) translate-and-expand or (ii) expand-
then-translate within cross-lingual retrieval pipelines.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 9
Table 1. Pre-trained model families and typical QE affordances.
Family Pre-training gist QE strengths (typical use) Common trade-offs
Encoder-only (BERT,
RoBERTa) [23, 62]Bidirectional masked LM;
strong contextual encodingsTerm/phrase selection, PRF in em-
bedding space; robust disambigua-
tion; efficient rerankingLimited free-form generation;
depends on candidate sources
Encoder–Decoder
(BART, T5) [53, 89]Denoising/text-to-text
seq2seq; encoder+decoderStructured generative QE:
pseudo-docs, titles, rationales;
promptable output controlDecoding latency;
quality control of generations
Decoder-only LLM
(GPT-3/4, PaLM, LLaMA)
[3, 4, 27]Autoregressive LM at scale;
in-context learning [112]Zero-/few-shot expansion;
reasoning-guided reformulation;
broad domain coverageHigher cost; hallucination risk
without grounding
Instruction-tuned (Instruct-
GPT, FLAN-T5, LLaMA-2
Chat) [82, 101, 111]SFT + (optionally) RLHF;
preference-aligned outputsFormat adherence
(keywords/entities/facets); safer,
more controllable expansionsMay over-adhere to instructions;
still needs retrieval grounding
Domain-specific / Multilin-
gual [11, 47, 71]Further pre-training on tar-
get domains/languagesBridges domain/lingual vocabulary
gaps; better entity/term recallNarrow transfer outside domain;
requires domain corpora
3.6 Architectural Affordances for QE: A Practitioner’s View
Table 1 summarizes typical affordances and trade-offs by model family. In short, encoder-only models excel at
selection/embedding-space QE with strong efficiency; encoder–decoders add structured generation under tighter
control; decoder-only LLMs unlock zero-shot, reasoning-driven expansion; instruction-tuned models improve format
adherence and safety; domain/multilingual variants mitigate vocabulary and language gaps. Section 4 details concrete
algorithms that instantiate these affordances.
Practical guidance.When precision and latency dominate (e.g., production web search, e-commerce), start with
encoder-only selection/PRF and add small encoder–decoder components for targeted generation. For recall-limited or
long-tail queries, leverage decoder-only LLMs or instruction-tuned seq2seq for zero-shot multi-query expansion, but
incorporate corpus-steering/feedback (see Sec . 4) to curb drift. In domain or cross-lingual settings, prefer specialized
encoders and multilingual models to reduce reliance on noisy translation.
From Model Selection to QE Implementation.The preceding guidance maps PLM/LLM families to retrieval priorities.
Realizing their QE value requires addressing four key questions: how to inject expansion signals, ground expansions in
corpus evidence, align generation with retrieval goals, and integrate structured knowledge. Chapter 4 operationalizes
these as four axes to bridge model capabilities with practical techniques, forming a unified framework for neural query
expansion. These questions and axes structure the technical discussion in Chapter 4.
4 PLM/LLM-Driven QE Techniques
Modern query expansion revisits a classic idea—add missing evidence to a query—under the lens of PLMs and LLMs.
What changes in the LLM era is not the goal butwhereandhowthe additional signal is injected: into the query’s vector
representation, into a curated set of explicit terms or spans, or into freshly produced text that stands in for what the user
might have asked if they had been more specific. Equally important are the choices ofgrounding(does expansion come

10 Li et al.
from the corpus or from a model’s world knowledge?),learning regime(prompt-only vs. alignment and distillation),
andinteraction pattern(single shot vs. feedback-in-the-loop).
Our organizing view.We structure the literature along four complementary axes that unify “traditional” QE with
PLM/LLM-based methods.
(i)Point of injection.Implicit, embedding-levelmethods strengthen the query vector or token embeddings
without emitting new terms (e.g., ANCE-PRF, ColBERT-PRF, Eclipse, QB-PRF, LLM-VPRF).Selection-based
explicitmethods keep the vocabulary corpus-grounded by ranking terms or spans from pseudo-relevant
documents or curated resources (e.g., CEQE, SQET, BERT-QE, CQED, PQEWC).
(ii)Grounding and interaction.Zero-grounding, non-interactiveapproaches rely mainly on an LLM’s prior
knowledge, sometimes with minimal corpus hints (e.g., Query2Doc, CoT-QE, HyDE, Exp4Fuse), and issue the
expanded query once without additional retrieval–generation cycles.Grounding-only, non-interactivemethods
explicitly condition generation on pseudo-relevant documents or other corpus evidence in a single retrieve–
generate–requery pass, often using constraints, selection, or feedback calibration to control drift (e.g., MILL,
AGR, EAR, GenPRF, CSQE, MUGI, PromptPRF, FGQE).Grounding-aware interactivedesigns run multiple
retrieve–expand loops (e.g., InteR, ProQE, ThinkQE), using feedback from earlier retrieval stages to guide later
expansions.
(iii)Learning and alignment.Beyond zero/few-shot prompting, systemsaligngeneration to retrieval utility via
supervised fine-tuning, preference optimization, or distillation (e.g., SoftQE, RADCoT, ExpandR, AQE).
(iv)Knowledge Graph augmentation.KG-augmentedstrategies are categorized into two types: those using only
knowledge graph (KG) signals (e.g., KGQE and CL-KGQE) and those integrating KG with document-level graphs
(e.g., KAR and QSKG).
4.1 Point of injection
Modern QE methods can be roughly categorized into two levels. On one hand, the query can be implicitly enhanced in
the vector space, adjusting its representation to improve retrieval performance. On the other hand, QE can be explicitly
performed at the text level, by adding extended words or documents to directly enrich the query content. Clarifying this
distinction provides a clear foundation for understanding how and where expansion information can be injected into
the retrieval pipeline, which is the focus of this section.As shown in Figure 2, it illustrates the difference in injection
points between Implicit QE and Explicit QE.
4.1.1 Implicit Embedding-based QE.
Setting and notation.Let 𝑞be the input query and R𝑘={𝑑 1,...,𝑑𝑘}the top-𝑘texts returned by a first pass (when
available). A document 𝑑has a fixed embeddingd ∈R𝑚; a query is represented either by a single vectorq ∈R𝑚
(bi-encoder) or by a sequence of token embeddings {𝝓𝑞𝑖}|𝑞|
𝑖=1(late interaction). We write inner products as ⟨a,b⟩=a⊤b
and use sim(·,·) for cosine similarity when needed. Implicit (embedding-level) QEdoes notemit new terms; it strengthens
the query representation using signals distilled fromR 𝑘and then re-queries the unchanged index.
Why implicit QE helps.Compared to generative expansion, implicit QE avoids vocabulary drift and decoding latency.
It is especially effective when the first pass already surfaces some on-topic material, because the model can amplify
directions in embedding space that correlate with relevance and suppress distracting ones.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 11
Fig. 2. Application Scenarios of Query Expansion in Information Retrieval
Having outlined the basic principles and advantages of implicit QE, we now turn our attention to several representative
methods. These approaches illustrate how implicit signals—such as contextual information—can be effectively utilized
to enhance query expansion and improve retrieval performance.
ANCE-PRF: refining a single query vector with PRF.ANCE-PRF[ 124] augments a dual-encoder retriever without
touching the document index. A dedicated PRF encoder re-encodes the query together with feedback texts:
qPRF=BERT PRF [CLS]𝑞[SEP]𝑑 1[SEP]···[SEP]𝑑 𝑘[SEP].(12)
The second-stage score is then the usual inner product 𝑠(𝑞,𝑑)=⟨ qPRF,d⟩, wheredis the fixed document vector from
the original ANCE index. Training uses a standard softmax contrastive loss over one positive and sampled negatives.
Specifically, let 𝑑+represent the positive (relevant) document, and Ndenote the set of sampled negative (irrelevant)
documents (each𝑑−∈Nis an irrelevant document). The contrastive loss is calculated as:
L=−logexp⟨q PRF,d+⟩
exp⟨q PRF,d+⟩+Í
d−∈Nexp⟨q PRF,d−⟩.(13)
Intuition.The encoder learns to attend to feedback tokens that complement the query signal, nudgingqtoward regions
populated by relevant documents.Cost/risks.One extra encoder forward and one extra nearest-neighbor search at
inference; effectiveness depends onR 𝑘containing at least some relevant content.
ColBERT-PRF: injecting discriminative token directions.ColBERT-PRF[ 109] adapts pseudo-relevance feedback to late
interaction [ 41]. ColBERT scores by matching each query token to its best document token, 𝑠(𝑞,𝑑)=Í
𝑖max𝑗⟨𝝓𝑞𝑖,𝝓𝑑𝑗⟩.
ColBERT-PRF enriches the query side with a small set of feedback centroids distilled from R𝑘: (i) collect all token
embeddings fromR𝑘; (ii) cluster them into 𝐾centroids{𝝂ℓ}𝐾
ℓ=1(where 𝝂ℓdenotes the ℓ-th centroid vector); (iii) assign
each centroid a discriminativeness weight 𝜎ℓ=log𝑁+1
𝑁ℓ+1, where𝑁ℓis the number of passages containing the nearest
lexical token to 𝝂ℓ(an IDF proxy) and 𝑁is the total number of passages in the collection. The final score appends these

12 Li et al.
centroids as extra “query tokens” with weight𝛽:
𝑠PRF(𝑞,𝑑)=∑︁
𝑖max
𝑗⟨𝝓𝑞𝑖,𝝓𝑑𝑗⟩ +𝛽∑︁
(𝝂ℓ,𝜎ℓ)∈𝐹𝑒𝜎ℓmax
𝑗⟨𝝂ℓ,𝝓𝑑𝑗⟩,(14)
where𝐹𝑒contains the top-weighted centroids.Intuition.Feedback centroids capture discriminative semantic directions
seen inR𝑘without committing to specific surface forms, which reduces polysemy drift.Cost/risks.Clustering adds a
modest overhead (mitigable with approximate methods). If R𝑘is off-topic, centroids import noise; selecting small 𝐾
and using IDF-like weighting helps.
Eclipse: reweighting embedding dimensions with positive and negative signals.Eclipse[ 22] stays in the single-vector
regime and adjusts dimensions rather than tokens. From the first pass it takes a positive set 𝐷+(top results) and a
negative set𝐷−(tail results). For each embedding dimension𝑟∈{1,...,𝑚}it estimates an importance
imp(𝑟)=avg𝑑∈𝐷+𝑣𝑑,𝑟−avg𝑑∈𝐷−𝑣𝑑,𝑟
𝜎𝑟+𝜖,(15)
where𝑣𝑑,𝑟is the𝑟-th coordinate of document vectorv 𝑑,𝜎𝑟is the standard deviation of that coordinate across 𝐷+∪𝐷−,
and𝜖stabilises the ratio. The augmented query is a Hadamard (elementwise) reweighting,q aug=q⊙imp , and the
second pass uses⟨qaug,d⟩.Intuition.Dimensions that consistently separate positives from negatives are amplified;
dimensions activated by off-topic material are damped. This is a soft, data-driven denoiser in latent space.Cost/risks.
No extra encoding, only statistics over 𝐷+and𝐷−. Choosing𝐷−too aggressively can over-suppress useful but rare
signals; small negative pools and standard deviation normalization mitigate this.
QB-PRF: Query-Bag Pseudo-Relevance Feedback (embedding-level).QB-PRF[ 130] augments the query representation
using a small set of semantically equivalent query variants selected from an initial retrieval stage, and then fuses them
in embedding space. Let 𝑞be the input query and 𝑓(·)the query encoder used by the retriever. A first pass returns a
candidate poolC(𝑞)={𝑞′
1,...,𝑞′
𝑀}(e.g., related user queries or automatically mined paraphrases).
Selection.Candidates are embedded and filtered for semantic equivalence and lexical diversity to form a query bag
B(𝑞)⊂C(𝑞) . In practice, a contrastive encoder or variational autoencoder scores equivalence; we denote the retained
items as{𝑞′
𝑖}𝑛
𝑖=1.
Fusion.Letq =𝑓(𝑞) andq′
𝑖=𝑓(𝑞′
𝑖). QB-PRF produces an augmented representation by attentional pooling over the
bag:
𝑎𝑖=exp 𝜏−1q⊤q′
𝑖
Í𝑛
𝑗=1exp 𝜏−1q⊤q′
𝑗,q aug=(1−𝛽)q+𝛽𝑛∑︁
𝑖=1𝑎𝑖q′
𝑖,(16)
where𝜏>0is a temperature and 𝛽∈[ 0,1]controls the contribution of the bag. The second pass scores documents with
𝑠(𝑑|𝑞+)=q⊤
augd(for dual-encoders) or by replacing the query side in a late-interaction scorer (for multi-representation
models).
QB-PRF operates entirely in the latent space: it does not append tokens nor modify the index, and it complements
classical PRF by injecting paraphrastic evidence that is robust to lexical mismatch. Reported results show consistent
gains in both early precision and deep recall for conversational and ad hoc retrieval settings, with negligible online cost
beyond one additional query encoding.
LLM-VPRF: Vector Pseudo-Relevance Feedback for LLM-based retrievers.LLM-VPRF[ 56] extends pseudo-relevance
feedback to dense retrievers whose query encoders are derived from large language models, such as PromptReps,

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 13
RepLLaMA, and LLM2Vec. Given an initial query vectorq 0=𝑓(𝑞) , the first-pass retrieval returns the top- 𝑘documents
R𝑘(𝑞)={𝑑 1,...,𝑑𝑘}with embeddings{d𝑗}and scores{𝑠𝑗}. The updated query is obtained by combining the original
query and the feedback documents:
qvprf=(1−𝛼)q 0+𝛼Í𝑘
𝑗=1𝑤𝑗d𝑗
Í𝑘
𝑗=1𝑤𝑗,(17)
where𝛼∈[ 0,1]controls the contribution of feedback, and 𝑤𝑗can be uniform, score-based (e.g., softmax over {𝑠𝑗}), or
IDF-based.
The refined query vector is then used for a second retrieval pass on the same index, without text generation,
tokenization changes, or model fine-tuning. This approach incorporates corpus-specific semantics from the initial
results while remaining lightweight. Experiments on MS MARCO, BEIR, and LoTTE report consistent gains in nDCG@10
and Recall@1000 over zero-shot LLM retrievers, with larger improvements on queries with low first-pass recall, and it
can be seamlessly integrated into dual-encoder pipelines alongside fusion or re-ranking methods.
Takeaways.Implicit query expansion offers a lightweight way to strengthen retrieval signals without generating
new terms or modifying the index. The main difference across methods lies inwherethe feedback signal is injected:
ANCE-PRF re-encodes the whole query jointly with feedback documents, ColBERT-PRF adds token-like centroids to
enrich late interaction, Eclipse rescales embedding dimensions using positive and negative signals, QB-PRF pools over
paraphrastic query variants, and LLM-VPRF updates the query vector with document embeddings.
Compared to generative expansion, these methods avoid hallucination and reduce latency, making them suitable
when the first-pass retrieval already yields partially relevant results. Their effectiveness ultimately depends on the
quality of the initial feedback set, and they are best viewed as precision-preserving, representation-level refinements
that can be combined with rerankers or fused with other expansion strategies for robustness.
4.1.2 Selection-based Explicit QE.
Problem setup and notation.Let 𝑞denote the input query and R𝑘={𝐷 1,...,𝐷𝑘}the pseudo-relevant set returned by
a first pass (e.g., BM25 or a dense retriever). From R𝑘we derive a pool of candidate units 𝐶to consider for expansion. A
candidate can be a term 𝑡, an𝑛-gram, or a short chunk 𝑐. We write sim( a,b)for cosine similarity between vectorsaand
b, and⟨a,b⟩for an inner product. Selection-based explicit QE uses a pretrained language model (PLM) not to generate
text, but to score candidates in context and keep only those that are likely to help retrieval; the chosen items are then
inserted into the query or used as additional evidence at scoring time. This design gives tight control over vocabulary
and makes it easy to incorporate domain or user constraints.
The core idea of selection-based explicit QE with PLMs is to identify salient terms or chunks from the input query
and incorporate them into the reformulated query. Building on this principle, several representative approaches have
been proposed, which differ in how they select informative units and integrate them with the original query. In the
following, we present some of these representative methods in detail.
CEQE: Context-aware scoring in an RM3 pipeline.CEQE[ 74] replaces frequency counts in RM3 (a relevance model
variant that performs linear interpolation of expansion terms with the original query using the Query Likelihood score)
with context-sensitive similarities computed by BERT. First, retrieve R𝑘and encode the query to a vectorq. For each
document𝐷∈R𝑘and each occurrence of a term 𝑡in𝐷, obtain the contextual embeddingm𝐷
𝑡from an intermediate
BERT layer. Letm𝐷
∗denote the set of all contextual embeddings of all terms in 𝐷The contribution of 𝑡in𝐷is the

14 Li et al.
normalised similarity
𝑝(𝑡|𝑞,𝐷)=Í
m𝐷
𝑡sim(q,m𝐷
𝑡)
Í
m𝐷∗sim(q,m𝐷∗).(18)
Aggregate over feedback documents with an RM3-style weighting 𝑝(𝑞|𝐷) (the Query Likelihood score of 𝑞given𝐷)
to obtain a feedback model
𝑝(𝑡|𝜃R) ∝∑︁
𝐷∈R𝑘𝑝(𝑡|𝑞,𝐷)𝑝(𝑞|𝐷).(19)
Finally, interpolate the top- 𝑚terms with the original query using a weight 𝜆. Intuitively, a term that appears near
query-relevant contexts (even if rare) is promoted, while spurious high-frequency terms are down-weighted. CEQE is
fully unsupervised and drops into existing PRF pipelines with only additional BERT text encodes.
SQET: Supervised filtering of term candidates.SQET[ 75] turns term selection into a learned binary decision. Given 𝑞
and a candidate 𝑡(with or without a short context window), a cross-encoder processes the pair and outputs a relevance
probability 𝑠(𝑡|𝑞)∈[ 0,1]. When multiple mentions are available, instance-level scores {𝑠𝑖}are combined into a
term-level score using simple rules:
Max:max
𝑖𝑠𝑖,Weighted-sum:1
𝑍∑︁
𝑖tf(𝑡;ctx𝑖)𝑠𝑖,invRank:1
𝑍∑︁
𝑖𝑠𝑖
log2(rank(ctx 𝑖)+1).(20)
The top-ranked terms are interpolated with the original query exactly as in PRF. Because the model observes query–term
pairs during training, SQET better rejects topical but off-target terms than purely unsupervised scoring.
BERT-QE: Selecting chunks as expansion evidence for re-ranking.BERT-QE[ 133] selects short text chunks from R𝑘
and reuses them as evidence when re-ranking candidates. Pipeline: (1) obtain R𝑘and, if desired, re-rank it once with a
cross-encoder to improve the feedback pool; (2) split each 𝐷∈R𝑘into overlapping chunks {𝑐𝑖}and score rel(𝑞,𝑐𝑖)
with the cross-encoder; (3) keep the top-𝑘 𝑐chunks𝐶and, for each candidate document𝑑, compute an evidence score
rel(𝐶,𝑑)=∑︁
𝑐𝑖∈𝐶𝜋𝑖rel(𝑐𝑖,𝑑), 𝜋 𝑖=exp(rel(𝑞,𝑐 𝑖))Í
𝑐𝑗∈𝐶exp(rel(𝑞,𝑐 𝑗)).(21)
The final score interpolates the original and evidence components:
rel(𝑞,𝐶,𝑑)=(1−𝛼)rel(𝑞,𝑑)+𝛼rel(𝐶,𝑑).(22)
Rather than adding new tokens to 𝑞, BERT-QE carries forward small, query-vetted excerpts that the re-ranker can
match against candidate documents, yielding robust gains with a controllable compute budget.
CQED: Domain-aware term selection for biomedical search.CQED[ 39] targets scholarly biomedical search by com-
bining general BERT with UMLS-BERT and lightweight preprocessing. Named entities in 𝑞are first identified and
protected to avoid drift. A masking module creates masked variants of 𝑞to elicit semantically close substitutes. Each
mask is filled by both encoders to form a candidate pool 𝐶={𝑡} that mixes general and domain-specific vocabulary.
A pairwise learning-to-rank model then scores 𝑡∈𝐶 by the improvement it brings when appended to 𝑞; multi-head
attention fuses signals from the two encoders before a final scoring layer. The expanded query 𝑞+=𝑞∪Top-𝑘(𝐶) is
used for retrieval. By anchoring term proposals in biomedical knowledge while keeping a general encoder in the loop,
CQED reduces off-domain expansions and has reported strong gains on TREC-COVID.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 15
PQEWC: Personalizing term choices from a user’s history.PQEWC[ 10] personalizes explicit expansion using the user’s
past interactions. From clicked titles or prior reads for user 𝑢, the method builds an embedding pool 𝐸𝑢={w(𝑢)}with a
ColBERT encoder. Two selection strategies are considered. Cluster-based: cluster 𝐸𝑢(e.g., HDBSCAN), rank cluster
centroids by similarity to the current query vector, and pick the nearest token to each top centroid as expansion terms.
Fast approximate: directly retrieve the 𝑛nearest tokens to the query vector from 𝐸𝑢using an ANN index. Each selected
term𝑡𝑖receives a weight 𝑤𝑖proportional to its similarity to 𝑞, and the scoring interpolates original and personalized
terms:
Score(𝑞′,𝑑)=(1−𝛼)Score(𝑞,𝑑)+𝛼∑︁
𝑖𝑤𝑖Score(𝑡𝑖,𝑑).(23)
This design steers ambiguous or broad queries toward a user’s long-term interests at millisecond-level overhead.
Design notes.Selection-based explicit QE keeps the expansion vocabulary under control, is easy to constrain with
domain ontologies or user profiles, and plays well with both sparse and neural rankers. Unsupervised scoring (CEQE) is
simple and strong; supervised filtering (SQET) improves precision when labels exist; chunk selection (BERT-QE) lets
a re-ranker exploit richer contextual evidence; domain and personalization layers (CQED, PQEWC) target drift and
ambiguity in practice.
4.2 Grounding and interaction
Global notation.Throughout this subsection, unless otherwise noted: 𝑞denotes the original input query, and 𝑔a
generated expansion (e.g., a passage, rationale, pseudo-document, or clue). 𝐺(𝑞)={𝑔 1,...,𝑔𝑚}is the set of generated
expansions. Concatenation at the string level is [𝑞;𝑔], and an augmented query is denoted 𝑞+(for either textual or
vector-space form). 𝑅𝑘(𝑞)={𝑑 1,...,𝑑𝑘}represents the top- 𝑘pseudo-relevant documents from a retriever, scored by
𝑠(𝑞,𝑑).qanddare vector embeddings of𝑞and𝑑, andrank(𝑑|𝑅(·))gives the position of𝑑in the ranked list.
4.2.1 Zero-Grounding, Non-Interactive QE: LLM-Driven Single-Stage Expansion.
Scope and specific notation.This category comprises zero- or few-shot query expansion methods that rely solely
on prompting a large language model ( L) without supervised fine-tuning, additional retriever training, or systematic
grounding in the target corpus. Minor uses of corpus evidence in control variants (e.g., injecting a few top-ranked
passages into the prompt) are still considered in scope when the main expansion process remains prompt-only and
non-interactive. Beyond the global notation, we write vfor the mean embedding over multiple 𝑔∈𝐺(𝑞) , used in vector
aggregation, and𝑝(𝑤|𝜃 •)for a relevance model estimated from generated feedback.
Query2Doc: Few-shot pseudo-documents.Q2D[ 107] promptsLto write a short passage that would plausibly answer
𝑞using𝑘in-context exemplars (e.g., 𝑘=4). The pseudo-document 𝑔is then concatenated with 𝑞to obtain𝑞+=[𝑞;𝑔].
For sparse retrieval, repeating the original query tokens before concatenation (e.g., five repetitions) preserves term
salience; for dense models, 𝑞and𝑔are joined with a separator and encoded jointly or separately. Q2D reliably improves
BM25 on MS MARCO and TREC DL (e.g., up to +15.2%nDCG@10 on DL’20) and gives modest gains when used to
augment dense training. Trade-offs include single-shot hallucinations and non-trivial decoding latency.
CoT-QE: Reason-then-expand prompting.CoT-QE[ 35] elicits an explanatory chain before the answer with a zero-shot
prompt of the form: “Answer the following query: {q}. Give the rationale before answering.” The produced rationale
𝑔contains definitions, disambiguation cues, and facet hints; [𝑞;𝑔]is issued to the retriever. A PRF-enhanced variant

16 Li et al.
(CoT/PRF) injects the top-3 BM25 passages into the prompt to bias reasoning toward corpus-specific terminology.
CoT-QE outperforms classical PRF baselines (e.g., Bo1, KL) and other prompt baselines on MS MARCO/BEIR, especially
on top-heavy metrics (MRR@10, nDCG@10), and remains effective with smaller models.
GAR: Generation-Augmented Retrieval.GAR[ 69] trains a seq2seq model (BART-large) on open-domain QA pairs
to emit one of three query-specific strings: an answer, a declarative sentence containing the answer, or the title of a
relevant passage.
At inference, the generated string 𝑔is concatenated to the original query, yielding [𝑞;𝑔], which is sent to BM25 or a
dense retriever. Running the three generators produces three rankings that can be combined by simple rank fusion for
additional recall. GAR is attractive because it is plug-and-play and task-aligned (the model learns to verbalise what a
good answer looks like), but it pays generation latency and can drift if𝑔over-commits to a specific interpretation.
GRF: Generative Relevance Feedback without First-Pass Dependence.GRF[ 66] replaces𝑅𝑘(𝑞)with an LLM-produced
set𝐺(𝑞)and estimates a relevance model directly over𝐺(𝑞). Concretely, the RM3-style term distribution is
𝑝(𝑤|𝜃𝐺)∝∑︁
𝑔∈𝐺(𝑞)𝑝(𝑤|𝑔)𝑝(𝑞|𝑔),(24)
where𝑝(𝑤|𝑔) comes from term statistics of 𝑔and𝑝(𝑞|𝑔) is approximated by a language-model likelihood. GRF
prompts for corpus-matched genres (keywords, entities, pseudo-documents, news-style summaries), which aligns
expansion with collection discourse and yields sizable MAP and nDCG@10 gains on Robust04, CODEC, and TREC DL.
GRF for dense and learned-sparse, and fusion with PRF.A follow-up study applies GRF to dense and learned-
sparse retrievers by encoding each 𝑔∈𝐺(𝑞) and aggregating with the original query embedding via a Rocchio-style
update; for learned sparse, term weights from 𝐺(𝑞) are pruned before indexing [ 65]. Query-level analyses show
complementarity with classical PRF: GRF helps low-recall queries; PRF strengthens already-recalled facets. A weighted
reciprocal-rank fusion of GRF and PRF improves Recall@1000 in the majority of cases.
HyDE: Hypothetical document embeddings.HyDE[ 29] promptsLto draft one or more short, plausible passages
answering𝑞(e.g., “write a paragraph that answers the question”), then uses their embeddings as surrogates for the
missing context in zero-shot dense retrieval. Let 𝐺={𝑔 1,...,𝑔𝑁}be generated passages and 𝑓(·) the fixed dense
encoder. Form a pseudo-context vector v=1
𝑁Í𝑁
𝑖=1𝑓(𝑔𝑖)and update the query embedding by a convex combination
˜q=(1−𝛼)𝑓(𝑞)+𝛼 v, 𝛼∈[0,1].(25)
Retrieving with ˜qimproves zero-shot dense baselines on heterogeneous benchmarks by bridging the representation
gap when no relevance labels or in-domain training are available; the main risks are drift from fanciful generations and
added encoding cost for multiple𝑔 𝑖.
Exp4Fuse: Rank fusion for LLM-augmented sparse retrieval.Exp4Fuse[ 60] prompts an LLM in a zero-shot setting to
generate a single hypothetical document 𝑟𝑞. The original query 𝑞𝑜is repeated𝜆times for balance and concatenated
with𝑟𝑞:
𝑞𝑒=
𝑞𝑜,...,𝑞𝑜|     {z     }
𝜆times,𝑟𝑞
, 𝜆=5.(26)

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 17
Using the same sparse retriever (e.g., SPLADE++), two ranked lists are obtained: 𝐼𝑜𝑞from𝑞𝑜and𝐼𝑒𝑞from𝑞𝑒. A modified
reciprocal-rank fusion then scores each document𝑑as:
𝑠FR(𝑑)=
𝑤𝑖+𝑛
102∑︁
𝑖=11
𝑘+𝑟𝑖,(27)
where𝑘=60mitigates outliers, 𝑛∈{ 1,2}counts the number of list appearances, 𝑤𝑖=1weights routes equally, and 𝑟𝑖
is the rank of 𝑑in list𝑖(or∞if absent). This indirect QE boosts nDCG@10 and Recall@1000 for learned sparse models
on MS MARCO and BEIR, often achieving SOTA with low overhead; regressions on strong queries suggest selective
invocation via query-quality predictors.
Contextual clue sampling and fusion.Contextual clue sampling [ 59] targets breadth and control. A generator produces
many short, answer-oriented “clues” for 𝑞using stochastic decoding. Candidates are clustered (e.g., by edit distance),
keeping one representative per cluster to reduce redundancy. Each retained clue 𝑐𝑖forms an expanded query [𝑞;𝑐𝑖]and
yields a ranked list with score𝑠([𝑞;𝑐 𝑖],𝑑). Final ranking fuses these lists with generation-aware weights:
𝑠fused(𝑞,𝑑)=𝑀∑︁
𝑖=1𝑃gen(𝑐𝑖|𝑞)𝑠([𝑞;𝑐 𝑖],𝑑).(28)
This scheme encourages multiple reasoning paths while placing more trust in clues the model deems probable. It is
particularly effective for multi-hop or under-specified questions.
SEAL: Generating substrings as executable keys.SEAL[ 12] constrains generation to 𝑛-gram substrings that are
guaranteed to occur in the target corpus. A BART model is fine-tuned to produce such substrings, and decoding is
enforced with an FM-index (a compressed full-text substring index) so that every token sequence 𝑡is corpus-valid. Each
candidate substring 𝑡is scored by a rarity-adjusted language-model score that balances "query plausibility" (from the
BART model) and "corpus rarity" (from the FM-Index):
𝑠SEAL(𝑡|𝑞)=log𝑃 LM(𝑡|𝑞) −𝜆log𝑃 corpus(𝑡),(29)
where𝑃corpus(𝑡)is the prior probability of 𝑡in the target corpus, calculated as𝐹(𝑡,R)Í
𝑑∈R|𝑑|—here,Rrepresents the target
retrieval corpus, and 𝐹(𝑡,R) is the total number of occurrences of 𝑡inR;𝜆is an interpolation hyperparameter that
balances the two terms. This scoring formula biases toward informative (rarer) substrings that remain plausible under
𝑞. Documents containing high-scoring substrings are retrieved via FM-index lookups, and evidence from multiple
substrings is aggregated. SEAL unifies expansion and retrieval: generated outputs are directly executable keys, avoiding
hallucination and enabling GPU-free, low-latency search with strong accuracy on KILT-style benchmarks.
This family covers methods that condition expansion on retrieval evidence and/or run multiple retrieve–expand–requery
cycles. Compared with free-running prompt-only generation, these approaches explicitly close the loop with the collec-
tion, which improves grounding, controllability, and robustness.
4.2.2 Grounding-Only, Non-Interactive QE: Corpus-Evidence Anchored Single-Pass Expansion.
Scope and specific notation.This category coversretrieval-conditionedexpansion methods in which Lis explicitly
grounded in the target collection during a single retrieve–generate–requery pass. Grounding can involve: (i) conditioning
on pseudo-relevant passages 𝑅𝑘(𝑞); (ii) constraining generation to corpus-valid substrings or entities; or (iii) applying
selection/calibration based on first-pass retrieval scores. Compared with Zero-Grounding QE, these methods access

18 Li et al.
corpus evidence; unlike interactive QE, they do not loop over multiple passes. Beyond the global notation, we introduce
𝑃gen(𝑔|𝑞,𝑅𝑘(𝑞))as the generation probability given corpus context, and𝜆∈[0,1]as the interpolation parameter for
fusion between retrieval runs.
Practical notes.Retrieval-conditioned QE methods vary in how they incorporate collection evidence. Constraint-based
approaches such as SEAL ensure grounding by limiting outputs to corpus-valid substrings, eliminating hallucination
and reducing latency. Selection-based methods like EAR filter diverse generations through learned rerankers, while
feedback-calibrated methods such as MUGI and FGQE adjust expansion impact using retrieval signals or fairness
objectives. Hybrid schemes, e.g., GenPRF and PromptPRF, combine statistical PRF signals with LLM-generated semantics
to balance recall and precision. CSQE mixes grounded evidence from 𝑅𝑘(𝑞)with unguided world-knowledge generations
to increase robustness. Overall, effective designs couple generative expressiveness with reliable grounding—via hard
constraints, learned selection, feedback calibration, or explicit interpolation with traditional signals.
MILL: Mutual verification with corpus evidence.MILL[ 36] constructs two complementary contexts and keeps only
what they agree on. Stage 1 generates multiple explanations by a query–query–document style prompt (decompose 𝑞
into sub-queries and write short passages), and in parallel retrieves pseudo-relevant documents with BM25. Stage 2
performs mutual verification: generated texts are scored by their semantic consistency with retrieved documents (cosine
between encoder representations), and PRF documents are rescored by their agreement with the generated explanations.
Only top-agreement items from both sides are retained, concatenated to 𝑞, and reissued. Across TREC-DL’19/20 and
BEIR, MILL improves NDCG@1000, MRR@1000, and Recall@1000, with largest gains in specialized domains, indicating
that agreement filtering curbs hallucination while preserving useful diversity.
AGR: Analyze–Generate–Refine.AGR[ 18] structures prompting into three passes.Analyze: extract key phrases and
a brief need statement for 𝑞.Generate: produce several answer-oriented candidates and, for each, run a light retrieval to
fetch a small set of supporting passages; ask Lto enrich each candidate using that context.Refine: perform a self-review
over all enriched candidates to remove off-topic or redundant content and return a compact expansion 𝑔. The final[𝑞;𝑔]
improves zero-shot retrieval and end-to-end OpenQA EM across NQ, TriviaQA, WebQ, and CuratedTREC, showing that
staged quality control reduces error propagation without supervision.
EAR: Expand, rerank, and retrieve.EAR[ 19] decouples generation from selection. A generator (e.g., BART-large
or T0-3B) first proposes a diverse set of candidate expansions 𝐸={𝑒 1,...,𝑒𝑁}. A learned reranker then predicts the
utility of each candidate. Two variants are common: a retrieval-independent reranker that scores pairs (𝑞,𝑒) , and a
retrieval-dependent reranker that also conditions on a top passage (𝑝1), scoring triples(𝑞,𝑒,𝑝 1). With a pairwise loss,
the reranker learns to prefer expansions that raise the rank of the gold passage under BM25. The best candidate ˆ𝑒is
concatenated with the query and used for the final retrieval. By filtering the generator through a PLM-based selector,
EAR preserves diversity while curbing noisy𝑒 𝑖, delivering consistent gains over single-shot generators.
GenPRF: Generative pseudo-relevance feedback.GenPRF[ 108] marries classical RM3 with neural reformulation. From
the top-ranked texts of a first pass, short passages are selected (e.g., FirstP/TopP/MaxP) and fed to a seq2seq model (T5
or FLAN-T5) to produce a corpus-aware reformulation 𝑞𝑟. Rather than replacing RM3, GenPRF mixes the two query
models. Let 𝑝(𝑤|𝜃 RM3)be RM3’s term distribution and 𝑝(𝑤|𝜃 Gen)the term distribution estimated from 𝑞𝑟(e.g.,

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 19
normalised token weights). GenPRF interpolates
𝑝(𝑤|𝜃†)=𝑘 RM3𝑝(𝑤|𝜃 RM3)+𝑘 Gen𝑝(𝑤|𝜃 Gen), 𝑘 RM3+𝑘Gen=1,(30)
and uses𝑝(·|𝜃†)in the second pass. This hybrid retains RM3’s high-recall term statistics while injecting semantically
richer paraphrases from the generator, yielding consistent MAP and nDCG gains across ad hoc collections.
CSQE:Corpus-Steered query expansion.CSQE[ 50]balances LLM world knowledge with collection specificity by
combining two sources: (i) sentences extracted from 𝑅𝑘(𝑞)that an LLM deems query-relevant and (ii) a hypothetical
document generated without corpus input. Concatenating both with 𝑞improves robustness on TREC DL and BEIR and
excels on NovelEval, where the target knowledge is absent from LLM pretraining.
MUGI:Multi-text generation integration for query expansion.MUGI[ 128] prompts an LLM in a zero-shot setting
to generate multiple pseudo-reference documents 𝑅={𝑟 1,...,𝑟𝑛}(e.g.,𝑛=5), which are then integrated with the
original query 𝑞to enrich both sparse and dense retrieval without additional training. For sparse retrieval (BM25), the
pseudo-references are concatenated with repeated copies of 𝑞to balance their influence, given BM25’s sensitivity to
term frequency and document length. The repetition factor𝜆is adapted to reference length:
𝑞sparse=
𝑞,...,𝑞
|  {z  }
𝜆times, 𝑟1,...,𝑟𝑛
,(31)
where𝜆is a length-normalized integer (hyperparameters described in the appendix). For dense retrieval, pairwise
concatenations
𝑞,𝑟𝑖
are encoded independently and pooled:
𝑒𝑞=1
𝑛𝑛∑︁
𝑖=1𝑓 [𝑞,𝑟𝑖],(32)
yielding a query embedding enriched with multiple contextualized references. A light Rocchio-style calibration shifts
the query embedding toward high-agreement positives (generated references and top- 𝐾documents) and away from
low-ranked negatives, mitigating noisy generations. This unified approach improves nDCG@10 and Recall@1000 across
sparse and dense models on TREC DL and BEIR, enabling smaller LLMs to match or surpass larger models.
PromptPRF:a feature-based PRF framework.Rather than free-form text,PromptPRF[ 55]extracts structured features
from𝑅𝑘(𝑞)using an LLM—keywords, entities, focused summaries, and chain-of-thought-derived terms—and encodes
the result once as an augmented query representation for the second pass. Because features are grounded in PRDs and
can be produced offline, PromptPRF reduces online LLM cost while outperforming term-statistics PRF baselines.
FGQE:Fair Generative Query Expansion.FGQE[ 34]measures exposure disparities in the first-pass ranking using
Average Weighted Rank Fairness and then conditions the LLM to generate entities, keywords, or pseudo-documents
that target underexposed groups. The resulting expansions improve fairness with minimal loss in nDCG, illustrating
that feedback signals can optimize objectives beyond relevance.
4.2.3 Grounding-Aware Interactive QE: Multi-Round Retrieve-Expand Loops.
Scope and specific notation.This category coversiterativequery expansion methods that interleave retrieval and
generation over 𝑇> 1rounds, with at least some expansions grounded in documents retrieved in earlier stages. This
closed-loop setting enables progressive query refinement, going beyond the single-pass grounding of the previous

20 Li et al.
category. Building on the global notation, let the initial query be 𝑞(1)=𝑞. At iteration 𝑡(1≤𝑡≤𝑇 ), the model produces
an expansion𝑔(𝑡)(from prior knowledge or conditioned on𝑅 𝑘(𝑞(𝑡))) and forms:
𝑞(𝑡)
exp=[𝑞(𝑡);𝑔(𝑡)].(33)
Retrieved sets 𝑅𝑘(𝑞(𝑡))may be combined, filtered, or re-used in later iterations, and stopping criteria may depend on
retrieval budgets or observed gains.
InteR:a novel framework that facilitates information refinement through synergy between RMs and LLMs.InteR[ 26]
alternates expansion and retrieval. An LLM first produces expansion text to pull in an initial set; those passages are
then summarized into a knowledge collection 𝑆that is interleaved with 𝑞to requery. The method supports hybrid
sparse–dense stacks and shows strong results on large-scale and low-resource search.
ProQE:progressive query expansion.When retrieval calls are expensive,ProQE[ 92] adopts a progressive loop that
retrieves only one document per iteration, lets an LLM judge and extract terms, updates the query, and repeats until
a budget is met. A final chain-of-thought expansion is added before a single high-recall sweep, delivering higher
effectiveness than single-shot PRF or zero-shot LLM QE under tight budgets.
LameR:the Language language model as Retriever.LameR[ 99]performs a small first-pass retrieval, then prompts the
LLM to produce multiple likely answers conditioned on those few passages, and finally issues an answer-augmented
query for the second pass. This answer-first strategy improves recall without retriever fine-tuning and sets competitive
zero-shot baselines on DL’19/DL’20 and BEIR.
ThinkQE: Iterative thinking with corpus interaction.ThinkQE[ 51] alternates explicit reasoning and retrieval. At
round𝑡, the model lists interpretations and related concepts, then writes a short expansion 𝑔𝑡; the expanded query
[𝑞;𝑔1;...;𝑔𝑡]is issued, and only new top- 𝑘documents are fed back as context for the next round to encourage
coverage. After a small number of rounds, all expansions are concatenated and searched. On TREC DL and BRIGHT,
this retrieve–expand–filter loop increases both nDCG@10 and Recall@1000 versus single-shot prompting, suggesting
that iterative, diversity-seeking reasoning is an effective antidote to early bias.
4.2.4 Takeaways.Across the grounding–interaction spectrum, LLM-based query expansion trades between prior-
knowledge breadth and corpus-specific precision.Zero-groundingmethods (e.g., Q2D, CoT-QE, HyDE) are simple to
deploy and can lift recall without first-pass cost, but risk drift and hallucination without corrective signals.Grounding-
only, non-interactivedesigns (e.g., SEAL, MILL, EAR, GenPRF, PromptPRF) add a light first-pass to curb hallucination,
align terminology, and enable richer control via constraints, rerankers, or interpolation with statistical PRF—often
delivering stable gains for both sparse and dense retrievers.Grounding-aware interactiveloops (e.g., InteR, ProQE,
ThinkQE) further close the feedback loop, progressively refining expansions to recover missed facets, at higher latency
and engineering cost. Effective practice couples generative expressiveness with targeted grounding and, where feasible,
iterative refinement; parameter choices ( 𝑘,𝜆, iteration budget) and gating strategies determine the cost–quality trade-off
in production.
4.3 Learning and Alignment for QE with PLMs/LLMs
Scope and notation.This section covers methods that explicitlyalignlarge language models or retrievers to produce
query expansions that improve downstream retrieval. We use 𝑞for a query, 𝑔for an expansion text, 𝑑+for a gold

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 21
(relevant) passage, and 𝐷−for a set of negatives. Encoders map text to vectors: 𝑓(·) denotes the student retriever,
𝑓𝑇(·)a teacher retriever. When needed, Ldenotes an LLM used to generate 𝑔. For dense training, we write a standard
contrastive (InfoNCE) objective
Lcont(𝑞)=−logexp(sim(𝑓(𝑞),𝑓(𝑑+))/𝜏)
exp(sim(𝑓(𝑞),𝑓(𝑑+))/𝜏)+Í
𝑑−∈𝐷−exp(sim(𝑓(𝑞),𝑓(𝑑−))/𝜏),(34)
with temperature 𝜏and cosine similarity sim(·,·) . Supervised fine-tuning (SFT) minimizes negative log-likelihood of a
target expansion sequence 𝑔under a generator 𝜋𝜃:LSFT=−log𝜋𝜃(𝑔|𝑞) . Direct Preference Optimization (DPO) aligns
a generator𝜋 𝜃to prefer𝑔+over𝑔−for the same𝑞:
LDPO=−log𝜎
𝛽
log𝜋𝜃(𝑔+|𝑞)−log𝜋 𝜃(𝑔−|𝑞)−log𝜋 ref(𝑔+|𝑞)+log𝜋 ref(𝑔−|𝑞)
,(35)
with inverse temperature 𝛽and a frozen reference model 𝜋ref. Parameter-efficient fine-tuning (PEFT) such as LoRA can
be used in any generator or encoder below to reduce memory and time.
4.3.1 SoftQE: distilling LLM-generated expansions into query representations.SoftQE[ 87] converts the benefit of LLM
expansions into a cheaper retriever that no longer requires the LLM at inference. Offline, a large language model (e.g.,
GPT-3.5-turbo) generates a pseudo-document 𝑔for each𝑞(e.g., Query2Doc-style). Ateacherdual-encoder (denoted as
𝑓𝑇(·)) is trained on expanded inputs [𝑞;𝑔]with contrastive loss, producing a strong representation 𝑓𝑇([𝑞;𝑔]). Astudent
encoder (denoted as 𝑓(·)) then learns, from only the raw query 𝑞, to approximate the teacher’s expanded representation
via an MSE distillation term,
Ldist=𝑓(𝑞)−𝑓𝑇([𝑞;𝑔])2
2,(36)
combined with the usual contrastive objective on raw queries:
LSoftQE =𝛼L dist+(1−𝛼)L cont(𝑞).(37)
Intuitively, the student is taught to embed 𝑞as if the helpful expansion were already present. The result is an expansion-
aware retriever with the latency of a standard encoder. In practice, SoftQE yields sizable nDCG@10 gains on MS MARCO
and BEIR while avoiding on-the-fly decoding costs; adding cross-encoder distillation further sharpens decision bound-
aries.
4.3.2 RADCoT: retrieval-augmented distillation of chain-of-thought expansions.RADCoT[ 48] compresses a reasoning-
capable teacher (e.g., GPT-3) into a smaller retrieval-augmented student that can produce chain-of-thought (CoT) style
expansions efficiently. Stage one collects pairs (𝑞,𝑔CoT)by prompting the teacher to explain and answer 𝑞. Stage two
trains a Fusion-in-Decoder (FiD) student conditioned on retrieved contexts 𝑃={𝑝 1,...,𝑝𝐾}, minimizing the likelihood
loss
LRADCoT =−log𝑃 FiD 𝑔CoT|𝑞,𝑃.(38)
At inference, the student generates a short rationale 𝑔CoT
stugrounded in the passages and appends it to 𝑞. Because the
student relies on retrieval rather than memorized world knowledge, it achieves teacher-like expansion quality with
orders-of-magnitude fewer parameters; using PEFT (e.g., LoRA adapters) is sufficient for stable training.

22 Li et al.
4.3.3 ExpandR: alternating retriever training and preference-aligned generation.ExpandR[ 121] jointly optimizes a
retriever and an LLM so that expansions are both informative andusefulto the ranker. The procedure alternates two
steps.
Retriever step.Given 𝑞and an LLM expansion 𝑔, construct a query representation by averaging the embeddings of
the original query and the expanded query (gating is optional for weight adaptation):
˜𝑞=1
2(𝑓(𝑞)+𝑓([𝑞;𝑔])) (39)
where𝑓(·)denotes the query encoder, and ˜𝑞is the augmented query embedding. The retriever is then trained with
L𝑐𝑜𝑛𝑡(𝑞)to learn to exploit the additional semantic signal from𝑔.
Generator step.Freeze the retriever and update the LLM with DPO using pairs (𝑔+,𝑔−)scored by a reward that
combines (i) self-consistency with the LLM’s own answer conditioned on 𝑑+and (ii) retrieval utility, e.g., the rank of 𝑑+
under[𝑞;𝑔]. The DPO loss nudges the LLM toward expansions that lift the actual ranking objective without requiring a
separate learned reward model.
By repeating these steps, the retriever becomes expansion-aware while the generator becomes retriever-aware,
producing generalizable gains across MS MARCO and BEIR, especially on collections with large lexical–semantic gaps.
4.3.4 Aligned Query Expansion: SFT and DPO for retrieval-oriented generation.AQE[ 120] starts from a base LLM and
explicitly reshapes its output distribution toward retrieval-helpful expansions. For each 𝑞, the base model proposes a
pool of candidates{𝑔𝑖}𝑛
𝑖=1; each candidate is scored by how highly 𝑑+ranks when searching with [𝑞;𝑔𝑖]. AQE supports
two alignment regimes.
RSFT.Keep only top candidates as targets and run supervised fine-tuning with the token-level loss LSFT. This anchors
the model on high-utility expansions.
DPO.Form preference pairs (𝑔+,𝑔−)from the top and bottom of the pool and optimize LDPOagainst a frozen
reference. DPO is a light-weight alternative to RLHF that avoids training a separate reward model while still encoding
utility-aware preferences.
At inference, AQE generates a single short 𝑔via greedy decoding and concatenates it to 𝑞, eliminating gener-
ate–rank–filter loops. On MS MARCO and NQ, both RSFT and DPO variants improve nDCG@10 and Recall@1000;
combining RSFT (to stabilize behavior) followed by DPO (to sharpen preferences) is often strongest. PEFT substantially
reduces fine-tuning cost without hurting effectiveness.
Implementation notes and takeaways.(i) Alignment converts brittle prompt-only QE into a reliable component: the
generator learns what helps the ranker, and the retriever internalizes expansion effects to amortize decoding cost. (ii)
Distillation paths (SoftQE, RADCoT) are preferable when inference budget is tight or privacy rules preclude online
LLM calls. (iii) Preference optimization (AQE, ExpandR) directly ties generation quality to ranking gains and is robust
across domains; DPO is a pragmatic substitute for full RLHF. (iv) In all regimes, PEFT (LoRA/adapters) makes alignment
feasible on modest hardware, and caching expansions or training expansion-aware students keeps serving costs close
to vanilla retrieval.
4.4 KG-augmented Query Expansion
Scope and specific notation.We consider a knowledge graph 𝐺=(𝑉,𝐸,𝑅) with entity nodes 𝑉, relation types 𝑅, and
edges𝐸⊆𝑉×𝑅×𝑉 . Given a query 𝑞, letE𝑞denote entitys extracted from 𝑞(via LLM), EL(𝑞)⊆𝑉 denote entities linked
from𝑞(via NER+entity linking), and Nℎ(𝑣)theℎ-hop neighborhood of 𝑣∈𝑉 . We write text(𝑣) for a textual field of an

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 23
entity (label, aliases, or description), and use concat(·) to denote textual concatenation. Unless stated otherwise, the
final retrieval is performed with a sparse or dense retriever scoring𝑠(𝑑|𝑞+)on an expanded query𝑞+.
It should be noted that KG-augmented QE here is not independent of the "grounding" framework in Sec.4.2: Sec.4.2
defines "grounding" as "expansion signals relying on real-time evidence from the current retrieval corpus", while this
section takes external KGs as a new grounding source, forming a complementation of "corpus grounding (Sec.4.2) + KG
grounding (this section)". Corpus grounding ensures alignment with the target corpus, while KG grounding supplements
structured semantics missing in the corpus, jointly addressing complex retrieval needs.
After introducing the basic concepts of knowledge graphs, we note that the key factor in KG-augmented query
expansion lies in how the extracted entities are utilized. Existing approaches can be broadly divided into two categories:
(i) leverages knowledge graph retrieval to incorporate related entities and relations.(ii) links the extracted entities to
external corpora or text chunks to obtain additional expansion information.
4.4.1 Entity-based Expansion via Knowledge Graph Retrieval.
Scope and notation.In this category of methods, entities are first extracted from the query, and then their neighboring
nodes or relations in the knowledge graph are retrieved to enrich the query representation. The expanded query can
then be expressed as:
𝑞+=
𝑞;Ø
𝑒∈E𝑞𝑓(N(𝑣),R(𝑣))
,(40)
where𝑓(·)is a function that selects or weights expansion terms from neighboring entities and relations, and [𝑞;·]
denotes concatenation or fusion with the original query. In the following, we present specific approaches for Entity-based
Expansion.
Entity-aware text injection (KGQE).KGQE[ 85] injects compact KG facts directly into the query text to improve
disambiguation and coverage. First, entities are detected and linked,E 𝑞=EL(𝑞). Then short textual snippets for each
𝑒∈E𝑞(e.g., label, type, canonical alias) are selected and concatenated:
𝑞+=concat
𝑞;
text(𝑒)
𝑒∈E𝑞
.(41)
Optional guards include type filters (retain only domain-relevant types), length caps, and de-duplication. The enriched
query is issued to the retriever, or used to prompt an LLM when a downstream generator is present. A minimal prompt
template is:
{
QUERY: "When is the last episode of season 8 of The Walking Dead?",
INJECTION: {"head": "The Walking Dead", "type": "TV Series", "publisher": "AMC"},
ENRICHED QUERY: "When is the last episode of season 8 of The Walking Dead? [SEP]
The Walking Dead — TV Series — AMC"
}
This family keeps expansions corpus-grounded and controllable while reducing ambiguity of short or entity-heavy
queries. KGQE addresses entity ambiguity (e.g., disambiguating "Apple" as a company vs. fruit) by injecting type-specific
KG facts (e.g., "Apple Inc. - Tech Company - Cupertino")—a key advantage over traditional WordNet-based expansion,
which fails to resolve short-query polysemy. However, KGQE struggles with low-resource domains (e.g., niche medical

24 Li et al.
subfields) where KG coverage is sparse; in such cases, hybrid schemes (e.g., KGQE + BioBERT term selection [38]) are
required to maintain effectiveness.
Cross-lingual KG-aware expansion (CL-KGQE).CL-KGQE[ 90] targets cross-language mismatch by combining
translation, KG linking, and distributional neighbors. An input query in language ℓ𝑠is translated or linked cross-
lingually to entities in a target KG; expansions are assembled from three sources: (i) distributional neighbors of query
tokens (word embeddings), (ii) KG categories or types (e.g., DBpedia dct:subject ), and (iii) hypernym or hyponym
terms from lexical graphs. The final expansion pool is merged with the translated keywords and executed with a
standard ranker (e.g., BM25). This hybrid design improves robustness when parallel data are scarce and terminology
diverges across languages.
4.4.2 Entity-based Expansion via Hybrid KG and Document Graph.
Scope and notation.In this approach, each query entity 𝑒∈E𝑞is first linked to a set of relevant documents or
chunks containing 𝑒, denoted as 𝐷(𝑒) . A document-level graph is then constructed where nodes correspond to these
documents, and edges represent relationships such as citations, authorship, or co-occurrence of entities. Expansion
terms are obtained from the graph nodes through a selection function𝑔(·), and concatenated with the original query:
𝑞+=
𝑞;Ø
𝑒∈E𝑞𝑔(𝐷(𝑒))
,(42)
where𝑞′is the expanded query, [𝑞;·]denotes concatenation, and 𝑔(·)extracts the most relevant terms from the
document graph.
KG-guided generation and retrieval (KAR).KAR[ 117] leverages structural neighborhoods to guide expansion. After
linkingE𝑞, retrieve an ℎ-hop subgraphS𝑞=Ð
𝑒∈E𝑞Nℎ(𝑒)and align documents to nodes/edges. From S𝑞, construct
lightweight relational snippets (“document triples”):
𝑇𝑞=
(𝑑𝑖,𝑟𝑖,𝑗,𝑑𝑗)|(𝑣𝑖,𝑟𝑖,𝑗,𝑣𝑗)∈S𝑞, 𝑑𝑖↔𝑣𝑖, 𝑑𝑗↔𝑣𝑗	
.(43)
An LLM is then prompted with (𝑞,𝑇𝑞)to write a short, knowledge-grounded expansion 𝑔𝑞=LLM(𝑞,𝑇 𝑞), and the final
expanded query is 𝑞+=concat(𝑞 ;𝑔𝑞). Compared with pure text injection, KAR exploits relational context to surface
salient, query-specific facets, which is particularly effective on semi-structured corpora (e.g., author–paper–venue
graphs).
Query-specific KG construction (QSKG).QSKG[ 67] builds a small, task-specific graph on the fly. A first-pass retrieval
for𝑞yields𝑅𝑘(𝑞); entities in these documents are linked and added as nodes, with edges created via within-document
co-mentions or KG links, expanding iteratively to form 𝐺𝑞. Node salience is computed with a simple centrality or
TF–IDF style score over𝐺 𝑞; top entities and relations are verbalized into short expansions and concatenated with the
query:
𝑞+=concat
𝑞;
text(𝑣)
𝑣∈TopK(𝐺𝑞)
.(44)
QSKG adapts to collection-specific vocabulary and senses, and tends to improve recall on ambiguous or multi-facet
queries while limiting drift through graph-based salience.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 25
Table 2. Comparison of traditional vs. PLM/LLM-based query expansion.
Dimension Traditional QE PLM/LLM-based QE
Knowledge Source Static resources, co-occurrence statsParametric knowledge from pretraining,
adaptable via prompts/fine-tuning
Contextual Sensitivity Isolated terms, context-agnostic Context-aware, sequence-level semantics
Retrieval Dependence Relies on initial rankingReduced dependence on initial retrieval via
zero-/few-shot generation
Computational Cost Lightweight, efficient Resource-intensive, higher latency
Domain Adaptability Domain-specific ontologies/dictionaries Cross-domain generalization, flexible adaptation
Retrieval Effectiveness Recall–precision trade-off Simultaneous gains in recall and precision
Integration Complexity Easy to plug into IR pipelineRequires model serving (GPUs); optionally dense
indices or cross-encoders depending on pipeline
4.4.3 Design considerations.KG-augmented QE improves disambiguation and coverage but introduces choices about
scope and control. In practice: (i) keep injected text short (labels, types, and 1–2 discriminative facts); (ii) prefer entity
linking confidence and type filters to avoid drift; (iii) in KAR/QSKG, cap graph radius and use salience to select few
high-yield nodes; and (iv) for cross-lingual settings, favor KG categories and lexical graphs that transfer well across
languages. These strategies preserve efficiency and mitigate noise while delivering consistent gains on entity-centric
and long-tail queries.
5 Comparative Analysis: Traditional vs. PLM/LLM-Based Query Expansion
Traditional query expansion and modern PLM/LLM-driven approaches differ fundamentally in how they bridge the
gap between sparse user queries and semantically rich documents. Table 2 summarizes these differences across key
dimensions. Overall, traditional methods rely on static resources and lightweight heuristics, while PLM/LLM-based
methods leverage parametric knowledge and contextual generation, trading efficiency for adaptability and effectiveness.
5.1 Knowledge Sources
Traditional QE depends on predefined dictionaries, ontologies, or co-occurrence statistics, which are limited in coverage
and slow to update. In contrast, PLMs and LLMs internalize broad linguistic and factual knowledge through large-scale
pretraining, and can be further adapted via fine-tuning or prompting, offering greater flexibility in dynamic or specialized
domains.
5.2 Contextual Sensitivity
Classical methods treat terms in isolation, often failing to disambiguate polysemous words (e.g., “Apple” as fruit vs.
company). PLM/LLM methods encode full-sequence context, enabling accurate disambiguation and context-aware
expansions, which improves retrieval in domains with complex semantics (e.g., legal, biomedical).
5.3 Retrieval Dependence
Pseudo-relevance feedback exemplifies traditional reliance on initial retrieval quality; poor initial results lead to topic
drift. In contrast, LLMs can generate zero-shot expansions or pseudo-documents directly, reducing dependence on
initial rankings and enabling more robust performance in ambiguous or long-tail queries.

26 Li et al.
5.4 Computational Cost
Traditional QE methods are lightweight and easy to deploy (e.g., RM3 with BM25). PLM/LLM-based methods, however,
are resource-intensive due to large model inference and sometimes retraining, incurring higher latency and energy
costs. Recent work on distillation and quantization mitigates this gap but trade-offs remain.
5.5 Domain Adaptability
Traditional QE requires curated resources for each domain (e.g., UMLS for medicine), limiting portability. PLM/LLM
models, pretrained on diverse corpora, generalize better across domains and can be efficiently adapted with small
amounts of domain-specific data or tailored prompts.
5.6 Retrieval Effectiveness
Traditional QE often improves recall at the expense of precision, requiring careful tuning. PLM/LLM methods, with
contextual understanding and generative ability, tend to boost both recall and precision simultaneously (e.g., Query2Doc,
GRF), showing stronger and more consistent gains across benchmarks.
5.7 Integration Complexity
Classical QE is plug-and-play and easily integrated into IR systems. PLM/LLM-based methods require heavier infras-
tructure (e.g., dense indices, model serving, GPU support), leading to higher deployment and maintenance overhead,
though APIs and smaller models are reducing barriers.
6 Application Domains and Use Cases
This section reviews where query expansion matters in practice and how traditional methods and PLM/LLM-based
approaches are instantiated across domains. We emphasize the mismatch patterns that QE must bridge, what evidence
exists that PLM/LLM-based QE helps beyond classical techniques, and the remaining deployment constraints.
6.1 Web Search Engines
From lexical rewriting to semantic interpretation.Commercial web search has long relied on query rewriting and
synonym expansion to cope with short, diverse user queries. Early pipelines (e.g., spelling correction, query substitution)
increased recall but lacked robust intent understanding, leading to noisy expansions. Modern AI-powered search
integrates large neural language models (e.g., BERT, MUM) to implicitly interpret and expand queries in context [ 3,100],
improving recall–precision balance compared to purely lexical heuristics.
Behavioral signals at scale.Large query logs provide rich signals to mine reformulations and near-synonyms from
user behavior [ 21]. Beyond pairwise mining, industrial systems aggregate candidates from behavior, taxonomy, and
KBs, followed by ML-based filtering to retain expansions that improve precision. Research also leverages multi-session
clicks, dwell time, and query co-occurrence to select expansion terms aligned with user intent [21, 45].
Event-centric expansion with PLMs for time-sensitive intents.To better handle breaking-news scenarios,
Event-Centric Query Expansion (EQE) [ 131] extends a query with the most salient ongoing event via a four-stage
pipeline: event collection, event reformulation (leveraging pre-trained language models like BART and mT5 with
prompt-tuning and contrastive learning), semantic retrieval (using a fine-tuned RoBERTa-based dual-tower model with

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 27
E-commerce Search
Comfy running shoes 
for gymComfy running shoes 
for gym
Behavior -driven personalized expansionData source
Behavior Pattern Mining
Extract user shopping patternsBehavior Pattern Mining
Extract user shopping patterns
Attribute Normalization
Align colloquial terms with
catalog terminologyAttribute Normalization
Align colloquial terms with
catalog terminology
Generate user -preference -drivenPersonalized Expansion
expansionsGenerate user -preference -drivenPersonalized Expansion
expansionsBehavior Pattern Mining
Extract user shopping patterns
Attribute Normalization
Align colloquial terms with
catalog terminology
Generate user -preference -drivenPersonalized Expansion
expansionsBehavior Pattern Mining
Extract user shopping patterns
Attribute Normalization
Align colloquial terms with
catalog terminology
Generate user -preference -drivenPersonalized Expansion
expansionsData source
Behavior Pattern Mining
Extract user shopping patterns
Attribute Normalization
Align colloquial terms with
catalog terminology
Generate user -preference -drivenPersonalized Expansion
expansionscomfortable gym trainers 
with arch support
lightweight running footwear
Nike/Adias cushioned sneakersExpansion increases
Product selectionE-commerce Search
Comfy running shoes 
for gym
Behavior -driven personalized expansionData source
Behavior Pattern Mining
Extract user shopping patterns
Attribute Normalization
Align colloquial terms with
catalog terminology
Generate user -preference -drivenPersonalized Expansion
expansionscomfortable gym trainers 
with arch support
lightweight running footwear
Nike/Adias cushioned sneakersExpansion increases
Product selectionCross -Lingual & Multilingual Search
climate change effectsclimate change effectsMultilingual Encoder(PLM/LLM)
Zero -shot transfer
mulitingual representations
Translation Branch
supports low -resource via pivotingEn→ES/FR/… via MT
LLM Expansion Branch
Generate -the-Translate
Backtranslation loop
Monolingual QE per Language
Refine terms within
each language spaceMultilingual Encoder(PLM/LLM)
Zero -shot transfer
mulitingual representations
Translation Branch
supports low -resource via pivotingEn→ES/FR/… via MT
LLM Expansion Branch
Generate -the-Translate
Backtranslation loop
Monolingual QE per Language
Refine terms within
each language spaceMultilingual Encoder(PLM/LLM)
Zero -shot transfer
mulitingual representations
Translation Branch
supports low -resource via pivotingEn→ES/FR/… via MT
LLM Expansion Branch
Generate -the-Translate
Backtranslation loop
Monolingual QE per Language
Refine terms within
each language spaceEN
global
warming
consequencesEN
global
warming
consequencesFR
changement
climatiqueFR
changement
climatiqueLLM -Enhanced CLIR
Generate -then-Translate Generate -then-TranslateData Sources & Flow Cross -Lingual & Multilingual Search
climate change effectsMultilingual Encoder(PLM/LLM)
Zero -shot transfer
mulitingual representations
Translation Branch
supports low -resource via pivotingEn→ES/FR/… via MT
LLM Expansion Branch
Generate -the-Translate
Backtranslation loop
Monolingual QE per Language
Refine terms within
each language spaceEN
global
warming
consequencesFR
changement
climatiqueLLM -Enhanced CLIR
Generate -then-TranslateData Sources & FlowBiomedical Information Retrieval
myocardial infarction signs acute coronary syndrome angina pectoris indicatorsMedical KGMedical KG
Domain PLMsDomain PLMsOntologiesOntologies Medical KG
Domain PLMsOntologies Medical KG
Domain PLMsOntologies
general paingeneral painHeart attack symptoms
myocardial 
infarctionHeart attack symptoms
myocardial 
infarctionData source
Expansion TermsBiomedical Information Retrieval
myocardial infarction signs acute coronary syndrome angina pectoris indicatorsMedical KG
Domain PLMsOntologies
general painHeart attack symptoms
myocardial 
infarctionData source
Expansion Terms
Apple Inc. product launch event Iphone release dateWeb Search Engines
apple launch.apple launch.
Expansion TermsQuery log User 
Signals
Web 
Pagesapple Inc.(ios)
ordinary apple(fruit)Data source
PLM
LLMsApple Inc. product launch event Iphone release dateWeb Search Engines
apple launch.
Expansion TermsQuery log User 
Signals
Web 
Pagesapple Inc.(ios)
ordinary apple(fruit)Data source
PLM
LLMs
Open -Domain Question Answering
2018 FIFA World Cup winner France
FIFA tournament results
final match Croatia 4 -22018 FIFA World Cup winner France
FIFA tournament results
final match Croatia 4 -2Precise Answer Extraction Who won the World Cup in 2018?
Problem: Low Recall from Underspecified Query
Supervised
OptimizationContext 
GenerationContext 
GenerationSemantic
MatchingSemantic
Matching
Supervised
OptimizationContext 
GenerationSemantic
Matching
Supervised
OptimizationContext 
GenerationSemantic
Matching
Higher
Recallaccurate
output
AnswerLow
RecallLow
RecallSemantic
GapSemantic
GapData Sources Open -Domain Question Answering
2018 FIFA World Cup winner France
FIFA tournament results
final match Croatia 4 -2Precise Answer Extraction Who won the World Cup in 2018?
Problem: Low Recall from Underspecified Query
Supervised
OptimizationContext 
GenerationSemantic
Matching
Higher
Recallaccurate
output
AnswerLow
RecallSemantic
GapData Sources Conversational Search
Turn 
SelectorQuery 
ResolverQuery 
ExpanderRetriever Multi -View
RerankerData Sources & Flow
Expanded Reply
Code Search
How to parse JSON stream 
in Python without timeout?
   Python JSON stream parser example
   Async JSON parser without blocking
   Python rapidjson streaming   Python JSON stream parser example
   Async JSON parser without blocking
   Python rapidjson streamingAPI Document GitHub Issues
Query
ExpanderMulti -Index
RetrievalReranker
Code Symbol DocExtended queryCode Search
How to parse JSON stream 
in Python without timeout?
   Python JSON stream parser example
   Async JSON parser without blocking
   Python rapidjson streamingAPI Document GitHub Issues
Query
ExpanderMulti -Index
RetrievalReranker
Code Symbol DocExtended queryRetrieval -Augmented Generation
What is the latest treatment for diabetes
New therapies in diabetes management
Updated clinical guidelines for diabetes
Recent advances in diabetes treatmentPubMed Knowledge
GraphDiabetes Insulin
NEJM 2024
Lancet 2023
PubMed Review
Metformin | Glucose Control
GLP -1 agonists
Cardiovascular protectionExtended queryRetrieval -Augmented Generation
What is the latest treatment for diabetes
New therapies in diabetes management
Updated clinical guidelines for diabetes
Recent advances in diabetes treatmentPubMed Knowledge
GraphDiabetes Insulin
NEJM 2024
Lancet 2023
PubMed Review
Metformin | Glucose Control
GLP -1 agonists
Cardiovascular protectionExtended queryBest laptops 
for programming?Best laptops 
for programming?
What about for 
gaming as well?What about for 
gaming as well?What about for 
gaming as well?
And under $1000?And under $1000?And under $1000?Historical records
Best laptops 
for programming?
What about for 
gaming as well?
And under $1000?Historical records
Budget gaming laptops
with good performanceBudget gaming laptops
with good performance
Acer Nitro 5Acer Nitro 5Dell XPS 15Dell XPS 15Dell XPS 15
Asus ROG ZephyrusAsus ROG ZephyrusAsus ROG Zephyrus
Affordable laptops 
for programming and 
gaming under $1000Affordable laptops 
for programming and 
gaming under $1000
Fig. 3. Application Scenarios of Query Expansion in Information Retrieval
two-stage contrastive training), and lightweight online ranking. This system has been deployed in Tencent QQ Browser
Search to serve hundreds of millions of users.

28 Li et al.
6.2 Biomedical Information Retrieval
Specialized vocabulary and ontologies.Biomedical IR faces severe lay–technical vocabulary gaps (“heart attack” vs.
“myocardial infarction”). Ontology-based QE (UMLS, MeSH) maps free text to controlled concepts and spelling variants
(e.g., MetaMap + ATM), improving classic metrics and reducing ambiguity. Aronson and Rindflesch [6]report up to
+14.1% 11-pt AP on the Hersh collection versus non-concept baselines, illustrating the benefit of structured medical
knowledge.
Domain PLMs.Domain-adapted PLMs (e.g., BioBERT, PubMedBERT) recommend contextual terms and capture
fast-evolving terminology. Kelly [38] show BioBERT-based embedding QE for rare diseases substantially improves
Precision@10 (0.42 →0.68) and nDCG (0.48 →0.72) over BM25/MeSH, with strong gains on acronym queries. For
systematic reviews, Khader et al . [40] combine BioBERT signals with MeSH features in a learning-to-rank scheme,
further boosting P@10/MAP/Recall; hybridizing semantic and ontology signals is especially effective.
Interactive LLM workflows.Ateia and Kruschwitz [7]design an interactive biomedical RAG agent that prompts
an LLM to emit an editable expanded query (Elasticsearch DSL), enabling experts to inspect and prune LLM-generated
terms before BM25 retrieval. This strikes a balance between semantic priors and expert control.
Precision and hallucination control.High-stakes settings require drift control: UMLS term re-weighting with
self-information can mitigate topic drift [ 24]. For LLMs, Niu et al . [80] propose self-refinement guided by predictive
uncertainty to detect risky entities and selectively verify against medical KGs, improving factuality with limited
overhead.
6.3 E-commerce Search
Catalog mismatch and normalization.User queries are short and colloquial, while product catalogs are verbose and
taxonomy-driven; QE aligns lay intent with normalized attributes [ 30]. Synonym mining from logs plus ML filtering
(e.g., eBay) improves recall and engagement in production [68].
Generative and taxonomy-aware enrichment.Beyond synonymy, mQE-CGAN [ 15] leverages GANs to synthesize
expansions conditioned on lexical/semantic signals, outperforming strong baselines (e.g., BART/CTRL) on semantic
similarity and coverage. ToTER [ 37] uses topical taxonomies for silver labeling, topic distillation, and query enrichment,
improving Recall/NDCG/MAP over SPLADE++, ColBERT, PRF, and recent generative augmentation across Amazon
ESCI.
LLM rewrites at scale.LLM-based personalization and on-the-fly rewrites normalize attributes and surface rare
variants [ 106]. Incorporating intent signals (name/category prediction) with auxiliary losses improves long-tail rewriting
in practice [ 129]. Taobao’s BEQUE [ 84] aligns rewrites to retrieval utility via offline feedback and Bradley–Terry-style
objectives, yielding measurable GMV/CTR gains in a 14-day A/B test, with pronounced lifts on tail and few-recall
queries.
6.4 Cross-Lingual & Multilingual Search
Traditional CLIR pipelines translate queries via dictionaries/MT and then apply monolingual QE; low-resource coverage
and ambiguity remain pain points. In contrast, multilingual PLMs/LLMs enable zero-shot transfer and support generate-
then-translate or backtranslation strategies. For example, Rajaei et al . [91] reformulate queries via multi-language
backtranslation and fuse ranked lists with RRF, improving MAP on classic testbeds (e.g., Robust04, GOV2). In practice,
mixing MT with LLM-driven expansions can raise cross-language recall while controlling drift via fusion and filtering.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 29
6.5 Open-Domain Question Answering
Answer-oriented expansions.Generation-Augmented Retrieval (GAR) [ 69] enriches queries with generated contexts
and matches BM25-level sparse retrieval to DPR, showing that properly scoped pseudo-text can rival dense pipelines.
EAR [ 19] further couples expansion with reranking. Recent AQE [ 120] aligns expansions with downstream passage
utility.
Zero-shot and process supervision.LLMs can yield high-quality expansions without task-specific fine-tuning
(e.g., MILL [ 36]). Analyze–Generate–Refine (AGR) [ 18] decomposes needs and quality-controls the expansions. Hybrid
text generation with PRF (HTGQE) [ 134] combines multiple LLM-generated contexts (answer/sentence/title) to improve
EM on NQ/Trivia.
Topic-aware ICL and vocabulary projection.TDPR [ 57] clusters queries to select ICL demonstrations, generates
pseudo-passages, and projects embeddings onto vocabulary to produce interpretable keywords, improving R@20 and
end-to-end accuracy across NQ/Trivia/WebQ/CuratedTREC.
6.6 Retrieval-Augmented Generation (RAG)
Why QE helps RAG.RAG quality hinges on retrieving the right passages; vague queries increase hallucination
risk [79, 132]. Refining/expanding the query boosts retrieval coverage and answer grounding [35, 127].
Representative strategies.QE-RAGpro [ 31] uses domain-tuned embeddings to fetch Q&A exemplars for prompting
LLMs to synthesize pseudo-documents, improving top- 𝑘accuracy on finance datasets. QOQA [ 42] generates multiple
LLM rewrites, selects them by query–document alignment (hybrid scores), and lifts nDCG@10 on SciFact/TREC-
COVID/FiQA. KG-Infused RAG [ 116] performs KG-guided spreading activation, infuses KG facts into LLM expansions,
and improves across multi-hop QA benchmarks. Backtranslation-based multi-view queries fused by RRF [ 91] yield
consistent gains in unsupervised settings.
Practical note.Selective invocation, caching, and small/distilled models reduce latency and cost; filtering (e.g.,
document-aligned scoring) limits drift from generative variability.
6.7 Conversational Search
Multi-turn settings require resolving coreference, filling ellipses, and adapting to evolving intents; traditional QE
struggles to selectively incorporate history. Learning to select useful turns (PRL) [ 73] forms expanded queries that
improve MRR/NDCG and handle topic switches better. MISE [ 44] combines Conversational Term Selection with
Multi-View Reranking to fuse dialog-, passage-, and paraphrase-based views, outperforming strong neural baselines.
QuReTeC [ 105] formulates term selection as lightweight classification over context and boosts MAP/NDCG@3 on
CAsT.
6.8 Code Search
From thesauri to generation.Early work enriches queries using WordNet, API names, and crowd knowledge to reduce
NL–code mismatch [52, 63, 78, 98, 125]. Neural methods predict keywords and control over-expansion [32, 33, 58].
PLM-based expansion.GACR [ 54] augments documentation queries with generated code snippets, substantially
improving CodeSearchNet across languages. SSQR [ 70] treats reformulation as self-supervised masked completion
(T5), improving MRR on CodeBERT/Lucene without parallel data. SG-BERT/GPT2 [ 61] trains on large query corpora to

30 Li et al.
Table 3. Representative Methods for Conversational Query Expansion
Method Granularity Model Notable strengths
PRL [73] Query BERT/ANCETurn selection; topic-switch robustness;
higher MRR/NDCG
MISE (CTS + MVR)[44] Term / Multi-view BERT/GPT-2CTS + MVR; multi-view (dialog, passage, paraphrase);
outperforms strong neural baselines
QuReTeC[105] Term BERTLightweight term selection over context;
higher MAP/NDCG@3
generate expansions that lift top- 𝑘accuracy. Hybrid embedding+sequence models (ECSQE) further enhance ranking [ 14].
Transformers and pretraining provide stronger semantics for bridging NL queries and code artifacts [102].
6.9 Lessons Learned Across Applications
(i) Traditional QE remains valuable for efficiency, transparency, and as a robust baseline (especially when latency budgets
are tight or supervision is scarce). (ii) PLM/LLM-based QE offers consistent gains in contextual disambiguation, cross-
domain/ cross-lingual adaptation, and recall–precision balance; interactive/inspectable workflows mitigate reliability
concerns in high-stakes domains. (iii) Cost, controllability, and factuality remain the main deployment barriers. Selective
invocation, fusion-based drift control, ontology/KB integration, and small/distilled models are practical levers. (iv)
These observations align with our comparative analysis (Section 5) and motivate the challenges and research avenues
discussed next (e.g., hallucination mitigation, feedback-grounded QE, efficiency–effectiveness trade-offs).
7 Future Directions and Open Challenges
Although PLM/LLM-driven QE has delivered strong gains across web, biomedical, e-commerce, ODQA, RAG, conversa-
tional and code search settings, our synthesis also reveals recurring failure modes and systems bottlenecks. We group
forward-looking directions by the concrete issues they address, and anchor each to empirical evidence summarized in
Sections 2–6.
7.1 Reliability on unfamiliar and ambiguous queries
Empirical basis.When domain knowledge is sparse or the query is highly polysemous, LLM expansions may hallucinate
entities or collapse onto a single popular sense, reducing facet coverage and recall [ 2]. HyDE/Q2D-style expansions can
be fluent yet off-topic in low-overlap domains (see Section 4 and the RAG discussion).
Opportunities.(i) Retrieval-augmented prompting that injects corpus-specific evidence pre- or in-generation [ 50,66].
(ii) Facet-aware diversification via multi-intent expansion, multi-query issuing and rank fusion, with ambiguity detectors
that trigger sense-splitting pipelines; complemented by expansion-level quality metrics (e.g., Expansion Gain@k,
Factuality Score) to detect and quantify hallucinations in unfamiliar or polysemous query settings. (iii) Difficulty-aware
routing that classifies queries as "familiar" (use prompt-only QE) or "unfamiliar" (use feedback-grounded QE) via
query-quality predictors [17].

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 31
7.2 Knowledge leakage and generalization
Empirical basis.On zero-shot verification and ODQA, gains sometimes coincide with expansions that paraphrase
gold passages, indicating leakage from LLM pretraining [123].
Opportunities.(i) Leakage-controlled benchmarks with temporal splits and held-out entities, plus reporting of
evidence originality. (ii) Detectors that measure n-gram or embedding overlap against evaluation corpora and pretraining
proxies. (iii) Dynamic grounding that biases generation toward fresh external sources or KGs rather than static parametric
memory.
7.3 Controlling expansion quality (safety, drift, usefulness)
Empirical basis.Even strong QE can degrade precision through subtle drift; automated filtering remains challeng-
ing [113].
Opportunities.(i) Pre-use vetting with lightweight judges for relevance, faithfulness and verifiability, rejecting
expansions before retrieval. (ii) Counterfactual checks such as drop-term and swap-sense ablations to ensure monotonic
utility of each expansion unit. (iii) Human-in-the-loop editing in high-stakes domains, in the spirit of editable DSL
queries in biomedical search.
7.4 Efficiency, scalability, and selective invocation
Empirical basis.Prompt-only QE (e.g., Q2D) often adds seconds-level latency per query; dense+LLM stacks are costly.
Distillation shows large savings with limited loss [48, 87].
Opportunities.(i) Distillation and PEFT: compress LLM-QE into student encoders (representation-level SoftQE;
reasoning-level RADCoT) for online budgets. (ii) Selective invocation: classifiers route only ambiguous/short/tail queries
to expensive QE [ 17]. (iii) Caching and reuse: query-semantic clustering with shared expansions, TTLs by domain, and
offline precomputation for tail traffic.
7.5 Dynamic domains and continual adaptation
Empirical basis.Static PLMs/LLMs lag emerging terms and events; classical PRF is corpus-specific but brittle without
guards.
Opportunities.(i) Continual retrieval-augmented QE that blends LLM priors with on-the-fly corpus/KG snippets [ 50,
66]. (ii) Low-shot domain transfer with instruction packs and adapter-based PEFT to learn domain-specific expansion
styles. (iii) Drift detectors that monitor expansion–click divergence and term novelty, auto-switching to conservative
selection methods such as CEQE/SQET when risk is high [74, 75].
7.6 Evaluation beyond end-to-end IR metrics
Empirical basis.MAP/nDCG reflect end impact but hide why QE helped or hurt.
Opportunities.(i) Expansion-level diagnostics: Expansion Gain@ 𝑘(fraction of new relevant docs attributable to
QE), Precision of Added Terms (share of QE-induced clicks that are relevant), and Facet Coverage (number of gold
facets touched by multi-query QE). (ii) User-centric measures: satisfaction on ambiguous tasks and edit-acceptance
rates in interactive QE. (iii) RAG-specific groundedness/attribution under QE-augmented retrieval.

32 Li et al.
7.7 Fairness, robustness, and governance
Empirical basis.Exposure can skew toward head facets/providers; fairness-aware QE is promising but early-stage [ 34].
Opportunities.(i) Fairness-aware generation optimizing exposure parity under minimal nDCG loss (e.g., AWRF-
style). (ii) Robustness audits across tail intents, languages and domain shifts. (iii) Privacy and transparency: disclose
expansion edits; prefer on-device or anonymized QE when logs are sensitive.
7.8 Integration patterns with neural IR and RAG
Empirical basis.QE helps at different pipeline points: first-stage recall (GAR/EAR/AGR), multi-query fusion, re-ranking
(BERT-QE), and RAG query optimization (QOQA, KG-infused RAG) [18, 19, 42, 69, 116].
Opportunities.(i) Design guides that choose explicit (term/text), implicit (embedding) or hybrid QE by corpus
lexicality and compute budget. (ii) Constrained generation using corpus-valid substrings as actionable keys for latency-
critical search [ 12]. (iii) Multimodal and cross-lingual QE leveraging multilingual/domain encoders and back-translation
fusion.
7.9 From research artifacts to reproducible systems
Empirical basis.Results are sensitive to prompts, decoding, PRF pools, and retrieval backends; leakage and cost
reporting are often omitted.
Checklist.Release prompts/in-context examples/decoding parameters; report compute, latency and cost per 1k
queries for online/offline modes; include leakage controls and temporal splits; ablate selection vs. generation vs. hybrid;
ship reproducible GRF/PRF/QE fusion scripts.
Takeaway.Near term, the frontier is reliable, economical, and controllable QE: diversification-aware generation
grounded in corpus/KG evidence, selective invocation with strong filters, and distilled/PEFT students for production.
Longer term, QE will co-evolve with neural IR and RAG through constrained, leakage-aware expansion for recall,
interpretable and fairness-aware exposure, and diagnostics that make expansion effects explicit.
8 Conclusion
This survey traced query expansion from classical corpus- and lexicon-driven methods to contemporary PLM/LLM
paradigms. We organized the space along four analytical dimensions: point of injection, which distinguishes explicit
versus implicit expansion within retrieval or generation pipelines; grounding and interaction, which captures query
anchoring to knowledge bases, multi-turn retrieval, and model-internal capabilities; learning alignment, which addresses
how models acquire and optimize expansion strategies; and knowledge graph–based argumentation, which leverages
structured knowledge and multi-hop reasoning to support interpretable expansion, pipeline position, and application
arenas. Across domains, empirical findings show that effective query expansion extends beyond pre-retrieval reformu-
lation to multiple stages of IR pipelines, including first-stage retrieval, re-ranking, and retrieval-augmented generation.
The four dimensions provide complementary levers: the injection point guides impact, grounding, and interaction
ensure relevance, learning alignment supports domain/task adaptation, and knowledge graph–based argumentation
enables interpretable, structured reasoning. Together, they offer a principled framework for designing and combining QE
techniques under practical constraints, while pointing to future challenges in quality control, cost-aware deployment,
adaptation, and fairness-aware evaluation.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 33
We also distilled practical guidance: choose explicit vs. implicit vs. hybrid QE by corpus lexicality and latency
budget; prefer feedback-grounded or constrained generation in safety-critical settings; and use selective invocation plus
distillation to control cost. Finally, we outlined open challenges with concrete, evidence-backed directions, including
reliability under ambiguity, leakage-aware evaluation, efficiency at scale, continual adaptation, fairness and governance,
and reproducible system building.
Overall, the path forward is a synthesis of old and new: combine human-curated knowledge (ontologies, user
behavior) with machine intelligence (instruction-following, reasoning, alignment) to deliver robust, transparent, and
efficient expansion that consistently improves retrieval and downstream generation.
References
[1]Ahmed Abdelali, Jim Cowie, and Hamdy S Soliman. 2007. Improving query precision using semantic expansion. Information processing &
management 43, 3 (2007), 705–716.
[2] Kenya Abe, Kunihiro Takeoka, Makoto P Kato, and Masafumi Oyamada. 2025. LLM-based Query Expansion Fails for Unfamiliar and Ambiguous
Queries. In Proceedings ofthe48th International ACM SIGIR Conference onResearch andDevelopment inInformation Retrieval. 3035–3039.
[3]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam
Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).
[4] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey,
Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023).
[5]Alan R Aronson and Thomas C Rindflesch. 1997. Query expansion using the UMLS Metathesaurus. In Proceedings oftheAMIA annual fall
symposium. 485.
[6] Ph.D Alan R. Aronson and Ph.D Thomas C. Rindflesch. 1997. Query Expansion Using the UMLS®Metathesaurus®. https://api.semanticscholar.
org/CorpusID:8713491
[7] Samy Ateia and Udo Kruschwitz. 2025. BioRAGent: A Retrieval-Augmented Generation System for Showcasing Generative Query Expansion and
Domain-Specific Search for Scientific Q&A. In European Conference onInformation Retrieval. Springer, 1–5.
[8] Hiteshwar Kumar Azad and Akshay Deepak. 2019. Query expansion techniques for information retrieval: a survey. Information Processing &
Management 56, 5 (2019), 1698–1735.
[9] Ricardo Baeza-Yates, Carlos Hurtado, and Marcelo Mendoza. 2004. Query recommendation using query logs in search engines. In International
conference onextending database technology. Springer, 588–596.
[10] Elias Bassani, Nicola Tonellotto, and Gabriella Pasi. 2023. Personalized query expansion with contextual word embeddings. ACM Transactions on
Information Systems 42, 2 (2023), 1–35.
[11] Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBERT: A Pretrained Language Model for Scientific Text. In Conference onEmpirical Methods in
Natural Language Processing. https://api.semanticscholar.org/CorpusID:202558505
[12] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Scott Yih, Sebastian Riedel, and Fabio Petroni. 2022. Autoregressive search engines:
Generating substrings as document identifiers. Advances inNeural Information Processing Systems 35 (2022), 31668–31683.
[13] Jagdev Bhogal, Andrew MacFarlane, and Peter Smith. 2007. A review of ontology based query expansion. Information processing &management
43, 4 (2007), 866–886.
[14] Nazia Bibi, Muhammad Usman Tariq, Zabeeh Ullah, Muhammad Babar, and Zahid Khan. 2025. Enhancing Code Search through Query Expansion:
A Fusion of LSTM with GloVe and BERT Model (ECSQE). Results inEngineering (2025), 105979.
[15] Altan Cakir and Mert Gurkan. 2023. Modified query expansion through generative adversarial networks for information extraction in e-commerce.
Machine Learning with Applications 14 (2023), 100509.
[16] Claudio Carpineto and Giovanni Romano. 2012. A survey of automatic query expansion in information retrieval. Acm Computing Surveys (CSUR)
44, 1 (2012), 1–50.
[17] Shufan Chen, He Zheng, and Lei Cui. 2025. When and How to Augment Your Input: Question Routing Helps Balance the Accuracy and Efficiency
of Large Language Models. In Findings oftheAssociation forComputational Linguistics: NAACL 2025. 3621–3634.
[18] Xinran Chen, Xuanang Chen, Ben He, Tengfei Wen, and Le Sun. 2024. Analyze, generate and refine: Query expansion with LLMs for zero-shot
open-domain QA. In Findings oftheAssociation forComputational Linguistics ACL 2024. 11908–11922.
[19] Yung-Sung Chuang, Wei Fang, Shang-Wen Li, Wen-tau Yih, and James Glass. 2023. Expand, rerank, and retrieve: Query reranking for open-domain
question answering. arXiv preprint arXiv:2305.17080 (2023).
[20] Hang Cui, Ji-Rong Wen, Jian-Yun Nie, and Wei-Ying Ma. 2002. Probabilistic query expansion using query logs. In Proceedings ofthe11th
international conference onWorld Wide Web. 325–332.
[21] Hang Cui, Ji-Rong Wen, Jian-Yun Nie, and Wei-Ying Ma. 2003. Query expansion by mining user logs. IEEE transactions onknowledge anddata
engineering 15, 4 (2003), 829–839.

34 Li et al.
[22] Giulio D’Erasmo, Giovanni Trappolini, Fabrizio Silvestri, and Nicola Tonellotto. 2025. ECLIPSE: Contrastive Dimension Importance Estimation
with Pseudo-Irrelevance Feedback for Dense Retrieval. In Proceedings ofthe2025 International ACM SIGIR Conference onInnovative Concepts
andTheories inInformation Retrieval (ICTIR). 147–154.
[23] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language
understanding. In Proceedings ofthe2019 conference oftheNorth American chapter oftheassociation forcomputational linguistics: human
language technologies, volume 1(long andshort papers). 4171–4186.
[24] Lijuan Diao, Hong Yan, Fuxue Li, Shoujun Song, Guohua Lei, and Feng Wang. 2018. The research of query expansion based on medical terms
reweighting in medical information retrieval. EURASIP Journal onWireless Communications andNetworking 2018, 1 (2018), 105.
[25] Efthimis N Efthimiadis. 1996. Query Expansion. Annual review ofinformation science andtechnology (ARIST) 31 (1996), 121–87.
[26] Jiazhan Feng, Chongyang Tao, Xiubo Geng, Tao Shen, Can Xu, Guodong Long, Dongyan Zhao, and Daxin Jiang. 2023. Synergistic interplay
between search and large language models for information retrieval. arXiv preprint arXiv:2305.07402 (2023).
[27] Luciano Floridi and Massimo Chiriatti. 2020. GPT-3: Its nature, scope, limits, and consequences. Minds andmachines 30, 4 (2020), 681–694.
[28] Gaihua Fu, Christopher B Jones, and Alia I Abdelmoty. 2005. Ontology-based spatial query expansion in information retrieval. In OTM Confederated
International Conferences" OntheMove toMeaningful Internet Systems". Springer, 1466–1482.
[29] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings ofthe61st
Annual Meeting oftheAssociation forComputational Linguistics (Volume 1:Long Papers). 1762–1777.
[30] Silviu Homoceanu and Wolf-Tilo Balke. 2014. Querying concepts in product data by means of query expansion. Web intelligence andagent
systems 12, 1 (2014), 1–14.
[31] Junying Hu, Kai Sun, Cong Ma, Hai Zhang, Jiangshe Zhang, et al .[n. d.]. Query Expansion by Retrieval-Augmented Generation Based on Deepseek.
KaiandMa,Cong andZhang, HaiandZhang, Jiangshe, Query Expansion byRetrieval-Augmented Generation Based onDeepseek ([n. d.]).
[32] Qing Huang, Yang Yang, and Ming Cheng. 2019. Deep learning the semantics of change sequences for query expansion. Software: Practice and
Experience 49, 11 (2019), 1600–1617.
[33] Qing Huang, Yangrui Yang, Xue Zhan, Hongyan Wan, and Guoqing Wu. 2018. Query expansion based on statistical learning from code changes.
Software: Practice andExperience 48, 7 (2018), 1333–1351.
[34] Thomas Jaenich, Graham McDonald, and Iadh Ounis. 2025. Fair Exposure Allocation Using Generative Query Expansion. In European Conference
onInformation Retrieval. Springer, 267–281.
[35] Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui Wang, and Michael Bendersky. 2023. Query expansion by prompting large language models.
arXiv preprint arXiv:2305.03653 (2023).
[36] Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li, Changying Hao, Shuaiqiang Wang, and Dawei Yin. 2023. Mill: Mutual verification with large
language models for zero-shot query expansion. arXiv preprint arXiv:2310.19056 (2023).
[37] SeongKu Kang, Shivam Agarwal, Bowen Jin, Dongha Lee, Hwanjo Yu, and Jiawei Han. 2024. Improving retrieval in theme-specific applications
using a corpus topical taxonomy. In Proceedings oftheACM Web Conference 2024. 1497–1508.
[38] Sam Kerr Kelly. 2021. Enhancing Query Expansion for Rare Diseases in PubMed Using Embedding-Based Semantic Representations. (2021).
[39] Ayesha Khader and Faezeh Ensan. 2023. Learning to rank query expansion terms for COVID-19 scholarly search. Journal ofBiomedical Informatics
142 (2023), 104386 – 104386. https://api.semanticscholar.org/CorpusID:258659102
[40] Ayesha Khader, Hamid Sajjadi, and Faezeh Ensan. 2022. Contextual Query Expansion for Conducting Technology-Assisted Biomedical Reviews..
InCanadian AI.
[41] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings
ofthe43rd International ACM SIGIR conference onresearch anddevelopment inInformation Retrieval. 39–48.
[42] Hamin Koo, Minseon Kim, and Sung Ju Hwang. 2024. Optimizing query generation for enhanced document retrieval in rag. arXiv preprint
arXiv:2407.12325 (2024).
[43] Robert Krovetz and W Bruce Croft. 1992. Lexical ambiguity and information retrieval. ACM Transactions onInformation Systems (TOIS) 10, 2
(1992), 115–141.
[44] Vaibhav Kumar and Jamie Callan. 2020. Making information seeking easier: An improved pipeline for conversational search. In Findings ofthe
Association forComputational Linguistics: EMNLP 2020. 3971–3980.
[45] Zhu Kunpeng, Wang Xiaolong, and Liu Yuanchao. 2009. A new query expansion method based on query logs mining. International Journal on
Asian Language Processing 19, 1 (2009), 1–12.
[46] Victor Lavrenko and W Bruce Croft. 2017. Relevance-based language models. In ACM SIGIR Forum , Vol. 51. ACM New York, NY, USA, 260–267.
[47] Jinhyuk Lee, WonJin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2019. BioBERT: a pre-trained biomedical
language representation model for biomedical text mining. Bioinformatics 36 (2019), 1234 – 1240. https://api.semanticscholar.org/CorpusID:
59291975
[48] Sung-Min Lee, Eunhwan Park, Donghyeon Jeon, Inho Kang, and Seung-Hoon Na. 2024. RADCoT: Retrieval-augmented distillation to specialization
models for generating chain-of-thoughts in query expansion. In Proceedings ofthe2024 Joint International Conference onComputational
Linguistics, Language Resources andEvaluation (LREC-COLING 2024). 13514–13523.
[49] Jiayin Lei, Weijiang Li, Feng Wang, and Hui Deng. 2011. A survey on query expansion based on local analysis. In 2011 4thInternational Conference
onIntelligent Networks andIntelligent Systems. IEEE, 1–4.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 35
[50] Yibin Lei, Yu Cao, Tianyi Zhou, Tao Shen, and Andrew Yates. 2024. Corpus-Steered Query Expansion with Large Language Models. In Proceedings
ofthe18th Conference oftheEuropean Chapter oftheAssociation forComputational Linguistics (Volume 2:Short Papers). 393–401.
[51] Yibin Lei, Tao Shen, and Andrew Yates. 2025. ThinkQE: Query Expansion via an Evolving Thinking Process. arXiv preprint arXiv:2506.09260
(2025).
[52] Otávio AL Lemos, Adriano C de Paula, Felipe C Zanichelli, and Cristina V Lopes. 2014. Thesaurus-based automatic query expansion for
interface-driven code search. In Proceedings ofthe11th working conference onmining software repositories. 212–221.
[53] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdel rahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019.
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In Annual Meeting of
theAssociation forComputational Linguistics. https://api.semanticscholar.org/CorpusID:204960716
[54] Dong Li, Yelong Shen, Ruoming Jin, Yi Mao, Kuan Wang, and Weizhu Chen. 2022. Generation-augmented query expansion for code retrieval.
arXiv preprint arXiv:2212.10692 (2022).
[55] Hang Li, Xiao Wang, Bevan Koopman, and Guido Zuccon. 2025. Pseudo Relevance Feedback is Enough to Close the Gap Between Small and Large
Dense Retrieval Models. arXiv preprint arXiv:2503.14887 (2025).
[56] Hang Li, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon. 2025. LLM-VPRF: Large Language Model Based Vector Pseudo Relevance
Feedback. arXiv preprint arXiv:2504.01448 (2025).
[57] Ronghan Li, Mingze Cui, Benben Wang, Yu Wang, and Qiguang Miao. 2025. Query Expansion with Topic-Aware In-Context Learning and
Vocabulary Projection for Open-Domain Dense Retrieval. Available atSSRN 5367307 (2025).
[58] Jason Liu, Seohyun Kim, Vijayaraghavan Murali, Swarat Chaudhuri, and Satish Chandra. 2019. Neural query expansion for code search. In
Proceedings ofthe3rdacm sigplan international workshop onmachine learning andprogramming languages. 29–37.
[59] Linqing Liu, Minghan Li, Jimmy Lin, Sebastian Riedel, and Pontus Stenetorp. 2022. Query expansion using contextual clue sampling with language
models. arXiv preprint arXiv:2210.07093 (2022).
[60] Lingyuan Liu and Mengxiang Zhang. 2025. Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based
Query Expansion. arXiv:2506.04760 [cs.IR] https://arxiv.org/abs/2506.04760
[61] XiangZheng Liu. 2023. When self-supervision met Query Expansion. Authorea Preprints (2023).
[62] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019).
[63] Meili Lu, Xiaobing Sun, Shaowei Wang, David Lo, and Yucong Duan. 2015. Query expansion via wordnet for effective code search. In 2015 IEEE
22nd International Conference onSoftware Analysis, Evolution, andReengineering (SANER). IEEE, 545–549.
[64] Zhiyong Lu, Won Kim, and W John Wilbur. 2009. Evaluation of query expansion using MeSH in PubMed. Information retrieval 12, 1 (2009), 69–80.
[65] Iain Mackie, Shubham Chatterjee, and Jeff Dalton. 2023. Generative and Pseudo-Relevant Feedback for Sparse, Dense and Learned Sparse Retrieval.
InWorkshop onLarge Language Models’ Interpretation andTrustworthiness, CIKM 2023.
[66] Iain Mackie, Shubham Chatterjee, and Jeffrey Dalton. 2023. Generative relevance feedback with large language models. In Proceedings ofthe46th
international ACM SIGIR conference onresearch anddevelopment ininformation retrieval. 2026–2031.
[67] Iain Mackie and Jeffrey Dalton. 2022. Query-specific knowledge graphs for complex finance topics. arXiv preprint arXiv:2211.04142 (2022).
[68] Aritra Mandal, Ishita K Khan, and Prathyusha Senthil Kumar. 2019. Query Rewriting using Automatic Synonym Extraction for E-commerce
Search.. In eCOM@ SIGIR.
[69] Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. 2020. Generation-augmented retrieval for
open-domain question answering. arXiv preprint arXiv:2009.08553 (2020).
[70] Yuetian Mao, Chengcheng Wan, Yuze Jiang, and Xiaodong Gu. 2023. Self-supervised query reformulation for code search. In Proceedings ofthe
31st acm joint european software engineering conference andsymposium onthefoundations ofsoftware engineering. 363–374.
[71] George Michalopoulos, Yuanxin Wang, Hussam Kaka, Helen H Chen, and Alexander Wong. 2020. UmlsBERT: Clinical Domain Knowledge
Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus. In North American Chapter ofthe
Association forComputational Linguistics. https://api.semanticscholar.org/CorpusID:224803491
[72] Jack Minker, Gerald A Wilson, and Barbara H Zimmerman. 1972. An evaluation of query expansion by the addition of clustered terms for a
document retrieval system. Information Storage andRetrieval 8, 6 (1972), 329–348.
[73] Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. 2023. Learning to relate to previous turns in conversational
search. In Proceedings ofthe29th ACM SIGKDD Conference onKnowledge Discovery andData Mining. 1722–1732.
[74] Shahrzad Naseri, Jeffrey Dalton, Andrew Yates, and James Allan. 2021. Ceqe: Contextualized embeddings for query expansion. In European
conference oninformation retrieval. Springer, 467–482.
[75] Shahrzad Naseri, Jeffrey Dalton, Andrew Yates, and James Allan. 2022. CEQE to SQET: A study of contextualized embeddings for query expansion.
Information Retrieval Journal 25, 2 (2022), 184–208.
[76] Jamal Abdul Nasir, Iraklis Varlamis, and Samreen Ishfaq. 2019. A knowledge-based semantic framework for query expansion. Information
processing &management 56, 5 (2019), 1605–1617.
[77] Roberto Navigli, Paola Velardi, et al .2003. An analysis of ontology-based query expansion strategies. In Proceedings ofthe14th European
Conference onMachine Learning, Workshop onAdaptive Text Extraction andMining, Cavtat-Dubrovnik, Croatia. 42–49.

36 Li et al.
[78] Liming Nie, He Jiang, Zhilei Ren, Zeyi Sun, and Xiaochen Li. 2016. Query expansion based on crowd knowledge for code search. IEEE Transactions
onServices Computing 9, 5 (2016), 771–783.
[79] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2023. Ragtruth: A hallucination
corpus for developing trustworthy retrieval-augmented language models. arXiv preprint arXiv:2401.00396 (2023).
[80] Mengjia Niu, Hao Li, Jie Shi, Hamed Haddadi, and Fan Mo. 2024. Mitigating hallucinations in large language models via self-refinement-enhanced
knowledge retrieval. arXiv preprint arXiv:2405.06545 (2024).
[81] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).
[82] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex
Ray, et al .2022. Training language models to follow instructions with human feedback. Advances inneural information processing systems 35
(2022), 27730–27744.
[83] Helen J Peat and Peter Willett. 1991. The limitations of term co-occurrence data for query expansion in document retrieval systems. Journal ofthe
american society forinformation science 42, 5 (1991), 378–383.
[84] Wenjun Peng, Guiyang Li, Yue Jiang, Zilong Wang, Dan Ou, Xiaoyi Zeng, Derong Xu, Tong Xu, and Enhong Chen. 2024. Large language model
based long-tail query rewriting in taobao search. In Companion Proceedings oftheACM Web Conference 2024. 20–28.
[85] Massimo Perna. 2025. Knowledge graph for query enrichment in retrieval augmented generation in domain specific application . Master’s thesis.
University of Twente.
[86] Matthew E Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018. Dissecting contextual word embeddings: Architecture and
representation. arXiv preprint arXiv:1808.08949 (2018).
[87] Varad Pimpalkhute, John Heyer, Xusen Yin, and Sameer Gupta. 2024. SoftQE: Learned representations of queries expanded by LLMs. In European
Conference onInformation Retrieval. Springer, 68–77.
[88] Yonggang Qiu and Hans-Peter Frei. 1993. Concept based query expansion. In Proceedings ofthe16th annual international ACM SIGIR conference
onResearch anddevelopment ininformation retrieval. 160–169.
[89] Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019.
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J.Mach. Learn. Res. 21 (2019), 140:1–140:67. https:
//api.semanticscholar.org/CorpusID:204838007
[90] Muhammad Mahbubur Rahman, Sorami Hisamoto, and Kevin Duh. 2019. Query expansion for cross-language question re-ranking. arXiv preprint
arXiv:1904.07982 (2019).
[91] Delaram Rajaei, Zahra Taheri, and Hossein Fani. 2024. Enhancing RAG’s Retrieval via Query Backtranslations. In International Conference on
Web Information Systems Engineering. Springer, 270–285.
[92] Muhammad Shihab Rashid, Jannat Ara Meem, Yue Dong, and Vagelis Hristidis. 2024. Progressive query expansion for retrieval over cost-constrained
data sources. arXiv preprint arXiv:2406.07136 (2024).
[93] Muhammad Ahsan Raza, Rahmah Mokhtar, and Noraziah Ahmad. 2019. A survey of statistical approaches for query expansion. Knowledge and
information systems 61, 1 (2019), 1–25.
[94] Andreia Rodrıguez Rivas, Eva Lorenzo Iglesias, and L Borrajo. 2014. Study of query expansion techniques and their application in the biomedical
information retrieval. TheScientific World Journal 2014, 1 (2014), 132158.
[95] Stephen E Robertson and K Sparck Jones. 1976. Relevance weighting of search terms. Journal oftheAmerican Society forInformation science 27,
3 (1976), 129–146.
[96] Joseph John Rocchio Jr. 1971. Relevance feedback in information retrieval. TheSMART retrieval system: experiments inautomatic document
processing (1971).
[97] Fabio Sartori. 2009. A comparison of methods and techniques for ontological query expansion. In Research Conference onMetadata andSemantic
Research. Springer, 203–214.
[98] Abdus Satter and Kazi Sakib. 2016. A search log mining based query expansion technique to improve effectiveness in code search. In 2016 19th
International Conference onComputer andInformation Technology (ICCIT). IEEE, 586–591.
[99] Tao Shen, Guodong Long, Xiubo Geng, Chongyang Tao, Tianyi Zhou, and Daxin Jiang. 2023. Large language models are strong zero-shot retriever.
arXiv preprint arXiv:2304.14233 (2023).
[100] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth,
Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
[101] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
[102] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is
all you need. Advances inneural information processing systems 30 (2017).
[103] Ellen M Voorhees. 1993. Using WordNet to disambiguate word senses for text retrieval. In Proceedings ofthe16th annual international ACM
SIGIR conference onResearch anddevelopment ininformation retrieval. 171–180.
[104] Ellen M Voorhees. 1994. Query expansion using lexical-semantic relations. In SIGIR’94: Proceedings oftheSeventeenth Annual International
ACM-SIGIR Conference onResearch andDevelopment inInformation Retrieval, organised byDublin City University. Springer, 61–69.

Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey 37
[105] Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, and Maarten De Rijke. 2020. Query resolution for conversational search with limited
supervision. In Proceedings ofthe43rd International ACM SIGIR conference onresearch anddevelopment inInformation Retrieval. 921–930.
[106] Gaike Wang, Xin Ni, Qi Shen, and Mingxuan Yang. 2024. Leveraging Large Language Models for Context-Aware Product Discovery in E-commerce
Search Systems. Journal ofKnowledge Learning andScience Technology ISSN: 2959-6386 (online) 3, 4 (2024), 300–312.
[107] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with large language models. arXiv preprint arXiv:2303.07678 (2023).
[108] Xiao Wang, Sean MacAvaney, Craig Macdonald, and Iadh Ounis. 2023. Generative Query Reformulation for Effective Adhoc Search. ArXiv
abs/2308.00415 (2023). https://api.semanticscholar.org/CorpusID:260351198
[109] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2023. ColBERT-PRF: Semantic pseudo-relevance feedback for dense passage and
document retrieval. ACM Transactions ontheWeb 17, 1 (2023), 1–39.
[110] Zhichao Wang, Bin Bi, Shiva Kumar Pentyala, Kiran Ramnath, Sougata Chaudhuri, Shubham Mehrotra, Xiang-Bo Mao, Sitaram Asur, et al .2024. A
comprehensive survey of llm alignment techniques: Rlhf, rlaif, ppo, dpo and more. arXiv preprint arXiv:2407.16216 (2024).
[111] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned
language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021).
[112] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler,
et al. 2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682 (2022).
[113] Orion Weller, Kyle Lo, David Wadden, Dawn J Lawrie, Benjamin Van Durme, Arman Cohan, and Luca Soldaini. 2023. When do Generative
Query and Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and Datasets. ArXiv abs/2309.08541 (2023). https:
//api.semanticscholar.org/CorpusID:262012661
[114] Ji-Rong Wen, Jian-Yun Nie, and Hong-Jiang Zhang. 2002. Query clustering using user logs. ACM Transactions onInformation Systems 20, 1
(2002), 59–81.
[115] Peter Willett. 1988. Recent trends in hierarchic document clustering: a critical review. Information processing &management 24, 5 (1988), 577–597.
[116] Dingjun Wu, Yukun Yan, Zhenghao Liu, Zhiyuan Liu, and Maosong Sun. 2025. KG-Infused RAG: Augmenting Corpus-Based RAG with External
Knowledge Graphs. arXiv preprint arXiv:2506.09542 (2025).
[117] Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A Rossi, Haoliang Wang, and Julian McAuley. 2025. Knowledge-Aware Query Expansion with
Large Language Models for Textual and Relational Retrieval. In Proceedings ofthe2025 Conference oftheNations oftheAmericas Chapter of
theAssociation forComputational Linguistics: Human Language Technologies (Volume 1:Long Papers). 4275–4286.
[118] Jinxi Xu and W Bruce Croft. 2000. Improving the effectiveness of information retrieval with local context analysis. ACM Transactions on
Information Systems (TOIS) 18, 1 (2000), 79–112.
[119] Jinxi Xu and W Bruce Croft. 2017. Quary expansion using local and global document analysis. In Acm sigir forum , Vol. 51. ACM New York, NY,
USA, 168–175.
[120] Adam Yang, Gustavo Penha, Enrico Palumbo, and Hugues Bouchard. 2025. Aligned Query Expansion: Efficient Query Expansion for Information
Retrieval through LLM Alignment. arXiv preprint arXiv:2507.11042 (2025).
[121] Sijia Yao, Pengcheng Huang, Zhenghao Liu, Yu Gu, Yukun Yan, Shi Yu, and Ge Yu. 2025. ExpandR: Teaching Dense Retrievers Beyond Queries
with LLM Guidance. arXiv:2502.17057 [cs.IR] https://arxiv.org/abs/2502.17057
[122] Zhijun Yin, Milad Shokouhi, and Nick Craswell. 2009. Query expansion using external evidence. In European conference oninformation retrieval .
Springer, 362–374.
[123] Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, and Kunwoo Park. 2025. Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based
Query Expansion. In Findings oftheAssociation forComputational Linguistics: ACL 2025. Association for Computational Linguistics, Vienna,
Austria, 19170–19187. https://doi.org/10.18653/v1/2025.findings-acl.980
[124] HongChien Yu, Chenyan Xiong, and Jamie Callan. 2021. Improving query representations for dense retrieval with pseudo relevance feedback. In
Proceedings ofthe30th ACM International Conference onInformation &Knowledge Management. 3592–3596.
[125] Feng Zhang, Haoran Niu, Iman Keivanloo, and Ying Zou. 2017. Expanding queries for code search using semantically related api class-names.
IEEE Transactions onSoftware Engineering 44, 11 (2017), 1070–1082.
[126] Jiacheng Zhang. 2024. Instruction Tuning for Domain Adaptation of Large Language Models. (2024).
[127] Kepu Zhang, Zhongxiang Sun, Weijie Yu, Xiaoxue Zang, Kai Zheng, Yang Song, Han Li, and Jun Xu. 2025. QE-RAG: A Robust Retrieval-Augmented
Generation Benchmark for Query Entry Errors. arXiv preprint arXiv:2504.04062 (2025).
[128] Le Zhang, Yihong Wu, Qian Yang, and Jian-Yun Nie. 2024. Exploring the Best Practices of Query Expansion with Large Language Models. In
Findings oftheAssociation forComputational Linguistics: EMNLP 2024. 1872–1883.
[129] Mengxiao Zhang, Yongning Wu, Raif Rustamov, Hongyu Zhu, Haoran Shi, Yuqi Wu, Lei Tang, Zuohua Zhang, and Chu Wang. 2022. Advancing
query rewriting in e-commerce via shopping intent learning. (2022).
[130] Xiaoqing Zhang, Xiuying Chen, Shen Gao, Shuqi Li, Xin Gao, Ji-Rong Wen, and Rui Yan. 2024. Selecting query-bag as pseudo relevance feedback
for information-seeking conversations. arXiv preprint arXiv:2404.04272 (2024).
[131] Yanan Zhang, Weijie Cui, Yangfan Zhang, Xiaoling Bai, Zhe Zhang, Jin Ma, Xiang Chen, and Tianhua Zhou. 2023. Event-centric query expansion
in web search. arXiv preprint arXiv:2305.19019 (2023).
[132] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al .2025. Siren’s song
in the ai ocean: A survey on hallucination in large language models. Computational Linguistics (2025), 1–45.

38 Li et al.
[133] Zhi Zheng, Kai Hui, Ben He, Xianpei Han, Le Sun, and Andrew Yates. 2020. BERT-QE: contextualized query expansion for document re-ranking.
arXiv preprint arXiv:2009.07258 (2020).
[134] Wenhao Zhu, Xiaoyu Zhang, Qiuhong Zhai, and Chenyun Liu. 2023. A hybrid text generation-based query expansion method for open-domain
question answering. Future Internet 15, 5 (2023), 180.
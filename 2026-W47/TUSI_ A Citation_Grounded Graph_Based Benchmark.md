# TUSI: A Citation-Grounded Graph-Based Benchmark

**Authors**: Marzieh Karimi, Mehran Safayani, Alireza Basiri

**Published**: 2026-11-16 19:38:41

**PDF URL**: [https://arxiv.org/pdf/2611.21823v1](https://arxiv.org/pdf/2611.21823v1)

## Abstract
Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI-Bench provides two complementary evaluation tasks: (i) TUSI-QA, an open-ended question answering benchmark containing 1,999 questions that requires grounded responses with explicit legal citations, and (ii) TUSI-Bar, a multiple-choice benchmark containing approximately 500 four-option questions derived from Iranian Bar examination questions. Furthermore, we propose a graph-based retrieval baseline that leverages relational structure within the legal knowledge graph. Experimental results demonstrate that incorporating graph-based relations improves retrieval recall by up to 8 percentage points compared to standard baselines, highlighting the importance of structured legal representations for RAG systems. The proposed benchmark provides a standardized resource for evaluating grounded legal reasoning in Persian and enables future research on structured legal intelligence.

## Full Text


<!-- PDF content starts -->

Highlights
TUSI: A Citation-Grounded Graph-Based Benchmark for Retrieval-Augmented
Generation in Persian Legal Question Answering
Marzieh Karimi, Mehran Safayani, Alireza Basiri
•TUSI-Bench enables grounded RAG evaluation in Persian legal QA.
•TUSI-QA and TUSI-Bar cover complementary legal reasoning tasks.
•A legal knowledge graph supports structure-aware retrieval.
•Graph-based retrieval improves recall by up to 8 percentage points.
arXiv:2611.21823v1  [cs.CL]  16 Nov 2026

TUSI: A Citation-Grounded Graph-Based Benchmark for
Retrieval-Augmented Generation in Persian Legal Question
Answering
Marzieh Karimia,¡, Mehran Safayaniaand Alireza Basiria
aDepartment of Electrical and Computer Engineering, Isfahan University of Technology, Isfahan, 84156-83111, Iran
ARTICLE INFO
Keywords:
Retrieval-Augmented Genera-
tion
Legal Information Retrieval
Knowledge Graph Representa-
tion
Evaluation Benchmark
Structured Legal DatasetABSTRACT
Reliable evaluation of retrieval-augmented generation (RAG) systems in the legal
domain remains challenging, particularly for low-resource languages such as Per-
sian, where available benchmarks are limited in realism, grounding, and coverage of
legalreasoning.Existingresourcesareeithersyntheticallygeneratedfromstatutory
texts, lacking natural information needs, or derived from informal legal question–
answer sources without explicit linkage to authoritative legal documents, making
themunsuitableforrigorousevaluationofgroundedlegalreasoning.Toaddressthis
gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for
the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered
legal knowledge framework containing 7,261 legal articles and integrating codified
statutes, legal commentaries, doctrinal writings, and Supreme Court unification
rulings. In addition, we construct a legal knowledge graph that explicitly models
relationshipsamonglegalarticlesandconceptstosupportstructure-awareretrieval.
Based on this foundation, TUSI-Bench provides two complementary evaluation
tasks:(i)TUSI-QA,anopen-endedquestionansweringbenchmarkcontaining1,999
questions that requires grounded responses with explicit legal citations, and (ii)
TUSI-Bar, a multiple-choice benchmark containing approximately 500 four-option
questions derived from Iranian Bar examination questions. Furthermore, we pro-
pose a graph-based retrieval baseline that leverages relational structure within the
legalknowledgegraph.Experimentalresultsdemonstratethatincorporatinggraph-
based relations improves retrieval recall by up to 8 percentage points compared to
standard baselines, highlighting the importance of structured legal representations
for RAG systems. The proposed benchmark provides a standardized resource for
evaluating grounded legal reasoning in Persian and enables future research on
structured legal intelligence.
1. Introduction
Recent advances in artificial intelligence have led to the widespread adoption of large language models
(LLMs) across diverse domains, including education, healthcare, and particularly the legal domain [1]. In
legal applications, these models are increasingly used to support tasks such as legal information retrieval,
question answering, and case analysis [2, 3]. The appeal is clear: legal knowledge is expressed in dense
natural-language text, and access to professional legal advice is costly and unevenly distributed. However,
the reliability of such systems remains a critical concern in a high-stakes domain such as law, where incorrect
or unsupported outputs may lead to serious legal and ethical consequences.
Working with real legal data exposes several well-documented weaknesses of LLM-based systems. First,
LLMs are prone to hallucination, i.e., generating fluent but factually unsupported content [4]. In the legal
domain this takes a particularly harmful form: recent studies show that general-purpose models frequently
fabricate case law and misstate legal holdings [5], and that even commercial legal research tools built on
retrieval-augmented pipelines still produce incorrect or misgrounded answers at a non-trivial rate [6].
Second, legal texts are long, densely cross-referenced, and written in a formal register with complex
syntax and specialized terminology, so that even foundational tasks such as named entity recognition require
mkarimi@ec.iut.ac.ir(M. Karimi);safayani@iut.ac.ir(M. Safayani);basiri@iut.ac.ir(A. Basiri)
orcid(s):
Karimi et al.:Preprint submitted to ElsevierPage 1 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
domain-specific adaptation [7]; this register also differs markedly from the everyday language in which
laypeople describe their problems, and as a result the lexical and semantic gap between a user’s narrative
question and the governing provision is large, which makes retrieval difficult [3, 8]. Third, a legally sound
answer rarely depends on a single passage: it often requires combining several articles, their authoritative
interpretations, and judicial rulings, whereas LLMs tend to under-use evidence placed in long contexts [9].
Fourth, legal knowledge is jurisdiction-specific and changes over time, whereas the knowledge stored in
model parameters is static, and may reflect a different jurisdiction or an outdated version of the law. Finally,
evaluating legal outputs is intrinsically hard: legal tasks demand multi-step reasoning that goes beyond
surface-level text understanding [10], and open-ended answers admit multiple acceptable formulations, so
reference-based metrics alone are insufficient.
Legal systems worldwide are broadly classified into two major traditions: civil law (Romano-Germanic)
systemsandcommonlawsystems.Incivillawsystems—includingtheIranianlegalsystem,whichisthefocus
of this study—sound legal reasoning requires strict adherence to codified statutory texts, legal doctrines, and
authoritative sources, in contrast to common law systems, where judicial precedent plays the primary role.
This dependence on explicit, codified legal evidence highlights a fundamental limitation of current LLMs:
their knowledge is implicitly encoded within static parameters and cannot naturally adapt to continuously
evolving legal systems. Consequently, ensuring factual grounding, interpretability, and up-to-date legal
reasoning remains a persistent challenge, and it is particularly acute in codified legal systems such as Iran’s,
where a legally sound answer must be explicitly traceable to a specific statutory article.
To address these limitations, Retrieval-Augmented Generation (RAG) [11] has emerged as a promising
approach. By integrating external knowledge sources at inference time, RAG-based methods reduce reliance
on parametric memory, enable dynamic retrieval of relevant legal documents, and make it possible to attach
explicit citations to generated answers. Nevertheless, the development of robust legal RAG systems critically
depends on high-quality evaluation benchmarks that can jointly assess retrieval effectiveness and grounded
generation in domain-specific settings, since errors in either component propagate to the final answer [12].
Several English-language legal benchmarks have been introduced in this direction. Among the most
notable are the Competition on Legal Information Extraction and Entailment (COLIEE) [13] and CLERC
[14], which primarily focus on legal retrieval and entailment, as well as reasoning-oriented resources such
as the Reasoning-Focused Legal Retrieval Benchmark [3], which introduces more complex legal question
answering grounded in document retrieval. Broader suites such as LegalBench [10] cover a wide range of
legal reasoning tasks, but they are built on English-language legal materials and do not cover Persian or the
Iranian legal system.
These English-centric resources cannot simply be reused for Persian legal applications, for three main
reasons. First, law is jurisdiction-bound. Substantive rules, legal concepts, and procedural requirements
differ across countries, so a model or benchmark grounded in U.S., European, or other legal systems cannot
validate answers about Iranian law; an answer that is correct under one jurisdiction may be wrong, or even
meaningless, under another. The Iranian legal system is a codified civil-law system whose sources, structure,
anddoctrinalfoundationsdiffersubstantiallyfromthoseofcommon-lawjurisdictions,anditincludesjudicial
institutions specific to its system, such as the binding Unification Rulings of the General Assembly of the
Supreme Court. Second, the structure of legal evidence differs: many English benchmarks are case-based,
with queries and relevant documents linked through judicial citations, and those that are statute-based cover
other jurisdictions, whereas Iranian legal reasoning is anchored in individually numbered statutory articles,
so retrieval must be evaluated at the article level. Third, translation does not bridge the gap: translating
English benchmarks into Persian would change the language but not the underlying law, and it would also
erase the terminology, register, and phrasing of real Persian legal consultations.
ThePersianlanguageaddsafurtherlayerofdifficulty.Persianisacomparativelylow-resourcelanguagein
NLP: annotated corpora, retrieval benchmarks, and domain-specific resources are scarce relative to English,
and multilingual models are trained on far less Persian text. Persian is written in a right-to-left Perso-Arabic
script with frequent orthographic variation (e.g., Arabic versus Persian character forms, and inconsistent
use of the zero-width non-joiner), which complicates normalization, matching, and indexing. Persian legal
language is moreover a specialized register that mixes modern Persian with Arabic-derived legal terminology
and archaic constructions, and its vocabulary is rarely seen in general-purpose training data. In addition,
tokenizers trained on predominantly English data tend to fragment non-English text into more tokens, which
Karimi et al.:Preprint submitted to ElsevierPage 2 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
increases cost and shortens the effective context available to the model [15], a significant handicap when
legal provisions and their interpretations are long. As a result, both retrieval models and generators tend to
perform less reliably on Persian legal text.
Despite these needs, existing resources for Persian legal NLP remain very limited. Available datasets
are confined to a few repositories on HuggingFace [16, 17] and generally fall into two categories: either
synthetically generated question sets derived directly from civil law articles—lacking realistic information
needs and natural query complexity—or crawled question–answer pairs from legal forums, where answers
provided by legal practitioners are not systematically linked to legal articles and are neither curated nor
structured for evaluation purposes. General-purpose Persian QA datasets, in turn, are mostly extractive and
non-legal. As a result, no existing Persian resource provides a reliable benchmark for rigorous evaluation of
retrieval-augmented legal systems.
To address this gap, we introduce TUSI-Bench, a structured Persian legal evaluation benchmark
specifically designed for assessing retrieval-augmented legal systems. The name TUSI is inspired by
Nasir al-Din al-Tusi, the renowned Persian polymath and scholar. TUSI-Bench is built on a multi-layer
legal knowledge framework of 7,261 statutory articles, linked to expert interpretations and Supreme
Court Unification Rulings through a legal knowledge graph. On top of this framework, it provides
two complementary components: TUSI-QA, an open-ended question answering benchmark of 1,999 real
lawyer–client questions with explicit, validated legal citations, and TUSI-Bar, a four-option multiple-choice
benchmark derived from Iranian Bar examination questions. We further propose a structure-aware retrieval
baseline that exploits the relational structure of the knowledge graph.
Our main contributions are summarized as follows:
•We introduce TUSI-Bench, to the best of our knowledge among the first structured Persian legal
benchmarks designed for evaluating RAG systems, addressing both the scarcity of Persian legal
resources and the jurisdictional mismatch of English-centric benchmarks. It comprises two comple-
mentary components: TUSI-QA, covering citation-grounded open-ended legal question answering, and
TUSI-Bar, covering four-option multiple-choice legal reasoning.
•We construct a legal knowledge graph that links statutory articles, expert interpretations, and
Unification Rulings, and use it to define a structure-aware retrieval baseline that improves Recall@5
by up to 8 percentage points over its dense-retrieval counterpart.
•Unlike existing Persian legal datasets, TUSI-Bench connects statutory texts with naturally formulated
legal questions and evidence-grounded answers, validated through an iterative agent-and-expert
filtering process, providing a realistic and challenging evaluation setting.
The remainder of this paper is organized as follows. Section 2 reviews related work in legal NLP and
retrieval-augmentedsystems.Section3describesthedatasetconstructionprocessanddatasources.Section4
presents the structure of the proposed benchmarks and annotation methodology. Section 5 reports the
experimental setup, evaluation results, and discussion. Finally, Section 6 concludes the paper and outlines
future research directions.
2. Related Work
This section situates TUSI-Bench with respect to four lines of prior work: general-domain retrieval and
question answering (QA) benchmarks, legal retrieval and QA benchmarks, graph-based retrieval-augmented
generation(RAG)forlegaltext,andPersian-languagebenchmarks.Foreach,westatewhatexistingresources
provide and which requirements of Persian statutory legal QA they leave unaddressed.
2.1. General-Domain Retrieval and Question Answering
Open-domain QA benchmarks such as Natural Questions (NQ) [18] and HotpotQA [19] evaluate systems
that retrieve evidence from Wikipedia and answer questions over it, and they are widely used to evaluate
RAG [11]. BEIR [20] unifies 18 retrieval datasets across 9 task types. In these benchmarks, answers are
typically short spans (or yes/no), and relevant passages share considerable vocabulary with the query;
Zheng et al. [3] show that the query–passage lexical similarity of NQ and HotpotQA is markedly higher than
Karimi et al.:Preprint submitted to ElsevierPage 3 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
in reasoning-intensive legal retrieval. HotpotQA adds multi-hop reasoning across several documents, but its
evidence remains unstructured text rather than a structured body of rules with cross-references.
Recent retrieval benchmarks reduce this reliance on surface matching. BRIGHT [21] collects realistic
reasoning-intensive queries in 12 datasets from domains such as economics, psychology, mathematics, and
coding, and BIRCO [22] evaluates retrieval under multi-faceted objectives. Neither targets legal text, so
neither captures the structure of statutory provisions, their cross-references, or the need to ground an
answer in specific articles.
2.2. Legal Retrieval and Question Answering Benchmarks
Legal text differs from general-domain text in its formal register, its dense cross-referencing, and the
requirement that conclusions be traceable to authoritative sources.
Case-law retrieval.Many legal retrieval datasets are built from citation links between court decisions.
CLERC [14] derives queries from passages of U.S. federal case law and asks systems to retrieve the precedents
cited in them; it also includes a task of generating legal analysis from retrieved cases, in which models
frequentlyhallucinatecitations.Thecase-lawtasksofCOLIEE[13]followasimilarquery-case-to-supporting-
case setup over Canadian Federal Court decisions, with entire case documents serving as queries (thousands
of tokens on average, Table 1). Because cited rules are often restated in the citing text, such queries tend to
share much vocabulary with their relevant documents [3]. These resources therefore differ from ours in two
ways: their queries are long judicial texts rather than short natural-language questions from people seeking
legal advice, and their evidence is unstructured precedent text rather than numbered statutory articles.
Statute retrieval and question answering.The statute-law tasks of COLIEE [13] are the closest precedent
to our setting: given a yes/no question from the Japanese bar exam, systems retrieve relevant Civil Code
articles (Task 3) and decide whether they entail the answer (Task 4). These tasks cover a single legal system
and use short, hypothetical exam questions with binary answers, so they cannot assess whether a system
can explain, interpret, and cite the supporting provisions in a free-text answer to a real consultation. The
reasoning-focused benchmark of Zheng et al. [3] comprises Bar Exam QA (bar exam fact patterns paired
with judicial-opinion passages) and Housing Statute QA (6,853 questions over U.S. state statutes, with a
corpus of about 1.8 million passages). It shows that low lexical overlap between queries and relevant text
defeats lexical retrievers such as BM25, a property that TUSI-QA shares and even exceeds (Table 2). It
is, however, built entirely on English-language U.S. materials and therefore cannot validate answers about
Iranian law.
Broader legal suites.LegalBench [10] is a collaboratively built benchmark of 162 tasks covering six types
of legal reasoning (issue-spotting, rule-recall, rule-application, rule-conclusion, interpretation, and rhetorical
understanding). Because the relevant legal text is supplied in the prompt, it evaluates reasoning but not
retrieval. LegalBench-RAG [12] isolates the retrieval step of legal RAG, with 6,858 query–snippet pairs over
a corpus of about 79M characters; however, it is built from contract and privacy-policy tasks, evaluates
span-level rather than article-level retrieval, and does not evaluate answer generation. MSLR [23] evaluates
multi-step legal reasoning over 1,389 Chinese insider-trading judgments, with about 60,000 annotated step-
level reasoning paths based on the IRAC (Issue–Rule–Application–Conclusion) framework. It shows that
current models remain limited in multi-step legal reasoning, but it targets reasoning quality within a single
offense category of the Chinese legal system rather than statute-grounded retrieval and citation.
Overall, these benchmarks cover a handful of jurisdictions and languages (English/U.S., Japanese, and
Chinese),andnonecombinesastatutorycorpuswithbothopen-ended,citation-groundedQAandexam-style
multiple-choice evaluation.
2.3. Graph-Based Retrieval-Augmented Generation for Legal Text
Legal evidence is rarely confined to a single passage: a sound answer may depend on an article, the
provisions it refers to, expert interpretations, and judicial rulings. Conventional RAG treats a corpus as a
flat collection of passages and cannot follow such relations. Graph-based RAG (GraphRAG) addresses this
by organizing the corpus as a graph and using its structure during retrieval and generation. In the original
GraphRAG [24], an LLM extracts entities and relations from the corpus into a knowledge graph, the graph
Karimi et al.:Preprint submitted to ElsevierPage 4 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
is partitioned into communities, and answers to corpus-level questions are built from community summaries,
improving the comprehensiveness and diversity of the answers. Surveys describe a common workflow of
graph-based indexing, graph-guided retrieval, and graph-enhanced generation [25].
Structure-aware retrieval has also been explored for statutes. For statutory article retrieval, Louis et
al. [26] derive a graph of the legislative structure from the division headings of the Belgian codes and
apply graph neural networks to it, reporting that the hierarchy of the law is a useful retrieval signal.
More recent work adapts GraphRAG to legal corpora explicitly. LegalGraphRAG [27] argues that legal
corpora are heterogeneous, mixing cases, articles, and interpretations, and that a flat graph cannot separate
them. It organizes legal knowledge into a hierarchical graph with ontology, fact, and rule layers and reasons
over it with three cooperating agents (researcher, auditor, and adjudicator), and it is applied to criminal
charge prediction. Oliveira et al. [28] model Portuguese normative documents as graphs whose nodes are
structural units (documents, articles, paragraphs, and items) and whose edges are normative relationships;
their retriever assembles evidence along root-to-leaf paths before generation, and they evaluate it with
RAGAS on 203 internal resolutions of a single energy company.
These works propose methods for individual corpora: the Portuguese study is limited to the internal
resolutions of one organization, and LegalGraphRAG targets criminal charge prediction rather than statute-
grounded question answering. We are not aware of any publicly available benchmark for Persian law that
pairs an article-level legal knowledge graph with questions whose supporting provisions are annotated.
2.4. Persian Benchmarks
Persian is a comparatively low-resource language in NLP, and the available benchmarks are dominated
by general-domain tasks.
General question answering and evaluation.ParSQuAD [29], PQuAD [30], and PersianQuAD [31] follow
an extractive reading-comprehension format and are commonly used for Persian RAG-style evaluation; they
assessspanextractionratherthanreasoningoverlegalstatutes.Multiple-choicebenchmarksevaluategeneral
model ability. The Khayyam Challenge (PersianMMLU) [32] contains 20,192 four-choice questions from 38
subjects of Persian school examinations. PerMMLU [33] is a localized and extended Persian counterpart
of MMLU that covers school-level, general, and specialized knowledge as well as knowledge specific to
Persian-speaking societies; its general-knowledge subset touches on topics such as national laws, but only as
isolated questions. PerCoR [34] offers 106K multiple-choice sentence-completion problems for commonsense
reasoning. None of these benchmarks includes a statutory corpus or involves retrieval.
Persian legal resources.Publicly released resources for Persian law are scarce. To our knowledge, the only
openly available Persian legal QA collections are a few repositories on Hugging Face, such as the Iran Legal
FAQ datasets [16, 17]. They consist either of synthetically generated questions derived directly from legal
articles, which lack realistic information needs, or of question–answer pairs crawled from legal forums, whose
answers are explanatory, context-dependent, and not curated for evaluation. Crucially, neither kind provides
ground-truth labels linking each question to the statutory articles that support its answer. To the best of
our knowledge, no publicly released Persian legal dataset combines real-world legal questions with verified,
article-level ground-truth labels, and as a result there is currently no way to evaluate whether a Persian
legal system retrieves and cites the correct provisions.
To the best of our knowledge, no Persian benchmark combines
1. a corpus of statutory legal texts,
2. real-world legal questions,
3. bar exam-style multiple-choice questions,
4. reasoning-intensive retrieval with low lexical overlap between queries and relevant laws, and
5. an article-level legal knowledge graph with annotated supporting provisions.
TUSI-Bench addresses this gap by providing a Persian legal benchmark grounded in statutory law, designed
to evaluate retrieval-based legal reasoning under realistic exam and practice settings.
Karimi et al.:Preprint submitted to ElsevierPage 5 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Article-level Statutory corpusExpert legal interpretationsJudicial unification rulingsStructured legal knowledge layer7,261 articles across 50 statutes, segmented for exact addressing27.7% of articles enriched with doctrinal commentaryLegal relationship graph5,183 undirected cross-references · 33.6% node coverageTusi-BenchTusi-QAReal lawyer–client consultations, filtered and validatedTusi-BarIranian Bar Exam questions, past three yearsSupreme Court General Assembly precedentsarticleinterpretationunification rulingsarticle-articlearticle-interpretationarticle-ruling Train 1490Test 509
Figure 1:Overview of the structured legal knowledge layer and the proposed TUSI-Bench. TUSI-Bench consists of two
complementarysubsets:TUSI-QA,acitation-groundedquestionansweringsetderivedfromreallawyer–clientconsultations;
and TUSI-Bar, a multiple-choice question set derived from Iranian Bar Exam questions.
3. Structured Legal Knowledge Layer
Our dataset is designed to support two primary tasks: Legal Information Retrieval and Legal Question
Answering. In this setting, a system is expected to retrieve the relevant legal provisions for a given query and
generate an answer grounded in statutory law and authoritative legal sources. Accordingly, the dataset is not
limitedtoisolatedquestion–answerpairs;instead,itincorporatesastructuredlegalknowledgelayertoenable
systematic evaluation of retrieval-based and evidence-grounded legal systems. An overview of the resulting
dataset architecture is presented in Figure 1. The framework integrates a structured legal knowledge layer
with benchmark-oriented evaluation resources, forming a unified foundation for legal information retrieval
and citation-grounded question answering.
The dataset is organized as a multi-layer legal knowledge architecture consisting of: (1) statutory laws
and executive regulations, (2) Judicial Unification Rulings, (3) expert interpretations and legal doctrine, and
(4) a graph of inter-article relationships. This structure enables evaluation not only of surface-level retrieval,
but also of multi-hop reasoning, cross-source retrieval, and interpretation-aware legal question answering.
The outcome of this stage is a segmented statutory corpus accompanied by a legal relationship graph and
complementary interpretative and judicial layers.
3.1. Article-Level Statutory Corpus
To construct the core legal knowledge layer, we selected a set of principal Iranian statutes that serve
as foundational sources in formal legal education and the national bar examination. The selection criteria
included: (i) frequency of citation in lawyers’ responses within the raw QA corpus, and (ii) recognized
doctrinal and examination authority. The full texts of these statutes, relevant executive regulations, and
Judicial Unification Rulings were collected from official and authoritative sources.
A preprocessing pipeline was applied, including removal of non-textual elements, orthographic normal-
ization, and standardization of spacing and punctuation. For fine-grained retrieval and precise referencing,
statutory texts were segmented at the level of articles. Each segment was assigned a unique identifier to
enable exact addressing during retrieval, evaluation, and QA alignment. This design allows performance
assessment not only at the statute level but also at the article level, which is essential for rigorous Legal
Information Retrieval evaluation.
To enrich the semantic depth of the corpus, a complementary layer of expert legal interpretation was
incorporated. These interpretations were extracted from a specialized legal knowledge source, where legal
experts provide doctrinal explanations of complex statutory provisions grounded in authoritative legal
scholarshipandinterpretativepractice[35].Eachinterpretativeunitwasexplicitlylinkedtoitscorresponding
statutory article. This enrichment supports interpretation-aware retrieval and retrieval-augmented legal
reasoning. An example of a statutory article and its corresponding expert interpretation is presented in
Figure A.1.
Karimi et al.:Preprint submitted to ElsevierPage 6 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
101102103
Article length (tokens, log scale)0100200300400Count(a) Article Length Distribution
Median = 242 tokens
Mean = 316 tokens
102 103
Explanation length (tokens, log scale)020406080100Count(c) Explanation Length Distribution
Median = 523 tokens
Mean = 654 tokens
Figure 2:Length distributions of statutory articles (left) and their corresponding legal interpretations (right).
3.2. Interpretation-Enriched Knowledge Graph
Beyond statutory text and interpretation, inter-article relationships were systematically extracted in two
ways. First, if one article explicitly referenced another within its text, we established a direct link between
those two articles. Second, in the expert interpretations, when a connection to a particular statutory article
was mentioned, a link was created between the interpretation and the referenced article.
Based on these references, we constructed a legal relationship graph. Formally, the legal knowledge graph
is defined asG= .V,E/, where:
•Vconsists of nodes representing:
–statutory articles,
–interpretative units, and
–Unification Rulings, examples of which are shown in Figure A.2;
•Econsists of typed edges representing:
–article-article references,
–article-interpretation links, and
–article-ruling associations.
This graph structure enables the evaluation of graph-aware retrieval methods and multi-hop reasoning
scenarios, particularly in cases where a legally sound answer requires combining multiple interrelated
provisions. By incorporating these structured dependencies, the dataset supports experimentation with
graph-based ranking, multi-source retrieval, and structured legal inference.
3.3. Corpus Statistics
To characterize the structural, textual, and temporal properties of the legal corpus, we conduct a
comprehensive quantitative analysis covering document length distributions, graph connectivity, coverage
patterns, and temporal dynamics (Figures 2, 3, 4 and 5). The dataset comprises 50 statutes and 7,261
legal articles, forming a heterogeneous legal knowledge graph with 9,272 nodes and 5,183 undirected cross-
references.
The corpus exhibits strong textual heterogeneity with highly skewed, heavy-tailed length distributions.
As illustrated in Figure 2, article lengths show a pronounced imbalance, with a median of 240 tokens and
a mean of 370 tokens, indicating the presence of a large number of short provisions alongside a smaller set
Karimi et al.:Preprint submitted to ElsevierPage 7 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
1930s 1940s 1950s 1960s 1970s 1980s 1990s 2000s 2010s 2020s
Decade0500100015002000Number of Articles1,049
389
11959457
3513082702,161
160n = 5,323
Figure 3:Temporal distribution of legal articles in the dataset across decades. The total number of articles is (n = 5,323).
0 5 10 15 20
Out-degree k100101102103104Count (log scale)Power-law alpha ~ 2.50R2 = 0.961(a) Out-degree count distribution
27.7%(b) Explanation coverage
Has data (27.7%) Missing (72.3%)33.6%(c) Reference coverage
Has data (33.6%) Missing (66.4%)
Figure 4:Structural properties of the legal reference graph. (a) Out-degree distribution on a log scale, showing a heavy-
tailed pattern consistent with a power-law (α= 2.50,R2= 0.961), indicating highly uneven citation behavior. (b)
Coverage of explanation annotations across nodes, where 27.7% of articles have associated interpretations. (c) Coverage
of reference annotations, showing that 33.6% of nodes participate in at least one reference link, highlighting moderate but
sparse connectivity in the graph structure.
of substantially longer legal texts. In contrast, legal interpretations (when available) are consistently longer,
with a median of 538 tokens and a mean of 695 tokens, reflecting their role as expert-driven expansions of
selectedstatutoryprovisions.Importantly,theseinterpretationsarenotuniformlyrequiredacrossallarticles;
rather, they are selectively added by domain experts to clarify complex or ambiguous provisions, while many
articles remain semantically self-contained.
Karimi et al.:Preprint submitted to ElsevierPage 8 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
0 200 400 600 800 1000 1200 1400
Number of articlesCivil Judgments Enforcement ActLabour LawDirect Taxes ActNon-Contentious Matters ActCode of Civil ProcedureLaw for the Procedure of the 
General and Revolution CourtsCode of Criminal ProcedureCommercial CodeIslamic Penal CodeCivil Code
1641802032883795296998271,0171,339
Figure 5:Top 10 laws by article count in the dataset.
Theinterpretativelayercovers27.7%ofarticles(Figure4),whichshouldnotbeinterpretedasadeficiency
in the corpus, but rather as an intentional and selective enrichment process targeting legally or semantically
complex cases. This results in a partially augmented knowledge structure where only a subset of provisions
is accompanied by explanatory commentary.
From a structural perspective, the reference network is highly sparse and exhibits a clear power-law
degree distribution (α= 2.50,R2= 0.961), as shown in Figure 4. A large proportion of nodes (66.4%) have
zero out-degree, while a small number of highly connected nodes act as structural hubs, indicating an uneven
distribution of cross-referencing activity across legal provisions.
The corpus also demonstrates strong temporal heterogeneity (Figure 3), spanning multiple decades
from early legislative records to contemporary statutes, with a noticeable concentration in recent decades,
particularly the 2010s. This reflects both the historical depth and the modern expansion of the legal system
represented in the dataset.
Finally, the distribution of articles across statutes is highly imbalanced (Figure 5), with a small number
of laws accounting for a large fraction of the corpus. This long-tail pattern further reinforces the non-uniform
nature of the dataset at both the document and structural levels.
By capturing the full and realistic diversity of Iranian legal rules across textual, structural, and temporal
dimensions, this corpus forms a highly complex and heterogeneous legal knowledge graph, which in turn
makes tasks such as information retrieval, reasoning, and graph-based learning particularly challenging.
4. TUSI-Bench: A Citation-Grounded Legal QA Benchmark
This section details the process of creating TUSI-Bench (Persian LEgal eXam & consultation Bench-
mark), a hybrid question-answer dataset tailored for the Persian legal domain. The objective of this
benchmarkistoprovideacomprehensiveresourceforthesimultaneousevaluationoflegaldocumentretrieval
systems and legal question-answering models. TUSI-Bench consists of two complementary components: (1)
TUSI-QA, a citation-grounded Question-Answer set extracted from real user-lawyer interactions, and (2)
TUSI-Bar, a Multiple-Choice Question set extracted from formal legal exams.
TUSI-QA includes real user questions answered by legal professionals and is based on two core principles:
first, the use of actual data derived from real user-lawyer interactions; and second, ensuring that the answers
are directly traceable to specific legal texts. TUSI-Bar is derived from the Iranian bar exam questions of the
past three years.
Karimi et al.:Preprint submitted to ElsevierPage 9 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Legal consultancy siteHugging Face datasetMerge and standardizeTarget-law filtering
Final validated set~7,900 Q&A pairs600K Q&A pairs
Drop answers with no citationfiltering AgentThree automatic criteria1,939 validated samplesHuman expert reviewStructured feedbackFeedbackData Collection
Figure 6:Overview of the TUSI-QA dataset construction pipeline.
4.1. TUSI-QA: Consultation-Driven QA with Legal Citations
The first component of TUSI-Bench focuses on real-world legal queries and expert open-ended responses.
The construction of TUSI-QA follows a multi-stage pipeline designed to transform raw legal question-answer
interactions into a citation-grounded and quality-controlled benchmark. As illustrated in Figure 6, the
process begins with collecting real-world legal queries and expert responses from multiple sources, followed
by data integration and standardization. Subsequently, citation-based filtering is applied to retain only
legally grounded responses, and an iterative validation process combining intelligent agent-based evaluation
and expert legal review is performed to ensure citation correctness, semantic alignment, and overall legal
reliability. The final outcome is a high-quality collection of validated legal question-answer pairs suitable for
evaluating retrieval and reasoning capabilities in Legal RAG systems.
4.1.1. Data Collection and Structural Integration
In the initial stage, question-answer data was gathered from two public sources to ensure that the samples
reflect real-world legal issues. The first source was an Iranian legal consultancy website, where users post
questions and receive responses from lawyers. Approximately 7,900 questions along with their corresponding
answerswereextractedfromthissource.Thedatacollectionwasstrictlylimitedtopubliclyavailablecontent,
adhering to technical constraints.
The second source was a publicly available dataset published on the Hugging Face platform [16],
containing 600,000 real Persian legal question-answer pairs with multiple answers from lawyers. This source
was used to increase the thematic and stylistic diversity of TUSI-QA.
The data from both sources was standardized and integrated into a uniform structure, ensuring each
sample included the question text and the lawyer’s response. The output of this stage was a preliminary
legal question-answer repository.
4.1.2. Selection of Target Laws and Explicit References
Aftercompilingthepreliminarydataset,asetofTargetLawswasdefinedtofocustheselectionofsamples
on legally-grounded responses. This set of laws includes the primary laws from the bar exam and all relevant
Karimi et al.:Preprint submitted to ElsevierPage 10 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
legal provisions. These laws serve as both a reference framework for locating legal citations and as a retrieval
corpus for subsequent evaluations.
To ensure the quality of the answers, only those responses were selected that explicitly referenced at least
one of the target laws, while those without any legal citations were excluded. Once the lawyer responses
containing legal references were selected, a step was added to link these responses to the corresponding legal
articles. This process involved using an intelligent agent to automatically extract relevant legal provisions
from the dataset of laws and link articles to the corresponding answers. Representative examples of the
resulting citation-grounded QA instances are shown in Figure A.3.
4.1.3. Iterative Citation Filtering via Intelligent Agent and Human Evaluation
Initial qualitative analysis revealed that the mere presence of a legal citation does not guarantee the
quality or correctness of the reference. In several cases, cited provisions were partially irrelevant, insufficient
for supporting the reasoning, or only loosely connected to the generated answer. In other cases, although
legal articles were cited, the reasoning itself was not explicitly grounded in the referenced legal text.
As illustrated in Figure 6, citation validation for TUSI-QA was therefore formulated as an iterative
human-in-the-loop process combining an intelligent agent with expert legal review. In the first stage, the
intelligent agent automatically evaluated every sample by jointly analyzing the question, answer, and cited
legal provisions according to three criteria: (1) the relevance and sufficiency of the cited legal provisions
with respect to the question, (2) explicit grounding of the answer in the cited legal text, and (3) semantic
consistency between the question and the generated answer. Samples violating any of these criteria were
discarded.
In the second stage, a legal expert inspected a randomly selected subset of the automatically filtered
samples. Rather than reviewing the entire dataset, the expert assessed the correctness of the citations, the
adequacy of legal grounding, and the consistency between questions and answers. Whenever deficiencies
were identified, structured feedback was provided to refine the filtering strategy. The intelligent agent was
subsequently updated based on this feedback, the filtering process was repeated, and another random subset
was reviewed. This iterative procedure continued until the expert confirmed that the retained samples
consistently satisfied both the predefined automatic criteria and the expected legal quality standards.
After the iterative filtering process, the final TUSI-QA dataset contained 1,999 validated question-answer
pairs. These samples were subsequently partitioned into training and test sets using a stratified sampling
strategy designed to preserve the citation-frequency distribution of legal provisions across both subsets.
Instead of performing a purely random split, the sampling process maintained approximately the same
proportion of highly frequent and infrequently cited legal articles in the training and test sets. This strategy
prevents distributional bias and yields a more representative evaluation benchmark for both retrieval and
downstream legal reasoning.
Figure 7 illustrates the resulting citation-frequency distributions. The legal references exhibit a
pronounced long-tail distribution, where a small number of legal provisions account for a large fraction
of all citations, whereas most provisions appear only a few times. Despite this highly skewed distribution,
the train-test split preserves a similar citation profile, ensuring that both subsets contain representative
proportions of frequent and rare legal provisions.
Finally, we analyzed the number of supporting legal provisions required to answer each question. Unlike
many existing legal question answering datasets that assume a single supporting statute, each question in
TUSI-QAmayrequireone,two,orthreerelevantlegalarticles.AsshowninFigure8,althoughmostquestions
are grounded in a single legal provision, a considerable number require reasoning over two provisions, and
a smaller subset depends on three legal articles. This multi-reference characteristic substantially increases
retrieval complexity, as successful systems must retrieve the complete set of supporting legal provisions
rather than only the most relevant article, providing a more realistic evaluation of retrieval completeness
and legal reasoning capabilities.
4.1.4. Quantitative Analysis of the Filtering Effect
To quantitatively evaluate the impact of citation filtering on TUSI-QA, we computed two semantic align-
ment metrics using BGE-M3 embeddings [36]: (1) answer–legal provision similarity and (2) question–legal
provision similarity, both measured via cosine similarity.
Karimi et al.:Preprint submitted to ElsevierPage 11 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
010203040Train
referencesReference Count Distribution Across Ranked IDs
Smoothed trend
Reference count
0 150 300 450 600
Ranked IDs sorted by reference count in the training set06121824Test
referencesSmoothed trend
Reference count
Figure 7:Distribution of citation frequencies across ranked legal provisions in the training and test sets of TUSI-QA. Legal
provisions are ranked according to their citation frequency in the training set. The figure shows that the long-tail citation
distribution is preserved after the stratified train-test split.
As shown in Figure 9, the KDE distributions shift to the right after filtering, indicating higher overall
semantic alignment. For answer–legal provision similarity, the mean increased from 0.526 to 0.556. A similar
improvement was observed for question–legal provision similarity. To assess whether these differences are
statistically meaningful, we conducted Welch’s t-test [37], which is robust to unequal variances and large
sample sizes. The test yielded a statistically significant difference for answer–legal provision similarity
(p ⃗0.001), with a small-to-medium effect size (Cohen’sd= 0.36). The very smallp-value indicates that
the observed improvement is highly unlikely to have occurred by random variation alone. Comparable
statistically significant results were obtained for question–legal provision similarity.
Importantly, while a clear rightward shift is observed, the overall shape of the distributions remains
largely unchanged. This suggests that the filtering process did not merely retain samples with extremely
high lexical or semantic similarity while discarding the rest. Instead, it selectively reduced low-alignment
cases while preserving the natural variability of the dataset.
Overall, the filtering improves semantic grounding and citation relevance without artificially compressing
the similarity distribution, resulting in a more stable and realistic component of TUSI-Bench for evaluating
Legal RAG and explainable QA systems.
4.2. TUSI-Bar: Bar Exam Multiple-Choice QA
The second component of TUSI-Bench, TUSI-Bar, consists of multiple-choice questions and answers
extracted from the Iranian Bar Exam held over the past three years. Each question in this set follows a
four-choice format, with one correct answer and three distractors. Importantly, every question in TUSI-Bar
is associated with a specific legal provision, although the annotation is at the law level rather than the
individual article level.
The questions, derived from the actual Bar Exam, cover various areas of Iranian law, ensuring thematic
diversity. In contrast to TUSI-QA, where each answer directly references a legal article, TUSI-Bar does not
provide article-level citations but rather links each question to the broader legal provision or law it addresses.
Examples of TUSI-Bar instances, including their answer choices, ground-truth answers, and associated legal
sources, are presented in Figure A.4.
Karimi et al.:Preprint submitted to ElsevierPage 12 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
1 2 3
Number of relevant articles020406080100Percentage of questions (%)1258
139
33364
95
50Train
Test
Figure 8:Distribution of the number of relevant legal articles required to answer each question in TUSI-QA (training
and test sets). Each question may depend on one, two, or three legal provisions, reflecting different levels of retrieval and
reasoning complexity.
0.0 0.2 0.4 0.6 0.8 1.0
Cosine Similarity012345Density (KDE)Before
After
0.0 0.2 0.4 0.6 0.8 1.0
Cosine Similarity0.00.51.01.52.02.53.0Density (KDE)Before
After
Figure 9:KDE plots of cosine similarity scores before and after citation filtering in TUSI-QA. Left: question–legal provision
similarity. Right: answer–legal provision similarity. The rightward shift after filtering indicates stronger semantic alignment.
The multiple-choice nature of the questions also allows for more controlled evaluation of system
performance, as it provides a set of discrete answer options that can be easily compared against model
predictions. This structure is ideal for benchmarking, as it allows for clear, quantifiable measures of accuracy
based on the system’s ability to select the correct answer from the provided choices.
4.3. Comparison with Existing Datasets
To better contextualize TUSI-Bench, we compare it with existing legal and general-domain question-
answering datasets in terms of scale, passage structure, query and answer length, and lexical overlap. As
summarized in Table 1, TUSI-Bench differs from prior resources along multiple dimensions. In terms of scale,
it contains fewer passages than several existing datasets, but its passages are relatively longer. In terms of
task formulation, unlike datasets that focus primarily on retrieval, TUSI-QA defines a complete open-ended
question-answering task. With respect to query and answer structure, it includes long, context-rich questions
and explanatory, analytical answers. Finally, as shown in Table 2 and Figure 10, it exhibits a distinct lexical
Karimi et al.:Preprint submitted to ElsevierPage 13 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Table 1
Summary of datasets. We report number of queries (Q), number of passages (P), number of answers (A), and the average
length of queries, passages and answers (calculated with the GPT-2 tokenizer [38]).
Dataset SplitTotal Number Average Length
Q P A Q P A
TUSI-QA (Ours)Train 1490 7261 1490 448.04 555.22 397.50
Test 509 7261 509 426.85 548.42 362.95
Comparison
PersianQuAD [31]Train 18567 3149 18567 59.07 1032.47 16.14
Test 1000 35 2339 57.79 1525.66 12.05
PQuAD [30]Train 63994 8979 48273 65.43 782.99 33.63
Test 8002 1059 8540 67.93 771.51 33.13
ParsQuAD [29]Train 18906 5723 15517 59.36 582.09 11.20
Test 5711 1165 8742 59.53 762.67 12.59
Bar Exam QA [3] – 1,195 / 1,815 856,835 – 172 / 157 131 –
Housing Statute QA [3] – 6,853 1,837,403 – 15 349 –
Natural Questions [18] – 3,452 2,681,468 – 11 102 –
HotpotQA [19] – 7,405 5,233,329 – 20 63 –
COLIEE [13] – 1,278 5,616 – 6,730 6,768 –
CLERC [14] – 2,851 1,842,422 – 415 3,303 –
Table 2
Lexical similarity scores for the retrieval task and downstream QA task across datasets. Values are reported as mean,95%
confidence interval.
Dataset LS retrieval LSQA
TUSI-QA (Ours)0.03 , 0.00 0.19 , 0.02
PersianQuAD [31]0.12 , 0.00 0.09 , 0.00
PQuAD [30]0.16 , 0.00 0.12 , 0.00
ParSQuAD [29]0.10 , 0.00 0.12 , 0.00
Bar Exam QA [3]0.07 , 0.00 0.14 , 0.01
Housing Statute QA [3]0.09 , 0.00 0.00 , 0.00
Natural Questions [18]0.27 , 0.01 0.25 , 0.01
HotPotQA [19]0.26 , 0.00 0.24 , 0.00
COLIEE [13]0.27 , 0.00–
CLERC [14]0.26 , 0.01–
similarity pattern, with a larger gap between users’ natural-language descriptions and formal legal texts,
while the answers remain grounded in legal sources.
In terms of scale, the number of passages in TUSI-Bench is smaller than in some legal benchmarks, such
as Bar Exam QA and Housing Statute QA [3]. However, as shown in Table 1, the average passage length
in TUSI-QA is higher than in these two benchmarks. Thus, although the retrieval corpus is smaller, its
textual units are longer and more content-rich. By contrast, datasets such as COLIEE [13] and CLERC [14]
have a different task formulation and contain both a substantial number of passages and considerably longer
average passage lengths. The long query lengths in these datasets are mainly due to the structure of legal
retrieval tasks, where long legal texts or case descriptions are used as retrieval inputs, rather than natural
user questions in a question-answering framework.
From the perspective of language and QA formulation, PersianQuAD [31], PQuAD [30], and ParsQuAD
[29] are among the main Persian resources for evaluating question-answering and RAG systems in the general
domain. These datasets mainly consist of questions whose answers are short, specific, and usually extractable
Karimi et al.:Preprint submitted to ElsevierPage 14 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
from the reference text. This characteristic is also reflected in the answer-length statistics in Table 1: the
average answer length in these datasets is substantially lower than in TUSI-QA. Therefore, although these
resources are valuable for evaluating Persian information retrieval and answer extraction, they generally do
not require the generation of long, analytical, or explanatory answers.
In contrast, TUSI-QA is designed as an open-ended QA benchmark. Its answers are not merely short or
factoidresponses;rather,theyrequirelegalexplanation,interpretationofstatutoryprovisions,andalignment
between the cited law and the conditions described in the question. This distinction is reflected in Table 1,
where the answers in both the training and test splits are considerably longer than those in Persian QuAD-
based datasets. Moreover, among the QA datasets considered in this study, TUSI-QA stands out in terms of
both query and answer length. This reflects its focus on context-dependent legal problems, where users often
need to describe the facts of the case, temporal conditions, relationships between parties, and relevant legal
details. Although COLIEE has the highest average query length statistically, its task formulation differs
from QA datasets: its queries are not necessarily natural-language questions and often correspond to legal
cases or text segments. Therefore, its query length is not directly comparable to the question length in QA
datasets.
TUSI-QA also exhibits distinctive lexical similarity characteristics. As shown in Table 2 and Figure 10,
its lexical similarity in the retrieval setting is lower than that of the other compared datasets. This indicates
limited surface-level overlap between user questions and the corresponding legal passages, which is consistent
with the nature of legal consulting scenarios: users typically describe their problems in natural and narrative
language, whereas statutory texts are written in a formal and structured style. In contrast, the higher lexical
similarity in the QA setting indicates that the final answers remain connected to the relevant legal texts and
do not drift away from legal grounding.
Overall, TUSI-Bench is not necessarily the largest resource in terms of the number of passages, but it
offers a distinctive combination of properties: relatively long passages, context-rich user questions, long open-
ended answers, and explicit grounding in legal texts. In addition, among the QA datasets considered in this
study, TUSI-QA is particularly prominent in terms of average query and answer length. These characteristics
make TUSI-Bench a suitable benchmark for evaluating Legal RAG systems in scenarios where both accurate
evidence retrieval and explainable answer generation are essential.
For all experiments, retrieval is performed over the complete collection of legal provisions introduced in
Section 3, and all evaluations are conducted on the corresponding test splits of TUSI-Bench.
5. Experimental Setup and Results
This section presents the experimental results of the proposed benchmark. We consider three evaluation
tasks: retrieval of relevant legal provisions, open-ended question answering, and legal reasoning. Retrieval
and open-ended QA are evaluated on the proposed Legal QA benchmark, while legal reasoning is evaluated
on the Bar Exam benchmark as a four-choice question answering task.
All experiments were conducted on a workstation equipped with an NVIDIA RTX A6000 GPU. For large
language model inference, including the iterative citation filtering and quality control pipeline described in
Section4,weemployGemma327B[39]quantizedto8-bitprecision.Documentembeddingsareindexedusing
the Weaviate vector database, and retrieval is performed using dense, BM25, or hybrid search depending
on the experimental setting. Unless otherwise specified, all reported results are obtained using identical
preprocessing and retrieval settings for fair comparison.
5.1. Evaluation Metrics
The proposed benchmark supports the evaluation of both Legal Information Retrieval (LIR) and Legal
Question Answering (LQA). Accordingly, we employ two complementary evaluation protocols. For retrieval
experiments, we adopt standard information retrieval metrics that assess the quality of document ranking.
For question answering, we employ an LLM-as-a-Judge framework to evaluate the legal quality and practical
usefulness of generated responses.
Retrieval MetricsLetQdenote the set of evaluation queries and letR qbe the set of relevant documents
associated with queryq.
Karimi et al.:Preprint submitted to ElsevierPage 15 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
0.00.3
0.00.2
0.00.1
0.00.2
0.00.2
0.00.3
0.00.2
0.00.2
0.00.5
0.00.2
0.00.5
0.00.7
0.00.1
0.00.1
0.00.2
0.00.1
0.00.1N/A
0.0 0.5 1.00.00.2
0.0 0.5 1.0N/ACitation-Grounded Legal QA
PersianQuAD
PQuAD
ParSQuAD
Bar Exam QA
Housing Statute QA
Natural Questions
HotpotQA
COLEE (Task 1.1)  †
CLERC  †Query–Gold Passages Answer–Gold Passages
Cosine Similarity Score
Proportion Median
† Query–Answer similarity not available.
Figure 10:Distribution of cosine similarity scores across datasets for Query–Gold Passage and Answer–Gold Passage
settings. The figure illustrates lexical similarity patterns and highlights the distinct behavior of the proposed dataset
compared with existing benchmarks.
Precision@k.Precision measures the proportion of retrieved documents that are relevant among the top-k
retrieved results:
PŽk=Relk
k, (1)
Karimi et al.:Preprint submitted to ElsevierPage 16 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
whereRel kdenotes the set of relevant documents retrieved within the firstkpositions. Precision reflects
the ability of a retrieval system to avoid returning irrelevant legal provisions.
Recall@k.Recall measures the proportion of relevant documents that are successfully retrieved:
RŽk=Relk
Rq. (2)
Recall is particularly important in legal retrieval since missing a relevant statutory provision may result
in incomplete legal reasoning.
Mean Average Precision (MAP).For a queryq, Average Precision (AP) is defined as
AP.q/ =1
RqN
k=1PŽkİrel.k/,(3)
whereNis the total number of retrieved documents andrel.k/equals 1 if the document at rankkis relevant
and 0 otherwise. The Mean Average Precision is computed as
MAP=1
QqËQAP.q/. (4)
MAP rewards systems that consistently rank relevant legal documents near the top of the retrieval list.
Mean Reciprocal Rank (MRR).MRR [40] evaluates the rank of the first relevant retrieved document:
MRR=1
QqËQ1
rank q, (5)
whererank qdenotes the position of the first relevant document for queryq. MRR is particularly useful
in legal retrieval scenarios where identifying a key statutory provision early can significantly improve
downstream reasoning.
Normalized Discounted Cumulative Gain (NDCG).NDCG [41] measuresranking quality while assigning
greater importance to highly ranked relevant documents. The Discounted Cumulative Gain is defined as
DCGŽk=k
i=12reli* 1
log2.i+ 1/,(6)
whererel idenotes the relevance score of the document at positioni. The normalized version is
NDCGŽk=DCGŽk
IDCGŽk,(7)
whereIDCGŽkis the maximum achievable DCG. NDCG is well suited for legal retrieval because multiple
statutory provisions may be relevant to a single legal question, and their ranking order is important.
LLM-as-a-Judge EvaluationWhile retrieval metrics evaluate the ability to identify relevant legal
documents,theycannotfullyassessthequalityofgeneratedlegalresponses.Therefore,fortheLegalQuestion
Answering task, we adopt an LLM-as-a-Judge [42] evaluation framework.
We employ DeepSeek-V4-Pro [43] as an automatic evaluator. Given the user question, the retrieved legal
context, the reference lawyer answer, and the model-generated answer, the judge model produces both a
qualitative justification and a score for each evaluation criterion.
The generated answers are evaluated according to five complementary dimensions, each scored on a
five-point integer scale.
Karimi et al.:Preprint submitted to ElsevierPage 17 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
jina-v3 qwen3-0.6b e5-large bge-m3 persian-embedding BM2501020304050Score (%)
+1.9+8.0
+5.4+5.5 +6.1
+0.6+3.2
+2.4+2.4 +2.5
+1.0+4.3
+3.3+3.7 +3.6
+0.4+1.4
+2.6+2.5 +2.3
+2.6+11.3
+7.1+8.0+8.4
-0.3-0.6
+0.3+0.3 +0.0Retrieval Evaluation — All Metrics per Model
Retrieval Method
Baseline
Structure-Aware Legal RAG
Metric
Precision@5
Recall@5
MAP@5
MRR@5
NDCG@5
Figure 11:Retrieval evaluation results across different embedding models and the proposed Structure-Aware Legal RAG
framework. The figure illustrates consistent improvements over the baseline in most configurations.
•Correctness: legal correctness and consistency with the reference answer and supporting legal
evidence.
•Effectiveness: practical usefulness and the ability to guide the user toward an appropriate legal course
of action.
•Objectiveness: professional and impartial legal tone while avoiding unsupported certainty.
•Clarity: readability, organization, and accessibility of legal concepts.
•Completeness: adequate coverage of legally relevant aspects without introducing unnecessary or
unrelated content.
Each criterion receives an integer score from 1 to 5, where higher scores indicate better performance. In
addition to the individual criterion scores, the overall QA performance is reported as the average score
across all five dimensions. This evaluation protocol enables assessment not only of factual legal correctness
but also of the practical, professional, and communicative quality of generated legal advice.
5.2. Baselines
We evaluate two primary retrieval baselines: BM25 [44] and a vanilla dense retrieval-based RAG
pipeline [11]. BM25 serves as a sparse lexical matching baseline, while the dense retrieval setting encodes
each segmented legal chunk into vector representations stored in a vector database, where relevant passages
are retrieved based on semantic similarity with the input query.
Toensureafaircomparisonacrossembeddingstrategies,weevaluatethedenseretrievalpipelineusingfive
different embedding models:jina-embeddings-v3[45],bge-m3[36],multilingual-e5-large[46], and a
domain-adaptedPersianembeddingmodelbasedonfine-tuningofapre-trainedmultilingualXLM-RoBERTa
encoder, trained on Persian legal corpora, andQwen3-Embedding-0.6B[47]. These models are selected to
cover both multilingual and Persian-specialized embedding spaces, enabling a comprehensive evaluation of
retrieval performance across different representation paradigms.
In particular, the first three models are widely adopted in prior Persian and multilingual retrieval studies
due to their strong semantic retrieval performance [48], while the Persian-specific model is included as
a domain-adapted baseline for Persian legal text retrieval. Finally,Qwen3-Embedding-0.6Bis added as a
recently released large-scale multilingual embedding model from the Qwen3 family, representing a stronger
modern baseline.
Karimi et al.:Preprint submitted to ElsevierPage 18 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Table 3
Downstream evaluation results on the Cited Question-Answer Set.
Method Correctness Effectiveness Objectiveness Clarity Completeness
Gemma3 27B 8bit [39] 1.947 2.092 3.314 3.827 1.782
Gemma3 27B 8bit [39] + Structure-Aware
Legal RAG2.420 2.5403.487 3.6442.206
GPT-5-nano [49] 2.110 2.450 4.150 4.0002.010
5.3. Structure-Aware Legal RAG
Recent studies on Graph Retrieval-Augmented Generation show that conventional RAG is not always
sufficientforspecializeddomains[24].Insuchdomains,relevantknowledgeisoftendistributedacrossmultiple
documents, and understanding it depends on conceptual and structural relations among knowledge units. In
this setting, flat text retrieval may fail to preserve the structural dependencies required for domain-specific
reasoning. This limitation is particularly important in legal question answering, where the correct answer
may depend not only on the semantic similarity between a query and a statutory provision, but also on the
relation of that provision to other articles, authoritative interpretations, and Judicial Unification Rulings.
To address this issue, we propose a structure-aware legal retrieval method that combines semantic
retrieval withgraph-basedlegalcontext expansion. The method uses thelegalknowledge graphintroducedin
our dataset. In this graph, nodes represent statutory articles, expert interpretations, and Judicial Unification
Rulings, while edges represent legal relations such as article-to-article references, article–interpretation links,
and article–ruling associations.
In the proposed method, all legal texts are first segmented into semantically coherent chunks. Each
chunk is then encoded as a dense vector and stored in a vector database. Given a user query, the system first
retrieves the chunks that are most semantically similar to the query. This step provides the initial evidence
required for answering the question.
Next, the retrieved chunks are mapped to their corresponding nodes in the legal knowledge graph. For
each matched node, the method retrieves all nodes located in its one-hop neighborhood. The resulting
one-hop expanded node set is returned as the graph-augmented retrieval output and is used as the evidence
set for downstream answer generation.
As a result, the final evidence set includes not only the chunks that are semantically similar to the
query, but also the legally connected articles and sources. This structural expansion allows the system to
retrieve materials that may be legally necessary for reasoning, even if they are not ranked highly by semantic
similarity alone.
Finally, the expanded evidence set is provided to the language model as grounded context, and the
model generates the answer based on the retrieved statutory articles, interpretations, and judicial decisions.
In this way, the proposed method combines semantic similarity with the explicit structure of legal knowledge,
making retrieval more aligned with statute-based legal question answering than flat retrieval.
5.4. Retrieval Evaluation
In the first stage, we evaluate the retrieval performance of the proposedStructure-Aware Legal RAG
framework in comparison with the BM25 [44] baseline and embedding-based retrieval models, as shown in
Figure 11. The objective is to assess the impact of incorporating structural legal information and relational
signals into the retrieval process.
The results indicate that the proposed framework improves retrieval performance across all embedding-
based configurations. The only exception is BM25, where a decrease is observed in both Precision@5 and
Recall@5, dropping by 0.6 and 0.3 percentage points, respectively, as reported in Figure 11. This suggests
that purely lexical matching methods are limited in capturing the semantic and structural complexity of
legal documents.
Among embedding-based models, Jina-v3 achieves the best performance, reaching a Recall@5 of 41.22%
with the proposed Structure-Aware Legal RAG framework, an improvement of up to 8 percentage points over
Karimi et al.:Preprint submitted to ElsevierPage 19 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Table 4
Downstream evaluation results on Bar Exam multiple-choice QA.
Method Accuracy (%)
Gemma3 27B 8bit [39] 33.0
Gemma3 27B 8bit [39] + RAG 36.8
Gemma3 27B 8bit [39] + Structure-Aware Legal RAG 37.7
Deepseek V4 flash [43]57.5
its dense-retrieval-only counterpart, as shown in Figure 11. This demonstrates the effectiveness of combining
dense embeddings with structure-aware retrieval for improving relevant passage identification.
Overall, these results highlight the importance of structural signals in legal retrieval, particularly in
scenarios where lexical similarity between queries and legal provisions is low. As illustrated in Figure 11,
relational and structural cues play a critical role in bridging the semantic gap between queries and legal
texts.
5.5. Downstream Evaluation
To comprehensively evaluate the proposed framework, we conduct experiments on two downstream
legal reasoning tasks: (i) a Cited Question Answering task, which requires generating open-ended, citation-
grounded legal responses, and (ii) a Bar Exam multiple-choice question answering task, designed to assess
structured legal reasoning under constrained answer selection.
In the Cited QA setting, model performance is evaluated across five dimensions, including correctness,
effectiveness, objectiveness, clarity, and completeness. As shown in Table 3, the Structure-Aware Legal RAG
framework improves performance on four of the five evaluation criteria compared to the non-RAG baseline,
with the largest gains observed in correctness, effectiveness, and completeness. In particular, incorporating
structured legal knowledge leads to more reliable and better-grounded responses, improving both reasoning
quality and factual consistency. Clarity is the one criterion that decreases with the addition of retrieved
context (from 3.827 to 3.644), a plausible trade-off in which longer, evidence-grounded answers become
somewhat less concise and readable even as they become more accurate and complete. Objectiveness shows
a modest improvement, indicating that grounding responses in retrieved legal sources does not come at the
cost of a professional and impartial tone. For additional reference, GPT-5-nano [49] is also evaluated in a
zero-shot setting without access to the proposed retrieval framework. Notably, the Structure-Aware Legal
RAG framework still yields improvements over the open-source Gemma3 baseline, highlighting the benefits
of incorporating structured legal knowledge even with an open-source model.
In the Bar Exam multiple-choice setting, we further evaluate the model’s ability to perform structured
legal reasoning and select the correct option among four candidates. As reported in Table 4, the proposed
Structure-Aware Legal RAG framework consistently outperforms both the vanilla model and the standard
RAG baseline, improving accuracy from 33.0% to 37.7%. These results demonstrate that structure-aware
retrieval improves not only open-ended generation tasks but also strengthens discriminative reasoning in
exam-style legal problems, leading to more robust and reliable decision-making.
5.6. Discussion
The proposed work addresses a critical gap in legal information retrieval and question answering for low-
resource languages, particularly Persian. Existing benchmarks in this domain are either synthetic, limited
in scope, or lack explicit grounding in authoritative legal structures. In contrast, this study introduces a
multi-layered Persian legal dataset that integrates statutes, legal texts, doctrinal writings, interpretative
sources, and judicial decisions into a unified and structured framework.
The experimental results demonstrate that leveraging structured legal knowledge significantly improves
bothretrievalandreasoningperformance.Theobservedimprovementsareprimarilyattributedtotheexplicit
modelingoflegalrelationshipsthroughaknowledgegraph,whichenablestheretrievalsystemtomovebeyond
surface-level semantic similarity and incorporate legally meaningful dependencies.
Furthermore, the introduction of two complementary components under TUSI-Bench—TUSI-QA, an
open-ended citation-grounded question answering task, andTUSI-Bar, a multiple-choice legal reasoning
Karimi et al.:Preprint submitted to ElsevierPage 20 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
exam—provides a more realistic and comprehensive evaluation setting for Persian legal NLP systems. These
benchmarks directly address the limitations of existing datasets, which typically fail to capture real-world
legal reasoning complexity or lack reliable ground truth alignment with authoritative sources.
Despite these advantages, the proposed framework also introduces certain challenges. The construction
of a high-quality legal knowledge graph requires careful curation and expert validation, as incorrect or
incomplete relational links may propagate noise into the retrieval process. Additionally, evaluating open-
ended legal answers remains inherently challenging due to the complexity of legal interpretation and
variability in acceptable responses.
6. Conclusion
This study introduced TUSI-Bench, a Persian legal benchmark for evaluating retrieval-augmented
generation systems, together with the structured legal knowledge layer on which it is built. The knowledge
layer comprises 7,261 articles from 50 statutes, linked to expert interpretations and Supreme Court
Unification Rulings through a legal knowledge graph of 9,272 nodes and 5,183 cross-references. On top
of this layer, TUSI-Bench provides two complementary tasks: TUSI-QA, an open-ended question answering
benchmark of 1,999 questions, and TUSI-Bar, a four-option multiple-choice benchmark derived from Iranian
Bar examination questions.
The main novelty of TUSI-Bench lies in the nature and quality of its data. TUSI-QA is built from real
lawyer–client consultations rather than synthetically generated questions, so its queries reflect the narrative,
fact-rich way in which people actually describe legal problems, and they exhibit very low lexical overlap with
the governing statutory text. Each retained answer is explicitly linked to one to three supporting articles of
the target laws, which makes citation-level evaluation of both retrieval and generation possible. The data
werecleanedthroughaniterativefilteringprocessthatcombinesanintelligentagentwithexpertlegalreview,
discarding samples whose citations were irrelevant, insufficient, or not actually used in the reasoning; the
resulting semantic alignment between questions, answers, and cited provisions is significantly higher after
filtering. Finally, the train–test split preserves the long-tail citation distribution of the legal provisions, and
all questions are linked to a shared retrieval corpus, so results are comparable across systems.
Our experiments show that incorporating structured legal knowledge helps. A simple structure-aware
baseline,whichexpandsretrievedchunkswiththeirone-hopneighborsinthelegalknowledgegraph,improves
Recall@5byupto8percentagepointsoveritsdense-retrievalcounterpart,raisesanswerqualityonTUSI-QA,
and increases accuracy on TUSI-Bar. At the same time, the absolute scores remain low, which indicates that
TUSI-Bench is a challenging benchmark with substantial room for improvement.
Several directions remain for future work. First, we plan to expand the corpus beyond the current 50
statutes by adding further laws and other legal documents, such as executive regulations, additional judicial
rulings,andlegalcommentaries,whichwouldbroadencoverageandmakethebenchmarkmorerepresentative
of legal practice. Second, the current graph is unweighted: all edges of a given type contribute equally, and
one-hop expansion can therefore introduce noise, especially around highly connected hub articles. Assigning
weights to edges, for example based on reference frequency, relation type, semantic similarity, or co-citation
patterns in TUSI-QA, would allow the graph to model the strength of the relations between provisions
more faithfully and to rank expanded neighbors rather than include them all. Third, we plan to improve
automatic relation extraction to enhance the scalability and coverage of the graph, and to extend TUSI-Bar
with article-level annotations in addition to the current law-level links.
Data AvailabilityThe dataset developed in this study is publicly available in the TUSI-Bench repository
on the Hugging Face Hub.
References
[1] Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein,
Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models.arXiv
preprint arXiv:2108.07258, 2021.
[2] Bo Li, Shuang Fan, Shaolin Zhu, and Lijie Wen. Cole: A collaborative legal expert prompting framework for large language
models in law.Knowledge-Based Systems, 311:113052, 2025.
Karimi et al.:Preprint submitted to ElsevierPage 21 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
[3] Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D. Manning, Peter Henderson, and
Daniel E. Ho. A reasoning-focused legal retrieval benchmark. InProceedings of the 4th ACM Symposium on Computer
Science and Law (CS&Law ’25). ACM, 2025.
[4] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale
Fung. Survey of hallucination in natural language generation.ACM Computing Surveys, 55(12):1–38, 2023.
[5] Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E. Ho. Large legal fictions: Profiling legal hallucinations in large
language models.Journal of Legal Analysis, 16(1):64–93, 2024.
[6] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. Hallucination-free?
assessing the reliability of leading ai legal research tools.Journal of Empirical Legal Studies, 2025.
[7] Simon Pietro Romano, Giancarlo Sperlí, Mario Varlese, and Andrea Vignali. Ner in the courtroom: A data-driven
framework for legal entity extraction.Information Processing & Management, 64(2, Part A):105132, 2027.
[8] Antoine Louis, Gijs van Dijck, and Gerasimos Spanakis. Interpretable long-form legal question answering with retrieval-
augmented large language models. InProceedings of the AAAI Conference on Artificial Intelligence, 2024.
[9] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost
in the middle: How language models use long contexts.Transactions of the Association for Computational Linguistics,
12:157–173, 2024.
[10] NeelGuha,JulianNyarko,DanielE.Ho,etal. Legalbench:Acollaborativelybuiltbenchmarkformeasuringlegalreasoning
in large language models. InNeurIPS Datasets and Benchmarks Track, 2023.
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler,
Mike Lewis, Wen-tau Yih, Tim Rocktaschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. In
Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.
[12] Nicholas Pipitone and Ghita Houir Alami. Legalbench-rag: A benchmark for retrieval-augmented generation in the legal
domain.arXiv preprint arXiv:2408.10343, 2024.
[13] Randy Goebel et al. Overview of benchmark datasets and methods for coliee 2024. InNew Frontiers in Artificial
Intelligence, pages 109–124, 2024.
[14] Abe Bohan Hou et al. Clerc: A dataset for us legal case retrieval and retrieval-augmented analysis generation. InFindings
of ACL: NAACL, pages 7898–7913, 2025.
[15] Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. Language model tokenizers introduce unfairness
between languages. InAdvances in Neural Information Processing Systems, 2023.
[16] PerSets. Iran legal persian qa, 2025. Hugging Face Dataset, accessed June 2026.
[17] Amir7440. Iran-madani-law, 2025. Hugging Face Dataset, accessed June 2026.
[18] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein,
Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: A benchmark for question answering research.
Transactions of the Association for Computational Linguistics, 7:452–466, 2019.
[19] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. Hotpotqa:Adatasetfordiverse,explainablemulti-hopquestionanswering. InProceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
[20] Nandan Thakur, Nils Reimers, et al. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval
models.arXiv preprint arXiv:2104.08663, 2021.
[21] Hongjin Su et al. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. InInternational
Conference on Learning Representations, pages 48941–48991, 2025.
[22] Xiaoyue Wang et al. Birco: A benchmark of information retrieval tasks with complex objectives.arXiv preprint
arXiv:2402.14151, 2024.
[23] Wenhan Yu, Xinbo Lin, Lanxin Ni, Jinhua Cheng, and Lei Sha. Benchmarking multi-step legal reasoning and analyzing
Chain-of-Thought effects in Large Language Models.Information Processing & Management, 2026.
[24] DarrenEdge,HaTrinh,NewmanCheng,JoshuaBradley,AlexChao,ApurvaMody,StevenTruitt,DashaMetropolitansky,
Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130, 2024.
[25] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph
retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921, 2024.
[26] Antoine Louis, Gijs Van Dijck, and Gerasimos Spanakis. Finding the law: Enhancing statutory article retrieval via graph
neural networks. InProceedings of the 17th Conference of the European Chapter of the Association for Computational
Linguistics, 2023.
[27] Zerui Chen, Qinggang Zhang, Zhishang Xiang, Zhimin Wei, Linfeng Gao, Xiao Huang, Zhihong Zhang, and Jinsong Su.
LegalGraphRAG: Multi-agent graph retrieval-augmented generation for reliable legal reasoning. InProceedings of the
64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 37455–37484, San
Diego, California, United States, 2026. Association for Computational Linguistics.
[28] Vinícius Teles de Oliveira, Deivison Oliveira da Silva, Mateus de Almeida Souza, Maurício Rodrigues Lima, Sávio Salvarino
Teles de Oliveira, and Thierson Couto Rosa. Retrieval-augmented generation and knowledge graphs in Portuguese-
languagelegaldocuments. InProceedings of the 17th International Conference on Computational Processing of Portuguese
(PROPOR 2026) – Vol. 1, pages 1–10, Salvador, Brazil, 2026. Association for Computational Linguistics.
[29] N. Abadani, J. Mozafari, A. Fatemi, M. Nematbakhsh, and A. Kazemi. Parsquad: Persian question answering dataset
based on machine translation of squad 2.0.International Journal of Web Research, 4(1):34–46, 2021.
Karimi et al.:Preprint submitted to ElsevierPage 22 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
[30] Kasra Darvishi, Newsha Shahbodaghkhan, Zahra Abbasiantaeb, and Saeedeh Momtazi. Pquad: A persian question
answering dataset.Computer Speech & Language, 80:101486, 2023.
[31] ArefehKazemi,JamshidMozafari,andMohammadAliNematbakhsh. Persianquad:Thenativequestionansweringdataset
for the persian language.IEEE Access, 10:26045–26057, 2022.
[32] Omid Ghahroodi et al. Khayyam challenge (persianmmlu): Is your llm truly wise to the persian language?arXiv preprint
arXiv:2404.06644, 2024.
[33] MCINext. Permmlu benchmark: Persian multiple-choice qa dataset, 2025.
[34] Morteza Alikhani et al. Percor: Evaluating commonsense reasoning in persian via multiple-choice sentence completion. In
Proceedings of NAACL IJCNLP, pages 2205–2224, 2025.
[35] Edaalat Sara. Edaalat sara: Iranian legal knowledge platform, 2026. Accessed: July 29, 2026.
[36] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-embedding: Multi-linguality, multi-
functionality, multi-granularity text embeddings through self-knowledge distillation. InFindings of ACL 2024, pages
2318–2335, 2024.
[37] Bernard L. Welch. The generalisation of ‘student’s’ problem when several different population variances are involved.
Biometrika, 34(1–2):28–35, 1947.
[38] AlecRadford,JeffreyWu,RewonChild,DavidLuan,DarioAmodei,andIlyaSutskever. Languagemodelsareunsupervised
multitask learners.OpenAI Blog, 1(8):9, 2019.
[39] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang
Du, Chu Wei, et al. Kimi-vl technical report.arXiv preprint arXiv:2504.07491, 2025.
[40] Ellen M. Voorhees. The trec-8 question answering track report. InTREC, 1999.
[41] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques.ACM Transactions on
Information Systems, 20(4):422–446, 2002.
[42] Lianmin Zheng et al. Judging llm-as-a-judge with mt-bench and chatbot arena.arXiv preprint arXiv:2306.05685, 2023.
[43] DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.
[44] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond.Foundations and
Trends in Information Retrieval, 3(4):333–389, 2009.
[45] Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang,
Georgios Mastrapas, Andreas Koukounas, Nan Wang, et al. jina-embeddings-v3: Multilingual embeddings with task
lora.arXiv preprint arXiv:2409.10173, 2024.
[46] LiangWang,NanYang,XiaolongHuang,LinjunYang,RanganMajumder,andFuruWei. Multilinguale5textembeddings:
A technical report.arXiv preprint arXiv:2402.05672, 2024.
[47] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng
Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176, 2025.
[48] Erfan Zinvandi, Morteza Alikhani, Mehran Sarmadi, Zahra Pourbahman, Sepehr Arvin, Reza Kazemi, and Arash Amini.
Famteb: Massive text embedding benchmark in persian language.Preprint, 2025.
[49] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low,
AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card.arXiv preprint arXiv:2601.03267, 2025.
liographycas-refs
Appendix
A. Examples from TUSI-Bench
This appendix provides qualitative examples from the proposed TUSI-Bench evaluation suite. The
examples illustrate the structure and content of the two introduced benchmarks, TUSI-QA and TUSI-Bar.
Specifically, TUSI-QA is designed as a citation-grounded open-ended legal question answering benchmark,
where each instance connects a legal query with an answer supported by authoritative legal sources. TUSI-
Bar, on the other hand, evaluates legal reasoning ability through multiple-choice questions derived from
professional legal examination settings.
Figure A.1 presents an example instance from TUSI-QA, consisting of a statutory provision from the
Iranian legal system and its corresponding expert interpretation. Such paired representations enable the
evaluation of whether models can retrieve and reason over relevant legal texts beyond surface-level matching.
Figure A.2 shows examples of judicial precedents included in TUSI-QA. These examples demonstrate
the incorporation of Supreme Court unification judgments, which represent an important source of legal
reasoning and interpretation in the Iranian judicial system.
Figure A.3 provides examples of citation-grounded question answering instances. Each example includes
a legal question, the generated answer, and the supporting legal citation, allowing evaluation of both answer
correctness and the reliability of cited sources.
Karimi et al.:Preprint submitted to ElsevierPage 23 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Finally, Figure A.4 illustrates examples from TUSI-Bar. Each instance contains a multiple-choice legal
reasoning question, candidate answers, the correct option, and the corresponding legal reference. These
exampleshighlightthebenchmark’sfocusonevaluatingstructuredlegalreasoningandknowledgeapplication
in an exam-style setting.
Karimi et al.:Preprint submitted to ElsevierPage 24 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
 .اﺳت ﻧﺎﻓذ ﻧﺑﺎﺷد ﻗﺎﻧون ﺻرﯾﺢ ﻣﺧﺎﻟف ﮐﮫ ﺻورﺗﯽ در اﻧد ﻧﻣوده ﻣﻧﻌﻘد را آن ﮐﮫ ﮐﺳﺎﻧﯽ ﺑﮫ ﻧﺳﺑت ﺧﺻوﺻﯽ ﻗراردادھﺎیPrivate agreements are legally binding upon the parties who have entered into them, provided that they do not expressly contravene the law.
(a) Original legal article (Article 10 of the Iranian Civil Code).
.1 رﻋﺎﯾت و اﺳت اﻟزاﻣﯽ ﻣطﻠق طور ﺑﮫ ﻗواﻧﯾن ﺑرﺧﯽ رﻋﺎﯾت ﮐﮫ آن ﺗوﺿﯾﺢ ﺗﮑﻣﯾﻠﯽ؛ ﻧﮫ اﺳت، «اﻣری» ﻗواﻧﯾن ﺑر ﻧﺎظر ﻣﺎده اﻧﺗﮭﺎی در «ﻗﺎﻧون» ی واژه ﻗواﻧﯾن را دوم ی دﺳﺗﮫ و «اﻣری» ﻗواﻧﯾن را اول دﺳﺗﮫ ﺑﺎﺷد، ﻧﺷده ﺑﯾﻧﯽ ﭘﯾش ﻋرف ﯾﺎ ﻗرارداد در آن ﺧﻼف ﮐﮫ اﺳت اﻟزاﻣﯽ ﺻورﺗﯽ در ﻗواﻧﯾن ﺑرﺧﯽ .ﮔوﯾﻧد ﻣﯽ «ﺗﺧﯾﯾری ﯾﺎ ﺗﻌوﯾﺿﯽ ﯾﺎ ﺗﻔﺳﯾری ﯾﺎ ﺗﮑﻣﯾﻠﯽ».2 .اﺳت ﻗﺎﻧون ﺑودن اﻣری ﺑر اﺻل ﺷﺧﺻﯾﮫ اﺣوال در و ﻗﺎﻧون ﺑودن ﺗﮑﻣﯾﻠﯽ ﺑر اﺻل ﻣﺎﻟﯽ اﻣور در.3 ﻧظم ﺑرﺧﻼف و ﺑﺎﺷد ﺷده رﻋﺎﯾت آﻧﮭﺎ در ﻗراردادھﺎ ﻋﻣوﻣﯽ ﺷراﯾط ﮐﮫ ﺻورﺗﯽ در ﻗراردادھﺎ ﮐﻠﯾﮫ ﻗراردادھﺎ، آزادی و اراده ﺣﺎﮐﻣﯾت اﺻل ﺑﮫ ﻧظر  .ﺑود ﺧواھد اﻻﺗﺑﺎع ﻻزم و ﻣﻌﺗﺑر ﻧﺑﺎﺷد( آﻣره ﻗواﻋد )ﻋﻣوﻣﯽ.4 ﻘدـﻋ ﻌﻘﺎدـاﻧ ﻠﮫـرﺣـﻣ در مـھ ﮐﮫ تـاﺳ اﺻﻠﯽ ﺎ،ـراردادھـﻗ آزادی لـاﺻ. تـﺎﺳـراردادھـﻗ ﻘوقـﺣ در اراده ﺎﮐﻣﯾتـﺣ لـاﺻ ﮭمـﻣ ﺗﺎﯾﺞـﻧ از ﯾﮑﯽ ﺎـراردادھـﻗ آزادی لـاﺻ در رارداد،ـﻗ رفـط زﯾﻧشـﮔ و ﺗﺧﺎبـاﻧ در رارداد،ـﻗ ﻌﻘﺎدـاﻧ دمـﻋ ﯾﺎ ﻌﻘﺎدـاﻧ در ﮫـآزاداﻧ ﻧدـواﻧـﺗ ﻣﯽ راردادـﻗ رﻓﯾنـط ذاـﻟ. راردادـﻗ رایـاﺟ ﻠﮫـرﺣـﻣ در مـھ و دارد رـاﺛ ونـﺎﻧـﻗ ﺎـﺑ رﯾﺢـﺻ ﻔتـﺧﺎﻟـﻣ ﮐﻧد، ﻣﯽ رﺳﯾمـﺗ راردادیـﻗ آزادی رایـﺑ دﻧﯽـﻣ ونـﺎﻧـﻗ 10 ﺎدهـﻣ ﮐﮫ زیـرﻣـﻗ طـﺧ. دـﺑﮕﯾرﻧ ﺗﺻﻣﯾم راردادـﻗ ﺿﻣونـﻣ و راﯾطـﺷ ﺗﻌﯾﯾن واﻧﯾنـﻗ ﺎـﺑ ﻔتـﺧﺎﻟـﻣ دﻟﯾل ﮫـﺑ راردادـﻗ ﻌضـﺑ ﯾﺎ ﮐل ورتـﺻ اﯾن در د،ـﺎﺷـﺑ رﯾﺢـﺻ ﻔتـﺧﺎﻟـﻣ در ریـاﻣ واﻧﯾنـﻗ ﺎـﺑ آن روطـﺷ از رﺧﯽـﺑ ﯾﺎ راردادـﻗ ﭼﮫـﻧﺎﻧـﭼ ذاـﻟ. تـاﺳ .اﺳت ﺑﺎطل.5 رﯾﺢـﺻ ﮫـﭼ )دـﺎﺷـﺑ ﻧﯾز ونـﺎﻧـﻗ ﺿﻣﻧﯽ فـﺧﺎﻟـﻣ رﻓﯾنـط قـواﻓـﺗ رـاﮔ ﮐﮫ ﻣﻌﻧﯽ دﯾنـﺑ! وﻋﯾتـوﺿـﻣ ﮫـﻧ دارد رﯾﻘﯾتـط دﻧﯽ،ـﻣ ونـﺎﻧـﻗ ﺗﮭﺎﯾﯽـاﻧ ﺧشـﺑ در رﯾﺢـﺻ ی واژه.اﺳت ﺑﺎطل( ﺿﻣﻧﯽ ﭼﮫ1.The term "law" in the final part of the provision refers specifically to mandatory rules rather than supplementary or default rules. This distinction arises from the fact that some legal rules are absolutely binding and cannot be modified by private agreement, whereas others apply only in the absence of an alternative arrangement established by contract or customary practice. The former category constitutes mandatory rules, while the latter includes supplementary, interpretative, optional, or default rules. 2.In financial and property-related matters, the general presumption is that legal provisions are supplementary, allowing parties to deviate from them through agreement. Conversely, in matters concerning personal status, the presumption is that legal provisions are mandatory and cannot generally be overridden by private arrangements. 3.Based on the principle of party autonomy and freedom of contract, agreements are considered valid and enforceable when the general requirements for contractual validity are satisfied and when their provisions do not violate public order or mandatory legal rules. Accordingly, private parties are generally free to regulate their legal relationships within the limits established by mandatory provisions of law. 4.The principle of freedom of contract is one of the fundamental consequences of party autonomy in contract law. This principle operates both at the stage of contract formation and during contract performance. Accordingly, parties are free to decide whether to enter into a contract, select their contractual counterpart, and determine the terms, conditions, and substance of their agreement. The limitation imposed by Article 10 of the Civil Code on contractual freedom is the express violation of law. Therefore, where a contract or any of its clauses expressly conflicts with mandatory legal provisions, the entire contract or the relevant contractual provision may be considered invalid due to such violation. 5.The term "expressly" in the final part of Article 10 has a procedural rather than substantive function. In other words, the invalidity of an agreement is not limited only to cases where the conflict with law is explicit. If the agreement is inconsistent with mandatory legal principles even implicitly, whether through an express or implied violation, it may still be deemed invalid.
(b) Corresponding expert interpretation of Article 10.
Figure A.1:An example instance from TUSI-QA, consisting of a statutory article and its corresponding expert
interpretation.
Karimi et al.:Preprint submitted to ElsevierPage 25 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
ایران مدنی قانون در غیرمتعارف قیمت با اموال فروش در وکالت های محدودیت خصوص در کشور عالی دیوان عمومی هیات ۷۴۸ شماره رویه وحدت رای همان ۷۶۶ ماده مطابق اما. است مجاز وکالت طریق از دیگری مال به نسبت معامله انجام مدنی، قانون ۷۴۲ ماده طبق: آن اساس بر که باشد می اختیار وکیل هرگاه بنابراین،. نکند تجاوز شده داده اختیارات از و کند رعایت را موکل مصلحت خود اقدامات و تصرفات در است موظف وکیل قانون، ًافرع که )غیرمتعارف و پایین بسیار قیمتی به را مال وکیل اگر. است متعارف قیمت معنای به عبارت این باشد، داشته" قیمت هر به "را موکل مال فروش است باطل معامله موکل، تنفیذ بدون و شده محسوب فضولی وی اقدام بفروشد، دیگری یا خود به( باشد قبول قابل غیر و بخس ثمن.Unification Judgment No. 847 of the General Assembly of the Supreme Court of Iran addresses the limitations on an attorney’s authority to sell a principal’s property at an unreasonable price under Iranian civil law.According to this judgment, although Article 247 of the Iranian Civil Code permits transactions involving another person's property through legal representation (agency), Article 667 of the same Code requires the agent to act in accordance with the interests of the principal and not exceed the scope of the granted authority.Therefore, when an agent is authorized to sell the principal’s property “at any price,” this expression shall be interpreted as referring to a reasonable and customary market price. If the agent sells the property at a significantly low and unreasonable price (considered an unacceptable or nominal price according to customary standards) either to themselves or to another person, such action constitutes an unauthorized transaction and the sale is invalid unless subsequently approved by the principal.
(a) Summary of Unification Judgment No. 847 of the General Assembly of the Supreme Court of Iran, issued
on May 14, 2024.
که باشد می قراردادها اعتبار و بانکی سود تنظیم در مرکزی بانک آمره نقش خصوص در کشور عالی دیوان عمومی هیات ۴۹۷ شماره رویه  وحدت  رای مصوب ربا بدون بانکی عملیات قانون ۰۲ ماده و ۱۵۳۱/۴۰/۸۱ مصوب کشور بانکی و پولی قانون ۷۳ و ۴۱ ،۱۱ ،۰۱ مواد اساس بر: آن اساس بر سهم حداکثر و حداقل به راجع مذکور بانک مصوبات و بوده کشور اعتباری و پولی نظام کننده تنظیم ایران اسالمی جمهوری مرکزی بانک ،۲۶۳۱/۶۰/۸۰ شوند تنظیم مرکزی بانک مقررات با مطابق باید بانکی قراردادهای همچنین،. دارد آمره جنبه غیردولتی و دولتی از اعم اعتباری موسسات و ها بانک سود است باطل باشد، مقررات این خالف بر که تحمیلی شرط هرگونه و.Unification Judgment No. 794 of the General Assembly of the Supreme Court of Iran concerns the mandatory regulatory authority of the Central Bank of Iran in determining banking profit rates and its impact on the validity of banking contracts.According to this judgment, based on Articles 10, 11, 14, and 37 of the Monetary and Banking Act of Iran, enacted on 8 July 1972, as well as Article 20 of the Law on Usury-Free Banking Operations, enacted on 30 August 1983, the Central Bank of the Islamic Republic of Iran is responsible for regulating the country’s monetary and credit system.The regulations and resolutions issued by the Central Bank regarding the minimum and maximum profit shares applicable to banks and credit institutions, whether governmental or private, have a mandatory (binding) legal nature.Furthermore, banking contracts must be concluded in accordance with the regulations and directives issued by the Central Bank. Any imposed contractual condition that contradicts these mandatory regulations is considered invalid and unenforceable.
(b) Summary of Unification Judgment No. 794 of the General Assembly of the Supreme Court of Iran, issued
on August 11, 2020.
Figure A.2:Examples of judicial precedents included in TUSI-QA: Unification Judgments No. 847 and No. 794 of the
General Assembly of the Supreme Court of Iran. Each example presents a concise summary of the corresponding judicial
decision.
Karimi et al.:Preprint submitted to ElsevierPage 26 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
Related Legal ArticlesArticle IDsLegal Question
Legal AnswerCitation/ Grounding       امالک مشاور در شهرستان مسکن ودیعه وام اخذ برای گذشته سال اینجانب وتحیت سالم با از معذرت بسرآمد،باوعرض آن تاریخ وام دریافت از قبل لکن نمودم ثبت رهگیری کد با قولنامه یک نمود؟با تمدید بنگاه مهر با مندرج شرایط همان با را قولنامه همان میشود آیا اینه سئوالم مزاحمت تشکرHello. Last year, in order to obtain a housing deposit loan in my city, I registered a private sale agreement with a tracking code at a real estate agency. However, its validity period expired before I received the loan.I apologize for the inconvenience. My question is whether the same agreement can be extended with the same conditions and with the official stamp of the real estate agency?Thank you.        ندارد منعی مشروع توافق هرگونه مدنی قانون 10 ماده مطابقAccording to Article 10 of the Iranian Civil Code, any lawful agreement between individuals is permitted and valid.10000010Article 10 of the Civil Codeﻣﺪٮ"ﯽ نو#ٮﺎ&ڡ  ۰۱ ﻣﺎده
(a) Example instance 1 from TUSI-QA.
        واقعی قیمت ولی. تومن میلیون هشتصدوپنجاه قیمت با بخرم ای خونه میخوام. سالم خانه واقعی قیمت ندانسنت بعلت خریدخانه ایا. است بیشتر تومن صدمیلیون تا پنجاه حدودا بخرد من قیمت از باالتر قیمت به را خانه که نداره رو کسی فروشنده اینکه و فروشنده توسط چون. نمیگیرد صورت معامالتی معموال بگن فروشنده به رو واقعی قیمت همه اگه چیست؟ حکمش است؟ جایز مشتری باالی سود بخاطر خانه این خرید حکم میکنن؟ معامله سود برای همهHello. I intend to purchase a house for 850 million tomans, but the actual market value of the property is approximately 50 to 100 million tomans higher. What is the legal and religious ruling regarding purchasing the house when the seller is unaware of its actual value and does not have anyone willing to buy it at a higher price?If everyone informs the seller of the actual market value, transactions would usually not take place, since everyone engages in transactions to gain profit. Is purchasing this house in order to obtain a significant profit considered permissible?        و ندارد معنی هیچ شرع نظر از نه و قانون نظر از نه ، باالتر قیمت با خانه خرید سالم با جایز و معتبر ، اشخاص بین خصوصی توافق ، مدنی قانون 10 ماده طبق است صحیح کامال استHello. Purchasing a house at a higher or lower price has no legal or religious restriction and is completely valid. According to Article 10 of the Iranian Civil Code, private agreements between individuals are legally valid and permissible, provided that they do not violate the law.Related Legal ArticlesArticle IDsﻣﺪٮ"ﯽ نو#ٮﺎ&ڡ  ۰۱ ﻣﺎدهLegal Question
Legal Answer
Citation/ Grounding10000010Article 10 of the Civil Code
(b) Example instance 2 from TUSI-QA.
Figure A.3:Examples of citation-grounded legal question answering instances in TUSI-QA. Each instance contains a legal
question, the corresponding answer, and the referenced legal article providing the supporting citation.
Karimi et al.:Preprint submitted to ElsevierPage 27 of 24

TUSI: A Citation-Grounded Graph-Based Benchmark
تابعیت حال عین در که کشور آن مقیم ایرانی زرتشتی شوهر و زن درخواست به سوئیس دادگاه و زرتشتی دین موازین و مقررات رعایت با طالق و میکند صادر طالق حکم دارند هم سوئیسی کدام ایران قوانین، لحاظ از حکم این اجرای و اعتبار خصوص در شود می ثبت سوئیس قوانین است؟ صحیح موردصالحیتدار دادگاه و زرتشتی عالی مراجع توسط تنفیذ و بررسی نیازمند طالق، حکم( الف است ایرانیاست ایرانی صالحیتدار دادگاه توسط تنفیذ و بررسی نیازمند صادره، طالق حکم( باست زرتشتی عالی مراجع توسط تنفیذ و بررسی نیازمند طالق، حکم( جاست اجرا قابل و دارد اعتبار ایران در طالق، حکم( دA Swiss court issues a divorce judgment at the request of an Iranian Zoroastrian husband and wife residing in Switzerland, who also hold Swiss citizenship. The divorce is registered in accordance with Zoroastrian religious rules and Swiss legal regulations. Regarding the validity and enforceability of this judgment under Iranian law, which option is correct? Options: A) The divorce judgment requires review and approval by the supreme Zoroastrian authorities as well as the competent Iranian court. B) The issued divorce judgment requires review and recognition by the competent Iranian court. C) The divorce judgment requires review and approval only by the supreme Zoroastrian authorities. D) The divorce judgment is valid in Iran and can be directly enforced.Option BIranian Family Protection Act (2013)Multiple-Choice Question
Legal AnswerCitation/ Groundingب گزینه ١٣٩١/١٢/١ ﻣﺻوب ﺧﺎﻧواده ﺣﻣﺎﯾت ﻗﺎﻧون
(a) Example of a TUSI-Bar multiple-choice question regarding the recognition of a foreign divorce judgment
and its related legal reference.
مدنی مسئولیت قانون لحاظ با باشد مثلی، مال آن و شود مالی تلف سبب شخصی چنانچه داد؟ خواهد خسارت جبران به حکم نحوی چه به دادگاهکند مثل دادن به محکوم را زبان کننده وارد باید ًاامزلا دادگاه شود یافت، مال آن مثل اگر(  الفشده تلف مال مثل اگر حتی کند قیمت، دادن به محکوم را زیان کننده وارد میتواند دادگاه(  ب شود یافتشده تلف مال مثل اینکه مگر کند مثل، دادن به محکوم را زیان کننده وارد است ملزم دادگاه( ج نباشد موجوددادگاه تصمیم با زیان کننده وارد اینکه بر مشروط دهد قیمت پرداخت به حکم میتواند دادگاه( د باشد موافقIf a person causes the destruction of another person's property and the property is fungible, according to the Civil Liability Law, how should the court order compensation for the damage? Options: A) If an equivalent item exists, the court must order the person causing the damage to provide an equivalent item. B) The court may order the person causing the damage to pay the value of the property, even if an equivalent item is available. C) The court must order the person causing the damage to provide an equivalent item unless no equivalent item exists. D) The court may order payment of the property's value only if the person causing the damage agrees with the court’s decision. Option BCivil Liability LawMultiple-Choice Question
Legal AnswerCitation/ Groundingب گزینهمدنی مسیولیت قانون
(b) Example of a TUSI-Bar multiple-choice question regarding compensation for destroyed fungible property.
Figure A.4:Illustrative examples from TUSI-Bar. Each example consists of a multiple-choice legal question, four candidate
answers, the ground-truth answer, and the associated legal source for citation-based legal reasoning.
Karimi et al.:Preprint submitted to ElsevierPage 28 of 24
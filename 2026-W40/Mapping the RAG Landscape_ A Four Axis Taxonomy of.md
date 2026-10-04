# Mapping the RAG Landscape: A Four Axis Taxonomy of Efficiency, Defense, Interactivity, and Reasoning

**Authors**: Meghana Sunil, Shravya V, Shravan Venkatraman, Joe Dhanith PR

**Published**: 2026-10-01 16:06:05

**PDF URL**: [https://arxiv.org/pdf/2610.01936v1](https://arxiv.org/pdf/2610.01936v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable fluency across many tasks but remain limited by their static, parameter bound knowledge and their susceptibility to hallucinating information. Retrieval Augmented Generation (RAG) addresses these issues by incorporating external retrieval into the generation process, grounding model outputs in verifiable and up to date sources. While prior surveys primarily focus on core RAG architectures and standard pipelines, recent research explores broader challenges and capabilities that extend beyond these foundational designs. This survey provides a consolidated and structured examination of contemporary RAG developments, organizing the field into a four axis taxonomy: improving retrieval efficiency, strengthening robustness and security, supporting user driven and interactive workflows, and enabling multi step or complex reasoning. We formalize key components of the RAG framework and review methods spanning dense and sparse retrieval, fusion strategies, embedding optimizations, and reinforcement learning based retrieval policies, highlighting how these advances influence practical deployment and system design. We also synthesize evaluation practices, domain specific applications, and architectural variants such as Naive, Advanced, and Modular RAG. Finally, we outline persistent challenges related to retrieval quality, reliability, domain adaptation, scalability, and explainability, and identify opportunities for building RAG systems that are more reliable, adaptable, and transparent.

## Full Text


<!-- PDF content starts -->

1
Mapping the RAG Landscape: A Four-Axis Taxonomy
of Efficiency, Defense, Interactivity, and Reasoning
Meghana Sunil1†, Shravya V1†, Shravan Venkatraman2, Joe Dhanith P R1∗
Abstract—Large Language Models (LLMs) have demonstrated remarkable fluency across many tasks but remain limited by their
static, parameter-bound knowledge and their susceptibility to hallucinating information. Retrieval-Augmented Generation (RAG)
addresses these issues by incorporating external retrieval into the generation process, grounding model outputs in verifiable and
up-to-date sources. While prior surveys primarily focus on core RAG architectures and standard pipelines, recent research explores
broader challenges and capabilities that extend beyond these foundational designs. This survey provides a consolidated and structured
examination of contemporary RAG developments, organizing the field into a four-axis taxonomy: improving retrieval efficiency,
strengthening robustness and security, supporting user-driven and interactive workflows, and enabling multi-step or complex reasoning.
We formalize key components of the RAG framework and review methods spanning dense and sparse retrieval, fusion strategies,
embedding optimizations, and reinforcement-learning–based retrieval policies, highlighting how these advances influence practical
deployment and system design. We also synthesize evaluation practices, domain-specific applications, and architectural variants such
as Naive, Advanced, and Modular RAG. Finally, we outline persistent challenges related to retrieval quality, reliability, domain
adaptation, scalability, and explainability, and identify opportunities for building RAG systems that are more reliable, adaptable, and
transparent.
Index Terms—Retrieval-Augmented Generation, Large Language Models, Information Retrieval, Contextual Generation, Semantic
Search
✦
1 Introduction
LargeLanguage Models (LLMs) [1, 2, 3, 4, 5, 6, 7]
have achieved remarkable progress in natural language
generation, supporting applications ranging from conversa-
tional systems to complex summarization and question an-
swering [7, 8]. Despite this progress, a longstanding limitation
of these models lies in their reliance on static, pre-trained
parameters, which restricts their ability to incorporate new or
domain-specific information. As a result, LLMs may produce
inaccurate or hallucinated content, particularly in settings
involving sparse inputs or attribute-heavy queries [9]. These
issues pose significant challenges in knowledge-intensive tasks
such as review generation [10, 11, 12, 13], dialogue systems
[14, 15, 16, 17], and data-to-text (D2T) generation [18,
19, 20], where factual correctness and contextual relevance
are essential [21]. Retrieval-Augmented Generation (RAG)
addresses these challenges by coupling LLMs with external
retrieval systems that provide access to relevant, verifiable
information at inference time [22, 23, 24]. Instead of depend-
ing solely on internal parametric knowledge, RAG models
dynamicallyretrievesemanticallyrelevantcontentfromstruc-
tured or unstructured corpora such as relational databases,
RDF graphs, or large-scale text collections [25, 26, 27]. This
retrieval-grounded formulation enables more accurate gen-
•1School of Computer Science and Engineering, Vellore Institute
of Technology, Chennai, India.
•2Mohamed bin Zayed University of Artificial Intelligence, Abu
Dhabi, UAE.
•∗Corresponding author(s). E-mail(s): joedhanith.pr@vit.ac.in
•Contributing authors: meghana.sunil2023@vitstudent.ac.in;
shravya.v2023@vitstudent.ac.in;
shravan.venkatraman@mbzuai.ac.ae
•†These authors contributed equally to this work.eration, supports rapid adaptation to new information, and
improves model transparency.
This retrieval-capable architecture also allows RAG sys-
tems to support personalized or user-specific workflows [28,
29, 30], such as product reviews, recommendation systems,
and user-facing assistants, as well as high-stakes domains
such as medical diagnosis [31, 32, 33], education [34], and
customer support where factual grounding is critical [35].
Reinforcement learning for LLMs [36, 37, 38] has further been
used to optimize retrieval strategies dynamically, improving
relevance and efficiency in real-world deployments [39]. We
review the formal structure of RAG and its architectural
variants-Naïve, Advanced, and Modular RAG-together with
their evaluation practices in §3.1.
Figure 1 summarizes the four core dimensions emphasized
in recent RAG research-efficiency, security, interactivity, and
complex reasoning-which reflect emerging priorities as RAG
transitions from experimental prototypes to widely deployed
systems and is increasingly integrated into training pipelines
and domain-adapted workflows [40]. This survey organizes
recent advancements along these four themes-compression
and efficiency [41], defensive RAG [40, 39], interactive and
user-centric RAG [27, 26], and complex reasoning [42, 21, 43]-
each tied to a central research question (RQ1-RQ4 below)
and characterized by representative contributions in retrieval
scoring, context construction, safety-aware filtering, personal-
ization, and multi-step reasoning.
These axes were selected because they represent emerg-
ing, orthogonal, and under-reviewed dimensions of the RAG
ecosystem-dimensions that extend beyond traditional tax-
onomies centered solely on retrieval or generation pipelines.
Previous surveys have primarily focused on dense-versus-
sparse retrieval comparisons, broad architectural overviews,
arXiv:2610.01936v1  [cs.AI]  1 Oct 2026

2
Fig. 1: Overview of the four taxonomic dimensions of RAG
systems explored in this survey:Compression and Efficiency,
Defensive RAG,User-Centric Interaction, andComplex Rea-
soning. These axes represent key directions in optimizing
performance, improving security and fairness, aligning with
user intent, and supporting structured reasoning.
or generic RAG workflows, leaving deeper conceptual dis-
tinctions underexplored. Our perspective differs by empha-
sizing the theoretical motivations, algorithmic shifts, and
design trade-offs that uniquely characterize each axis. Rather
than treating RAG as a monolithic augmentation strategy,
we highlight concerns such as retrieval–generation coupling,
personalization constraints, safety-aware filtering, and multi-
step reasoning pipelines-areas that have expanded rapidly
yet remain dispersed across the literature. By consolidating
these developments into four coherent axes, this survey of-
fers a structured and forward-looking understanding of how
retrieval-augmented systems are evolving and where they
diverge from traditional LLM enhancement strategies.
To provide a concrete and answerable structure, we orga-
nize this review around four research questions that reflect
central, unresolved tensions in the RAG literature:
RQ1 (Efficiency):How can RAG systems maintain high
factualaccuracywhileoperatingundertheinference-time
constraints imposed by large-scale and heterogeneous
corpora-and which retrieval and compression strategies
most effectively navigate the accuracy–cost trade-off?
RQ2 (Defense):What principal vulnerabilities does exter-
nal retrieval introduce into LLM pipelines-including cor-
pus poisoning, demographic bias amplification, and pri-
vacy leakage-and how effectively do current defensive
architectures mitigate them?
RQ3 (Interactivity):To what extent can RAG systems
adapt to evolving user intent, interaction history, and
preference signals-and what mechanisms best balance
personalization with factual precision?
RQ4 (Reasoning):How does integrating retrieval into iter-
ative, step-conditioned reasoning pipelines improve per-formance on multi-hop and knowledge-intensive tasks
compared to single-pass retrieval-and what architectural
properties prevent error accumulation across reasoning
steps?
These questions are neither fully solved nor fully open: §3
surveys what the literature has achieved on each, §5 maps
where each question remains unresolved, and §6 identifies the
research directions most likely to produce progress. Together,
they provide a lens for evaluating whether a given system
advances the field along a specific, meaningful dimension
ratherthantreatingRAGasanundifferentiatedbodyofwork.
2 Survey Methodology
To ensure methodological transparency and reproducibility,
we adopt a structured literature review protocol inspired
by Snyder’s framework for systematic and semi-systematic
reviews [44] and aligned with the PRISMA 2020 principles.1
Our goal was to capture developments in RAG across the
rapid expansion period from January 2020 to January 2025.
Wequeriedthreemajorscholarlyrepositories-GoogleScholar,
Semantic Scholar, and arXiv-using both broad and domain-
targeted keywords such as “retrieval-augmented genera-
tion”, “RAG LLM”, “RAG pipeline”, “RAG benchmark”,
“RAG evaluation”, “RAG defense”, “RAG fairness”, “multi-
hop retrieval LLM”, “RAG hallucination”, and “knowledge-
grounded generation”. These searches yielded approximately
780 records across peer-reviewed venues, preprints, and work-
shop papers, forming the identification stage of our PRISMA-
inspired workflow.
Screeningproceededunderexplicitinclusionandexclusion
criteria designed to ensure consistency across the surveyed
literature. We included works that (i) integrate retrieval and
generation within a unified pipeline, (ii) introduce method-
ological, architectural, or empirical innovations relevant to
RAG, (iii) propose RAG-specific benchmarks or evaluation
frameworks, or (iv) investigate security, robustness, fairness,
or safety aspects of retrieval-augmented LLMs. We excluded
works that (i) lack a retrieval component, (ii) describe stan-
dalone retrievers or standalone LLMs without integration, or
(iii) offer no technical or empirical contribution. The resulting
corpus forms the evidence base synthesized throughout this
survey and directly supports the four-axis taxonomy intro-
duced in later sections. This ensures that the review reflects
both breadth and depth, provides balanced coverage across
research directions, and maintains alignment with established
systematic review standards.
To further control for completeness, we supplemented the
keyword-based search with a targeted citation-snowballing
pass. Starting from one representative paper per taxonomy
axis plus the foundational RAG paper (Lewis et al., 2020 [45],
for the field as a whole; RAPTOR [46] for compression and
efficiency; SELF-RAG [47] for defensive RAG; Chat-REC [48]
for interactive, user-centric RAG; and GenGround [49] for
complex multi-step reasoning), we traced citations in both
directions, examining each seed paper’s own references (back-
ward) and the works that cite it (forward). This pass was
scoped as a targeted completeness check on the four axes
rather than a re-execution of the full identification stage,
1. https://www.prisma-statement.org/

3
and it surfaced several additional candidate papers directly
relevant to the corresponding axis, including HippoRAG [50]
(compression and efficiency), RuleRAG [51] (defensive RAG),
MemoCRS [52] (interactive, user-centric RAG), and recent
iterative multi-hopreasoning workextending GenGround[53]
(complex reasoning).
3 Taxonomy of RAG-LLM Architectures
3.1 Conceptual Overview
RAG is a hybrid framework that enhances the capabilities
of Large Language Models (LLMs) by integrating real-time
information retrieval into the text generation process. Unlike
conventional LLMs that rely entirely on static, paramet-
ric knowledge learned during pre-training, RAG introduces
an external retrieval step that surfaces relevant, up-to-date
information from knowledge bases, databases, or document
repositories. This augmentation allows the model to pro-
duce more reliable and contextually grounded outputs while
mitigating hallucinations that arise when the model lacks
sufficient internal knowledge. The dynamic nature of retrieval
enables RAG systems to remain responsive in domains where
information evolves rapidly, making them especially effective
in knowledge-intensive or time-sensitive applications.
The RAG framework operates in two principal stages.
During the retrieval stage, the system identifies documents
or passages that are relevant to an input query using methods
such as Dense Passage Retrieval (DPR), which leverages neu-
ral embeddings for semantic matching, or sparse techniques
like BM25, which use term-frequency and inverse-document-
frequency scoring. Following retrieval, the generation stage
conditions a pre-trained language model, such as GPT or T5,
on the combined query and retrieved content to synthesize
a coherent and factually grounded response. This fusion step
ensuresthatgeneratedtextreflectsboththemodel’sparamet-
ric knowledge and the contextual evidence provided through
retrieval.
A key advantage of RAG lies in its adaptability. Tra-
ditional LLMs require costly retraining to incorporate new
information, whereas RAG systems can integrate updates at
inference time simply by retrieving from refreshed or domain-
specific corpora, improving factual accuracy in domains with
frequent updates (e.g., medicine, finance, law). This design
also improves interpretability, as users can trace generated
statementsbacktosupportingevidenceratherthananopaque
parametric prior, creating a more transparent generation
processwellsuitedtoknowledge-intensive,time-sensitive,and
high-stakes applications.
3.2 Theoretical Background
By separating knowledge retrieval from parametric genera-
tion, RAG systems overcome constraints imposed by static
training pipelines, remaining responsive in fields such as
healthcare, legal reasoning, and technical support where ac-
curacy and specificity are critical. This conceptual shift lays
the groundwork for hybrid systems that match the fluency
of LLMs with the factual grounding of information retrieval,
formalized below.
The retrieval component in RAG plays a central role in
enabling this hybrid functionality. Rather than generating
Fig. 2: Overview of the Retrieval-Augmented Generation
(RAG) pipeline. The workflow is divided into two phases:
(1) offline indexing, where source documents are chunked, en-
codedintodenseembeddings,andstoredinavectordatabase;
and(2)onlineretrievalandgeneration,wheretheuserqueryis
embeddedandusedtoretrievethetop-ksemanticallyrelevant
document chunks via vector similarity search. The retrieved
context is combined with the original query and supplied
to the large language model (LLM), enabling grounded and
context-aware response generation.
responses solely from internal representations, RAG systems
activelysearchforrelevantdocumentsfromlarge-scaleknowl-
edge bases, web corpora, or structured datasets. Retrieval
methods [54] typically fall into two categories. Sparse re-
trieval techniques such as BM25 rely on term-frequency and
inverse-document-frequency scoring, making them effective
for keyword-driven matching. Dense retrieval techniques such
as Dense Passage Retrieval (DPR) use neural embeddings to
capture semantic similarity between queries and documents,
enabling retrieval even when wording differs. The retrieved
context is subsequently fused with the user query during
generation, allowing the model to produce responses that are
both grounded and explainable. This integration ensures that
retrieved knowledge directly influences the generative process
rather than serving as an auxiliary signal.
This hybridization of parametric and non-parametric
knowledge addresses key shortcomings of standard LLMs,
particularly their inability to access or incorporate new in-
formation after pre-training. While parametric models like
GPT provide fluent, general-purpose generation, they are
inherently static and cannot adapt without retraining. RAG
systems overcome this rigidity by retrieving new informa-
tion on demand, reducing the need for continual model up-
dates and improving factual consistency. Emerging hybrid
approachesfurtherrefinethisbalancebycombiningthegener-
alization capabilities of parametric models with the precision
and relevance offered by retrieval-based augmentation. These
methods create a more flexible and adaptive architecture that
supports dynamic reasoning, domain-aware responses, and
improved robustness across diverse tasks. Figure 2 illustrates
the overall structure of a standard RAG pipeline, highlight-
ing how retrieval and generation interact to form grounded
responses.

4
In traditional parametric models, a language modelP θ(y|
x)is learned purely from training data, where:
Pθ(y|x) =Decoder(x;θ)(1)
Here,xis the input,yis the output, andθrepresents the
fixed parameters learned during training. This formulation
captures the standard setting where all knowledge required
for generation is internal to the model.
In Retrieval-Augmented Generation, this paradigm is
extended by incorporating an external retrieval mecha-
nism. Given a queryx, a set of relevant documentsD=
{d1,d2,...,dk}is retrieved from an external corpusCusing
a retriever functionR:
D=R(x,C)(2)
This retrieved context augments the generative process by
providing non-parametric evidence that supports or refines
the model’s output.
The generation probability is then reformulated as:
P(y|x) =/summationdisplay
d∈DP(y|x,d)·P(d|x)(3)
where:
•P(d|x)is the probability of retrieving documentdgiven
queryx.
•P(y|x,d)is the generation probability conditioned on
both the query and the retrieved document.
This decomposition explicitly models how retrieval influences
generation and provides an interpretable formulation for
grounding outputs in external knowledge.
Retrieval Models.Sparse and dense retrieval are two pri-
maryfamiliesofmethodsusedtoidentifyrelevantdocuments.
Sparse retrieval techniques such as BM25 employ TF-IDF-
based scoring and match queries to documents based on exact
or near-exact keyword overlap. The BM25 scoring function is
defined as:
Score BM25 (x,d) =/summationdisplay
t∈x∩dIDF(t)f(t,d) (k+ 1)
f(t,d) +k/parenleftig
1−b+b|d|
avgdl/parenrightig
(4)
wheref(t,d)is the frequency of termtin documentd, andk,
bare hyperparameters controlling term frequency scaling and
document length normalization.
Dense retrieval techniques such as DPR compute similar-
ity in a learned embedding space:
Score DPR(x,d) =ϕ q(x)⊤ϕd(d)(5)
whereϕqandϕdare neural encoders for queries and docu-
ments, respectively, enabling semantic matching even in the
absence of exact term overlap.
The retrieval distribution is typically normalized using a
softmax:
P(d|x) =exp(Score(x,d))/summationtext
d′∈Dexp(Score(x,d′))(6)
This converts raw similarity scores into a proper probability
distribution over the retrieved set. As introduced in the orig-
inal RAG framework [45], this probabilistic treatment allows
the generator to marginalize over all retrieved documents-
weighting each document’s contribution by its retrieval score-
rather than conditioning generation on a single top-rankedAlgorithm 1RAG Pipeline
Require:Input queryx, corpusC, embedding modelϕ,
retrieverR, generatorG, top-kparameterk
Ensure:Generated responseˆy
1://Offline: Index Construction
2:for alld i∈Cdo
3:vi←ϕd(di)▷Encode documents
4:endfor
5:Store{v i}in vector indexI
6://Online: Inference Pipeline
7:q←ϕq(x)▷Encode query
8:D←R(q,I)▷Retrieve top-kdocs
9:C←concat(x,D)▷Combine query and retrieved
context
10:ˆy←G(C)▷Generate grounded output
11:returnˆy
passage. The resulting distribution directly feeds into Equa-
tion (3), whereP(d|x)governs how each retrieved docu-
ment’sgeneratedprobabilityiscombinedintothefinaloutput
distribution.
This softmax-marginalization treatment is a specific mod-
elingchoice,notauniversalrequirementofRAGsystems,and
it is worth being explicit about what it assumes and what it
costs. First, the normalization in Eq. (6) is computed only
over the top-kretrieved setD, not the full corpusC:P(d|x)
is a distribution over thekdocuments the retriever has
alreadyselected,notacalibratedestimateofrelevanceoverall
possible evidence, so the formulation inherits whatever recall
errors occurred upstream inR(x,C). Second, this is the RAG-
Sequence/RAG-Tokenmarginalizationoriginallyproposedby
Lewis et al. [45], which treats each retrieved document as
conditionallyindependentgiventhequery-anassumptionthat
simplifies inference but does not allow the model to reason
jointly across documents the way a single concatenated con-
text would. Not all systems in this survey adopt it: several ar-
chitectures discussed in §3 instead condition deterministically
on a fixed top-kconcatenation (treatingP(d|x)as implicitly
uniform or thresholded rather than softmax-weighted), or
replace the marginal in Eq. (6) entirely with learned re-
ranking or critique scores (e.g., Algorithm 3’s /tildewiderRel(q,d)). We
retain the softmax formulation here because it is the most
common closed-form baseline against which compression-,
defense-, interaction-, and reasoning-aware variants in later
sections can be compared, not because it is the only valid
choice.
The full retrieval-augmented inference process is summarized
in Algorithm 1, which outlines both offline index construction
and online document selection during generation.
This retrieval-augmented framework enables the system
to dynamically incorporate new information without retrain-
ing, improving factual grounding, explainability, and domain
adaptability. By separating retrieval from generation while
coupling them at inference time, RAG provides a flexible
architecturecapableofsupportingevolvingdomainsandhigh-
stakesapplications.Thistheoreticalgroundingformsthebasis
for the taxonomic analysis presented in subsequent sections.

5
TABLE 1: Comparison of prior RAG surveys and the contributions of this work.
Survey Year FocusArea WhatTheyCover WhatTheyDon’tCover What This Survey
Adds
Brehme et al. [55] 2025 RAG evaluation LLM judges, dataset
scoringIndexing and component-
wise evaluationFour-axis taxonomy (ef-
ficiency,defense,interac-
tivity,reasoning)beyond
evaluation methodology
alone
Hindi et al. [56] 2025 Legal-domain
RAGInterpretability, legal
datasetsGeneralizable evaluation
methodsDomain-independent
four-axis taxonomy,
not limited to a single
domain
Ni et al. [57] 2025 Trustworthy
RAGSafety, robustness, fair-
nessDataset/index/generation
evaluationExtends safety/robust-
nessfocuswiththreefur-
ther axes: efficiency, in-
teractivity, and reason-
ing
Zheng et al. [58] 2025 Vision-based
RAGMultimodal retrieval
and generationText-only methodological
evaluationFour-axis taxonomy for
text-based RAG, com-
plementing multimodal-
focused coverage
Oche et al. [59] 2025 Systematic RAG
reviewYear-wise progress, in-
dustry trendsLimited methodological
depthFormalizes each axis via
objective, algorithm,
and literature synthesis,
beyond trend-level
review
Singh et al. [26] 2025 Agentic RAG Planning, tool-use, au-
tonomous agentsEvaluation of core RAG
componentsClarifies boundary be-
tween agentic RAG and
this survey’s four-axis
architectural taxonomy
3.3 Historical Development
Early retrieval systems such as TF-IDF and BM25 enabled
large-scale lexical document search but were limited by
surface-level term matching, which hindered semantic flex-
ibility. This motivated the development of neural retrieval
methods-and eventually the tight coupling of retrieval with
generative language models-that provide deeper semantic
alignment and more flexible knowledge integration.
The introduction of neural retrieval models in the late
2010s marked a transformative shift toward more expressive
retrieval mechanisms. Dense retrieval systems such as DPR
employeddeepneuralnetworkstomapqueriesanddocuments
into a shared embedding space, enabling semantic similarity
computation via vector distance rather than relying solely
on lexical overlap. This innovation significantly improved
retrieval quality and made it possible to integrate external
evidence more effectively into downstream language genera-
tion tasks. The formalization of the RAG framework in 2020
by researchers at Facebook AI and OpenAI further advanced
this trajectory by unifying dense retrieval with generative
LLMs into a tightly coupled inference workflow. Unlike earlier
pipeline-style approaches, this unified architecture allowed
models to generate responses that were contextually coher-
ent and grounded in verifiable external knowledge, resulting
in substantial improvements on knowledge-intensive bench-
marks. More recently, long-context retrieval systems have ex-
tended this trajectory to entire documents and heterogeneous
knowledge sources, supporting multi-hop inference, dynamic
knowledge adaptation, and domain-specific reasoning.
3.4 Positioning Relative to Prior RAG Surveys
Recent surveys have examined RAG systems from a variety
of perspectives, yet each tends to focus on a specific subset
of challenges rather than providing a unified methodologicalanalysis. Brehme et al. [55] focus on evaluation methodolo-
gies, particularly the reliability of LLM-based judges and the
challenges associated with automated assessment of gener-
ated content. Hindi et al. [56] emphasize precision and in-
terpretability within legal-domain RAG, offering application-
specific insights but lacking generalizable evaluation frame-
works that extend beyond legal settings. Ni et al. [57] ap-
proach RAG from the standpoint of trustworthiness, dis-
cussing robustness, safety, and fairness but not addressing
dataset construction, indexing behavior, or generator-side
evaluation. Zheng et al. [58] extend RAG analysis into the
multimodal space by examining retrieval-augmented vision-
language systems, while Oche et al. [59] provide a broad sys-
tematic review that highlights year-wise progress and indus-
trial adoption trends. Finally, Singh et al. [26] survey agentic
RAG, emphasizing planning, tool use, and autonomous work-
flows in retrieval-augmented agents. However, none of these
surveys synthesize a unified evaluation perspective that spans
the core components of RAG pipelines, including datasets,
retrievers, indexing strategies, and generation modules.
Table 1 summarizes the distinctions between these prior
works and the focus of this survey. Existing literature pri-
marily centers on trust, domain specificity, multimodal rea-
soning, or agentic behavior, each typically treated as an
isolated concern rather than part of a broader organizing
structure. By contrast, this survey organizes RAG research
along four orthogonal axes-compression and efficiency, de-
fensive and safety-aware design, interactive and user-centric
personalization, and complex multi-step reasoning-each for-
malized through a dedicated objective, reference algorithm,
and synthesis of representative contributions. This structure
addressespracticalneedsforsystembuilderswhomustjointly
reason about retrieval cost, safety, personalization, and rea-
soning depth rather than treating these as disconnected de-
sign choices. Although agentic RAG represents an important

6
emerging direction [26], it is aligned with different goals-
autonomous planning and tool use-and remains orthogonal to
the architectural and methodological taxonomy developed in
thiswork.Instead,weconcentrateonthefoundationalmecha-
nisms that govern RAG behavior and shape the effectiveness,
safety, personalization, and reasoning capacity of deployed
retrieval-augmented systems. Concretely, the distinctive con-
tribution of this survey is not its choice of topics but its
organizing principle. Where prior surveys partition the field
either by a single concern (trust, domain, modality, or agentic
behavior)orbypipelinestage(retrievalversusgeneration),we
organize it by thedesign tension a system is built to optimize:
accuracy under cost (RQ1), robustness under adversarial or
biased evidence (RQ2), alignment with evolving user state
(RQ3), and fidelity across multi-step reasoning (RQ4). This
is the dimension along which a practitioner actually chooses,
and we make it comparable by pairing each axis with a formal
objective (Eq. (7)-(10)), a reference algorithm (Algorithms 2-
5), and the cross-axis method comparison in Table 2-a level of
structured, side-by-side analysis that the prose-driven organi-
zation of prior surveys does not provide.
3.5 Taxonomy Structure and Cross-Axis Comparison
This taxonomy is organized along two levels with explicit
branching criteria. At thefirst level, we partition the field
into four problem domains-compression and efficiency, defen-
sive RAG, interactive and user-centric RAG, and complex
reasoning-each corresponding to one of the research questions
RQ1-RQ4 introduced in §1. At thesecond level, we deliber-
ately do not branch by chronology; instead, within each axis
we group contributions intofamilies of solutions defined by
thesub-problemtheysolve(forefficiency,e.g.,utility-costopti-
mization, structural compression, domain-adaptive retrieval,
and evaluation tooling). The branching criterion at this level
is therefore themechanisma method uses to address its axis’s
central tension, not its date of publication. This produces
subsubsections that are comparable across axes and that map
onto the shared design features summarized in Table 2.
Although all four axes share the retrieve-then-generate
backbone of Algorithm 1, their objectives differ inwhat they
optimizeandwhichcontrolvariabletheyexpose.Theefficiency
objective (Eq. (7)) maximizes a utility-to-costratio, exposing
the compression costC ϕ(di)as its lever; the defensive objec-
tive (Eq. (8)) replaces that ratio with an additiverisk penalty
(1−T(d))+P(d)+B(d)weighted by relevance, exposing per-
document toxicity, privacy, and bias scores; the interactive
objective (Eq. (9)) introducesuser-conditionedtermsH(u,d)
andF(d,u)under a personalized policyπ u(d|q), exposing
interaction history and feedback; and the reasoning objective
(Eq. (10)) is the onlymulti-stepformulation, summing a
per-step retrieval-grounded term with coherence and consis-
tency regularizers acrossTreasoning steps. The reference
algorithms differ correspondingly: the efficiency and defensive
pipelines (Algorithms 2 and 3) re-score a single retrieved
set-by cost and by risk, respectively-whereas the interactive
pipeline(Algorithm4)samplesfromauser-conditioneddistri-
bution and the reasoning pipeline (Algorithm 5) re-enters re-
trieval at every step. Read together, each formulation relaxes
a different limitation of the plain RAG objective (Eq. (3)):
efficiency adds cost-awareness, defense adds risk-awareness,interactivity adds user state, and reasoning adds multi-hop
temporal structure.
Table 2 operationalizes this comparison at the method
level:itcharacterizesrepresentativesystemsfromeachaxisby
thecross-cuttingdesignfeaturestheyexhibitandtheresearch
question(s) they primarily address, making explicit how a
given subset of features maps to progress on a specific RQ.
3.6 Compression and Efficiency-Driven RAG
The compression and efficiency axis is included in the taxon-
omy because a large portion of recent RAG research explicitly
examines the computational bottlenecks introduced by re-
trievalatscale[46,60,61,62].Asexternalcorporacontinueto
grow and RAG systems increasingly integrate heterogeneous
sources, a consistent theme across the literature is the need to
reduce retrieval latency, limit token overhead, and improve
the relevance–cost trade-off when selecting context. These
concerns are especially salient in production systems, where
inference cost and throughput place tight constraints on how
much context can be retrieved and processed. Many works
therefore treat retrieval as an optimization problem that
balances the benefits of additional evidence against the cost
of longer input sequences and slower decoding. This pattern
recurs across studies published from 2020-2025 and motivates
treating compression- and efficiency-oriented approaches as
a distinct dimension within the broader RAG landscape. In
practice, this axis captures methods that rethink how much
to retrieve, how to compress it, and how to allocate scarce
context budget without sacrificing model performance.
We define a retrieval- and compression-aware objective for
compression- and efficiency-driven RAG systems as follows:
min
θ,ϕEq∼Q/bracketleftiggk/summationdisplay
i=1U(q,di)·log/parenleftbiggpθ(di|q)
Cϕ(di)/parenrightbigg/bracketrightigg
.(7)
where:
•q∈Q: input query drawn from the query distribution.
•di: thei-th retrieved document for queryq.
•pθ(di|q): retrieval probability of documentd igiven
queryq, parameterized by retriever parametersθ.
•U(q,di)∈[0,1]: utility function measuring the relevance
or helpfulness ofd ifor queryq.
•Cϕ(di)>0: compression cost or token-level cost of
including documentd i, parameterized by compression
moduleϕ.
•log/parenleftig
pθ(di|q)
Cϕ(di)/parenrightig
: log-efficiency score balancing relevance
and cost.
This objective prioritizes high-utility, low-cost documents for
inclusion in the context window, reflecting the core trade-offs
incompression-andefficiency-drivenRAG.Theretrieveraims
to surface documents that are semantically aligned with the
query, while the compression model estimates the expected
cost of including each document in the LLM’s input con-
text (e.g., token length, redundancy, or entropy). The log-
ratio quantifies how much useful information each document
provides relative to its cost, thereby guiding the selection of
documents that maximize informativeness per token under
constrained budgets. In other words, the retrieval model at-
tempts to approximate the ideal allocation of context, where
each additional token contributes meaningful evidence rather

7
TABLE 2: Cross-axis comparison of representative RAG methods. Each method is marked (✓) with the cross-cutting design
features it exhibits and the research question(s) it primarily addresses. Feature columns:Comp./cost-compression- or cost-
aware context selection;Multi-step-retrieval interleaved across reasoning or refinement steps;Safety-toxicity, bias, or privacy
filtering;Personal.-user/history-conditioned retrieval;Structured-retrieval over graphs, tables, or other structure;Verify-
critique or verification of retrieved evidence.
Method Retrieval Comp./cost Multi-step Safety Personal. Structured Verify PrimaryRQ
Compression & Efficiency (RQ1)
RAPTOR [46] Dense✓− − −✓−RQ1
xRAG [60] Dense✓− − − − −RQ1
RQ-RAG [61] Dense✓ ✓− − − −RQ1
Stochastic RAG [62] Dense✓− − − − −RQ1
Defensive RAG (RQ2)
FILCO [63] Dense− −✓− −✓RQ2
SELF-RAG [47] Dense−✓ ✓− −✓RQ2, RQ4
PoisonedRAG [64] Dense− −✓− − −RQ2
FairRAG [65] Dense− −✓− − −RQ2
Interactive, User-Centric (RQ3)
Chat-REC [48] Dense− − −✓− −RQ3
PersonaRAG [66] Dense− − −✓− −RQ3
ERAGent [67] Dense✓ ✓−✓−✓RQ3
ISEEQ [68] Hybrid−✓− −✓−RQ3
Complex Reasoning (RQ4)
IRCoT [69] Dense−✓− − − −RQ4
GenGround [49] Dense−✓− − −✓RQ4
PlanRAG [70] Dense−✓− −✓−RQ4
G-Retriever [71] Graph− − − −✓−RQ4
than redundant or distracting information. This formulation
captures a principle that appears implicitly across many sys-
tems, even when it is not written explicitly as an optimization
objective.
The expression also highlights the central relevance–cost
trade-off that defines this axis. The termp θ(di|q)repre-
sents the retrieval probability of a document being useful
for answering the query, whileC ϕ(di)captures token-level
compression cost, redundancy, entropy, or other length-based
constraints. The log-ratio thus prioritizes documents that
provide high expected utility while incurring minimal context
overhead,aprincipleunderlyingmanyefficiency-focusedRAG
systems. Algorithm 2 sketches a generic compression-aware
pipeline that reflects common design choices: scoring docu-
mentsbyutilityandcost,filteringnoisyorlow-yieldevidence,
and constructing a compact context set for downstream gen-
eration. Variants of this pipeline appear in systems that use
hierarchical retrieval [46], minimal-context prompting [60],
reranking [72], and other strategies [73] aimed at controlling
inference-time cost while preserving answer quality.
This pipeline reflects operations found across the liter-
ature: utility estimation, compression-aware scoring, noise
filtering, and selection of high-efficiency documents. Variants
of these steps appear in works on few-shot retrieval, hierarchi-
cal access, query reformulation, and token-minimal context
construction, often combined with model-side optimizations
such as pruning or distillation. Together, they illustrate how
efficiencyconsiderationsincreasinglyshapethedesignofmod-
ern RAG systems and motivate this axis of the taxonomy.
Figure 3 contrasts this compression-aware pipeline with a
normal RAG pipeline, showing how compressed-index lookup
and contextual compression narrow a full candidate chunk set
down to the minimal evidence passed to the generator.Algorithm 2Compression-Aware Retrieval-Augmented
Generation
Require:Queryq, corpusC, retriever parametersθ, com-
pressor parametersϕ, top-kretrieval size
Ensure:Compressed document contextD compfor genera-
tion
1:qvec←ϕq(q)▷Encode query
2:D←RetrieveTopK(q vec,C,θ)
3:for alld i∈Ddo
4:ComputeU(q,d i)▷Utility or relevance score
5:ComputeC ϕ(di)▷Compression or cost estimate
6:Compute efficiency scoreE i←log/parenleftig
pθ(di|q)
Cϕ(di)/parenrightig
7:endfor
8:D comp←SelectTopDocuments(D,based onE i)
9:returnD comp
3.1.1 Utility-Cost Optimization and Query Refinement
The most direct expression of this axis is the effort to maxi-
mize the usefulness of retrieved evidence relative to its cost,
mirroring the utility termU(q,d i)and the log-efficiency score
attheheartoftheobjectivedefinedforthisaxis.Severalmeth-
ods establish the building blocks of utility- and cost-aware re-
trieval.REINA(RetrievingfromthetraINingdatA)proposed
enhancing NLP task performance by retrieving semantically
similar labeled examples from the training set [74]. Although
REINA did not involve external retrieval, it highlighted the
potential of instance-level context reuse and drew attention
to the computational bottlenecks involved in large-scale index
traversal-an issue central to efficient RAG. In a similar spirit,
Improving Language Models by Retrieving from Trillions of
Tokens[75] demonstrated that integrating retrieval over vast
external corpora could substantially improve language mod-

8
USER QUER Y
How many vacation days do employees
get per  year?
   EMBED THE QUER Y
COMPRESSED INDEX
LOOKUP
Chunk A: “Full-time employees accrue 1.5 days per
month…”
Chunk B: “V acation cannot exceed 18 days annually
unless appr oved…”
Chunk C: “Leave balance r esets at year -end…"
Chunk D: “Part-time employees accrue leave
proportionally…”
Chunk E: “Unused vacation expir es after  12
months…”
PASS TO LLM
LLM GENERA TES FINAL  
ANSWERRETRIEVE TOP K
CHUNKS FROM VECT OR
DB 
“Full-time employees get 18 vacation days per
year. Unused days expir e after  12 months.
Part-time employees may get less depending on
hours worked.”CONTEXTUAL  COMPRESSION 
“Employees accrue 1.5 vacation
days/month.”
“Annual vacation cap: 18 days.”
LLM GENERA TES FINAL
ANSWER
“Employees earn 1.5 vacation days each month,
totaling 18 days per  year .”PASS TO LLM NORMAL  RAGCOMPRESSION AND
EFFICIENCY  DRIVEN RAGCOSINE SIMILARITY  ANN SEARCH 
“Employees accrue 1.5 vacation days per  month.”
“Annual vacation cap: 18 days.”
“Unused days may carry over  if appr oved.”
“Part-time leave is pr orated.”
Fig. 3: Schematic of Compression and Efficiency-Driven RAG systems.
eling, especially in low-resource or domain-specific settings.
By retrieving over vast external corpora, it makes the cost of
indiscriminateretrievalafirst-orderconcern-preciselythecost
that the utility-cost objective is designed to manage.
A complementary line of work made clear why cost-aware
selection matters.LLMs Can Be Easily Distracted by Irrel-
evant Context[76] showed that even high-performing LLMs
could suffer substantial performance drops when presented
with extraneous or misleading context. These findings un-
derscore the need for careful filtering and query refinement
so that compression and pruning do not discard useful ev-
idence, a principle that methods operationalize in several
ways. Augmentation-aware retrievers explicitly model the
interaction between retrieved context and generative objec-
tives, encouraging retrieval policies that better align with the
downstream use of evidence and treating the retriever and
generator as co-adaptive modules rather than disjoint com-
ponents[77].Prompt-guidedretrievalstrategiesextendedthis
idea to non-knowledge-intensive tasks such as classification
and text generation, using prompt reranking to better align
retrieved context with task-specific requirements [78]. Query-
side refinement is more targeted still: RQ-RAG provides
modules that dynamically rewrite retrieval queries to improve
documentrelevanceandaddressthebrittlenessofstaticquery
formulations [61], while broader studies of context-selection
strategies across multiple domains argued that one-size-fits-
all retrieval policies fail to generalize and advocated adaptive
mechanisms informed by dataset characteristics and task ob-
jectives [42]. Stochastic RAG made the cost-utility trade-offexplicit by adopting utility-maximization formulations that
guide retrieval based on expected generation quality, directly
instantiating the utility termU(q,d i)that the efficiency ob-
jective seeks to maximize per unit of context cost [62].
3.1.2 Structural Compression and Hierarchical Retrieval
A second family of methods targets the compression cost
Cϕ(di)directly, reducing the token and computational foot-
print of retrieved evidence through structural means. On the
model side, structured pruning methods compress generative
backbones without significant loss in performance, offering a
lightweight alternative for latency-sensitive applications [79].
On the representation side, SANTA pretrains language mod-
els with an inductive bias toward structured data formats,
improving dense retrieval from semi-structured corpora such
as tables or knowledge graphs [80]. Rather than retrieving
flat sets of passages, RAPTOR organizes information into
recursive tree-based abstractions that can be traversed or
summarized,enablingmoreefficientreasoningovermulti-level
content;ineffect,itimplementstheSelectTopDocumentsstep
of Algorithm 2 over a hierarchy of higher-level units rather
than a flat candidate set [46]. Compression itself was pushed
to an extreme by xRAG, which distills relevant information
into minimal tokens without compromising response quality-
an approach that directly minimizes the compression-cost
termCϕ(di)in the efficiency objective [60]. Superposition
prompting complements these methods by exploring parallel
retrieval streamsthatreduceinference latencyandfilternoisy
documents [73].

9
3.1.3 Domain-Adaptive and Low-Supervision Retrieval
Athirdthemeconcernsadaptingretrievaltonewdomainsand
operating under limited supervision, where the cost being op-
timized is the cost of labeled data and domain transfer rather
than tokens alone. The challenge of handling diverse retrieval
sources was tackled inRAG across Heterogeneous Knowl-
edge[81], which addressed integrating information across
corpora with varying formats, quality, and domain alignment
through multi-source retrieval and fusion. To reduce the cost
of labeled supervision, PROMPTAGATOR proposed a few-
shot dense retrieval pipeline that required as few as eight
examples, using LLMs to synthesize training queries in data-
scarcesettingsandloweringtheentrybarrierfornewretrieval
systems [82]. Retrieval was also used as a form of implicit
supervision in model synthesis for domain-specific languages
(DSLs), where in-domain exemplars guide generation and
helpsystemscopewithout-of-distributiongeneralization[83].
Generalization to unseen tasks was advanced by UPRISE,
a universal prompt retrieval system for zero-shot evaluation
that leverages retrieval to design effective prompts without
task-specifictuning[84],whileintherecommendationdomain
RAG techniques were adapted for open-world personalization
by incorporating retrieval-augmented knowledge into recom-
mender pipelines [85].
As RAG expanded into specialized domains, structure-
and language-aware retrieval became central. Chunking
strategies tailored to financial documents were proposed to
improve retrieval granularity in dense, highly technical cor-
pora [86], and benchmarks such as LegalBench-RAG [87]
togetherwithArabic-specificevaluationshighlightedthechal-
lenges of non-English, morphologically rich, and highly struc-
tured legal data retrieval [88, 89]. The same low-supervision,
domain-adaptive philosophy carried RAG into new applica-
tion areas: retrieval-augmented test generation surfaced rel-
evant code or documentation during test synthesis to sup-
port software engineering workflows [90], generative retrieval
methods incorporated richer item and user context into rec-
ommender systems [91], and retrieval-guided generation was
used for synthetic dataset creation to improve diversity and
realism [92]. Collectively, these methods show how retrieval
can act as both a knowledge source and a control signal
that supports domain adaptation and personalization under
constrained supervision.
3.1.4 Evaluation and Toolkits
Progress on efficiency-driven RAG has been accompanied by
infrastructure for measuring and reproducing it. Modular
toolkits such as FlashRAG, RAG Foundry, and RAGLAB
emerged as open-source platforms that support configurable,
reproducible experimentation across datasets and architec-
tures, providing standardized interfaces for retrievers, genera-
tors, and evaluators that lower the barrier to entry and enable
more systematic comparisons [93, 94, 95]. On the evaluation
side, RAGAS introduced automated, reference-free metrics
that assess both retrieval quality and generation fidelity [96],
while RAGBench provided a multi-domain benchmark for ex-
plainableRAGevaluationacrosstasksandretrievalconfigura-
tions [97]. Other studies focused on the impact of embedding
models on retrieval relevance, underscoring that embedding
choice remains a critical but often underappreciated factor in
RAGperformance[98].Together,theseeffortsmakeclearthatevaluating efficiency involves not only latency and context
length but also how compression and retrieval decisions affect
downstream quality.
Takenasawhole,compression-andefficiency-drivenRAG
spans four complementary families: utility-aware query re-
finement, structural and hierarchical compression, domain-
adaptive retrieval under limited supervision, and compre-
hensive evaluation ecosystems. This collective shift has en-
abled RAG to scale across domains, adapt to new modal-
ities, and maintain high utility even under compute and
data constraints, making efficiency a cornerstone of modern
knowledge-augmented generation that connects retrieval me-
chanics, model architecture, and evaluation under a common
cost-aware perspective.
Key Takeaways & Insights
Compression- and efficiency-driven RAG research reveals sev-
eral consistent themes across the literature. A central insight
is the importance of optimizing the trade-off between docu-
ment utility and context cost, as formalized in Eq. (7) and
operationalized by Algorithm 2. Across heterogeneous modal-
ities, languages, and structured corpora, the recurring need
is for adaptive methods that manage distractors and balance
relevanceagainstcomputationaloverhead.Recentadvancesin
compression such as pruning, distillation, and augmentation-
aware retrieval show a shift toward systems capable of op-
erating efficiently under tight latency or memory constraints
while maintaining generation quality. Despite this progress,
current approaches face limitations including brittle retrieval
scoring,lossofsemanticfidelityunderaggressivecompression,
sensitivity to ambiguous queries, insufficient robustness to
domain shift, and a lack of unified evaluation standards for
cost-aware retrieval performance.
Common failure modes mirror these limitations: models
often over-retrieve redundant or irrelevant evidence, under-
retrieve essential information when compression is overly
aggressive, or become susceptible to distractors that mislead
generation. Existing systems also struggle with static query
formulations and inconsistent handling of structured or mul-
tilingual sources. Several gaps remain open, including the
absenceofageneralframeworkforjointlyoptimizingretrieval
and compression, limited understanding of how compres-
sion affects multi-step reasoning, and the need for adaptive,
intent-aware retrieval strategies. For practitioners, effective
deployment requires balancing retrieval depth with context
budget, incorporating reranking or utility scoring to filter
noise, tuning compression to the specific task, and adopt-
ing structure-aware retrieval when working with specialized
corpora. Monitoring common failure modes and leveraging
query-refinement mechanisms can further improve robustness
in real-world, latency-sensitive RAG applications.
3.7 Defensive RAG
Figure 4 contrasts a normal RAG pipeline with a defensive
RAG pipeline, in which an explicit harmful-content/bias de-
tection stage either reframes the prompt or routes retrieval
through a bias-aware, safe-document branch before gener-
ation. Defensive RAG is included as a core axis of this
taxonomy because a significant portion of recent work (2022-
2024)explicitlyfocusesonmitigatinghallucinations,reducing

10
Normal RAG Pipeline Defensive RAG Pipeline
 User Prompt
Embed the prompt
Retrieve top
documents 
Pass retrieved
documents
LLM
generates
final
response Harmful content /
Bias detection
Bias aware/
  Safe document
retrieval
LLM
generates
neutral
responseUser Prompt
If detected
(Modify/ Block/
Reframe
prompt)Embed clean
prompt
Fig. 4: Comparison between standard and defensive RAG pipelines, illustrating safeguards for harmful content detection,
prompt sanitization, and bias-aware document retrieval.
bias, improving adversarial robustness, and preventing pri-
vacy leakage in retrieval-augmented systems [63, 47, 64, 65].
As RAG enters high-risk domains such as healthcare, finance,
law, and public policy, safety and trustworthiness have be-
come fundamental design criteria rather than optional add-
ons. Standard RAG pipelines that simply retrieve and condi-
tion on documents may inadvertently propagate misinforma-
tion, expose sensitive information, or amplify harmful biases
embedded in external corpora. These risks motivate special-
ized architectures and objectives that treat safety, fairness,
and privacy as first-class optimization goals. Defensive RAG
systems therefore modify both retrieval and generation to
respect risk-aware constraints, filter or downweight harmful
evidence, and introduce verification or critique stages before
producing final outputs. Within this taxonomy, these efforts
are grouped under a single axis to highlight shared techniques
and challenges in building robust, trustworthy RAG systems.
We define a bias-mitigation, vulnerability-aware, and
privacy-preserving objective for Defensive RAG systems as
follows:
LDefRAG =Eq∼Q/bracketleftigg/summationdisplay
d∈Dq/parenleftig
(1−T(d)) +P(d)
+B(d)/parenrightig
·Rel(q,d)/bracketrightigg
(8)
where:
•Qis the distribution over user queriesq.
•Dqis the set of retrieved documents for queryq.
•Rel(q,d)∈[0,1]is the retrieval relevance score betweenq
andd.
•T(d)∈[0,1]is the probability that documentdis non-
toxic.•P(d)∈[0,1]is the probability that documentdcontains
private or sensitive content.
•B(d)∈[0,1]is the bias score of documentd.
This formulation penalizes documents that are toxic, biased,
or privacy-sensitive. Intuitively,(1−T(d)) +P(d) +B(d)
estimates the expected harm associated with a retrieved doc-
ument, while multiplication withRel(q,d)penalizes harmful
documents in proportion to how influential they would be
in generation. Documents that are both highly relevant and
highly risky thus receive large penalties, encouraging the
system to either exclude them or replace them with safer
alternatives.Inpractice,thisobjectivecanberealizedthrough
filtering, reweighting, or constrained optimization over the
retrieved set.
Algorithm 3 outlines the core retrieval logic used in defensive
RAGpipelines,illustratinghowsafetyandfairnessconstraints
can be integrated into the retrieval loop.
Algorithm3implementsafiltereddocumentselectionloop
that enforces ethical and safety constraints during retrieval.
Candidate documents are first retrieved using a standard rel-
evance function and are then re-scored using toxicity, privacy,
and bias estimates before being passed to the generator. Simi-
lar filtering and reweighting strategies appear in systems such
as FILCO [63], SELF-RAG [47], and LLM-based reranking
pipelines [72], which dynamically rescore documents using
signals linked to safety, factuality, or alignment with system
policies. These approaches illustrate how retrieval itself can
be treated as a controllable, safety-critical component rather
than a neutral preprocessing step.

11
Algorithm 3Bias-, Privacy-, and Safety-Aware Defensive
RAG
Require:Queryq, corpusC, scoring modelsT,P,B, rele-
vance functionRel
Ensure:Filtered document setD safe, generated responseˆy
1:qvec←ϕq(q)▷Encode query
2:D cand←RetrieveTopK(q vec,C)
3:for alld∈D canddo
4:Compute relevanceRel(q,d)
5:Compute risk score:S d←(1−T(d)) +P(d) +B(d)
6:Compute safe-weighted score: /tildewiderRel(q,d)←Rel(q,d)·
(1−Sd)
7:endfor
8:D safe←SelectTop(D cand,/tildewiderRel)
9:ˆy←G(q,D safe)
10:returnˆy
3.2.1 Filtering, Reranking, and Retrieval-Aware Critique
A first family of defensive methods treats retrieval itself as
a controllable, safety-critical stage by filtering, reranking, or
critiquing candidate evidence before it reaches the generator.
One subgroup shows how the choice of retrieved context
can steer model behavior: UPRISE uses universal prompt
retrieval to dynamically select task-relevant prompts from
curated pools, enabling LLMs to perform well on unseen
tasks without manual prompt tuning [99]. Although initially
focusedonpromptretrievalratherthandocument-levelRAG,
it demonstrated that retrieved context functions as a control
mechanism influencing both the style and content of gener-
ation. At the document level, the FILCO framework applies
fine-grained filtering to select high-quality documents after
retrieval, improving factual alignment and reducing halluci-
nation; in effect, FILCO performs the risk-aware re-scoring of
/tildewiderRel(q,d)that Algorithm 3 applies before passing documents
to the generator [63].
LLMs were also found useful not only as generators but
as rerankers and retrieval-aware critics. While LLMs often
struggle with few-shot information extraction, they can be
highly effective as rerankers for hard-to-disambiguate queries,
allowinginitialcandidatedocumentstobefilteredorre-scored
before final generation [72]. The synergy between retrieval
and generation was pushed further by SELF-RAG, which
combinesretrieval,generation,andalearnedcritiquemodelto
increasefactuality;SELF-RAGeffectivelyimplementsthecri-
tique loop implicit in Algorithm 3, re-querying and re-scoring
evidence whenever a learned quality-estimation signal for the
current output falls below a threshold [47]. These retrieval-
aware critique mechanisms show how defensive behavior can
bebuiltdirectlyintothearchitectureratherthanboltedonas
an external safety filter, giving models a natural interface for
applying policy constraints when certain topics or evidence
types warrant additional scrutiny.
3.2.2 Bias, Fairness, and Sociocultural Safety
Beyond factual reliability, a growing body of work investi-
gates fairness and ethical concerns in RAG systems. Studies
such as [100] and [101] revealed persistent demographic and
cultural biases in retrieved and generated outputs, indicat-
ing that retrieval may replicate or amplify existing societal
stereotypes. These findings were reinforced by [102], whichquestioned the objectivity of LLM-based evaluation pipelines
in domains with high sociocultural sensitivity and highlighted
theneedfordomain-awareretrievalstrategiesthataccountfor
the trustworthiness and appropriateness of external knowl-
edge sources. As RAG systems increasingly mediate access
to information, such biases risk reinforcing misinformation or
systemic prejudice at scale, since the combination of biased
retrieval and uncritical generation can present skewed or
harmful content as authoritative. In terms of the defensive
objective, this line of work targets the bias termB(d), ar-
guing that fairness and safety must be treated as end-to-
end properties of the RAG pipeline rather than attributes of
the language model alone, and motivating the integration of
bias detectors, content filters, and policy constraints across
retrieval and generation stages.
Figure 5 illustrates how user-in-the-loop mechanisms re-
structure retrieval through clarification, iterative refinement,
and contextual adjustment-concepts that parallel the defen-
sive filtering, safety-aware reranking, and bias suppression
strategies discussed here. Although the figure primarily high-
lights interactive RAG, similar design principles apply when
humans or policy modules provide feedback on retrieved ev-
idence and generated outputs, closing the loop between user
preferences, safety constraints, and system behavior.
3.2.3 Privacy, Leakage, and Adversarial Robustness
A third family of methods addresses the privacy and adver-
sarial risks that arise when retrieval draws on external or un-
trustedcorpora,correspondingtotheprivacytermP(d)inthe
defensive objective. Foundational studies compared retrieval-
based augmentation with fine-tuning for injecting external
knowledge, highlighting trade-offs between flexibility, safety,
and control [103], while analyses of RAG failure modes doc-
umented engineering bottlenecks such as hallucination per-
sistence, attribution ambiguity, and irrelevant retrieval [104].
Privacy and information leakage emerged as especially press-
ing concerns, as researchers showed how RAG systems might
expose sensitive data through unfiltered retrieval, corrupted
knowledge bases, or misconfigured access controls [105, 106].
The risk is acute in high-stakes settings: in healthcare, for
instance, retrieving sensitive or misaligned documents can
leadtoethicallyproblematicoutputsorinadvertentdisclosure
of private information [107, 108].
Several approaches respond by constraining or reshap-
ing the retrieval corpus. Some systems use synthetic data
as a retrieval corpus to reduce direct exposure of sensi-
tive information while still providing useful evidence [109],
and others enforce strict access-control policies so that re-
trieval respects organizational or regulatory boundaries [110].
Work on adversarial robustness developed strategies to de-
tect and neutralize poisoned content in knowledge bases,
including defenses proposed in BADRAG and related stud-
ies [111, 107, 64]. Filtering-based approaches aim to suppress
distracting or malicious documents, while controlled noise
injection-introducing random but non-toxic documents-was
surprisingly found to improve model calibration and accu-
racy by reducing overconfidence [112]. Denial-of-service-style
attacks on RAG retrieval were studied as well, emphasizing
the need for secure ranking systems and robust retrieval
infrastructures [113].

12
3.2.4 Verification and Benchmarking
A final family of work focuses on verifying outputs and
benchmarkingrobustness,complementingretrieval-sidefilter-
ing with post-hoc checks on what the system is willing to
assert. The need for such checks is shown by HaluEval, which
benchmarks hallucination prevalence across multiple LLMs
and emphasizes the role of retrieval filtering in minimizing
such errors [114]. Complementing this, chain-of-verification
methods validate claims retrieved from external sources be-
fore generation, allowing models to cross-check evidence and
flag inconsistencies [115], while multilingual robustness tech-
niques [116] and factual boundary analysis [117] improve un-
certainty estimation and reduce overconfident errors in sparse
or misaligned settings.
Benchmarking efforts emerged to evaluate RAG robust-
ness and factual consistency more systematically. IRSC and
UDA proposed real-world and zero-shot benchmarks for
measuring resilience under distribution shift and noisy re-
trieval [118, 119], BERGEN provided a modular library for
unified evaluation across diverse datasets and metrics [120],
and domain-specific metrics such as Face4RAG for Chinese
factuality evaluation emphasized the need for culturally in-
formedevaluationpipelines[121].Fairnessconcerns[122]were
examined inTowards Fair RAG[65, 123], which showed
how retrieval ranking can skew information exposure and
exacerbate representational imbalances. Collectively, these
developments define the emerging field of Defensive RAG:
by combining filtering, reranking, uncertainty modeling, bias
mitigation,privacypreservation,andethicalsafeguards,these
systems move beyond naive retrieval toward principled, risk-
aware architectures. Defensive RAG thus complements the
efficiency-focused axis by emphasizing not just how much and
how fast we retrieve, but also what we retrieve and how safely
it can be used.
Key Takeaways & Insights
Defensive RAG research highlights the growing need for ro-
bustness, trustworthiness, and ethical safeguards in retrieval-
augmented systems. The literature emphasizes that factual
reliability depends not only on retrieving relevant evidence
but also on filtering out toxic, biased, misleading, or privacy-
sensitivedocuments,asformalizedinEq.(8)andAlgorithm3,
establishing retrieval as a controllable, safety-critical compo-
nent of the pipeline rather than a neutral preprocessing step.
Despite this progress, Defensive RAG faces several limita-
tions and failure modes that remain open challenges. Current
approaches struggle with reliably detecting subtle bias, pri-
vacyrisks,andtoxiccontent,andsafetyfiltersmayincorrectly
suppress benign documents or fail to block harmful ones.
Retrieval-scoring methods remain sensitive to adversarial or
noisy corpora, and systems can still over-retrieve mislead-
ing evidence, under-retrieve safe but essential information,
or propagate biases embedded in external knowledge bases.
Existing work does not fully address the difficulty of jointly
optimizing relevance, safety, and fairness, nor the problem of
ensuring robustness under distribution shift, poisoned docu-
ments, or large-scale misinformation campaigns.
For practitioners, deploying Defensive RAG requires
adopting risk-aware scoring, performing rigorous filtering and
reranking of retrieved documents, and monitoring failurecases such as hallucinations, leakage, and biased responses.
Applying verification or self-reflection mechanisms when un-
certainty is high, and carefully designing privacy controls,
fairness-sensitive retrieval policies, and adversarial resilience
strategies are critical for safe deployment in sensitive do-
mains. These practices can help bridge the gap between
research prototypes and production-ready systems, ensuring
that retrieval-augmented architectures are not only powerful
but also responsible and trustworthy in real-world use.
3.8 Interactive, User-Centric RAG Systems
Traditional RAG systems are typically designed around static
retrieval pipelines that optimize document relevance with
respect to a query. However, such pipelines often ignore
the dynamic, user-driven nature of real-world interactions,
where user goals may shift over time, context evolves across
turns, and retrieved knowledge must adapt accordingly. In
many deployed systems, retrieval remains fixed and query-
only, even though users provide rich signals through feedback,
preferences,andinteractionhistory.Thisaxisofthetaxonomy
captures RAG architectures that are interactive, user-aware,
and context-sensitive, enabling LLMs to refine, adapt, or per-
sonalize their retrieval and generation processes based on user
intent and evolving input. These systems aim to move from
a one-shot, query-centric view toward iterative, session-based
behaviorthatreflectslong-termuserneeds.Giventhegrowing
volume of work that explicitly models user intent, feedback,
and personalization over time [48, 66, 67, 68], interactive,
user-centric RAG constitutes a distinct axis in the taxonomy,
separate from purely efficiency- or defense-oriented methods.
We define a user-conditioned, interaction-aware objective
for interactive RAG systems as:
LInterRAG =E (q,u)∼U/bracketleftigg/summationdisplay
d∈Dq,u/parenleftig
αS(q,d) +βH(u,d)
+γF(d,u)/parenrightig
πu(d|q)/bracketrightigg
(9)
where:
•Uis the distribution over query–user pairs(q,u).
•Dq,uis the set of retrieved documents for queryqand
useru.
•S(q,d)∈[0,1]is the semantic relevance score between
queryqand documentd.
•H(u,d)∈[0,1]is the historical interaction alignment
score between useruand documentd.
•F(d,u)∈[0,1]is the feedback compatibility score esti-
matinghowwelldocumentdalignswithuserpreferences.
•πu(d|q)is the user-conditioned document selection
policy.
•α,β,γ∈R +are tunable weights controlling the trade-
off between query relevance, historical consistency, and
personalization.
Intuitively, this objective encourages the system to select
documents that are simultaneously semantically relevant to
the current query (S), consistent with the user’s prior in-
teractions or history (H), and compatible with explicit or
implicit feedback signals (F). The weightsα,β,γgovern how
strongly each of these dimensions contributes to the final
score, while the user-conditioned policyπ u(d|q)translates

13
Algorithm 4Interactive, User-Aware RAG Pipeline
Require:Queryq, user profileu, document corpusC, selec-
tion policyπ u, scoring functionsS,H,F
Ensure:Personalized contextD q,uand generated responseˆy
1:qvec←ϕq(q)▷Encode query
2:D cand←RetrieveTopK(q vec,C)
3:for alld∈D canddo
4:ComputeS(q,d)▷Query–document semantic
relevance
5:ComputeH(u,d)▷User history alignment score
6:ComputeF(d,u)▷Feedback/personalization score
7:Compute total score:R d←αS(q,d) +βH(u,d) +
γF(d,u)
8:endfor
9:πu(d|q)←Softmax overR d
10:SampleD q,u∼πu(d|q)▷Personalized document
selection
11:ˆy←G(q,D q,u)▷Generate response with selected context
12:returnˆy
these scores into a personalized distribution over documents.
Inpractice,thesequantitiesmaybeimplementedusingneural
scorers, heuristic features, or combinations thereof, but the
objective highlights the shared goal of aligning retrieval with
evolving user intent. This formulation captures the essence
of interactive RAG: retrieval is no longer purely query-driven
but shaped by user identity, history, and feedback.
The end-to-end interactive personalization pipeline is de-
scribed in Algorithm 4.
Algorithm 4 outlines a retrieval–generation loop that inte-
grates user feedback, profile information, and intent modeling
to personalize outputs. Given a query and user identity, can-
didate documents are scored not just by semantic relevance
to the query (S(q,d)) but also by how well they align with
the user’s historical interactions (H(u,d)) and expressed or
inferred preferences (F(d,u)). A learned or heuristic docu-
ment selection policyπ ucombines these scores and samples
a final set of documentsD q,uto serve as input context for
the generator. This interactive loop enables adaptation over
sessions, dynamic query expansion, and alignment with user
expectations, making it well-suited for recommendation, con-
versational agents, and domain-specific assistants. Variants
of this pipeline are instantiated in systems such as Chat-
REC [48], ERAGent [67], PersonaRAG [66], and adaptive
conversational RAG frameworks [124], which implement user-
aware scoring and policy-based document selection. Figure 5
(introduced earlier) conceptually contrasts naive RAG with
interactive architectures that incorporate user feedback, in-
tentclarification,anditerativerefinement,mirroringthecom-
ponents formalized in Algorithm 4.
3.3.1 Intent Clarification and Information-Seeking
A first family of interactive RAG work focuses on closing the
gapbetweenauser’sinitialqueryandtheiractualinformation
need, treating clarification and proactive question-asking as
part of the retrieval loop. The introduction of metaprompts,
as proposed inPrompt Programming for LLMs: Beyond the
Few-Shot Paradigm, enabled models to generate their own
prompts based on task goals, shifting the burden of instruc-tion design away from the user [125]. This approach demon-
stratedthatwell-structuredzero-shotpromptscouldmatchor
even outperform traditional few-shot examples, especially in
general-purpose reasoning tasks. However, despite improving
the interface between user input and model response, this
strategy remained fundamentally reactive: models could only
operate within the constraints of the given prompt and could
not independently identify or seek missing information.
Addressing this constraint, ISEEQ framed LLMs as in-
teractive agents capable of generating Information-Seeking
Questions (ISQs) to clarify, refine, or expand a user’s ini-
tial query [68]. By leveraging knowledge graphs and dy-
namic meta-information retrieval, the system enriched se-
mantic representations of user intent and enabled models to
initiate clarifying sub-questions during generation, integrat-
ing a knowledge-aware passage retriever with a generative-
adversarial reinforcement learning framework [126] to keep
those questions coherent and retrieval-relevant. This marked
a conceptual shift from static prompt execution to dialogue-
like, multi-turn interaction in which the model actively
queriesexternalknowledgeinresponsetointernaluncertainty.
The same proactive philosophy motivated generate-then-read
paradigmssuchasGENREAD,whichreversedthetraditional
retrieve-then-generate pipeline by letting LLMs synthesize
their own contextual documents before generating an an-
swer [127]. While promising in terms of focus and relevance,
this raised questions about factual grounding and led to hy-
brid strategies that combine generative context construction
with retrieval-based verification. Together, these early meth-
ods established a key design principle for interactive RAG:
user intent must be modeled explicitly, and retrieval should
respond not only to the initial query but also to evolving
information needs.
3.3.2 Personalization and Dialogue-Driven Retrieval
A second family of systems adapts retrieval and generation to
individual users and to the evolving state of a conversation,
instantiating the historical-alignment termH(u,d)and feed-
backtermF(d,u)oftheinteractiveobjective.Onethreadtar-
gets stylistic adaptation:Diversify Question Generation with
Retrieval-Augmented Style Transfershows that retrieval can
support not only factual accuracy but also creative diversity,
adapting outputs to user tone, task framing, or information
preferences [128]. Personalization and transparency were ex-
plored more directly in systems likeChat-REC, which embed-
ded RAG within recommender systems to enhance explain-
ability and interactive refinement, treating user interaction
history as a dynamic prompt space and retrieving contextual
cues to personalize recommendations in real time [48]. This
history-driven selection is precisely what the user-conditioned
policyπu(d|q)in Algorithm 4 is meant to capture. For
explicit user modeling,PersonaRAGandAdaptive RAG for
Conversational Systemsuse user-specific modeling and adap-
tive retrieval invocation to reduce redundancy and improve
fluency in open-ended conversations [66, 124]; PersonaRAG
in particular scores documents against an explicit user profile,
operationalizing the historical-alignment termH(u,d)that
distinguishes interactive retrieval from purely query-driven
retrieval.
Personalization also proved valuable in instructional and
assistive settings. Personalized learning systems leveraged

14
Fig. 5: Comparison between Normal RAG and Interactive, User-Centric RAG architectures.
RAG to deliver contextually appropriate content, as seen in
applications for children with developmental disabilities [129]
and in feedback generation from lecture materials in pro-
gramming education [130], whileRAMOandMOOC-RAG
addressed course recommendation by drawing on course cat-
alogs and student behavior [131]. Dialogue-driven retrieval
was further extended toward diverse viewpoints, as multi-
perspective query interfaces surfaced contrasting positions
via RAG-enhanced synthesis [132]. The same task-adaptive
instinct appeared in high-precision applications, where inter-
active RAG was evaluated on math question answering [133],
intelligence reporting [134], and cross-lingual information ac-
cess[135];ineachcaseretrievalwastunedtotheusertaskand
optimized for objectives such as factuality, contextual align-
ment, or language sensitivity, marking a shift from retrieval-
as-lookup to retrieval-as-dialogue.
3.3.3 Retriever-LLM Alignment and Agent Architectures
A third family of work concerns how interactive systems
are built-aligning retrievers with LLM needs and assembling
modular, agent-like architectures around the retrieval loop.
Onesubgrouptreatsretrievalassomethingthemodelactively
manages:Teaching LLMs to Self-Debugprovides mechanisms
forevaluatingandimprovingthemodel’sownretrievalchoices
and reasoning chains [136], whileActive RAGtreats retrieval
as a dynamic decision space rather than a fixed preprocessing
step [137]. Flexible retrieval substrates supported these be-
haviors:LLM-Embedderprovided a unified embedding model
for cross-source retrieval, accommodating user demonstra-tions,memorystores,andstructureddatabaseswithinasingle
framework [138], and document- or tool-oriented interaction
was enabled by RAG-based form filling and structured input
parsing[139]andbyTableGPT,whichunifiedinteractionwith
tables, commands, and natural language through external
functional interfaces [140]. Multimodal grounding extended
these architectures beyond text, asMiniGPT-4incorporated
image-grounded context into generation [141] andmPLUG-
Owldemonstrated modular designs combining vision, audio,
and text retrieval [142].
Acentralchallengeinthisfamilyisthedivergencebetween
retriever relevance scoring and the actual utility of retrieved
content for LLM inference. This “preference gap” was directly
addressed inBridging the Preference Gap between Retrievers
and LLMs, which argued that retrievers must be trained not
only for surface relevance but for alignment with model rea-
soning and user-specific information needs-in effect learning
theuser-conditionedpolicyπ u(d|q)ratherthanagenericrel-
evance score [143]. Related training-time strategies included
RA-DIT, which applied dual instruction tuning to both re-
triever and generator for holistic alignment [144], and RADA,
which generated training samples via retrieval-guided context
construction, shifting RAG from a purely inference-time tool
to one that also shapes model training [145]. Modular agent
architectures brought these components together:ERAGent
presented a RAG agent equipped with question rewriting, re-
trieval triggers, knowledge filtering, and user-specific reading
modules-a concrete instantiation of the rewrite-trigger-filter-
read pipeline formalized in Algorithm 4 [67]-whileIM-RAG

15
modeled internal reasoning chains via learned inner mono-
logues across retrieval rounds to enhance context continuity
in dialogue [146]. Interactive RAG was also extended to new
modalities and to transparency: RAG was applied to large-
scale video libraries [147],RAG-Exintroduced explainable
pipelines that let users trace how each retrieved chunk con-
tributed to the output [148], andPromptBenchoffered a
unified framework for evaluating prompt-response dynamics
across LLMs and RAG configurations [149].
Across these three families, interactive, user-centric RAG
combines intent clarification and information-seeking ques-
tiongeneration,personalizationanddialogue-drivenretrieval,
and retriever-LLM alignment within agent architectures to
support dynamic, personalized, and multimodal interaction.
Intent-clarification methods structure input and enable LLMs
to proactively seek missing context; personalization and
dialogue-driven methods incorporate user profiles, dialogue
history, and domain-specific knowledge to guide retrieval
and generation; and alignment-focused methods use inner-
monologue reasoning, retriever-LLM alignment, and explain-
able outputs to redefine RAG as an interactive, user-aware
layer within intelligent systems. This trajectory reflects a
fundamental shift from static augmentation to real-time,
adaptive knowledge integration, positioning RAG as a central
component in human-aligned, feedback-driven AI.
Key Takeaways & Insights
Interactive, user-centric RAG systems represent a major shift
from static, query-only retrieval pipelines toward adaptive,
intent-aware, and personalized knowledge integration. Algo-
rithm 4 formalizes this paradigm by integrating semantic rel-
evance, historical alignment, and user feedback (Eq. (9)) into
a unified scoring and sampling policy for document selection,
treating retrieval as a dynamic loop rather than a one-shot
preprocessing step. Across domains such as recommendation,
education, and cross-lingual information access, interactive
RAG mechanisms improve contextual alignment, enhance
transparency, and support richer user-driven workflows by
incorporatinguserprofiles,preferencemodeling,andadaptive
reasoning strategies.
Despite these advances, key limitations and failure modes
remain. Interactive RAG systems often rely on incomplete or
brittle user models that may misinterpret intent, overfit to
short-term interaction history, or incorrectly generalize user
preferences across contexts. Personalization signals may con-
flictwithsemanticrelevance,leadingtoretrievalofdocuments
that align with user history but degrade factual accuracy.
Feedback signals can be noisy or ambiguous, causing insta-
bility in sampling-based policies such asπ u(d|q); inter-
active agents may over-query, under-query, or reinforce user
misconceptions. Systems built on multimodal pipelines suffer
fromcascadingerrorsandmisalignmentbetweencomponents,
whileproactivequeryingapproachesmayintroduceadditional
latency, compounding retrieval costs. Existing work does not
fully resolve how to balance personalization with safety, how
to reconcile divergent user preferences in multi-user or col-
laborative settings, or how to evaluate user-centric retrieval
across diverse tasks and modalities. For practitioners, effec-
tivedeploymentrequirescarefulcalibrationofpersonalization
weights, mechanisms for uncertainty-aware query expansion,transparent feedback channels, and alignment checks that
ensure retrieved context remains factual, relevant, and ap-
propriate for the user’s goals. As interactive RAG becomes
more deeply integrated into real-world systems, robust user
modeling, adaptive retrieval policies, and principled evalua-
tion frameworks will be essential to support reliable and user-
aligned performance.
3.9 RAG for Complex Reasoning and Multi-Step Tasks
While baseline RAG systems focus primarily on factual
groundingandinformationretrieval,complexreal-worldtasks
often demand structured reasoning, multi-step planning, and
sequentialdecision-making.Insuchsettings,simplyretrieving
a static set of documents and generating a one-shot answer is
rarelysufficient,sincesolutionsmustintegrateevidenceacross
multiple hops, track intermediate decisions, and revise earlier
assumptions when new information appears. This axis of
the taxonomy encompasses RAG architectures that explicitly
support or enhance such capabilities, either through integra-
tion with symbolic structures, iterative generation pipelines,
or domain-specific adaptations for reasoning-intensive tasks.
These systems typically interleave retrieval with chain-of-
thought–style reasoning, allowing models to query external
knowledge at intermediate steps rather than only at the
outset. Given the growing volume of work that explores
retrieval-conditioned chain-of-thought reasoning, structured
state updates, and multi-hop inference across diverse do-
mains [69, 49, 70, 71], complex reasoning forms a distinct
axis in the taxonomy, separate from efficiency-, safety-, or
user-centric RAG. It represents a shift from treating retrieval
as a static augmentation mechanism to viewing it as a core
component of the reasoning process itself.
A reasoning-aligned objective for complex reasoning and
multi-step RAG systems can be formally defined as:
LReason =T/summationdisplay
t=1/bracketleftigg
Ert∼R(ht)[DKL(pt(yt|ht,rt)∥ˆyt)]/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
Retrieval-grounded step loss
+λ· C(r t,ht)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Contextual Coherence+γ· J(h t,ht−1)/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Reasoning Consistency/bracketrightigg
(10)
where:
•T: Total number of reasoning steps.
•ht: Hidden state (or intermediate reasoning output) at
stept.
•rt∼R(ht):Retrieveddocumentsconditionedonh tusing
the retrieval functionR.
•pt(yt|ht,rt): Model’s output distribution at stept,
conditioned on prior reasoning and retrieved context.
•ˆyt: Ground-truth or target output at stept.
•DKL: Kullback–Leibler divergence measuring discrep-
ancy between the generated and target distribution.
•C(rt,ht): Differentiable coherence function quantifying
alignment between retrieved contextr tand reasoning
stateht.
•J(ht,ht−1): Regularizer penalizing incoherent jumps be-
tween successive reasoning states.
•λ,γ: Weighting hyperparameters for coherence and rea-
soning consistency.

16
Algorithm 5Multi-Step Retrieval-Augmented Reasoning
Require:Initial queryq, retrieverR, generatorG, reasoning
stepsT
Ensure:Final outputˆy
1:h 0←Encode(q)▷Initial reasoning state from input
2:fort= 1toTdo
3:rt∼R(ht−1)▷Retrieve context based on prior state
4:yt←G(ht−1,rt)▷Generate intermediate output
5:ht←UpdateState(h t−1,yt)▷Advance reasoning
state
6:Compute step loss:L t←D KL(pt(yt|ht−1,rt)∥ˆyt)
7:Add coherence penalty:L t←Lt+λ·C(rt,ht)
8:Add consistency penalty:L t←Lt+γ·J(h t,ht−1)
9:endfor
10:returnFinal outputˆy←y T
Intuitively, this objective captures the interleaved nature of
complex reasoning in RAG. The first term ensures that each
intermediate generation step is grounded in retrieved evi-
dence, rather than relying solely on parametric memory. The
coherence term encourages retrieved documents to match the
evolving reasoning state, so that context remains relevant as
the model’s understanding progresses. The consistency term
enforces a stable, logically continuous trajectory across steps,
discouraging abrupt or contradictory shifts in intermediate
conclusions. Together, these components express the multi-
hop, interdependent structure of reasoning in complex RAG
pipelines, where retrieval, intermediate states, and final out-
puts are tightly coupled.
The complete multi-step reasoning pipeline is formalized in
Algorithm 5.
Algorithm 5 describes a structured reasoning loop in
RAG systems designed for multi-hop tasks such as complex
question answering, procedural generation, or scientific ex-
planation. The process begins by encoding the query into an
initial hidden stateh 0, which represents the model’s initial
interpretation of the problem. At each timestept, relevant
documents are retrieved based on the current reasoning state,
and this context then conditions the generator to produce
an intermediate outputy t. The output is used to update the
hidden stateh t, enabling chained inference and accumulation
of partial conclusions over time. Each step is supervised
via KL divergence against step-specific ground truths, while
coherence and consistency penalties ensure that retrieved
content aligns semantically with the current reasoning state
and that the reasoning path remains logically valid across
timesteps. This formulation mirrors the mechanics of systems
such as interleaved retrieval with chain-of-thought reason-
ing [69], GenGround [49], and multi-view multi-hop reasoning
pipelines that integrate feedback-based retrieval throughout
the reasoning chain. Figure 6 visually illustrates the distinc-
tion between naive single-shot RAG and multi-step reasoning
RAG architectures, aligning with the multi-step flow formal-
ized in Algorithm 5.
3.4.1 Structured and Symbolic Reasoning (Code, Tables, and
Knowledge Graphs)
A first family of complex-reasoning RAG systems operates
in structured domains, where the logical form of the data-code, tables, or graphs-forces retrieval to respect task-specific
structureratherthantreatcontentasflattext.Attheprompt
level, UPRISE dynamically selects prompts for knowledge-
intensive NLP tasks, reducing reliance on manual instruction
design and partially structuring downstream reasoning [150].
In code, Liu et al. introduced a hybrid Graph Neural Network
to model local and global code dependencies, integrating
retrieval to improve summarization fidelity and logical coher-
ence [151]. REDCODER reframes retrieval as an emulation of
real-world development workflows rather than a content-fetch
mechanism: by mimicking how programmers consult related
code snippets, documentation, and examples, it improves
both generation quality and task efficiency [152]. Across
these approaches-prompt selection, structure-aware retrieval,
and task-mimetic workflows-the common requirement is close
alignment between the retrieval strategy and the structure of
the problem being solved.
Beyond code, structured reasoning also spans tables and
graphs. T-RAG proposed an end-to-end model that jointly
trained dense retrievers and generative decoders for struc-
tured table inputs, eliminating the need for separate modules
and reducing error propagation, and establishing a template
for tightly coupled retrieval-generation architectures in struc-
tured settings [153]. Across this application space, core rea-
soning limitations in standard LLMs-weak planning, fragile
multi-step inference, and limited self-correction-motivate the
retrieval-augmented and chain-of-thought–style mitigations
discussed throughout this axis. Knowledge-graph ground-
ing addressed some of these limitations directly:Knowledge
Graph-AugmentedLanguageModelsemployedstructuredKG-
based retrieval to offer a more interpretable and verifiable
grounding layer for dialogue generation [154]. This subgroup
also includes dedicated graph-reasoning frameworks such as
G-RetrieverandGRAG, which advance structured ques-
tion answering by aligning retrieval with graph traversal
paths [71, 155]. Across these systems, retrieval is organized
around explicit structure-syntax trees, table schemas, or
graph topology-so that evidence selection mirrors the rela-
tionships the task depends on.
3.4.2 Interleaved and Iterative Retrieval
A second, cross-cutting family makes retrieval part of the
reasoning loop itself, fetching new evidence as intermediate
conclusions evolve-exactly the structure formalized by the
multi-step objective and Algorithm 5. The defining example
isInterleaving Retrieval with Chain-of-Thought Reasoning,
whichallowedretrievaltooccurateachreasoningstep,adapt-
ing context as the model’s understanding evolved and sub-
stantially improving multi-hop performance by fetching new
evidence when earlier assumptions proved insufficient [69];
thisispreciselytheper-stepretrievalr t∼R(ht)andretrieval-
grounded step loss that Algorithm 5 supervises. Related work
pushed iterative refinement further:Tree of Clarifications
introduced a structured decomposition of ambiguous ques-
tions so that RAG systems resolve intent before generating
finalresponses[156],whilezero-shotinformationretrievaland
entailment-aware filtering enabled models to function in low-
supervision settings and reduced the impact of noisy context
injection [157, 158]. Auxiliary memory played a complemen-
tary role: self-memory mechanisms allow LLMs to revisit and
refine previous outputs, improving long-horizon consistency

17
Query
Prompt : How can I start a
business as a student with less
than $500, and scale it to
$10k/month in 1 year?
You can start dropshipping or a
tutoring service. Use your savings
wisely and reinvest in marketing.
a) What are low-capital
business models for
students?
b) How can I market with
no money?
c) What scaling tactics work
in Year 1?Response : Month 1–3: Start a
freelance design gig using Canva
and Upwork (capital: $0–$100).
Month 4–6: Reinvest in portfolio site
and paid ads. Month 7–12: Scale via
referrals + hire 1 contractor . Here's a
breakdown by month…  Retrieves per stepRAG for Complex
reasoning Naive RAG
Fig. 6: Illustration of Normal RAG versus Complex Reasoning RAG. While the Naive RAG pipeline retrieves documents
and generates a direct response to the initial query, the Complex Reasoning RAG decomposes the query into sub-questions,
performs stepwise retrieval, and incrementally builds a structured, multi-step response tailored to complex information needs.
and reducing reasoning drift-directly serving the consistency
regularizerJ(h t,ht−1)that penalizes incoherent jumps be-
tween successive reasoning states [159]. Planning and emer-
gent reasoning behaviors form a further group, withLLM+P
improving planning proficiency by aligning retrieval with op-
timal policy reasoning [160], studies onRetrieve-and-Sample
and long-tail knowledge representation examining how RAG
handles rare facts [161], and work on analogical reasoning
showing that LLMs begin to exhibit human-like analogy
formation when scaled and structured appropriately [162].
A further group scales these loops to longer contexts
and more explicit planning.Retrieval Meets Long Context
LLMsintroducedretrieval-enhancedstrategiesforintegrating
extensive textual input, a necessity for academic research and
case-based analysis [163], andPlanRAGintroduced a struc-
tured plan-then-retrieve methodology that injects decision-
making logic before retrieval so that context selection re-
flects task structure [70].InstructRetrofine-tuned retrieval-
pretrained models via instruction tuning so that retrieval was
informed by model intent rather than static query similar-
ity [164]. Multi-hop reasoning itself was made more flexible:
GenGround: Generate-then-Groundreversed the standard
pipeline by letting the model propose intermediate reasoning
steps that are then verified or corrected through targeted
retrieval-tightening the coherence termC(r t,ht)between re-
trieved context and the evolving reasoning state [49]-while
Unlocking Multi-View Insightsintegrated multiple perspec-
tives during multi-hop retrieval to improve coverage and
contextual coherence [165]. Robustness of the retrieval sub-
strate became a concern as these loops grew more power-
ful:Black-Box Opinion Manipulation Attacksexposed how
retrieval pipelines could be compromised to inject biased
or adversarial information [166, 167], prompting missing-information–guidedretrievalthatidentifiesgapsincurrentev-
idenceandactivelyseekscomplementarydocuments[168]and
dynamic relevance scoring with entailment-aware filtering, as
inDR-RAG, to preserve precision under noisy or adversarial
conditions[169].Layered,multi-intentgenerationwaslikewise
handled iteratively byRichRAG, which decomposes complex
user goals into ordered sub-intents [170].
3.4.3 Domain-Specific Complex Reasoning (Medical, Legal, and
Mathematical)
A third family adapts these reasoning mechanisms to high-
stakes domains whose corpora, conventions, and accuracy re-
quirements demand specialized pipelines. In medicine, LLMs
augmentedwithminimalsupervisionandretrievalaccesshave
been shown to extract clinically relevant data and encode
medical knowledge effectively [171], and knowledge-grounded
conversation benefited from retrieving and fusing external
documents to produce more informative, context-aware dia-
logue that adapts its retrieval choices to unpredictable user
input [172]. Domain adaptability was tackled head-on by
Improving the Domain Adaptation of RAG Models, which
combinedmulti-domainfine-tuningandretrievaloptimization
to improve open-domain question answering across new set-
tings [173], while reading-comprehension–style training was
used to facilitate knowledge transfer across domains [174].
High-stakes and formal domains received targeted treatment:
in mathematics,MATHPROMPTERextended RAG to for-
mal reasoning by retrieving formulas and theorems relevant
tonatural-languageproblems[175];inmedicine,Almanacand
Towards Expert-Level Medical Question Answeringgrounded
retrieval in trusted medical corpora to increase factual ac-
curacy and safety [176]; and in scientific synthesis,PaperQA

18
TABLE 3: Performance metrics of QA-centric Retrieval-Augmented Generation models across multiple evaluation criteria. “-”
indicates data not reported.
Method Dataset/Setting Accuracy/F1 Hallucination /
ConsistencyRetrieval
Effectiveness
FoRAGWebGPT (en), WebCPM
(zh)Factuality:
0.82–0.99Coherence:
0.91–0.98TrainingTime(Holis-
tic): 33.1h
WeKnow-RAG4 domains, classification,
chunk-size evalAccuracy:
0.10–0.41Hallucination:
0.025–0.35Confidence-aware re-
trieval analysis
GenGroundHotpotQA, MuSiQue,
StrategyQAF1: 27.3–52.3 Semantic Acc:
24.7–55.7-
FRAMESInternal multi-hop QA
benchmarkAcc: 0.408–0.729 - Prompting strategy
comparison
AdobeRAGAdobe product corpus nDCG:
0.692–0.822- Dataset coverage
breakdown
Self-RAG(TA-ARE)RetrievalQA (various
LLMs)Match Acc:
6.0–46.4- Retrieval Acc: up to
100%
QA-RAGRAPTOR, HFusion, QR-
FusionBLEU-1: up to
1.33; METEOR:
0.99ROUGE-L: 0.9
(best)Multi-type QA re-
trieval fusion
RQ-RAGARC, POPQA, 2Wiki,
MUSIQUEAcc: 41.7–79.4 - Retrieval Source
Comparison (Wiki,
DDG, Bing)
PRCASQuAD, HotpotQA, Top-
iQCQA- - Contextual Adapter
across 3 QA datasets
Face4RAGSynthetic + Real QA data
(Chinese)- Pos Rate: 30–63% Segment-level evalua-
tion
expanded RAG’s reach into long-context, evidence-based re-
search workflows [177].
Domainspecializationextendstomoredemandingdeploy-
ments. In the medical domain,Development and Testing of
RAGforPreoperativeInstructionGenerationstressedprecise,
interpretable reasoning for high-stakes deployments [178],
andi-MedRAGimproved medical QA by enabling dynamic,
iterative question refinement throughout a reasoning tra-
jectory [179]. In law,CBR-RAGfused retrieval with case-
basedreasoning,retrievinglegalprecedentsandaligningthem
with user queries [180] to emulate expert legal reasoning
patterns [181]. Engineering and education were addressed
bySAPPhIRE modeling with RAG, which supported design-
rationale retrieval and structured explanation [182], and by
Lecture-RAG for Feedback Generation, which tailored feed-
back based on lecture content and student performance [183].
Evaluation kept pace with these domain-specific demands:
FACT, FETCH, AND REASONintroduced FRAMES, a
benchmark for reasoning fidelity across fact-checking, re-
trieval precision, and multi-hop inference [21];DomainRAG
andBenchmarking RAG for Medicinehighlighted the impor-
tance of field-adapted datasets for understanding reasoning
under real-world constraints [184, 185]; andAutomated Exam
Generation for RAG Modelsprobed reasoning depth and ro-
bustnessusingexam-stylequestions[186].Table3summarizes
representative QA-centric systems across these accuracy, con-
sistency, and retrieval-effectiveness dimensions, while Table 8
details retrieval and reasoning metrics for representative legal
and scientific RAG systems such as CaseGPT, CBR-RAG,
LegalBench-RAG, and HyPA-RAG.
Taken together, complex-reasoning RAG spans structured
and symbolic retrieval over code, tables, and graphs; inter-
leaved and iterative retrieval woven into the reasoning loop;
and domain-specific pipelines for high-stakes settings such as
medicineandlaw.Acrossthesefamilies,theunifyingprincipleis retrieval-mediated rather than retrieval-enhanced reason-
ing: structured memory, symbolic representations, multi-view
retrieval,anddynamicfeedbackarecombinedsothatretrieval
participates in each reasoning step rather than only at the
outset, positioning RAG as a central enabler of trustworthy,
adaptive, and structured intelligence.
Key Takeaways & Insights
RAG systems for complex reasoning and multi-step tasks
mark a transition from single-shot factual grounding to struc-
tured, iterative inference that more closely resembles human
problem solving. The objective in Eq. (10) and the multi-
step procedure in Algorithm 5 formalize this paradigm by su-
pervising each reasoning step with retrieval-grounded losses,
enforcing semantic coherence between retrieved context and
intermediate states, and regularizing the consistency of the
reasoning trajectory across timesteps. Together, the methods
surveyed in this axis form a rich design space that blends
symbolic structure, auxiliary memory, dynamic retrieval, and
iterative feedback to support planning, counterfactuals, and
high-stakes decision-making far beyond naive retrieve-then-
read pipelines.
At the same time, current approaches exhibit impor-
tant limitations and characteristic failure modes. Multi-step
architectures are vulnerable to error accumulation: spuri-
ous intermediate steps, poorly decomposed sub-questions,
or misaligned retrieval at early timesteps can cascade into
incoherent final answers, even when individual components
(retriever, generator, planner) perform well in isolation. Re-
trieval triggers and step counts are often heuristic, leading
to over-retrieval that bloats context and under-retrieval that
starveslaterreasoningstagesofcriticalevidence;long-context
methods can still lose track of earlier steps or rely on shallow
pattern matching rather than genuine multi-hop reasoning.
Benchmarks, while increasingly sophisticated, only partially

19
capture real-world requirements such as robustness to adver-
sarially injected evidence, domain shift, and incomplete or
missing information, leaving gaps in how reasoning fidelity
is measured and optimized [166, 168, 169]. For practitioners,
effective use of complex-reasoning RAG entails constraining
the number of reasoning steps, instrumenting systems to log
and inspect intermediate states, and combining interleaved
retrieval with explicit decomposition, clarification, and verifi-
cation(e.g.,missing-information–guidedretrieval,entailment-
aware filtering, multi-view evidence checks). It is crucial to
tuneretrievalandplanningjointly,prioritizetrusted,domain-
specific corpora in high-stakes applications, and evaluate on
multi-hop, domain-adapted benchmarks to ensure that added
architectural complexity translates into more reliable, inter-
pretable, and safe reasoning rather than merely longer chains
of brittle steps.
4 Applications of RAG-LLMs
The applications discussed in this section span all four axes
of the taxonomy developed in §3: open-domain QA benefits
primarily from compression and efficiency techniques (§3.6)
and complex reasoning architectures (§3.9); code generation
leverages interactive, retriever-generator alignment strategies
(§3.8); and educational and corporate deployments require
both defensive safeguards (§3.7) and personalization from
interactive RAG (§3.8). Within each subsection, we dis-
tinguish betweenresearch benchmark systems-which demon-
strate RAG capabilities on controlled evaluation tasks-and
deployed applications, which face real-world constraints of
reliability, privacy, latency, and domain specificity. We or-
ganize our discussion along a spectrum that progresses from
canonicalresearchbenchmarkstocross-domaingeneralization
and finally to production systems. Open-domain question
answering (§4.1) remains the foundational use case: RAG
was originally introduced to address knowledge-intensive QA
tasks [45], and the ODQA literature consequently repre-
sents the most mature body of retrieval-augmented meth-
ods, evaluation protocols, and failure-mode analyses. Code
generation (§4.2) serves as a critical test of RAG’s ability
to generalize beyond natural-language prose into structured,
syntacticdomainswhereretrievalmustrespectprogramming-
language semantics and repository-level context [187, 188].
Finally, educational and corporate deployments (§4.3) cap-
ture RAG’s transition from research prototype to produc-
tioninfrastructure-encompassingintelligenttutoringsystems,
enterprise knowledge management, and institutional work-
flows that must satisfy reliability, privacy, and scalability
constraints absent in benchmark settings [189, 190]. Together
these three areas span the full arc of RAG application ma-
turity while remaining tractable for substantive quantitative
comparison.
4.1 Information Retrieval and Question Answering
Research Systems.Open-domain Question Answering
(ODQA) has seen substantial improvement through the in-
tegration of RAG techniques with Large Language Models
(LLMs).Muchoftherecentworkfocusesonenhancinganswer
accuracy, factual grounding, and coherence while reducing
hallucinations. FoRAG improves long-form QA by combining
an outline-enhanced generator with a doubly fine-grainedRLHF framework to address factual inaccuracies and logical
inconsistencies [191]. WeKnow-RAG incorporates Web search
and Knowledge Graphs to strengthen retrieval robustness,us-
ing multi-stage retrieval and self-assessment to minimize hal-
lucinations [192]. The Generate-then-Ground (GenGround)
framework alternates between answer deduction and knowl-
edgegrounding,offeringamoreflexibleapproachtomulti-hop
reasoningcomparedtorigidretrieve-then-readpipelines[193].
Collectively, these systems highlight a trend toward richer re-
trieval pipelines that dynamically refine context and improve
factual reliability in ODQA.
Benchmarks and Evaluation.Alongside model innova-
tion, recent progress has produced comprehensive evalua-
tion frameworks and datasets tailored for RAG-based QA.
FRAMES (Factuality, Retrieval, And reasoning MEasure-
ment Set) provides a unified benchmark for assessing factual
accuracy, retrieval precision, and reasoning fidelity across
challenging multi-hop questions [21]. A domain-specific RAG
framework for Adobe products addresses the limitations of
general-purpose models by leveraging retrieval over propri-
etary corpora and user-behavior data [194]. RetrievalQA,
a benchmark of 1,271 questions, evaluates Adaptive RAG
(ARAG) methods and introduces Time-Aware Adaptive Re-
trieval (TA-ARE), which adjusts retrieval depth without ad-
ditional fine-tuning [195]. These resources reflect a growing
emphasis on systematic evaluation and highlight the need for
QA systems that integrate knowledge more effectively, refine
reasoning steps, and operate reliably across diverse contexts.
Beyondevaluationframeworks,severalrecentsystemstar-
get improvements in context representation, refinement, and
factualconsistency.QA-RAGenhancescontextstructuringby
transforming retrieved evidence into question–answer pairs,
reducing hallucinations and improving grounding [196]. RQ-
RAG introduces dynamic query refinement to better handle
complex or ambiguous questions, yielding performance gains
in both single-hop and multi-hop QA tasks [197]. PRCA
proposes a Pluggable Reward-Driven Contextual Adapter
that filters and restructures retrieved information before
it is passed to a black-box LLM, improving ReQA per-
formance without full-model fine-tuning [198]. Face4RAG
provides a benchmark for evaluating Chinese RAG sys-
tems and introduces L-Face4RAG for detecting logical fal-
lacies and assessing reasoning quality [199]. These develop-
ments collectively demonstrate increasing attention to re-
trieval–generation alignment and error detection across lan-
guages and domains, as well as the need for methods robust
to varied error distributions.
Table 3 presents a consolidated view of RAG mod-
els designed for open-domain and multi-hop QA. Systems
such as FoRAG and QA-RAG demonstrate strong factu-
ality and BLEU-based performance, while frameworks like
WeKnow-RAG emphasize hallucination mitigation and con-
fidence calibration. Table 4 complements this view with per-
formance data for RAG systems designed specifically for
security- and fairness-sensitive settings, such as Poisone-
dRAG, BadRAG, and FairRAG, illustrating the additional
safeguards required when QA is deployed in adversarial or
regulated environments-concerns that directly connect to the
Defensive RAG axis discussed in §3.7. GenGround and RQ-
RAG highlight the importance of iterative retrieval strate-
gies for complex question decomposition, and Self-RAG and

20
TABLE 4: Overview of Security, Fairness, and Privacy-focused RAG Systems
Model Datasets/EvaluationContext KeyMetrics/Findings
PoisonedRAG NQ, HotpotQA, MS-MARCO ASR: 0.97–0.99; F1-Score: 0.96–1.00
RAG (CoCondenser + MiniLM) Government, Education, Society, Health ASR: 0.17–0.50; ASV: –0.17 to 0.67
RC-RAG Internal eval on ChatGPT and Mistral Risk: 14.94–19.00; Carefulness: 52.87–65.37
Towards Fair RAG Exposure Disparity Benchmarks EE-D: 0.14; EE-R: 0.28
LLaMA2-7B-Chat Health, Enron Attacks ROUGE Prompts: 73–111; Repeat Contexts: 55–135
BadRAG GPT-4, Claude-3 Evaluation Retrieval Success: 98.9%; Rejection Rate: 74.6%
SAGE HealthcareMagic, Wiki-PII BLEU-1: 0.01–0.11; ROUGE-L: 0.02–0.09
TA-ARE showcase retrieval improvements through filtering,
thresholding, and temporal adaptation. Across benchmarks
suchasHotpotQA,StrategyQA,SQuAD,anddomain-specific
corpora, recent results reflect a clear trend toward tighter re-
trieval–generation coupling and improved context-aware rea-
soning. This diversity underscores the expanding landscape
of QA-centric RAG systems and the wide range of evaluation
methodologies used to study them.
Looking forward, several challenges remain central for the
next generation of RAG-based QA systems. These include
the integration of heterogeneous knowledge sources, improved
robustness in query reformulation, and consistent factual
grounding across diverse tasks and domains. As real-world in-
formation needs become more dynamic and domain-sensitive,
future RAG systems must offer stronger reasoning capabil-
ities, more adaptive retrieval pipelines, and more reliable
calibration under uncertainty. Continued progress in these
directions will be essential for building trustworthy, high-
utility ODQA systems capable of supporting both general-
purpose and specialized information retrieval.
4.2 Retrieval-Based Code Generation
RAG has also demonstrated strong utility in code generation
and synthesis, particularly in scenarios involving domain-
specific languages, rare programming patterns, or large, het-
erogeneous codebases [200]. This subsection reviews both
deployed, user-facing coding assistants and research systems
that are not themselves shipped products but demonstrate
the application potential of retrieval-augmented techniques
in this domain; we distinguish between the two explicitly
below. By augmenting LLMs with relevant code snippets,
documentation, and usage examples, RAG systems improve
context awareness and help models generalize beyond memo-
rized patterns. This is especially valuable in zero-shot or few-
shot settings where the model lacks extensive prior exposure
to a particular API or language.
Deployed Applications.GitHub Copilot [201], Amazon
CodeWhisperer, and ChatGPT-based [1] coding assistance
are the most widely used production tools in this space. As
Table 5 reflects, these systems are characterized by product-
level features-IDE integration, language coverage, reference/-
explanation support, and pricing-rather than the held-out
quantitative benchmarks reported for research systems, since
their evaluation criteria for end users are usability and cover-
age rather than benchmark accuracy.
ResearchSystems.ArepresentativesystemisProCC[202],
which combines prompt-based retrieval with a contextual
multi-armed bandit algorithm to dynamically choose amongmultiple semantic perspectives of source code. By leveraging
a multi-retriever architecture and adaptive retrieval policies,
ProCC improves alignment between retrieved examples and
userintent.Itdemonstratessignificantperformancegainsover
strong baselines and generalizes effectively across both open-
source and proprietary codebases, highlighting the value of
retrieval diversity in code-related tasks. Retrieval has also
proven valuable for synthesizing programs in rare or domain-
specific languages. A combined RAG and Few-Shot Learning
(FSL) method [203] retrieves structurally similar examples
from DSL repositories, enabling models to generate correct
syntaxandsemanticswithminimalsupervision.Thisstrategy
improves performance in niche development settings where
annotated examples are scarce.
CodeT5+[204], a flexible encoder–decoder architecture,
further demonstrates the potential of retrieval-enhanced code
models. Incorporating span denoising, contrastive learning,
text–codematching,andcausallanguagemodeling,CodeT5+
achieves strong results across tasks such as code completion,
defectdetection,summarization,andmulti-stepreasoning.Its
unified design exemplifies how retrieval-aware training can
improve downstream generalization in diverse code domains.
Table 5 summarizes representative retrieval-based code gen-
eration systems; because it spans both commercial coding
assistants and research systems, the “Attribute / Metric”
column intentionally mixes two kinds of entries: product-level
features (e.g., IDE support, pricing, training data source) for
tools such as GitHub Copilot and CodeWhisperer, and quan-
titative performance metrics (e.g., EM, BLEU, pass@k) for
research systems such as ProCC and CodeT5+. Foundational
tools such as GitHub Copilot and CodeWhisperer emphasize
accessibility,IDEsupport,andbroadlanguagecoverage,while
newer systems such as ProCC and CodeT5+ deliver stronger
exact-match and reasoning performance via task-aware re-
trieval and training objectives. Retrieval-enhanced few-shot
synthesis for uncommon DSLs also shows substantial error
reduction with GPT-4 as shot count increases, underscoring
its applicability in low-resource scenarios. The diversity of
reported metrics-from BLEU and pass@k to clone detection
F1-highlights the multifaceted evaluation landscape in code
generation research.
Future work in this area includes improving retrieval qual-
ityfornoisyorsparselydocumentedrepositories,enablingon-
the-fly index updates for fast-changing codebases, and ensur-
ingprivacyinproprietarydeveloperenvironments.Additional
goals involve bridging the performance gap between open-
source and closed-source models, developing retrieval mecha-
nisms that reason across multi-file and multi-library contexts,
and unifying RAG with program verification frameworks to

21
TABLE 5: Comparison of Retrieval-Based Code Generation Systems and Metrics
Model Category Attribute / Metric Score / Description
GitHub CopilotIDE Support Supported IDEs IntelliJ, VSCode, PyCharm, etc.
Reference / Explanation Provides References No
Reference / Explanation Explains Suggestions No
Suggestion Variety Options Returned Up to 10
Training Source Data Public Repositories
Languages Supported Best With C, C++, Java, Python, etc.
Accessibility Offline / Local Access No / Yes
Release Info Developer / Release OpenAI–Microsoft / Oct 2021
Pricing Subscriptions Free for Students; $10–$19/month
Amazon CodeWhispererIDE Support Supported IDEs JetBrains, VS Code, AWS Cloud9
Reference / Explanation Provides References Yes
Suggestion Variety Options Returned Up to 5
Accessibility Offline / Local Files No / Yes
Release Info Developer / Release AWS / June 2022
ChatGPT [1]IDE Support Supported IDEs None
Reference / Explanation Explains Suggestions Yes
Suggestion Variety Options Returned One per request
Training Source Data GitHub, GitLab, Codex
Accessibility Offline / Local Files No / No
Release Info Developer / Release OpenAI / Nov 2022
ProCCAccuracy EM (Open-Source / Private) 8.6% / 10.1%
Fine-Tuning Gain Gain After FT 5.6%
Model Performance CodeLlama EM / ES 54.66 / 75.85
Model Performance StarCoder EM / ES 49.14 / 72.69
RAG + Few-Shot for DSLsGPT-3.5 Error Rate 1–4 Shots 100%, 100%, 96%, 73%
GPT-4 Error Rate 1–4 Shots 40%, 49%, 42%, 29%
CodeT5+Code Completion pass@1 / @10 / @100 35.0%, 54.5%, 77.9%
Math QA (Python) pass@80 / @100 87.4%, 73.8%
GSM8K (Python) pass@80 / @100 73.8%, 87.4%
Summarization BLEU-4 33.83%
Completion Exact Match 44.86%
Retrieval MRR (Text-to-Code) 77.4
Defect Detection Accuracy 66.7%
Clone Detection F1 Score 95.0%
enhance correctness and usability of generated code.
4.3 Educational and Corporate Use Cases
Applications.RAG-LLMs are increasingly adopted in edu-
cational settings for personalized tutoring, automated feed-
back, and intelligent content generation. By combining re-
trieval with generative modeling, these systems can adapt
to individual learning needs, fill contextual knowledge gaps,
and ground explanations in curricular material. In automated
tutoring, RAG has been used to assess social-emotional com-
petencies in tutors, generate personalized feedback for pro-
gramming learners, and support children with developmental
disabilities.Forexample,PicotTointegratesofflineLLMswith
RAG-basedretrievaltodelivertailorededucationalcontentto
children with learning challenges [205]. Similarly, frameworks
such asRASTenable automated question generation and
exam preparation, producing diverse and targeted items that
scale instructional support [206]. These systems collectively
illustrate how RAG can enhance both formative and summa-
tive assessment workflows.Benchmarks and Evaluation.Table 6 highlights the per-
formance and cost characteristics of RAG-based educational
systems. The LLaMA2 baseline achieves strong precision and
accuracy on basic classification tasks, while RAG-enhanced
GPT-4 setups for lecture feedback show rapid response times,
grounded explanations via linked lecture segments, and posi-
tivestudentsatisfaction,trust,andusabilityscores.Costcom-
parisonsacrosspromptingstrategiesindicatethatRAG-based
prompts can be significantly more economical than zero-shot
or more elaborate reasoning prompts, particularly for large
models such as GPT-4. Results from Xwin-LM-70B further
suggest that high-capacity models can achieve near-perfect
performance on targeted instructional tasks, reinforcing the
viability of RAG-LLMs as scalable educational assistants.
Applications.In corporate environments, RAG is increas-
ingly used for intelligent knowledge management over hetero-
geneous, domain-specific data. Recent work explores embed-
ding fine-tuning, vector space segmentation, and LLM-based
reranking to optimize retrieval from enterprise corpora and
reduce hallucinations in high-stakes business queries [207].

22
TABLE 6: Evaluation of RAG-based and LLM-enhanced systems in educational and corporate contexts
Model Metric Details
LLaMA2Precision 89%
Recall 84.5%
Accuracy 85%
RAG (Cost across prompting strategies)Zero-shot Prompt Type I (GPT-3.5
Turbo)$0.100
Zero-shot Prompt Type II (GPT-3.5
Turbo)$0.014
Tree of Thoughts Prompt (GPT-3.5
Turbo)$0.013
RAG Prompt (GPT-3.5 Turbo) $0.008
Zero-shot Prompt Type I (GPT-4
Turbo)$1.035
Zero-shot Prompt Type II (GPT-4
Turbo)$0.188
Tree of Thoughts Prompt (GPT-4
Turbo)$0.137
RAG Prompt (GPT-4 Turbo) $0.137
RAG with GPT-4 (Lecture Feedback System)Avg. Feedback Time (with lecture) 18 seconds
Avg. Feedback Time (without lecture) 1–2 seconds
Linked Lecture Segments 160 segments
Avg. Segments per Feedback 1.67
Student Satisfaction (Length) Neutral to Agree
Student Trust (Accuracy) Neutral to Agree
System Usability Score 74.8
Xwin-LM-70B-V0.1-GPTQ Question-wise Accuracy Q1–Q6: 100%, Overall: 100%
These methods improve contextual accuracy in enterprise
search, analytics, and decision-support workflows by ground-
ing responses in internal documents rather than generic web
data.Innovationssuchaselement-basedchunkingforfinancial
documents [208] and automated form-filling pipelines [209]
demonstrate how RAG can process structured and semi-
structured content, enabling downstream applications in le-
gal, financial, and operational domains.
Benchmarks and Evaluation.Table 7 illustrates the
growing maturity of RAG systems for corporate knowledge
management. TheRASTframework demonstrates measur-
able BLEU improvements across QA datasets, indicating the
benefits of style-aware augmentation. Thebge-large-en-v1.5
model achieves consistently high NDCG and Accuracy@k
scoresacrossseparateandcombinedvectorspaces,underscor-
ing the impact of embedding design in complex document
corpora. DS-RAG emphasizes flexibility in multi-provider,
policy-aware environments, where retrieval architecture must
stay consistent even as infrastructure changes. TheChipper
framework provides the most granular view of RAG behavior
over financial reports [208], showing how different chunking
strategies affect retrieval accuracy, ROUGE/BLEU scores,
and Q&A performance relative to GPT-4 and human base-
lines.
These findings aim to bring to light the importance of
hybrid chunking strategies, structured content modeling, and
robust embedding spaces in enterprise-grade RAG deploy-
ments. Ongoing research explores using LLMs as early-stage
classifiers for query routing, refining data comparison tech-
niques for chunk selection, and improving retrieval efficiency
in time-sensitive decision pipelines. As organizations scale
their digital infrastructure, RAG offers a principled frame-work for transforming enterprise search and knowledge access
into an intelligent, context-aware service.
5 Challenges and Open Problems in RAG-LLMs
The remainder of the paper forms a single, consolidated dis-
cussion, presented as four complementary lenses on the same
material rather than as independent sections. This section
catalogs the openchallengesthat limit current systems; §6
maps those challenges to concreteresearch directions; §7
translates the resulting understanding intoactionable guid-
ancefor practitioners; and §8 synthesizes the overall outlook.
We retain these as separate labeled sections so that readers
can navigate directly to challenges, directions, or guidance,
but they are intended to be read together as the survey’s
discussion of where the field stands and where it should go-
each organized, like the taxonomy, around the four research
questions RQ1-RQ4.
The challenges facing modern RAG systems directly limit
factual reliability, adaptability, and safe deployment across
the four major axes of the field: efficiency-driven retrieval,
defensive and safety-aware methods, user-centric personal-
ization, and complex multi-step reasoning. These difficulties
revealstructuralgapsinhowretrieval,generation,andreason-
ingpipelinesinteractunderreal-worldconstraints.Mappedto
the research questions introduced in §1: retrieval bottlenecks
(§5.1) and scalability (§5.4) expose the limits of current
answers toRQ1; hallucination and reliability issues (§5.2)
and explainability gaps (§5.5) cut acrossRQ1,RQ2, and
RQ4; domain adaptation and personalization failures (§5.3)
directly challengeRQ3; and safety, privacy, and adversarial
vulnerabilities spanRQ2throughout. Together, these chal-

23
TABLE 7: Metrics and Evaluation Results for RAG-based Corporate Knowledge Management Systems
Model Metric ValuesandEvaluationDetails
RAST (Style Transfer)BLEU Scores (SQuAD/1) Top-1: 19.25, Oracle: 23.23, Pairwise: 48.91, Overall: 9.14
BLEU Scores (SQuAD/2) Top-1: 19.36, Oracle: 22.59, Pairwise: 56.42, Overall: 7.75
BLEU Scores (NewsQA) Top-1: 11.02, Oracle: 16.26, Pairwise: 23.16, Overall: 7.74
bge-large-en-v1.5NDCG@10 0.85
Accuracy@k (Combined vs. Separate DBs) k=1: 0.85 / 0.91, k=5: 0.92 / 0.95, k=10: 0.93 / 0.97
Training/Test Split Written: 1450 / 500, Transcribed: 1450 / 500
Corpus Articles: 165 (2004 chunks), Transcribed Data: 59 (4167 chunks)
DS-RAGRetrieval Latency Network-dependent
Embedding Consistency Same across providers
Ranking Time Tech-dependent
Similarity Accuracy High (model-dependent)
Chipper (Financial Reports)Element Distribution NarrativeText: 61,780; Title: 29,664; Tables: 7,700; etc.
Chunking Stats Base128: 64k chunks (Mean 800), Base256: 32k, Base512: 16k, Chipper: 20.8k
Retrieval Accuracy (Page Level) Base128: 72.34%, Base256: 73.05%, Base512: 68.09%, Aggregated: 83.69%
Retrieval ROUGE/BLEU Base128: 0.383 / 0.181, Base256: 0.433 / 0.231, Base512: 0.455 / 0.250
Chipper Aggregated Scores ROUGE: 0.568, BLEU: 0.452, Accuracy: 84.40%
Keywords Chipper Scores ROUGE: 0.444, BLEU: 0.315, Accuracy: 46.10%
Summary Chipper Scores ROUGE: 0.473, BLEU: 0.350, Accuracy: 62.41%
Prefix+Table Chipper Scores ROUGE: 0.514, BLEU: 0.400, Accuracy: 67.38%
Q&A Results (Base128) No answer: 35.46%, GPT-4: 29.08%, Manual: 35.46%
Q&A Results (Base256) No answer: 25.53%, GPT-4: 32.62%, Manual: 36.88%
Q&A Results (Base512) No answer: 24.82%, GPT-4: 41.84%, Manual: 48.23%
Q&A Results (Keywords) No answer: 22.70%, GPT-4: 43.97%, Manual: 53.19%
Q&A Results (Summary) No answer: 17.73%, GPT-4: 43.97%, Manual: 51.77%
Q&A Results (Prefix+Table) No answer: 20.57%, GPT-4: 41.13%, Manual: 53.19%
lengeshighlighttheneedformoreunifiedretrieval–generation
objectives, robust uncertainty modeling, adaptive user- and
domain-aware retrieval strategies, and stronger safeguards to
ensure that next-generation RAG systems remain reliable,
interpretable, and resilient.
5.1 Retrieval Bottlenecks
One of the most pressing limitations in RAG systems lies in
the retrieval pipeline, particularly when operating over large-
scale or heterogeneous knowledge bases. The recurring failure
modes are:
•As dataset sizes grow, retrieval efficiency becomes a
major bottleneck, often resulting in increased latency,
reduced relevance of retrieved results, and diminished
system scalability-especially in enterprise and real-time
applications where timely and accurate responses are
critical [210, 83, 211].
•Traditional retrieval methods, such as dense or sparse
vector search, can degrade in performance under high
retrieval loads or when faced with domain shifts, forcing
practitioners into difficult trade-offs between retrieval
accuracy and computational efficiency [212, 213]; more
sophisticated retrievers typically yield higher precision
butincurgreaterinferencecosts,whichcanbeprohibitive
at scale.
•Optimizing the size and granularity of text chunks re-
mains a difficult balancing act: large chunks may provide
richer context but introduce semantic noise and redun-
dancy, while small chunks can fragment information and
reduceeffectiverelevance[214,215];theseissuesarecom-
pounded in heterogeneous and dynamic corpora, where
documents vary widely in structure and length.
•Retrieval bottlenecks are further exacerbated by the
limitations of static indexing in evolving knowledge en-
vironments: indices are often expensive to rebuild, andmany systems lack efficient mechanisms to update or re-
rankindiceswithoutextensivereprocessing[83,216],and
current work still lacks adaptive, self-updating retrieval
mechanisms that preserve relevance and efficiency as
corpora change over time.
•In deployment, this leads to outdated retrieval outputs,
higher latency, and reduced reliability-problems that be-
come especially severe in enterprise, time-sensitive, and
mission-critical applications.
5.2 Hallucination and Reliability Issues
AlthoughRAGframeworkssignificantlyreducehallucinations
in large language models by grounding responses in retrieved
evidence, reliability remains a persistent concern:
•Hallucinations still occur when the retrieved content is
irrelevant, outdated, or misleading; because generation
is conditioned on this context, inaccuracies in retrieval
propagate into the final response and can create a false
sense of factuality [217, 211, 218].
•Even with rich external sources, incomplete or noisy
knowledge bases can lead to erroneous answers, partic-
ularly in open-domain settings; LLMs also struggle to
interpret subtle nuances or contradictions in retrieved
text, sometimes producing overly confident responses
that misrepresent the evidence [219, 216].
•Integrating retrieved content fluently and coherently into
generated outputs is itself a challenge: misalignment be-
tween retrieved evidence and the model’s internal priors
canyieldunnaturalphrasing,semanticinconsistencies,or
selective use of evidence that omits key qualifiers [220].
•In cases where user queries extend beyond the scope or
coverage of the knowledge base, RAG systems may even
underperform relative to purely parametric LLMs, be-
causetheirdependencyonexternalcontentcanconstrain
generative flexibility [221].

24
TABLE 8: Evaluation of RAG Models for Scientific and Legal Text Processing
Model Setting/Task Precision/Recall/F1 Other Metrics /
Notes
CaseGPT Medical, Legal Domain
QAMedical:Precision@10:
0.90, F1: 0.89
Legal:Precision@10:
0.93, F1: 0.91MRR: 0.92 (med),
0.94 (legal)
NDCG@10:
0.91/0.93
Human Scores:
Quality: 4.3,
Relevance: 4.5
CBR-RAG Case Law Legal QA – Top Statutes: Fed.
Court Rules (9),
Civil Aviation (8),
etc.
Improves grounding
using legal precedent
LegalBench-RAG Benchmark Evaluation – Q&A Counts:
CUAD (4042),
MAUD (1676), etc.
Total Q&A: 6858
HyPA-RAG Hybrid Retrieval +
AdaptationContext Recall: 0.9046
Faithfulness: 0.8430
F1 Similarity: 0.8621Correctness (1-5):
4.25
PA-Class (2/3):
0.90/0.89
Answer Relevancy:
0.79 / 0.77
•Ongoing work explores dynamic context filtering,
entailment-aware generation, hallucination detection
mechanisms, and training objectives that better align
factual grounding with semantic fluency [222, 223], but
a major open problem is the absence of unified, fine-
grained frameworks that jointly model retrieval quality,
hallucination detection, and generation-time factual ver-
ification; most systems still treat retrieval and generation
as loosely coupled components, limiting their ability to
assess evidence reliability or correct misleading context
during inference.
5.3 Domain Adaptation and Personalization
Adapting RAG systems to specialized domains such as sci-
entific research, finance, or medicine introduces unique chal-
lenges for both retrieval and generation, and personalization
adds a further layer of complexity:
•Distribution shifts between general-purpose LLMs and
domain-specificlanguageoftenleadtofailuresinextract-
ing relevant information or correctly interpreting spe-
cialized terminology, which in turn degrades factuality,
coherence, and task relevance [224, 225].
•Onepromisingdirectioninvolvesself-trainingapproaches
that jointly develop question answering and genera-
tion capabilities tailored to a target domain: by fine-
tuning LLMs on instruction-following, search-centric,
and domain-specific QA tasks-and then prompting them
to generate domain-relevant queries over unlabeled
corpora-researchers have shown that iterative feedback
loops can improve performance without exhaustive man-
ual annotation [226].
•Complementary frameworks such asRAG-end2end
jointly train the retriever and generator on domain-
specific datasets, allowing all components, including the
knowledge index, to adapt in unison to specialized re-
trieval demands [227].•Tailoring RAG systems to individual users or enterprise
use cases requires fine-grained control over retrieval con-
tent, indexing granularity, and context-aware prompt-
ing, often under strict security and compliance con-
straints;inenterprisedeployments,systemsmustsupport
safe access to proprietary information, raising concerns
about data leakage from stored embeddings or vector
databases [228, 229].
•To mitigate these risks, researchers advocate security-
first RAG designs that incorporate access-controlled re-
trieval, embedding encryption, and AI-driven data clas-
sification at ingestion time [230, 231], and advanced
implementations support multi-dimensional access con-
trol based on user roles, data sensitivity, and contextual
business relevance [232].
•Despite progress, current work still lacks unified frame-
works that can simultaneously adapt retrieval, genera-
tion, indexing, and personalization signals across het-
erogeneous domains and user profiles, leading to brittle
performance under domain shift, inconsistent personal-
ization quality, and heightened privacy or compliance
risks.
5.4 Scalability and Latency Constraints
Scalability and latency present persistent bottlenecks in real-
time RAG systems:
•As underlying knowledge bases expand, the compu-
tational overhead required to retrieve and rank rele-
vant content increases, and even modest delays can
disrupt user experience or degrade decision quality in
settings such as customer support [233], healthcare, or
autonomous systems [234].
•Recent efforts explore distributed and federated retrieval
as practical strategies for handling large-scale data en-
vironments: distributed retrieval architectures shard the

25
corpus across multiple nodes, enabling parallel search
and improved scalability, while federated retrieval ex-
tends this paradigm by querying decentralized and het-
erogeneous data sources without centralizing storage-a
crucial capability in enterprises with siloed knowledge
assets [235].
•These architectures, however, introduce new engineer-
ing challenges: ensuring consistency across distributed
caches, managing load balancing, and maintaining syn-
chronized updates are all non-trivial, especially when
corpora change frequently [236].
•Integrating generation into a distributed RAG setting
often requires asynchronous processing and sophisticated
caching strategies to avoid new bottlenecks in the infor-
mation flow; to reduce retrieval latency without sacrific-
ingaccuracy,researchersemployadvancedindexingtech-
niques, Approximate Nearest Neighbor (ANN) search,
pre-fetching, and contextual pre-filtering to narrow can-
didate sets before full scoring [237].
•Yet a key gap remains: there are few end-to-end scalable
RAG architectures that jointly optimize retrieval, index-
ing, and generation under real-world latency budgets. In
practice, treating retrieval as the sole bottleneck can lead
to unpredictable response times and degraded reliability
once downstream synchronization and coordination costs
are taken into account.
5.5 Explainability and Interpretability
One of the core promises of RAG systems is improved trans-
parency over traditional black-box LLMs, owing to their
ability to explicitly reference external sources:
•By anchoring responses in retrieved content, RAG sys-
temsallowuserstotracetheoriginsofgeneratedoutputs,
verify supporting documents, and understand how infor-
mation was synthesized [238, 83]. This provenance-aware
generation is particularly valuable in regulated domains
such as healthcare, finance, and law, where justification
andauditabilityareessential[27,224,83],makingRAGa
natural candidate for applications that require both high
performance and clear evidence trails.
•Evaluation metrics for RAG explainability have evolved
beyond traditional retrieval benchmarks to include con-
textual precision and contextual recall, which assess not
only relevance but also the semantic fit of retrieved
context within the generation pipeline [102, 239].
•Standard tools such as Mean Reciprocal Rank (MRR)
and Mean Average Precision (MAP) remain important
for retrieval diagnostics, while newer frameworks like
FRAMES [21] and Face4RAG [199] introduce hallucina-
tion detection and logical consistency scoring for end-to-
end system integrity.
•To further improve interpretability, researchers have pro-
posed visualizing attention flows between retrieved doc-
uments and generated tokens, enabling more intuitive
debugging and model introspection [102]; some systems
also integrate rule-based reasoning or knowledge graph
overlays to constrain generation within domain-specific
bounds,reducingerrorpropagationandsupportingsemi-
symbolic inference [240].
•Despite these advances, RAG systems remain susceptible
to opaque behaviors when retrieval fails, when LLMsoverride retrieved facts, or when evidence is selectively
used; ongoing work therefore investigates integrating ex-
plainability into training objectives, employing counter-
factual prompting, and designing interactive user inter-
facesthatexposethedecisionpathtakenbyRAGmodels
in real time [224, 239].
•A significant open problem is the absence of standard-
ized, fine-grained frameworks that unify retrieval trans-
parency, reasoning traceability, and generation-level at-
tribution into a single interpretable pipeline. In deploy-
ment, this fragmentation hinders trust, auditability, and
error analysis-especially in regulated or high-stakes envi-
ronments where stakeholders must understand not only
what the model retrieved, but how that evidence shaped
its final reasoning process.
6 Future Directions
As RAG systems gain traction across academic, industrial,
and applied domains, several promising directions are emerg-
ing to address current limitations and expand the utility of
RAG-enhanced language models. This section outlines four
research frontiers that can shape the next generation of RAG
architecturesandconnecttheaxesinourtaxonomy-efficiency,
defense, interactivity, and complex reasoning-to longer-term
system design.
6.1 RAG with Reinforcement Learning
Addresses:RQ1(accuracy–cost trade-off under constrained
inference) andRQ4(preventing error accumulation in multi-
step pipelines). Current systems answer RQ1 and RQ4 only
partially: retrieval policies are largely static and local, not
jointlyoptimizedwithgenerationobjectivesormulti-steptask
rewards.
A growing direction is the integration of reinforcement
learning (RL) into RAG pipelines to jointly optimize retrieval
and generation decisions. Most current systems treat retrieval
as a static, pre-tuned process that is only weakly aligned
with downstream objectives. RL instead provides a principled
framework to learn retrieval and generation policies from
rewardsignalstiedtotaskperformance,factualaccuracy,user
satisfaction, or safety. Early demonstrations of this approach-
such as the RLHF-based generation loop in FoRAG [191] and
the trust-driven retrieval policy in TrustRAG [39]-show that
aligning retrieval with reward signals substantially reduces
hallucinations and improves factual grounding over static
retrieval baselines.
In RL-based RAG, retrieval policies can be trained to
select documents based on expected downstream utility, to
re-rank candidates using feedback from the generator, or to
decide adaptively when retrieval is needed at all. Techniques
such as reward shaping, inverse RL, and curriculum learning
could guide both retriever and generator toward globally
beneficial behaviors in multi-step or multi-hop tasks, rather
than optimizing only local relevance scores.
Combining RL with offline data such as logged user inter-
actions, historical retrieval traces, or prior evaluation signals
may enable fine-tuning without extensive new annotation.
As real-world deployments increasingly demand adaptive rea-
soning and low hallucination rates, RL-based RAG frame-
works are likely to become an important tool for aligning

26
retrieval strategies, generation style, and safety constraints
withapplication-levelgoals.Intermsoftheformalframework,
RL replaces the static retrieval distributionp θ(di|q)in
Eq. (7) and the fixed selection policies of Algorithms 2-5 with
reward-driven policies, and recasts the per-step term of the
reasoning objective (Eq. (10)) as a return to be maximized
rather than a divergence to be minimized; the state-update
step of Algorithm 5 becomes the natural locus for a learned
value or reward model.
6.2 Neuro-Symbolic Integration in RAG
Addresses:RQ4(multi-hop reasoning and error accumula-
tion) andRQ2(verifiable, constraint-satisfying retrieval in
high-stakes domains). Current graph-based systems advance
RQ4 but structured symbolic verification integrated end-to-
end with retrieval and generation remains largely unsolved.
While current RAG models rely primarily on dense or
sparse vector retrieval, they often lack the structured reason-
ing capabilities required for complex decision-making. Neuro-
symbolic approaches-where neural models are augmented
with symbolic inference engines, rule-based logic, or knowl-
edge graphs-offer a path toward stronger factual grounding,
constraint satisfaction, and interpretability. Current work has
begun exploring this direction through graph-augmented re-
trieval systems such as GraphRAG [27], G-Retriever [71], and
GRAG [155], as well as knowledge-graph-grounded dialogue
systems [154] that use structured KG traversal to constrain
and verify generation.
Integrating symbolic modules into RAG could enable sys-
tems to perform consistency checks, apply domain rules, and
conduct multi-hop reasoning over structured representations.
Ontology-guided retrieval, schema-aware indexing, or logical
filtering can help align generation with legal, scientific, or
safety constraints, especially in domains such as compliance,
scientificdiscovery,orsoftwareengineering.Symbolicmemory
components may also support long-horizon reasoning by stor-
ing and manipulating intermediate knowledge states across
turns.
Hybrid neuro-symbolic RAG architectures have the po-
tential to produce more explainable responses by pairing free-
form language with structured justifications, citations, or rule
traces. Progress in graph retrieval, semantic parsing, differ-
entiable reasoning, and rule-aware generation will be essen-
tial for scaling these capabilities to high-stakes settings that
demand verifiable outputs and clear guarantees. Concretely,
symbolic components would be injected into the coherence
and consistency functionsC(r t,ht)andJ(h t,ht−1)of the
reasoning objective (Eq. (10))-replacing soft differentiable
penalties with rule- or ontology-based checks-and into the
document-selection steps of Algorithms 2-4, where a logical-
constraint filter would gate documents before generation; in
the defensive objective (Eq. (8)) the same machinery could
supply a verifiable surrogate for the toxicity, privacy, and bias
scoresT,P,B.
6.3 ExpandingBeyondText:Multi-ModalandStructured
Retrieval
Addresses:RQ1(retrievalprecisionacrossheterogeneouscor-
pora),RQ3(adapting to richer, multi-modal user contexts),
andRQ4(reasoning over evidence that spans text, images,tables, and code). Early demonstrations exist, but unified
cross-modalretrievalwithcoherentreasoningremainsanopen
challenge across all four RQs.
As LLMs are increasingly deployed in multi-modal envi-
ronments, extending RAG systems beyond plain text has be-
comeakeychallenge.Manyreal-worldtasksrequireretrieving
and reasoning over images, tables, code, videos, audio, and
other structured artefacts. Traditional RAG pipelines, which
assume unstructured text corpora, struggle to capture the
full spectrum of such information. Early explorations of this
direction include modular vision–language architectures such
as mPLUG-Owl [142] and MiniGPT-4 [141], video retrieval
systems [147, 35], and systematic surveys of vision-based
RAG [58] that characterize the technical challenges unique
to cross-modal grounding.
Future RAG architectures will need to learn unified or
well-aligned embedding spaces that support cross-modal re-
trieval, cross-attention between textual queries and non-
textual evidence, and dynamic decisions about which modal-
ity is most informative for a given query. Structured retrieval
over knowledge graphs, relational databases, or APIs raises
additionalissuesinindexing,segmentation,querytranslation,
and semantic matching.
Applications in healthcare (e.g., combining imaging re-
ports, structured lab values, and clinical notes), education
(e.g., grounding answers in lecture videos and slides), and
scientific discovery (e.g., retrieving tables, figures, and code
from research papers) can particularly benefit from multi-
modal RAG. As modalities converge, expanding RAG beyond
text will be critical for building more comprehensive knowl-
edge agents that operate over the full range of digital artifacts
encountered in practice. Formally, this requires generalizing
the encodersϕ qandϕdand the similarity score of Eq. (5) to
a shared multi-modal embedding space, and broadening the
retrieverR(x,C)of Algorithm 1 so that the corpusCspans
images, tables, and code rather than text alone.
6.4 Improved Evaluation Benchmarks
Addresses:All four RQs. Progress on RQ1–RQ4 cannot be
reliably measured without benchmarks that jointly assess
efficiency, robustness, personalization quality, and multi-hop
reasoning fidelity; current frameworks cover each dimension
incompletely and rarely in combination.
Despite progress in separately benchmarking retrieval and
generation, comprehensive evaluation of end-to-end RAG sys-
temsremainsunderdeveloped.Existingmetricsoftenmisskey
dimensions such as attribution quality, hallucination rates,
contextual relevance, robustness to noisy retrieval, and long-
termreasoningconsistency.AsRAGbecomesintegraltohigh-
stakes workflows, standardized and interpretable evaluation
tools will be essential. General-purpose frameworks such as
RAGAS[96],RAGBench[97],andBERGEN[120]havebegun
providingmodular,reproducibleinfrastructureforend-to-end
evaluation, while task-specific benchmarks like FRAMES [21]
and Face4RAG [199] introduce multi-dimensional scoring for
reasoning fidelity and hallucination detection.
More work is needed to assess RAG in multilingual,
multi-modal, multi-hop, and user-centric settings. Bench-
marks should account not only for answer accuracy, but also
for grounding fidelity, the quality of evidence selection, and

27
user-centric factors such as trust, readability, and stability
over time. Dataset diversity is another bottleneck, with most
evaluationsconcentratedonEnglishandgeneral-purposeQA.
Future research should prioritize task-specific benchmarks
in domains like law, medicine, education, and programming,
along with protocols that explicitly test safety and fairness.
Explainability-aware evaluation measuring how clearly mod-
els link outputs to retrieved inputs and how reliably they
expose their evidence chains will be crucial for building RAG
systems that are not only effective, but also auditable and
trustworthy in real-world use.
Whether a single uniform benchmark can span all four
axes remains an open question. We argue that it is only
partially feasible: efficiency, robustness, personalization, and
reasoning fidelity rest on fundamentally different ground
truths-latency and token budgets, adversarial robustness,
user satisfaction, and multi-hop correctness-that cannot be
collapsed into a single score. A more realistic target is a
shared evaluation harness with axis-specific tracks-common
corpora, retrieval interfaces, and reporting formats, but per-
axis metrics-rather than one universal leaderboard. In terms
of the formal framework, such a harness would need to in-
strument the scoring functions that the four objectives leave
abstract: the utilityU(q,d)and costC ϕ(d)of Eq. (7), the
risk termsT,P,Bof Eq. (8), the user-alignment termsH,F
of Eq. (9), and the step-wise coherence and consistency of
Eq. (10), so that methods become directly comparable along
common dimensions.
7 Actionable Guidance for Practitioners
Practitioners designing real-world RAG systems must bal-
ance retrieval architecture, efficiency, safety, and evaluation
rigor. These recommendations map directly onto the sur-
vey’s research questions: retrieval-architecture and efficiency
choices operationalizeRQ1, safety and privacy controls ad-
dressRQ2,user-centricalignmentmetricsspeaktoRQ3,and
multi-hop evaluation targetsRQ4. Sparse retrieval methods
such as BM25 offer interpretability, strong performance on
long or keyword-rich documents, and low operational cost,
while dense retrievers excel at semantic matching for short
or concept-heavy queries, making them well suited for multi-
hop reasoning or personalized interactions. In many settings,
hybrid sparse–dense configurations provide the best balance
between recall and precision, especially when dense retrieval
rescuessemanticmatchesthatsparsemethodsmissandsparse
retrieval filters obviously irrelevant content.
Efficient RAG pipelines benefit from modularity, early
filtering, and adaptive retrieval. Lightweight query encoding,
hybrid retrieval, reranking layers, context compression, and
dynamic (rather than fixed top-k) retrieval loops can all
help reduce latency and improve grounding quality. Practical
deployments should combine these techniques with caching
strategies and domain-aware chunking (e.g., element-based
chunking for financial or legal documents) to avoid redun-
dant work and improve retrieval granularity. Enterprise use
cases introduce additional constraints. Systems must support
auditability of retrieval corpora, fine-grained access control at
retrieval time, and integration with structured data sources
such as databases, APIs, and knowledge graphs. Continuous
monitoring for hallucinations, retrieval drift, knowledge stale-
ness,andpolicyviolationsisessentialwhenRAGisembeddedin business-critical workflows. A robust evaluation workflow
should measure retrieval precision and recall, grounding fi-
delity, hallucination rates, latency, robustness to noisy or
shifted domains, multi-hop reasoning quality, and the relative
contributions of retrieval and generation, complemented by
user-centric alignment metrics such as satisfaction, trust, and
perceived helpfulness. Safety and privacy guidelines parallel
Defensive RAG principles. Systems should filter or rerank
retrieved content to exclude toxic, biased, or privacy-sensitive
documents; enforce strict role-based visibility for enterprise
knowledge; and preferentially use synthetic or redacted cor-
pora in high-risk domains. Regular audits, bias assessments,
and adversarial stress tests help ensure that retrieval and
generation behave safely under realistic and worst-case con-
ditions. In combination, these practices provide a practical
recipe for deploying RAG systems that are not only powerful
andefficient,butalsosecure,compliant,andalignedwithuser
and organizational goals.
8 Conclusion
RAG has become a central paradigm for extending LLMs be-
yond static, parametric knowledge. By coupling large models
with external retrieval, RAG addresses persistent issues such
as outdated information, hallucinations, and weak factual
grounding. This survey has traced the evolution of RAG
systems across architectures, training strategies, and appli-
cation domains, and has organized recent work along four
complementary axes:compression and efficiency,defensive
and safety-aware RAG,interactive and user-centric systems,
andcomplex reasoning and multi-step pipelines. These axes
corresponddirectlytothefourresearchquestionsposedin§1-
efficiency (RQ1), defense (RQ2), interactivity (RQ3), and
reasoning (RQ4)-so the survey’s contribution is best read as
a map of how far the literature has answered each. Viewed
together, these axes highlight how the field has moved from
simpleretrieve-then-generatepipelinestowardmoreadaptive,
risk-aware, and cognitively structured retrieval–generation
loops.
Across applications, we observe RAG being deployed in
open-domain QA, conversational agents, code generation,
educational and corporate assistants, and scientific and legal
workflows. These settings share a common requirement: sys-
tems must combine fluent generation with reliable access to
verifiable, task-specific knowledge. RAG meets this require-
ment by making retrieval a first-class component of system
design, but the diversity of deployment scenarios also exposes
recurringpainpointsaroundevaluation,domaintransfer,and
operational robustness.
The preceding sections have outlined key challenges in-
cluding retrieval bottlenecks, residual hallucinations, domain
and user adaptation, scalability and latency constraints, and
limited end-to-end explainability, as well as near-term re-
search directions in reinforcement learning for RAG (§6.1),
neuro-symbolic and knowledge-graph integration (§6.2), mul-
timodal and structured retrieval (§6.3), and improved bench-
marks (§6.4). Looking ahead, the impact of RAG will depend
not only on architectural innovations but also on how seam-
lessly these systems integrate with real-world infrastructure,
satisfy privacy and fairness requirements, and remain main-
tainableasunderlyingcorporaanduserneedsevolve.Wehope

28
this survey provides a clear map of the design space and a
practicalfoundationforresearchersandpractitionersbuilding
the next generation of retrieval-augmented systems.
Beyondthenear-termdirectionsdetailedin§6,twofurther
trajectories are likely to shape the longer-term evolution of
RAG.
Agentic RAG.RAG pipelines are likely to evolve from
single-shot retrieval modules into agents that actively manage
their own information needs. Rather than issuing a single
query and generating an answer, future systems will itera-
tivelyreformulatequeries,planmulti-stepretrievalstrategies,
test competing hypotheses, and call external tools when re-
trieval alone is insufficient. Retrieval will become a decision
taken at many points along the reasoning process (“Do I need
more evidence here? What kind? From where?”), rather than
a fixed preprocessingstep. This shift toward agentic RAG will
blur the line between retrieval, planning, and reasoning, with
policies that explicitly trade off additional retrieval against
latency, cost, and expected accuracy.
LLM-as-Retriever and Unified Agent Architectures.
As LLMs themselves improve at recall, reasoning, and tool
orchestration, we are likely to see architectures where the
same model acts as both generator and high-level retriever.
Instead of relying solely on external vector stores, future
systems may query latent memory buffers, learned key–value
stores, or lightweight external tools under the control of a
singleagenticpolicy.Insuchsetups,classicalvectorsearchwill
remainimportant,butitwillbewrappedinsidebroaderagent
frameworks that decide when to consult external indices,
when to rely on internal representations, and how to coor-
dinate multiple tools (search, databases, simulators) within
one coherent control loop.
Scaling Laws for Retrieval.Just as scaling laws have
helped characterize how model size and data volume affect
LLM performance, we expect analogous principles to emerge
for retrieval. Early evidence suggests systematic relationships
between corpus size, chunking strategy, retriever capacity,
context window, and downstream accuracy or hallucination
rates. Formalizing these “retrieval scaling laws” would give
practitioners concrete guidance on questions such as: How
should index size grow with model size? When does more
context stop helping? How aggressively can we compress
or filter documents before accuracy degrades? Such insights
would make RAG design less heuristic and more principled,
helping teams provision retrieval resources that match their
models and tasks.
Overall, we anticipate that RAG will continue to evolve
from a simple augmentation technique into a core organizing
principle for knowledge-intensive AI systems. Systems that
can proactively seek information, reason over structured and
multi-modal evidence, and expose clear links between re-
trieved sources and generated outputs will be best positioned
to meet real-world demands for accuracy, transparency, and
adaptability.
Author Contributions
Meghana Sunil and Shravya V contributed equally to this
work and jointly led the development of the review, including
conceptualization, literature survey, and analysis. Shravan
Venkatraman supported the work by assisting with writing,manuscript refinement, and providing technical feedback. Joe
Dhanith P R contributed through writing support, supervi-
sion, and overall mentorship that guided the direction and
quality of the work.
Conflict of Interests:
The authors declare that there are no conflicts of interest
associated with this work.
Funding Information:
This research did not receive any financial support or fund-
ing.
References
[1] OpenAI, J. Achiam, and S. Adler, “Gpt-4 technical report,”
2024. 1, 20, 21
[2] G. Team, R. Anil, S. Borgeaud, and J.-B. Alayrac, “Gemini: A
family of highly capable multimodal models,” 2025. 1
[3] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Pas-
sos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, E. Chu, J. H.
Clark, L. E. Shafey, Y. Huang, K. Meier-Hellstern, G. Mishra,
E. Moreira, M. Omernick, K. Robinson, S. Ruder, Y. Tay,
K. Xiao, Y. Xu, Y. Zhang, G. H. Abrego, J. Ahn, J. Austin,
P. Barham, J. Botha, J. Bradbury, S. Brahma, K. Brooks,
M. Catasta, Y. Cheng, C. Cherry, C. A. Choquette-Choo,
A. Chowdhery, C. Crepy, S. Dave, M. Dehghani, S. Dev, J. De-
vlin,M.Díaz,N.Du,E.Dyer,V.Feinberg,F.Feng,V.Fienber,
M. Freitag, X. Garcia, S. Gehrmann, L. Gonzalez, G. Gur-Ari,
S. Hand, H. Hashemi, L. Hou, J. Howland, A. Hu, J. Hui,
J. Hurwitz, M. Isard, A. Ittycheriah, M. Jagielski, W. Jia,
K. Kenealy, M. Krikun, S. Kudugunta, C. Lan, K. Lee, B. Lee,
E. Li, M. Li, W. Li, Y. Li, J. Li, H. Lim, H. Lin, Z. Liu,
F. Liu, M. Maggioni, A. Mahendru, J. Maynez, V. Misra,
M. Moussalem, Z. Nado, J. Nham, E. Ni, A. Nystrom, A. Par-
rish, M. Pellat, M. Polacek, A. Polozov, R. Pope, S. Qiao,
E. Reif, B. Richter, P. Riley, A. C. Ros, A. Roy, B. Saeta,
R. Samuel, R. Shelby, A. Slone, D. Smilkov, D. R. So, D. Sohn,
S.Tokumine,D.Valter,V.Vasudevan,K.Vodrahalli,X.Wang,
P. Wang, Z. Wang, T. Wang, J. Wieting, Y. Wu, K. Xu, Y. Xu,
L. Xue, P. Yin, J. Yu, Q. Zhang, S. Zheng, C. Zheng, W. Zhou,
D.Zhou,S.Petrov,andY.Wu,“Palm2technicalreport,”2023.
1
[4] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi,
Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale,
D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull,
D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao,
V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou,
H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann,
A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee,
D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov,
P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein,
R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith,
R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams,
J.X.Kuan,P.Xu,Z.Yan,I.Zarov,Y.Zhang,A.Fan,M.Kam-
badur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and
T. Scialom, “Llama 2: Open foundation and fine-tuned chat
models,” 2023. 1
[5] H.Touvron,T.Lavril,G.Izacard,X.Martinet,M.-A.Lachaux,
T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Ro-
driguez,A.Joulin,E.Grave,andG.Lample,“Llama:Openand
efficient foundation language models,” 2023. 1
[6] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra,
A.Roberts,P.Barham,H.W.Chung,C.Sutton,S.Gehrmann,
P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao,
P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du,
B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard,
G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat,
S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson,
L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph,
A.Spiridonov,R.Sepassi,D.Dohan,S.Agrawal,M.Omernick,
A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira,
R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta,

29
M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern,
D. Eck, J. Dean, S. Petrov, and N. Fiedel, “Palm: Scaling
language modeling with pathways,” 2022. 1
[7] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Ka-
plan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sas-
try, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger,
T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Win-
ter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford,
I. Sutskever, and D. Amodei, “Language models are few-shot
learners,” inAdvances in Neural Information Processing Sys-
tems(H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and
H. Lin, eds.), vol. 33, pp. 1877–1901, Curran Associates, Inc.,
2020. 1
[8] O. Thawakar, D. Dissanayake, K. P. More, R. Thawkar,
A. Heakl, N. Ahsan, Y. Li, I. Z. M. Zumri, J. Lahoud, R. M.
Anwer,et al., “Llamav-o1: Rethinking step-by-step visual rea-
soning in llms,” inFindings of the Association for Computa-
tional Linguistics: ACL 2025, pp. 24290–24315, 2025. 1
[9] S.Li,L.Stenzel,C.Eickhoff,andS.A.Bahrainian,“Enhancing
retrieval-augmented generation: a study of best practices,” in
Proceedings of the 31st International Conference on Computa-
tional Linguistics, pp. 6705–6717, 2025. 1
[10] S. Wu, X. Ma, D. Luo, L. Li, X. Shi, X. Chang, X. Lin, R. Luo,
C. Pei, C. Du,et al., “Automated literature research and
review-generation method based on large language models,”
National Science Review, vol. 12, no. 6, p. nwaf169, 2025. 1
[11] M. D’Arcy, T. Hope, L. Birnbaum, and D. Downey, “Marg:
Multi-agent review generation for scientific papers,” 2024. 1
[12] Z. Gao, K. Brantley, and T. Joachims, “Reviewer2: Optimizing
review generation through prompt generation,”arXiv preprint
arXiv:2402.10886, 2024. 1
[13] W. Jiang, J. Chen, X. Ding, J. Wu, J. He, and G. Wang,
“Review summary generation in online systems: Frameworks
forsupervisedandunsupervisedscenarios,”ACMTransactions
on the Web (TWEB), vol. 15, no. 3, pp. 1–33, 2021. 1
[14] Z. Yi, J. Ouyang, Z. Xu, Y. Liu, T. Liao, H. Luo, and Y. Shen,
“A survey on recent advances in llm-based multi-turn dialogue
systems,”ACM Computing Surveys, vol. 58, no. 6, pp. 1–38,
2025. 1
[15] Y. Fan and X. Luo, “A survey of dialogue system evaluation,”
in2020 IEEE 32nd International Conference on Tools with
Artificial Intelligence (ICTAI), pp. 1202–1209, IEEE, 2020. 1
[16] S. E. Finch and J. D. Choi, “Towards unified dialogue system
evaluation: A comprehensive analysis of current evaluation
protocols,” inProceedings of the 21th annual meeting of the
special interest group on discourse and dialogue, pp. 236–245,
2020. 1
[17] J.Ni,T.Young,V.Pandelea,F.Xue,andE.Cambria,“Recent
advancesindeeplearningbaseddialoguesystems:Asystematic
survey,”arXiv preprint arXiv:2105.04387, 2021. 1
[18] M.Keymanesh,A.Benton,andM.Dredze,“Whatmakesdata-
to-text generation hard for pretrained language models?,” in
Proceedings of the 2nd workshop on natural language genera-
tion, evaluation, and metrics (GEM), pp. 539–554, 2022. 1
[19] Y. Lin, T. Ruan, J. Liu, and H. Wang, “A survey on neural
data-to-text generation,”IEEE Transactions on Knowledge
and Data Engineering, vol. 36, no. 4, pp. 1431–1449, 2023. 1
[20] M. Sharma, A. K. Gogineni, and N. Ramakrishnan, “Neural
methods for data-to-text generation,”ACM Transactions on
Intelligent Systems and Technology, vol. 15, no. 5, pp. 1–46,
2024. 1
[21] S. Krishna, K. Krishna, A. Mohananey, S. Schwarcz, A. Stam-
bler, S. Upadhyay, and M. Faruqui, “Fact, fetch, and reason:
A unified evaluation of retrieval-augmented generation,” in
Proceedingsofthe2025ConferenceoftheNationsoftheAmer-
icas Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 1: Long Papers),
pp. 4745–4759, 2025. 1, 18, 19, 25, 26
[22] Y. Ke, L. Jin, K. Elangovan, H. R. Abdullah, N. Liu, A. T. H.
Sia, C. R. Soh, J. Y. M. Tung, J. C. L. Ong, and D. S. W. Ting,
“Development and testing of retrieval augmented generation
in large language models–a case study report,”arXiv preprint
arXiv:2402.01733, 2024. 1
[23] G. Chen, W. Yu, X. Lu, X. Zhang, E. Meng, and L. Sha,
“Unlocking multi-view insights in knowledge-dense retrieval-
augmented generation,”IEEE Transactions on Audio, Speechand Language Processing, 2025. 1
[24] X. Wang, P. Sen, R. Li, and E. Yilmaz, “Adaptive retrieval-
augmented generation for conversational systems,” inFindings
oftheAssociationforComputationalLinguistics:NAACL2025
(L.Chiruzzo,A.Ritter,andL.Wang,eds.),(Albuquerque,New
Mexico), pp. 491–503, Association for Computational Linguis-
tics, Apr. 2025. 1
[25] Y.Wang,P.Li,M.Sun,andY.Liu,“Self-knowledgeguidedre-
trievalaugmentationforlargelanguagemodels,”inFindingsof
the Association for Computational Linguistics: EMNLP 2023,
pp. 10303–10315, 2023. 1
[26] A. Singh, A. Ehtesham, S. Kumar, T. T. Khoei, and A. V.
Vasilakos, “Agentic retrieval-augmented generation: A survey
on agentic rag,”arXiv preprint arXiv:2501.09136, 2025. 1, 5, 6
[27] H. Han, Y. Wang, H. Shomer, K. Guo, J. Ding, Y. Lei,
M. Halappanavar, R. A. Rossi, S. Mukherjee, X. Tang,et al.,
“Retrieval-augmented generation with graphs (graphrag),”
arXiv preprint arXiv:2501.00309, 2024. 1, 25, 26
[28] X. Li, P. Jia, D. Xu, Y. Wen, Y. Zhang, W. Zhang, W. Wang,
etal.,“Asurveyofpersonalization:FromRAGtoagent,”ACM
Transactions on Information Systems, vol. 44, no. 4, pp. 1–39,
2026. 1
[29] L. Chen, X. Wei, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen,
H. Duan, B. Lin, Z. Tang, L. Yuan, Y. Qiao, D. Lin, F. Zhao,
andJ.Wang,“Sharegpt4video:Improvingvideounderstanding
and generation with better captions,” inAdvances in Neural
Information Processing Systems(A. Globerson, L. Mackey,
D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang,
eds.), vol. 37, pp. 19472–19495, Curran Associates, Inc., 2024.
1
[30] Y. Feng, H. A. Rahmani, A. Lipani, and E. Yilmaz, “Towards
asking clarification questions for information seeking on task-
oriented dialogues,”arXiv preprint arXiv:2305.13690, 2023. 1
[31] S. Venkatraman, J. D. PR, and M. S. Kavitha, “Hierarchical
graph-guided contextual representation learning for neurode-
generative pattern recognition in mri,”Computers in Biology
and Medicine, vol. 199, p. 111276, 2025. 1
[32] S. Venkatraman, M. S. Kavitha, V. Manikandarajan, J. Wu,
et al., “Can we go beyond visual features? neural tissue rela-
tion modeling for relational graph analysis in non-melanoma
skin histology,” inProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 6427–6437,
2026. 1
[33] S. Venkatraman, R. R. Madavan,et al., “Ugpl: Uncertainty-
guided progressive learning for evidence-based classification
in computed tomography,” inProceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 958–968,
2025. 1
[34] S.Kanthimathi,P.Nanda,S.Venkatraman,V.Renganayagan,
and S. Eswaran, “Transforming education through ai-powered
personalized assessment models,” inAdopting Artificial Intel-
ligence Tools in Higher Education, pp. 136–154, CRC Press,
2025. 1
[35] S. Jeong, K. Kim, J. Baek, and S. J. Hwang, “Videorag:
Retrieval-augmented generation over video corpus,” inFind-
ings of the Association for Computational Linguistics: ACL
2025, pp. 21278–21298, 2025. 1, 26
[36] O. Thawakar, S. Venkatraman, R. Thawkar, A. Shaker,
H. Cholakkal, R. M. Anwer, S. Khan, and F. Khan, “Evolmm:
Self-evolving large multimodal models with continuous re-
wards,” 2026. 1
[37] M. Sunil, M. Venmathimaran, and M. S. Kavitha, “irea-
soner: Trajectory-aware intrinsic reasoning supervision for self-
evolving large multimodal models,” 2026. 1
[38] S. Venkatraman, R. Thawkar, O. Thawakar, R. M. Anwer,
H.Cholakkal,S.Khan,andF.Khan,“Payingmoreattentionto
visual tokens in self-evolving large multimodal models,” 2026.
1
[39] H. Zhou, K.-H. Lee, Z. Zhan, Y. Chen, Z. Li, Z. Wang, H. Had-
dadi, and E. Yilmaz, “Trustrag: Enhancing robustness and
trustworthiness in retrieval-augmented generation,” 2025. 1,
25
[40] X. Liang, S. Niu, Z. Li, S. Zhang, H. Wang, F. Xiong, Z. Fan,
B. Tang, J. Zhao, J. Yang,et al., “Saferag: benchmarking
security in retrieval-augmented generation of large language
model,” inProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long

30
Papers), pp. 4609–4631, 2025. 1
[41] T.Fan,J.Wang,X.Ren,andC.Huang,“Minirag:Towardsex-
tremelysimpleretrieval-augmentedgeneration,”arXivpreprint
arXiv:2501.06713, 2025. 1
[42] L. Wang, H. Chen, N. Yang, X. Huang, Z. Dou, and F. Wei,
“Chain-of-retrieval augmented generation,” inAdvances in
Neural Information Processing Systems, vol. 38, pp. 59888–
59915, 2026. 1, 8
[43] Z. Qi, R. Xu, Z. Guo, C. Wang, H. Zhang, and W. Xu,
“Long2rag: Evaluating long-context & long-form retrieval-
augmented generation with key point recall,” inFindings of
the Association for Computational Linguistics: EMNLP 2024,
pp. 4852–4872, 2024. 1
[44] H. Snyder, “Literature review as a research methodology: An
overviewandguidelines,”Journalofbusinessresearch,vol.104,
pp. 333–339, 2019. 2
[45] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rock-
täschel,etal., “Retrieval-augmented generation for knowledge-
intensivenlptasks,”Advancesinneuralinformationprocessing
systems, vol. 33, pp. 9459–9474, 2020. 2, 4, 19
[46] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and
C.Manning,“Raptor:Recursiveabstractiveprocessingfortree-
organized retrieval,” inInternational Conference on Learning
Representations, vol. 2024, pp. 32628–32649, 2024. 2, 6, 7, 8
[47] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-
rag: Learning to retrieve, generate, and critique through self-
reflection,” inInternational conference on learning representa-
tions, vol. 2024, pp. 9112–9141, 2024. 2, 7, 10, 11
[48] Y. Gao, T. Sheng, Y. Xiang, Y. Xiong, H. Wang, and
J. Zhang, “Chat-rec: Towards interactive and explain-
able llms-augmented recommender system,”arXiv preprint
arXiv:2303.14524, 2023. 2, 7, 12, 13
[49] Z. Shi, S. Zhang, W. Sun, S. Gao, P. Ren, Z. Chen, and Z. Ren,
“Generate-then-ground in retrieval-augmented generation for
multi-hop question answering,” inProceedings of the 62nd An-
nual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 7339–7353, 2024. 2, 7, 15, 16, 17
[50] B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, “Hip-
porag: Neurobiologically inspired long-term memory for large
language models,”Advances in neural information processing
systems, vol. 37, pp. 59532–59569, 2024. 3
[51] Z. Chen, C. Xu, D. Wang, Z. Huang, Y. Dou, X. Jiang, and
J. Guo, “Rulerag: Rule-guided retrieval-augmented generation
with language models for question answering,”arXiv preprint
arXiv:2410.22353, 2024. 3
[52] Y. Xi, W. Liu, J. Lin, B. Chen, R. Tang, W. Zhang, and Y. Yu,
“Memocrs:Memory-enhancedsequentialconversationalrecom-
mender systems with large language models,” inProceedingsof
the 33rd ACM International Conference on Information and
Knowledge Management, pp. 2585–2595, 2024. 3
[53] Z.Chu,H.Fan,J.Chen,Q.Wang,M.Yang,J.Liang,Z.Wang,
H. Li, G. Tang, M. Liu,et al., “Self-critique guided itera-
tive reasoning for multi-hop question answering,” inFindings
of the Association for Computational Linguistics: ACL 2025,
pp. 2415–2438, 2025. 3
[54] S. Zhao, Y. Yang, Z. Wang, Z. He, L. K. Qiu, and L. Qiu,
“Retrieval augmented generation (rag) and beyond: A compre-
hensivesurveyonhowtomakeyourllmsuseexternaldatamore
wisely,”arXiv preprint arXiv:2409.14924, 2024. 3
[55] L. Brehme, T. Ströhle, and R. Breu, “Can LLMs be trusted for
evaluatingRAGsystems?asurveyofmethodsanddatasets,”in
2025 IEEE Swiss Conference on Data Science (SDS), pp. 16–
23, IEEE, 2025. 5
[56] M. Hindi, L. Mohammed, O. Maaz, and A. Alwarafy, “Enhanc-
ing the precision and interpretability of retrieval-augmented
generation(RAG)inlegaltechnology:Asurvey,”IEEEAccess,
2025. 5
[57] B. Ni, Z. Liu, L. Wang, Y. Lei, Y. Zhao, X. Cheng, Q. Zeng,
L. Dong, Y. Xia, K. Kenthapadi, R. Rossi, F. Dernoncourt,
M.M.Tanjim,N.Ahmed,X.Liu,W.Fan,E.Blasch,Y.Wang,
M. Jiang, and T. Derr, “Towards trustworthy retrieval aug-
mented generation for large language models: A survey,” 2025.
5
[58] X. Zheng, Z. Weng, Y. Lyu, L. Jiang, H. Xue, B. Ren,
D. Paudel, N. Sebe, L. V. Gool, and X. Hu, “Retrieval aug-
mented generation and understanding in vision: A survey andnew outlook,” 2025. 5, 26
[59] A. J. Oche, A. G. Folashade, T. Ghosal, and A. Biswas, “A
systematic review of key retrieval-augmented generation (rag)
systems: Progress, gaps, and future directions,” 2025. 5
[60] X. Cheng, X. Wang, X. Zhang, T. Ge, S.-Q. Chen, F. Wei,
H. Zhang, and D. Zhao, “xrag: Extreme context compression
for retrieval-augmented generation with one token,”Advances
inNeuralInformationProcessingSystems,vol.37,pp.109487–
109516, 2024. 6, 7, 8
[61] C.-M. Chan, C. Xu, R. Yuan, H. Luo, W. Xue, Y. Guo, and
J. Fu, “Rq-rag: Learning to refine queries for retrieval aug-
mentedgeneration,”arXivpreprintarXiv:2404.00610,2024. 6,
7, 8
[62] H. Zamani and M. Bendersky, “Stochastic rag: End-to-end
retrieval-augmented generation through expected utility maxi-
mization,” 2024. 6, 7, 8
[63] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang,
W. Zhang, J. Jiang, and B. Cui, “Retrieval-augmented gen-
eration for ai-generated content: A survey,”Data Science and
Engineering, pp. 1–29, 2026. 7, 10, 11
[64] W. Zou, R. Geng, B. Wang, and J. Jia, “{PoisonedRAG}:
Knowledge corruption attacks to{Retrieval-Augmented}gen-
eration of large language models,” in34th USENIX Security
Symposium(USENIXSecurity25),pp.3827–3844,2025. 7,10,
11
[65] T. E. Kim and F. Diaz, “Towards fair rag: On the impact of
fair ranking in retrieval-augmented generation,” inProceedings
of the 2025 International ACM SIGIR Conference on Innova-
tive Concepts and Theories in Information Retrieval (ICTIR),
pp. 33–43, 2025. 7, 10, 12
[66] S. Zerhoudi and M. Granitzer, “Personarag: Enhancing
retrieval-augmented generation systems with user-centric
agents,” 2026. 7, 12, 13
[67] Y. Shi, X. Zi, Z. Shi, H. Zhang, Q. Wu, and M. Xu, “Er-
agent: Enhancing retrieval-augmented language models with
improved accuracy, efficiency, and personalization,”arXiv
preprint arXiv:2405.06683, 2024. 7, 12, 13, 14
[68] M. Gaur, K. Gunaratna, V. Srinivasan, and H. Jin, “Iseeq:
Information seeking question generation using dynamic meta-
information retrieval and knowledge graphs,” inProceedings
of the AAAI conference on artificial intelligence, vol. 36,
pp. 10672–10680, 2022. 7, 12, 13
[69] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabhar-
wal, “Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions,” inProceedings of
the 61st annual meeting of the association for computational
linguistics (volume 1: long papers), pp. 10014–10037, 2023. 7,
15, 16
[70] P.Verma,S.P.Midigeshi,G.Sinha,A.Solin,N.Natarajan,and
A. Sharma, “Plan-rag: Planning-guided retrieval augmented
generation,” 2024. 7, 15, 17
[71] X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun,
X. Bresson, and B. Hooi, “G-retriever: Retrieval-augmented
generation for textual graph understanding and question an-
swering,”Advances inNeuralInformation ProcessingSystems,
vol. 37, pp. 132876–132907, 2024. 7, 15, 16, 26
[72] Y. Ma, Y. Cao, Y. Hong, and A. Sun, “Large language model is
not a good few-shot information extractor, but a good reranker
for hard samples!,” inFindings of the association for computa-
tional linguistics: EMNLP 2023, pp. 10572–10601, 2023. 7, 10,
11
[73] T. Merth, Q. Fu, M. Rastegari, and M. Najibi, “Superposition
prompting: Improving and accelerating retrieval-augmented
generation, 2024,”URL https://arxiv. org/abs/2404.06910. 7,
8
[74] S. Wang, Y. Xu, Y. Fang, Y. Liu, S. Sun, R. Xu, C. Zhu, and
M. Zeng, “Training data is more valuable than you think: A
simple and effective method by retrieving from training data,”
inProceedingsofthe60thAnnualMeetingoftheAssociationfor
ComputationalLinguistics(Volume1:LongPapers),pp.3170–
3179, 2022. 7
[75] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford,
K.Millican,G.B.VanDenDriessche,J.-B.Lespiau,B.Damoc,
A.Clark,etal.,“Improvinglanguagemodelsbyretrievingfrom
trillions of tokens,” inInternational conference on machine
learning, pp. 2206–2240, PMLR, 2022. 7
[76] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. H. Chi,

31
N. Schärli, and D. Zhou, “Large language models can be easily
distracted by irrelevant context,” inInternational Conference
on Machine Learning, pp. 31210–31227, PMLR, 2023. 8
[77] T. Shen, G. Long, X. Geng, C. Tao, Y. Lei, T. Zhou, M. Blu-
menstein, and D. Jiang, “Retrieval-augmented retrieval: Large
language models are strong zero-shot retriever,” inFindings
of the Association for Computational Linguistics: ACL 2024,
pp. 15933–15946, 2024. 8
[78] Z. Guo, S. Cheng, Y. Wang, P. Li, and Y. Liu, “Prompt-guided
retrieval augmentation for non-knowledge-intensive tasks,” in
Findings of the Association for Computational Linguistics:
ACL 2023, pp. 10896–10912, 2023. 8
[79] M. Santacroce, Z. Wen, Y. Shen, and Y. Li, “What matters in
the structured pruning of generative language models?,”arXiv
preprint arXiv:2302.03773, 2023. 8
[80] J. Liu, T. Zhou, Y. Chen, J. Zhao, and K. Liu, “Enhancing
large language models with pseudo-and multisource-knowledge
graphs for open-ended question answering,” in2025 IEEE
41stInternationalConferenceonDataEngineeringWorkshops
(ICDEW), pp. 97–106, IEEE, 2025. 8
[81] W. Yu, “Retrieval-augmented generation across heterogeneous
knowledge,” inProceedings of the 2022 conference of the North
American chapter of the association for computational linguis-
tics: human language technologies: student research workshop,
pp. 52–58, 2022. 9
[82] Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov,
K. Guu, K. B. Hall, and M.-W. Chang, “Promptagator:
Few-shot dense retrieval from 8 examples,”arXiv preprint
arXiv:2209.11755, 2022. 9
[83] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai,
J. Sun, M. Wang, and H. Wang, “Retrieval-augmented gen-
eration for large language models: A survey,”arXiv preprint
arXiv:2312.10997, 2023. 9, 23, 25
[84] D. Cheng, S. Huang, J. Bi, Y. Zhan, J. Liu, Y. Wang, H. Sun,
F. Wei, W. Deng, and Q. Zhang, “Uprise: Universal prompt
retrieval for improving zero-shot evaluation,” inProceedings of
the 2023 Conference on Empirical Methods in Natural Lan-
guage Processing, pp. 12318–12337, 2023. 9
[85] J.Lin,X.Dai,Y.Xi,W.Liu,B.Chen,H.Zhang,Y.Liu,C.Wu,
X. Li, C. Zhu,et al., “How can recommender systems benefit
from large language models: A survey,”ACM Transactions on
Information Systems, vol. 43, no. 2, pp. 1–47, 2025. 9
[86] S. Setty, H. Thakkar, A. Lee, E. Chung, and N. Vidra, “Im-
proving retrieval for rag based question answering models on
financial documents,”arXiv preprint arXiv:2404.07221, 2024.
9
[87] N. Pipitone and G. H. Alami, “Legalbench-rag: A benchmark
for retrieval-augmented generation in the legal domain,”arXiv
preprint arXiv:2408.10343, 2024. 9
[88] A. Mahboub, M. E. Za’ter, B. Al-Rfooh, Y. Estaitia, A. Jaljuli,
and A. Hakouz, “Evaluation of semantic search and its role
in retrieved-augmented-generation (rag) for arabic language,”
arXiv preprint arXiv:2403.18350, 2024. 9
[89] F. Hijazi, S. AlHarbi, A. AlHussein, H. Shairah, R. Alzahrani,
H. AlShamlan, G. Turkiyyah, and O. Knio, “Arablegaleval: A
multitask benchmark for assessing arabic legal knowledge in
large language models,” inProceedings of the Second Arabic
Natural Language Processing Conference, pp. 225–249, 2024. 9
[90] J.Shin,N.S.Harzevili,R.Aleithan,H.Hemmati,andS.Wang,
“Retrieval-augmented test generation: How far are we?,”arXiv
preprint arXiv:2409.12682, 2024. 9
[91] S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu,
L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost,et al., “Recom-
mender systems with generative retrieval,”Advances in Neural
Information Processing Systems, vol. 36, pp. 10299–10315,
2023. 9
[92] Y. Lu, L. Chen, Y. Zhang, M. Shen, H. Wang, X. Wang, C. van
Rechem, T. Fu, and W. Wei, “Machine learning for synthetic
data generation: a review,”arXiv preprint arXiv:2302.04062,
2023. 9
[93] J. Jin, Y. Zhu, Z. Dou, G. Dong, X. Yang, C. Zhang, T. Zhao,
Z. Yang, and J.-R. Wen, “Flashrag: A modular toolkit for effi-
cient retrieval-augmented generation research,” inCompanion
ProceedingsoftheACMonWebConference2025,pp.737–740,
2025. 9
[94] D.Fleischer,M.Berchansky,M.Wasserblat,andP.Izsak,“Rag
foundry: A framework for enhancing llms for retrieval aug-mented generation,”arXiv preprint arXiv:2408.02545, 2024. 9
[95] X.Zhang,Y.-Z.Song,Y.Wang,S.Tang,X.Li,Z.Zeng,Z.Wu,
W. Ye, W. Xu, Y. Zhang,et al., “Raglab: A modular and
research-oriented unified framework for retrieval-augmented
generation,” inProceedings of the 2024 Conference on Empir-
ical Methods in Natural Language Processing: System Demon-
strations, pp. 408–418, 2024. 9
[96] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas:
Automated evaluation of retrieval augmented generation,” in
Proceedings of the 18th conference of the european chapter of
the association for computational linguistics: system demon-
strations, pp. 150–158, 2024. 9, 26
[97] R. Friel, M. Belyi, and A. Sanyal, “Ragbench: Explainable
benchmarkforretrieval-augmentedgenerationsystems,”arXiv
preprint arXiv:2407.11005, 2024. 9, 26
[98] M.Nandakishor,“Deeprag:buildingacustomhindiembedding
model for retrieval augmented generation from scratch,”arXiv
preprint arXiv:2503.08213, 2025. 9
[99] B. Wan, F. Zhang, Z. Qi, J. Ding, J. Li, B. Fan, Y. Zhang, and
J. Zhang, “Cognitive-aligned document selection for retrieval-
augmented generation,”arXiv preprint arXiv:2502.11770,
2025. 11
[100] Y. Ji, H. Zhang, and Y. Wang, “Bias evaluation and mitigation
in retrieval-augmented medical question-answering systems,”
2025. 11
[101] W. Seo, Z. Yuan, and Y. Bu, “Valuesrag: Enhancing cultural
alignment through retrieval-augmented contextual learning,”
inProceedings of the AAAI/ACM Conference on AI, Ethics,
and Society, vol. 8, pp. 2307–2318, 2025. 11
[102] Y. Zhou, W. Zhang, J. Shao, Y. Liu, X. Li, J. Jin, H. Qian,
Z. Liu, C. Li, J. C. Zhang,et al., “Trustworthiness in retrieval-
augmented generation systems: A survey,”arXiv preprint
arXiv:2409.10102, 2024. 11, 25
[103] O. Ovadia, M. Brief, M. Mishaeli, and O. Elisha, “Fine-tuning
or retrieval? comparing knowledge injection in llms,” inPro-
ceedingsofthe2024conferenceonempiricalmethodsinnatural
language processing, pp. 237–250, 2024. 11
[104] S. Tonmoy, S. Zaman, V. Jain, A. Rani, V. Rawte, A. Chadha,
and A. Das, “A comprehensive survey of hallucination miti-
gation techniques in large language models,”arXiv preprint
arXiv:2401.01313, 2024. 11
[105] P. Zhou, Y. Feng, and Z. Yang, “Privacy-aware rag:
Secure and isolated knowledge retrieval,”arXiv preprint
arXiv:2503.15548, 2025. 11
[106] S. Zeng, J. Zhang, P. He, Y. Xing, Y. Liu, H. Xu, J. Ren,
S. Wang, D. Yin, Y. Chang, and J. Tang, “The good and the
bad:Exploringprivacyissuesinretrieval-augmentedgeneration
(RAG),” inFindings of the Association for Computational
Linguistics: ACL 2024(L.-W. Ku, A. Martins, and V. Sriku-
mar, eds.), (Bangkok, Thailand), pp. 4505–4524, Association
for Computational Linguistics, Aug. 2024. 11
[107] J.Xue,M.Zheng,Y.Hu,F.Liu,X.Chen,andQ.Lou,“Badrag:
Identifying vulnerabilities in retrieval augmented generation
of large language models,”arXiv preprint arXiv:2406.00083,
2024. 11
[108] S. Zeng, J. Zhang, P. He, Y. Xing, Y. Liu, H. Xu, J. Ren,
S. Wang, D. Yin, Y. Chang, and J. Tang, “The good and the
bad:Exploringprivacyissuesinretrieval-augmentedgeneration
(RAG),” inFindings of the Association for Computational
Linguistics: ACL 2024(L.-W. Ku, A. Martins, and V. Sriku-
mar, eds.), (Bangkok, Thailand), pp. 4505–4524, Association
for Computational Linguistics, Aug. 2024. 11
[109] I. Ziegler, A. Köksal, D. Elliott, and H. Schütze, “Craft your
dataset:Task-specificsyntheticdatasetgenerationthroughcor-
pus retrieval and augmentation,”Transactions of the Associ-
ation for Computational Linguistics, vol. 13, pp. 1693–1721,
2025. 11
[110] S. H. Jayasundara, N. A. G. Arachchilage, and G. Russello,
“Ragent: Retrieval-based access control policy generation,”
arXiv preprint arXiv:2409.07489, 2024. 11
[111] Z. Chen, J. Liu, H. Liu, Q. Cheng, F. Zhang, W. Lu, and
X. Liu, “Black-box opinion manipulation attacks to retrieval-
augmented generation of large language models,” 2024. 11
[112] J. Wu, S. Zhang, F. Che, M. Feng, P. Shao, and J. Tao,
“Pandora’s box or aladdin’s lamp: A comprehensive analysis
revealing the role of rag noise in large language models,” in
Proceedings of the 63rd Annual Meeting of the Association for

32
ComputationalLinguistics(Volume1:LongPapers),pp.5019–
5039, 2025. 11
[113] F. Ye, S. Li, Y. Zhang, and L. Chen, “R2ag: Incorporating
retrieval information into retrieval augmented generation,” in
Findings of the Association for Computational Linguistics:
EMNLP 2024, pp. 11584–11596, 2024. 11
[114] Z. Zhu, Y. Yang, and Z. Sun, “Halueval-wild: Evaluating hal-
lucinations of language models in the wild,”arXiv preprint
arXiv:2403.04307, 2024. 12
[115] B. He, N. Chen, X. He, L. Yan, Z. Wei, J. Luo, and Z.-H. Ling,
“Retrieving, rethinking and revising: The chain-of-verification
can improve retrieval augmented generation,” inFindings of
the Association for Computational Linguistics: EMNLP 2024,
pp. 10371–10393, 2024. 12
[116] A.C.Stickland,S.Sengupta,J.Krone,S.Mansour,andH.He,
“Robustification of multilingual language models to real-world
noise in crosslingual zero-shot settings with robust contrastive
pretraining,”inProceedingsofthe17thConferenceoftheEuro-
peanChapteroftheAssociationforComputationalLinguistics,
pp. 1375–1391, 2023. 12
[117] R.Ren,Y.Wang,Y.Qu,W.X.Zhao,J.Liu,H.Wu,J.-R.Wen,
and H. Wang, “Investigating the factual knowledge boundary
of large language models with retrieval augmentation,” inPro-
ceedingsofthe31stInternationalConferenceonComputational
Linguistics, pp. 3697–3715, 2025. 12
[118] L. Zeng, R. Gupta, D. Motwani, D. Yang, and Y. Zhang,
“Worse than zero-shot? a fact-checking dataset for evaluating
the robustness of rag against misleading retrievals,” 2025. 12
[119] Y. Hui, Y. Lu, and H. Zhang, “Uda: A benchmark suite for re-
trievalaugmentedgenerationinreal-worlddocumentanalysis,”
Advances in Neural Information Processing Systems, vol. 37,
pp. 67200–67217, 2024. 12
[120] D. Rau, H. Déjean, N. Chirkova, T. Formal, S. Wang, S. Clin-
chant, and V. Nikoulina, “Bergen: A benchmarking library for
retrieval-augmentedgeneration,”inFindingsoftheAssociation
for Computational Linguistics: EMNLP 2024, pp. 7640–7663,
2024. 12, 26
[121] Y. Xu, T. Cai, J. Jiang, and X. Song, “Face4rag: Factual
consistency evaluation for retrieval augmented generation in
chinese,” inProceedings of the30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, pp. 6083–6094,
2024. 12
[122] T. E. Kim and F. Diaz, “Towards fair rag: On the impact of
fair ranking in retrieval-augmented generation,” inProceedings
of the 2025 International ACM SIGIR Conference on Innova-
tive Concepts and Theories in Information Retrieval (ICTIR),
ICTIR ’25, (New York, NY, USA), p. 33–43, Association for
Computing Machinery, 2025. 12
[123] F.PeritiandS.Saha,“Thesynergisticintegrationofaccesscon-
trol management and large language model agents: A survey,”
TechRxiv, vol. 2026, no. 0210, 2026. 12
[124] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park,
“Adaptive-rag: Learning to adapt retrieval-augmented large
language models through question complexity,” inProceedings
of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pp. 7036–7050, 2024.
13
[125] L. Reynolds and K. McDonell, “Prompt programming for large
language models: Beyond the few-shot paradigm,” inExtended
abstracts of the 2021 CHI conference on human factors in
computing systems, pp. 1–7, 2021. 13
[126] L.Yang,H.Chen,Z.Li,X.Ding,andX.Wu,“Giveusthefacts:
Enhancing large language models with knowledge graphs for
fact-aware language modeling,”IEEE Transactions on Knowl-
edgeandDataEngineering, vol. 36, no. 7, pp. 3091–3110, 2024.
13
[127] G.Frisoni,A.Cocchieri,A.Presepi,G.Moro,andZ.Meng,“To
generateortoretrieve?ontheeffectivenessofartificialcontexts
formedicalopen-domainquestionanswering,”inProceedingsof
the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 9878–9919, 2024. 13
[128] Q.Gou,Z.Xia,B.Yu,H.Yu,F.Huang,Y.Li,andN.Cam-Tu,
“Diversify question generation with retrieval-augmented style
transfer,” inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pp. 1677–1690, 2023.
13[129] D.S.Mitra,“Ai-poweredadaptiveeducationfordisabledlearn-
ers,”Available at SSRN 5042713, 2024. 14
[130] H.Keuning,J.Jeuring,andB.Heeren,“Asystematicliterature
review of automated feedback generation for programming ex-
ercises,”ACMTransactionsonComputingEducation(TOCE),
vol. 19, no. 1, pp. 1–43, 2018. 14
[131] J. Rao and J. Lin, “Ramo: Retrieval-augmented genera-
tion for enhancing moocs recommendations,”arXiv preprint
arXiv:2407.04925, 2024. 14
[132] R. Taiwo, I. T. Bello, S. F. Abdulai, A.-M. Yussif, B. A.
Salami, A. Saka, and T. Zayed, “Generative ai in the con-
struction industry: A state-of-the-art analysis,”arXiv preprint
arXiv:2402.09939, 2024. 14
[133] Z. Levonian, C. Li, W. Zhu, A. Gade, O. Henkel, M.-E.
Postle, and W. Xing, “Retrieval-augmented generation to im-
prove math question-answering: Trade-offs between grounded-
ness and human preference,”arXivpreprintarXiv:2310.03184,
2023. 14
[134] S. McGregor, A. Ettinger, N. Judd, P. Albee, L. Jiang, K. Rao,
W. H. Smith, S. Longpre, A. Ghosh, C. Fiorelli,et al., “To err
is ai: A case study informing llm flaw reporting practices,” in
Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 39, pp. 28938–28945, 2025. 14
[135] Y. Jin, M. Chandra, G. Verma, Y. Hu, M. De Choudhury, and
S.Kumar,“Bettertoaskinenglish:Cross-lingualevaluationof
largelanguagemodelsforhealthcarequeries,”inProceedingsof
the ACM Web Conference 2024, pp. 2627–2638, 2024. 14
[136] N.Jiang,X.Li,S.Wang,Q.Zhou,S.B.Hossain,B.Ray,V.Ku-
mar, X. Ma, and A. Deoras, “Ledex: Training llms to better
self-debug and explain code,”Advances in Neural Information
Processing Systems, vol. 37, pp. 35517–35543, 2024. 14
[137] S. Huang, X. Tao, S. Yuan, Y. Zhang, P. Li, H. A. Beilinson,
Y. Zhang, W. Yu, P. Pontarotti, H. Escriva,et al., “Discovery
of an active rag transposon illuminates the origins of v (d) j
recombination,”Cell, vol. 166, no. 1, pp. 102–114, 2016. 14
[138] C.Tao,T.Shen,S.Gao,J.Zhang,Z.Li,K.Hua,W.Hu,Z.Tao,
and S. Ma, “Llms are also effective embedding models: An in-
depth overview,”arXiv preprint arXiv:2412.12591, 2024. 14
[139] M.Dassen,R.Kotula,K.Murray,A.Yates,D.Lawrie,E.Kayi,
J. Mayfield, and K. Duh, “Factum: mechanistic detection of ci-
tation hallucination in long-form rag,” inEuropeanConference
on Information Retrieval, pp. 272–288, Springer, 2026. 14
[140] L. Zha, J. Zhou, L. Li, R. Wang, Q. Huang, S. Yang, J. Yuan,
C. Su, X. Li, A. Su,et al., “Tablegpt: Towards unifying tables,
nature language and commands into one gpt,”arXiv preprint
arXiv:2307.08674, 2023. 14
[141] D. Zhu, X. Shen, X. Li, M. Elhoseiny,et al., “Minigpt-4:
Enhancing vision-language understanding with advanced large
language models,” inInternational Conference on Learning
Representations, vol. 2024, pp. 18378–18394, 2024. 14, 26
[142] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu,
P. Shi, Y. Shi,et al., “mplug-owl: Modularization empowers
large language models with multimodality,”arXiv preprint
arXiv:2304.14178, 2023. 14, 26
[143] Z. Ke, W. Kong, C. Li, M. Zhang, Q. Mei, and M. Bendersky,
“Bridging the preference gap between retrievers and llms,”
inProceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Papers),
pp. 10438–10451, 2024. 14
[144] V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James,
P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis,et al., “Ra-dit:
Retrieval-augmenteddualinstructiontuning,”inInternational
ConferenceonLearningRepresentations, vol. 2024, pp. 19138–
19162, 2024. 14
[145] M. Seo, J. Baek, J. Thorne, and S. J. Hwang, “Retrieval-
augmented data augmentation for low-resource domain tasks,”
arXiv preprint arXiv:2402.13482, 2024. 14
[146] D. Yang, J. Rao, K. Chen, X. Guo, Y. Zhang, J. Yang, and
Y. Zhang, “Im-rag: Multi-round retrieval-augmented genera-
tion through learning inner monologues,” inProceedings of the
47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, pp. 730–740, 2024. 15
[147] Y. Tevissen, K. Guetari, and F. Petitpont, “Towards retrieval
augmented generation over large video libraries,” in2024 16th
InternationalConferenceonHumanSystemInteraction(HSI),
pp. 1–4, IEEE, 2024. 15, 26
[148] F. Mumuni and A. Mumuni, “Explainable artificial intelligence

33
(xai): from inherent explainability to large language models,”
arXiv preprint arXiv:2501.09967, 2025. 15
[149] G. d. A. e Aquino, N. d. S. de Azevedo, L. Y. S. Okimoto,
L. Y. S. Camelo, H. L. de Souza Bragança, R. Fernandes,
A. Printes, F. Cardoso, R. Gomes, and I. G. Torné, “From rag
to multi-agent systems: A survey of modern approaches in llm
development,” 2025. 15
[150] M. Chen, Y. Li, Y. Yang, S. Yu, B. Lin, and X. He, “Automan-
ual:Constructinginstructionmanualsbyllmagentsviainterac-
tive environmental learning,”Advances in Neural Information
Processing Systems, vol. 37, pp. 589–631, 2024. 16
[151] S. Liu, Y. Chen, X. Xie, J. Siow, and Y. Liu, “Retrieval-
augmentedgenerationforcodesummarizationviahybridgnn,”
arXiv preprint arXiv:2006.05405, 2020. 16
[152] C. Guo, X. Liu, C. Xie, A. Zhou, Y. Zeng, Z. Lin, D. Song,
and B. Li, “Redcode: Risky code execution and generation
benchmark for code agents,”Advances in Neural Information
Processing Systems, vol. 37, pp. 106190–106236, 2024. 16
[153] F. Pan, M. Canim, M. Glass, A. Gliozzo, and J. Hendler,
“End-to-end table question answering via retrieval-augmented
generation,”arXiv preprint arXiv:2203.16714, 2022. 16
[154] M. Kang, J. M. Kwak, J. Baek, and S. J. Hwang, “Knowledge
graph-augmented language models for knowledge-grounded di-
aloguegeneration,”arXivpreprintarXiv:2305.18846,2023. 16,
26
[155] T.T.ProckoandO.Ochoa,“Graphretrieval-augmentedgener-
ation for large language models: A survey,” in2024 Conference
onAI,science,engineering,andtechnology(AIxSET),pp.166–
169, IEEE, 2024. 16, 26
[156] G. Kim, S. Kim, B. Jeon, J. Park, and J. Kang, “Tree of
clarifications: Answering ambiguous questions with retrieval-
augmented large language models,” inProceedings of the 2023
Conference on Empirical Methods in Natural Language Pro-
cessing, pp. 996–1009, 2023. 16
[157] D. Arora, A. Kini, S. R. Chowdhury, N. Natarajan, G. Sinha,
and A. Sharma, “Gar-meets-rag paradigm for zero-shot infor-
mation retrieval,”arXiv preprint arXiv:2310.20158, 2023. 16
[158] F. Luo and M. Surdeanu, “Divide & conquer for entailment-
aware multi-hop evidence retrieval,”arXiv preprint
arXiv:2311.02616, 2023. 16
[159] Y. Bai, Y. Miao, L. Chen, D. Wang, D. Li, Y. Ren,
H. Xie, C. Yang, and X. Cai, “Pistis-rag: Enhancing retrieval-
augmented generation with human feedback,”arXiv preprint
arXiv:2407.00072, 2024. 17
[160] T. W. Webb, K. J. Holyoak, and H. Lu, “Evidence from coun-
terfactualtaskssupportsemergentanalogicalreasoninginlarge
language models,”PNAS nexus, vol. 4, no. 5, p. pgaf135, 2025.
17
[161] X.Wang,H.Peng,Y.Guan,K.Zeng,J.Chen,L.Hou,X.Han,
Y.Lin,Z.Liu,R.Xie,etal.,“Maven-arg:Completingthepuzzle
of all-in-one event understanding dataset with event argument
annotation,” inProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long
Papers), pp. 4072–4091, 2024. 17
[162] T. Webb, K. J. Holyoak, and H. Lu, “Emergent analogical
reasoninginlargelanguagemodels,”NatureHumanBehaviour,
vol. 7, no. 9, pp. 1526–1541, 2023. 17
[163] P.Xu,W.Ping,X.Wu,L.McAfee,C.Zhu,Z.Liu,S.Subrama-
nian, E. Bakhturina, M. Shoeybi, and B. Catanzaro, “Retrieval
meets long context large language models,” inInternational
ConferenceonLearningRepresentations, vol. 2024, pp. 49569–
49584, 2024. 17
[164] B. Wang, W. Ping, L. McAfee, P. Xu, B. Li, M. Shoeybi, and
B.Catanzaro,“Instructretro:Instruction tuningpostretrieval-
augmented pretraining,”arXiv preprint arXiv:2310.07713,
2023. 17
[165] G. Chen, W. Yu, X. Lu, X. Zhang, E. Meng, and L. Sha,
“Unlocking multi-view insights in knowledge-dense retrieval-
augmented generation,”IEEE Transactions on Audio, Speech
and Language Processing, 2025. 17
[166] Z. Chen, J. Liu, H. Liu, Q. Cheng, F. Zhang, W. Lu,
and X. Liu, “Black-box opinion manipulation attacks to
retrieval-augmented generation of large language models,”
arXiv preprint arXiv:2407.13757, 2024. 17, 19
[167] L. Chen, R. Zhang, J. Guo, Y. Fan, and X. Cheng, “Con-
trolling risk of retrieval-augmented generation: A counterfac-
tual prompting framework,” inFindings of the Associationfor Computational Linguistics: EMNLP 2024(Y. Al-Onaizan,
M. Bansal, and Y.-N. Chen, eds.), (Miami, Florida, USA),
pp. 2380–2393, Association for Computational Linguistics,
Nov. 2024. 17
[168] K. Wang, F. Duan, P. Li, S. Wang, and X. Cai, “Llms know
what they need: Leveraging a missing information guided
framework to empower retrieval-augmented generation,” in
Proceedings of the 31st International Conference on Compu-
tational Linguistics, pp. 2379–2400, 2025. 17, 19
[169] W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu, “Dragin: Dynamic
retrieval augmented generation based on the real-time infor-
mation needs of large language models,” inProceedings of the
62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 12991–13013, 2024.
17, 19
[170] S. Wang, X. Yu, M. Wang, W. Chen, Y. Zhu, and Z. Dou,
“Richrag: Crafting rich responses for multi-faceted queries
in retrieval-augmented generation,” inProceedings of the
31st International Conference on Computational Linguistics,
pp. 11317–11333, 2025. 17
[171] Y.Lu,X.Zhao,andJ.Wang,“ClinicalRAG:Enhancingclinical
decision support through heterogeneous knowledge retrieval,”
inProceedings of the 1st Workshop on Towards Knowledgeable
Language Models (KnowLLM 2024)(S. Li, M. Li, M. J. Zhang,
E. Choi, M. Geva, P. Hase, and H. Ji, eds.), (Bangkok, Thai-
land), pp. 64–68, Association for Computational Linguistics,
Aug. 2024. 17
[172] Y. Ahn, S.-G. Lee, J. Shim, and J. Park, “Retrieval-augmented
response generation for knowledge-grounded conversation in
the wild,”IEEE Access, vol. 10, pp. 131374–131385, 2022. 17
[173] S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi,
R. Rana, and S. Nanayakkara, “Improving the domain adap-
tation of retrieval augmented generation (rag) models for open
domain question answering,”Transactions of the Association
for Computational Linguistics, vol. 11, pp. 1–17, 01 2023. 17
[174] T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica,
andJ.E.Gonzalez,“Raft:Adaptinglanguagemodeltodomain
specific rag,”arXiv preprint arXiv:2403.10131, 2024. 17
[175] S. Imani, L. Du, and H. Shrivastava, “MathPrompter: Mathe-
matical reasoning using large language models,” inProceedings
of the 61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 5: Industry Track)(S. Sitaram,
B. Beigman Klebanov, and J. D. Williams, eds.), (Toronto,
Canada), pp. 37–42, Association for Computational Linguis-
tics, July 2023. 17
[176] K. Verma, M. Moore, S. Wottrich, K. R. López, N. Aggar-
wal, Z. Bhatt, A. Singh, B. Unroe, S. Basheer, N. Sachdeva,
et al., “Emulating human cognitive processes for expert-level
medicalquestion-answeringwithlargelanguagemodels,”arXiv
preprint arXiv:2310.11266, 2023. 17
[177] J. Lála, O. O’Donoghue, A. Shtedritski, S. Cox, S. G. Ro-
driques, and A. D. White, “Paperqa: Retrieval-augmented
generative agent for scientific research,”arXiv preprint
arXiv:2312.07559, 2023. 18
[178] Y. Ke, L. Jin, K. Elangovan, H. R. Abdullah, N. Liu, A. T. H.
Sia, C. R. Soh, J. Y. M. Tung, J. C. L. Ong, and D. S. W. Ting,
“Development and testing of retrieval augmented generation
in large language models–a case study report,”arXiv preprint
arXiv:2402.01733, 2024. 18
[179] Y. Shi, T. Yang, C. Chen, Q. Li, T. Liu, X. Li, and N. Liu,
“Searchrag:cansearchenginesbehelpfulforllm-basedmedical
question answering?,”arXiv preprint arXiv:2502.13233, 2025.
18
[180] R. Kalra, Z. Wu, A. Gulley, A. Hilliard, X. Guan,
A. Koshiyama, and P. C. Treleaven, “HyPA-RAG: A hybrid
parameter adaptive retrieval-augmented generation system for
AI legal and policy applications,” inProceedings of the 2025
Conference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human Language
Technologies(Volume3:IndustryTrack),pp.1036–1054,Asso-
ciation for Computational Linguistics, 2025. 18
[181] N. Wiratunga, R. Abeyratne, L. Jayawardena, K. Mar-
tin, S. Massie, I. Nkisi-Orji, R. Weerasinghe, A. Liret, and
B. Fleisch, “CBR-RAG: case-based reasoning for retrieval aug-
mented generation in LLMs for legal question answering,” in
International Conference on Case-Based Reasoning, pp. 445–
460, Springer Nature Switzerland, 2024. 18

34
[182] A.Majumder,K.Bhattacharya,andA.Chakrabarti,“Develop-
ment and evaluation of a retrieval-augmented generation tool
for creating sapphire models of artificial systems,” inInter-
national Conference on Research into Design, pp. 489–504,
Springer, 2025. 18
[183] S. Kahl, F. Löffler, M. Maciol, F. Ridder, M. Schmitz,
J. Spanagel, J. Wienkamp, C. Burgahn, and M. Schilling, “En-
hancing ai tutoring in robotics education: evaluating the effect
of retrieval-augmented generation and fine-tuning on large lan-
guage models,”Autonomous Intelligent Systems Group, 2024.
18
[184] S. Wang, J. Liu, S. Song, J. Cheng, Y. Fu, P. Guo, K. Fang,
Y.Zhu,andZ.Dou,“Domainrag:Achinesebenchmarkforeval-
uating domain-specific retrieval-augmented generation,”arXiv
preprint arXiv:2406.05654, 2024. 18
[185] P. Xia, K. Zhu, H. Li, T. Wang, W. Shi, S. Wang, L. Zhang,
J. Y. Zou, and H. Yao, “Mmed-rag: Versatile multimodal rag
system for medical vision language models,” inInternational
ConferenceonLearningRepresentations, vol. 2025, pp. 66188–
66217, 2025. 18
[186] G. Guinet, B. Omidvar-Tehrani, A. Deoras, and L. Cal-
lot, “Automated evaluation of retrieval-augmented language
models with task-specific exam generation,”arXiv preprint
arXiv:2405.13622, 2024. 18
[187] Y. Tao, Y. Li, Y. Qin, and Y. Liu, “Retrieval-augmented
code generation: A survey with focus on repository-level ap-
proaches,”arXiv preprint arXiv:2510.04905, 2025. 19
[188] K. Du, J. Chen, R. Rui, H. Chai, L. Fu, W. Xia, Y. Wang,
R. Tang, Y. Yu, and W. Zhang, “Codegrag: Bridging the
gap between natural language and programming language
via graphical retrieval augmented generation,”arXiv preprint
arXiv:2405.02355, 2024. 19
[189] Z. Yu, S. Liu, P. Denny, A. Bergen, and M. Liut, “Integrating
small language models with retrieval-augmented generation
in computing education: Key takeaways, setup, and practical
insights,”inProceedingsofthe56thACMTechnicalSymposium
on Computer Science Education V. 1, pp. 1302–1308, 2025. 19
[190] Y.-Z. Lin, K. Petal, A. H. Alhamadah, S. Ghimire, M. W.
Redondo,D.R.V.Corona,J.Pacheco,S.Salehi,andP.Satam,
“Personalized education with generative ai and digital twins:
Vr, rag, and zero-shot sentimentanalysisfor industry 4.0 work-
forcedevelopment,”arXivpreprintarXiv:2502.14080,2025. 19
[191] T. Cai, Z. Tan, X. Song, T. Sun, J. Jiang, Y. Xu, Y. Zhang,
and J. Gu, “Forag: Factuality-optimized retrieval augmented
generationforweb-enhancedlong-formquestionanswering,”in
Proceedings of the 30th ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining, pp. 199–210, 2024. 19, 25
[192] W. Xie, X. Liang, Y. Liu, K. Ni, H. Cheng, and Z. Hu,
“Weknow-rag: An adaptive approach for retrieval-augmented
generation integrating web search and knowledge graphs,”
arXiv preprint arXiv:2408.07611, 2024. 19
[193] Z. Shi, S. Zhang, W. Sun, S. Gao, P. Ren, Z. Chen, and Z. Ren,
“Generate-then-ground in retrieval-augmented generation for
multi-hop question answering,” inProceedings of the 62nd An-
nual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 7339–7353, 2024. 19
[194] S. Sharma, D. S. Yoon, F. Dernoncourt, D. Sultania, K. Bagga,
M. Zhang, T. Bui, and V. Kotte, “Retrieval augmented gener-
ation for domain-specific question answering,”arXiv preprint
arXiv:2404.14760, 2024. 19
[195] Z. Zhang, M. Fang, and L. Chen, “Retrievalqa: Assessing
adaptive retrieval-augmented generation for short-form open-
domain question answering,” inFindingsoftheAssociationfor
ComputationalLinguistics:ACL2024,pp.6963–6975,2024. 19
[196] S. Setty, H. Thakkar, A. Lee, E. Chung, and N. Vidra, “Im-
proving retrieval for rag based question answering models on
financial documents,” 2024. 19
[197] C.-M. Chan, C. Xu, R. Yuan, H. Luo, W. Xue, Y. Guo, and
J. Fu, “Rq-rag: Learning to refine queries for retrieval aug-
mented generation,” 2024. 19
[198] H. Yang, Z. Li, Y. Zhang, J. Wang, N. Cheng, M. Li, and
J. Xiao, “PRCA: Fitting black-box large language models for
retrieval question answering via pluggable reward-driven con-
textualadapter,”inProceedingsofthe2023ConferenceonEm-
pirical Methods in Natural Language Processing(H. Bouamor,
J. Pino, and K. Bali, eds.), (Singapore), pp. 5364–5375, Associ-
ation for Computational Linguistics, Dec. 2023. 19[199] Y. Xu, T. Cai, J. Jiang, and X. Song, “Face4rag: Factual
consistency evaluation for retrieval augmented generation in
chinese,” inProceedings of the30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining, KDD ’24, (New
York, NY, USA), p. 6083–6094, Association for Computing
Machinery, 2024. 19, 25, 26
[200] M. Chen, J. Tworek, and H. Jun, “Evaluating large language
models trained on code,” 2021. 20
[201] GitHub, “Introducing github copilot: Your ai pair program-
mer.” https://github.blog, 2021. 20
[202] H. Tan, Q. Luo, L. Jiang, Z. Zhan, J. Li, H. Zhang, and
Y. Zhang, “Prompt-based code completion via multi-retrieval
augmented generation,”ACM Trans. Softw. Eng. Methodol.,
vol. 35, Dec. 2025. 20
[203] N. Baumann, J. S. Diaz, J. Michael, L. Netz, H. Nqiri,
J. Reimer, and B. Rumpe, “Combining retrieval-augmented
generation and few-shot learning for model synthesis of un-
commonDSLs,”inModellierung2024SatelliteEvents,pp.10–
18420, Gesellschaft für Informatik e.V., 2024. 20
[204] Y. Wang, H. Le, A. Gotmare, N. Bui, J. Li, and S. Hoi,
“Codet5+: Open code large language models for code under-
standing and generation,” inProceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Processing,
pp. 1069–1088, 2023. 20
[205] Z. F. Han, J. Lin, A. Gurung, D. R. Thomas, E. Chen,
C. Borchers, S. Gupta, and K. R. Koedinger, “Improving as-
sessmentoftutoringpracticesusingretrieval-augmentedgener-
ation,”arXiv preprint arXiv:2402.14594, 2024. 21
[206] S. Jacobs and S. Jaschke, “Leveraging lecture content for
improved feedback: Explorations with GPT-4 and retrieval
augmented generation,” in2024 36th International Conference
on Software Engineering Education and Training (CSEE&T),
pp. 1–5, IEEE, 2024. 21
[207] M. Bucur, “Exploring large language models and retrieval
augmented generation for automated form filling,” bachelor’s
thesis, University of Twente, 2023. 21
[208] Y. Hu and Y. Lu, “RAG and RAU: A survey on retrieval-
augmented language model in natural language processing,”
arXiv preprint arXiv:2404.19543, 2024. 22
[209] G. Chen, W. Yu, X. Lu, X. Zhang, E. Meng, and L. Sha,
“Unlocking multi-view insights in knowledge-dense retrieval-
augmented generation,”IEEE Transactions on Audio, Speech
and Language Processing, 2025. 22
[210] H. Zhou, H. Gu, Z. Zhan, X. Liu, K. Zhou, Y. Xiao, M. Liang,
et al., “The efficiency vs. accuracy trade-off: Optimizing RAG-
enhanced LLM recommender systems using multi-head early
exit,” inProceedings of the 63rd Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long Papers),
pp. 26443–26458, 2025. 23
[211] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large
language models in retrieval-augmented generation,” inPro-
ceedings of the AAAI Conference on Artificial Intelligence,
vol. 38, pp. 17754–17762, 2024. 23
[212] M. Shen, M. Umar, K. Maeng, G. E. Suh, and U. Gupta, “To-
wards understanding systems trade-offs in retrieval-augmented
generationmodelinference,”arXivpreprintarXiv:2412.11854,
2024. 23
[213] A. Balaguer, V. Benara, R. L. d. F. Cunha, T. Hendry, D. Hol-
stein,J.Marsman,N.Mecklenburg,etal.,“RAGvsfine-tuning:
pipelines, tradeoffs, and a case study on agriculture,”arXiv
preprint arXiv:2401.08406, 2024. 23
[214] Q. Zhang, S. Chen, Y. Bei, Z. Yuan, H. Zhou, Z. Hong,
H. Chen,et al., “A survey of graph retrieval-augmented gen-
eration for customized large language models,”arXiv preprint
arXiv:2501.13958, 2025. 23
[215] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Pas-
sos,S.Shakeri,etal.,“PaLM2technicalreport,”arXivpreprint
arXiv:2305.10403, 2023. 23
[216] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large
language models in retrieval-augmented generation,” inPro-
ceedings of the AAAI Conference on Artificial Intelligence,
vol. 38, pp. 17754–17762, 2024. 23
[217] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang,
W. Zhang, J. Jiang, and B. Cui, “Retrieval-augmented genera-
tion for ai-generated content: A survey,” 2026. 23
[218] Z. R. Wang, Z. Wang, L. Le, H. S. Zheng, S. Mishra, V. Perot,
Y. Zhang, A. Mattapalli, A. Taly, J. Shang,et al., “Speculative

35
rag: Enhancing retrieval augmented generation through draft-
ing,” 2025. 23
[219] Y. Zhu, H. Yuan, S. Wang, J. Liu, W. Liu, C. Deng, H. Chen,
Z. Liu, Z. Dou, and J.-R. Wen, “Large language models for
information retrieval: A survey,” 2025. 23
[220] S. Li, L. Stenzel, C. Eickhoff, and S. A. Bahrainian, “Enhanc-
ing retrieval-augmented generation: a study of best practices,”
2025. 23
[221] P. Wu, X. Zhang, W. Yu, X. Liu, X. Du, and Z. Z. Chen, “Do
retrieval-augmented language models adapt to varying user
needs?,” 2025. 23
[222] R. Buchmann, J. Eder, H.-G. Fill, U. Frank, D. Karagiannis,
E. Laurenzi, J. Mylopoulos, D. Plexousakis, and M. Y. Santos,
“Large language models: Expectations for semantics-driven
systemsengineering,”Data&KnowledgeEngineering,vol.152,
p. 102324, 2024. 24
[223] Z. Sun, X. Zang, K. Zheng, J. Xu, X. Zhang, W. Yu, Y. Song,
and H. Li, “Redeep: Detecting hallucination in retrieval-
augmented generation via mechanistic interpretability,” inIn-
ternationalConferenceonLearningRepresentations,vol.2025,
pp. 50250–50279, 2025. 24
[224] S. Gupta, R. Ranjan, and S. N. Singh, “A comprehensive sur-
veyofretrieval-augmentedgeneration(rag):Evolution,current
landscape and future directions,” 2024. 24, 25
[225] S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi,
R. Rana, and S. Nanayakkara, “Improving the domain adap-
tation of retrieval augmented generation (rag) models for open
domain question answering,”Transactions of the Association
for Computational Linguistics, vol. 11, pp. 1–17, 2023. 24
[226] A. Misrahi, N. Chirkova, M. Louis, and V. Nikoulina,
“Adapting large language models for multi-domain retrieval-
augmented-generation,” 2025. 24
[227] S. Rakin, M. A. R. Shibly, Z. M. Hossain, M. M. Akbar,
and Z. Khan, “Leveraging the domain adaptation of retrieval
augmented generation models for question answering and for
hallucinationreduction,”inInternationalConferenceonInfor-
mation Technology-New Generations, pp. 482–493, Springer,
2025. 24
[228] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu,
H. Liu, Y. Xing, and J. Tang, “Mitigating the privacy issues in
retrieval-augmented generation (rag) via pure synthetic data,”
inProceedingsofthe2025ConferenceonEmpiricalMethodsin
Natural Language Processing, pp. 24538–24569, 2025. 24
[229] S. Wu, Y. Xiong, Y. Cui, H. Wu, C. Chen, Y. Yuan, L. Huang,
X. Liu, T.-W. Kuo, N. Guan,et al., “Retrieval-augmented
generation for natural language processing: A survey,” 2024.
24
[230] V. Kanka,Scaling big data: leveraging LLMs for enterprise
success. Libertatem Media Private Limited, 2024. 24
[231] M. A. Ferrag, F. Alwahedi, A. Battah, B. Cherif, A. Mechri,
N. Tihanyi, T. Bisztray, and M. Debbah, “Generative ai in
cybersecurity: A comprehensive review of llm applications and
vulnerabilities,”Internet of Things and Cyber-Physical Sys-
tems, vol. 5, pp. 1–46, 2025. 24
[232] R. Fayyazi, S. H. Trueba, M. Zuzak, and S. J. Yang,
“Proverag: Provenance-driven vulnerability analysis with au-
tomated retrieval-augmented llms,”IEEE Access, vol. 13,
pp. 212815–212826, 2025. 24
[233] D. Sukhwal, “Retrieval augmented generation: An evaluation
of RAG-based chatbot for customer support,”Retrieval Aug-
mented Generation: An Evaluation of RAG-based Chatbot for
Customer Support, 2024. 24
[234] D.Zhao,“Frag:Towardfederatedvectordatabasemanagement
for collaborative and secure retrieval-augmented generation,”
2024. 24
[235] P. Shojaee, S. S. Harsha, D. Luo, A. Maharaj, T. Yu, and
Y. Li, “Federated retrieval augmented generation for multi-
product question answering,” inProceedings of the 31st In-
ternationalConferenceonComputationalLinguistics:Industry
Track, pp. 387–397, 2025. 25
[236] C. Jin, Z. Zhang, X. Jiang, F. Liu, S. Liu, X. Liu, and
X. Jin, “Ragcache: Efficient knowledge caching for retrieval-
augmented generation,” 2025. 25
[237] Z. Yue, H. Zhuang, A. Bai, K. Hui, R. Jagerman, H. Zeng,
Z. Qin, D. Wang, X. Wang, and M. Bendersky, “Inference
scalingforlong-contextretrievalaugmentedgeneration,”inIn-
ternationalConferenceonLearningRepresentations,vol.2025,pp. 72914–72938, 2025. 25
[238] H.Zhao,F.Yang,B.Shen,H.Lakkaraju,andM.Du,“Towards
uncovering how large language model works: An explainability
perspective,”arXiv preprint arXiv:2402.10688, 2024. 25
[239] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu, “Eval-
uation of retrieval-augmented generation: A survey,” inCCF
Conference on Big Data, pp. 102–120, Springer, 2024. 25
[240] A. Salemi and H. Zamani, “Evaluating retrieval quality in
retrieval-augmented generation,” inProceedings of the 47th
International ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval, pp. 2395–2400, 2024. 25
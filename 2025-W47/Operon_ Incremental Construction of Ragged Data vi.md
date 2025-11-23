# Operon: Incremental Construction of Ragged Data via Named Dimensions

**Authors**: Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, Minhyeong Lee

**Published**: 2025-11-20 06:16:31

**PDF URL**: [https://arxiv.org/pdf/2511.16080v1](https://arxiv.org/pdf/2511.16080v1)

## Abstract
Modern data processing workflows frequently encounter ragged data: collections with variable-length elements that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of named dimensions with explicit dependency relations. Operon provides a domain-specific language where users declare pipelines with dimension annotations that are statically verified for correctness, while the runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We formalize the mathematical foundation for reasoning about partial shapes and prove that Operon's incremental construction algorithm guarantees deterministic and confluent execution in parallel settings. The system's explicit modeling of partially-known states enables robust persistence and recovery mechanisms, while its per-task multi-queue architecture achieves efficient parallelism across heterogeneous task types. Empirical evaluation demonstrates that Operon outperforms an existing workflow engine with 14.94x baseline overhead reduction while maintaining near-linear end-to-end output rates as workloads scale, making it particularly suitable for large-scale data generation pipelines in machine learning applications.

## Full Text


<!-- PDF content starts -->

Operon: Incremental Construction of Ragged Data via Named
Dimensions
SUNGBIN MOON,Asteromorph, Republic of Korea
JIHO PARK,Asteromorph, Republic of Korea
SUYOUNG HWANG,Asteromorph, Republic of Korea
DONGHYUN KOH,Asteromorph, Republic of Korea
SEUNGHYUN MOON,Asteromorph, Republic of Korea
MINHYEONG LEE∗,Asteromorph, Republic of Korea
Modern data processing workflows frequently encounter ragged data: collections with variable-length elements
that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI
agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent
to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We
present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of
named dimensions with explicit dependency relations. Operon provides a domain-specific language where
users declare pipelines with dimension annotations that are statically verified for correctness, while the
runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We
formalize the mathematical foundation for reasoning about partial shapes and prove that Operon’s incremental
construction algorithm guarantees deterministic and confluent execution in parallel settings. The system’s
explicit modeling of partially-known states enables robust persistence and recovery mechanisms, while its
per-task multi-queue architecture achieves efficient parallelism across heterogeneous task types. Empirical
evaluation demonstrates that Operon outperforms an existing workflow engine with 14.94 ×baseline overhead
reduction while maintaining near-linear end-to-end output rates as workloads scale, making it particularly
suitable for large-scale data generation pipelines in machine learning applications.
CCS Concepts:•Software and its engineering →Data flow architectures;Domain specific languages;
Automated static analysis;•Theory of computation→Concurrent algorithms; Operational semantics.
Additional Key Words and Phrases: ragged arrays, named dimensions, order theory, incremental computation,
workflow engines
1 Introduction
Modern data processing workflows often involve collections of recurring data with variable length.
Such forms of data, known asragged data, arise naturally in many domains:
•In natural language processing, bodies of text contain varying numbers of paragraphs,
sentences, and tokens [18, 28].
•Repeated scientific measurements may yield records of differing lengths on each run.
•Vision tasks introduce images with an unknown number of detected regions, captions, or
annotations depending on their content [12, 19].
•Autonomous large language model (LLM) agents routinely generate action traces or message
streams of unpredictable size [25].
∗Correspondence to Minhyeong Lee.
Authors’ Contact Information: Sungbin Moon, sb.moon@asteromorph.com, Asteromorph, Seoul, Republic of Korea; Jiho
Park, jh.park@asteromorph.com, Asteromorph, Seoul, Republic of Korea; Suyoung Hwang, sy.hwang@asteromorph.com,
Asteromorph, Seoul, Republic of Korea; Donghyun Koh, dh.koh@asteromorph.com, Asteromorph, Seoul, Republic of
Korea; Seunghyun Moon, sh.moon@asteromorph.com, Asteromorph, Seoul, Republic of Korea; Minhyeong Lee, mh.lee@
asteromorph.com, Asteromorph, Seoul, Republic of Korea.arXiv:2511.16080v1  [cs.PL]  20 Nov 2025

2 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
SciCapRow
extract_captioned_figures
CaptionedFigget_paper_id
PaperId
extract_body_text
BodyText
regex_match
MentionPgocr_extract
OcrToken
collect_row
Row(a) SciCap+
PaperId
parse_paper
ParsedPaper
extract_captioned_figures
CaptionedFigextract_sections
Section
extract_paragraphs
Paragraph
vlm_evaluate
Relevance
filter_aggregate
RelevantPgocr_extract
OcrToken
collect_row
Row(b) Ours
Fig. 1. Workflows for scientific figure captioning. Rounded boxes denote data entries, and rectangles denote
processing tasks. (a) Original SciCap+ pipeline [ 39] extracts a single paragraph 𝐾textper figure𝐼using regex
matching. (b) Our pipeline introduces a vision-language model (VLM) agent to assess and gather multiple
relevant paragraphs𝐾′
text.
As these workflows scale to process large chunks of data, indexing, batching, and dependency
management become increasingly important. However, the variation in length complicates handling
such data, and the fact that some lengths remain unknown before execution only exacerbates this
complexity. Existing workflow engines do not reason about the shapes and dependencies integral to
ragged data, and the burden of bookkeeping falls on the user [ 27,33]. To address these challenges,
we presentOperon, a Rust-based workflow engine that natively supports ragged data pipelines
throughnamed dimensions with dependencies.
1.1 Motivating Example
Let us consider an example workflow (Fig. 1b) to motivate our work. The SciCap dataset [ 15]
defines the task of caption generation as the prediction of a caption 𝐶given a scientific figure 𝐼;
the extension SciCap+ [ 39] augments this task by providing additional knowledge 𝐾extracted
from the associated paper. The resulting dataset contains rows of (𝐼,𝐾,𝐶) tuples, where 𝐾consists
of one paragraph 𝐾textthat directly mentions the figure and OCR-extracted tokens 𝐾vision from
the figure itself. As shown in Fig. 1a, the workflow used to generate SciCap+ extracts 𝐾textusing
regular expression matching on the paper text and persists at most one paragraph per figure.
Our example workflow in Fig. 1b is a proposed enhancement that addresses limitations of regex
matching by introducing a vision-language model (VLM) agent to assess paragraph relevance [ 37].
We begin the workflow from the raw paper PDFs and extract all necessary components using
existing tools such as PDFFigure 2.0 [ 8]. Given a figure-caption pair (𝐼,𝐶) , the agent independently

Operon: Incremental Construction of Ragged Data via Named Dimensions 3
scores all paragraphs extracted from the paper’s body text. Paragraphs that meet a fixed relevance
threshold are then aggregated to form 𝐾′
text. This approach allows the dataset to encapsulate
relevant information spread across multiple paragraphs, even potentially those that do not explicitly
reference the figure. In this example, we observe a degree ofraggednessthroughout the flow: the
number of OCR tokens varies per figure and per paper; the number of VLM inferences depends on
both the number of figures and the number of paragraphs.
1.2 Challenges
Expressing and concurrently executing ragged data pipelines pose several challenges, as listed
below. We narrow our focus to pipelines that can be described as directed acyclic graphs (DAGs)
with many-to-one edges, where each node represents a type of data entry and each edge represents
a data processing task that transforms input entries to output entries.
Unintuitive code structure.When expressing data pipelines as code, each task would typically be
represented as a function call, and the overall pipeline would be structured as a sequence of such
calls. However, this structure quickly becomes unintuitive when tasks need to be repeated, nested, or
parallelized, as the overall sequence becomes cluttered with control flow constructs and dependency
bookkeeping [ 4,6]. This problem calls for a higher-level abstraction that clearly separates the
task definitions from their execution logic [ 20]: the task definitions should immediately match
the logical structure of the workflow DAG, while the implementation of each task should remain
self-contained.
Ambiguity in repetition.When describing each task as a function in a data pipeline, the repetition
behavior of a multidimensional task remains unclear when provided only with the usual function
signature. Certain batch-operation tasks, such as zipping, masking, or aggregating, require their
input lengths and shapes to be aligned along specific axes [ 13,36]. This information, while evident to
the user due to the context of the pipeline, cannot be inferred and enforced by the function signature
alone [ 24]. Due to this, it remains a challenge to design a system that can express relationships
across axes clearly and unambiguously.
Late discovery of tasks and data lengths.Since the DAG structure is not fully known before
execution, static DAG scheduling algorithms cannot be directly applied. The uncertainty in the
number of upstream tasks and data entries complicates dependency management and parallelism.
Tasks can only be lazily scheduled when a quota of dependencies has been met, where the quota
needs to be dynamically updated as the number of dependencies gradually becomes known during
execution. Prior works on dynamic DAG scheduling [ 31,35] mainly focus on optimizing resource
utilization in known DAG structures rather than the runtime discovery of tasks and data lengths;
our task is to exploit the characteristics that emerge from ragged data pipelines to design a dynamic
scheduling system that fits the use case.
1.3 Our Design
Our design, Operon, addresses the above challenges by providing a domain-specific language
(DSL) for pipeline definition and a runtime system for dynamic scheduling. Figure 2 demonstrates
how Operon expresses the motivating example shown earlier in Section 1.1. In a macro-implemented
DSL, users declare their pipelines as combinations of tasks, where each task definition resembles a
function signature with additional dimension annotations (named dimensions). After some static
checks during macro expansion, Operon provides the runtime system that dynamically schedules
user-defined tasks as specified by the pipeline definition.

4 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
1operon::define_operon! {
2sci_cap_enhanced = {
3PaperId<p> = get_paper_id();
4ParsedPaper = parse_paper(PaperId) for p;
5CaptionedFig<f> = extract_captioned_figures(ParsedPaper) for p;
6Section<s> = extract_sections(ParsedPaper) for p;
7Paragraph<g> = extract_paragraphs(Section) for p, s;
8Relevance = vlm_evaluate(CaptionedFig, Paragraph) for p, f, s, g;
9RelevantPg<r> = filter_aggregate(Paragraph<s, g>, Relevance<s, g>) for p, f;
10OcrToken<t> = ocr_extract(CaptionedFig) for p, f;
11Row = collect_row(CaptionedFig, RelevantPg<r>, OcrToken<t>) for p, f;
12}
13}
Fig. 2. Operon pipeline definitions for the motivating example. Dimensions are explicitly declared and tracked
through the pipeline. Angle brackets denote iteration and aggregation axes.
Here, dimensions are equipped with an inferreddependency relationthat describes how their
lengths depend on one another. For example, in Fig. 2, the dimension fover different figures
depends on the prior dimension p, as the number of figures varies per paper. Operon tracks these
relationships throughout the pipeline and statically verifies that nonsense iterations or aggregations
do not occur. While the idea of naming dimensions has been explored in several prior works and
frameworks ([ 14,26,32]), Operon holds the novelty of elevating this concept to accommodate
dependency relationships. This abstraction provides an implicit control flow logic that relieves
users from manually managing iterations, repetitions, and dependencies.
The strong theoretical foundation of Operon creates several advantages. By explicating the
partially-known states during execution, Operon holds a unique ability to persist intermediate
states and recover from previous runs. Moreover, Operon’s per-task multi-queue system allows
tasks to be scheduled as soon as their dependencies are met, which is crucial for parallelism across
task types. The resulting system shows consistently low latency, high scalability, and notably a
steady end-to-end output rate compared to an existing workflow engine, as we demonstrate later
in this paper.
1.4 Contributions
Our main contribution is the design and implementation ofOperon, an incremental workflow
scheduling engine with a statically verified DSL interface and an automatically generated runtime
system. Technical contributions presented in this paper include:
•Formalism of dimensional dependencies(Section 2). We introduce a mathematical framework
for reasoning about named dimensions and their dependencies and show how ragged data
can be represented within this framework.
•Structured model for partial data and incrementality(Section 3). We give explicit representa-
tions for partially-known data states that arise during the execution of ragged data pipelines.
We further find which transformations are compatible with each given state and prove that
this model enables confluent execution in parallel.
•Operon DSL and runtime system(Section 4). We present the syntax and verification methods
for Operon pipelines and describe how the runtime system dynamically schedules tasks
based on the pipeline definition and the current data states.

Operon: Incremental Construction of Ragged Data via Named Dimensions 5
•Evaluation of Operon(Section 5). Empirical experiments demonstrate that Operon outper-
forms an existing workflow engine, Prefect, exhibiting a14 .94×baseline overhead reduction
while maintaining a near-linear end-to-end output rate as the workload scales.
2 Ragged and Named Dimensions
In this section, we formalize the concepts to describe ragged data. For this, we present a system of
named dimensions equipped with an explicit dependency relation, and develop a generalization of
multidimensional arrays on top of this system.1
2.1 Dimensions
We define named dimensions, or simply dimensions, as identifiers for each axis of repeated data.
In rectangular arrays, the behavior of each dimension is invariant with respect to the others, and
hence we may treat each dimension independently. However, in ragged arrays, the size of one
dimension may depend on the position along another dimension. To make this dependency explicit,
we introduce the following definition.
Definition 2.1 (Dimension spaces).Adimension space (D,≺) is a finite setDof dimensions with
a strict partial order ≺. We denote the reflexive closure of ≺as⪯. The relation 𝑑≺𝑒 means that
𝑒dependson𝑑; in this relationship, 𝑑is theancestor, and 𝑒is thedescendant. If neither 𝑑⪯𝑒 nor
𝑒⪯𝑑, we write𝑑∥𝑒and say𝑑and𝑒areindependent.
Example 2.2.Recall the motivating example shown in Section 1.1. The dimension space induced
by this example would be:
D={p,f,t,r,s,g}.𝑒
𝑑≺𝑒p f t r s gDescription of𝑑
𝑑pFT T T T TPapers
fF FT TF F Figures
tF F F F F F OCR Tokens
rF F F F F F Relevant paragraphs
sF F F F FTSections
gF F F F F F Paragraphs
We may confirm that the intuitive dependencies translate well into the relation ≺: for example,
p≺f as figures depend on papers, s≺g as paragraphs depend on sections, and f∥s as figures
and sections are independent.
Since dimension spaces are finite posets, we adopt the following standard notions in our setting.
Definition 2.3 (Structure of dimension spaces).Given a dimension space (D,≺) , we define the
following terms:
(1)AsubspaceofDis an induced subposet (E,≺|E)for anyE ⊆D . All subsetsE ⊆D
discussed here and below are assumed to be subspaces with the induced order≺| E.
(2)Aprimary dimensionis a minimal element of D. That is, a dimension 𝑑∈D such that there
exists no𝑒∈D with𝑒≺𝑑 . Every nonempty dimension space contains at least one primary
dimension since finite nonempty posets always have minimal elements.
(3)Downward closures and downward closed subposets are simply referred to asclosuresand
closed subspaces. We useclosednessin place ofdownward closednesssince upward closedness
is irrelevant to the discussion. A closure of a subspace EisE↓=Ð
𝑒∈E{𝑑∈D|𝑑⪯𝑒},
1All proofs are provided in Appendix B.

6 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
while a closed subspace FsatisfiesF↓=F; for a singleton subspace {𝑑}, we write𝑑↓={𝑑}↓
and call such closuresprincipal.
(4)Thedependency spaceof a subspace Eis defined as Dep(E)=E↓\E. For a singleton
subspace{𝑑}, we write Dep(𝑑)=Dep({𝑑})and also call such dependency spacesprincipal.
(5) A subspaceEisconvexif and only ifDep(E)is closed.
Intuitively, a dimension is primary if it does not require reference to other dimensions. A closed
subspace extends this idea to multiple dimensions. In a closed subspace, all ancestors of each
dimension can be found within itself, making the subspace self-contained. Thus, discussions about
closed subspaces typically do not require context beyond the subspace itself.
However, this is not generally true for every subspace. While discussing each subspace E, we
must also handle the external dependencies, which the dependency space Dep(E) represents.
In most cases, we would like to establish these dependencies beforehand, which would require
inspecting Dep(E) as a standalone subspace. To this end, convexity provides a helpful guarantee
that all ancestors of Ecan be fixed without referring back to E. Applications such as subcoordinates
(Def. 2.10) and subarrays (Def. 2.13), therefore, rely on convexity to avoid circular dependencies.
We conclude this section with a justification of the nameconvex.
Lemma 2.4.A subspace E⊆D is convex if and only if it is an order-convex subposet, that is, if
𝑑,𝑒∈E,𝑓∈D, and𝑑⪯𝑓⪯𝑒, then𝑓∈E.
Corollary 2.5.Every principal dependency space is closed.
2.2 Shapes and Coordinates
To associate the dimensions defined in Section 2.1 with a structure that holds data, we must decide
how to index data across dimensions. The obvious answer is to use “coordinates”—maps from
dimensions to nonnegative integers—to specify which “cell” a piece of data belongs in.
In rectangular arrays, coordinates are confined with a simple tuple of lengths, theshape, that
specifies the acceptable indices along each dimension. We generalize this notion to ragged arrays
by introducingresolutions, which specify lengths in a dependency-aware manner.
Definition 2.6 (Resolutions).On a dimension space (D,≺) , aresolutionof a dimension 𝑑∈D is
a tuple(𝑑,𝑐,ℓ) where𝑐∈[Dep(𝑑)→N 0]andℓ∈N 0.2Aresolution mapis a set of resolutions 𝑅
that satisfies(𝑑,𝑐,ℓ 1),(𝑑,𝑐,ℓ 2)∈𝑅→ℓ 1=ℓ2.
In this definition, a single resolution is a mapping that, given a dimension 𝑑and a total map 𝑐
over the ancestor dimensions of 𝑑, returns a nonnegative length ℓ. A lengthℓwould accommodate
values in[0,ℓ)along𝑑; we explicitly allow ℓ=0for “empty” dimensions with no valid values.
We henceforth interpret a resolution map 𝑅as a partial function D×[D⇀N 0]⇀N 0(or, more
precisely,Ð
𝑑∈D({𝑑}×[Dep(𝑑)→N 0])⇀N 0).
As shown in Figure 3, a well-chosen resolution map may shape a ragged profile. However,
since each resolution carries information about the ancestor dimensions, we must verify that the
resolutions do not contradict themselves. Specifically, in each occurrence of a position 𝑐, we must
check each dimension 𝑑∈dom(𝑐) to see if the resolution of 𝑑allows the value of 𝑐(𝑑) at that
position. For this, we define a condition that verifies whether a position 𝑐over some dimensions is
valid under a resolution map𝑅.
2We use the notation [·→·] to denote the set of total functions from the domain to the codomain, and similarly, [·⇀·]
for partial functions.

Operon: Incremental Construction of Ragged Data via Named Dimensions 7
𝑅(s,{p↦→0})=5𝑅(g,{p↦→0,s↦→4})=3𝑅(f,{p↦→0})=3
𝑅(p,∅)=1𝑅={
(p,∅,1),
(s,{p↦→0},5),
(g,{p↦→0,s↦→0},4),
(g,{p↦→0,s↦→1},3),
(g,{p↦→0,s↦→2},2),
(g,{p↦→0,s↦→3},0),
(g,{p↦→0,s↦→4},3),
(f,{p↦→0},3)
}
Fig. 3. A resolution map 𝑅on the dimension space {p,s,g,f}from Example 2.2 defining a ragged profile. For
the single paper shown, there are 3 figures and 5 sections; each section contains 4, 3, 2, 0, and 3 paragraphs,
respectively. This configuration uniquely defines the 36 possible positions for relevance scores, which are
computed for each paragraph and each figure.
Definition 2.7 (In-bounds condition).For a resolution map 𝑅defined on a dimension space (D,≺)
and a partial function𝑐:D⇀N 0, we call the following thein-bounds condition.
Ib(𝑅;𝑐) ⇐⇒ ∀𝑑∈dom(𝑐). 𝑑,𝑐| Dep(𝑑)∈dom(𝑅)∧𝑅 𝑑,𝑐| Dep(𝑑)>𝑐(𝑑)
Note that for Ib(𝑅;𝑐)to hold true, dom(𝑐) must be closed, since it requires 𝑐|Dep(𝑑) be total over
Dep(𝑑) for all𝑑. When dom(𝑐) is a principal dependency space Dep(𝑑′), we refer to each resolu-
tion 𝑑,𝑐| Dep(𝑑),ℓas anancestorof any resolution (𝑑′,𝑐,ℓ′)withℓ′∈N 0, whereas(𝑑′,𝑐,ℓ′)is a
descendant.
As the in-bounds condition provides the means to verify the validity of positions, we may now
define which resolution maps are well-formed.
Definition 2.8 (Shapes).On a dimension space (D,≺) , ashapeis a resolution map 𝑅that satisfies
the following condition.
∀𝑑∈D.∀𝑐∈NDep(𝑑)
0.(𝑑,𝑐)∈dom(𝑅)↔Ib(𝑅;𝑐)
Theonly-ifdirection (𝑑,𝑐) ∈dom(𝑅) →Ib(𝑅 ;𝑐)necessitates that all 𝑐be constrained by
ancestor resolutions. Theifdirection Ib(𝑅;𝑐)→(𝑑,𝑐)∈dom(𝑅) further enforces that there are
no unresolved lengths; that is, if the ancestor resolutions rule that a position 𝑐:Dep(𝑑)→N 0
is in-bounds, then there must be a resolution (𝑑,𝑐,ℓ) . Also note that for primary dimensions 𝑑,
Ib(𝑅;∅)is vacuously true for the only function ∅:∅→N 0, so the definition maintains that there
is precisely one resolution(𝑑,∅,ℓ)for such dimensions.
Coordinates are now naturally defined as maps within the bounds set by a shape.
Definition 2.9 (Coordinates).Given a dimension space (D,≺) , a shape𝑅, and a closed subspace
F⊆D, we have thecoordinate spaceC(D;𝑅;F):
C(D;𝑅;F)={𝑐:F→N 0|Ib(𝑅;𝑐)},
where each𝑐∈C(D;𝑅;F)is acoordinateoverF.
As promised in Section 2.1, we extend this definition to support indexing over convex subspaces.

8 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Definition 2.10 (Subcoordinates).Given a dimension space (D,≺) , a shape𝑅, a convex sub-
spaceE ⊆ D , and a coordinate 𝑐Dep(E)∈ C(D ;𝑅;Dep(E)) , we have thesubcoordinate space
C∗(D;𝑅;E,𝑐 Dep(E)):
C∗(D;𝑅;E,𝑐 Dep(E))=
𝑐|E|𝑐∈C(D;𝑅;E↓)∧𝑐| Dep(E) =𝑐Dep(E)	
.
Each𝑐∗∈C∗(D;𝑅;E,𝑐 Dep(E))is asubcoordinateoverEand𝑐 Dep(E) .
We conclude by stating some properties of coordinates and subcoordinates to demonstrate their
well-behavedness.
Proposition 2.11.Given a dimension space(D,≺)and a shape𝑅, we have the following:
(1) For closed subspacesF′⊆F⊆D,𝑐∈C(D;𝑅;F)=⇒𝑐| F′∈C(D;𝑅;F′).
(2) For a closedF⊆D,C∗(D;𝑅;F,∅)=C(D;𝑅;F).
(3)For a convexE⊆D and a coordinate 𝑐Dep(E)∈C(D ;𝑅;Dep(E)) , there exists arestricted
shape𝑅|(E,𝑐 Dep(E)), a shape on(E,≺| E), such that
C∗(D;𝑅;E,𝑐 Dep(E))=C
E;𝑅|(E,𝑐 Dep(E));E
.
That is, we can interpret each subcoordinate space as a coordinate space when the shape is
appropriately restricted. We have𝑅| (E,∅)⊆𝑅whenEis closed.
2.3 Arrays
We arrive at the final step in formulating the dimension system, which is associating the system
with real-life ragged arrays. Since we have already established the shape of possible coordinates
over dimensions, this process is straightforward.
Definition 2.12 (Arrays).For a dimension space (D,≺) , a shape𝑅, a closed subspace F⊆D , and
a space of values𝑉, anarrayis a function
arr:C(D;𝑅;F)→𝑉.
When this function is not total, we call it apartial array arr:C(D;𝑅;F)⇀𝑉.
Also, when we fix a coordinate over the ancestor dimensions, we get a smaller array over the
descendant dimensions.
Definition 2.13 (Subarrays).For an array arr:C(D ;𝑅;F)→𝑉 , a convex subspace E⊆F such
thatF\Eis closed, and a coordinate𝑐∈C(D;𝑅;F\E), thesubarrayarr[𝑐]is a function
arr[𝑐]:C∗(D;𝑅;E,𝑐| Dep(E))→𝑉
that satisfies
∀𝑐∗∈C∗(D;𝑅;E,𝑐| Dep(E)).arr[𝑐](𝑐∗)=arr(𝑐∗∪𝑐).
Note that the above equation is valid sincedom(𝑐)=F\E⊇Dep(E).
3 Incremental Resolutions
Operon handles data processing tasks and data entries as arrays on a global dimension space
and a shared shape. As such, we may understand Operon as a system that computes for the
completion of all defined arrays arr:C(D,𝑅,F)→𝑉 , where arrconceptually represents either
instances of a processing task (e.g., vlm_evaluate ) or a data collection (e.g., Relevance ). Assuming
that the variables in the signature D,𝑅,F,𝑉 are known upfront, the system becomes a simple
fill-in-the-blanks engine that populates the values for all coordinates inC(D,𝑅,F).

Operon: Incremental Construction of Ragged Data via Named Dimensions 9
However, whileD,F, and𝑉can indeed be determined statically (§4.1), the shape 𝑅does not
follow suit. The system does not have any knowledge of the desired shape until the execution of
user-defined tasks. Instead, specific tasks (e.g., extract_sections orfilter_aggregate ) produce
new resolutions that would ideally accumulate to form a final shape. For the runtime system to
behave predictably, we must be able to express the intermediate states of 𝑅as new resolutions are
added. To this end, we establish which states are acceptable as intermediate resolution maps, provide
a confluent and terminating algorithm that maintains this property, and extend our definitions of
coordinates to handle unknowns.
3.1 Partial Shapes
Recall the definition of shapes in Definition 2.8. For a resolution map to be a shape, coordinates
mentioned in its domain must be in-bounds of itself ( (𝑑,𝑐)∈dom(𝑅)→Ib(𝑅 ;𝑐)), and all in-bounds
coordinates must appear in its domain ( Ib(𝑅;𝑐)→(𝑑,𝑐)∈dom(𝑅) ). While the former connotes
noncontradiction, the latter condition enforces the resolution map to becompletein the sense that
no obvious holes are left unfilled. Relaxing the condition to allow incompleteness gives us a natural
definition forpartial shapes.
Definition 3.1 (Partial shapes).On a dimension space (D,≺) , apartial shapeis a resolution map
𝑅that satisfies the following condition.
∀𝑑∈D.∀𝑐∈[Dep(𝑑)→N 0].(𝑑,𝑐)∈dom( 𝑅)→Ib(𝑅;𝑐)
In particular, if a partial shape 𝑅is not a shape, we call 𝑅anincomplete shape. By contrast, we may
use the termsshapeandcomplete shapeinterchangeably.
Partial shapes behave as valid intermediate states while building towards a complete shape,
starting from the trivial empty map ∅. From a top-down perspective, partial shapes are initial
segments of complete shapes when topologically sorted with respect to the ancestor-descendant
relation of resolutions (as defined in Def. 2.7). Any such topological sorting would therefore list a
sequence of resolutions whose cumulative addition produces a chain of partial shapes, eventually
resulting in the desired complete shape.
However, since we do not have the final shape in advance, we take an approach where we start
from a partial shape (often the empty map), repeatedly produce a resolution that does not contradict
the current partial shape, and extend the partial shape with that resolution. We call such resolutions
compatiblewith the given partial shape.
Definition 3.2 (Compatible resolutions).For a partial shape 𝑅on(D,≺) , if a dimension 𝑑∈D
and a function𝑐∈[Dep(𝑑)→N 0]satisfy
Comp(𝑅;𝑑,𝑐) ⇐⇒Ib( 𝑅;𝑐)∧(𝑑,𝑐)∉dom( 𝑅),
then the pair(𝑑,𝑐) iscompatiblewith 𝑅. Any resolution(𝑑,𝑐,ℓ) withℓ∈N 0is also said to be
compatible with 𝑅.
The following lemma states that a compatible resolution, as defined above, indeed preserves the
partial shape property on extension.
Lemma 3.3.For a partial shape 𝑅and a resolution(𝑑,𝑐,ℓ) , the extension 𝑅{(𝑑,𝑐)↦→ℓ} stays a
partial shape if and only ifComp( 𝑅;𝑑,𝑐).

10 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Algorithm 1:Incremental construction of a shape
Input :A dimension space(D,≺)and a function𝜋:(𝑑,𝑐)↦→ℓ
Output :A complete shape𝑅
1proceduremainbegin
2𝑅←−∅;
3while 𝑅is incompletedo
4𝐶←−
(𝑑,𝑐)|Comp( 𝑅;𝑑,𝑐)	
;// never empty due to Thm. 3.4
5(𝑑,𝑐)←−element in𝐶;
6ℓ←−𝜋(𝑑,𝑐);
7𝑅←−𝑅{(𝑑,𝑐)↦→ℓ};
8return 𝑅;
3.2 Incremental Construction
As briefly mentioned in the previous subsection, we aim to incrementally construct a complete
shape by starting from an initial partial shape ∅and repeatedly adding compatible resolutions. A
simple linear algorithm that performs this task is shown in Alg. 1. This algorithm repeatedly finds
a compatible(𝑑,𝑐) pair, queries an oracle function 𝜋for the desired length ℓat that coordinate,
and extends the current partial shape 𝑅with the new resolution (𝑑,𝑐,ℓ) . We assume the oracle
function𝜋:Ð
𝑑∈D({𝑑}×[Dep(𝑑)→N 0])→N 0is a total and deterministic function for purpose
of this discussion; in practice, a query to 𝜋would represent the execution of a user-defined task
that produces the desired length.
The correctness of Alg. 1 relies on two assumptions: first, that there is always at least one
compatible resolution to add to an incomplete shape, and second, that the process of adding
compatible resolutions eventually leads to a complete shape. We formalize these assumptions in
Thms. 3.4 and 3.5, respectively.
Theorem 3.4 (Progress).A partial shape has a compatible resolution if and only if it is incomplete.
Theorem 3.5 (Termination).There is no infinite sequence of partial shapes where each step adds
a resolution.
We may further extend the above algorithm for parallel execution, as shown in Alg. 2. In
this version, multiple worker threads each own a compatible (𝑑,𝑐) pair to process, which allows
concurrent queries to the oracle function 𝜋. To avoid duplicate work, the main thread keeps track
of the(𝑑,𝑐) pairs that workers are currently processing in a thread-local set 𝑆. If all compatible
pairs are being processed, the main thread waits for any worker to finish and update the shared
shape𝑅before proceeding.
For this parallelization to be correct, each worker’s (𝑑,𝑐) pair must remain compatible with the
shared shape 𝑅regardless of other workers’ actions. Thm. 3.6 ensures this by stating that adding a
compatible resolution does not invalidate other compatible resolutions.
Theorem 3.6 (Local commutativity).If Comp(𝑅;𝑑,𝑐) andComp(𝑅;𝑑′,𝑐′)with(𝑑,𝑐)≠(𝑑′,𝑐′),
thenComp( 𝑅{(𝑑,𝑐)↦→ℓ};𝑑′,𝑐′).
We conclude with the following corollary, which guarantees the consistency of the resulting
complete shape under a fixed oracle function𝜋.
Corollary 3.7 (Determinism).Under a fixed function 𝜋:Ð
𝑑∈D({𝑑}×[Dep(𝑑)→N 0])→N 0,
any fair execution of Alg. 1 or Alg. 2 terminates and returns the same complete shape.

Operon: Incremental Construction of Ragged Data via Named Dimensions 11
Algorithm 2:Parallel incremental construction of a shape
Input :A dimension space(D,≺)and a function𝜋:(𝑑,𝑐)↦→ℓ
Output :A complete shape𝑅
1proceduremainbegin
2𝑅←−∅;// as a shared reference with concurrent appends
3𝑆←−∅;// "seen"(𝑑,𝑐)pairs, thread-local
4repeat
5𝑅𝑠←−snapshot of 𝑅;
6if 𝑅𝑠is completethen return 𝑅𝑠;
7𝐶←−
(𝑑,𝑐)|Comp( 𝑅𝑠;𝑑,𝑐)	
;// never empty due to Thm. 3.4
8if𝐶⊆𝑆then// all compatible pairs are being processed
9wait until 𝑅≠𝑅𝑠;
10continue;
11else
12(𝑑,𝑐)←−element in𝐶\𝑆;
13𝑆←−𝑆∪ {(𝑑,𝑐)};
14spawnworker( 𝑅,𝑑,𝑐);
15procedureworker( 𝑅,𝑑,𝑐)begin
16ℓ←−𝜋(𝑑,𝑐);
17𝑅←−𝑅{(𝑑,𝑐)↦→ℓ};// atomic append to shared reference
18return;
3.3 Coordinates with Unknowns
Incomplete shapes naturally lead to the question of how to define coordinates over them. Operon
might find some data entries before the shape for that data is fully known, and those entries must
still be addressable. For example, the relevance score at {p↦→0,s↦→0,g↦→0,f↦→0 }in Figure 3
could be computed before the resolution (g,{p↦→0,s↦→4},3)becomes known. Since the array of
relevance scores is defined over F={p,s,g,f}, the coordinate space C(D ;𝑅;F)cannot be defined
as in Def. 2.9 without the above resolution. We wish to extend the definition of coordinates to allow
suchunknownvalues while keeping fully-resolved coordinates accessible.
The following definition achieves this by permitting coordinates to be partial functions over
F. The coordinate must be in-bounds for its domain, but when a resolution is missing for some
dimension𝑑, the coordinate must also omit𝑑from its domain.
Definition 3.8 (Coordinates with unknowns).For a partial shape 𝑅on(D,≺) and a closedF⊆D ,
acoordinateoverFis a partial function𝑐:F⇀N 0that satisfies the following.
(1)∀𝑑∈dom(𝑐). 𝑑,𝑐| Dep(𝑑)∈dom(𝑅)∧𝑅 𝑑,𝑐| Dep(𝑑)>𝑐(𝑑).
(2)∀𝑑∈F\dom(𝑐). 𝑑,𝑐| Dep(𝑑)∉dom(𝑅).
Thecoordinate spaceC(D; 𝑅;F)denotes the set of such coordinates. Note that (1) isIb( 𝑅;𝑐).
This extension aligns well with the original definition, as the definition without unknowns
becomes a special case where 𝑅is complete. We therefore characterize subcoordinates, arrays, and
subarrays without change in their definitions, except that the shapes in those definitions may now
be partial.

12 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Proposition 3.9.Def. 3.8 is a strict extension of the original definition of coordinates in Def. 2.9.
That is,𝐶(D;𝑅;F)is unchanged under either definition when𝑅is a complete shape.
The following theorem portrays how the coordinate space changes as we add new compatible
resolutions to a partial shape. Once again, this characterization aligns well with our intuition:
adding a new resolution for a dimension 𝑑effectivelyexplodesthe coordinate space along that
dimension, producing ℓnew options for each existing coordinate that match the ancestor positions.
Computing the coordinate space incrementally in this manner allows Operon to avoid recomputing
the entire space from scratch after each resolution addition.
Theorem 3.10 (Coordinate explosion).For a partial shape 𝑅on(D,≺) , a closedF ⊆D , a
dimension𝑑∈F, and a coordinate𝑐: Dep(𝑑)→N 0withComp( 𝑅;𝑑,𝑐),
(1)∀𝑐′∈C(D;𝑅;F).𝑐′|Dep(𝑑) =𝑐=⇒𝑑∉dom(𝑐′);
(2) writing𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ},
C(D;𝑅ℓ;F)=
C(D;𝑅;F)\𝑈
∪ℓ−1Ø
𝑖=0{𝑐′{𝑑↦→𝑖}|𝑐′∈𝑈}
where𝑈=
𝑐′∈C(D;𝑅;F)|𝑐′|Dep(𝑑) =𝑐	
.
Let us revisit the behavior of arrays defined over coordinates with unknowns. For a complete
shape𝑅, an intermediate state in computing an array arr:C(D ;𝑅;F)→𝑉 may be expressed as a
partial array arr:C(D ;𝑅;F)⇀𝑉 , where the shape is fully known but some entries are yet to be
computed. When the shape is still incomplete with 𝑅⊊𝑅 , the array may be expressed as a partial
array arr′:C(D ;𝑅;F)⇀𝑉 with dom( arr′)⊆[F→N 0](i.e., only defined for coordinates total
overF). Even if a new compatible resolution is added to 𝑅, thefilled-incoordinates in dom( arr′)
are not affected by the explosion due to Thm. 3.10(1).
In other cases (such asticketswhich we will discuss in Section 4.2), we may define the array to
span the entire coordinate space C(D ;𝑅;F)even when𝑅is incomplete. This definition allows the
array to hold entries that are not fully resolved yet, which Operon utilizes to track dependency
counts lazily.
4 Operon
In this section, we present the system design for Operon.
4.1 Overview
As mentioned earlier, Operon takes advantage of Rust’s procedural macro feature to accept pipeline
definitions in a concise DSL. During macro expansion, it inspects the declared pipeline and generates
the code necessary for execution. Figure 4 describes the syntax of our DSL.
The pipeline consists of one or more tasks that collectively define the overarching data flow. In a
static analysis as shown in Figure 5, we check whether the pipeline 𝑝is well-formed according
to these rules, i.e., whether (∅,∅,∅)|𝑝⊢(D,≺,Σ) holds for some(D,≺,Σ) . For a well-formed
pipeline, the triple (D,≺,Σ) from this analysis gains meaning as the global dimension space,
the dependency relation, and the map from entity type to itscharacteristicdimension subspace,
respectively; the meaning ofΣwill be elaborated shortly.
The checking rules ensure that the inferred (D,≺,Σ) satisfies several well-formedness properties,
as stated in the following lemma.
Lemma 4.1.Given(∅,∅,∅)|𝑝⊢(D,≺,Σ),
(1) the relation≺is a strict partial order overD;

Operon: Incremental Construction of Ragged Data via Named Dimensions 13
Pipeline𝑝::=®𝑡
Task𝑡::=
𝑓,𝑠out,−−→𝑠in,𝑖,F,𝑛
unique𝜏 in,𝑖
Entity signature𝑠::= ⟨𝜏,E⟩
Entity signature mapΣ::=®𝑠unique𝜏
Dim. spaceD,E,F::= ®𝑑
Concurrency𝑛∈Z+Entity type𝜏∈Type vars Dimension𝑑,𝑒∈Idents
Function𝑓:Ö
𝑖
list ... list|              {z              }
|Ein,𝑖|𝜏in,𝑖
→list ... list|              {z              }
|Eout|𝜏out Dep. rel.≺⊆D×D
Fig. 4. Syntax of the Operon domain-specific language.
(∅,∅,∅)|()⊢(∅,∅,∅)(Unit)
(∅,∅,∅)|®𝑡⊢(D 1,≺1,Σ1) (D 1,≺1,Σ1)|𝑡′⊢(D 2,≺2,Σ2)
(∅,∅,∅)|®𝑡::𝑡′⊢(D 2,≺2,Σ2)(Chain)
𝜏out∉dom(Σ) E out∩D=∅ |E out|≤1
∀𝑖.𝜏 in,𝑖∈dom(Σ). 
Ein,𝑖⊆Σ(𝜏 in,𝑖)
Σ(𝜏 in,𝑖)\E in,𝑖⊆F
Σ(𝜏 in,𝑖)\E in,𝑖closed under≺
F⊆Ð
𝑖Σ(𝜏 in,𝑖) Fclosed under≺
(D,≺,Σ)|
𝑓,⟨𝜏out,Eout⟩,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
⊢

D⊔E out,≺⊔F×E out,Σ{𝜏out↦→F⊔E out}(TaskDef)
Fig. 5. Static checking rules for the DSL.
(2) for all entity types𝜏∈dom(Σ), the characteristic dimension spaceΣ(𝜏)is closed under≺;
(3)for all tasks 𝑡=
𝑓,𝑠out,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
in𝑝, the dimension spaces FandΣ(𝜏 in,𝑖)\E in,𝑖
are closed under≺.
Once the checks are complete,entitiescan be defined based on the inferred information.
Definition 4.2 (Entities).Given (∅,∅,∅)|𝑝⊢(D,≺,Σ) , for all entity types 𝜏∈dom(Σ) , anentity
array𝐸(𝜏)is defined as a partial array
𝐸(𝜏):C(D; 𝑅;Σ(𝜏))⇀𝜏; dom( 𝐸(𝜏))⊆[Σ(𝜏)→N 0]
for some partial shape 𝑅. Elements of this array are calledentities.
Entities are the data units that Operon aims to produce and process. For each entity type 𝜏
mentioned in the pipeline definition, Σ(𝜏) characterizes the dimension subspace that entities of
type𝜏are indexed over. The problem situation of Operon now becomes clearer: given a pipeline

14 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
definition𝑝, run the pipeline to incrementally construct a partial shape 𝑅and fully populate the
entity arrays 𝐸(𝜏)for all𝜏∈dom(Σ).
Therefore, at its core, Operon is a state machine that continuously transforms the state (𝑅,𝐸)by
executing user-defined functions specified in the pipeline. Starting from the trivial state (∅,𝜆𝜏.∅) ,
Operon undergoes the following state transitions, known asjobs, until it reaches a terminal state
where𝑅is complete and all entity arrays are total.
Definition 4.3 (Jobs).Consider a task
𝑓,⟨𝜏out,Eout⟩,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
in a pipeline 𝑝, the in-
ferred(D,≺,Σ) , and a current state (𝑅,𝐸). For a total coordinate 𝑐∈C(D ;𝑅;F)∩[F→N 0], if
the subarrays 𝐸(𝜏 in,𝑖)[𝑐|Σ(𝜏 in,𝑖)\E in,𝑖]are all total over their respective domains, the function 𝑓can
be invoked with these subarrays as inputs. We refer to this call as ajobat coordinate 𝑐, denoted
as𝑗𝑡(𝑐). A job𝑗𝑡(𝑐)transforms the current state (𝑅,𝐸)into a new state(𝑅′,𝐸′)as follows. This
transition is exactly once valid for each𝑡and𝑐.
•If|Eout|=0, then𝑅′=𝑅. Otherwise, writeEout={𝑒}and let𝑙be the length of the output
array returned by𝑓. Then, 𝑅′=𝑅{(𝑒,𝑐)↦→𝑙}.
•𝐸′is identical to 𝐸except for the subarray 𝐸′(𝜏out)[𝑐], which is assigned the output array.
This definition is only possible under the constraints imposed by the static checking rules and
Lemma 4.1. First of all, the notation C(D ;𝑅;F)assumes thatFis closed under≺. Similarly, the
subarray𝐸(𝜏 in,𝑖)[𝑐|Σ(𝜏 in,𝑖)\E in,𝑖]is well-defined only when Σ(𝜏 in,𝑖)\E in,𝑖is a subset ofFand is
closed under≺. We may also note that the assignment 𝑅′=𝑅{(𝑒,𝑐)↦→𝑙}is only valid because
Eoutis disjoint with previously defined dimensions. Therefore, the coordinate (𝑒,𝑐) is only seen
once across(𝑡,𝑐)pairs during the pipeline execution.
The batch assignment 𝐸′(𝜏out)[𝑐]=𝑓(...) is valid because of the following reason. The subarray
𝐸(𝜏 out)[𝑐] :C∗(D;𝑅;Eout,𝑐|Dep(E out))⇀𝜏 outis an empty function prior to the job with the sub-
coordinate space (with unknowns) C∗(D;𝑅;Eout,𝑐|Dep(E out))={∅}. After the job and the partial
shape update, the subcoordinate space appropriately explodes to fit the output array 𝑓(...) . The
assignment 𝐸′(𝜏out)[𝑐]=𝑓(...) while keeping all other entities unchanged is therefore valid and
completes the state transition.
The normalizing constraint |Eout|≤1inTaskDef was chosen to simplify the usage of Operon.
While it is possible with minimal changes in Definition 4.3 to allow multiple output dimensions,
doing so would require knowledge of dimensional dependencies within the output dimension set
Eout. Since no universally used array data structure supports our formulation of ragged arrays,
the burden of providing and enforcing the dependency information would be left to the user. By
restrictingEoutto at most one dimension, the 0- or 1-dimensional output array trivially translates
to the corresponding ragged subarray, letting us avoid this complexity. While this restriction may
seem limiting, it is possible to work around it by splitting a desired multi-dimensional output
into multiple tasks that each produce a single dimension, albeit with some loss of usability or
performance.
4.2 Implementation
The primary goal of Operon is to launch each job as soon as it becomes executable. Jobs become
ready to run when (1) their coordinates have fully resolved over Fand (2) all their input entities
have been computed. Since both resolutions and input entities are produced by some other jobs in
the pipeline, the readiness of a job relies on the status of others.

Operon: Incremental Construction of Ragged Data via Named Dimensions 15
Operon manages this by usingtickets, which are lightweight objects that represent the state of
each job in the system.
Definition 4.4 (Tickets).Consider a task 𝑡=
𝑓,𝑠out,−−→𝑠in,𝑖,F,𝑛
in a pipeline(∅,∅,∅)|𝑝⊢(D,≺
,Σ). Theticket arrayfor task𝑡is atotalarray
𝑗𝑡:C(D;𝑅;F)→N 0×N 0×{Waiting,Queued,Done},
where each entry 𝑗𝑡(𝑐)=(count,quota,status)is called aticket.
A ticket𝑗𝑡(𝑐)conceptually corresponds to a job 𝑗𝑡(𝑐), even though the latter is not defined
unless the coordinate 𝑐is total overF. For a partially defined coordinate 𝑐, the ticket represents
all potential jobs that could arise from 𝑐as more resolutions are added to 𝑅. In the initial state
(𝑅,𝐸)=(∅,𝜆𝜏.∅) , the coordinate space with unknowns C(D ;𝑅;F)starts as{∅}, and hence there
exists a single ticket 𝑗𝑡(∅)representing all jobs of task 𝑡. As resolutions for dimensions in Fare
added to𝑅, the coordinate space expands according to Theorem 3.10 and tickets are duplicated along
the newly resolved dimensions. When a ticket’s coordinate becomes total over F, it corresponds to
exactly one job, and we say it isfully resolved. If the task has no input entities, as in the first task in
all pipelines, its ticket is fully resolved from the start and can have the corresponding job launched
immediately; otherwise, the ticket must wait for upstream jobs to resolve its coordinates.
Throughout the ticket’s lifetime, its count andquota fields are updated to track the number of
completed dependencies and the total number of dependencies, respectively. While the dependencies
are most intuitively explained as the completion of all input entities, i.e., 𝐸(𝜏 in,𝑖)[𝑐|Σ(𝜏𝑖𝑛,𝑖)\E in,𝑖]
being total for all 𝑖, we can track back to the tasks that produce these entities and count their tickets
as dependencies instead. Expressing 𝑡𝑖=
_,
𝜏in,𝑖,E𝑖
,_,F𝑖,_
as the task that produces the input
entity𝜏 in,𝑖, the dependencies of ticket 𝑗𝑡(𝑐)can be defined as the set of tickets
Ä
𝑖𝑗𝑡𝑖
𝑐|F𝑖\Ein,𝑖
.
The size of this set,∑︁
𝑖C∗(D;𝑅;F𝑖∩E in,𝑖,𝑐|F𝑖\Ein,𝑖),
determines the quota of the ticket, while the count is incremented each time one of these dependent
tickets reaches the Done status. Once the two counts are equal, it can be assumed that the ticket
is fully resolved (Lemma 4.5), and the ticket becomes ready for execution ( Queued ). The ticket is
further updated toDonewhen the corresponding job completes.
Lemma 4.5.When a ticket’scountequals itsquota, the ticket is fully resolved.
A dedicated scheduler for each task manages these tickets, as illustrated in Figure 6. These
schedulers operate independently and communicate solely through peer events:
•Job completion event: A job of its type has completed, and schedulers of downstream tasks
should increment thecountof applicable tickets.
•Resolution event: A new resolution for a dimension in Eouthas been produced, and down-
stream ticket arrays whoseFcontains that dimension should explode.
•Ticket explosion event: A ticket array has expanded due to a resolution event, and downstream
tickets depending on the exploded tickets should update theirquotas accordingly.
Each scheduler runs a main event loop that governs the execution of its task’s jobs:
•Peer event handling: Perform necessary updates to the tickets of its task according to the
incoming peer event. Enqueue newly ready tickets for execution.

16 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Initialize scheduler
Main event loop
event kind?
peer event type?
resolve blank
coordinates
propagate
explosion events
enqueue newly
ready ticketsupdate dependency
countershandle coordinate
explosionpropagate
job/resolution events
no pending tickets
and events?
returnOk(())spawn worker task
perform storage ops execute job emit internal eventWorker Taskinternal event
peer eventworker_permit∧∃job∈queue
YesNojob finished
resolution foundpredecessor exploded
Fig. 6. Idealized flowchart of an individual scheduler.
•Job execution: Spawn concurrent workers, up to a set concurrency limit 𝑛, to execute the
jobs corresponding to the queued tickets.
•Internal event handling: Process signals from its workers, collecting results from the com-
pleted jobs, and emitting peer events as necessary.
The loop continues until there are no pending tickets or events, at which point the scheduler returns
Ok(()) to signal completion of its task. The theorems and corollary in Section 3.2 guarantee the
well-behavedness of the progress of the partial shape 𝑅.
The entities 𝐸, resolutions 𝑅, and tickets 𝑗𝑡are all stored in an underlying storage system,
currently implemented in PostgreSQL. The persistence allows straightforward pause-resume and
crash-recovery functionality, as the entire state of the schedulers can be reconstructed from the
database.
5 Evaluation
In this section, we evaluate the performance of Operon with a comparative analysis against
Prefect [ 29] and discuss its limitations.3Prefect was chosen as the baseline for a few reasons.
First, Prefect is most similar to Operon design-wise, as both frameworks are built around the
asynchronous execution of a workflow composed of user-defined tasks. Second, Prefect is relatively
lightweight compared to other workflow orchestration frameworks and allows fine-grained control
over the execution environment [ 21]. We found that other widely used frameworks, such as Apache
3Raw data from the experiments are provided in Appendix C.

Operon: Incremental Construction of Ragged Data via Named Dimensions 17
20 40 60 80 100
N025050075010001250Execution Time (s)
Performance Comparison (tsleep=3)
Theory
Operon
Prefect
(a)
0 1 2 3 4 5
tsleep0100200300Execution Time (s)
Performance Comparison (N=20)
Theory
Operon
Prefect (b)
Fig. 7. Total execution times of Operon (red) and Prefect (blue) by number of PaperId s𝑁and sleep time
𝑡sleep. (a) Measured execution time about 𝑁with𝑡sleep=3s. (b) Measured execution time about 𝑡sleepwith
𝑁= 20. In both graphs, the reference line “Theory” (gray) indicates the theoretical minimum execution time,
given asl
# of vlm_evaluate
64m
+1
×𝑡sleep.
Airflow [ 3] and Luigi [ 34], impose more structural constraints on the workflow definition and
execution, making them less suitable for a direct comparison with Operon.
5.1 Performance Analysis
For a quantitative comparison of workflow processing performance, we measured the total exe-
cution time of the same workflow under various settings. The workflow used in the experiment
was based on the example presented in Section 1.1. However, to establish a consistent experimen-
tal environment, all tasks were implemented as mock tasks with negligible computation while
maintaining all dimensional structures. Additionally, as the parse_paper ,vlm_evaluate , and
ocr_extract tasks would require relatively long execution times in a real environment due to
the use of third-party programs or ML models, they were classified as heavy tasks and assigned
additional sleep intervals.
To control the influence of hardware resources, the worker pool was limited to 64 for heavy
tasks and 1 for other general tasks, with a total thread count capped at 4,000. Both systems were
configured to use a local PostgreSQL server as the storage. The size of each dimension was randomly
generated but pre-defined and fixed for consistency across experiments. All experiments were
conducted in a controlled environment in a single device (Mac mini, M4, 16GB RAM).
We chose two variables for the experiments: the number of PaperId s to process ( 𝑁) and the
sleep interval for heavy tasks ( 𝑡sleep). To independently analyze the impact of each variable, we
measured execution times by varying one variable while keeping the other fixed, conducting three
trials for each setting.
The experimental results in Figure 7 show that Operon consistently outperforms Prefect in terms
of execution time, remaining close to the theoretical minimum across various configurations. The
vertical intercept of Figure 7(b), where 𝑡sleep=0, signifies the baseline scheduling overhead of
each system with near-zero task execution time. We observe that Operon completes the workflow
14.94 times faster than Prefect in this configuration. As 𝑡sleepincreases, the scheduling overhead
becomes amortized over the longer task execution times, which is reflected in the narrowing
performance gap between the two systems. When 𝑁increases (Figure 7(a)), the execution times of
both systems behave roughly proportional to the number of total tasks, as to be expected from a flat

18 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
0 200 400 600 800 1000 1200
Elapsed time (s)02004006008001000# of rowsRow generation time comparison (N=100, tsleep=3)
Operon
Prefect
Fig. 8. Generated rows over time in the experiment with𝑁=100and𝑡 sleep=3s.
increase in quantity. The gap between the two systems therefore widens as 𝑁increases, signaling
an accumulating advantage for Operon in larger-scale workflows.
The following structural factors can explain the overall performance difference between the two
systems.
(1)Implementation language. Prefect is implemented in Python, while Operon is implemented
in Rust, which inevitably leads to performance differences. Python’s Global Interpreter Lock
(GIL) acts as an inherent constraint in multithreaded environments [2].
(2)State persisting method. Operon only stores minimal data, such as outputs, indices across
dimensions, and timestamps, whereas Prefect additionally stores various metadata for
tracking the workflow.
(3)Scheduling architecture. Prefect employs a centralized server architecture to manage the
entire workflow, which incurs network communication overhead. We minimized the latency
by using localhost, but there is still additional overhead compared to Operon, which operates
as a standalone multithreaded process apart from the database.
Total execution time is not the only metric for evaluating performance. In large-scale data
generation tasks for ML, which Operon targets as a primary use case, the time to the 𝑛th result
also holds significant practical value. Quicker generation of partial results opens the door to
early commencement of model training, which enables parallelizing tasks after the data generation
stage [ 17]. Additionally, early availability of intermediate results allows for rapid error identification
and debugging. We plotted the number of end-to-end results over time as an additional performance
metric from this perspective.
Results in Figure 8 show that Operon holds a clear advantage in this regard as well. Operon
generates rows uniformly throughout the execution time, demonstrating strong parallelism across
tasks, whereas Prefect exhibits a pattern where generation stagnates in the early stages and then
surges sharply towards the end of the workflow.
The difference stems from the task management mechanisms of the two systems. Operon
employs a work-stealing scheduler that efficiently distributes currently executable tasks, managing
thousands of lightweight tasks concurrently through a limited number of OS thread pools. This, in
tandem with the per-task multi-scheduler design, allows for balanced scheduling even in scenarios
where heavy and light tasks are mixed [ 5]. As tailing tasks do not starve, rows are generated at a
consistent rate throughout the execution.
In contrast, Prefect’s ThreadPoolTaskRunner uses a fixed-size thread pool, adding tasks to the
thread pool’s queue for sequential processing upon creation. As heavy tasks (such as vlm_evaluate

Operon: Incremental Construction of Ragged Data via Named Dimensions 19
in this workflow) clog the queue, lighter tasks that tail behind them (such as collect_row ) are
forced to wait, hence the observed stagnation in the end-to-end generation rate.
5.2 Limitations
Database overhead.A critical limitation of Operon is the overhead introduced by database
operations, as well as the practical requirement of maintaining a running PostgreSQL instance.
Operon keeps the runtime state—the shape, entities, and tickets of all tasks—in a persistent storage,
and each scheduler event opens a database transaction to update and pull necessary information.
The drawback in performance ties to some design choices regarding reliability and suitability
for target use cases. As mentioned in Section 4.2, the persistent storage allows Operon to provide
strong data-consistency and recovery options. The runtime state model of Operon is designed to
guarantee reachability from the current state to the final completed state, allowing it to reference
the store to recover the exact execution point and continue the remaining work, regardless of
when the workflow was paused. Persisting the state also prevents memory overflow in large-scale
workflows, which may occur if all metadata were kept in memory. These features are integral to
Operon’s target use cases, which focus on CPU-based data-parallel processing [ 10], rather than
extremely low-latency GPU-based workloads.
Structural constraints.Operon supports only DAG-structured workflows, making it impossible to
express workflows requiring cyclic structures directly. Cases where the number of cycles is statically
determined can be rewritten into a DAG through loop unrolling; however, when the number of
cycles is dynamically determined at runtime, it cannot be expressed in Operon’s declarative model.
Imperative frameworks like Prefect would be more suitable for such scenarios.
6 Related Work
Named dimensions and ragged tensors.The demands for named dimensions manifest in various
practical packages such as xarray [ 14], TensorFlow named tensors [ 1], Dex [ 24], einops [ 32],
and Awkward [ 26]. The shared goal of these packages is to describe machine learning models
or operations accurately. Operations between multidimensional tensors are prevalent in modern
deep-learning workloads [ 7], calling for the need for named dimensions to avoid ambiguity and
errors.
On the other hand, ragged tensors are motivated by real-world problems involving variable-
length data. Typical implementations of ragged tensors are based on padding into rectangular
tensors [ 11] or a pointer-based layout such as Iliffe vectors [ 16]. More recently, TensorFlow ragged
tensor [ 1], AccelerateHS [ 9], and Awkward [ 26] support ragged tensors natively. Awkward provides
a design for ragged data over a totally ordered set of named axes, which focuses primarily on
the memory layout and low-level representation. In contrast, Operon emphasizes the abstract
formulation of named dimensions and ragged data, along with the integration into a workflow
orchestration framework.
Workflow engines.We summarize how Operon compares with several widely used workflow
engines in Table 1. The criteria reflect core aspects of Operon’s design: tasks with runtime-known
cardinality, data-centric structure (tasks as data-spawning procedures), type enforcement, first-class
ragged semantics, and native named dimensions.
Among existing systems, Prefect [ 29] aligns most closely with Operon. It supports dynamic work-
flows, a data-oriented model, and partially typesafe configuration, making it the most comparable
platform and the basis for our evaluation. Apache Airflow [ 3] primarily targets DAG scheduling
and monitoring. While it offers limited dynamic expansion through mapped tasks, the mechanism
is restricted and does not generalize to multidimensional and data-driven patterns. Luigi [ 34]

20 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Table 1. Comparison table of Operon against widely used workflow engines (∗: Partially supported)
Operon Prefect[29] Apache Airflow[3] Luigi[34]
General scheduling ✓ ✓ ✓ ✓
Runtime-discovered tasks ✓ ✓ ✓∗×
Data-centric structure ✓ ✓× ×
Type enforcement ✓ ✓∗× ×
First-class ragged semantics ✓× × ×
Native named dimensions ✓× × ×
focuses on batch-oriented pipelines with statically defined task graphs, but lacks dynamic workflow
generation, data-centric abstractions, and type guarantees.
DAG-based agentic frameworks.Agentic frameworks are actively studied in response to the rapid
evolution of large language models (LLMs). In particular, a vast number of agentic frameworks take
the form of DAGs [ 21–23,38], which brings interest to the investigation of their underlying structure.
Most notably in recent studies, AFlow [ 40] examines the iterative refinement of workflows through
feedback on the code structure, and MacNet [ 30] demonstrates optimal DAG structures through
empirical evaluations. These LLM-driven systems often exhibit unpredictability in data structure
and size, as well as high error rates and long execution times. Operon’s targeted design addresses
these difficulties and contributes a structured approach to building robust agentic frameworks.
7 Conclusion
In this paper, we have presented Operon, a dynamic workflow engine designed to declare and
execute acyclic ragged data pipelines with minimal overhead. Our novel theoretical framework
using named dimensions is core to the design of Operon, as it allows for precise tracking of data
shapes and dependencies throughout the pipeline execution. Its declarative DSL separates control
flow from data processing logic with a static check for well-formedness of the iterative structure.
In practice, we have demonstrated that Operon’s parallelism across tasks leads to a near-linear end-
to-end output rate even with discrepancies in task durations. The explicit modeling and persistence
of intermediate data states trivialize robust fault tolerance and recovery mechanisms. As such,
Operon sets a strong foundation for expressing and processing ragged data at scale.
An interesting future direction for this work would be designing a type system to represent
ragged arrays with our dimension system, along with a corresponding data structure. Currently,
Operon only incorporates elementary array operations, such as aggregation, slicing, and Cartesian
products; establishing the algebra over ragged arrays and shapes would facilitate the handling of
more complex operations.
References
[1]Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy
Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael
Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar,
Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg,
Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015.TensorFlow, Large-scale machine learning on heterogeneous systems.
doi:10.5281/zenodo.4724125
[2]Anton Malakhov. 2016. Composable Multi-Threading for Python Libraries. InProceedings of the 15th Python in Science
Conference, Sebastian Benthall and Scott Rostrup (Eds.). 15 – 19. doi:10.25080/Majora-629e541a-002

Operon: Incremental Construction of Ragged Data via Named Dimensions 21
[3]Apache Software Foundation. 2025.Apache Airflow: A platform to programmatically author, schedule and monitor
workflows. https://github.com/apache/airflow
[4]Timothy G. Armstrong, Justin M. Wozniak, Michael Wilde, and Ian T. Foster. 2014. Compiler Techniques for Massively
Scalable Implicit Task Parallelism. InSC ’14: Proceedings of the International Conference for High Performance Computing,
Networking, Storage and Analysis. 299–310. doi:10.1109/SC.2014.30
[5]Robert D. Blumofe and Charles E. Leiserson. 1999. Scheduling multithreaded computations by work stealing.J. ACM
46, 5 (Sept. 1999), 720–748. doi:10.1145/324133.324234
[6]Craig Chambers, Ashish Raniwala, Frances Perry, Stephen Adams, Robert R. Henry, Robert Bradshaw, and Nathan
Weizenbaum. 2010. FlumeJava: easy, efficient data-parallel pipelines. InProceedings of the 31st ACM SIGPLAN Conference
on Programming Language Design and Implementation(Toronto, Ontario, Canada)(PLDI ’10). Association for Computing
Machinery, New York, NY, USA, 363–375. doi:10.1145/1806596.1806638
[7]David Chiang, Alexander M Rush, and Boaz Barak. 2021. Named tensor notation.arXiv preprint arXiv:2102.13196
(2021).
[8]Christopher Clark and Santosh Divvala. 2016. Pdffigures 2.0: Mining figures from research papers. InProceedings of
the 16th ACM/IEEE-CS on Joint Conference on Digital Libraries. 143–152.
[9]Robert Clifton-Everest, Trevor L. McDonell, Manuel M. T. Chakravarty, and Gabriele Keller. 2017. Streaming Irregular
Arrays. InHaskell ’17: The 10th ACM SIGPLAN Symposium on Haskell. ACM, 174–185.
[10] Gianpaolo Cugola and Alessandro Margara. 2012. Low latency complex event processing on parallel hardware.J.
Parallel and Distrib. Comput.72, 2 (2012), 205–218. doi:10.1016/j.jpdc.2011.11.002
[11] Pratik Fegade, Tianqi Chen, Phillip Gibbons, and Todd Mowry. 2022. The CoRa Tensor Compiler: Compilation for
Ragged Tensors with Minimal Padding. InProceedings of Machine Learning and Systems, D. Marculescu, Y. Chi, and C. Wu
(Eds.), Vol. 4. 721–747. https://proceedings.mlsys.org/paper_files/paper/2022/file/afe8a4577080504b8bec07bbe4b2b9cc-
Paper.pdf
[12] Guangshuai Gao, Junyu Gao, Qingjie Liu, Qi Wang, and Yunhong Wang. 2025. A survey of deep learning methods for
density estimation and crowd counting.Vicinagearth2, 1 (Feb. 2025), 2. doi:10.1007/s44336-024-00011-8
[13] Troels Henriksen and Martin Elsman. 2021. Towards size-dependent types for array programming. InProceedings of
the 7th ACM SIGPLAN International Workshop on Libraries, Languages and Compilers for Array Programming(Virtual,
Canada)(ARRAY 2021). Association for Computing Machinery, New York, NY, USA, 1–14. doi:10.1145/3460944.3464310
[14] S. Hoyer and J. Hamman. 2017. xarray: N-D labeled arrays and datasets in Python.Journal of Open Research Software
5, 1 (2017). doi:10.5334/jors.148
[15] Ting-Yao Hsu, C Lee Giles, and Ting-Hao Huang. 2021. SciCap: Generating Captions for Scientific Figures. InFindings
of the Association for Computational Linguistics: EMNLP 2021, Marie-Francine Moens, Xuanjing Huang, Lucia Specia,
and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Punta Cana, Dominican Republic, 3258–3264.
doi:10.18653/v1/2021.findings-emnlp.277
[16] J.K. Iliffe. 1961. The use of the genie system in numerical calculation.Annual Review in Automatic Programming2
(1961), 1–28. doi:10.1016/S0066-4138(61)80002-5
[17] Hannah Kim, Jaegul Choo, Changhyun Lee, Hanseung Lee, Chandan Reddy, and Haesun Park. 2017. PIVE: Per-Iteration
Visualization Environment for Real-Time Interactions with Dimension Reduction and Clustering.Proceedings of the
AAAI Conference on Artificial Intelligence31, 1 (Feb. 2017). doi:10.1609/aaai.v31i1.10628
[18] Mario Michael Krell, Matej Kosec, Sergio P. Perez, and Andrew W Fitzgibbon. 2023. Efficient Sequence Packing without
Cross-contamination: Accelerating Large Language Models without Impacting Performance. https://openreview.net/
forum?id=ZAzSf9pzCm
[19] Yiming Li, Yi Wang, Wenqian Wang, Dan Lin, Bingbing Li, and Kim-Hui Yap. 2025. Open World Object Detection: A
Survey.IEEE Trans. Cir. and Sys. for Video Technol.35, 2 (Feb. 2025), 988–1008. doi:10.1109/TCSVT.2024.3480691
[20] Dragos A. Manolescu. 2002. Workflow enactment with continuation and future objects. InProceedings of the 17th ACM
SIGPLAN Conference on Object-Oriented Programming, Systems, Languages, and Applications(Seattle, Washington,
USA)(OOPSLA ’02). Association for Computing Machinery, New York, NY, USA, 40–51. doi:10.1145/582419.582425
[21] Charlie Masters, Advaith Vellanki, Jiangbo Shangguan, Bart Kultys, Jonathan Gilmore, Alastair Moore, and Ste-
fano V. Albrecht. 2025. Orchestrating Human-AI Teams: The Manager Agent as a Unifying Research Challenge.
arXiv:2510.02557 [cs.AI] https://arxiv.org/abs/2510.02557
[22] Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. 2025. Flow: Modularized Agentic
Workflow Automation. InThe Thirteenth International Conference on Learning Representations. https://openreview.
net/forum?id=sLKDbuyq99
[23] Chiwan Park, Wonjun Jang, Daeryong Kim, Aelim Ahn, Kichang Yang, Woosung Hwang, Jihyeon Roh, Hyerin
Park, Hyosun Wang, Min Seok Kim, and Jihoon Kang. 2025. A Practical Approach for Building Production-Grade
Conversational Agents with Workflow Graphs. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 6: Industry Track), Georg Rehm and Yunyao Li (Eds.). Association for Computational

22 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Linguistics, Vienna, Austria, 1508–1519. doi:10.18653/v1/2025.acl-industry.107
[24] Adam Paszke, Daniel D. Johnson, David Duvenaud, Dimitrios Vytiniotis, Alexey Radul, Matthew J. Johnson, Jonathan
Ragan-Kelley, and Dougal Maclaurin. 2021. Getting to the point: index sets and parallelism-preserving autodiff for
pointful array programming.Proc. ACM Program. Lang.5, ICFP, Article 88 (Aug. 2021), 29 pages. doi:10.1145/3473593
[25] Grzegorz Piotrowski, Mateusz Bystroński, Mikołaj Hołysz, Jakub Binkowski, Grzegorz Chodak, and Tomasz Jan
Kajdanowicz. 2025. When Will the Tokens End? Graph-Based Forecasting for LLMs Output Length. InProceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop),
Jin Zhao, Mingyang Wang, and Zhu Liu (Eds.). Association for Computational Linguistics, Vienna, Austria, 843–848.
doi:10.18653/v1/2025.acl-srw.61
[26] Jim Pivarski, Ianna Osborne, Ioana Ifrim, Henry Schreiner, Angus Hollands, Anish Biswas, Pratyush Das, Santam
Roy Choudhury, Nicholas Smith, and Manasvi Goyal. 2018.Awkward Array. doi:10.5281/zenodo.4341376
[27] Pivarski, Jim, Osborne, Ianna, Das, Pratyush, Lange, David, and Elmer, Peter. 2021. AwkwardForth: accelerating Uproot
with an internal DSL.EPJ Web Conf.251 (2021), 03002. doi:10.1051/epjconf/202125103002
[28] Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar,
and Oncel Tuzel. 2025. Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum.
arXiv:2405.13226 [cs.CL] https://arxiv.org/abs/2405.13226
[29] PrefectHQ. 2025.Prefect: A workflow orchestration framework for building resilient data pipelines in Python. https:
//github.com/PrefectHQ/prefect
[30] Chen Qian, Zihao Xie, YiFei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng
Yang, Zhiyuan Liu, and Maosong Sun. 2025. Scaling Large Language Model-based Multi-Agent Collaboration. InThe
Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=K3n5jPkrU6
[31] I. Riakiotakis and P. Tsanakas. 2005. Dynamic scheduling of nested loops with uniform dependencies in heterogeneous
networks of workstations. In8th International Symposium on Parallel Architectures,Algorithms and Networks (ISPAN’05).
IEEE, 6 pp.–. doi:10.1109/ISPAN.2005.40
[32] Alex Rogozhnikov. 2022. Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation. InInternational
Conference on Learning Representations. https://openreview.net/forum?id=oapKSVM2bcj
[33] Oliver Rübel, Andrew Tritt, Benjamin Dichter, Thomas Braun, Nicholas Cain, Nathan Clack, Thomas J. Davidson,
Max Dougherty, Jean-Christophe Fillion-Robin, Nile Graddis, Michael Grauer, Justin T. Kiggins, Lawrence Niu,
Doruk Ozturk, William Schroeder, Ivan Soltesz, Friedrich T. Sommer, Karel Svoboda, Ng Lydia, Loren M. Frank,
and Kristofer Bouchard. 2019. NWB:N 2.0: An Accessible Data Standard for Neurophysiology.bioRxiv(2019).
arXiv:https://www.biorxiv.org/content/early/2019/01/17/523035.full.pdf doi:10.1101/523035
[34] Spotify. 2025.Luigi: A python module that helps you build complex pipelines of batch jobs. https://github.com/spotify/luigi
[35] Min-You Wu, Wei Shu, and Yong Chen. 2000. Runtime parallel incremental scheduling of DAGs. InProceedings 2000
International Conference on Parallel Processing. IEEE, 541–548.
[36] Hongwei Xi and Frank Pfenning. 1998. Eliminating array bound checking through dependent types.SIGPLAN Not.33,
5 (May 1998), 249–257. doi:10.1145/277652.277732
[37] Jheng-Hong Yang and Jimmy Lin. 2024. Toward Automatic Relevance Judgment using Vision–Language Models for
Image–Text Retrieval Evaluation. arXiv:2408.01363 [cs.IR] https://arxiv.org/abs/2408.01363
[38] Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, and Weinan Zhang. 2025. AgentNet:
Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems. InThe Thirty-ninth Annual Conference
on Neural Information Processing Systems. https://openreview.net/forum?id=tXqLxHlb8Z
[39] Zhishen Yang, Raj Dabre, Hideki Tanaka, and Naoaki Okazaki. 2024. Scicap+: A knowledge augmented dataset to
study the challenges of scientific figure captioning.Journal of Natural Language Processing31, 3 (2024), 1140–1165.
[40] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng,
Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. 2025. AFlow: Automating Agentic
Workflow Generation. InThe Thirteenth International Conference on Learning Representations. https://openreview.net/
forum?id=z5uVAKwmjf

Operon: Incremental Construction of Ragged Data via Named Dimensions 23
A Compatibility with Nested Containers
We show that our design of arrays is compatible with traditional multidimensional arrays. Nested
linear containers serve as natural baselines for this comparison, as they are most commonly used
in many languages to express ragged data. In our formulation, such arrays correspond to systems
with atotally ordereddimension space. Each resolution in a shape then represents an individual
container, with the 𝑑,𝑐, andℓvalues denoting the depth, the indices to this container, and the
length, respectively. As such, we discuss how shapes and arrays transform when we linearly extend
the dimension space to a total order.
Hereafter, we understand a linear extension 𝐿as a permutation of Dthat preserves the original
order≺; the extended order𝑑≺ 𝐿𝑒means that𝑑appears before𝑒in𝐿.
Definition A.1 (Canonical expansions).For a shape 𝑅on(D,≺) with a linear extension 𝐿, if a
shape𝑅𝐿on(D,≺𝐿)satisfies
(𝑑,𝑐𝐿,ℓ)∈𝑅𝐿=⇒ (𝑑,𝑐 𝐿|Dep(𝑑),ℓ)∈𝑅,
we call𝑅𝐿acanonical expansionof𝑅relative to𝐿.
The following theorem asserts that all arrays can be rewritten into a nested container while
preserving their data.
Theorem A.2.For a shape𝑅on(D,≺)with a linear extension𝐿,
(1) the canonical expansion𝑅 𝐿uniquely exists;
(2)𝑅𝐿preserves coordinate spaces, that is, if F⊆D is closed in both(D,≺) and(D,≺𝐿), then
C(D;𝑅𝐿;F)=C(D;𝑅;F).
Finally, by comparing |𝑅|to|𝑅𝐿|, we obtain an upper bound on the additional storage required
by our design. Here, we exclude zero-length resolutions to avoid degenerate expansions that make
|𝑅𝐿|artificially small: an independent dimension with zero length could nullify everything else if
𝐿has it as the first element. The following theorem states that the number of resolutions never
exceeds the number of nested containers, provided the above assumption holds.
Theorem A.3.Consider a shape 𝑅on(D,≺) and a linear extension 𝐿. Assuming that ℓ>0for all
(𝑑,𝑐,ℓ)∈𝑅, the number of resolution entries|𝑅|satisfies|𝑅|≤|𝑅 𝐿|.
B Proofs
Lemma (2.4).A subspace E⊆D is convex if and only if it is an order-convex subposet, that is, if
𝑑,𝑒∈E,𝑓∈D, and𝑑⪯𝑓⪯𝑒, then𝑓∈E.
Proof.(⇒) IfEis convex, thenEis an order-convex subposet.
Assume for contradiction thatEis not order-convex. Then, there exists𝑑,𝑒,𝑓∈Dsuch that
𝑑⪯𝑒⪯𝑓, 𝑑,𝑓∈E, 𝑒∉E.
Since𝑒⪯𝑓 and𝑓∈E while𝑒∉E ,𝑒∈Dep(E) . Then, since 𝑑⪯𝑒 , it follows that 𝑑∈Dep(E)↓.
However, since 𝑑∈ E ,𝑑∉Dep(E) . This implies Dep(E)↓≠Dep(E) , which contradicts the
convexity ofE. Therefore,Emust be an order-convex subposet.
(⇐) IfEis an order-convex subposet, thenEis convex:
Assume for contradiction thatEis not convex, then there exists𝑑,𝑒such that
𝑑∉Dep(E), 𝑒∈Dep(E), 𝑑⪯𝑒.
By definition ofDep(E),𝑒∉E, and there exists𝑓∈Esuch that𝑒⪯𝑓.

24 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
If𝑑∈E , since𝑑⪯𝑒⪯𝑓 and𝑑,𝑓∈E , the order convexity of Ewould force 𝑒∈E , a contradiction.
However, if 𝑑∉E , since𝑑⪯𝑓 , it follows that 𝑑∈Dep(E) , also a contradiction. Therefore, Emust
be convex. □
Corollary (2.5).Every principal dependency space is closed.
Proof.Singletons are order-convex.□
Proposition (2.11-(1)).Given a dimension space(D,≺)and a shape𝑅, we have:
For closed subspacesF′⊆F⊆D,𝑐∈C(D;𝑅;F)=⇒𝑐| F′∈C(D;𝑅;F′).
Proof. Consider a dimension 𝑑∈F′. From the in-bounds condition Ib(𝑅;𝑐), we have(𝑑,𝑐| Dep(𝑑))
∈dom(𝑅) and𝑅(𝑑,𝑐| Dep(𝑑))>𝑐(𝑑) . SinceF′is closed, Dep(𝑑) ⊆ F′, from which we have
(𝑐|F′)|Dep(𝑑) =𝑐| Dep(𝑑) . Then, we can rewrite the statements as (𝑑,(𝑐|F′)|Dep(𝑑))∈dom(𝑅) and
𝑅(𝑑,(𝑐|F′)|Dep(𝑑))>𝑐|F′(𝑑). This means that Ib(𝑅;𝑐|F′)is satisfied. Therefore, 𝑐|F′∈C(D ;𝑅;F′).
□
Proposition (2.11-(2)).Given a dimension space(D,≺)and a shape𝑅, we have:
For a closedF⊆D,C∗(D;𝑅;F,∅)=C(D;𝑅;F).
Proof.SinceFis closed,Dep(F)=∅.
C∗(D;𝑅;F,∅)=
𝑐|F|𝑐∈C(D;𝑅;F↓)∧𝑐| Dep(F) =∅	
={𝑐|𝑐∈C(D;𝑅;F)}
=C(D;𝑅;F)
□
Proposition (2.11-(3)).Given a dimension space(D,≺)and a shape𝑅, we have:
For a convexE⊆D and a coordinate 𝑐Dep(E)∈C(D ;𝑅;Dep(E)) , there exists arestricted shape
𝑅|(E,𝑐 Dep(E)), a shape on(E,≺| E), such that
C∗(D;𝑅;E,𝑐 Dep(E))=C
E;𝑅|(E,𝑐 Dep(E));E
.
That is, we can interpret each subcoordinate space as a coordinate space when the shape is appropriately
restricted. We have𝑅| (E,∅)⊆𝑅whenEis closed.
Proof.We show that the proposed equality holds for
𝑅|(E,𝑐 Dep(E))= 
 𝑒,𝑐| Dep(𝑒)\Dep(E) ,ℓ(𝑒,𝑐,ℓ)∈𝑅∧
𝑒∈E∧
∀𝑑∈Dep(𝑒)∩Dep(E).𝑐(𝑑)=𝑐 Dep(E)(𝑑) 

by proving two inclusions.
(⊆)C∗(D;𝑅;E,𝑐 Dep(E))⊆C(E;𝑅|(E,𝑐 Dep(E));E)
For𝑐∈C∗(D;𝑅;E,𝑐 Dep(E)), from the definition of subcoordinate, there exists a corresponding
𝑐+∈C(D;𝑅;E↓)such that𝑐+|E=𝑐and𝑐+|Dep(E) =𝑐Dep(E) .
Now, consider a dimension 𝑑∈E . From the in-bounds condition Ib(𝑅;𝑐+), we have(𝑑,𝑐+|Dep(𝑑))
∈dom(𝑅) and𝑅(𝑑,𝑐+|Dep(𝑑))>𝑐+(𝑑). Also, for all 𝑒∈Dep(𝑑)∩Dep(E) , the coordinate 𝑐+|Dep(𝑑)
satisfies𝑐+|Dep(𝑑)(𝑒)=𝑐 Dep(E)(𝑒). Therefore, from the definition of𝑅| (E,𝑐 Dep(E)), we have:
𝑅|(E,𝑐 Dep(E))(𝑑,𝑐+|Dep(𝑑)\Dep(E))=𝑅(𝑑,𝑐+|Dep(𝑑)).

Operon: Incremental Construction of Ragged Data via Named Dimensions 25
Also, since we can write𝑐+as𝑐+=𝑐⊔𝑐 Dep(E) . we can induce the following equality:
𝑐+|Dep(𝑑)\Dep(E) =𝑐| Dep(𝑑)\Dep(E)⊔𝑐Dep(E)|Dep(𝑑)\Dep(E)
=𝑐| Dep(𝑑)⊔∅
=𝑐| Dep(𝑑).
Combining these results, we get
(𝑑,𝑐| Dep(𝑑))∈dom(𝑅|(E,𝑐 Dep(E)))∧𝑅|(E,𝑐 Dep(E))(𝑑,𝑐| Dep(𝑑)\Dep(E))>𝑐(𝑑).
This means thatIb(𝑅| (E,𝑐 Dep(E));𝑐)is satisfied, and thus𝑐∈C(E;𝑅| (E,𝑐 Dep(E));E).
Therefore,C∗(D;𝑅;E,𝑐 Dep(E))⊆C(E;𝑅|(E,𝑐 Dep(E));E).
(⊇)C∗(D;𝑅;E,𝑐 Dep(E))⊇C(E;𝑅|(E,𝑐 Dep(E));E)
For𝑐∈C(E;𝑅| (E,𝑐 Dep(E));E), let𝑐+=𝑐⊔𝑐 Dep(E) Naturally,dom(𝑐+)=E⊔Dep(E)=E↓.
Now, consider a dimension𝑑∈dom(𝑐+).
•If𝑑∈E , since the in-bounds condition Ib(𝑅|(E,𝑐 Dep(E));𝑐)holds, we have(𝑑,𝑐| Dep(𝑑))∈
dom(𝑅|(E,𝑐 Dep(E)))and𝑅|(E,𝑐 Dep(E))(𝑑,𝑐| Dep(𝑑))>𝑐(𝑑) . From the definition of 𝑅|(E,𝑐 Dep(E)),
there exists a corresponding(𝑑,𝑐′,𝑙′)∈𝑅that satisfies the following conditions:
𝑐′|Dep(𝑑)\Dep(E) =𝑐;𝑙′=𝑙;∀𝑒∈Dep(𝑑)∩Dep(E),𝑐′(𝑒)=𝑐 Dep(E)(𝑒).
Now, consider a dimension𝑒∈dom(𝑐′)=Dep(𝑑).
–If𝑒∈Dep(E), then𝑐′(𝑒)=𝑐 Dep(E)(𝑒)=𝑐+(𝑒).
–Otherwise,𝑒∈Dep(𝑑)\Dep(E). Then,𝑐′(𝑒)=𝑐(𝑒)=𝑐+(𝑒).
Therefore,𝑐′=𝑐+|Dep(𝑑) . We can conclude that (𝑑,𝑐+|Dep(𝑑))∈dom(𝑅) and𝑅(𝑑,𝑐+|Dep(𝑑))
>𝑐+(𝑑).
•Otherwise, 𝑑∈Dep(E) . Then, since Ib(𝑅;𝑐Dep(E))holds, we have(𝑑,𝑐 Dep(E)|Dep(𝑑)) ∈
dom(𝑅) and𝑅(𝑑,𝑐 Dep(E)|Dep(𝑑))>𝑐 Dep(E)(𝑑). Also, sinceEis convex, we have Dep(𝑑)⊆
Dep(E)↓=Dep(E) , which means that 𝑐+|Dep(𝑑) =𝑐Dep(E)|Dep(𝑑) . Therefore, we can con-
clude that(𝑑,𝑐+|Dep(𝑑))∈dom(𝑅)𝑅(𝑑,𝑐+|Dep(𝑑))>𝑐+(𝑑).
This means that Ib(𝑅;𝑐+)is satisfied, and thus 𝑐+∈C(E ;𝑅;E↓). Then, since 𝑐+|Dep(E) =𝑐Dep(E) ,
𝑐+|E=𝑐∈C∗(D;𝑅;E,𝑐 Dep(E)).
Therefore,C∗(D;𝑅;E,𝑐 Dep(E))⊇C(E;𝑅;E).
From the two inclusions, the stated equality holds.□
Lemma (3.3).For a partial shape 𝑅and a resolution(𝑑,𝑐,ℓ) , the extension 𝑅{(𝑑,𝑐)↦→ℓ} stays a
partial shape ifComp( 𝑅;𝑑,𝑐).
Proof.Let’s call the extended resolution map 𝑅′=𝑅{(𝑑,𝑐)↦→ℓ}.
Consider a pair(𝑑∗,𝑐∗)∈dom(𝑅′).
•If(𝑑∗,𝑐∗)∈dom(𝑅), we haveIb( 𝑅;𝑐∗)since𝑅is a partial shape.
•Otherwise, if it is a newly added(𝑑,𝑐),Ib( 𝑅;𝑐)holds by the definition ofComp( 𝑅;𝑑,𝑐).
In both cases,Ib( 𝑅;𝑑∗,𝑐∗)holds.
Now, consider a dimension 𝑒∈dom(𝑐) . From Ib(𝑅;𝑑∗,𝑐∗), we have(𝑒,𝑐∗|Dep(𝑒))∈dom(𝑅)and
𝑅(𝑒,𝑐∗|Dep(𝑒))>𝑐∗(𝑒). Since Comp(𝑅;𝑑,𝑐) requires(𝑑,𝑐) to not be in dom(𝑅), the same resolution
is also present in 𝑅′. Therefore,(𝑒,𝑐∗|Dep(𝑒))∈dom(𝑅′)and𝑅′(𝑒,𝑐∗|Dep(𝑒))>𝑐∗(𝑒). This means
thatIb(𝑅′,𝑐∗)is satisfied.
Since Ib(𝑅′,𝑐∗)holds for any pair (𝑑∗,𝑐∗) ∈dom(𝑅′), the extended resolution map 𝑅′=
𝑅{(𝑑,𝑐)↦→ℓ}is also a partial shape.□

26 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Theorem (3.4).A partial shape has a compatible resolution if and only if it is incomplete.
Proof.Let 𝑅denote the partial shape on(D,≺).
(⇒): If the partial shape has a compatible resolution, then the partial shape is incomplete.
Let(𝑑,𝑐,ℓ) be the compatible resolution. Since Comp(𝑅;𝑑,𝑐) holds, Ib(𝑅;𝑐)∧(𝑑,𝑐)∉dom( 𝑅).
This is a counterexample to Ib(𝑅;𝑐)→(𝑑,𝑐)∈dom( 𝑅), which is a required condition for 𝑅to be
complete. Therefore, 𝑅is incomplete.
(⇐): If the partial shape is incomplete, it has a compatible resolution.
Assume for contradiction that for all 𝑑∈D and all𝑐:Dep(𝑑)→N 0, ifIb(𝑅;𝑐)holds, then
(𝑑,𝑐)∈dom( 𝑅). From the definition of partial shape, we already know that for all 𝑑∈D and all
𝑐: Dep(𝑑)→N 0, if(𝑑,𝑐)∈dom( 𝑅), thenIb(𝑅;𝑐). Combining these two yields:
∀𝑑∈D.∀𝑐: Dep(𝑑)→N 0.(𝑑,𝑐)∈dom( 𝑅)⇔Ib(𝑅;𝑐).
This is the defining condition for 𝑅to be complete, a contradiction. Therefore, there exists (𝑑,𝑐)
such that Ib(𝑅;𝑐)and(𝑑,𝑐)∉dom( 𝑅). Then, for any ℓ∈N 0,(𝑑,𝑐,ℓ) is a resolution compatible
with𝑅. □
Theorem (3.5).There is no infinite sequence of partial shapes where each step adds a resolution.
Proof. It suffices to show that for each dimension 𝑑∈D , the number of resolutions (𝑑,𝑐,ℓ)
that you can add to the partial shape 𝑅on a dimension space (D,≺) is finite. Adding a resolution
(𝑑,𝑐,ℓ) for which Ib(𝑅;𝑐)does not hold would violate the partial shape condition in the resulting
resolution map. Hence, it suffices to show the set of coordinates 𝑐:Dep(𝑑)→N 0that satisfies
Ib(𝑅;𝑐)is finite.
For a base step, consider a primary dimension 𝑑0. In this case, dom(𝑐)=Dep(𝑑)=∅ , so there
exists exactly one valid coordinate: an empty function.
For an inductive step, let 𝑑be a non-primary dimension. Assume that for each 𝑒∈Dep(𝑑) , the
number of resolutions(𝑒,𝑐 𝑒)such thatIb( 𝑅;𝑐𝑒)is finite. By the definition ofIb( 𝑅;𝑐), we have
∀𝑒∈dom(𝑐)=Dep(𝑑).(𝑒,𝑐| Dep(𝑒))∈𝑅∧𝑅(𝑒,𝑐| Dep(𝑒))≥𝑐(𝑒).
By the inductive hypothesis, each 𝑒admits only finitely many valid coordinates 𝑐𝑒. Hence, there
exists a finite maximal 𝜋(𝑒,𝑐𝑒)for the function 𝜋, which is the function used to determine 𝑅(𝑒,𝑐𝑒).
Since Dep(𝑑) is finite and each component 𝑐(𝑒) is bounded above by a finite value, the total number
of coordinates𝑐: Dep(𝑑)→N 0satisfyingIb( 𝑅;𝑐)is also finite.
Therefore, by induction on the partial order ≺, for every dimension 𝑑∈D , the set of coordinates
𝑐:Dep(𝑑)→N 0satisfying Ib(𝑅;𝑐)is finite. Since there is a finite number of dimensions in D, this
means that there are only a finite number of resolutions that can be added to 𝑅.□
Theorem (3.6).If Comp(𝑅;𝑑,𝑐) andComp(𝑅;𝑑′,𝑐′)with(𝑑,𝑐)≠(𝑑′,𝑐′), then Comp(𝑅{(𝑑,𝑐)↦→
ℓ};𝑑′,𝑐′).
Proof.FromComp( 𝑅;𝑑,𝑐), we haveIb( 𝑅;𝑐′)and(𝑑′,𝑐′)∉dom(𝑅).
Now, consider a dimension 𝑒∈dom(𝑐′). From the in-bounds condition Ib(𝑅;𝑐′), we have
(𝑒,𝑐′|Dep(𝑒))∈dom(𝑅)and𝑅(𝑒,𝑐′|Dep(𝑒))>𝑐′(𝑒). Since Comp(𝑅;𝑑,𝑐) requires(𝑑,𝑐) to not be
indom(𝑅), the same resolution is also present in 𝑅′. Therefore,(𝑒,𝑐′|Dep(𝑒)) ∈dom(𝑅′)and
𝑅′(𝑒,𝑐′|Dep(𝑒))>𝑐′(𝑒). This means thatIb( 𝑅′;𝑐′)is satisfied.
Also, since(𝑑′,𝑐′)∉dom(𝑅), it is trivial that(𝑑′,𝑐′)∉dom(𝑅{(𝑑,𝑐)↦→ℓ}).
Therefore, we can conclude thatComp( 𝑅{(𝑑,𝑐)↦→ℓ};𝑑′,𝑐′).□

Operon: Incremental Construction of Ragged Data via Named Dimensions 27
Corollary (3.7).Under a fixed function 𝜋:Ð
𝑑∈D({𝑑}×[Dep(𝑑)→N 0])→N 0, any fair
execution of Alg. 1 or Alg. 2 terminates and returns the same complete shape.
Proof. Since Alg. 1 or Alg. 2 both execute the same function 𝜋(𝑑,𝑐) to acquire the resolution at
(𝑑,𝑐) , the values added to the shape at the same coordinate are identical between the two algorithms.
Only the order of additions may differ, as Alg 2 executes the functions in parallel.
However, since the shape is an unordered set of triples (𝑑,𝑐,ℓ) , the order of addition does not
matter. Furthermore, as stated in Theorem 3.6, adding a resolution to the shape does not affect the
availability of other resolutions. Therefore, the sets of resolutions added are identical, and the two
algorithms return the same shape.□
Proposition (3.9).Def. 3.8 is a strict extension of the original definition of coordinates in Def. 2.9.
That is,𝐶(D;𝑅;F)is unchanged under either definition when𝑅is a complete shape.
Proof. Note that throughout the proof, we refer to the two conditions in Def. 3.8 as conditions
(1) and (2).
Assume𝑅is a complete shape on(D,≺), i.e.
∀𝑑∈D.∀𝑐∈Dep(𝑑)→N 0.(𝑑,𝑐)∈dom(𝑅)↔Ib(𝑅;𝑐).
Let𝑐:F⇀N 0satisfy the conditions in Def. 3.8. We show that 𝑐must be total onFand satisfy
Ib(𝑅;𝑐).
Suppose for contradiction that 𝑐is partial, i.e. dom(𝑐)⊂F . Then, there exists a ≺-minimal
element𝑑∗∈F\dom(𝑐) . From the minimality of 𝑑∗, we can infer that Dep(𝑑∗)⊆dom(𝑐) , and thus
𝑐|Dep(𝑑∗)is total on Dep(𝑑∗). Now, consider a dimension 𝑒∈Dep(𝑑∗). Applying the condition (1),
we have(𝑒,𝑐| Dep(𝑒))∈dom(𝑅) and𝑅(𝑒,𝑐| Dep(𝑒))>𝑐(𝑒) . This means that Ib(𝑅;𝑐|Dep(𝑑∗))is satisfied.
Since𝑅is a complete shape, this means that (𝑑∗,𝑐|Dep(𝑑∗))∈dom(𝑅) , contradicting condition (2).
Hence,𝑐is total onF.
With totality, condition (1) is exactly
∀𝑑∈dom(𝑐). 𝑑,𝑐| Dep(𝑑)∈dom(𝑅)∧𝑅 𝑑,𝑐| Dep(𝑑)>𝑐(𝑑),
i.e.Ib(𝑅;𝑐).
Thus, Def. 3.8 simplifies to
{𝑐:F→N 0|Ib(𝑅;𝑐)}
which is precisely Def. 2.9. Therefore, the coordinate space remains unchanged under either
definition. □
Theorem (3.10-(1)).For a partial shape 𝑅on(D,≺) , a closedF⊆D , a dimension 𝑑∈F , and a
coordinate𝑐:Dep(𝑑)→N 0with Comp(𝑅;𝑑,𝑐),∀𝑐′∈C(D ;𝑅;F).𝑐′|Dep(𝑑) =𝑐=⇒𝑑∉dom(𝑐′).
Proof. Suppose for contradiction that 𝑑∈dom(𝑐′). Then by the definition of C(D ;𝑅;F), we
have(𝑑,𝑐| Dep(𝑑))=(𝑑,𝑐)∈dom( 𝑅). This contradicts Comp(𝑅;𝑑,𝑐), which states that (𝑑,𝑐)∉
dom(𝑅). Therefore,𝑑∉dom(𝑐′).□
Theorem (3.10-(2)).For a partial shape 𝑅on(D,≺) , a closedF⊆D , a dimension 𝑑∈F , and a
coordinate𝑐: Dep(𝑑)→N 0withComp( 𝑅;𝑑,𝑐), writing 𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ},
C(D;𝑅ℓ;F)=
C(D;𝑅;F)\𝑈
∪ℓ−1Ø
𝑖=0{𝑐′{𝑑↦→𝑖}|𝑐′∈𝑈}
where𝑈=
𝑐′∈C(D;𝑅;F)|𝑐′|Dep(𝑑) =𝑐	
.

28 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Proof. For a dimension 𝑑∗∈F and𝑐∗:F⇀N 0, we define two conditions corresponding to
each conditions in Def. 3.8:
•cond 1(𝑅;𝑑∗,𝑐∗) ⇐⇒𝑑∗∈dom(𝑐∗∧(𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅)∧𝑅(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗)
•cond 2(𝑅;𝑑∗,𝑐∗) ⇐⇒𝑑∗∈F\dom(𝑐∗)∧(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅)
Then, we can rewrite Def. 3.8 as follows:
C(D;𝑅;F)=
𝑐∗:F⇀N 0|∀𝑑∗∈F.cond 1(𝑅;𝑑∗,𝑐∗)∨cond 2(𝑅;𝑑∗,𝑐∗)	
.
Now, we prove the proposed equality by proving two inclusions.
(⊆): Let𝑐∗∈C(D;𝑅ℓ,F). Consider two cases.
Case 1:𝑐∗|Dep(𝑑) ≠𝑐.For all𝑑∗∈F,
•If𝑑∗∈dom(𝑐∗), then cond 1(𝑅ℓ;𝑑∗,𝑐∗)should hold, from which we get (𝑑∗,𝑐∗|Dep(𝑑∗))∈
dom(𝑅ℓ)∧𝑅ℓ(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗). Since𝑐∗|Dep(𝑑) ≠𝑐,(𝑑∗,𝑐∗|Dep(𝑑∗))is distinct from
(𝑑,𝑐) , and thus the same is true for 𝑅, i.e.(𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅)∧𝑅(𝑑∗,𝑐∗|Dep(𝑑∗))>
𝑐∗(𝑑∗). Therefore, cond 1(𝑅;𝑑∗,𝑐∗)is satisfied.
•Otherwise, 𝑑∗∈ F\dom(𝑐∗). Then, cond 2(𝑅ℓ;𝑑∗,𝑐∗)should hold, from which we get
(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅ℓ). Since𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ} , any resolution absent in 𝑅ℓis also
absent in𝑅. Therefore,(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅)which means that cond 2(𝑅;𝑑∗,𝑐∗)is satis-
fied.
Since either cond 1(𝑅;𝑑∗,𝑐∗)orcond 2(𝑅;𝑑∗,𝑐∗)is true for all 𝑑∗∈F,𝑐∗∈C(D ;𝑅;F). Also,
from𝑐∗|Dep(𝑑) ≠𝑐, we have𝑐∗∉𝑈. Therefore,𝑐∗∈C(D;𝑅;F)\𝑈.
Case 2:𝑐∗|Dep(𝑑) =𝑐.Suppose for contradiction that 𝑑∉dom(𝑐∗). Then, cond 2(𝑅ℓ;𝑑,𝑐∗)should
hold, which requies (𝑑,𝑐∗|Dep(𝑑))=(𝑑,𝑐)∉dom( 𝑅ℓ), which contradicts 𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ} . There-
fore,𝑑∈dom(𝑐∗). Then, the cond 1(𝑅ℓ;𝑑,𝑐∗)should hold, from which we get𝑐∗(𝑑)<𝑅ℓ(𝑑,𝑐)=ℓ.
Let𝑐′=𝑐∗|F\{𝑑}. We now show that𝑐′∈C(D;𝑅;F). For all𝑑∗∈F,
•For𝑑∗=𝑑, it is trivial that 𝑑∉dom(𝑐∗). SinceFis closed,𝑑𝑒𝑝(𝑑)⊆F\{𝑑}, and thus
𝑐′|Dep(𝑑) =𝑐∗|Dep(𝑑) =𝑐. Then from Comp(𝑅;𝑑,𝑐), we have(𝑑,𝑐′|Dep(𝑑))=(𝑑,𝑐)∉ 𝑅.
Therefore, cond 2(𝑅;𝑑,𝑐′)is satisfied.
•For other dimensions, if 𝑑∗∈dom(𝑐∗),cond 1(𝑅ℓ;𝑑∗,𝑐∗)should hold, from which we have
(𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅ℓ)∧𝑅ℓ(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗). Since(𝑑∗,𝑐∗|Dep(𝑑∗))is distinct from
(𝑑,𝑐), the same is true for 𝑅, i.e.(𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅)∧𝑅(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗).
Suppose for contradiction that 𝑑∈Dep(𝑑∗). By the definition of partial shape, (𝑑∗,𝑐∗|Dep(𝑑∗))
∈dom(𝑅)implies that Ib(𝑅;𝑐∗|Dep(𝑑∗)). Since we assumed that 𝑑∈Dep(𝑑∗), this requires
(𝑑,𝑐)∈𝑅, which contradicts Comp(𝑅;𝑑,𝑐). Therefore, 𝑑∉Dep(𝑑∗), from which we have
𝑐′|Dep(𝑑∗)=𝑐∗|Dep(𝑑∗).
Given this, the above condition is equivalent to (𝑑∗,𝑐′|Dep(𝑑∗))∈dom(𝑅)∧𝑅(𝑑∗,𝑐′|Dep(𝑑∗))>
𝑐′(𝑑∗), which is cond 1(𝑅;𝑑∗,𝑐′). Therefore, cond 1(𝑅;𝑑∗,𝑐′)is satisfied.
•Otherwise,𝑑∗∉dom(𝑐∗)and𝑑∗≠𝑑. Then, cond 2(𝑅ℓ;𝑑∗,𝑐∗)should hold, from which we
get(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅ℓ). If𝑑∈Dep(𝑑∗),𝑐′|Dep(𝑑∗)=𝑐∗|Dep(𝑑∗)\{𝑑}is never total on
Dep(𝑑∗), and thus it is trivial that (𝑑∗,𝑐′|Dep(𝑑∗))∉dom(𝑅). Otherwise, 𝑑∉Dep(𝑑∗), from
which we have 𝑐′|Dep(𝑑∗)=𝑐∗|Dep(𝑑∗). Since𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ} , any resolution absent in
𝑅ℓis also absent in 𝑅, i.e.(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅). Either way,(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅),
which means that cond 2(𝑅;𝑑∗,𝑐∗)is satisfied.

Operon: Incremental Construction of Ragged Data via Named Dimensions 29
Since either cond 1(𝑅;𝑑∗,𝑐′)orcond 2(𝑅;𝑑∗,𝑐′)is true for all 𝑑∗∈ F ,𝑐′∈ C(D ;𝑅;F). Fur-
thermore, since 𝑐′|Dep(𝑑) =𝑐,𝑐′∈𝑈, and since 𝑐∗=𝑐′{𝑑↦→𝑐∗(𝑑)} with𝑐∗(𝑑)<ℓ ,𝑐∗∈Ðℓ−1
𝑖=0{𝑐′{𝑑↦→𝑖}|𝑐′∈𝑈}.
Combining the two cases, we get
C(D;𝑅ℓ;F)⊆
C(D;𝑅;F)\𝑈
∪ℓ−1Ø
𝑖=0{𝑐′{𝑑↦→𝑖}|𝑐′∈𝑈}.
(⊇): Let𝑐∗be in the right-hand side. Like before, consider two cases:
Case 1:𝑐∗∈C(D ;𝑅;F)\𝑈 .Then,𝑐∗|Dep(𝑑) ≠𝑐. Now, we prove 𝑐∗∈C(D ;𝑅ℓ;F)in a similar
way as the first case in the⊆direction. For all𝑑∗∈F,
•If𝑑∗∈dom(𝑐∗), then cond 1(𝑅;𝑑∗,𝑐∗)should hold, from which we get (𝑑∗,𝑐∗|Dep(𝑑∗))∈
dom(𝑅)∧𝑅(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗). Since𝑅ℓ=𝑅{(𝑑,𝑐) ↦→ℓ} and(𝑑,𝑐)∉dom( 𝑅),
any resolution present in 𝑅is also present in 𝑅ℓ. Therefore,(𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅ℓ)∧
𝑅ℓ(𝑑∗,𝑐∗|Dep(𝑑∗))>𝑐∗(𝑑∗)which means that cond 1(𝑅;𝑑∗,𝑐∗)is satisfied.
•Otherwise, 𝑑∗∈ F\dom(𝑐∗). Then, cond 2(𝑅;𝑑∗,𝑐∗)should hold, from which we get
(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅). Since𝑐∗|Dep(𝑑) ≠𝑐,(𝑑∗,𝑐∗|Dep(𝑑∗))is distinct from(𝑑,𝑐) , and thus
the resolution is also absent in 𝑅ℓ, i.e.(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅ℓ). Therefore, cond 2(𝑅ℓ;𝑑∗,𝑐∗)
is satisfied.
Since either cond 1(𝑅ℓ;𝑑∗,𝑐∗)or cond 2(𝑅ℓ;𝑑∗,𝑐∗)is true for all𝑑∗∈F,𝑐∗∈C(D;𝑅ℓ;F).
Case 2:𝑐∗=𝑐′{𝑑↦→𝑖}with𝑐′∈𝑈and0≤𝑖<ℓ.Then,𝑐∗|Dep(𝑑) =𝑐. For all𝑑∗∈F,
•For𝑑∗=𝑑, it is trivial that 𝑑∈dom(𝑐∗). From the definition of 𝑅ℓ, we also have(𝑑,𝑐∗|Dep(𝑑))
=(𝑑,𝑐)∈dom( 𝑅ℓ)∧𝑅ℓ(𝑑,𝑐)=ℓ>𝑖. Therefore, cond 1(𝑅ℓ;𝑑,𝑐∗)is satisfied.
•For other dimensions, if 𝑑∗∈dom(𝑐∗)and𝑑∗≠𝑑, then𝑑∗∈dom(𝑐′). Then, cond 1(𝑅;𝑑∗,𝑐′)
should hold, from which we get (𝑑∗,𝑐′|Dep(𝑑∗))∈dom(𝑅)∧𝑅(𝑑∗,𝑐′|Dep(𝑑∗))>𝑐′(𝑑∗). Since
𝑅ℓ=𝑅{(𝑑,𝑐)↦→ℓ} and(𝑑,𝑐)∉dom( 𝑅), any resolution present in 𝑅is also present in 𝑅ℓ.
Therefore,(𝑑∗,𝑐′|Dep(𝑑∗))∈dom(𝑅ℓ)∧𝑅ℓ(𝑑∗,𝑐′|Dep(𝑑∗))>𝑐′(𝑑∗).
Suppose for contradiction that 𝑑∈Dep(𝑑∗). Then,𝑐′|Dep(𝑑∗)is not total on Dep(𝑑∗), which
contradicts(𝑑∗,𝑐∗). Therefore,𝑑∉Dep(𝑑∗), from which we have𝑐∗|Dep(𝑑∗)=𝑐′|Dep(𝑑∗).
Given this, the above condition is equivalent to (𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅ℓ)∧𝑅ℓ(𝑑∗,𝑐∗|Dep(𝑑∗))
>𝑐∗(𝑑∗), which is cond 1(𝑅ℓ;𝑑∗,𝑐∗). Therefore, cond 1(𝑅;𝑑∗,𝑐∗)is satisfied.
•Otherwise, 𝑑∗∈F\dom(𝑐∗) ⊆F\dom(𝑐′). Then, cond 2(𝑅;𝑑∗,𝑐′)should hold, from
which we get(𝑑∗,𝑐′|Dep(𝑑∗))∉dom(𝑅).
If𝑑∈Dep(𝑑∗), suppose for contradiction that (𝑑∗,𝑐∗|Dep(𝑑∗))∈dom(𝑅). By the definition
of partial shape, this implies that Ib(𝑅;𝑐∗|Dep(𝑑∗)). Then since 𝑑∈Dep(𝑑∗), this requires
(𝑑,𝑐)∈𝑅, which contradictsComp( 𝑅;𝑑,𝑐). Therefore,(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅).
Otherwise,𝑑∉Dep(𝑑∗), from which we have 𝑐∗|Dep(𝑑∗)=𝑐′|Dep(𝑑∗), and thus(𝑑∗,𝑐∗|Dep(𝑑∗))
∉dom(𝑅).
Either way,(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅). Since(𝑑∗,𝑐∗|Dep(𝑑∗))is distinct from(𝑑,𝑐) , the res-
olution is also absent in 𝑅ℓ, i.e.(𝑑∗,𝑐∗|Dep(𝑑∗))∉dom(𝑅ℓ). Therefore, cond 2(𝑅ℓ;𝑑∗,𝑐∗)is
satisfied.
Since either cond 1(𝑅ℓ;𝑑∗,𝑐∗)or cond 2(𝑅ℓ;𝑑∗,𝑐∗)is true for all𝑑∗∈F,𝑐∗∈C(D;𝑅ℓ;F).

30 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Combining the two cases, we get
C(D;𝑅ℓ;F)⊇
C(D;𝑅;F)\𝑈
∪ℓ−1Ø
𝑖=0{𝑐′{𝑑↦→𝑖}|𝑐′∈𝑈}.
From the two inclusions, the stated equality holds.□
Lemma (4.1).Given(∅,∅,∅)|𝑝⊢(D,≺,Σ),
(1) the relation≺is a strict partial order overD;
(2) for all entity types𝜏∈dom(Σ), the characteristic dimension spaceΣ(𝜏)is closed under≺;
(3)for all tasks 𝑡=
𝑓,𝑠out,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
in𝑝, the dimension spaces FandΣ(𝜏 in,𝑖)\E in,𝑖
are closed under≺.
Proof. We prove the three items simultaneously by induction on the derivation of (∅,∅,∅)|
𝑝⊢(D,≺,Σ) generated by the rules in Fig. 5. Let the invariant Inv(D,≺,Σ) be the conjunction of
the items (1)-(3) in Lemma 4.1.
Base step (Unit).For 𝑝=() , the ruleUnityields (∅,∅,∅) . The empty relation is a strict partial
order on∅.dom(Σ)=∅ , so (2) is vacuous. There are no tasks in 𝑝, so (3) is also vacuous. Therefore,
Inv holds.
Auxiliary step (TaskDef).Assume Inv(D,≺,Σ) and the premises of TaskDef . Since|Eout|≤1,
we can say thatE out=∅or{𝑑out}.
Define
D′=D⊔E out;≺′=≺⊔F×E out;Σ′=Σ{𝜏out↦→F⊔E out}.
We show that Inv(D′,≺′,Σ′)holds.
We start by making couple claims about the properties of≺′:
(i) For any𝑑,𝑒∈D′, if(𝑑,𝑒)∈≺′, then𝑑∈D.
(ii) For any𝑑,𝑒∈D, if(𝑑,𝑒)∈≺′, then(𝑑,𝑒)∈≺.
(iii) For anyF∗⊆D, ifF∗is closed under≺, then it is also closed under≺′.
The proof for (i) is trivial. Consider (𝑑,𝑒)∈≺′. If(𝑑,𝑒)∈≺ , then naturally 𝑑∈D . Otherwise,
(𝑑,𝑒)∈F×E out, in which case 𝑑∈F⊆D . In both cases, 𝑑∈D . The proof for (ii) is even simpler.
Since(𝑑,𝑒)∉F×E out, it must be that(𝑑,𝑒)∈≺ . Finally, for (iii), assume for contradiction that
there existsF∗⊆D that is closed under ≺but not under≺′. Then, there exists 𝑑∈D′\F∗,𝑒∈F∗
such that(𝑑,𝑒)∈≺′. By (i),𝑑∈D . Then, by (ii)(𝑑,𝑒)∈≺ , which contradicts the closedness of F∗
under≺. Therefore,F∗is closed under≺′.
Now, we prove the three items that constitute Inv(D′,≺′,Σ′).
(1)≺′is a strict partial order onD′.
We prove this by showing that≺′is irreflexive and transitive.
Irreflexivity: Assume for contradiction that there exists 𝑑∈D′such that(𝑑,𝑑)∈≺′. By (i),
𝑑∈D , and then by (ii),(𝑑,𝑑)∈≺ . This contradicts the irreflexivity of ≺by the inductive hypothesis.
Therefore, there is no such𝑑, and thus≺′is irreflexive.
Transitivity: Suppose 𝑎≺′𝑏and𝑏≺′𝑐. By (i),𝑎,𝑏∈D , and then by (ii),(𝑎,𝑏)∈≺ . If𝑐∈D ,
then by (ii),(𝑏,𝑐)∈≺ . Since≺is transitive by the inductive hypothesis, we have (𝑎,𝑐)∈≺⊆≺′.
Otherwise, 𝑐∈E out. In which case,(𝑎,𝑐)∈D×E out, so we have(𝑎,𝑐)∈≺′. Therefore,≺′is
transitive.
(2) For all𝜏∈dom(Σ′),Σ′(𝜏)is closed under≺′.

Operon: Incremental Construction of Ragged Data via Named Dimensions 31
For𝜏=𝜏 out,Σ′(𝜏out)=F⊔E out. Assume for contradiction that this is not closed under ≺′.
Then, there exists 𝑑∈D′\Σ′(𝜏out),𝑒∈Σ′(𝜏out)such that𝑑≺′𝑒. Note that by (i), we have 𝑑∈D .
If𝑒∈F ⊆D , then by (ii) we have (𝑑,𝑒)∈≺ , which contradicts the closedness of Funder≺.
Otherwise,𝑒∈E out. However, since(𝑑,𝑒)∉≺ and(𝑑,𝑒)∉F×E out,(𝑑,𝑒)∉≺′, a contradiction.
Therefore,Σ′(𝜏out)is closed under≺′.
For other entity types 𝜏≠𝜏 out,Σ′(𝜏)=Σ(𝜏) . By the inductive hypothesis, Σ(𝜏) is closed under
≺, and by (iii), it should also be closed under≺′.
(3) For all tasks 𝑡∗=
𝑓,𝑠out,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
in𝑝, the dimension spaces FandΣ(𝜏 in,𝑖)\E in,𝑖
are closed under≺.
For𝑡∗=𝑡, the dimension spaces are closed under ≺by the premises ofTaskDef. For other
tasks𝑡∗≠𝑡, the dimension spaces are closed under ≺by the inductive hypothesis. By (iii), these
dimension spaces are also closed under≺′.
Inductive step (Chain).Assume (∅,∅,∅) |®𝑡⊢ (D 1,≺1,Σ1),(D1,≺1,Σ1) |𝑡′⊢ (D 2,≺2,Σ2),
andInv(D 1,≺1,Σ1). Since the only step that yields (D1,≺1,Σ1) |𝑡′⊢(D 2,≺2,Σ2)isTaskDef ,
Inv(D 2,≺2,Σ2)also holds from applying the auxiliary step above.
Therefore, Inv(D,≺,Σ)holds for the final triple(D,≺,Σ).□
Lemma (4.5).When a ticket’scountequals itsquota, the ticket is fully resolved.
Proof. Consider a task 𝑡=
𝑓,𝑠out,−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
,F,𝑛
and its ticket 𝑗𝑡(𝑐). Furthermore, let(𝑅,𝐸)
be the current state of resolutions and entities.
Base step: If|−−−−−−−−−−→
𝜏in,𝑖,Ein,𝑖
|=0, the ticket is always fully resolved. Therefore, the proposition
holds.
Inductive step: Expressing 𝑡𝑖=
_,
𝜏in,𝑖,E𝑖
,_,F𝑖,_
as the task that produces input entity 𝜏in,𝑖,
assume that the proposition holds for each𝑡 𝑖.
In order for the ticket’scountto equal itsquota, all tickets in 𝑗𝑡(𝑐)’s dependencies
Ä
𝑖𝑗𝑡𝑖
𝑐|F𝑖\Ein,𝑖
.
has to be in the Done state. This means that each of these tickets must have their count equal to
theirquota. By the inductive hypothesis, each ticket in the dependencies is fully resolved.
Consider a dimension𝑑∈F. From the parsing rule, we have𝑑∈Ð
𝑖Σ(𝜏𝑡𝑖).
•If∃𝑖.𝑑∈E𝑖, then𝑗𝑡𝑖(𝑐|Dep(𝑑))is the job responsible for creating the resolution (𝑑,𝑐| Dep(𝑑)).
The ticket corresponding to this job is present in the dependencies set of 𝑗𝑡(𝑐), and thus is
marked as done. Therefore,(𝑑,𝑐| Dep(𝑑))must be present in 𝑅.
•Otherwise,∃𝑖.𝑑∈F 𝑖. Since the tickets 𝑗𝑡𝑖are fully resolved, the resolution (𝑑,𝑐| Dep(𝑑))
must also be present in 𝑅.
Since for all𝑑∈F, the resolution(𝑑,𝑐| Dep(𝑑))is present in 𝑅,𝑗𝑡(𝑐)is also fully resolved.
By induction on the task in the order they are introduced, we can conclude that for all task 𝑡, the
proposition holds for 𝑡. This induction is valid as the each the parsing ruleTaskDeffrom Fig. 5
necessitates that all input entities for a task must have been introduced already.□
Theorem (A.2-(1)).For a shape𝑅on(D,≺)with a linear extension𝐿:
The canonical expansion𝑅 𝐿uniquely exists.

32 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Proof. LetDep𝐿(𝑑)denote the dependency space of dimension 𝑑∈D under≺𝐿. Since≺⊆≺𝐿,
we have
Dep(𝑑)⊆Dep𝐿(𝑑)for all𝑑∈D.
(Existence): Define a resolution map𝑅 𝐿on(D,≺𝐿)as follows:
dom(𝑅𝐿):=
(𝑑,𝑐𝐿)|𝑑∈D,𝑐 𝐿: Dep𝐿(𝑑)→N 0,Ib(𝑅;𝑐𝐿)	
and for every(𝑑,𝑐 𝐿)∈dom(𝑅 𝐿), set
𝑅𝐿(𝑑,𝑐𝐿):=𝑅(𝑑,𝑐 𝐿|Dep(𝑑)).
For all𝑑∈Dand a partial function𝑐:D⇀N 0, we have:
•If(𝑑,𝑐| Dep𝐿(𝑑))∈dom(𝑅 𝐿), then𝑅𝐿(𝑑,𝑐| Dep𝐿(𝑑))=𝑅(𝑑,(𝑐| Dep𝐿(𝑑))|Dep(𝑑))=𝑅(𝑑,𝑐| Dep(𝑑)).
•Ib(𝑅;𝑐)⇒Ib(𝑅;𝑐| Dep𝐿(𝑑))⇒(𝑑,𝑐| Dep𝐿(𝑑))∈dom(𝑅 𝐿).
•
(𝑑,𝑐| Dep𝐿(𝑑))∈dom(𝑅 𝐿)⇔Ib(𝑅;𝑐| Dep𝐿(𝑑))
⇒Ib(𝑅;𝑐| Dep(𝑑)) (∵Dep(𝑑)⊆Dep𝐿(𝑑))
⇔(𝑑,𝑐| Dep(𝑑))∈dom(𝑅) (∵𝑅is a shape)
From these, we get:
(𝑑,𝑐)∈dom(𝑅 𝐿)⇔Ib(𝑅;𝑐)
⇔∀𝑒∈dom(𝑐).(𝑒,𝑐| Dep(𝑒))∈dom(𝑅)∧𝑅(𝑒,𝑐| Dep(𝑒))>𝑐(𝑒)
⇔∀𝑒∈dom(𝑐).(𝑒,𝑐| Dep𝐿(𝑒))∈dom(𝑅 𝐿)∧𝑅𝐿(𝑒,𝑐| Dep𝐿(𝑒))>𝑐(𝑒)
⇔Ib(𝑅𝐿;𝑐).
This proves that 𝑅𝐿is a valid shape. Since 𝑅𝐿obviously satisfies the condition (𝑑,𝑐𝐿,ℓ)∈𝑅𝐿⇒
(𝑑,𝑐𝐿,ℓ)∈𝑅,𝑅 𝐿is a canonical expression.
(Uniqueness): Assume for contradiction that two canonical expansions𝑅 1≠𝑅 2exist.
By the definition of canonical expansions, whenever(𝑑,𝑐)∈dom(𝑅 1)∩dom(𝑅 2),
𝑅1(𝑑,𝑐)=𝑅(𝑑,𝑐| Dep(𝑑))=𝑅 2(𝑑,𝑐),
so any disagreement must come from the domains.
Hencedom(𝑅 1)≠dom(𝑅 2). Without loss of generality, pick
(𝑑∗,𝑐∗)∈dom(𝑅 1)\dom(𝑅 2)
with≺𝐿-minimal𝑑∗among all such witnesses.
Since𝑅1and𝑅2are both shapes,(𝑑∗,𝑐∗)∈dom(𝑅 1)means that Ib(𝑅 1;𝑐∗)holds, and(𝑑∗,𝑐∗)∉
dom(𝑅 2)means that Ib(𝑅 2;𝑐∗)does not. This means that there exists 𝑒∈dom(𝑐∗)such that either
• (𝑒,𝑐∗|Dep(𝑒))∈dom(𝑅 1)∧(𝑒,𝑐∗|Dep(𝑒))∉dom(𝑅 2), or
•𝑅 1(𝑒,𝑐∗|Dep(𝑒))≥𝑐∗(𝑒)>𝑅 2(𝑒,𝑐∗|Dep(𝑒)).
The latter is impossible since 𝑅1and𝑅2cannot disagree on the shared domain. However, the
former is also impossible since 𝑒≺𝐿𝑑∗and(𝑒,𝑐∗|Dep(𝑒))∈dom(𝑅 1)\dom(𝑅 2), conflicting the
≺𝐿-minimality of𝑑∗.
Therefore, the canonical expansion is unique.□
Theorem (A.2-(2)).For a shape𝑅on(D,≺)with a linear extension𝐿:
𝑅𝐿preserves coordinate spaces, that is, if F ⊆ D is closed in both(D,≺) and(D,≺𝐿), then
C(D;𝑅𝐿;F)=C(D;𝑅;F).

Operon: Incremental Construction of Ragged Data via Named Dimensions 33
Proof. In the proof for Theorem A.2-(1), we already established that the proposed canonical
expansion𝑅𝐿satisfies∀𝑐.Ib(𝑅 ;𝑐)⇔Ib(𝑅 𝐿;𝑐). Since a canonical expansion uniquely exists, it
follows that this is always true. Therefore,C(D;𝑅 𝐿;F)=C(D;𝑅;F).□
Theorem (A.3).Consider a shape 𝑅on(D,≺) and a linear extension 𝐿. Assuming that ℓ>0for
all(𝑑,𝑐,ℓ)∈𝑅, the number of resolution entries|𝑅|satisfies|𝑅|≤|𝑅 𝐿|.
Proof. From the unique canonical extension established in proof for Theorem A.2-(1), we have:
dom(𝑅𝐿):=
(𝑑,𝑐𝐿)|𝑑∈D,𝑐 𝐿∈Dep𝐿(𝑑)→N 0,Ib(𝑅;𝑐𝐿)	
and from the definition of shape:
dom(𝑅):={(𝑑,𝑐)|𝑑∈D,𝑐∈Dep(𝑑)→N 0,Ib(𝑅;𝑐)}.
For each element(𝑑,𝑐)∈dom(𝑅) , let us say that 𝑐+=𝑐⊔((Dep𝐿(𝑑)\Dep(𝑑))×{0}). Since we
can incrementally constructDep𝐿(𝑑)\Dep(𝑑)by appending the dimensions in the order defined
by𝐿, we can apply Lemma B.1 to getIb(𝑅;𝑐 𝐿). Then, it follows that(𝑑,𝑐 𝐿)∈dom(𝑅 𝐿).
Since there exists at least one element of dom(𝑅𝐿)for each element of dom(𝑅) , we can conclude
that|𝑅|≤|𝑅 𝐿|. □
Lemma B.1 (Helper for Theorem A.3).Let 𝑅be a shape on(D,≺) . Fix a coordinate 𝑐that satisfies
Ib(𝑅;𝑐). Suppose moreover that every resolvable next dimension at𝑐has positive length:
∀(𝑑,𝑐| Dep(𝑑))∈dom(𝑅).𝑅(𝑑,𝑐| Dep(𝑑))>0
Then for anyE⊆
𝑑∈D|(𝑑,𝑐| Dep(𝑑))∈dom(𝑅)	
\dom(𝑐) , the in-bounds condition Ib(𝑅,𝑐⊔(E×
{0}))holds.
Proof of Lemma B.1.Consider a dimension𝑒∈dom(𝑐)⊔Eand let𝑐+=𝑐⊔(E×{0}).
•If𝑒∈dom(𝑐) , since the in-bounds condition Ib(𝑅;𝑐)holds, we have(𝑒,𝑐+|Dep(𝑒))∈dom(𝑅)
and𝑅(𝑒,𝑐+|Dep(𝑒))>𝑐+(𝑒).
•Otherwise,𝑒∈E . Then, by choice of E, we have(𝑒,𝑐+|Dep(𝑒))∈dom(𝑅) and𝑅(𝑒,𝑐+|Dep(𝑒))
>𝑐+(𝑒)=0.
This means thatIb(𝑅;𝑐⊔(E× {0}))is satisfied.□
C Evaluation Data
Table 2. Data of Figure 7(a)
𝑡sleep 3
𝑁 5 10 15 20 40 60 80 100
OperonTry1 58.15 109.45 140.65 173.19 340.87 535.84 690.11 838.13
Try2 57.83 109.91 146.84 173.03 339.00 518.40 700.80 838.73
Try3 57.98 110.60 140.32 170.17 339.59 516.14 689.66 838.38
PrefectTry1 74.93 141.06 185.30 213.59 452.68 734.26 998.51 1290.88
Try2 78.00 139.22 199.10 217.88 453.80 752.55 1024.64 1238.67
Try3 76.97 146.49 191.00 216.71 477.41 739.78 998.43 1262.26
Theory 57 108 138 168 336 510 684 831

34 Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, and Minhyeong Lee
Table 3. Data of Figure 7(b)
𝑁 20
𝑡sleep 0 1 2 3 4 5
OperonTry1 8.32 57.64 113.75 169.98 226.22 282.10
Try2 8.49 57.70 113.97 169.84 227.54 281.93
Try3 8.52 57.70 113.84 169.93 225.94 282.00
PrefectTry1 126.87 144.07 175.22 218.61 269.51 314.82
Try2 125.55 138.22 181.05 217.82 272.58 312.72
Try3 126.06 137.74 179.65 218.55 264.00 316.78
Theory 0 56 112 168 224 280
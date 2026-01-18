# A Generalizable Framework for Building Executable Domain-Specific LLMs under Data Scarcity: Demonstration on Semiconductor TCAD Simulation

**Authors**: Di Wang, Zhenhua Wu, Yu Liu, Kai Chang, Shaohua Wu

**Published**: 2026-01-15 07:13:34

**PDF URL**: [https://arxiv.org/pdf/2601.10128v1](https://arxiv.org/pdf/2601.10128v1)

## Abstract
Scientific and engineering verticals often suffer from data scarcity and strict executability requirements: models must generate not only fluent text, but also syntactically valid, tool-compilable scripts. We present a schema-first alignment framework for building compact, executable domain-specific LLMs in low-resource settings. The framework integrates three core components: (i) large-scale synthetic QA data generation from expert documentation to instill foundational domain knowledge; (ii) a code-centric IR->DPO workflow that converts verified tool decks into interpretable intermediate representations (IR), performs equivalence-preserving diversification, and constructs preference pairs to directly optimize instruction compliance and code executability; and (iii) a controlled evaluation of Retrieval-Augmented Generation (RAG), showing that while RAG benefits general LLMs, it can marginally degrade the performance of already domain-aligned models.
  We demonstrate the framework by instantiating TcadGPT for semiconductor Technology Computer-Aided Design (TCAD). Using 1.5M synthetic QA pairs and an IR-driven DPO dataset, TcadGPT attains 85.6% semantic accuracy and an 80.0% syntax pass rate on SDE executability tests, substantially outperforming state-of-the-art general LLMs such as GPT-4o. To probe portability beyond TCAD, we apply the same recipe to the open-source FEM solver Elmer, observing consistent improvements in script-level success rates over general-purpose baselines. All datasets, benchmarks, and code (including P1, P2, and IR->DPO) are released for reproducibility. Together, these results suggest that the proposed framework provides a robust and reproducible path toward executable LLMs in specialized, data-scarce professional domains.

## Full Text


<!-- PDF content starts -->

A Generalizable Framework for Building Executable
Domain-Specific LLMs under Data Scarcity: Demonstration on
Semiconductor TCAD Simulation
Di Wang1, Zhenhua Wu2, Yu Liu*1, Kai Chang*2, and Shaohua Wu1
1Inspur Electronic Information Industry Co., Ltd, Beijing, China.
2Center for Quantum Matters, Zhejiang University, Hangzhou, China.
*Correspondence to: Yu Liu (liuyubj@inspur.com) or Kai Chang (kchang@zju.edu.cn).
Abstract
Scientific and engineering verticals often suffer from
data scarcity and strict executability requirements:
models must generate not only fluent text, but
also syntactically valid, tool-compilable scripts. We
present a schema-first alignment framework for build-
ing compact, executable domain-specific LLMs in low-
resourcesettings. Theframeworkintegratesthreecore
components: (i) large-scale synthetic QA data gen-
eration from expert documentation to instill founda-
tionaldomain knowledge; (ii)acode-centric IR →DPO
workflow that converts verified tool decks into in-
terpretable intermediate representations (IR), per-
forms equivalence-preserving diversification, and con-
structs preference pairs to directly optimize instruc-
tion compliance and code executability; and (iii) a
controlled evaluation of Retrieval-Augmented Genera-
tion (RAG), showing that while RAG benefits general
LLMs, it can marginally degrade the performance of
already domain-aligned models.
We demonstrate the framework by instantiating
TcadGPTfor semiconductor Technology Computer-
Aided Design (TCAD). Using 1.5M synthetic QA
pairs and an IR-driven DPO dataset,TcadGPTat-
tains 85.6% semantic accuracy and an 80.0% syntax
pass rate on SDE executability tests, substantially
outperforming state-of-the-art general LLMs such asGPT-4o. To probe portability beyond TCAD, we
apply the same recipe to the open-source FEM solver
Elmer, observing consistent improvements in script-
level success rates over general-purpose baselines. All
datasets, benchmarks, and code (including P1, P2,
and IR →DPO) are released for reproducibility. To-
gether, these results suggest that the proposed frame-
work provides a robust and reproducible path toward
executable LLMs in specialized, data-scarce profes-
sional domains.
1 Introduction
Advanced semiconductor research and development in-
creasingly rely on high-fidelity digital twins that com-
bine predictive physics with rapid, data-driven itera-
tion. Within such workflows, Technology Computer-
Aided Design (TCAD) serves as the computational
backbone, resolving tightly coupled process–device
interactions at nanometer scales and producing simu-
lation artifacts that must be both physically accurate
andtool-executable. As device dimensions continue
to shrink, process complexity escalates and physical
effect coupling proliferates [ 1,2], leading to TCAD
workflows characterized by high-dimensional param-
eter spaces, strong nonlinearity, and script-intensive
iterative setup. Together, these factors create an ur-
1arXiv:2601.10128v1  [cs.CE]  15 Jan 2026

Figure 1:Executable result produced by a
model-generated SDE deck.A planar MOSFET
structure constructed from a natural-language in-
struction and rendered by our model into an SDE
script, then executed to completion without manual
edits. Shown is the dopant distribution (DopingCon-
centration, cm−3) after mesh generation and export
(.tdr/.bnd). The example includes multiple materi-
als (Silicon, SiO 2, PolySi, Si 3N4), contact definitions
(gate/source/drain/substrate), region-specific doping
profiles, and multi-level mesh refinement. This result
demonstrates bothinstruction complianceandtool
executability, motivating the schema-first IR →DPO
alignment strategy introduced later in this paper.
gent demand for intelligent automation that reduces
manual overhead while preserving the numerical rigor
required by industrial solvers.
Recent advances in artificial intelligence (AI) have
demonstrated clear potential to assist TCAD work-
flows, primarily through surrogate modeling and op-
timization. Neural networks have been applied to
approximate electrostatic potentials, current–voltage
characteristics, and other device responses, substan-
tially reducing simulation time and computational
cost [3–5]. Physics-informed neural networks (PINNs)
further integrate governing equations into learning
frameworks, bridging data-driven models with physi-
cal constraints [ 6–8]. Bayesian and Pareto-front-basedoptimization strategies have also shown success in re-
ducing the number of expensive TCAD simulations
required for design exploration [ 9–11]. However, these
approaches focus primarily on accelerating numerical
evaluation rather than addressing a central usability
bottleneck: the authoring, debugging, and iterative
refinement of complex TCAD scripts themselves. In
industrial practice, physics-based solvers remain in-
dispensable for accuracy, yet TCAD users continue
to face steep learning curves, error-prone scripting,
and limited automation across end-to-end simulation
workflows.
In parallel, Large Language Models (LLMs) have
rapidly advanced the state of the art in Elec-
tronic Design Automation (EDA). They enable
natural-language interaction with design tools (e.g.,
ChatEDA [ 12]), automate high-quality RTL and HDL
generation (e.g., ChipGPT [ 13], RTLLM [ 14]), assist
verification and rule checking (e.g., DRC-Coder [ 15]),
and support multimodal defect analysis in fabrica-
tion (e.g., FabGPT [ 16]). Despite these successes, the
application of LLMs to TCAD remains largely unex-
plored. TCAD differs fundamentally from many EDA
tasks in that it requires strict adherence to simulator-
specific syntax, hierarchical geometry definitions, and
execution order constraints, all of which must result
intool-acceptedartifacts rather than descriptive text.
To date, the only work directly applying LLMs to
TCAD is by Nguyenet al.[ 17], who fine-tuned a trans-
former on approximately 7,000 structure input files
to generate TCAD scripts. More broadly, existing at-
tempts typically rely on narrowly constructed datasets
with limited diversity, often produced by token-level
mutation of a single template. These studies lack
standardized benchmarks and systematic comparisons
with general-purpose LLMs, leaving both effectiveness
and portability unclear. Unlike domains such as Ver-
ilog or HLS, TCAD suffers from extreme data scarcity:
vendors do not release large-scale script corpora, and
publicly available materials—user guides, textbooks,
and bundled examples—were not designed for ma-
chine learning. As a result, TCAD input decks (e.g.,
.cmd,.tdr) pose a particularly challenging testbed
for LLMs due to their nested syntax, hierarchical
structure, and tight coupling between commands and
physical intent.
2

Our perspective and goal.Rather than propos-
ing a single-task system, we introduce aschema-
first alignment frameworkfor buildingexecutable
domain-specific LLMs underdata scarcity. We instan-
tiate this framework on TCAD asTcadGPT, where
the primary objective is to translate natural-language
intent into simulator-accepted scripts with high in-
struction fidelity. The framework combines large-scale
QA synthesis from expert documentation with a code-
centric IR →DPO alignment pipeline that explicitly
optimizes for syntactic validity and execution order.
Figure 1 presents a representative SDE result gen-
erated byTcadGPTand executed without manual
intervention. To avoid over-claiming generality from a
single domain, we further include a lightweight exten-
sion study on the open-source FEM solverElmer[ 18],
using the same alignment recipe to probe portability
beyond TCAD.
2A Generalizable Schema-First
Alignment Framework
Design principles.Our framework targets domains
where: (i) public corpora are scarce and fragmented;
(ii) solutions must yieldtool-executableartifacts (e.g.,
simulator scripts); and (iii) instruction-following fi-
delity is critical. It comprises three tightly coupled
stages:
•Knowledge acquisition via QA synthe-
sis.Two complementary pipelines transform
expert-curated documentation intoAlpaca-style
instruction–response pairs at scale:Pipeline 1
(segment-based QA) emphasizes broad coverage
and phrasing robustness;Pipeline 2(keyword-
guided QA) targets fine-grained concepts, com-
mands, and equations for higher precision. To-
gether they build domain priors for reasoning and
syntax awareness.
•Schema-first code alignment via IR →DPO.
Verified executable decks are parsed into anIn-
termediate Representation (IR)that preserves
semantics (dimension, geometry Boolean order,
contacts, doping, mesh/export). IR enables (a)equivalence-preserving diversificationand (b)de-
terministic instruction rendering. We then build
DPO preference pairs with single-factor, inter-
pretable violations (numeric scale, order/proce-
dure, export omissions) todirectly optimizein-
struction compliance and syntactic validity.
•Retrieval as an optional, controlled compo-
nent.RAG can substantially lift generic LLMs
by grounding them in domain text, but maydi-
lutepriors for already specialized compact mod-
els. Our framework treats RAG as optional and
schema-constrained—to be used when the base
is generic or when retrieval can be structured to
avoid lexical overfitting.
Outcomes.The framework yields compact models
capable of emittingtool-compilablecode and stable,
high-utility QA responses. In the sequel, we demon-
strate the framework on TCAD viaTcadGPTand
include a lightweight extension on the open-source
FEM solverElmerto probe portability.
3TcadGPT: Domain-Adapted
LLM for TCAD
The rapid progress of large language models (LLMs)
has enabled breakthroughs across general-purpose and
scientific applications, giving rise to the emerging field
of LLMs for Science (LLM4S). However, their applica-
bilitytohighlyspecializeddomainssuchasTechnology
Computer-Aided Design (TCAD) remains extremely
limited. In scientific question answering, ETH Zurich
introduced a multilingual LLM trained on scientific lit-
erature and Wikipedia [ 19]. While effective on general
QA tasks, it lacks support for code understanding or
procedural file generation. Other domain-specialized
LLMs [20–22] have targeted areas like quantum chem-
istry, EDA scripting, or chip placement. However,
these models typically rely on small-scale datasets,
high-level abstraction, or prompt-only tuning. For
instance, the ORCA-LLM [ 20] focuses on generating
input scripts for quantum chemistry simulations, but
only covers limited domain logic with minimal instruc-
tion diversity. LLM4EDA [ 21] surveys applications in
3

hardware design, but lacks tool-specific depth. Chip-
NeMo [22] adapts LLMs for EDA tasks, yet does not
target the code structural fidelity or low-level syntax
required in TCAD environments. By comparison, our
work explicitly targets both command syntax and
domain semantics in a tightly scoped, low-resource
setting.
This gap stems primarily from the scarcity of pub-
licly available training data and the intricate domain-
specific syntax, modeling principles, and simulation
workflows required in TCAD.
Unlike other hardware design languages such as Ver-
ilog or RTL, which benefit from an active open-source
ecosystem and abundant datasets via platforms like
GitHub, TCAD simulation scripts are rarely shared.
MostcommercialTCADtools—suchasSentaurusPro-
cess, Sentaurus Device, and Garand—are proprietary,
and the input files used for real-world semiconductor
simulations often contain sensitive information tied
to process technology, making them strictly confiden-
tial. Consequently, only limited resources such as
user guides or basic training examples are accessible
for model development. This data scarcity presents
a significant bottleneck in enabling LLMs to reason
effectively in the TCAD domain.
As a result, existing LLMs perform poorly on
TCAD-related tasks. Our preliminary experiments
reveal that even state-of-the-art models such as
DeepSeek V3 fail to answer code-intensive questions
reliably, with limited utility in practical scenarios.
To address these challenges, we proposeTcadGPT,
a compact (8B-parameter) domain-specialized LLM
tailored for semiconductor researchers working with
TCAD tools.
Our work makes the following contributions:
•Large-scale synthetic dataset.We design
two complementary synthetic data generation
pipelines that transform publicly available TCAD
manuals, textbooks, and training documents into
a large-scale instruction–response corpus contain-
ing over1.5 millionAlpaca-format samples un-
der non-commercial academic use.
•Reproducible benchmark with reference
answers.We construct a comprehensive264-
question TCAD benchmark suitecoveringphysical modeling, simulation logic, and tool com-
mand syntax. Each question is authored with
an expert-providedreference answer, and model
outputs are graded using a standardized rubric
that only accounts for acceptable equivalent ex-
pressions, enabling transparent and reproducible
comparison across models.
•IR→DPO alignment pipeline.We introduce
acode-centric IR →DPO alignment work-
flowthat extracts intermediate representations
from verified TCAD decks, performs equivalence-
preserving diversification, and constructs struc-
tured preference pairs with controlled instruction
violations. This stage builds upon the QA-tuned
TcadGPT-P1&2model and directly optimizes
instruction-following fidelity and code executabil-
ity, yielding the finalTcadGPT-P1&2+DPO
checkpoint used in Section 7.2.
•Model performance and resource efficiency.
The QA-tunedTcadGPT-P1&2model achieves
a benchmark accuracy of85.6%on the 264-
question QA suite, significantly outperforming
GPT-4o, DeepSeek V3, and R1 baselines. The
DPO-alignedTcadGPT-P1&2+DPOfurther
demonstrates strong code executability, achieving
an 80% syntax pass rate in tool-level validation
while remaining compact enough for on-premise
inference on a single GPU.
Together, these contributions establish TcadGPT
as a reproducible, domain-adapted foundation model
for intelligent TCAD automation and digital-twin
development.
4Experiment Setup and Bench-
mark
To rigorously evaluate the capabilities ofTcadGPT,
weconstructedabenchmarktailoredtoTCAD-related
tasks, designed to reflect the real-world challenges
faced by semiconductor researchers. The benchmark
was developed with three core principles: (1) ensur-
ing coverage of core functional modules across TCAD
4

Representative TCAD Question-Answer Examples
Q1 (Conceptual): What is the high-field mobility satu-
ration effect?
A: High-field mobility saturation refers to the phenomenon
where the drift velocity of carriers approaches a limit under
high electric fields.
Q2 (Syntax): How to control oxidation growth rate in
Sprocess?
A: Use commands pdbSet Diffuse MinGrowthStep and
MaxGrowthStep to control the rate.
Q3 (Math/Physics F ormula): What is the Shockley-
Read-Hall (SRH) recombination formula?
A:
U=pn−n2
i
τp(n+n1) +τn(p+p1)Q4 (Instruction Parsing): How to define a boundary
condition in Sentaurus Device?
A: Use the command BoundaryCondition (Electrode,
Type, Value) .
Q5 (Operation Instruction): How to load data in Sen-
taurus Visual?
A: Use the menu path File → Load to import result files
such as TDR.
Q6 (Simulation Configuration): How to adjust simu-
lation accuracy in Sdevice?
A: Set the parameter Math{Numerics{DampedNewton{
Tolerance=1e-6}}} for higher precision.Figure 2: Representative examples from the TCAD benchmark, covering multiple question types including
operational, conceptual, and model-related. These illustrate the diversity and depth of domain understanding
required for high-performance models.
toolchains, (2) including diverse question types rang-
ing from conceptual understanding to exact syntax
construction, and (3) grounding questions in realistic
user scenarios. The final set consists of 264 questions
carefully curated and reviewed by domain experts to
ensure both technical correctness and representative-
ness.
Figure 3 summarizes the question type distribution.
The benchmark spans six major TCAD functional
modules: general physical models, general simulation,
sprocess ,sde,sdevice, and svisual. These mod-
ules encompass both conceptual modeling knowledge
and tool-specific operations, with over 65% (173 out of
264) of the questions focusing on instruction parsing,
tool operation, or syntax construction—areas identi-
fied as the most significant pain points for researchers.
All responses were evaluated against expert-
providedreferenceanswersusingastandardizedrubric
toensureobjectiveandconsistentscoring. Specifically,
each benchmark item is authored by a domain expert
together with an unambiguous expected answer (e.g.,
a specific command, parameter meaning, formula, or
a short canonical explanation), so that correctness can
be verified without relying on subjective judgment.
The rubric is used only to handleacceptable equivalent
expressions(e.g., formatting differences, synonyms, or
Physical Model
36
(13.6%) Simulation
66
(25.0%)
SDE33
(12.5%)
SProcess31
(11.7%)
SDevice58
(22.0%)SVisual
40
(15.2%)Figure 3: Distribution of question types in the TCAD
benchmark. The six major categories—Physical
Model, Simulation, SDE, SProcess, SDevice, and SVi-
sual—cover all key simulation and modeling tasks
encountered in real-world workflows.
5

minor rewording) while preserving the same technical
content. For instruction-oriented questions, outputs
are marked correct only if they provide directly us-
able commands or code snippets conforming to the
expected tool syntax; for principle-oriented questions,
outputs are marked correct only if they match the
reference conclusion with sufficient completeness for
practical TCAD use.
Representative examples from the benchmark are
showninFigure2, illustratingthediversityofquestion
types, including conceptual definitions, syntax-level
commands, mathematical formulations, and simula-
tion configurations.
Retrieval-Augmented Generation Setup
To investigate whether external knowledge retrieval
improves model performance, we implemented a
Retrieval-Augmented Generation (RAG) pipeline us-
ing theLangFlowplatform.1The pipeline combines
offline document preparation and online query pro-
cessing. Domain documents—including Synopsys
Sentaurus user guides, TCAD textbooks, and offi-
cial training materials—were preprocessed into man-
ageable chunks using a custom splitter and embed-
ded with Ollama Embedding (nomic-embed-text) .
These embeddings were stored in Chroma DB for high-
speed semantic search.
At inference time, user queries are first translated
into English by a lightweight translation model to
align with the language of the domain corpus. Re-
trieved document chunks are then parsed and incor-
porated into a structured contextual prompt, which is
finally passed to the target LLM for response genera-
tion. This setup provides a consistent way to compare
the standalone performance of different LLMs against
their retrieval-augmented counterparts.
Models for Evaluation
We evaluated two complementary aspects of model
capability: (1)QA-level understandingand (2)
code-level executability. For QA-level evalua-
tion, we used LLaMA 3.1 8B as the base model and
1https://github.com/langflow-ai/langflowfine-tuned it using our synthetic QA corpus to ob-
tain three variants:TcadGPT-P1(trained only
on data from Pipeline 1, i.e., segment-based QA
generation),TcadGPT-P2(trained only on data
from Pipeline 2, i.e., keyword-guided QA genera-
tion), and the combined modelTcadGPT-P1&2
(trained on the union of both pipelines). These vari-
ants were compared with general-purpose models in-
cluding DeepSeek V3 ,DeepSeek R1 , and GPT-4o. To
study the impact of retrieval augmentation, we ad-
ditionally tested LLaMA 3.1 8B + RAG ,DeepSeek V3
+ RAG, andTcadGPT-P1&2 + RAG. This design
enables a systematic comparison between (i) general-
purpose versus domain-specialized LLMs, and (ii)
standalone versus retrieval-augmented setups.
For code-level executability evaluation, we fur-
ther assessed theDPO-aligned modelTcadGPT-
P1&2+DPO, which extends the QA-tuned check-
point with the IR →DPO alignment dataset. Unlike
the QA benchmark that measures scientific correct-
ness and descriptive reasoning, this evaluation focuses
on syntactic validity and instruction-following preci-
sion, validated directly within the Sentaurus sde -S
environment.
Reproducibility and Open Resources.To en-
sure transparency and reproducibility, all components
of this work—including the 264-question QA bench-
mark, the 20-instruction SDE executability test set,
the fine-tuned model weights, and the full data gener-
ation pipelines (P1,P2, andIR →DPO)—are pub-
licly released on GitHub.2The repository provides
scripts for dataset synthesis, model fine-tuning, and
benchmark evaluation, enabling direct replication and
further research on TCAD-oriented LLMs.
5QA Training Data Generation
DuetothescarcityofpubliclyavailableTCAD-related
code and documentation, we curated our training cor-
pus from officially accessible resources such as Sen-
taurus User Guides, TCAD textbooks, training man-
uals, and bundled sample files. All documents were
2https://github.com/wddddds1/TcadGPT
6

obtained from public sources and used under non-
commercial academic conditions.
User Guides
63.4%
(968,082)
Training Docs13.4%
(204,858)Textbooks23.2%
(354,536)Pipeline 2
48.4%
Pipeline 1
14.9%
Pipeline 2
5.8%Pipeline 1
7.6%Pipeline 2
23.2%
Figure 4: Composition of the synthetic training
dataset by data source (User Guides, Training Docs,
Textbooks) and data generation pipeline (Pipeline 1,
Pipeline 2). The outer ring shows the total contribu-
tion of each source, while the inner ring distinguishes
the two QA generation pipelines.
To systematically construct a high-quality instruc-
tion–response dataset in Alpaca format[ 23], we de-
signed two complementary pipelines to maximize both
the coverage and diversity of domain knowledge. Fig-
ure 4 visualizes the detailed breakdown of our training
set, which totals over 1.5 million QA pairs. The outer
ring represents the relative contributions of each docu-
ment type, while the inner ring further subdivides the
dataset by generation pipeline. Notably, the majority
of samples originate from user guides and textbooks,
with Pipeline 2 accounting for over 80% of the total
corpus.
To illustrate the nature of the generated data, Fig-
ure 5 shows typical JSON-formatted QA pairs pro-
duced by our pipeline.
Pipeline 1: Segment-Based QA Generation
We first converted source PDFs into text using open-
source OCR tools, followed by segmentation based
on structural markers such as section titles and para-
graph breaks. These segments were passed to an LLM
(DeepSeek V3) to generate instruction-following QA
pairs. To mitigate the model’s phrasing sensitivity,
Examples of JSON QA Pairs
{
"instruction": "What components are included in the total resistance calculation of
metal interconnects?",
"input": "",
"output": "The total resistance of metal interconnects includes all serially
connected conductor segments. Specifically in the example:
-resistance of the metal1 segment
-resistance of the metal2 segment
-resistance of the via (connecting metal1 and metal2)
These elements together form the overall interconnect impedance characteristics
between cont1 and cont2."
},
{
"instruction": "What is the purpose of the ’moderesistance’ command?",
"input": "",
"output": "The ’moderesistance’ command sets the simulator to operate in a purely
resistive network mode for interconnect structures. In this mode:
-capacitive and inductive effects are ignored
-focus is solely on DC impedance characteristics of the conductor network
-subsequent ’solve’ commands will compute based on this mode"
}Figure 5: Examples of JSON QA Pairs used for fine-
tuning.
each question was paraphrased into ten variants using
a separate LLM prompt, enhancing robustness. This
process produced 340,000 samples used to train the
TcadGPT-P1model.
Algorithm 1Pipeline 1: Segment-Based QA Gener-
ation
1:Input:Set of PDF documentsD
2:foreach documentdinDdo
3:T←OCR(d)
4:S←Segment(T)
5:foreach segmentsinSdo
6:q, a←LLM.generateQA(s)
7:Q′←LLM.paraphrase(q, n= 10)
8: Add( q, a)and each( q′, a)inQ′to dataset
9:end for
10:end for
Pipeline 2: Keyword-Guided QA Generation
To achieve greater coverage and domain precision,
we designed a second pipeline that extracts technical
keywords (e.g., material parameters, model names,
equations) from each segment and uses them to guide
QA generation. For each (segment, keyword) pair,
the LLM is prompted to synthesize a relevant and
focused QA instance. This method yielded over 1.2
million diverse samples and was used to train the
TcadGPT-P2model.
7

Algorithm 2Pipeline 2: Keyword-Guided QA Gen-
eration
1:Input:Set of PDF documentsD
2:foreach documentdinDdo
3:T←OCR(d)
4:S←Segment(T)
5:foreach segmentsinSdo
6:K←LLM.extractKeywords(s)
7:foreach keywordkinKdo
8:q, a←LLM.generateQA(s, k)
9:Add(q, a)to dataset
10:end for
11:end for
12:end for
Prompt Engineering and Training SetupTo
ensure the quality, coverage, and reliability of our syn-
thetic instruction–response pairs, we designed a multi-
stage prompt engineering framework. For Pipeline 1,
the prompt instructed the LLM to extract all techni-
cally meaningful elements from each document seg-
ment, with an emphasis on simulation commands,
modeling parameters, and domain logic. It enforced
strict formatting (Alpaca-style JSON), discouraged
shallow or duplicated questions, and emphasized con-
cise, operational answers including formulas or code
when applicable. To improve linguistic robustness,
a separate prompt was used to generate 10 diverse
paraphrases per question.
Pipeline 2 employed a two-step prompt structure.
The first extracted all relevant keywords—such as
model names, command names, or physical princi-
ples—from each segment. The second prompt used
each (segment, keyword) pair to generate QA exam-
ples, with different styles based on the keyword type
(e.g., definition vs. instruction). The prompt enforced
structural consistency, avoided generic references (e.g.,
“as mentioned above”), and required detailed, example-
driven answers. An overview of the prompt types and
their intended objectives is summarized in Table 1.
The datasets generated from the two pipelines con-
sisted of approximately 340,000 samples (Pipeline 1)
and 1,200,000 samples (Pipeline 2). Given the lim-
ited performance of the initial 340K-sample dataset
from Pipeline 1, Pipeline 2 was proposed to enhanceTable 1: Overview of Prompt Types and Objectives
Prompt Type Objective and Key Con-
straints
Segment QA
GenerationExtract domain logic; enforce
Alpaca format; emphasize com-
mand, equation, or process-level
questions; maximize coverage;
Chinese-only.
Paraphrasing
for QAGenerate 10 diverse phrasings
per question to improve model
robustness; strictly enforce non-
redundancy and valid list for-
mat.
Keyword Ex-
tractionIdentify all technical concepts
(commands, parameters, mod-
els); enforce strict output JSON
structure; ignore non-technical
pages.
QA from Key-
wordsStyle questions by keyword type;
prohibit contextual references;
ensure concise yet informative
answers with examples.
coverage and diversity, resulting in a larger and more
fine-grained corpus of 1.2M samples. Model train-
ing was conducted on 8×A100 80GB GPUs using the
LLaMA 3.1 8B model and LLaMA-Factory with Deep-
Speed Stage 3. Full-parameter fine-tuning was applied
with a maximum sequence length of 4096 tokens, a co-
sine learning rate decay starting from1 ×10−5, batch
size 4 per GPU with gradient accumulation of 8, and
5 total epochs. All training used bf16precision, with
2% of the dataset held out for validation.
6Code Augmentation and DPO
Dataset Construction
QA-style supervision provides broad domain knowl-
edge but rarely teaches a model tofollow instruc-
tions preciselywhen generating executable TCAD
scripts. Models trained only on QA or unstructured
code often produce nearly-correct yet unusable out-
8

Verified Source Code Input
DPO dataset(I, COT, Chosen code, Rejected code)Extract IR Flatten & Canonicalizebranch explosionnormalize units/numerical valuesunify aliases…Equivalence-Preserving Diversificationjitter numeric valuesswap commutable Boolean opstoggle optional steps…produces ~10×diversified IR variantsGenerateCOTand instructionsfor each IRRender Chosen Code from IRGenerate Rejected Code Variantsnumeric violationsprocedural violations (swap Boolean orders, missing steps)cross-sample impostors…Instruction Refinement with LLMCOT Refinement with LLMSerialize to DPO FormatIRextraction& DiversificationCode and Instruction GenerationDPO Dataset ConstructionFigure 6:Overview of the IR →DPO alignment
pipeline.Starting from verified TCAD source decks,
the pipeline extracts a schema-level intermediate rep-
resentation (IR), performs equivalence-preserving di-
versification, renders canonical instructions and refer-
ence code, and constructs preference pairs with con-
trolled numeric and procedural violations for Direct
Preference Optimization (DPO).
puts—commands in the wrong order, missing export
steps, or parameters off by an order of magnitude. To
bridge this gap, we introduce acode-centric align-
ment pipelinethat converts a small set of verified
decks into a large, interpretable dataset for Direct
Preference Optimization (DPO). Figure 6 illustrates
the complete IR →DPO alignment workflow, including
IR extraction and diversification, instruction and code
rendering, and the construction of preference pairs
with controlled violations.
IR construction and equivalence-
preserving diversification
Starting from ∼800 official Sentaurus
sde/sprocess /sdevice decks, we extract a
lightweightIntermediate Representation (IR)that
captures the essential simulation facts: dimensionality
and up-direction, material list, region geometry
with ordered Boolean operations, contact definitions,
doping specifications, and mesh/export directives.
Each record also contains a compactfact card(region
counts, Boolean-order string, contact presence,
expected outputs), which formalizes the invariants
that must not change across transformations.
(sdegeo:create-rectangle"Si" (0 0) (1 0.5))(sdedr:define-refinement-size"global" 10 10 0.0001)…{"geometry": [{"material": "Si" ,"type": "rectangle" ,"x1": 0.0,"y1": 0.0,"x2": 1.0,"y2": 0.5}],"refinement": {"scope": "global","nx": 10,"ny": 10,"min_step": 0.0001}}…"refinement": {"scope": "global","nx": 10,"ny": 10,"min_step": 0.0001}…"refinement": {"scope": "global","nx": 10,"ny": 10,"min_step": 0.0002}Create a rectangular silicon region from (0, 0) to (1, 0.5), then apply a global mesh refinement with parameters 10,10,0.0001First define the silicon rectangle, then apply the global mesh refinement with the specified parameters.{"instruction": "Create a Si rectangle (0,0)-(1,0.5) and apply global refinement [10,10,0.0001]. " ,"chosen": ” …(define-refinement-size global 10 10 0.0001)… " ,“rejected”: “…(define-refinement-size global 10 10 0.0002)… "}Verified Source Code 
Extracted IRChosen IR
Rejected IRInstructionCOT
DPO datasetFigure 7:Example of IR-driven DPO pair con-
struction.A verified TCAD source snippet is parsed
into an intermediate representation (IR), from which
a canonical natural-language instruction, a chosen
(instruction-compliant) code sample, and a rejected
variant with a single controlled violation (e.g., a nu-
meric mismatch) are deterministically generated.
We firstflattendecks by unifying aliases, standard-
izing numeric formats, and canonicalizing command
orders where tool semantics allow, so comparisons
reflect semantics rather than style. We then perform
equivalence-preserving diversificationon the IR
to enlarge coverage without altering physical mean-
ing: mildly reordering commutable Boolean/geometry
operations, applying small unit-aware numeric jitters,
canonicalizing aliases, and toggling optional mesh/ex-
port statements. This yields roughly a ×10 expansion
while remaining executable and semantically faithful.
Instruction rendering, validation, and
DPO packaging
Each diversified IR is deterministically rendered into
acanonical natural-language instructionand a ref-
erence code snippet. Anumeric whitelistrestricts
visible constants to those present in the IR or refer-
ence code. To make the intended reasoning explicit
without verbosity, we attach a concise, symmetric
chain-of-thought (COT)that enumerates geome-
try, doping, mesh, and export steps.
To explicitly optimize instruction following, we
9

Algorithm 3IR-to-DPO Alignment Workflow
1:Input:Verified TCAD decksC
2:foreach deckc∈ Cdo
3:IR←ExtractIR(c)
4:// dimension, geometry, doping, mesh
5:IR flat←Flatten(IR)
6:{IR i} ←Diversify(IR flat)
7:// order swaps, numeric jitters, toggles
8:foreachIR ido
9:I←RenderInstruction(IR i)
10:// numeric whitelist
11:c⋆←RenderCode(IR i)
12:// chosen code
13:{˜c j} ←MakeRejected(IR i, c⋆)
14:// controlled violations
15:ValidatePairs(I, c⋆,˜cj)
16:SerializeDPO(I, c⋆,˜cj)
17:end for
18:end for
buildDPO preference pairs. For every instruction,
the validated reference code is thechosensample. We
synthesize severalrejectedvariants that intentionally
violate a single interpretable dimension, including (i)
numeric deviations(near/step/mag/wide jitters or
×10/×0.1unit-scale errors), (ii)procedural violations
(e.g., swapping Boolean order, moving contact def-
initions before refinement, omitting build-mesh or
export), and (iii)cross-sample impostors(valid code
from another record with conflicting facts). A two-
stage checker validates negatives: non-subset multiset
tests for numbers (with weak-accept fallbacks) and
IR diffs/targeted regex for structure.
Finally, we serialize samples into JSON with sta-
ble IDs. All pairs share the same instruction and
COT—the only difference is inside the fenced code
block—so the preference signal cannot be confounded
by stylistic artifacts. We also produce five para-
phrased instruction variants (research/engineering/-
concise/stepwise/conversational) filtered by the same
whitelist for robustness.
Illustrative example.Figure 7 provides a concrete
example of the IR-driven DPO construction process.Starting from a verified TCAD source snippet, the
extracted IR is used to render a natural-language
instruction, a reference (chosen) code sample, and a
rejected variant that differs by a single, interpretable
violation.
Example of a DPO Pair (abbreviated)
Instruction:Construct a rectangular region on
a two-dimensional silicon substrate from coor-
dinates (0, 0, 0) to (1, 1, 0). Define two global
rectangular windows and place a virtual contact
point at (0, 0.5, 0). Perform boron doping with
a concentration of 9.8e+12 inside the windows,
then refine the global mesh with parameters [10,
10, 0.0001, 1, 1, 0.0001]. Finally, build the mesh
and export both BND and TDR files for use in
SDevice.
Chosen (excerpt):
(sdedr:define-refinement-size "global" 10 10←-
→0.0001 1 1 0.0001)
...
(sdeio:save-tdr-bnd "n@node@.tdr" "n@node@.←-
→bnd")
Rejected (excerpt):
(sdedr:define-refinement-size "global" 10 10←-
→0.0001 1 1 0.0002)
...
(sdeio:save-tdr-bnd "n@node@.tdr" "n@node@.←-
→bnd")
7Experiment Results and Abla-
tion Study
To comprehensively evaluate the effectiveness of
TcadGPT, we designed a two-tier evaluation frame-
work addressing bothsemantic understandingand
code executability.
(1) QA-level Evaluation.We constructed a cu-
rated benchmark of 264 expert-verified questions span-
ning six key TCAD subdomains—General Physical
Model,Simulation,SDE,SProcess,SDevice,
andSVisual. This benchmark measures the model’s
ability to comprehend physical principles, interpret
tool syntax, and produce scientifically correct and
10

usable responses. Each answer was manually scored
based on scientific correctness and syntactic validity,
as described in Section 4.
(2) Code-level Executability Evaluation.To
further assess the impact of our IR →DPO alignment
workflow on executable code generation, we created
a separate test set consisting of 20 unseen natural-
language SDE instructions automatically generated
from the IR renderer. For each instruction, the model
produced one or multiple candidate scripts, which
were directly validated using sde -Ssyntax check-
ing in the official Sentaurus Structure Editor. We
report two metrics: (i)Pass@1, the proportion of
single-sample generations passing syntax check, and
(ii)Pass@3, the proportion passing when up to three
samples are generated per instruction. We also distin-
guish betweendirect passesand cases requiring simple
placeholder replacement (e.g., substituting @height@
with numeric constants).
This dual-level evaluation provides complemen-
tary perspectives: the QA benchmark quantifies the
model’s reasoning and instruction-following accuracy,
whichreflects the model’s understandingof knowledge,
while the SDE test set measures its ability to produce
tool-valid, executablecode under realistic conditions,
which reflects the application of knowledge by the
model.
7.1 QA-level Benchmark Results
7.1.1 Overall Performance
Figure 8(a) summarizes the overall accuracy achieved
by each model across the entire benchmark (264 ques-
tions). Among all models,TcadGPT-P1&2(trained
with combined data pipelines) achieved the highest
accuracy of 85.6%, demonstrating significant superi-
ority over general-purpose LLM baselines, including
GPT-4o[ 24] (46.6%), DeepSeek R1[ 25] (52.3%), and
DeepSeek V3[ 26] (50.0%). Additionally, we observed
substantial performance improvements through re-
trieval augmentation (RAG), with DeepSeek V3 im-
proving from 50.0% to 67.0% and LLaMA 3.1 8B
improving from 24.6% to 58.0%. However, notably,
adding RAG toTcadGPT-P1&2reduced the accu-racy from 85.6% to 64.0%, indicating that an overly
broad retrieval context may dilute domain-specific
fine-tuning effectiveness.
7.1.2 Subdomain-Level Analysis
To gain deeper insight, we evaluated model perfor-
mance on each of the six TCAD subdomains sepa-
rately (Figure 8(b)).
General Physical Model:TcadGPT-P1&2
achieved near-perfect performance (93.9%), signifi-
cantly surpassing general models like GPT-4o (90.9%)
and DeepSeek V3+RAG (87.9%). The noticeable gap
betweenTcadGPT-P1(10.6%) andTcadGPT-P2
(92.4%) on theGeneral Physical Modelcategory
reflects a difference in data coverage rather than op-
timization stability.TcadGPT-P1was primarily
trained on segment-level extractions that were biased
toward tool usage and script patterns; as a result, it
frequently attempted to answer conceptual physics
questions by emitting procedural instructions or simu-
lator commands, which were graded as incorrect under
our rubric. In essence,TcadGPT-P1became overly
code-prone—responding to general physics questions
with instruction-like outputs instead of descriptive
reasoning. Pipeline 2, by contrast, was generated
via keyword-guided QA synthesis explicitly targeting
physical mechanisms, material parameters, and mod-
eling assumptions, which allowed it to produce high-
accuracy descriptive answers rather than instruction-
style code completions.
Simulation:TcadGPT-P2andTcadGPT-
P1&2both led performance (86.1%), outperforming
RAG-enhanced general models (DeepSeek V3+RAG
at 80.6%) and significantly exceeding baseline models
such as GPT-4o (61.1%) and LLaMA 3.1 8B (36.1%).
SDE: In tasks related to mesh and geometry se-
tups,TcadGPT-P2andTcadGPT-P1&2achieved
the best results (93.9%), demonstrating remarkable
superiority over general models, with the best gen-
eral model (LLaMA 3.1 8B+RAG) reaching only
60.6%. The sharp contrast highlights the complex,
tool-specific syntax that domain-adapted models are
uniquely suited to handle.
SProcess: Similarly,TcadGPT-P1&2(83.9%)
andTcadGPT-P2(80.6%) significantly outper-
11

LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG0102030405060708090100Accuracy (%)
24.6%58.0%
52.3%
50.0%67.0%
46.6%
28.0%84.1%85.6%
64.0%a) Overall Accuracy (%)
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
42.4%66.7%90.9%
81.8%87.9%90.9%
10.6%92.4% 93.9%
80.3%b) Physical Model
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
36.1%69.4%80.6%
75.0%80.6%
61.1%
16.7%86.1% 86.1%
63.9%Simulation
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
24.2%60.6%
30.3% 30.3%57.6%
21.2%42.4%93.9% 93.9%
51.5%SDE
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
3.2%54.8%
16.1%
6.5%38.7%
12.9%48.4%80.6%83.9%
51.6%SProcess
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
17.2%50.0%46.6%50.0%63.8%
43.1%44.8%81.0%84.5%
62.1%SDevice
LLaMA 3.1 8B
LLaMA 3.1 8B + RAGDeepSeek R1 DeepSeek V3
DeepSeek V3 + RAGGPT-4o
T cadGPT Pipeline1 T cadGPT Pipeline2T cadGPT Pipeline1&2
T cadGPT Pipeline1&2 + RAG020406080100
12.5%45.0%
17.5%25.0%55.0%
12.5%15.0%67.5% 67.5%
60.0%SVisualFigure 8: Model performance across all six TCAD subdomains and the overall benchmark (264 questions).
Left: overall accuracy comparison across all models. Right: subdomain-level accuracy onGeneral Physical
Model,Simulation,SDE,SProcess,SDevice, andSVisual.
formed all general-purpose counterparts, with the
best-performinggeneralmodel(LLaMA3.18B+RAG)
at 54.8%. This indicates the importance of domain-
specific training, particularly in complex simulation
tasks such as oxidation and diffusion.
SDevice:TcadGPT-P1&2andTcadGPT-P2
consistently outperformed other models, achieving
accuracies of 84.5% and 81.0%, respectively. General-
purpose models struggled substantially, with GPT-4o
(44.8%) and DeepSeek V3+RAG (63.8%) underscor-
ing the difficulty in correctly parsing detailed device
simulation commands and parameters without spe-
cialized training.
SVisual: Visualization tasks posed similar chal-
lenges, yetTcadGPT-P2andTcadGPT-P1&2
delivered outstanding accuracy (67.5%), surpassing
general baselines such as DeepSeek V3+RAG (55.0%)
and GPT-4o (15.0%). The data suggests that visu-
alization syntax, while slightly more intuitive, stillgreatly benefits from specialized domain data.
7.1.3 Ablation Study
Our ablation study compares several configurations
to examine the contributions of RAG and domain-
specific fine-tuning separately. General-purpose mod-
els significantly improved when supplemented with
retrieval (e.g., LLaMA 3.1 8B from 24.6% to 58.0%
and DeepSeek V3 from 50.0% to 67.0%), validating
retrieval augmentation’s effectiveness in providing nec-
essary contextual grounding. Conversely, adding re-
trieval to the already specializedTcadGPT-P1&2
reduced overall performance from 85.6% to 64.0%,
suggesting the carefully curated training data pro-
vided by Pipelines 1 and 2 sufficiently covers domain
knowledge, and overly broad retrieved contexts may
introduce noise rather than valuable insights.
Moreover, comparingTcadGPT-P1and
TcadGPT-P2individually highlighted that struc-
12

tured keyword-based QA generation (Pipeline 2)
dramatically enhanced the model’s capacity to
understand and correctly respond to intricate TCAD-
specific queries across all modules, underscoring the
importance of targeted and structured domain data.
These results collectively demonstrate that a tai-
lored domain-specific training approach, especially
structured and keyword-guided data augmentation,
is vital for LLM effectiveness in highly specialized,
low-resource domains like TCAD.
SummaryPipeline 2 delivers consistent and sizable
gains across both conceptual and code-intensive tasks,
substantially outperforming all baselines. These im-
provements highlight the value of targeted prompt
design and keyword-guided coverage when synthesiz-
ing domain-specific QA data.
Our results further show that domain-specific
fine-tuning is a stronger foundation than retrieval
alone: RAG markedly lifts general-purpose models,
yet naïvely adding RAG to the already specialized
TcadGPTdegrades accuracy. This suggests that un-
structured retrieval can dilute learned domain priors;
future work should investigate constrained or schema-
aware retrieval to complement specialization without
introducing noise.
7.1.4 Error Analysis
We conducted a qualitative analysis on failure cases
ofTcadGPT-P2to identify remaining challenges.
Three main categories of errors were observed:
(1) Invalid References:Some answers contained
hallucinated citations to equations or figures not
present in the context. These were often inherited
from training passages and incorrectly retained during
generation. For example:Invalid Equation Reference
User Query:What are the equations cor-
responding to transient simulation and small-
signal analysis?
Model Answer:The equation correspond-
ing to transient simulation isEquation 12,
which describes the basic transient formula-
tion; the small-signal analysis corresponds to
Equation 48, describing the basic principle
of small-signal behavior.
In this case, the model refers to "Equation 12" and
"Equation 48" even though neither is defined in the
input context. We introduced stricter prompts to
suppress such false references, which led to modest
improvements, but some cases still persist.
(2) Command Hallucination and Misunder-
standing:The model sometimes generates incorrect
or fabricated command syntax, particularly for less
common tasks or tools with sparse training coverage.
Consider the following example related to dopant def-
inition inSprocess:
Hallucinated Command in Sprocess
User Query:How to define a new dopant
species in Sprocess?
CorrectAnswer:Usethecommand implant
species=<name> <material> and optionally
specify imp.table to define the implant ta-
ble. Use the damageflag to enable damage
calculation.
Model Answer:In Sprocess, use the
dopant command: dopant add <name>
<new dopant name> <species> <existing
dopant name>
Here, the model invents a non-existent dopantcom-
mand, likely duetopattern confusionor partialphrase
overlap from other simulator languages.
(3) Tokenization and OCR Noise:Some errors
stemmed from training data issues such as OCR ar-
tifacts, broken LaTeX equations, or split command
lines. Although we applied cleanup heuristics, their
effects remain localized and can interfere with parsing
13

or token alignment.
Despite these challenges, TcadGPT-P2 consistently
outperforms baseline models in structural fidelity and
domain-specific syntax. Future work may benefit
from integrating simulation-aware validators, struc-
tured retrieval, or Abstract Syntax Tree (AST) level
constraints to further mitigate hallucinations.
7.1.5Analysis of RAG-Induced Performance
Degradation
Although retrieval-augmented generation (RAG) is
generally beneficial for open-domain tasks, we ob-
served that adding RAG toTcadGPT-P1&2re-
sulted in a notable drop in performance. To clarify
thisphenomenon, weanalyzearepresentativeexample
from our benchmark.
Example (Benchmark ID 78): Question
Question:What are the equations correspond-
ing to transient simulation and small-signal AC
analysis, respectively?
TcadGPT-P1&2
Transient simulation uses a time-dependent differ-
ential equation describing the device’s dynamic
behavior, in the form
d
dtq(t) +f(t) = 0;
small-signal AC analysis adopts a frequency-
domain linearized model, such as the admittance
matrix
Y=A+jωC,
which is used to compute device responses under
small-signal excitation.
TcadGPT-P1&2+RAG
In the given context, transient simulation corre-
sponds to equation 36, and small-signal AC analy-
sis corresponds to equation 38.DeepSeek V3 + RAG (for comparison)
Basedontheprovidedcontext, transientandsmall-
signal AC analyses correspond to the following
formulas.
Small-Signal AC Analysis:The admittance
matrix is described as
δI=Y δV(Eq. 22),
where δVandδIare the complex voltage and
current vectors at selected nodes.
...
This comparison clearly illustrates why RAG may
degrade the performance of a compact domain-specific
model. The retrieved documents included literal equa-
tion indices (e.g., “Eq. 36”, “Eq. 38”) from TCAD
manuals, which misled the 8B model intocopying nu-
meric referencesinstead of reasoning about the phys-
ical meaning of the equations. Because RAG directly
injects unfiltered text segments without semantic ab-
straction, a smaller model lacking strong reasoning
capacity tends tooverfit to lexical cuesrather than
synthesizing physically valid expressions.
In contrast, the standalone fine-tunedTcadGPT-
P1&2relies on its aligned internal knowledge and
produces consistent, interpretable, and executable for-
mulas. This case demonstrates that while RAG bene-
fits large general models (e.g., DeepSeek V3) through
contextual recall and reasoning, it can be detrimental
to lightweight domain-specific models, where noisy or
contextually inconsistent retrievals may impair both
instruction complianceandcode executability.
7.2 Code-level Executability Results
WhiletheQA-levelbenchmarkevaluatessemanticand
reasoning accuracy, it does not directly reflect whether
model-generated scripts aretool-executable. To quan-
tify the practical impact of the IR →DPO alignment
pipeline, we conducted a syntax-level executability
test on 20 unseen SDE instructions rendered from the
IR generator. Each instruction was converted into nat-
ural language and used as input for model generation.
The outputs were validated by directly invoking the of-
ficial sde -Ssyntax checker from Synopsys Sentaurus
14

0 5 10
CountDirect passPlaceholder-resolvedFail
12/201/207/20SDE executability breakdown (Pass@1)(a)
0 5 10 15
CountDirect passPlaceholder-resolvedFail
15/201/204/20SDE executability breakdown (Pass@3) (b)
Figure 9:SDE executability breakdown for
TcadGPT.(a)Pass@1resultsshowingdirectpasses,
placeholder-resolved passes, and failures. (b) Pass@3
results under three-sample decoding.
without any manual correction.
It is worth noting that the QA-level and code-level
evaluations are performed on two different model
checkpoints. The QA benchmark usesTcadGPT-
P1&2, which was fine-tuned solely on 1.5M Alpaca-
style QA pairs. The code-level executability test,
by contrast, evaluates theTcadGPT-P1&2+DPO
model, which further incorporates IR-based prefer-
ence alignment to enhance instruction compliance
and code correctness. This separation reflects the dis-
tinct objectives of the two evaluation tracks: semantic
understanding versus executable code generation.
Evaluation Protocol.For each instruction, we
collected both single-sample and multi-sample gener-
ations:
•Pass@1: proportion of first-sample generations
thatsuccessfullypassedsyntaxvalidationwithout
modification.
•Pass@3: proportion of instructions with at least
one syntactically valid script among three sam-
pled generations.
We further categorized results into: (i)Direct
pass— code compiled without modification; (ii)
Placeholder-resolved pass— code containing tem-
plate variables (e.g., @height@ ,@gate_len@ ) that
passed syntax check after substituting numerical con-
stants; and (iii)Fail— code rejected by the compiler
due to syntax or structural errors. All tests were con-
ducted using the same SDE version and configuration
to ensure consistency.Table 2: Syntax-level executability on a 20-instruction
SDE test set. Pass@k: proportion of instructions
whose generated code passed sde -S. Placeholder res-
olution refers to replacing template variables (e.g.,
@height@) with numeric constants.
Model Pass@1 Pass@3
TcadGPT (P1+P2+DPO)13/20 (65.0%) 16/20 (80.0%)
DeepSeek V30/20 (0.0%) 0/20 (0.0%)
Breakdown forTcadGPT:Direct-pass@1 = 12/20 (60.0%);
after placeholder resolution = +1⇒Pass@1 = 13/20.
Direct-pass@3 = 15/20 (75.0%); after placeholder resolution =
+1⇒Pass@3 = 16/20.
Results and Discussion.As summarized in Ta-
ble 2 and Fig. 9, the finalTcadGPTmodel—fine-
tuned on the combinedP1+P2 QA corpus and
the DPO alignment dataset—achieved aPass@1
rate of 65.0%(13 out of 20 instructions) on the
syntax-level SDE executability benchmark. As shown
inFig.9(a), themajorityofsuccessfulcasesweredirect
passeswithout any modification (12/20), with only a
single additional case passing after simple placeholder
substitution.
When sampling three generations per instruction
(Pass@3), the overall pass rate further increased
to80.0%(16/20). Importantly, as illustrated in
Fig. 9(b), this improvement was primarily driven by
an increase in direct passes (15/20), rather than re-
liance on placeholder resolution, indicating improved
structural robustness under multi-sample decoding.
In contrast,DeepSeek V3—despite producing out-
puts that superficially resembled SDE scripts—failed
syntax validation for all 20 test cases (0% Pass@1 /
Pass@3), confirming that its generated code lacked
structural correctness and command-level consistency.
SDE Test Set.The 20-instruction SDE test set was
designed to comprehensively evaluate the executabil-
ity of model-generated scripts across representative
TCAD scenarios. It covers both 2D and 3D structures,
diverse Boolean operation modes (ABA, BAB), mate-
rial stacks (Silicon, SiO 2, Si3N4, PolySi, GaN, AlGaN,
etc.), and a range of procedural commands including
contact definition, doping placement (constant and
15

Gaussian), mesh refinement, and export directives.
Instructions were automatically rendered into natural
language via the IR generator, ensuring that they are
unseen during training while remaining fully valid
within the Synopsys SDE syntax space. This mix-
ture of geometry, doping, and meshing tasks provides
a practical and challenging benchmark for assessing
syntax-level executabilityin real TCAD workflows.
Why DPO is Necessary.Before adopting Direct
Preference Optimization (DPO), we systematically ex-
plored several alignment strategies based on standard
supervised fine-tuning. Models trained solely on the
Alpaca-style QA corpus, even after introducing chain-
of-thought (COT) rationales, exhibited poor instruc-
tionadherenceandfrequentlyproducedalmost-correct
but non-executablecode: commands were placed in
the wrong order, mesh parameters were mis-scaled,
and export directives were often omitted. These is-
sues persisted regardless of prompt format or data
scale, revealing that simple SFT or COT supervision
was insufficient to instill strict syntactic discipline.
Integrating DPO on top of the QA-tuned base proved
essential, as it directly optimized the model to prefer
instruction-compliant outputs while penalizing nu-
meric or procedural deviations. This shift led to a
clear and reproducible improvement in executable ac-
curacy—evident in the 80% syntax pass rate achieved
in the SDE test set.
These results highlight that integrating DPO align-
ment on top of QA fine-tuning (P1+P2) significantly
enhances instruction compliance and syntactic fidelity.
Whereas QA-only fine-tuning improves general reason-
ing but not formal syntax, the DPO stage explicitly
optimizes for executable consistency by penalizing
numeric and procedural deviations (e.g., wrong mesh
parameters, export omissions). Consequently, the
resulting model can interpret simulation intent and
generate scripts that aredirectly compilable by com-
mercial TCAD tools.
Scope and Limitations.This evaluation focuses
onsyntax-level executabilityin the SDE environ-
ment. Runtime validation (e.g., full mesh build and
TDR/BND export) was not included. The sample
0 10 20 30 40 50 60
Correct answers (count)Our 8B
DeepSeek
GPT-4o52/100
34/100
32/100ELMER QA benchmark (100 questions)(a) QA accuracy on 100
questions (Pass@1).
0 5 10
Executable cases (count)Our 8B (IR DPO)
DeepSeek-3.2
GPT-4o12/20
7/20
4/20ELMER code executability (20 instructions)(b) Executable cases on 20
instructions (Pass@1).
Figure 10:Lightweight cross-domain check on
Elmer.Using the same QA synthesis and IR →DPO
alignment recipe, our 8B model outperforms general-
purpose LLMs on both QA and solver-level code exe-
cutability.
size (20 instructions) is limited but representative of
diverse geometry and doping scenarios. Future work
will expand this evaluation to SProcess and SDevice
modules, increase coverage to over 100 instructions
per tool, and introduce anend-to-end execution
success(Pass@k)metriccapturingsolver-completed
runs under a fixed environment.
Summary.The code-level experiment shows
that combining QA-based fine-tuning (P1+P2)
with IR →DPO alignment substantially improves
instruction-following fidelityandsyntax-level compi-
labilityin a commercial TCAD toolchain. With an
80% sde -Spass rate,TcadGPTmoves beyond de-
scriptive assistance and becomes a practical generator
oftool-acceptedSDE scripts, whereas the general-
purposeDeepSeek V3baseline fails all basic syntax
checks in this setting.
7.3LightweightCross-DomainDemon-
stration on Elmer
7.3.1 Setup
To assess the portability of our schema-first alignment
framework beyond TCAD, we conduct a lightweight
cross-domain study on the open-source finite-element
solverElmer. Unlike TCAD tools, which are special-
ized semiconductor simulators,Elmeris a general-
purpose multiphysics FEM solver with a fundamen-
tally different problem formulation, numerical work-
16

flow, andscriptinginterface. Thisstructuraldifference
makes it a suitable testbed for probing cross-domain
generalization rather than near-domain transfer.
We intentionally keep this study small-scale and
complementary. The goal is not to establish a bench-
mark comparable in depth to the TCAD suite, but to
verify whether the proposed alignment recipe can be
instantiated with minimal domain-specific engineering
effort.
We reuse the same two-stage recipe as in the TCAD
setting: (i) QA synthesis from domain documenta-
tion using Pipeline 2, followed by full-parameter fine-
tuning of an 8B base model; and (ii) an IR →DPO-
style alignment workflow to improve instruction com-
pliance and script executability. All QA training data
are generated automatically from publicly available
Elmer documentation, without access to proprietary
corpora.
We note that the DeepSeek baseline used in the
Elmer experiments corresponds to DeepSeek V3.2,
whereas the TCAD experiments use DeepSeek V3.
This difference arises from changes in the official API
during the course of the study and does not affect the
qualitative comparison reported here.
7.3.2 QA evaluation
For QA-level evaluation, we manually construct a
100-question Elmer test set derived from the same
documentation sources. The questions are intention-
ally restricted tosingle-factitems, each with a single
unambiguous answer, covering solver configuration,
boundary conditions, material specification, and com-
mon scripting patterns. This design avoids multi-step
reasoning and isolates factual recall and command-
level understanding.
As summarized in Fig. 10a, under Pass@1 evalua-
tion the Elmer-adapted 8B model achieves52/100,
compared with34/100for DeepSeek V3.2 and
32/100for GPT-4o. The consistent margin indi-
cates that domain-specific QA synthesis alone already
yields substantial gains over general-purpose LLMs in
a structurally distinct solver domain.7.3.3 Code executability evaluation
Wefurtherevaluatecode-levelgenerationon20unseen
natural-language instructions. For each instruction,
the model generates a .sif/.jis -style Elmer solver
input, which is executed directly using ElmerSolver ,
without relying on a separate syntax-only checker.
Before execution, we automatically strip any non-
.sifcontent (e.g., explanations or Markdown) from
model outputs and ensure that referenced mesh files
exist, either by correcting mesh database pointers
or by generating meshes via ElmerGrid . No man-
ual modification of solver commands or numerical
parameters is performed.
As shown in Fig. 10b, the IR →DPO-aligned 8B
model yields12/20executable cases under Pass@1,
compared with7/20for DeepSeek V3.2 and4/20
for GPT-4o. Despite the small scale, this executable
gap mirrors the QA trend in Fig. 10 and provides a
consistent signal that the proposed alignment recipe
transfers beyond TCAD to solver-level code genera-
tion in a different numerical and scripting paradigm.
We note that executability rates between TCAD
and Elmer are not directly comparable. TCAD scripts
exhibit tighter syntactic and numerical coupling with
stricter front-end validation, such that minor devi-
ations can lead to early rejection during parsing or
preprocessing. In contrast, Elmer adopts a more per-
missive key–value style input and defers some consis-
tencycheckstolatersolverstages. Therefore, absolute
executability levels reflect not only model capability
but also inherent differences in language strictness
and tool-chain behavior. The Elmer study is thus in-
tended to probe portability rather than to normalize
difficulty across domains.
8 Conclusion and Future Work
We presented aschema-first alignment frame-
workfor buildingexecutabledomain-specific LLMs
underdata scarcity. Instantiated on TCAD as
TcadGPT, the framework combines large-scale QA
synthesis (1.5M pairs) with IR-driven DPO alignment
to directly optimize instruction-following fidelity and
syntactic validity for tool-accepted code generation.
17

On a 264-question benchmark and a 20-instruction
SDEexecutabilitytest,TcadGPTachieves85.6%se-
mantic accuracy and an80.0%syntax pass rate under
the official sde -Schecker, substantially outperform-
ing general-purpose LLMs, which exhibit markedly
lower success rates in this setting. We further observe
that Retrieval-Augmented Generation (RAG) sub-
stantially benefits generic models but can marginally
degrade already specialized compact ones, motivating
the need forschema-constrainedretrieval strategies.
Beyond TCAD, we show that the same alignment
recipe—QA synthesis from expert materials, schema-
first IR extraction and diversification, and IR →DPO
alignment—can be applied with minimal modification
to an open-source FEM solver, yielding consistent
improvements in script-level success rates. These
results suggest that the proposed framework captures
areusablealignmentpatternfortool-executableLLMs
in data-scarce scientific and engineering domains that
demand strict artifact validity.
Limitations.Our evaluation deliberately empha-
sizessyntax-level executability(tool acceptance)
rather thanruntime-leveloutcomes such as numeri-
cal convergence or end-to-end completion. The latter
can depend sensitively on simulator-specific stability,
meshing strategies, and solver configurations, and are
therefore not fully isolated by the current benchmarks.
In addition, the present system does not yet perform
robust multi-step planning or interactive repair, and
residual hallucinations may occur under weakly struc-
tured retrieval settings.
Future Work.Future directions include: (i)
closed-loop integrationwith commercial solvers
to enable runtime-level verification and automatic re-
pair; (ii)reward modelingbased on convergence
signals, physical consistency checks, or explicit tool
feedback; (iii) strongerplanning and decomposi-
tionmechanisms for multi-stage workflows; and (iv)
schema-aware retrievalthat constrains evidence
injection to avoid lexical overfitting while preserving
factual grounding.References
[1]S. Salahuddin, K. Ni, and S. Datta, “The era of
hyper-scaling in electronics,”Nature Electronics,
vol. 1, no. 8, pp. 442–450, 2018.
[2]W. Cao, H. Bu, M. Vinet, M. Cao, S. Takagi,
S. Hwang, T. Ghani, and K. Banerjee, “The fu-
ture transistors,”Nature, vol. 620, no. 7974, pp.
501–507, 2023.
[3]K. Mehta and H.-Y. Wong, “Prediction of finfet
current-voltage and capacitance-voltage curves
using machine learning with autoencoder,”IEEE
Electron Device Letters, vol. 42, no. 2, pp. 136–
139, 2021.
[4]S. Myung, W. Jang, S. Jin, J. M. Choe, C. Jeong,
and D. S. Kim, “Restructuring tcad system:
Teaching traditional tcad new tricks,” in2021
IEEE International Electron Devices Meeting
(IEDM). IEEE, 2021, pp. 18.2.1–18.2.4.
[5]L. Li, M. Agrawal, S. Y. Yeh, K. T. Lam, J. Wu,
and B. Magyari-Köpe, “Towards accurate and
efficient process simulations based on atomistic
and neural network approaches,” in2022 Interna-
tional Electron Devices Meeting (IEDM). IEEE,
2022, pp. 15.6.1–15.6.4.
[6]S. K. Mitusch, S. W. Funke, and M. Kuchta, “Hy-
brid fem-nn models: Combining artificial neural
networks with the finite element method,”Jour-
nal of Computational Physics, vol. 446, p. 110651,
2021.
[7]B. Kim and M. Shin, “A novel neural-network
device modeling based on physics-informed ma-
chine learning,”IEEE Transactions on Electron
Devices, vol. 70, no. 11, pp. 6021–6028, 2023.
[8]A. Lu, Y. F. Chau, and H. Y. Wong, “Physics-
informed neural network for predicting out-of-
training-range tcad solution with minimized do-
main expertise,” inProceedings of the IEEE Elec-
tron Devices Technology and Manufacturing Con-
ference (EDTM). IEEE, 2025, accepted for
publication.
18

[9]T. Wu and J. Guo, “Multiobjective design of
2-d-material-based field-effect transistors with
machine learning methods,”IEEE Transactions
on Electron Devices, vol. 68, no. 11, pp. 5476–
5482, 2021.
[10]H.Xu, W.Gan, L.Cao, C.Yang, J.Wu, M.Zhou,
H. Qu, S. Zhang, H. Yin, and Z. Wu, “A machine
learning approach for optimization of channel ge-
ometryandsource/draindopingprofileofstacked
nanosheet transistors,”IEEE Transactions on
Electron Devices, vol. 69, no. 7, pp. 3568–3574,
2022.
[11]H. Jeong, J. Choi, H. Cho, S. Woo, Y. Kim, J.-
T. Kong, and S. Kim, “Mobo-driven advanced
sub-3-nm device optimization for enhanced pdp
performance,”IEEE Transactions on Electron
Devices, vol. 71, no. 5, pp. 2881–2887, 2024.
[12]Z. He, H. Wu, X. Zhang, X. Yao, S. Zheng,
H. Zheng, and B. Yu, “Chateda: A large lan-
guage model powered autonomous agent for eda,”
inProceedings of the 2023 ACM/IEEE Work-
shop on Machine Learning for CAD (MLCAD).
IEEE, 2023, p. to appear.
[13]K. Chang, Y. Wang, H. Ren, M. Wang, S. Liang,
Y. Han, H. Li, and X. Li, “Chipgpt: How far
are we from natural language hardware design,”
arXiv preprint arXiv:2305.14019, 2023.
[14]Y. Lu, S. Liu, Q. Zhang, and Z. Xie, “Rtllm:
An open-source benchmark for design rtl genera-
tion with large language model,”arXiv preprint
arXiv:2308.05345, 2023.
[15]C.-C. Chang, C.-T. Ho, Y. Li, Y. Chen, and
H. Ren, “Drc-coder: Automated drc checker code
generation using llm autonomous agent,” inPro-
ceedings of the 2025 International Symposium
on Physical Design (ISPD). ACM, 2025, p. to
appear.
[16]Y. Jiang, X. Lu, Q. Jin, Q. Sun, H. Wu, and
C. Zhuo, “Fabgpt: An efficient large multi-
modal model for complex wafer defect knowledge
queries,” inProceedings of the 2024 InternationalConference on Computer-Aided Design (ICCAD).
ACM, 2024, p. to appear.
[17]L. M. L. Nguyenet al., “Tcad structure input
file generation using large language model,” in
SISPAD 2024, 2024.
[18]P. Råback, M. Malinen, J. Ruokolainen, and
T. Zwinger, “Elmer finite element solver for mul-
tiphysics,”CSC – IT Center for Science, 2012.
[19]ETH Zurich, “A language model for public good,”
ETH News, 2025.
[20]R. Jacobset al., “Towards ai-accelerated quan-
tum chemistry: Orca-llm for code generation and
simulation planning,”Digital Discovery, 2024.
[21]H. Zhonget al., “Llm4eda: Large lan-
guage models for electronic design automa-
tion: A comprehensive survey,”arXiv preprint
arXiv:2401.12224, 2024.
[22]H. Liuet al., “Chipnemo: Domain-specialized
large language models for chip design,”arXiv
preprint arXiv:2311.00176, 2023.
[23]Y. Wang, Y. T. Peris, A. Bosselut, I. Gurevych,
and L. Zettlemoyer, “Stanford alpaca: An
instruction-following llama model,”arXiv
preprint arXiv:2305.10906, 2023.
[24]A. Hurst, J. Koon, S. McDonell, and T. Ope-
nAI, “Gpt-4o system card,”arXiv preprint
arXiv:2410.21276, 2024. [Online]. Available:
https://arxiv.org/abs/2410.21276
[25]DeepSeek-AI, “Deepseek-r1: Incentivizing
reasoning capability in llms via reinforce-
ment learning,” 2025. [Online]. Available:
https://arxiv.org/abs/2501.12948
[26]A. Liu, Z. Chen, D. Guo, J. Songet al.,
“Deepseek-v3: A 671b-parameter mixture-
of-experts language model,”arXiv preprint
arXiv:2412.19437, 2024. [Online]. Available:
https://arxiv.org/abs/2412.19437
19

Appendix A: Prompt Templates
for Pipeline 1
Pipeline 1: Prompt for Generating
Alpaca-format QA Pairs
Prompt: Generate QA Pairs
Your task is to generate high-quality fine-tuning
data based on the provided text. In content, you
will be given a domain-specific technical passage.
You need to generate data in the following format:
{ "instruction": "User instruction.",
"input": "", "output": "Expected system
output." }
Questions must be relevant to the material, profes-
sional, and challenging. Focus on commands, code,
formulas, operation steps, scientific principles, and
software logic. Questions should be diverse, and
answers must be rich, professional, and structured.
The output should be a normal response to your
generated instruction. Ensure the instruction
is general and not limited to phrases like in
this book . The format must be strictly followed.
The provided content may contain HTML tags
or unrendered LaTeX code—only output human-
readable text, and ensure formulas render cor-
rectly.
Content is mostly technical manuals or textbooks.
Ask a question about each technical point or com-
mand. Sample questions include:
•What is the impact of optical phonon scat-
tering on mobility?
•How to define new dopant types in Sprocess?
•How to save a simulated structure as a TDR
file?
•What are the key parameters in ion implan-
tation, and how to configure simulation com-
mands?
Avoid duplicate or equivalent questions. Always
generate at least 10 different questions per passage.
All content must be in Chinese.Pipeline 1: Prompt for Question Aug-
mentation
Prompt: Augment Questions
You will be given a single QA pair from a fine-
tuning dataset. Your task is to perform data
augmentation. Modify the question into different
forms.
Output should be a Python list containing 10 new
questions.
Ensure question diversity. Questions should be
inspired by both the original question and the
answer. Output must be a Python-readable list:
["question1", "question2", ...,
"question10"]
Only output the list; no extra characters. Must
be in Chinese.
Appendix B: Prompt Templates
for Keyword-Guided Pipeline 2
Prompt: Extract Keywords
You will be given a technical TCAD passage in
either English or Chinese. Your task is to extract
all keywords that can be further queried. These
include but are not limited to:
•Command names (e.g.,pdbSet,Electrode)
•Parameter names (e.g.,MinGrowthStep)
•Model names (e.g., Hydrodynamic model)
•Software module names (e.g., Sentaurus De-
vice)
•Physical mechanisms or equations (e.g.,
thermionic emission, carrier drift)
Output format must be:
{ "keywords": ["keyword1", "keyword2",
...] }
Strictly follow this format. Do not extract key-
words if the passage lacks technical content. Pre-
serve the original language of each keyword and
ensure complete coverage.
20

Prompt: Generate QA from Keywords
Your task: Based on the given technical paragraph
and keyword, generate structured Chinese QA
data for Alpaca-style fine-tuning.
Output must be a JSON array. Each entry follows
this format:
{ "instruction": "User question",
"input": "", "output": "System answer"
}
Guidelines:
1. Only generate in Chinese.
2.If the paragraph lacks semantic sentences or
relevant info for a keyword, skip it and return
[].
3.Style based on keyword type: concept, for-
mula, command, module, or configuration.
4.Cover all aspects of the keyword. Avoid rep-
etition.
5.Answers must be precise, executable, with
code or formulas if needed.
6. Use$...$for inline formulas.
7.Do not refer directly to the paragraph. Do
not add markdown markers.
8.When the keyword is a command/parameter,
the question should describe the function, not
the name.
21